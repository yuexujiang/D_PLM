"""
DDG Prediction Pipeline using XGBoost with Antisymmetric Features

Data:
  - Train/Valid: S8754.csv  (columns: name, wt_seq, mut_seq, ddG)
  - Test:        S669.csv

Encoder: D-PLM → mean-pooled per-residue embeddings

Antisymmetry strategy:
  The neural model (model_ddt.py Encoder) enforces antisymmetry structurally:
      pred = head(emb_mt) - head(emt_wt)  →  f(mt, wt) = -f(wt, mt)
  XGBoost cannot do this structurally, so we use three complementary tricks:

  1. Feature:  diff = emb(mt) - emb(wt)
               Swapping inputs negates the feature: diff(wt,mt) = -diff(mt,wt)

  2. Data augmentation: for every (wt→mt, ddG) sample, add (mt→wt, -ddG).
     This trains XGBoost to approximate an odd/antisymmetric function.

  3. Inference enforcement (exact antisymmetry):
         pred_sym(x) = ( f(x) - f(-x) ) / 2
     This guarantees pred(wt→mt) = -pred(mt→wt) for any trained model,
     regardless of how well the training actually converged to an odd function.
"""

import os
import argparse
import numpy as np
import pandas as pd
import pickle
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb

# ─────────────────────────────────────────────
# 1. Argument parser
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description='DDG XGBoost Pipeline')
parser.add_argument("--model-location", type=str,
                    default='../results/vivit_3/checkpoints/checkpoint_best_val_rmsf_cor.pth')
parser.add_argument("--config-path",    type=str,
                    default='../results/vivit_3/config_vivit3.yaml')
parser.add_argument("--model-type",     type=str, default='d-plm')
parser.add_argument("--sequence",       type=str, default='')
parser.add_argument("--output-path",    type=str, default='./ddt/S8754_xgboost')
parser.add_argument("--train-csv",      type=str, default='./ddt/S8754.csv',
                    help='Training CSV: name, wt_seq, mut_seq, ddG')
parser.add_argument("--test-csv",       type=str, default='./ddt/S669.csv',
                    help='Test CSV:     name, wt_seq, mut_seq, ddG')
parser.add_argument("--save-model",     type=str, default='xgb_ddg.pkl')
parser.add_argument("--early-stopping", type=int, default=50,
                    help='Early stopping rounds (0 to disable)')
parser.add_argument("--use-dummy-encoder", action='store_true',
                    help='Use random embeddings (for debugging without GPU)')
args, unknown = parser.parse_known_args()

os.makedirs(args.output_path, exist_ok=True)

# ─────────────────────────────────────────────
# 2. Encoder helpers  (mirrors phase_separation_xgboost.py)
# ─────────────────────────────────────────────
if not args.use_dummy_encoder:
    if args.model_type in ("d-plm", "ESM2"):
        import torch
        import esm
        import yaml
        from utils.utils import load_configs
        import esm_adapterH
        from collections import OrderedDict

        def load_checkpoints(model, checkpoint_path):
            if checkpoint_path is None:
                print("checkpoints not exist " + str(checkpoint_path))
                return
            checkpoint = torch.load(checkpoint_path,
                                    map_location=lambda storage, loc: storage)
            if any('adapter' in name for name, _ in model.named_modules()):
                if np.sum(["adapter_layer_dict" in k
                           for k in checkpoint['state_dict1'].keys()]) == 0:
                    new_dict = OrderedDict()
                    for k, v in checkpoint['state_dict1'].items():
                        if "adapter_layer_dict" not in k:
                            k = k.replace('adapter_layer', 'adapter_layer_dict.adapter_0')
                        new_dict[k] = v
                    model.load_state_dict(new_dict, strict=False)
                else:
                    new_dict = OrderedDict()
                    for k, v in checkpoint['state_dict1'].items():
                        new_dict[k.replace("esm2.", "")] = v
                    model.load_state_dict(new_dict, strict=False)
            elif any('lora' in name for name, _ in model.named_modules()):
                new_dict = OrderedDict()
                for k, v in checkpoint['state_dict1'].items():
                    new_dict[k.replace("esm2.", "")] = v
                model.load_state_dict(new_dict, strict=False)
            print("checkpoints were loaded from " + checkpoint_path)

        def load_model(args):
            if args.model_type == "d-plm":
                with open(args.config_path) as f:
                    config_file = yaml.full_load(f)
                configs = load_configs(config_file, args=None)
                if configs.model.esm_encoder.adapter_h.enable:
                    model, alphabet = esm_adapterH.pretrained.esm2_t33_650M_UR50D(
                        configs.model.esm_encoder.adapter_h)
                else:
                    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
                load_checkpoints(model, args.model_location)
            else:  # ESM2
                model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
                print("Transferred model to GPU")
            return model, alphabet

        def encoder_main(args):
            batch_converter = args.alphabet.get_batch_converter()
            _, _, tokens = batch_converter([("protein1", args.sequence)])
            with torch.no_grad():
                rep = args.model(tokens.cuda(),
                                 repr_layers=[args.model.num_layers]
                                 )["representations"][args.model.num_layers]
            return rep.squeeze(0)

        args.model, args.alphabet = load_model(args)


def get_embedding(seq: str) -> np.ndarray:
    """Return mean-pooled embedding vector for one amino-acid sequence."""
    if args.use_dummy_encoder:
        rng = np.random.default_rng(abs(hash(seq)) % (2**31))
        return rng.random(480).astype(np.float32)
    args.sequence = seq
    output = encoder_main(args)
    emb = output[1:-1].cpu().numpy()   # strip [CLS] / [EOS]
    return emb.mean(axis=0)


def embed_sequences(sequences: list, desc: str = '') -> np.ndarray:
    vecs = []
    n = len(sequences)
    for i, seq in enumerate(sequences):
        if (i + 1) % 50 == 0 or i == 0:
            print(f'  [{desc}] {i+1}/{n}')
        vecs.append(get_embedding(seq))
    return np.vstack(vecs)


def cached_embed(sequences, cache_path, desc=''):
    if os.path.exists(cache_path):
        print(f'  Loading cached embeddings from {cache_path}')
        return np.load(cache_path)
    print(f'  Computing embeddings for {len(sequences)} sequences ({desc}) …')
    X = embed_sequences(sequences, desc=desc)
    np.save(cache_path, X)
    print(f'  Saved to {cache_path}')
    return X

# ─────────────────────────────────────────────
# 3. Data loading
# ─────────────────────────────────────────────
def extract_protein_id(name):
    """rcsb_1A7V_A_A104H_6_25 → 1A7V_A  (mirrors data_ddt.py)"""
    parts = name.split("_")
    return f"{parts[1]}_{parts[2]}"


def load_csv(path):
    df = pd.read_csv(path)
    df["protein_id"] = df["name"].apply(extract_protein_id)
    print(f'  Loaded {len(df)} samples from {os.path.basename(path)}')
    return df

# ─────────────────────────────────────────────
# 4. Load & embed training / validation data
# ─────────────────────────────────────────────
print('\n=== Loading training data ===')
df_all = load_csv(args.train_csv)

# Protein-wise split to prevent leakage (same as data_ddt.py)
proteins = df_all["protein_id"].unique()
train_prots, val_prots = train_test_split(proteins, test_size=0.2, random_state=42)
df_train = df_all[df_all["protein_id"].isin(train_prots)].reset_index(drop=True)
df_valid = df_all[df_all["protein_id"].isin(val_prots)].reset_index(drop=True)
print(f'  Train: {len(df_train)} samples | Valid: {len(df_valid)} samples')

emb_wt_train = cached_embed(df_train["wt_seq"].tolist(),
                             os.path.join(args.output_path, 'train_wt_emb.npy'), 'train-wt')
emb_mt_train = cached_embed(df_train["mut_seq"].tolist(),
                             os.path.join(args.output_path, 'train_mt_emb.npy'), 'train-mt')

emb_wt_valid = cached_embed(df_valid["wt_seq"].tolist(),
                             os.path.join(args.output_path, 'valid_wt_emb.npy'), 'valid-wt')
emb_mt_valid = cached_embed(df_valid["mut_seq"].tolist(),
                             os.path.join(args.output_path, 'valid_mt_emb.npy'), 'valid-mt')

# ─────────────────────────────────────────────
# 5. Antisymmetric feature construction
# ─────────────────────────────────────────────
# diff(wt→mt) = emb(mt) - emb(wt)
# diff(mt→wt) = emb(wt) - emb(mt) = -diff(wt→mt)   ← antisymmetric by construction
diff_train = emb_mt_train - emb_wt_train          # [N_train, D]
y_train    = df_train["ddG"].values

# Augment with reversed pairs: (mt→wt, -ddG)
# This teaches the model: f(-x) ≈ -f(x)
diff_train_aug = np.vstack([diff_train, -diff_train])
y_train_aug    = np.concatenate([y_train, -y_train])

diff_valid = emb_mt_valid - emb_wt_valid
y_valid    = df_valid["ddG"].values

print(f'\nFeature dim:  {diff_train.shape[1]}')
print(f'Train samples (after augmentation): {len(y_train_aug)}')
print(f'ddG range: [{y_train.min():.2f}, {y_train.max():.2f}]')

# ─────────────────────────────────────────────
# 6. Train XGBoost
# ─────────────────────────────────────────────
print('\n=== Training XGBoost ===')

xgb_model = xgb.XGBRegressor(
    n_estimators=2000,
    max_depth=8,
    learning_rate=0.005,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    eval_metric='rmse',
    early_stopping_rounds=args.early_stopping if args.early_stopping > 0 else None,
    random_state=42,
    n_jobs=-1,
)
xgb_model.fit(diff_train_aug, y_train_aug,
              eval_set=[(diff_valid, y_valid)],
              verbose=50)

with open(os.path.join(args.output_path, args.save_model), 'wb') as f:
    pickle.dump(xgb_model, f)
print(f'Model saved to {os.path.join(args.output_path, args.save_model)}')

# ─────────────────────────────────────────────
# 7. Symmetric prediction (enforces exact antisymmetry)
# ─────────────────────────────────────────────
def predict_antisymmetric(model, diff_feat: np.ndarray) -> np.ndarray:
    """
    Enforce exact antisymmetry at inference time:
        pred_sym(x) = ( f(x) - f(-x) ) / 2

    Proof: pred_sym(-x) = (f(-x) - f(x)) / 2 = -pred_sym(x)  ✓

    This guarantees pred(wt→mt) = -pred(mt→wt) regardless of
    how closely XGBoost approximated an odd function during training.
    """
    return (model.predict(diff_feat) - model.predict(-diff_feat)) / 2

# ─────────────────────────────────────────────
# 8. Evaluation
# ─────────────────────────────────────────────
def evaluate(model, diff_feat: np.ndarray, y_true: np.ndarray, label: str) -> dict:
    y_pred   = predict_antisymmetric(model, diff_feat)
    rmse     = np.sqrt(mean_squared_error(y_true, y_pred))
    mae      = np.mean(np.abs(y_true - y_pred))
    spearman, _ = spearmanr(y_true, y_pred)
    pearson,  _ = pearsonr(y_true, y_pred)
    print(f'\n--- {label} (N={len(y_true)}) ---')
    print(f'  RMSE     : {rmse:.4f}')
    print(f'  MAE      : {mae:.4f}')
    print(f'  Spearman : {spearman:.4f}')
    print(f'  Pearson  : {pearson:.4f}')
    return dict(Dataset=label, N=len(y_true),
                RMSE=round(rmse, 4),
                MAE=round(mae, 4),
                Spearman=round(spearman, 4),
                Pearson=round(pearson, 4))

print('\n=== Evaluating ===')
results = []
results.append(evaluate(xgb_model, diff_valid, y_valid, label='Valid'))

# ─────────────────────────────────────────────
# 9. Load & embed test data
# ─────────────────────────────────────────────
print('\n=== Loading test data ===')
df_test = load_csv(args.test_csv)

emb_wt_test = cached_embed(df_test["wt_seq"].tolist(),
                            os.path.join(args.output_path, 'test_wt_emb.npy'), 'test-wt')
emb_mt_test = cached_embed(df_test["mut_seq"].tolist(),
                            os.path.join(args.output_path, 'test_mt_emb.npy'), 'test-mt')

diff_test = emb_mt_test - emb_wt_test
y_test    = df_test["ddG"].values

results.append(evaluate(xgb_model, diff_test, y_test, label='Test (S669)'))

# ─────────────────────────────────────────────
# 10. Summary
# ─────────────────────────────────────────────
summary_df   = pd.DataFrame(results)
summary_path = os.path.join(args.output_path, 'evaluation_results.csv')
summary_df.to_csv(summary_path, index=False)

print('\n' + '='*60)
print('SUMMARY')
print('='*60)
print(summary_df.to_string(index=False))
print(f'\nResults saved to {summary_path}')


################# saved model config
'''
args.early_stopping=50
print('\n=== Training XGBoost ===')

xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=10,
    learning_rate=0.005,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    eval_metric='rmse',
    early_stopping_rounds=args.early_stopping if args.early_stopping > 0 else None,
    random_state=42,
    n_jobs=-1,
)
xgb_model.fit(diff_train_aug, y_train_aug,
              eval_set=[(diff_valid, y_valid)],
              verbose=50)

  RMSE     : 1.7375
  Spearman : 0.3266
  Pearson  : 0.2674

'''
