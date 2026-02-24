"""
Phase Separation Protein Prediction Pipeline
- Training: Molphase_train_pos (label=1) + Molphase_train_neg (label=0)
- Encoder: D-PLM (user-provided main() function) → mean-pooled embeddings
- Classifier: XGBoost
- Evaluation: tableS1 ~ tableS5 (Positive=1, Negative=0)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import re
import pickle
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
)
import xgboost as xgb

# ─────────────────────────────────────────────
# 0.  Paste / import your existing code here
# ─────────────────────────────────────────────
# Assumed available in the environment:
#   from your_module import load_model, main   (the encoder)
# Replace the two lines below with your actual imports:
#
# from your_module import load_model, main
#
# For standalone testing, a dummy encoder is provided at the bottom of this
# file that returns random 480-dim vectors; delete it once real imports work.

# ─────────────────────────────────────────────
# 1.  Argument parser (mirrors your original snippet)
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Phase Separation XGBoost Pipeline')
parser.add_argument("--model-location", type=str,
                    default='./results/vivit_3/checkpoints/checkpoint_best_val_rmsf_cor.pth')
parser.add_argument("--config-path",    type=str,
                    default='./results/vivit_3/config_vivit3.yaml')
parser.add_argument("--model-type",     type=str, default='d-plm')
parser.add_argument("--sequence",       type=str, default='')
parser.add_argument("--output_path",    type=str, default='./evaluate/phase_sep/xgboost')
parser.add_argument("--train-pos",      type=str,
                    default='./evaluate/phase_sep/Molphase_train_pos.xlsx',
                    help='Path to positive training Excel file')
parser.add_argument("--train-neg",      type=str,
                    default='./evaluate/phase_sep/Molphase_train_neg.xlsx',
                    help='Path to negative training Excel file')
parser.add_argument("--test-dir",       type=str, default='./evaluate/phase_sep/',
                    help='Directory containing tableS1.xlsx … tableS5.xlsx')
parser.add_argument("--save-model",     type=str, default='xgb_phase_sep.pkl',
                    help='Where to save the trained XGBoost model')
parser.add_argument("--use-dummy-encoder", action='store_true',
                    help='Use random embeddings (for debugging without GPU)')
args, unknown = parser.parse_known_args()

os.makedirs(args.output_path, exist_ok=True)

# ─────────────────────────────────────────────
# 2.  Encoder helpers
# ─────────────────────────────────────────────
if not args.use_dummy_encoder:
    if args.model_type=="d-plm":
        import torch
        import esm
        import numpy as np
        import yaml
        from utils.utils import load_configs
        import esm_adapterH
        from collections import OrderedDict
        
        def load_checkpoints(model,checkpoint_path):
                if checkpoint_path is not None:
                    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
                    if  any('adapter' in name for name, _ in model.named_modules()):
                        if np.sum(["adapter_layer_dict" in key for key in checkpoint[
                            'state_dict1'].keys()]) == 0:  # using old checkpoints, need to rename the adapter_layer into adapter_layer_dict.adapter_0
                            new_ordered_dict = OrderedDict()
                            for key, value in checkpoint['state_dict1'].items():
                                if "adapter_layer_dict" not in key:
                                    new_key = key.replace('adapter_layer', 'adapter_layer_dict.adapter_0')
                                    new_ordered_dict[new_key] = value
                                else:
                                    new_ordered_dict[key] = value
                            
                            model.load_state_dict(new_ordered_dict, strict=False)
                        else:
                            #this model does not contain esm2
                            new_ordered_dict = OrderedDict()
                            for key, value in checkpoint['state_dict1'].items():
                                    key = key.replace("esm2.","")
                                    new_ordered_dict[key] = value
                            
                            model.load_state_dict(new_ordered_dict, strict=False)
                    elif  any('lora' in name for name, _ in model.named_modules()):
                        #this model does not contain esm2
                        new_ordered_dict = OrderedDict()
                        for key, value in checkpoint['state_dict1'].items():
                                print(key)
                                key = key.replace("esm2.","")
                                new_ordered_dict[key] = value
                        
                        model.load_state_dict(new_ordered_dict, strict=False)
                    
                    print("checkpoints were loaded from " + checkpoint_path)
                else:
                    print("checkpoints not exist "+ checkpoint_path)
        
        def load_model(args):
            if args.model_type=="d-plm":
                with open(args.config_path) as file:
                    config_file = yaml.full_load(file)
                    configs = load_configs(config_file, args=None)
                if configs.model.esm_encoder.adapter_h.enable:
                    model, alphabet = esm_adapterH.pretrained.esm2_t33_650M_UR50D(configs.model.esm_encoder.adapter_h)
                elif configs.model.esm_encoder.lora.enable:
                    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
                    lora_targets =  ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj","self_attn.out_proj"]
                    target_modules=[]
                    if configs.model.esm_encoder.lora.esm_num_end_lora > 0:
                        start_layer_idx = np.max([model.num_layers - configs.model.esm_encoder.lora.esm_num_end_lora, 0])
                        for idx in range(start_layer_idx, model.num_layers):
                            for layer_name in lora_targets:
                                target_modules.append(f"layers.{idx}.{layer_name}")
                        
                    peft_config = LoraConfig(
                        inference_mode=False,
                        r=configs.model.esm_encoder.lora.r,
                        lora_alpha=configs.model.esm_encoder.lora.alpha,
                        target_modules=target_modules,
                        lora_dropout=configs.model.esm_encoder.lora.dropout,
                        bias="none",
                        # modules_to_save=modules_to_save
                    )
                    peft_model = get_peft_model(model, peft_config)
                
                # inference for each model
                # model, alphabet = esm_adapterH.pretrained.esm2_t33_650M_UR50D(configs.model.esm_encoder.adapter_h)/
                load_checkpoints(model,args.model_location)
            elif args.model_type=="ESM2":
                #if use ESM2
                model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            
            
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
                print("Transferred model to GPU")
            
            return model,alphabet
        
        def encoder_main(args):
            batch_converter = args.alphabet.get_batch_converter()
            
            data = [
                ("protein1", args.sequence),
            ]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            
            with torch.no_grad():
                wt_representation = args.model(batch_tokens.cuda(),repr_layers=[args.model.num_layers])["representations"][args.model.num_layers]
            
            wt_representation = wt_representation.squeeze(0) #only one sequence a time
            return wt_representation
        
        args.model,args.alphabet = load_model(args)

def get_embedding(seq: str) -> np.ndarray:
    """Return mean-pooled embedding vector for one amino-acid sequence."""
    if args.use_dummy_encoder:
        rng = np.random.default_rng(abs(hash(seq)) % (2**31))
        return rng.random(480).astype(np.float32)           # dummy 480-d vector
    args.sequence = seq
    wt_output = encoder_main(args)                          # your encoder call
    emb = wt_output[1:-1].cpu().numpy()                     # strip [CLS]/[EOS]
    return emb.mean(axis=0)

def embed_sequences(sequences: list, desc: str = '') -> np.ndarray:
    """Embed a list of sequences with a progress counter."""
    vecs = []
    n = len(sequences)
    for i, seq in enumerate(sequences):
        if (i + 1) % 50 == 0 or i == 0:
            print(f'  [{desc}] {i+1}/{n}')
        vecs.append(get_embedding(seq))
    return np.vstack(vecs)

# ─────────────────────────────────────────────
# 3.  Data-loading helpers
# ─────────────────────────────────────────────
def load_flat_sequences(path: str, col: str = 'Sequence') -> list:
    """Load sequences from a simple tabular Excel (Molphase format)."""
    df = pd.read_excel(path)
    seqs = df[col].dropna().str.strip().tolist()
    seqs = [s for s in seqs if s]
    print(f'  Loaded {len(seqs)} sequences from {os.path.basename(path)}')
    return seqs


def load_labeled_sequences(path: str) -> tuple:
    """
    Parse the interleaved label / header / sequence format used in
    tableS1 – tableS5.

    Layout (0-indexed, no header row used):
      col-0: label cell ('Positive' or 'Negative', then NaN for body rows)
      col-1: alternating >header lines and sequence lines
              (col-1 may hold the sequence while col-2 holds the header in S2)

    Returns (sequences, labels) as parallel lists.
    """
    df = pd.read_excel(path, header=None)

    sequences, labels = [], []
    current_label = None
    pending_header = False   # True when we just saw a >header line

    # Determine which column contains the sequence data
    # For tableS2 the layout has 3 columns; sequences live in column 2.
    n_cols = df.shape[1]

    for _, row in df.iterrows():
        # --- update current label ---
        cell0 = str(row.iloc[0]).strip()
        if cell0.lower() in ('positive', 'negative'):
            current_label = 1 if cell0.lower() == 'positive' else 0

        if current_label is None:
            continue

        # --- walk through remaining columns looking for header/sequence ---
        for col_idx in range(1, n_cols):
            val = str(row.iloc[col_idx]).strip() if pd.notna(row.iloc[col_idx]) else ''
            if not val or val.lower() == 'nan':
                continue

            # S2 has a sub-label like "Disordered" / "Folded" in col-1 —
            # skip any short non-sequence token
            if len(val) < 10 and not val.startswith('>'):
                continue

            if val.startswith('>'):
                pending_header = True          # next meaningful cell = sequence
            elif pending_header:
                # Basic sanity: looks like an amino-acid sequence
                clean = re.sub(r'[^A-Za-z]', '', val)
                if len(clean) > 5:
                    sequences.append(clean.upper())
                    labels.append(current_label)
                    pending_header = False

    print(f'  Loaded {len(sequences)} sequences '
          f'(pos={labels.count(1)}, neg={labels.count(0)}) '
          f'from {os.path.basename(path)}')
    return sequences, labels

# ─────────────────────────────────────────────
# 4.  Build / load embeddings cache
# ─────────────────────────────────────────────
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
# 5.  Load training data
# ─────────────────────────────────────────────
print('\n=== Loading training data ===')
pos_seqs = load_flat_sequences(args.train_pos)
neg_seqs = load_flat_sequences(args.train_neg)

X_pos = cached_embed(pos_seqs,
                     os.path.join(args.output_path, 'train_pos_emb.npy'),
                     desc='train-pos')
X_neg = cached_embed(neg_seqs,
                     os.path.join(args.output_path, 'train_neg_emb.npy'),
                     desc='train-neg')

X_train = np.vstack([X_pos, X_neg])
y_train = np.array([1] * len(pos_seqs) + [0] * len(neg_seqs))
print(f'Training set: {X_train.shape}, pos={y_train.sum()}, neg={(y_train==0).sum()}')

# ─────────────────────────────────────────────
# 6.  Train XGBoost
# ─────────────────────────────────────────────
print('\n=== Training XGBoost ===')
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1,
)
xgb_model.fit(X_train, y_train,
              eval_set=[(X_train, y_train)],
              verbose=50)

with open(os.path.join(args.output_path, args.save_model), 'wb') as f:
    pickle.dump(xgb_model, f)
print(f'Model saved to {os.path.join(args.output_path, args.save_model)}')

# ─────────────────────────────────────────────
# 7.  Evaluate on tableS1 – tableS5
# ─────────────────────────────────────────────
test_files = {
    'TableS1': os.path.join(args.test_dir, 'tableS1.xlsx'),
    'TableS2': os.path.join(args.test_dir, 'tableS2.xlsx'),
    'TableS3': os.path.join(args.test_dir, 'tableS3.xlsx'),
    'TableS4': os.path.join(args.test_dir, 'tableS4.xlsx'),
    'TableS5': os.path.join(args.test_dir, 'tableS5.xlsx'),
}

results = []

print('\n=== Evaluating on test sets ===')
for name, fpath in test_files.items():
    if not os.path.exists(fpath):
        print(f'  {name}: file not found, skipping.')
        continue

    print(f'\n--- {name} ---')
    seqs, y_true = load_labeled_sequences(fpath)

    if len(seqs) == 0:
        print(f'  No sequences parsed, skipping.')
        continue

    cache_path = os.path.join(args.output_path, f'{name.lower()}_emb.npy')
    X_test = cached_embed(seqs, cache_path, desc=name)

    y_true = np.array(y_true)
    y_pred  = xgb_model.predict(X_test)
    y_prob  = xgb_model.predict_proba(X_test)[:, 1]

    # Handle edge-case: only one class present in y_true
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float('nan')

    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)

    print(f'  ROC-AUC  : {auc:.4f}')
    print(f'  Accuracy : {acc:.4f}')
    print(f'  F1       : {f1:.4f}')
    print(f'  Precision: {prec:.4f}')
    print(f'  Recall   : {rec:.4f}')

    results.append(dict(
        Dataset=name,
        N=len(y_true),
        N_pos=int(y_true.sum()),
        N_neg=int((y_true == 0).sum()),
        ROC_AUC=round(auc,  4),
        Accuracy=round(acc,  4),
        F1=round(f1,   4),
        Precision=round(prec, 4),
        Recall=round(rec,  4),
    ))

# ─────────────────────────────────────────────
# 8.  Summary table
# ─────────────────────────────────────────────
summary_df = pd.DataFrame(results)
summary_path = os.path.join(args.output_path, 'evaluation_results.csv')
summary_df.to_csv(summary_path, index=False)

print('\n' + '='*60)
print('SUMMARY')
print('='*60)
print(summary_df.to_string(index=False))
print(f'\nResults saved to {summary_path}')

#######################disorder only and folded only
name = 'TableS2'
fpath = os.path.join(args.test_dir, 'tableS2.xlsx')
seqs, y_true = load_labeled_sequences(fpath)
cache_path = os.path.join(args.output_path, f'{name.lower()}_emb.npy')
X_test = cached_embed(seqs, cache_path, desc=name)
y_true = np.array(y_true)

disorder_X=np.vstack([X_test[:100],X_test[200:300]])
disorder_y_true=np.hstack([y_true[:100],y_true[200:300]])
fold_X=np.vstack([X_test[100:200],X_test[300:]])
fold_y_true=np.hstack([y_true[100:200],y_true[300:]])

y_pred  = xgb_model.predict(disorder_X)
y_prob  = xgb_model.predict_proba(disorder_X)[:, 1]
try:
    auc = roc_auc_score(disorder_y_true, y_prob)
except ValueError:
    auc = float('nan')
acc  = accuracy_score(disorder_y_true, y_pred)
f1   = f1_score(disorder_y_true, y_pred, zero_division=0)
prec = precision_score(disorder_y_true, y_pred, zero_division=0)
rec  = recall_score(disorder_y_true, y_pred, zero_division=0)

print(f'  ROC-AUC  : {auc:.4f}')
print(f'  Accuracy : {acc:.4f}')
print(f'  F1       : {f1:.4f}')
print(f'  Precision: {prec:.4f}')
print(f'  Recall   : {rec:.4f}')

y_pred  = xgb_model.predict(fold_X)
y_prob  = xgb_model.predict_proba(fold_X)[:, 1]
try:
    auc = roc_auc_score(fold_y_true, y_prob)
except ValueError:
    auc = float('nan')
acc  = accuracy_score(fold_y_true, y_pred)
f1   = f1_score(fold_y_true, y_pred, zero_division=0)
prec = precision_score(fold_y_true, y_pred, zero_division=0)
rec  = recall_score(fold_y_true, y_pred, zero_division=0)

print(f'  ROC-AUC  : {auc:.4f}')
print(f'  Accuracy : {acc:.4f}')
print(f'  F1       : {f1:.4f}')
print(f'  Precision: {prec:.4f}')
print(f'  Recall   : {rec:.4f}')