import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm


import argparse
import torch
import esm
import esm_adapterH
import yaml
from utils.utils import load_configs
from collections import OrderedDict
from peft import LoraConfig, get_peft_model


import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
# ----------------------------
# Setup
# ----------------------------



# ----------------------------
# Define your embedding function
# ----------------------------

def embed_sequence(mt_seq, wt_seq,batch_converter, args):
    data = [
        ("protein1", mt_seq),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    with torch.no_grad():
        mt_representation = args.model(batch_tokens.cuda(),repr_layers=[args.model.num_layers])["representations"][args.model.num_layers]
    mt_representation = mt_representation.squeeze(0) #only one sequence a time
    data = [
        ("protein1", wt_seq),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    with torch.no_grad():
        wt_representation = args.model(batch_tokens.cuda(),repr_layers=[args.model.num_layers])["representations"][args.model.num_layers]
    wt_representation = wt_representation.squeeze(0) #only one sequence a time

    # score = np.linalg.norm((mt_representation-wt_representation).to('cpu').detach().numpy(),axis=1)
    # return np.log(np.mean(score))*-1

    return (mt_representation[0]-wt_representation[0]).to('cpu').detach().numpy()
    """
    Replace this with your actual embedding model.
    This dummy version returns a fixed-length random vector.
    """
    np.random.seed(hash(seq) % 2**32)  # deterministic randomness per sequence
    return np.random.rand(128)

# ----------------------------
# Mutate sequence + embed
# ----------------------------

def apply_mutation(seq, mutation):
    """
    Apply mutation like 'Q2C' on seq.
    """
    ref = mutation[0]
    pos = int(mutation[1:-1]) - 1  # convert to 0-index
    alt = mutation[-1]
    assert seq[pos] == ref, f"Reference mismatch at pos {pos+1}: expected {ref}, got {seq[pos]}"
    mutated = seq[:pos] + alt + seq[pos+1:]
    return mutated

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
    if args.model_type=="s-plm":
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

# Generate embeddings for each mutant sequence


# Your original sequence
#SEQ = "MASTQ..."  # Replace with your full WT sequence

uniprot_id='RL401_YEAST' #
if uniprot_id=='RL401_YEAST':
    SEQ='MQIFVKTLTGKTITLEVESSDTIDNVKSKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGGIIEPSLKALASKYNCDKSVCRKCYARLPPRATNCRKRKCGHTNQLRPKKKLK'
    df = pd.read_csv("../ESM-1v data/data_wedownloaded/41 mutation dataset/ESM-1v-41/test/RL401_YEAST_Bolon2014.csv")
    df = df.iloc[:,:3]

# Load your mutation dataset (adjust if your CSV has headers etc.)
# df = pd.read_csv("your_file.csv", header=None)
df.columns = ["A", "mutation", "effect"]  # mutation in column B, effect in column C

parser = argparse.ArgumentParser(description="DMS regression tasks")
parser.add_argument('--model_type', type=str, required=False, default="s-plm")
parser.add_argument('--config_path', type=str, required=False, default="../results/vivit_2/config.yaml")
parser.add_argument('--model_location', type=str, required=False, 
                    default="../results/vivit_2/checkpoints/checkpoint_best_val_dms_corr.pth")

# safe parse
args = parser.parse_args(args=['--model_type', "s-plm", 
                               '--config_path', "../results/vivit_2/config.yaml",
                               '--model_location', "../results/vivit_2/checkpoints/checkpoint_best_val_dms_corr.pth"])

args.model, args.alphabet = load_model(args)
batch_converter = args.alphabet.get_batch_converter()
    

X = []
y = []
print("Generating embeddings...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    mut = row['mutation']
    effect = row['effect']
    mutated_seq = apply_mutation(SEQ, mut)
    if mutated_seq==SEQ:
        continue
    embedding = embed_sequence(mutated_seq, SEQ, batch_converter, args)
    X.append(embedding)
    y.append(effect)

X = np.array(X)
y = np.array(y)

print("Any NaN in X_train?", np.isnan(X).any())
print("Any inf in X_train?", np.isinf(X).any())

# ----------------------------
# Train and Evaluate at various data splits
# ----------------------------

if X.ndim==1:
    X=X.reshape(-1,1)

splits = [0.8, 0.6, 0.4, 0.2]
# for train_frac in splits:
#     X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_frac, random_state=42)

#     model = Ridge()  # or any other regressor
#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_val)
#     mse = mean_squared_error(y_val, y_pred)
#     r2 = r2_score(y_val, y_pred)

#     print(f"\n=== Split: {int(train_frac*100)}% train / {int((1-train_frac)*100)}% val ===")
#     print(f"R^2 score: {r2:.4f}")
#     print(f"MSE: {mse:.4f}")
# === Hyperparameters ===
INPUT_DIM = X.shape[1]         # your embedding dimension (e.g., 128)
HIDDEN_DIM = 64
EPOCHS = 200
LR = 1e-3
PATIENCE = 15
DROPOUT = 0.2

# === Normalize y ===
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# === Convert to tensors ===
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

# === Train/Val split (e.g., 80/20) ===
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, train_size=0.2, random_state=42)

# === MLP Model ===
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, 1)
        )

    def forward(self, x):
        return self.net(x)

model = MLP()
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# === Early Stopping ===
best_val_loss = float('inf')
best_model_state = None
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    pred_train = model(X_train)
    loss = loss_fn(pred_train, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_val = model(X_val)
        val_loss = loss_fn(pred_val, y_val).item()

    if val_loss < best_val_loss - 1e-5:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        break

# === Evaluation with best model ===
model.load_state_dict(best_model_state)
model.eval()
with torch.no_grad():
    final_pred = model(X_val).numpy()
    final_true = y_val.numpy()
    # Undo y normalization
    final_pred = scaler_y.inverse_transform(final_pred)
    final_true = scaler_y.inverse_transform(final_true)

    r2 = r2_score(final_true, final_pred)
    mse = mean_squared_error(final_true, final_pred)

print(f"\n✅ Best model performance on validation set:")
print(f"R² Score: {r2:.4f}")
print(f"MSE: {mse:.4f}")