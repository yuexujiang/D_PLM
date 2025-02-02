import os
import random
import shutil

# Set the seed for reproducibility
random_seed = 42
random.seed(random_seed)

# Set the path to your folder containing HDF5 files
folder_path = "/data/duolin/swiss-pos-512-v2"

# Set the paths for training, validation, and test folders
train_path = "/data/duolin/swiss-pos-512-v2_splitdata/train"
val_path = "/data/duolin/swiss-pos-512-v2_splitdata/val"
test_path = "/data/duolin/swiss-pos-512-v2_splitdata/test"
if not os.path.exists(train_path):
    os.makedirs(train_path)

if not os.path.exists(val_path):
    os.makedirs(val_path)

if not os.path.exists(test_path):
    os.makedirs(test_path)

# List all files in the folder
all_files = os.listdir(folder_path)

# Shuffle the list of files randomly
random.shuffle(all_files)
numoffiles = len(all_files)  # 542378
# Define the number of files for each split
num_train = 540000
num_val = 1000
# num_test = 339

# Split the shuffled files into training, validation, and test sets
train_files = all_files[:num_train]
val_files = all_files[num_train:num_train + num_val]
test_files = all_files[num_train + num_val:]

# copy files to their respective folders
for file in train_files:
    shutil.copy(os.path.join(folder_path, file), os.path.join(train_path, file))

for file in val_files:
    shutil.copy(os.path.join(folder_path, file), os.path.join(val_path, file))

for file in test_files:
    shutil.copy(os.path.join(folder_path, file), os.path.join(test_path, file))
