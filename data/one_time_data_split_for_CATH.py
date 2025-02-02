import os
import random
import shutil

# Set the seed for reproducibility
random_seed = 42
random.seed(random_seed)

# Set the path to your folder containing HDF5 files
folder_path = "/data/duolin/CATH/CATH_4_3_0_nonredundant_basedon_S40pdb_gvp"

# Set the paths for training, validation, and test folders
train_path = "/data/duolin/CATH/CATH-pos-512-gvp_splitdata/train"
val_path = "/data/duolin/CATH/CATH-pos-512-gvp_splitdata/val"
test_path = "/data/duolin/CATH/CATH_4_3_0_non-rep_gvp/"
all_test_files = os.listdir(test_path)

if not os.path.exists(train_path):
    os.makedirs(train_path)

if not os.path.exists(val_path):
    os.makedirs(val_path)


# List all files in the folder
all_files = os.listdir(folder_path)#31847
#remove all_test_files from all_files, but current all_test_files is only prefix of fullname
valid_files = [item for item in all_files if not any(item.startswith(prefix.replace('.h5','')) for prefix in all_test_files)] #30316



# Shuffle the list of files randomly
random.shuffle(valid_files)
numoffiles = len(valid_files)  # 30316
# Define the number of files for each split
num_train = 27285 #30316 *0.9
#num_val = 3031

# Split the shuffled files into training, validation, and test sets
train_files = valid_files[:num_train]
val_files = valid_files[num_train:]

# copy files to their respective folders
for file in train_files:
    shutil.copy(os.path.join(folder_path, file), os.path.join(train_path, file))

for file in val_files:
    shutil.copy(os.path.join(folder_path, file), os.path.join(val_path, file))



import os
train_path = "./train"
val_path = "./val"

for file in os.listdir(val_path):
    basename=os.path.basename(file)
    if len(basename.split("."))<5:
        print(file)
        os.remove(os.path.join(val_path,file))


for file in os.listdir(train_path):
    basename=os.path.basename(file)
    if len(basename.split("."))<5:
        print(file)
        os.remove(os.path.join(train_path,file))

