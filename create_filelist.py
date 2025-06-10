import os

data_root = r"E:\MINI_0.1\preprocessed_data"
filelist_path = r"E:\MINI_0.1\Wav2Lip\filelists\train.txt"

# Create directory if it doesn't exist
os.makedirs(os.path.dirname(filelist_path), exist_ok=True)

with open(filelist_path, "w") as f:
    for folder_name in os.listdir(data_root):
        # Assuming each folder_name corresponds to a training sample
        f.write(folder_name + "\n")