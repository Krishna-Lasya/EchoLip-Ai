import os
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import logging
from datetime import datetime

# Set up logging
log_dir = r"E:\MINI_0.1\Wav2Lip\logs"
checkpoint_dir = r"E:\MINI_0.1\Wav2Lip\checkpoints"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"train_colorsyncnet_{datetime.now().strftime('%Y%m%d_%H%M')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),  # Save to file
        logging.StreamHandler()  # Print to console
    ]
)
logger = logging.getLogger(__name__)

# SyncNet model (based on Wav2Lip's SyncNet: https://github.com/Rudrabha/Wav2Lip/blob/master/models/syncnet.py)
class SyncNet(nn.Module):
    def __init__(self):
        super(SyncNet, self).__init__()
        # Video stream: [batch, 1, 5, 112, 112] (grayscale, 5 frames)
        self.video_cnn = nn.Sequential(
            nn.Conv3d(1, 96, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3)),
            nn.BatchNorm3d(96),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            nn.Conv3d(96, 256, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 2, 2)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
        )
        # Output: [batch, 256, 1, 6, 6] -> 256 * 1 * 6 * 6 = 9216
        self.video_fc = nn.Linear(256 * 1 * 6 * 6, 128)

        # Audio stream: [batch, 13, 20]
        self.audio_cnn = nn.Sequential(
            nn.Conv1d(13, 96, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Conv1d(96, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.audio_fc = nn.Linear(256 * 5, 128)

    def forward(self, video, audio):
        try:
            video_feat = self.video_cnn(video)
            video_feat = video_feat.view(video_feat.size(0), -1)
            video_feat = self.video_fc(video_feat)

            audio_feat = self.audio_cnn(audio)
            audio_feat = audio_feat.view(audio_feat.size(0), -1)
            audio_feat = self.audio_fc(audio_feat)

            return video_feat, audio_feat
        except Exception as e:
            logger.error(f"Error in SyncNet forward pass: {str(e)}")
            raise

# Contrastive loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, video_feat, audio_feat, label):
        try:
            dist = torch.norm(video_feat - audio_feat, p=2, dim=1)
            loss = label * dist**2 + (1 - label) * torch.clamp(self.margin - dist, min=0)**2
            return loss.mean()
        except Exception as e:
            logger.error(f"Error in ContrastiveLoss: {str(e)}")
            raise

# Custom dataset
class SyncNetDataset(Dataset):
    def __init__(self, data_root, sequence_length=5, img_size=(112, 112), mfcc_length=20):
        self.data_root = data_root
        self.sequence_length = sequence_length
        self.mfcc_length = mfcc_length
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.img_size = img_size
        self.videos = [os.path.join(data_root, d) for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        self.frame_data = []
        logger.info(f"Found {len(self.videos)} video directories in {data_root}")
        for video_dir in self.videos:
            meta_path = os.path.join(video_dir, "faces.json")
            if not os.path.exists(meta_path):
                logger.warning(f"Missing faces.json in {video_dir}")
                continue
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                frame_meta = meta["frame_meta"]
                faces = meta["faces"]
                frame_names = sorted(frame_meta.keys(), key=lambda x: int(x.split("_")[1].split(".")[0]))
                for i in range(len(frame_names) - self.sequence_length + 1):
                    seq_frames = frame_names[i:i + self.sequence_length]
                    seq_faces = []
                    for fn in seq_frames:
                        face = next((f for f in faces if f["frame"] == fn), None)
                        if face:
                            seq_faces.append(face)
                    if len(seq_faces) == self.sequence_length:
                        self.frame_data.append({
                            "video_dir": video_dir,
                            "frame_names": seq_frames,
                            "face_paths": [f["face_path"] for f in seq_faces],
                            "mfcc_start": frame_meta[seq_frames[self.sequence_length // 2]]["mfcc_start"],
                            "mfcc_end": frame_meta[seq_frames[self.sequence_length // 2]]["mfcc_end"]
                        })
            except Exception as e:
                logger.error(f"Error processing {meta_path}: {str(e)}")
                continue
        logger.info(f"Created {len(self.frame_data)} sequences for training")

    def __len__(self):
        return len(self.frame_data)

    def __getitem__(self, idx):
        try:
            data = self.frame_data[idx]
            video = []
            for face_path in data["face_paths"]:
                img = cv2.imread(face_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    logger.error(f"Failed to load image: {face_path}")
                    raise ValueError(f"Failed to load image: {face_path}")
                img = cv2.resize(img, self.img_size)
                img = self.transform(img)
                video.append(img)
            video = torch.stack(video, dim=1)

            mfcc_path = os.path.join(data["video_dir"], "mfccs.npy")
            mfccs = np.load(mfcc_path)
            mfcc_start = data["mfcc_start"]
            mfcc_end = data["mfcc_end"]
            audio = mfccs[:, mfcc_start:mfcc_end]
            if audio.shape[1] < self.mfcc_length:
                pad = np.zeros((audio.shape[0], self.mfcc_length - audio.shape[1]))
                audio = np.concatenate([audio, pad], axis=1)
            elif audio.shape[1] > self.mfcc_length:
                audio = audio[:, :self.mfcc_length]
            audio = torch.FloatTensor(audio)

            label = 1
            if random.random() > 0.5:
                label = 0
                other_idx = random.randint(0, len(self.frame_data) - 1)
                other_data = self.frame_data[other_idx]
                other_mfcc_path = os.path.join(other_data["video_dir"], "mfccs.npy")
                other_mfccs = np.load(other_mfcc_path)
                other_start = other_data["mfcc_start"]
                other_end = other_data["mfcc_end"]
                other_audio = other_mfccs[:, other_start:other_end]
                if other_audio.shape[1] < self.mfcc_length:
                    pad = np.zeros((other_audio.shape[0], self.mfcc_length - other_audio.shape[1]))
                    other_audio = np.concatenate([other_audio, pad], axis=1)
                elif other_audio.shape[1] > self.mfcc_length:
                    other_audio = other_audio[:, :self.mfcc_length]
                audio = torch.FloatTensor(other_audio)

            return video, audio, label
        except Exception as e:
            logger.error(f"Error in dataset __getitem__ at index {idx}: {str(e)}")
            raise

# Training loop
def train(model, dataloader, optimizer, criterion, device, epochs=300, checkpoint_dir=checkpoint_dir):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        try:
            for batch_idx, (video, audio, label) in enumerate(dataloader):
                video, audio, label = video.to(device), audio.to(device), label.to(device)
                optimizer.zero_grad()
                video_feat, audio_feat = model(video, audio)
                loss = criterion(video_feat, audio_feat, label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f"syncnet_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error in training epoch {epoch+1}: {str(e)}")
            raise

if __name__ == "__main__":
    logger.info("Starting training script")
    # Paths and hyperparameters
    data_root = r"E:\MINI_0.1\preprocessed_data"
    batch_size = 8  # For CPU
    epochs = 300  # Set to 300 epochs
    learning_rate = 0.001
    device = torch.device("cpu")  # Explicitly use CPU
    logger.info(f"Using device: {device}")

    # Dataset and DataLoader
    try:
        dataset = SyncNetDataset(data_root, sequence_length=5, img_size=(112, 112), mfcc_length=20)
        if len(dataset) == 0:
            logger.error("Dataset is empty. Re-run process_videos.py with --frame-interval 1 to extract more frames.")
            print("ERROR: Dataset is empty. Run: python process_videos.py --input-dir \"E:\\MINI_0.1\\input_videos_mp4_only\" --output-dir \"E:\\MINI_0.1\\preprocessed_data\" --frame-interval 1 --max-workers 5")
            exit(1)
        logger.info(f"Dataset size: {len(dataset)} samples")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    except Exception as e:
        logger.error(f"Error initializing dataset or dataloader: {str(e)}")
        raise

    # Model, loss, optimizer
    try:
        model = SyncNet().to(device)
        criterion = ContrastiveLoss(margin=1.0)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        logger.info("Model, criterion, and optimizer initialized")
    except Exception as e:
        logger.error(f"Error initializing model or optimizer: {str(e)}")
        raise

    # Train
    try:
        train(model, dataloader, optimizer, criterion, device, epochs, checkpoint_dir)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise