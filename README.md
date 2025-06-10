# MINI Project

This repository contains a lip-syncing project using Wav2Lip technology. It includes video processing scripts, input videos, and audio files for generating lip-synced videos in multiple languages.

## Structure

- `input_audios/` - Original and translated audio files
- `input_videos/` - Source videos for lip-syncing
- `input_videos_mp4_only/` - MP4 format videos ready for processing
- `preprocessed_data/` - Intermediate processed data
- `results/` - Output files including transcripts and processed videos
- `Wav2Lip/` - The Wav2Lip model and implementation

## Usage

1. Place source videos in the `input_videos` folder
2. Place audio files in the `input_audios` folder
3. Run the preprocessing scripts from the Wav2Lip folder
4. Generate lip-synced videos using the inference script

## Requirements

See `Wav2Lip/requirements.txt` for dependencies.