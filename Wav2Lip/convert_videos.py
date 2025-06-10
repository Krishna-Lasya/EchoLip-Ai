import os
import subprocess

input_folder = r"E:\MINI_0.1\input_videos_flat"

# Collect all .mpg files first
mpg_files = []
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.lower().endswith(".mpg"):
            mpg_files.append(os.path.join(root, file))

total_files = len(mpg_files)
print(f"Total videos to convert: {total_files}")

for idx, input_path in enumerate(mpg_files, start=1):
    output_path = os.path.splitext(input_path)[0] + ".mp4"

    if not os.path.exists(output_path):
        print(f"Processing video {idx}/{total_files}: {os.path.basename(input_path)}")
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-strict", "experimental",
            output_path
        ])
    else:
        print(f"Skipping video {idx}/{total_files} (already converted): {os.path.basename(input_path)}")

print(f"Successfully converted all {total_files} videos to mp4!")
