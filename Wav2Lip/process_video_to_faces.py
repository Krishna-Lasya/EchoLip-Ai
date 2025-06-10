import cv2
import os
import json

def process_video(video_path, output_base_dir, frame_interval=10):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(output_base_dir, video_name)
    frames_dir = os.path.join(output_dir, "frames")
    faces_dir = os.path.join(output_dir, "faces")
    meta_path = os.path.join(output_dir, "faces.json")

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(faces_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open: {video_path}")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    frame_count = 0
    saved_faces = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_name = f"frame_{frame_count:05d}.jpg"
            frame_path = os.path.join(frames_dir, frame_name)
            cv2.imwrite(frame_path, frame)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for i, (x, y, w, h) in enumerate(faces):
                face_img = frame[y:y+h, x:x+w]
                face_path = os.path.join(faces_dir, f"{frame_name[:-4]}_face_{i}.jpg")
                cv2.imwrite(face_path, face_img)

                saved_faces.append({
                    "frame": frame_name,
                    "face_index": i,
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "face_path": face_path
                })

        frame_count += 1

    cap.release()

    with open(meta_path, 'w') as f:
        json.dump(saved_faces, f, indent=2)

    print(f"✅ {video_name}: Processed {frame_count} frames, saved {len(saved_faces)} faces to {output_dir}")


def process_all_videos(input_folder, output_folder):
    for file in os.listdir(input_folder):
        if file.lower().endswith(".mp4"):
            video_path = os.path.join(input_folder, file)
            process_video(video_path, output_folder)


# === Paths ===
input_videos_dir = r"E:\MINI_0.1\input_videos_mp4_only"
output_processed_dir = r"E:\MINI_0.1\preprocessed_data"

process_all_videos(input_videos_dir, output_processed_dir)
