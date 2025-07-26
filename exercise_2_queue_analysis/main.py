import os
import cv2
import matplotlib.pyplot as plt
import torch
import torchreid

from utils.video_processing import frame_capture, generate_video
from model.queue_analyzer import QueueAnalyzer


def main():
    """
    Main function to run the queue analysis system.
    """
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Path to the input video
    video_path = "first-video.mp4"

    # Extract frames from the video
    print(f"Extracting frames from {video_path}...")
    frames = frame_capture(video_path, frames_per_second=3)
    print(f"Extracted {len(frames)} frames from the video.")

    # Display a sample frame with queue boundaries
    if frames:
        sample_frame = frames[0].copy()
        sample_frame = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB)

        # Draw queue boundaries
        sample_frame = cv2.line(sample_frame, (965, 1000), (965, 0), (0, 255, 0), 3)
        sample_frame = cv2.line(sample_frame, (980, 1000), (980, 0), (0, 255, 0), 3)
        sample_frame = cv2.line(sample_frame, (1910, 1000), (1910, 0), (0, 255, 0), 3)
        sample_frame = cv2.line(sample_frame, (0, 650), (1910, 650), (0, 255, 0), 3)

        plt.figure(figsize=(12, 8))
        plt.imshow(sample_frame)
        plt.title("Sample Frame with Queue Boundaries")
        plt.show()

    # Initialize the queue analyzer
    print("Initializing Queue Analyzer...")
    analyzer = QueueAnalyzer(
        yolo_model_path="yolov3u.pt",
        reid_model_name="resnet50",
        reid_model_path="resnet50_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
        device=device,
    )

    # Process the video frames
    print("Processing video frames...")
    output_folder = "output_frames"
    os.makedirs(output_folder, exist_ok=True)
    processed_frames = analyzer.process_video(frames, output_folder=output_folder)

    # Generate output video
    print("Generating output video...")
    output_video = generate_video(
        output_folder, "queue_analysis_result.mp4", frame_rate=10
    )
    print(f"Video generated successfully: {output_video}")


if __name__ == "__main__":
    main()
