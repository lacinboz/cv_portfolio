import cv2
import os


def frame_capture(path, frames_per_second=3):
    """
    Extracts frames from a video file at a specified rate.

    Args:
        path (str): Path to the video file
        frames_per_second (int, optional): Number of frames to extract per second. Defaults to 3.

    Returns:
        list: List of extracted image frames
    """
    image_list = []

    vidObj = cv2.VideoCapture(path)

    # Get the frame rate of the video
    fps = int(vidObj.get(cv2.CAP_PROP_FPS))

    # Used as counter variable
    count = 0

    # checks whether frames were extracted
    success = True

    # Calculate how many frames between each save
    save_interval = fps // frames_per_second

    while success:
        success, image = vidObj.read()
        if not success:
            break

        # Save frames at the specified rate
        if count % save_interval == 0:
            image_list.append(image)
        count += 1

    return image_list


def generate_video(image_folder, video_name, frame_rate=10):
    """
    Generates a video from a sequence of images.

    Args:
        image_folder (str): Path to the folder containing images
        video_name (str): Name of the output video file
        frame_rate (int, optional): Frame rate of the output video. Defaults to 10.

    Returns:
        str: Path to the generated video file
    """
    images = [
        img
        for img in os.listdir(image_folder)
        if img.endswith((".jpg", ".jpeg", ".png"))
    ]
    images.sort(
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )  # Sort by frame number

    # Set frame from the first image
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Video writer to create video file
    video = cv2.VideoWriter(
        video_name, cv2.VideoWriter_fourcc(*"DIVX"), frame_rate, (width, height)
    )

    # Appending images to video
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    # Release the video file
    video.release()
    cv2.destroyAllWindows()

    return video_name
