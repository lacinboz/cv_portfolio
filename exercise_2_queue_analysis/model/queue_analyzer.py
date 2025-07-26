import cv2
import matplotlib.pyplot as plt
import torch
import torchreid
from model.detection import PersonDetector
from model.reid import PersonReID


class QueueAnalyzer:
    """
    Queue Analyzer for camera-based queue analysis with person re-identification.

    This class integrates person detection and re-identification to analyze queues
    and estimate waiting times.
    """

    def __init__(
        self,
        yolo_model_path="yolov3u.pt",
        reid_model_name="resnet50",
        reid_model_path="resnet50_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
        device=None,
    ):
        """
        Initialize the Queue Analyzer.

        Args:
            yolo_model_path (str, optional): Path to the YOLO model. Defaults to 'yolov3u.pt'.
            reid_model_name (str, optional): Name of the ReID model. Defaults to 'resnet50'.
            reid_model_path (str, optional): Path to the ReID model weights. Defaults to 'resnet50_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth'.
            device (str, optional): Device to run models on. Defaults to None (auto-select).
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Initialize detector
        self.detector = PersonDetector(model_path=yolo_model_path)

        # Initialize feature extractor
        self.extractor = torchreid.utils.FeatureExtractor(
            model_name=reid_model_name, model_path=reid_model_path, device=self.device
        )

        # Initialize ReID system
        self.reid = PersonReID(self.extractor)

        # Queue boundaries
        self.queue_boundary_y = 650
        self.exit_boundary_x = 1910

    def draw_queue_boundaries(self, frame):
        """
        Draw queue boundaries on the frame.

        Args:
            frame: Input frame

        Returns:
            frame: Frame with queue boundaries
        """
        # Draw entry gates
        frame = cv2.line(frame, (965, 1000), (965, 0), (0, 255, 0), 3)
        frame = cv2.line(frame, (980, 1000), (980, 0), (0, 255, 0), 3)

        # Draw exit boundary
        frame = cv2.line(
            frame,
            (self.exit_boundary_x, 1000),
            (self.exit_boundary_x, 0),
            (0, 255, 0),
            3,
        )

        # Draw queue area boundary
        frame = cv2.line(
            frame,
            (0, self.queue_boundary_y),
            (self.exit_boundary_x, self.queue_boundary_y),
            (0, 255, 0),
            3,
        )

        return frame

    def process_frame(self, frame):
        """
        Process a single frame for queue analysis.

        Args:
            frame: Input frame

        Returns:
            tuple: (processed_frame, people_in_queue, queue_order)
        """
        processed_frame = frame.copy()

        # Detect people
        results = self.detector.detect(processed_frame)

        # Reset queue order dictionary for this frame
        queue_order_dict = {}
        self.reid.queue_order_dict = queue_order_dict

        people_location = []

        for result in results:
            queue_order_dict = {}
            self.reid.queue_order_dict = queue_order_dict
            people_location = []

            for det, conf in zip(
                result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()
            ):
                # Filter detections
                valid, x1, y1, x2, y2, width, height, area, aspect_ratio = (
                    self.detector.filter_detections(det, conf)
                )
                if not valid:
                    continue

                # Crop the person from the frame
                crop = frame[y1:y2, x1:x2]

                # Extract features and assign ID
                embedding = self.reid.extract_features(crop)
                person_id = self.reid.assign_id_for_person(embedding, conf)

                # Update queue order if person is in queue area
                in_queue = self.reid.update_queue_order(
                    person_id, x1, y1, self.queue_boundary_y
                )

                # Check if person has exited
                exited = self.reid.check_exit(person_id, x2, self.exit_boundary_x)
                if exited and self.reid.avg_time is None:
                    self.reid.update_average_time(person_id)

                # Store person location for ordering
                avg_x = (x1 + x2) / 2
                people_location.append([x1, y1, person_id, avg_x])

                print("person_id", person_id)
                print("queue_order_dict", queue_order_dict)

                # Draw detection box
                processed_frame = self.detector.draw_detection(
                    processed_frame, x1, y1, x2, y2, person_id
                )

            # Draw queue information
            processed_frame = self.detector.draw_queue_info(
                processed_frame, len(self.reid.people_in_queue)
            )

            print("people_location", people_location)

        # Sort people by position (right to left)
        people_location.sort(key=lambda x: x[3], reverse=True)

        # Assign queue positions and ETAs
        order = 1
        taken_into_account = {}

        for person in people_location:
            x1, y1, person_id, avg_x = person
            if person_id is not None:
                if person_id not in taken_into_account:
                    # Calculate ETA
                    eta = self.reid.calculate_eta(order)

                    print("taken_into_account", taken_into_account)

                    # Draw order and ETA
                    processed_frame = self.detector.draw_detection(
                        processed_frame,
                        x1,
                        y1,
                        None,
                        None,
                        person_id=person_id,
                        order=order,
                        eta=eta,
                    )

                    taken_into_account[person_id] = order
                    order += 1
                else:
                    # Use existing order for this person
                    existing_order = taken_into_account[person_id]
                    eta = self.reid.calculate_eta(existing_order)

                    # Draw order and ETA
                    processed_frame = self.detector.draw_detection(
                        processed_frame,
                        x1,
                        y1,
                        None,
                        None,
                        person_id=person_id,
                        order=existing_order,
                        eta=eta,
                    )

        return processed_frame, len(self.reid.people_in_queue), taken_into_account

    def process_video(
        self, frames, output_folder="output_frames", display_results=True
    ):
        """
        Process a sequence of frames for queue analysis.

        Args:
            frames (list): List of video frames
            output_folder (str, optional): Folder to save output frames. Defaults to 'output_frames'.
            display_results (bool, optional): Whether to display results. Defaults to True.

        Returns:
            list: List of processed frames
        """
        import os

        os.makedirs(output_folder, exist_ok=True)

        processed_frames = []
        frame_count = 1

        for frame in frames:
            # Process the frame
            processed_frame, people_count, queue_order = self.process_frame(frame)

            # Save the processed frame
            cv2.imwrite(f"{output_folder}/frame_{frame_count}.jpg", processed_frame)
            processed_frames.append(processed_frame)

            # Display the result if requested
            if display_results:
                plt.figure(figsize=(12, 8))
                plt.imshow(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
                plt.title(f"Frame {frame_count} - People in queue: {people_count}")
                plt.show()

            frame_count += 1

        return processed_frames
