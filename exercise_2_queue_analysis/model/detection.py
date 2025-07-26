import cv2
from ultralytics import YOLO


class PersonDetector:
    """
    Person detector using YOLOv3 for queue analysis.

    This class handles person detection and filtering based on size and position.
    """

    def __init__(self, model_path="yolov3u.pt", conf_threshold=0.75):
        """
        Initialize the person detector.

        Args:
            model_path (str, optional): Path to the YOLO model. Defaults to 'yolov3u.pt'.
            conf_threshold (float, optional): Confidence threshold for detections. Defaults to 0.75.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_id = 0

        # Font settings for visualization
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.color = (0, 0, 255)
        self.thickness = 2

    def detect(self, frame):
        """
        Detect persons in a frame.

        Args:
            frame: Input frame for detection

        Returns:
            list: Detection results
        """
        return self.model.predict(
            source=frame, conf=self.conf_threshold, classes=[self.class_id]
        )

    def filter_detections(self, det, conf, min_area=16300, exclude_range=(950, 980)):
        """
        Filter detections based on size and position criteria.

        Args:
            det: Detection coordinates (x1, y1, x2, y2)
            conf: Detection confidence
            min_area (int, optional): Minimum area for valid detections. Defaults to 16300.
            exclude_range (tuple, optional): X-coordinate range to exclude. Defaults to (950, 980).

        Returns:
            tuple: (is_valid, x1, y1, x2, y2, width, height, area, aspect_ratio)
        """
        x1, y1, x2, y2 = map(int, det)

        # Check if detection is in the exclusion range
        if x1 in range(exclude_range[0], exclude_range[1]):
            return False, None, None, None, None, None, None, None, None

        width = abs(x2 - x1)
        height = abs(y2 - y1)
        area = width * height

        w = x2 - x1
        h = y2 - y1
        aspect_ratio = h / (w + 1e-5)

        print("conf", conf)
        print("area", area)
        print("aspect_ratio", aspect_ratio, "width", width, "height", height)

        # Filter by area
        if area < min_area:
            return False, None, None, None, None, None, None, None, None

        # if conf < 0.7 and area < 18100:
        #     return False, None, None, None, None, None, None, None, None
        # if height < 180:
        #     return False, None, None, None, None, None, None, None, None

        return True, x1, y1, x2, y2, width, height, area, aspect_ratio

    def draw_detection(
        self, frame, x1, y1, x2, y2, person_id=None, order=None, eta=None
    ):
        """
        Draw detection box and information on the frame.

        Args:
            frame: Frame to draw on
            x1, y1, x2, y2: Bounding box coordinates
            person_id (str, optional): Person ID. Defaults to None.
            order (int, optional): Queue order. Defaults to None.
            eta (int, optional): Estimated time of arrival. Defaults to None.

        Returns:
            frame: Frame with detection visualization
        """
        if x2 is not None and y2 is not None:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw ID and order information
        if person_id is not None:
            cv2.putText(
                frame,
                f"ID: {person_id[:5]}",
                (x1, y1 - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

            if order is not None:
                cv2.putText(
                    frame,
                    f"#: {order}",
                    (x1, y1 - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )

            if eta is not None:
                cv2.putText(
                    frame,
                    f"ETA:: {eta} sec",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )

        return frame

    def draw_queue_info(self, frame, people_count):
        """
        Draw queue information on the frame.

        Args:
            frame: Frame to draw on
            people_count (int): Number of people in the queue

        Returns:
            frame: Frame with queue information
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 5

        cv2.putText(
            frame,
            f" People count: {people_count}",
            (0, 100),
            font,
            2,
            (255, 255, 255),
            thickness,
        )
        return frame
