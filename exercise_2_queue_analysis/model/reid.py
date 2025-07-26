import torch
import torch.nn.functional as F
import uuid
import time
from collections import defaultdict, deque


class PersonReID:
    """
    Person Re-Identification system for tracking individuals across camera views.

    This class manages the embedding history, person tracking, and queue analysis.
    """

    def __init__(self, extractor, similarity_threshold=0.66):
        """
        Initialize the PersonReID system.

        Args:
            extractor: Feature extractor model for generating person embeddings
            similarity_threshold (float, optional): Threshold for matching identities. Defaults to 0.66.
        """
        self.extractor = extractor
        self.similarity_threshold = similarity_threshold

        # ReID gallery with embedding history
        self.embedding_history = defaultdict(lambda: deque(maxlen=1000))
        self.embedding_dict = {}
        self.people_in_queue = {}
        self.timing_dict = {}
        self.queue_order_dict = {}
        self.time_diff = []
        self.gallery_ids = []
        self.avg_time = None

    def assign_id_for_person(self, person_features, conf):
        """
        Assign an ID to a person based on feature similarity.

        Args:
            person_features: Extracted features for the person
            conf (float): Confidence score of the detection

        Returns:
            str: Person ID or None if no match found with sufficient confidence
        """
        person_features = (
            F.normalize(torch.tensor(person_features), p=2, dim=1).cpu().numpy()[0]
        )  # shape: (512,)

        best_id = None
        best_score = -1

        for pid, history in self.embedding_history.items():
            for prev_feat in history:
                score = F.cosine_similarity(
                    torch.tensor(person_features), torch.tensor(prev_feat), dim=0
                ).item()
                if score > best_score:
                    best_score = score
                    best_id = pid

        print("best_score", best_score)

        if best_score > self.similarity_threshold:
            self.embedding_history[best_id].append(person_features)
            return best_id
        else:
            if conf > 0.81:
                new_id = str(uuid.uuid4())
                self.embedding_dict[new_id] = person_features
                self.people_in_queue[new_id] = person_features
                self.embedding_history[new_id].append(person_features)
                self.gallery_ids.append(new_id)
                self.timing_dict[new_id] = time.time()
                return new_id
            else:
                return None

    def extract_features(self, crop):
        """
        Extract features from a person crop.

        Args:
            crop: Cropped image of a person

        Returns:
            tensor: Extracted features
        """
        return self.extractor(crop)

    def update_queue_order(self, person_id, x1, y1=0, y1_threshold=650):
        """
        Update the queue order for a person.

        Args:
            person_id (str): ID of the person
            x1 (int): X-coordinate of the person's bounding box
            y1 (int, optional): Y-coordinate of the person's bounding box. Defaults to 0.
            y1_threshold (int, optional): Y-threshold for queue consideration. Defaults to 650.

        Returns:
            bool: True if the person is in the queue area, False otherwise
        """
        if y1 < y1_threshold:
            if self.queue_order_dict.get(person_id) is None:
                self.queue_order_dict[person_id] = x1
            else:
                self.queue_order_dict[person_id] = max(
                    self.queue_order_dict[person_id], x1
                )
            return True
        return False

    def check_exit(self, person_id, x2, exit_threshold=1910):
        """
        Check if a person has exited the queue.

        Args:
            person_id (str): ID of the person
            x2 (int): Right X-coordinate of the person's bounding box
            exit_threshold (int, optional): X-threshold for exit consideration. Defaults to 1910.

        Returns:
            bool: True if the person has exited, False otherwise
        """
        if x2 > exit_threshold:
            if person_id is not None:
                self.time_diff.append(time.time() - self.timing_dict[person_id])
                if person_id in self.people_in_queue:
                    del self.people_in_queue[person_id]
                return True
        return False

    def calculate_eta(self, order):
        """
        Calculate estimated time of arrival for a person in the queue.

        Args:
            order (int): Position in the queue

        Returns:
            int: Estimated time in seconds or None if not available
        """
        if self.avg_time is not None:
            eta = (order - 1) * self.avg_time
            if eta < 0:
                eta = 0
            return round(eta)
        return None

    def update_average_time(self, person_id):
        """
        Update the average time based on a person exiting the queue.

        Args:
            person_id (str): ID of the person exiting
        """
        self.avg_time = (time.time() - self.timing_dict[person_id]) / 20
