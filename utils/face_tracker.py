# utils/face_tracker.py
# Face tracking with unique IDs
#
# What this does:
# 1. Detects multiple faces in each frame
# 2. Assigns a unique ID to each face
# 3. Tracks the same face across frames
# 4. Gives separate stress prediction for each face
# 5. Removes faces that disappear from frame
# ============================================================

import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict

class FaceTracker:
    """
    Tracks multiple faces across video frames using centroid tracking.

    How centroid tracking works:
    - Each face has a center point (centroid)
    - When new faces appear, we check which existing face
      centroid is closest to the new detection
    - If close enough = same face, keep same ID
    - If too far = new face, assign new ID
    """

    def __init__(self, max_disappeared=20, max_distance=80):
        """
        Parameters:
            max_disappeared (int): How many frames a face can be
                                   missing before we remove its ID
            max_distance    (int): Maximum pixel distance between
                                   centroids to consider same face
        """

        # Counter for assigning unique IDs
        # Starts at 0, increments for each new face
        self.next_id = 0

        # OrderedDict stores {face_id: centroid}
        # centroid = (x_center, y_center) of face bounding box
        self.faces = OrderedDict()

        # Tracks how many frames each face has been missing
        # {face_id: number_of_frames_disappeared}
        self.disappeared = OrderedDict()

        # Stores bounding box for each face ID
        # {face_id: (x, y, w, h)}
        self.bboxes = OrderedDict()

        # Stores stress prediction for each face ID
        # {face_id: {"label": "High Stress", "confidence": 85.2}}
        self.predictions = OrderedDict()

        self.max_disappeared = max_disappeared
        self.max_distance    = max_distance


    def register(self, centroid, bbox):
        """
        Registers a new face with a unique ID.

        Parameters:
            centroid: (cx, cy) center point of face
            bbox    : (x, y, w, h) bounding box
        """
        # Assign current counter value as this face's ID
        face_id = self.next_id

        # Store centroid and bbox
        self.faces[face_id]       = centroid
        self.bboxes[face_id]      = bbox
        self.disappeared[face_id] = 0

        # Initialize with unknown prediction
        self.predictions[face_id] = {
            "label"     : "Detecting...",
            "confidence": 0.0,
            "all_probs" : [0.33, 0.33, 0.34]
        }

        # Increment counter for next face
        self.next_id += 1

        return face_id


    def deregister(self, face_id):
        """
        Removes a face ID when it disappears for too long.
        """
        del self.faces[face_id]
        del self.disappeared[face_id]
        del self.bboxes[face_id]
        del self.predictions[face_id]


    def update_prediction(self, face_id, label, confidence, all_probs):
        """
        Updates the stress prediction for a specific face ID.

        Parameters:
            face_id   : the unique face ID
            label     : 'Low Stress', 'Moderate Stress', 'High Stress'
            confidence: confidence percentage
            all_probs : list of all 3 class probabilities
        """
        if face_id in self.predictions:
            self.predictions[face_id] = {
                "label"     : label,
                "confidence": confidence,
                "all_probs" : all_probs
            }


    def update(self, detections):
        """
        Updates tracker with new face detections from current frame.

        Parameters:
            detections: list of (x, y, w, h) bounding boxes

        Returns:
            dict: {face_id: (x, y, w, h)} for all tracked faces
        """

        # ── No faces detected in this frame ───────────────────
        if len(detections) == 0:

            # Increment disappeared counter for all tracked faces
            for face_id in list(self.disappeared.keys()):
                self.disappeared[face_id] += 1

                # Remove face if missing too long
                if self.disappeared[face_id] > self.max_disappeared:
                    self.deregister(face_id)

            return self.bboxes

        # ── Calculate centroids for new detections ─────────────
        # Centroid = center point of each bounding box
        new_centroids = []
        for (x, y, w, h) in detections:
            cx = x + w // 2   # center x
            cy = y + h // 2   # center y
            new_centroids.append((cx, cy))

        new_centroids = np.array(new_centroids)

        # ── No existing faces — register all as new ────────────
        if len(self.faces) == 0:
            for i, (centroid, bbox) in enumerate(
                    zip(new_centroids, detections)):
                self.register(centroid, bbox)

        # ── Match new detections to existing faces ─────────────
        else:
            # Get existing face IDs and their centroids
            existing_ids       = list(self.faces.keys())
            existing_centroids = np.array(list(self.faces.values()))

            # Calculate distance between every existing centroid
            # and every new centroid
            # Result is a matrix of shape (existing, new)
            D = dist.cdist(existing_centroids, new_centroids)

            # For each existing face, find closest new detection
            # rows = sorted by minimum distance value
            rows = D.min(axis=1).argsort()

            # For each new detection, find closest existing face
            cols = D.argmin(axis=1)[rows]

            # Track which rows and cols we have already used
            used_rows = set()
            used_cols = set()

            # Loop through sorted (row, col) pairs
            for (row, col) in zip(rows, cols):

                # Skip if already matched
                if row in used_rows or col in used_cols:
                    continue

                # Skip if distance is too large (different face)
                if D[row, col] > self.max_distance:
                    continue

                # Match found — update this face ID with new data
                face_id = existing_ids[row]
                self.faces[face_id]       = new_centroids[col]
                self.bboxes[face_id]      = detections[col]
                self.disappeared[face_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            # Find unmatched existing faces
            unused_rows = set(range(D.shape[0])) - used_rows

            # Increment disappeared for unmatched existing faces
            for row in unused_rows:
                face_id = existing_ids[row]
                self.disappeared[face_id] += 1
                if self.disappeared[face_id] > self.max_disappeared:
                    self.deregister(face_id)

            # Find unmatched new detections — register as new faces
            unused_cols = set(range(D.shape[1])) - used_cols
            for col in unused_cols:
                self.register(new_centroids[col], detections[col])

        return self.bboxes