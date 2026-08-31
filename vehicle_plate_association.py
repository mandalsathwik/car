from collections import Counter
import logging
import config

logger = logging.getLogger("IBVAP.VehiclePlateAssociation")

class TrackPlateState:
    def __init__(self, track_id, vehicle_class):
        self.track_id = track_id
        self.vehicle_class = vehicle_class
        self.best_plate_text = "NOT DETECTED"
        self.best_plate_confidence = 0.0
        self.ocr_history = []  # List of (plate_text, confidence)
        self.observation_count = 0
        self.last_ocr_frame = -999

    def add_ocr_observation(self, plate_text, confidence, frame_number):
        self.last_ocr_frame = frame_number

        if "UNKNOWN" in plate_text or "NOT DETECTED" in plate_text:
            return

        self.ocr_history.append((plate_text, confidence))
        self.observation_count += 1

        # Keep history within temporal voting window
        if len(self.ocr_history) > config.TEMPORAL_VOTING_WINDOW:
            self.ocr_history.pop(0)

        # Consolidate plate using temporal voting & confidence weighting
        self._consolidate()

    def _consolidate(self):
        """
        Consolidate plate readings using frequency voting weighted by confidence.
        Prevents overwriting a strong reading with a lower-confidence temporary artifact.
        """
        if not self.ocr_history:
            return

        plate_scores = {}
        plate_counts = Counter()

        for text, conf in self.ocr_history:
            plate_counts[text] += 1
            plate_scores[text] = plate_scores.get(text, 0.0) + conf

        # Calculate average confidence per candidate text
        avg_confidences = {text: plate_scores[text] / plate_counts[text] for text in plate_scores}

        # Select candidate with highest combined score (frequency * avg_confidence)
        best_candidate = max(plate_scores.keys(), key=lambda t: (plate_counts[t], avg_confidences[t]))
        candidate_conf = avg_confidences[best_candidate]

        # Do not overwrite if existing best plate has significantly higher confidence unless candidate has strong consensus
        if candidate_conf >= self.best_plate_confidence or plate_counts[best_candidate] >= 3:
            self.best_plate_text = best_candidate
            self.best_plate_confidence = candidate_conf


class VehiclePlateAssociator:
    def __init__(self):
        self.track_states = {}  # track_id -> TrackPlateState

    def get_or_create_state(self, track_id, vehicle_class):
        if track_id not in self.track_states:
            self.track_states[track_id] = TrackPlateState(track_id, vehicle_class)
        return self.track_states[track_id]

    def update_track_ocr(self, track_id, vehicle_class, plate_text, ocr_confidence, frame_number):
        state = self.get_or_create_state(track_id, vehicle_class)
        state.add_ocr_observation(plate_text, ocr_confidence, frame_number)
        return state

    def should_run_ocr(self, track_id, frame_number):
        """
        Rate-limit OCR runs per vehicle track to preserve performance FPS.
        Skip OCR if we already have a strong consolidated plate for this vehicle.
        """
        if track_id not in self.track_states:
            return True

        state = self.track_states[track_id]
        
        # If plate is already verified with high confidence (>0.85) over multiple frames, skip OCR
        if state.best_plate_confidence > 0.85 and state.observation_count >= 3:
            return False

        # Otherwise run OCR every OCR_INTERVAL frames
        return (frame_number - state.last_ocr_frame) >= config.OCR_INTERVAL

    def get_track_plate(self, track_id):
        if track_id in self.track_states:
            state = self.track_states[track_id]
            return state.best_plate_text, state.best_plate_confidence
        return "NOT DETECTED", 0.0
