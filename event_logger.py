import csv
import os
import logging
from datetime import datetime
import config

logger = logging.getLogger("IBVAP.EventLogger")

class EventLogger:
    def __init__(self, output_path=config.OUTPUT_CSV):
        self.output_path = output_path
        self.logged_events = {} # track_id -> last_logged_plate_text
        self._initialize_csv()

    def _initialize_csv(self):
        # Create CSV header if file doesn't exist
        if not os.path.exists(self.output_path):
            with open(self.output_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "frame_number",
                    "track_id",
                    "vehicle_type",
                    "vehicle_confidence",
                    "plate_text",
                    "plate_confidence",
                    "x1",
                    "y1",
                    "x2",
                    "y2"
                ])
            logger.info(f"Initialized CSV log file at '{self.output_path}'")

    def log_vehicle_event(self, frame_number, track_id, vehicle_type, vehicle_conf, plate_text, plate_conf, bbox):
        """
        Log event to CSV only when a meaningful plate update occurs or for new tracked vehicles.
        Prevents frame-by-frame duplicate spam.
        """
        # If we already logged this track ID with the exact same plate status, skip duplicate log
        if track_id in self.logged_events and self.logged_events[track_id] == plate_text:
            return

        # Don't re-log UNKNOWN if we already logged UNKNOWN
        if track_id in self.logged_events and "UNKNOWN" in self.logged_events[track_id] and "UNKNOWN" in plate_text:
            return

        x1, y1, x2, y2 = bbox
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.output_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                frame_number,
                track_id,
                vehicle_type,
                f"{vehicle_conf:.2f}",
                plate_text,
                f"{plate_conf:.2f}",
                x1,
                y1,
                x2,
                y2
            ])

        self.logged_events[track_id] = plate_text
        logger.info(
            f"Event Logged [Frame {frame_number}] | Track ID: {track_id} | Type: {vehicle_type} | "
            f"Plate: {plate_text} (Conf: {plate_conf:.2f})"
        )
