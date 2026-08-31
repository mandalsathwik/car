import logging
from ultralytics import YOLO
import config

logger = logging.getLogger("IBVAP.VehicleTracking")

class VehicleTracker:
    def __init__(self, detector):
        """
        Uses detector's YOLO model instance for persistent tracking via ByteTrack/BoTSORT.
        """
        self.detector = detector

    def track(self, frame, frame_number):
        """
        Runs model.track on the current frame.
        Returns a list of active track objects:
        [{
           'track_id': int,
           'vehicle_class': str,
           'confidence': float,
           'bbox': (x1, y1, x2, y2),
           'center_point': (cx, cy),
           'frame_number': int
        }]
        """
        results = self.detector.model.track(
            source=frame,
            persist=True,
            tracker=config.TRACKER_CONFIG,
            imgsz=config.IMG_SIZE,
            conf=config.CONF_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            classes=self.detector.target_class_ids if self.detector.target_class_ids else None,
            verbose=False
        )

        current_tracks = []

        if len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            class_ids = results[0].boxes.cls.int().cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()

            for box, track_id, cls_id, conf in zip(boxes, track_ids, class_ids, confidences):
                x1, y1, x2, y2 = map(int, box)
                track_id = int(track_id)
                cls_id = int(cls_id)
                cls_name = self.detector.class_id_to_name.get(cls_id, self.detector.model.names.get(cls_id, "vehicle")).upper()
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                track_info = {
                    'track_id': track_id,
                    'vehicle_class': cls_name,
                    'confidence': float(conf),
                    'bbox': (x1, y1, x2, y2),
                    'center_point': (cx, cy),
                    'frame_number': frame_number
                }

                current_tracks.append(track_info)

        return current_tracks
