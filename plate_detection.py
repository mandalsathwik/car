import os
import logging
from ultralytics import YOLO
import config

logger = logging.getLogger("IBVAP.PlateDetection")

class PlateDetector:
    def __init__(self, model_path=config.PLATE_MODEL_PATH):
        self.model_path = model_path
        self.model = None

        if not os.path.exists(model_path):
            logger.warning(
                f"License plate model NOT FOUND at '{model_path}'. "
                f"Plate detection is DISABLED. "
                f"Place a YOLOv8 license plate model at '{model_path}' to enable it."
            )
        else:
            logger.info(f"Loading License Plate YOLO model from '{model_path}'...")
            self.model = YOLO(model_path)
            logger.info("License Plate model loaded successfully.")

    def detect_plate(self, vehicle_crop):
        """
        Detect license plate regions inside a vehicle crop using the YOLO model.

        Input : vehicle_crop (BGR numpy array)
        Output: list of plate candidates:
                [{'bbox': (px1, py1, px2, py2), 'crop': plate_img, 'confidence': float}]
                Coordinates are relative to vehicle_crop.
        Returns empty list if model is not loaded.
        """
        if self.model is None:
            return []

        if vehicle_crop is None or vehicle_crop.size == 0:
            return []

        h, w = vehicle_crop.shape[:2]
        if h < 20 or w < 20:
            return []

        results = self.model.predict(
            source=vehicle_crop,
            conf=config.PLATE_CONF_THRESHOLD,
            verbose=False
        )

        plates = []
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()

            for box, conf in zip(boxes, confidences):
                px1, py1, px2, py2 = map(int, box)
                plate_crop = vehicle_crop[
                    max(0, py1):min(h, py2),
                    max(0, px1):min(w, px2)
                ]
                if plate_crop.size > 0:
                    plates.append({
                        'bbox': (px1, py1, px2, py2),
                        'crop': plate_crop,
                        'confidence': float(conf)
                    })

        return plates
