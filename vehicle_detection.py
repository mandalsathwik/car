import logging
from ultralytics import YOLO
import config

logger = logging.getLogger("IBVAP.VehicleDetection")

class VehicleDetector:
    def __init__(self, model_path=config.VEHICLE_MODEL_PATH):
        logger.info(f"Loading vehicle detection model from {model_path}...")
        self.model = YOLO(model_path)
        self.target_class_ids = []
        self.class_id_to_name = {}
        self._inspect_and_map_classes()

    def _inspect_and_map_classes(self):
        """
        Dynamically inspect model.names to find matching class IDs for target vehicle types.
        Prevents assuming hardcoded class IDs across different models.
        """
        names = self.model.names  # dict of {class_id: class_name}
        logger.info(f"Inspecting model class names: {names}")

        target_set = set(cls_name.lower() for cls_name in config.TARGET_VEHICLE_CLASSES)

        for class_id, class_name in names.items():
            clean_name = str(class_name).lower().strip()
            if clean_name in target_set:
                self.target_class_ids.append(int(class_id))
                display_label = clean_name
                if clean_name == "bicycle":
                    display_label = "bike/scooty"
                self.class_id_to_name[int(class_id)] = display_label

        logger.info(f"Mapped vehicle target classes: {self.class_id_to_name} (IDs: {self.target_class_ids})")

        if not self.target_class_ids:
            logger.warning("No target vehicle classes matched in model.names! Defaulting to all classes.")

    def detect(self, frame):
        """
        Run inference on a frame with configured image size, confidence, and IoU thresholds.
        Returns a list of detection dictionaries:
        [{'bbox': (x1, y1, x2, y2), 'confidence': float, 'class_id': int, 'class_name': str}]
        """
        results = self.model.predict(
            source=frame,
            imgsz=config.IMG_SIZE,
            conf=config.CONF_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            classes=self.target_class_ids if self.target_class_ids else None,
            verbose=False
        )

        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.int().cpu().numpy()

            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)
                cls_id = int(cls_id)
                cls_name = self.class_id_to_name.get(cls_id, self.model.names.get(cls_id, "vehicle"))

                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': float(conf),
                    'class_id': cls_id,
                    'class_name': cls_name
                })

        return detections
