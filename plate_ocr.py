import cv2
import re
import logging
import easyocr
import numpy as np
import config

logger = logging.getLogger("IBVAP.PlateOCR")

class PlateOCR:
    def __init__(self):
        logger.info("Initializing EasyOCR Engine...")
        # Use GPU if available, else CPU fallback handled internally by easyocr
        try:
            self.reader = easyocr.Reader(['en'], gpu=True)
        except Exception as e:
            logger.warning(f"EasyOCR GPU init failed, switching to CPU: {e}")
            self.reader = easyocr.Reader(['en'], gpu=False)

    def preprocess_plate(self, plate_crop):
        """
        Apply plate image preprocessing pipeline:
        1. Resize/Upscale if too small
        2. Grayscale conversion
        3. CLAHE contrast enhancement
        4. Bilateral filtering for noise reduction while keeping edges sharp
        """
        if plate_crop is None or plate_crop.size == 0:
            return None

        h, w = plate_crop.shape[:2]

        # Upscale if crop is too small for OCR engine
        if w < 160 or h < 50:
            scale_factor = max(160.0 / w, 50.0 / h)
            plate_crop = cv2.resize(plate_crop, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        # Grayscale
        if len(plate_crop.shape) == 3:
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_crop.copy()

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Bilateral filter to smooth noise while preserving character edges
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

        return denoised

    def clean_text(self, text):
        """
        Normalize and clean OCR output:
        - Uppercase
        - Remove spaces and special non-alphanumeric characters
        - Filter out common OCR hallucinated artifacts
        """
        if not text:
            return ""

        text = text.upper()
        # Remove non-alphanumeric
        cleaned = re.sub(r'[^A-Z0-9]', '', text)
        return cleaned

    def recognize(self, plate_crop):
        """
        Input: plate_crop (BGR or Grayscale image)
        Returns: (plate_text, ocr_confidence, processed_image)
        """
        processed = self.preprocess_plate(plate_crop)
        if processed is None:
            return "NOT DETECTED", 0.0, None

        # Run EasyOCR on preprocessed image
        results = self.reader.readtext(processed)

        best_text = ""
        best_conf = 0.0

        for (bbox, raw_text, prob) in results:
            cleaned = self.clean_text(raw_text)
            # Plate heuristic: valid plates typically have 4-12 characters
            if len(cleaned) >= 4 and prob > best_conf:
                best_text = cleaned
                best_conf = float(prob)

        # Handle low confidence / uncertain OCR
        if not best_text or best_conf < config.PLATE_CONF_THRESHOLD:
            if best_text and best_conf > 0.15:
                return f"UNKNOWN ({best_text})", best_conf, processed
            return "UNKNOWN / LOW CONFIDENCE", best_conf, processed

        return best_text, best_conf, processed
