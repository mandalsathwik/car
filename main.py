import cv2
import time
import logging
import sys
import os

import config
from vehicle_detection import VehicleDetector
from vehicle_tracking import VehicleTracker
from plate_detection import PlateDetector
from plate_ocr import PlateOCR
from vehicle_plate_association import VehiclePlateAssociator
from event_logger import EventLogger

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("IBVAP.MainPipeline")

def main():
    logger.info("==================================================")
    logger.info("Initializing IBVAP Vehicle Analytics Pipeline")
    logger.info("==================================================")

    # Check input video source
    video_source = config.VIDEO_SOURCE
    if isinstance(video_source, str) and not os.path.exists(video_source) and not video_source.isdigit():
        logger.error(f"Video file not found at path: '{video_source}'")
        sys.exit(1)

    cap = cv2.VideoCapture(int(video_source) if str(video_source).isdigit() else video_source)
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {video_source}")
        sys.exit(1)

    # Initialize Modules
    try:
        detector = VehicleDetector()
        tracker = VehicleTracker(detector)
        plate_detector = PlateDetector()
        ocr_engine = PlateOCR()
        associator = VehiclePlateAssociator()
        logger_module = EventLogger()
    except Exception as e:
        logger.critical(f"Pipeline initialization failed: {e}", exc_info=True)
        sys.exit(1)

    frame_count = 0
    start_time = time.time()
    
    total_det_time = 0.0
    total_ocr_time = 0.0
    total_ocr_runs = 0

    logger.info(f"Pipeline running on source '{video_source}'. Press 'q' in video window to exit.")

    while True:
        frame_start = time.time()
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video stream reached or frame read failed.")
            break

        frame_count += 1

        # 1. Multi-Object Detection & Tracking
        det_start = time.time()
        active_tracks = tracker.track(frame, frame_count)
        det_time = (time.time() - det_start) * 1000  # ms
        total_det_time += det_time

        # 2. Process each tracked vehicle
        for track in active_tracks:
            track_id = track['track_id']
            cls_name = track['vehicle_class']
            v_conf = track['confidence']
            bbox = track['bbox']
            x1, y1, x2, y2 = bbox

            # Safely crop vehicle region
            vh, vw = frame.shape[:2]
            vehicle_crop = frame[max(0, y1):min(vh, y2), max(0, x1):min(vw, x2)]

            # Check if OCR should be run for this track on this frame (only if plate detector model is loaded)
            if plate_detector.model is not None and vehicle_crop.size > 0 and associator.should_run_ocr(track_id, frame_count):
                ocr_start_t = time.time()
                
                # Detect plate candidate regions inside vehicle crop
                plate_candidates = plate_detector.detect_plate(vehicle_crop)

                best_p_text = "NOT DETECTED"
                best_p_conf = 0.0

                for candidate in plate_candidates:
                    p_crop = candidate['crop']
                    p_text, p_conf, _ = ocr_engine.recognize(p_crop)
                    
                    if p_conf > best_p_conf:
                        best_p_text = p_text
                        best_p_conf = p_conf

                ocr_duration = (time.time() - ocr_start_t) * 1000 # ms
                total_ocr_time += ocr_duration
                total_ocr_runs += 1

                # Update temporal voting and association for track_id
                associator.update_track_ocr(track_id, cls_name, best_p_text, best_p_conf, frame_count)

            # Get current consolidated plate reading for visualization and logging
            plate_text, plate_conf = associator.get_track_plate(track_id)

            # Log Event
            logger_module.log_vehicle_event(
                frame_number=frame_count,
                track_id=track_id,
                vehicle_type=cls_name,
                vehicle_conf=v_conf,
                plate_text=plate_text,
                plate_conf=plate_conf,
                bbox=bbox
            )

            # 3. Visualization Formatting
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_vehicle = f"{cls_name} ID:{track_id} {v_conf:.2f}"

            if plate_detector.model is not None:
                if "UNKNOWN" in plate_text:
                    label_plate = f"PLATE: UNKNOWN"
                    plate_color = (0, 165, 255)
                elif plate_text == "NOT DETECTED":
                    label_plate = "PLATE: NOT DETECTED"
                    plate_color = (128, 128, 128)
                else:
                    label_plate = f"PLATE: {plate_text} {plate_conf:.2f}"
                    plate_color = (0, 0, 255)

                cv2.rectangle(frame, (x1, max(0, y1 - 40)), (x1 + max(len(label_vehicle), len(label_plate)) * 10, y1), (0, 0, 0), -1)
                cv2.putText(frame, label_vehicle, (x1 + 5, y1 - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(frame, label_plate, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, plate_color, 2)
            else:
                cv2.rectangle(frame, (x1, max(0, y1 - 25)), (x1 + len(label_vehicle) * 11, y1), (0, 0, 0), -1)
                cv2.putText(frame, label_vehicle, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # Performance Overlay
        frame_time = (time.time() - frame_start) * 1000 # ms
        fps = 1000.0 / frame_time if frame_time > 0 else 0
        avg_fps = frame_count / (time.time() - start_time) if (time.time() - start_time) > 0 else 0

        status_bar = f"IBVAP Vehicle Analytics | FPS: {fps:.1f} (Avg: {avg_fps:.1f}) | Tracks: {len(active_tracks)} | Events: {len(logger_module.logged_events)}"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (40, 40, 40), -1)
        cv2.putText(frame, status_bar, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if config.SHOW_DISPLAY:
            # Resize frame for screen display if necessary
            disp_h, disp_w = frame.shape[:2]
            if disp_w > config.MAX_DISPLAY_WIDTH:
                scale = config.MAX_DISPLAY_WIDTH / float(disp_w)
                display_frame = cv2.resize(frame, (int(disp_w * scale), int(disp_h * scale)))
            else:
                display_frame = frame

            cv2.imshow("IBVAP Vehicle Analytics Module", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User interrupted stream processing.")
                break

    # Benchmark summary
    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    avg_det_ms = total_det_time / frame_count if frame_count > 0 else 0
    avg_ocr_ms = total_ocr_time / total_ocr_runs if total_ocr_runs > 0 else 0

    logger.info("==================================================")
    logger.info("PIPELINE PERFORMANCE SUMMARY")
    logger.info("==================================================")
    logger.info(f"Total Frames Processed : {frame_count}")
    logger.info(f"Total Processing Time  : {elapsed_time:.2f} s")
    logger.info(f"Average FPS            : {avg_fps:.2f}")
    logger.info(f"Avg Detection+Track Latency : {avg_det_ms:.2f} ms / frame")
    logger.info(f"Total OCR Executions        : {total_ocr_runs}")
    logger.info(f"Avg OCR Latency             : {avg_ocr_ms:.2f} ms / OCR run")
    logger.info(f"Total Logged Events         : {len(logger_module.logged_events)}")
    logger.info("==================================================")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
