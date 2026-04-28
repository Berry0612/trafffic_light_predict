import cv2
import csv
import time
import multiprocessing
import queue
import numpy as np
import os
from datetime import datetime

def process_camera(cam_id, url, roi_bbox, frame_queue):
    x, y, w, h = roi_bbox
    cap = cv2.VideoCapture(url)
    
    log_dir = os.path.join("logs", f"Camera_{cam_id}")
    os.makedirs(log_dir, exist_ok=True)
    
    csv_file = os.path.join(log_dir, "data_records.csv")
    txt_file = os.path.join(log_dir, "event_logs.txt")

    if not cap.isOpened():
        with open(txt_file, "a") as f: f.write(f"[{datetime.now().strftime('%H:%M:%S')}] Error: Stream open failed.\n")
        return

    current_state = "UNKNOWN"
    last_change_time = datetime.now()
    pending_state = None
    pending_state_start = None
    DEBOUNCE_TIME = 3.0  
    predicted_red_duration = None  
    current_red_start_time = None  
    TOLERANCE = 2.0                

    with open(csv_file, "w", newline="") as f_csv, \
         open(txt_file, "a") as f_txt:
        
        writer = csv.writer(f_csv)
        writer.writerow(["Timestamp", "State", "Duration", "Note"])
        
        start_msg = f"[{datetime.now().strftime('%H:%M:%S')}] [Cam {cam_id}] Monitoring started."
        f_txt.write(start_msg + "\n")
        print(start_msg)

        lower_red1 = np.array([0, 120, 150])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 150])
        upper_red2 = np.array([180, 255, 255])

        while True:
            ret, frame = cap.read()
            if not ret:
                cap = cv2.VideoCapture(url)
                time.sleep(1)
                continue

            roi = frame[y:y+h, x:x+w]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + \
            #            cv2.inRange(hsv, lower_red2, upper_red2)

            # red_pixels = cv2.countNonZero(mask_red)
            kernel = np.ones((3, 3), np.uint8)
        #     mask_green = cv2.inRange(
        #     hsv,
        #     np.array([40, 80, 80]),
        #     np.array([90, 255, 255])
        # )

            mask_red1 = cv2.inRange(
                    hsv,
                    np.array([0, 100, 100]),
                    np.array([10, 255, 255])
                )

            mask_red2 = cv2.inRange(
                    hsv,
                    np.array([170, 100, 100]),
                    np.array([180, 255, 255])
                )

            mask_red = mask_red1 + mask_red2

            mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
            mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)

            # mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
            # mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

            red_pixels = cv2.countNonZero(mask_red)
            # green_pixels = cv2.countNonZero(mask_green)

            #threshold = 15
            threshold = int((w * h) * 0.02)
            raw_state = "RED" if red_pixels > threshold else "NOT_RED"
            now = datetime.now()

            if raw_state != current_state:
                if pending_state != raw_state:
                    pending_state = raw_state
                    pending_state_start = now
                else:
                    if (now - pending_state_start).total_seconds() >= DEBOUNCE_TIME:
                        old_state = current_state
                        current_state = raw_state
                        duration = (now - last_change_time).total_seconds()
                        note = ""

                        if old_state == "NOT_RED" and current_state == "RED":
                            current_red_start_time = now
                        elif old_state == "RED" and current_state == "NOT_RED":
                            actual_red_duration = duration
                            if predicted_red_duration is None:
                                predicted_red_duration = actual_red_duration
                                note = f"Init cycle: {predicted_red_duration:.1f}s"
                            else:
                                error = abs(actual_red_duration - predicted_red_duration)
                                if error <= TOLERANCE:
                                    note = f"Predict OK (Err {error:.1f}s)"
                                else:
                                    note = f"Update cycle: {actual_red_duration:.1f}s"
                                    predicted_red_duration = actual_red_duration

                        time_str = now.strftime('%H:%M:%S')
                        log_msg = f"[{time_str}] -> {current_state} (Hold {duration:.1f}s) {note}"
                        
                        writer.writerow([now.strftime('%Y-%m-%d %H:%M:%S'), current_state, round(duration, 1), note])
                        f_csv.flush()
                        
                        f_txt.write(log_msg + "\n")
                        f_txt.flush()
                        print(f"[Cam {cam_id}] {log_msg}")

                        last_change_time = now
                        pending_state = None 
            else:
                pending_state = None

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            text_color = (0, 0, 255) if current_state == "RED" else (0, 255, 0)
            cv2.putText(frame, current_state, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

            if current_state == "RED" and predicted_red_duration is not None and current_red_start_time is not None:
                elapsed = (now - current_red_start_time).total_seconds()
                countdown = int(predicted_red_duration - elapsed)
                timer_text = "Wait..." if countdown < 0 else f"T-{countdown}s"
                cv2.putText(frame, timer_text, (x + w + 10, y + int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.putText(frame, f"Cam {cam_id}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            small_frame = cv2.resize(frame, (640, 360))
            try:
                frame_queue.put_nowait(small_frame)
            except queue.Full:
                pass

if __name__ == '__main__':
    cameras = [
        {"id": "A", "url": "https://trafficvideo3.tainan.gov.tw/4ba95398"},
        {"id": "B", "url": "https://trafficvideo2.tainan.gov.tw/942eb11e"},
        {"id": "C", "url": "https://trafficvideo2.tainan.gov.tw/685f3c55"},
        {"id": "D", "url": "https://trafficvideo2.tainan.gov.tw/d8a553fa"}
    ]

    ZOOM_FACTOR = 2.5

    print(f"[Main] Starting. Zoom factor: {ZOOM_FACTOR}. Please select RED light ROI.")
    rois = {}

    for cam in cameras:
        cap = cv2.VideoCapture(cam["url"])
        ret, frame = cap.read()
        if not ret:
            print(f"[Main] Error reading Cam {cam['id']}. Skipping.")
            continue
            
        zoomed_frame = cv2.resize(frame, None, fx=ZOOM_FACTOR, fy=ZOOM_FACTOR, interpolation=cv2.INTER_LINEAR)
        window_title = f"Select RED ROI - Cam {cam['id']} (Press ENTER)"
        
        roi_bbox_zoomed = cv2.selectROI(window_title, zoomed_frame, fromCenter=False, showCrosshair=False)
        cv2.destroyWindow(window_title)
        cap.release()

        if roi_bbox_zoomed != (0, 0, 0, 0):
            x = int(roi_bbox_zoomed[0] / ZOOM_FACTOR)
            y = int(roi_bbox_zoomed[1] / ZOOM_FACTOR)
            w = int(roi_bbox_zoomed[2] / ZOOM_FACTOR)
            h = int(roi_bbox_zoomed[3] / ZOOM_FACTOR)
            rois[cam["id"]] = (x, y, w, h)
            print(f"[Main] Cam {cam['id']} ROI set: ({x}, {y}, {w}, {h})")

    if len(rois) == 0:
        print("[Main] No ROI selected. Exiting.")
        exit()

    print("[Main] ROIs selected. Starting dashboard...")
    
    queues = {
        "A": multiprocessing.Queue(maxsize=3),
        "B": multiprocessing.Queue(maxsize=3),
        "C": multiprocessing.Queue(maxsize=3),
        "D": multiprocessing.Queue(maxsize=3)
    }

    processes = []
    for cam in cameras:
        if cam["id"] in rois:
            p = multiprocessing.Process(
                target=process_camera, 
                args=(cam["id"], cam["url"], rois[cam["id"]], queues[cam["id"]])
            )
            processes.append(p)
            p.daemon = True 
            p.start()

    blank_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(blank_frame, "Waiting...", (200, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    latest_frames = { "A": blank_frame, "B": blank_frame, "C": blank_frame, "D": blank_frame }

    print("[Main] Dashboard running. Press 'q' to quit.")

    while True:
        for cam_id in latest_frames.keys():
            try:
                latest_frames[cam_id] = queues[cam_id].get(timeout=0.01)
            except queue.Empty:
                pass

        top_row = np.hstack([latest_frames["A"], latest_frames["B"]])
        bottom_row = np.hstack([latest_frames["C"], latest_frames["D"]])
        dashboard = np.vstack([top_row, bottom_row])

        cv2.imshow("Dashboard", dashboard)

        if cv2.waitKey(1) == ord('q'):
            print("[Main] Quit signal received.")
            break

    cv2.destroyAllWindows()
    for p in processes:
        p.terminate()