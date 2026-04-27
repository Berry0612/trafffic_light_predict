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
        err_msg = f"[{datetime.now().strftime('%H:%M:%S')}]無法開啟串流。"
        with open(txt_file, "a", encoding="utf-8") as f: f.write(err_msg + "\n")
        return

    current_state = "UNKNOWN"
    last_change_time = datetime.now()
    pending_state = None
    pending_state_start = None
    DEBOUNCE_TIME = 2.0  
    predicted_red_duration = None  
    current_red_start_time = None  
    TOLERANCE = 2.0                

    with open(csv_file, "w", newline="", encoding="utf-8") as f_csv, \
         open(txt_file, "a", encoding="utf-8") as f_txt:
        
        writer = csv.writer(f_csv)
        writer.writerow(["Timestamp", "New State", "Duration (sec)", "Note"])
        
        start_msg = f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 [Cam {cam_id}] 開始監控！"
        f_txt.write(start_msg + "\n")
        print(start_msg)

        while True:
            ret, frame = cap.read()
            if not ret:
                cap = cv2.VideoCapture(url)
                time.sleep(1)
                continue

            # 影像分析
            roi = frame[y:y+h, x:x+w]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            mask_red = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255])) + \
                       cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
            mask_green = cv2.inRange(hsv, np.array([40, 90, 70]), np.array([90, 255, 255]))

            red_pixels = cv2.countNonZero(mask_red)
            green_pixels = cv2.countNonZero(mask_green)

            raw_state = current_state
            threshold = 30
            if red_pixels > threshold and red_pixels > green_pixels:
                raw_state = "RED"
            elif green_pixels > threshold and green_pixels > red_pixels:
                raw_state = "GREEN"

            now = datetime.now()

            # 防抖與預測邏輯
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

                        if old_state == "GREEN" and current_state == "RED":
                            current_red_start_time = now
                        elif old_state == "RED" and current_state == "GREEN":
                            actual_red_duration = duration
                            if predicted_red_duration is None:
                                predicted_red_duration = actual_red_duration
                                note = f"首次記錄週期：{predicted_red_duration:.1f}秒"
                            else:
                                error = abs(actual_red_duration - predicted_red_duration)
                                if error <= TOLERANCE:
                                    note = f"預測成功 (誤差 {error:.1f}s)"
                                else:
                                    note = f"校正週期：{actual_red_duration:.1f}秒"
                                    predicted_red_duration = actual_red_duration

                        # 雙重日誌寫入 (CSV 結構化資料 + TXT 詳細文字紀錄)
                        time_str = now.strftime('%H:%M:%S')
                        log_msg = f"[{time_str}] 轉為 {current_state} (維持 {duration:.1f} 秒) {note}"
                        
                        writer.writerow([now.strftime('%Y-%m-%d %H:%M:%S'), current_state, round(duration, 1), note])
                        f_csv.flush()
                        
                        f_txt.write(log_msg + "\n")
                        f_txt.flush()
                        print(f"[Cam {cam_id}] {log_msg}")

                        last_change_time = now
                        pending_state = None 
            else:
                pending_state = None

            # 畫面繪製
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            text_color = (0, 0, 255) if current_state == "RED" else (0, 255, 0)
            cv2.putText(frame, current_state, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

            if current_state == "RED" and predicted_red_duration is not None and current_red_start_time is not None:
                elapsed = (now - current_red_start_time).total_seconds()
                countdown = int(predicted_red_duration - elapsed)
                timer_text = "Wait..." if countdown < 0 else f"T-{countdown}s"
                cv2.putText(frame, timer_text, (x + w + 10, y + int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # 在畫面左上角標示攝影機 ID
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
        {"id": "D", "url": "https://trafficvideo3.tainan.gov.tw/d99425e8"}
    ]

    rois = {}

    for cam in cameras:
        cap = cv2.VideoCapture(cam["url"])
        ret, frame = cap.read()
        if not ret:
            continue
            
        window_title = f"Select ROI for Cam {cam['id']} (Press ENTER to confirm)"
        roi_bbox = cv2.selectROI(window_title, frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(window_title)
        cap.release()

        if roi_bbox != (0, 0, 0, 0):
            rois[cam["id"]] = roi_bbox

    if not rois:
        exit()

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
    cv2.putText(blank_frame, "Waiting for video...", (200, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

  
    latest_frames = { "A": blank_frame, "B": blank_frame, "C": blank_frame, "D": blank_frame }


    while True:
        for cam_id in latest_frames.keys():
            try:
               
                latest_frames[cam_id] = queues[cam_id].get(timeout=0.01)
            except queue.Empty:
                pass 
        top_row = np.hstack([latest_frames["A"], latest_frames["B"]])
        bottom_row = np.hstack([latest_frames["C"], latest_frames["D"]])
        dashboard = np.vstack([top_row, bottom_row])
        cv2.imshow("Traffic Monitoring Dashboard", dashboard)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
    for p in processes:
        p.terminate()