import cv2
import csv
from datetime import datetime
import numpy as np
# import time
import os
import glob

frames_dir = "frames"
print(f"等待 {frames_dir} 資料夾中出現影像 (請確保已在另一個終端機執行 capture_taipei_stream.py)...")

first_frame = None
while first_frame is None:
    jpg_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if jpg_files:
        first_frame = cv2.imread(jpg_files[-1])
        if first_frame is not None:
            break
    # time.sleep(1)
roi_bbox = cv2.selectROI("Select Traffic Light ROI", first_frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select Traffic Light ROI")

if roi_bbox == (0, 0, 0, 0):
    exit()

x, y, w, h = roi_bbox
print(f"Selected ROI: x={x}, y={y}, w={w}, h={h}")

current_state = "UNKNOWN"
last_change_time = datetime.now()


pending_state = None
pending_state_start = None
DEBOUNCE_TIME = 2.0  


predicted_red_duration = None  
current_red_start_time = None  
TOLERANCE = 2.0                

output_file = "tainan_traffic_prediction.csv"

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "New State", "Duration (sec)", "Note"])

    while True:
        # start_time = time.time()
        
        # 尋找最新的照片
        jpg_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
        if not jpg_files:
            if cv2.waitKey(100) == ord('q'): break
            continue
            
        latest_file = jpg_files[-1]
        
        if 'last_processed_file' in locals() and latest_file == last_processed_file:
            if cv2.waitKey(50) == ord('q'): break
            continue
            
        frame = cv2.imread(latest_file)
        if frame is None:
            if cv2.waitKey(50) == ord('q'): break
            continue
            
        last_processed_file = latest_file
        
        # 可選：自動刪除已經處理過的舊照片以節省硬碟空間 (保留最新的 10 張)
        if len(jpg_files) > 10:
            for old_file in jpg_files[:-10]:
                try:
                    os.remove(old_file)
                except OSError:
                    pass
            
        roi = frame[y:y+h, x:x+w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_red1, upper_red1 = np.array([0, 70, 50]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([170, 70, 50]), np.array([180, 255, 255])
        lower_green, upper_green = np.array([40, 100, 100]), np.array([90, 255, 255])

        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        red_rate = np.sum(mask_red > 0) / mask_red.size
        green_rate = np.sum(mask_green > 0) / mask_green.size

        raw_state = current_state
        if red_rate > green_rate and red_rate > 0.01:
            raw_state = "RED"
        elif green_rate > red_rate and green_rate > 0.01:
            raw_state = "GREEN"

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

                    if old_state == "GREEN" and current_state == "RED":
                        current_red_start_time = now
                        
                    elif old_state == "RED" and current_state == "GREEN":
                        actual_red_duration = duration
                        
                        if predicted_red_duration is None:
            
                            predicted_red_duration = actual_red_duration
                            note = f"首次記錄紅燈週期：{predicted_red_duration:.1f}秒"
                        else:
                            error = abs(actual_red_duration - predicted_red_duration)
                            if error <= TOLERANCE:
                                note = f"預測成功！(誤差 {error:.1f} 秒)"
                            else:
                                note = f"預測失敗！週期改變，重新校正為 {actual_red_duration:.1f}秒"
                                predicted_red_duration = actual_red_duration

                    log_msg = f"[{now.strftime('%H:%M:%S')}] 轉為 {current_state} (上個狀態維持 {duration:.1f} 秒) {note}"
                    print(log_msg)
                    writer.writerow([now.strftime('%Y-%m-%d %H:%M:%S'), current_state, round(duration, 1), note])
                    f.flush()

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
            
     
            if countdown < 0:
                timer_text = "Wait..."
            else:
                timer_text = f"{countdown}s"
                
  
            cv2.putText(frame, timer_text, (x + w + 10, y + int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Traffic Light Prediction POC', frame)
        # ⏳ 新增這段：強制鎖定在 30 FPS (1 秒 30 張，每張大約 0.033 秒)
        # process_time = time.time() - start_time
        # delay_time = 0.033 - process_time
            
        # if delay_time > 0:
        #         time.sleep(delay_time) # 如果算得太快，就強迫程式睡覺等待

        if cv2.waitKey(1) == ord('q'):
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 收到結束指令。")
                break

        # if cv2.waitKey(delay) == ord('q'):
        #     break

cv2.destroyAllWindows()