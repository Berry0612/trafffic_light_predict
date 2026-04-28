import cv2
import csv
import time
import multiprocessing
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# ==========================================
# 核心子進程 (Worker)：負責處理「單一」攝影機的所有邏輯
# ==========================================
def process_camera(cam_id, url):
    print(f"[Cam {cam_id}] 啟動進程，正在載入 YOLOv8 模型...")
    # 載入最輕量的 YOLOv8 Nano 模型 (第一次執行會自動下載 yolov8n.pt，約 6MB)
    model = YOLO('yolov8n.pt') 
    
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"❌ [Cam {cam_id}] 無法開啟串流。進程結束。")
        return

    # --- 階段一：YOLO 自動尋找 ROI (只執行一次) ---
    print(f"[Cam {cam_id}] 正在尋找紅綠燈位置...")
    x1, y1, x2, y2 = 0, 0, 0, 0
    roi_locked = False
    
    # 給它 30 次抓取畫面的機會來找紅綠燈 (避免剛好連線時畫面全黑或破圖)
    for _ in range(30):
        ret, frame = cap.read()
        if not ret:
            time.sleep(1)
            continue
            
        # 使用 YOLO 進行辨識，classes=[9] 代表只找 COCO 資料集裡的「Traffic Light」
        results = model(frame, classes=[9], conf=0.15, verbose=False) 
        
        if len(results[0].boxes) > 0:
            # 取得信心度最高的第一個紅綠燈框
            best_box = results[0].boxes[0]
            bx1, by1, bx2, by2 = map(int, best_box.xyxy[0])
            
            # 工程實務：YOLO 抓的框通常很貼合，為了怕燈號光暈被切掉，我們向外擴張 15 像素 (Padding)
            pad = 15
            y1, y2 = max(0, by1 - pad), min(frame.shape[0], by2 + pad)
            x1, x2 = max(0, bx1 - pad), min(frame.shape[1], bx2 + pad)
            
            roi_locked = True
            print(f"✅ [Cam {cam_id}] 成功鎖定紅綠燈座標：({x1},{y1}) - ({x2},{y2})")
            break
            
    if not roi_locked:
        print(f"⚠️ [Cam {cam_id}] 畫面上找不到紅綠燈，進程結束。")
        cap.release()
        return

    # --- 階段二：進入即時 HSV 狀態判斷與倒數預測 ---
    current_state = "UNKNOWN"
    last_change_time = datetime.now()
    pending_state = None
    pending_state_start = None
    DEBOUNCE_TIME = 2.0  

    predicted_red_duration = None  
    current_red_start_time = None  
    TOLERANCE = 2.0                

    output_file = f"camera_{cam_id}_log.csv"
    
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "New State", "Duration (sec)", "Note"])
        print(f"🚀 [Cam {cam_id}] 進入高速影像分析迴圈！日誌存入 {output_file}")

        while True:
            ret, frame = cap.read()
            if not ret:
                # 串流中斷，重新連線
                cap = cv2.VideoCapture(url)
                continue

            # 使用剛才 YOLO 找到的座標進行裁切 (高速處理，不佔用 YOLO 算力)
            roi = frame[y1:y2, x1:x2]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            mask_red = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255])) + \
                       cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
            mask_green = cv2.inRange(hsv, np.array([40, 90, 70]), np.array([90, 255, 255]))

            red_pixels = cv2.countNonZero(mask_red)
            green_pixels = cv2.countNonZero(mask_green)

            raw_state = current_state
            # YOLO 抓的框比較精準且較小，像素閥值調低為 15
            threshold = 15 
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
                                note = f"首次記錄：{predicted_red_duration:.1f}秒"
                            else:
                                error = abs(actual_red_duration - predicted_red_duration)
                                if error <= TOLERANCE:
                                    note = f"預測成功！(誤差 {error:.1f} 秒)"
                                else:
                                    note = f"校正週期：{actual_red_duration:.1f}秒"
                                    predicted_red_duration = actual_red_duration

                        log_msg = f"[Cam {cam_id}] [{now.strftime('%H:%M:%S')}] 轉為 {current_state} (維持 {duration:.1f} 秒) {note}"
                        print(log_msg)
                        writer.writerow([now.strftime('%Y-%m-%d %H:%M:%S'), current_state, round(duration, 1), note])
                        f.flush()

                        last_change_time = now
                        pending_state = None 
            else:
                pending_state = None

            # UI
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            text_color = (0, 0, 255) if current_state == "RED" else (0, 255, 0)
            cv2.putText(frame, current_state, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

            if current_state == "RED" and predicted_red_duration is not None and current_red_start_time is not None:
                elapsed = (now - current_red_start_time).total_seconds()
                countdown = int(predicted_red_duration - elapsed)
                timer_text = "Wait..." if countdown < 0 else f"T-{countdown}s"
                cv2.putText(frame, timer_text, (x2 + 10, y1 + int((y2-y1)/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # 為每個攝影機開一個獨立的視窗
            window_name = f"Camera {cam_id}"
            cv2.imshow(window_name, frame)
            
            if cv2.waitKey(1) == ord('q'):
                print(f"[Cam {cam_id}] 收到結束指令。")
                break

    cap.release()
    cv2.destroyWindow(window_name)

if __name__ == '__main__':
    cameras = [
        {"id": "A", "url": "https://trafficvideo3.tainan.gov.tw/4ba95398"},
        {"id": "B", "url": "https://trafficvideo2.tainan.gov.tw/942eb11e"},
        {"id": "C", "url": "https://trafficvideo2.tainan.gov.tw/685f3c55"},
        {"id": "D", "url": "https://trafficvideo3.tainan.gov.tw/d99425e8"}
    ]
    processes = []
    for cam in cameras:
        p = multiprocessing.Process(target=process_camera, args=(cam["id"], cam["url"]))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

