import cv2
# 來源串流網址
url = 'https://jtmctrafficcctv2.gov.taipei/NVR/8426d0e5-5201-4dc3-9d9a-c237bc5ae1c3/live.m3u8'
cap = cv2.VideoCapture(url)             # 讀取來源

if not cap.isOpened():
    print("Cannot open camera")
    exit()
# 取得影片的 FPS (每秒顯示張數)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps != fps: # 避免抓不到 fps 或是 NaN
    fps = 30.0
delay = int(1000 / fps) # 計算每張畫面應該停留的毫秒數 (例如 30fps 就是約 33 毫秒)
print(f"Stream FPS: {fps}, Delay per frame: {delay}ms")

while True:
    ret, frame = cap.read()             # 讀取影片的每一幀
    if not ret:
        print("Cannot receive frame")   # 如果讀取錯誤，印出訊息
        cap = cv2.VideoCapture(url)     # 有時候串流間隔時間較久會中斷，中斷時重新讀取
        continue
    cv2.imshow('oxxostudio', frame)     # 如果讀取成功，顯示該幀的畫面
    
    # 這裡原本是 waitKey(1)，現在改成 waitKey(delay) 來控制播放速度
    if cv2.waitKey(delay) == ord('q'):      
        break
cap.release()                           # 所有作業都完成後，釋放資源
cv2.destroyAllWindows()                 # 結束所有視窗