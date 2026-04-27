import subprocess
import os
import time

#路口 ID
CAMERA_IDS = ["300", "301", "302", "303", "304", "305", "306", "307", "308", "309"]

#真實 m3u8 網址
STREAM_URL_TEMPLATE = "https://hls.bote.gov.taipei/hls/{id}/index.m3u8" 

OUTPUT_DIR = "./cctv_records"
TEST_DURATION = 10  # 測試模式只錄 10 秒

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def run_test():
    print("=== 開始環境與連線測試 ===")
    for cam_id in CAMERA_IDS:
        url = STREAM_URL_TEMPLATE.format(id=cam_id)
        output_file = os.path.join(OUTPUT_DIR, f"TEST_cam_{cam_id}.mp4")
        
        print(f"測試錄影路口 {cam_id} ({TEST_DURATION}秒)...")
        # ffmpeg 指令：-t 代表錄影秒數, -c copy 代表不重新編碼直接儲存(極低CPU消耗)
        cmd = ["ffmpeg", "-y", "-i", url, "-t", str(TEST_DURATION), "-c", "copy", output_file]
        
        # 執行並隱藏 ffmpeg 落落長的輸出訊息
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if result.returncode == 0 and os.path.exists(output_file):
            print(f"路口 {cam_id} 測試成功！影片已存為 {output_file}")
        else:
            print(f"口 {cam_id} 測試失敗，請檢查網址或網路狀態。")
    print("=== 測試完成，請去資料夾檢查影片是否能正常播放 ===")

def run_collection():
    print("=== 開始 24 小時正式收集 ===")
    processes = []
    for cam_id in CAMERA_IDS:
        url = STREAM_URL_TEMPLATE.format(id=cam_id)
        # 這裡設定每 1 小時 (3600秒) 自動切一個檔案。
        output_pattern = os.path.join(OUTPUT_DIR, f"cam_{cam_id}_%03d.mp4")
        
        print(f"啟動路口 {cam_id} 錄影程序...")
        cmd = [
            "ffmpeg", "-i", url, 
            "-c", "copy", 
            "-f", "segment", "-segment_time", "3600", 
            output_pattern
        ]
        # 使用 Popen 在背景非同步執行
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(p)
    
    print("10 個路口的錄影程序已全數在背景啟動！")
    print("請保持電腦開機與網路連線。按 Ctrl+C 可以強制中斷。")
    
    try:
        # 讓主程式持續運行
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n收到中斷指令，正在關閉所有錄影程序...")
        for p in processes:
            p.terminate()
        print("已安全關閉。")

if __name__ == "__main__":
    mode = input("請選擇模式 (1: 測試 10 秒, 2: 正式收集 24 小時): ")
    if mode == "1":
        run_test()
    elif mode == "2":
        run_collection()
    else:
        print("輸入錯誤。")