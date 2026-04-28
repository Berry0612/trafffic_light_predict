import argparse
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import cv2
import imageio_ffmpeg
import requests

BASE_HOST = "https://hls.bote.gov.taipei"
LIVE_BASE = f"{BASE_HOST}/live/"
API_BASE = f"{BASE_HOST}/api"


def post_json(endpoint: str, payload: Dict[str, Any], timeout: int = 20) -> Any:
    url = f"{API_BASE}/{endpoint}"
    # response = requests.post(url, json=payload, timeout=timeout)
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning) # 隱藏討厭的警告訊息

    response = requests.post(url, json=payload, timeout=timeout, verify=False)
    
    response.raise_for_status()
    return response.json()


def resolve_stream_url(camera_id: str, poll_seconds: int = 120) -> Dict[str, str]:
    camera_data = post_json("getCameraFqid", {"ccd_id": camera_id})
    if not camera_data:
        raise RuntimeError("getCameraFqid 回傳空資料")

    if camera_data[0].get("Id") == "unKnown":
        raise RuntimeError("攝影機編號不存在")

    init_data_list = post_json("initVideoWithIP", {"ip": "0.0.0.0", "ccd_id": camera_id})
    if not init_data_list:
        raise RuntimeError("initVideoWithIP 回傳空資料")

    init_data = init_data_list[0]
    camera_name = init_data.get("Name", f"camera_{camera_id}")

    if init_data.get("Result") == "long_term_ready" and init_data.get("StreamPath"):
        stream_path = init_data["StreamPath"]
        return {
            "stream_url": urljoin(LIVE_BASE, stream_path),
            "camera_name": camera_name,
            "stream_type": "long_term",
        }

    deadline = time.monotonic() + poll_seconds
    while time.monotonic() < deadline:
        status = post_json("checkVideoStatus", {"ccd_id": camera_id})
        if status.get("status") == "ready" and status.get("path"):
            return {
                "stream_url": urljoin(LIVE_BASE, status["path"]),
                "camera_name": camera_name,
                "stream_type": status.get("stream_type", "temporary"),
            }
        time.sleep(2)

    raise RuntimeError("等待串流就緒逾時，請稍後重試")


def capture_frames_opencv(
    stream_url: str,
    output_dir: Path,
    interval: float,
    max_frames: Optional[int],
    max_seconds: Optional[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟串流: {stream_url}")

    # Reduce decode latency if backend supports this property.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    start = time.monotonic()
    next_capture = start
    frame_count = 0
    consecutive_failures = 0

    print(f"開始擷取: {stream_url}")
    print(f"輸出資料夾: {output_dir.resolve()}")
    print(f"每 {interval} 秒擷取 1 張")

    try:
        while True:
            ok, frame = cap.read()
            now = time.monotonic()

            if not ok:
                consecutive_failures += 1
                if consecutive_failures >= 30:
                    raise RuntimeError("連續讀取串流失敗，請重新執行")
                time.sleep(0.1)
                continue

            consecutive_failures = 0

            if now >= next_capture:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = output_dir / f"frame_{timestamp}_{frame_count:06d}.jpg"
                cv2.imwrite(str(output_path), frame)
                frame_count += 1
                print(f"[{datetime.now().strftime('%H:%M:%S')}] saved -> {output_path.name}")
                next_capture += interval

                if max_frames is not None and frame_count >= max_frames:
                    print("達到 max_frames，停止擷取")
                    break

            if max_seconds is not None and (now - start) >= max_seconds:
                print("達到 max_seconds，停止擷取")
                break

    finally:
        cap.release()


def capture_frames_ffmpeg(
    stream_url: str,
    output_dir: Path,
    max_frames: Optional[int],
    max_seconds: Optional[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    output_pattern = str(output_dir / "frame_%Y%m%d_%H%M%S.jpg")

    command = [
        ffmpeg_exe,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-i",
        stream_url,
        "-vf",
        "fps=1",
        "-q:v",
        "2",
    ]

    if max_seconds is not None:
        command.extend(["-t", str(max_seconds)])
    if max_frames is not None:
        command.extend(["-frames:v", str(max_frames)])

    command.extend(["-strftime", "1", output_pattern])

    print("使用 ffmpeg 後端擷取")
    print("執行命令:")
    print(" ".join(shlex.quote(part) for part in command))

    process = subprocess.run(command, check=False)
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg 擷取失敗，退出碼: {process.returncode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="每秒擷取台北市即時影像串流畫面")
    parser.add_argument("--camera-id", default="345", help="網頁上的 id 參數")
    parser.add_argument("--interval", type=float, default=1.0, help="擷取間隔秒數，預設 1.0")
    parser.add_argument("--output", default="frames", help="輸出資料夾")
    parser.add_argument("--max-frames", type=int, default=None, help="最多擷取張數")
    parser.add_argument("--max-seconds", type=int, default=None, help="最多執行秒數")
    parser.add_argument("--direct-url", default=None, help="若已知 m3u8，可直接指定")
    parser.add_argument(
        "--backend",
        choices=["ffmpeg", "opencv"],
        default="ffmpeg",
        help="擷取後端，預設 ffmpeg",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.interval <= 0:
        raise ValueError("interval 必須大於 0")

    if args.direct_url:
        stream_url = args.direct_url
        print("使用 direct-url 直接擷取")
    else:
        info = resolve_stream_url(args.camera_id)
        stream_url = info["stream_url"]
        print(
            f"攝影機: {info['camera_name']} | 型態: {info['stream_type']} | URL: {stream_url}"
        )

    if args.backend == "ffmpeg":
        capture_frames_ffmpeg(
            stream_url=stream_url,
            output_dir=Path(args.output),
            max_frames=args.max_frames,
            max_seconds=args.max_seconds,
        )
    else:
        capture_frames_opencv(
            stream_url=stream_url,
            output_dir=Path(args.output),
            interval=args.interval,
            max_frames=args.max_frames,
            max_seconds=args.max_seconds,
        )


if __name__ == "__main__":
    main()
