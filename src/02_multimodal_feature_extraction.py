import cv2
import numpy as np

def compute_shot_cut_rate(video_path: str, threshold: float = 0.45) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_min = max((total_frames / fps) / 60.0, 1e-6)
    prev_hist, cuts, frame_idx = None, 0, 0
    step = max(int(round(fps)), 1)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            if prev_hist is not None:
                if (np.sum(np.abs(hist - prev_hist)) / hist.size) > threshold:
                    cuts += 1
            prev_hist = hist
        frame_idx += 1
    cap.release()
    return round(cuts / duration_min, 2)

def compute_engagement_index(likes: int, comments: int, shares: int, collects: int) -> tuple[float, float]:
    ei = 1.0 * likes + 2.5 * comments + 4.0 * shares + 3.0 * collects
    return round(ei, 2), round(float(np.log(ei + 1.0)), 4)
