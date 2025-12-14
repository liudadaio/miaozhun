# 依赖: opencv-python, numpy
# 保存为 saomiao/aiming.py
import cv2
import numpy as np
import math
from typing import Optional, Tuple, List, Dict

# 参数（按需调整）
HOUGH_DP = 1.2
HOUGH_MIN_DIST = 30
HOUGH_PARAM1 = 100
HOUGH_PARAM2 = 30
MIN_BALL_RADIUS = 6
MAX_BALL_RADIUS_RATIO = 0.08  # 相对于最短边的最大半径比例

CANNY_LOW = 30
CANNY_HIGH = 100
HOUGH_LINE_THRESH = 40
HOUGH_LINE_MINLEN = 60
HOUGH_LINE_MAXGAP = 20

def _detect_circles(gray: np.ndarray, max_radius: int) -> List[Tuple[int,int,int]]:
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=HOUGH_DP, minDist=HOUGH_MIN_DIST,
                               param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
                               minRadius=MIN_BALL_RADIUS, maxRadius=max_radius)
    res = []
    if circles is None:
        return res
    circles = np.round(circles[0]).astype(int)
    for x,y,r in circles:
        res.append((int(x), int(y), int(r)))
    return res

def _choose_white(bgr: np.ndarray, circles: List[Tuple[int,int,int]]) -> Optional[Tuple[int,int,int]]:
    if not circles:
        return None
    best = None
    best_v = -1.0
    h,w = bgr.shape[:2]
    hsv_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    for (x,y,r) in circles:
        x1,y1 = max(0, x-r), max(0, y-r)
        x2,y2 = min(w, x+r), min(h, y+r)
        roi = hsv_full[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        v = float(np.mean(roi[:,:,2]))
        if v > best_v:
            best_v = v
            best = (x,y,r)
    return best

def _choose_target(bgr: np.ndarray, circles: List[Tuple[int,int,int]], white: Optional[Tuple[int,int,int]]) -> Optional[Tuple[int,int,int]]:
    if not circles:
        return None
    if white is None:
        return max(circles, key=lambda c: c[2])
    wx,wy,wr = white
    best = None
    bd = float('inf')
    for c in circles:
        # 忽略白球本身
        if abs(c[0]-wx) <= wr and abs(c[1]-wy) <= wr:
            continue
        d = (c[0]-wx)**2 + (c[1]-wy)**2
        if d < bd:
            bd = d
            best = c
    return best

def _detect_cue_line(gray: np.ndarray, white: Optional[Tuple[int,int,int]]) -> Optional[Tuple[int,int,int,int]]:
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                            threshold=HOUGH_LINE_THRESH,
                            minLineLength=HOUGH_LINE_MINLEN,
                            maxLineGap=HOUGH_LINE_MAXGAP)
    if lines is None:
        return None
    candidates = [tuple(l[0]) for l in lines]

    if white is not None:
        wx,wy,wr = white
        def dist_line_to_pt(line, px, py):
            x1,y1,x2,y2 = line
            vx,vy = x2-x1, y2-y1
            wxp, wyp = px-x1, py-y1
            seg_len2 = vx*vx + vy*vy
            if seg_len2 == 0:
                return math.hypot(wxp, wyp)
            t = (wxp*vx + wyp*vy)/seg_len2
            t = max(0.0, min(1.0, t))
            projx = x1 + t*vx
            projy = y1 + t*vy
            return math.hypot(px-projx, py-projy)
        best = None
        bd = float('inf')
        for line in candidates:
            d = dist_line_to_pt(line, wx, wy)
            if d < bd:
                bd = d
                best = line
        if best is not None and bd < max(150, wr*6):
            return best
    # 回退：最长线
    best = max(candidates, key=lambda ln: (ln[2]-ln[0])**2 + (ln[3]-ln[1])**2)
    return best

def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    ang1 = math.atan2(v1[1], v1[0])
    ang2 = math.atan2(v2[1], v2[0])
    diff = ang2 - ang1
    while diff <= -math.pi:
        diff += 2*math.pi
    while diff > math.pi:
        diff -= 2*math.pi
    return diff

def _to_overlay_lines_np(lines_list: List[Tuple[int,int,int,int]]) -> Optional[np.ndarray]:
    """
    Convert list of (x1,y1,x2,y2) into numpy array matching HoughLinesP format:
    shape (N,1,4), dtype=int, each entry like [[x1,y1,x2,y2]]
    Returns None if input empty.
    """
    if not lines_list:
        return None
    arr = np.array(lines_list, dtype=np.int32).reshape(-1, 1, 4)
    return arr

def process_frame_for_aim(img_bgr: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Return: (annotated_img, metadata)
    metadata keys:
      - white_ball: (x,y,r) or None
      - target_ball: (x,y,r) or None
      - cue_line: (x1,y1,x2,y2) or None
      - angle_deg: float or None
      - visual_lines_np: numpy.ndarray or None (shape (N,1,4)) <-- ready to pass to overlay.update_lines
    """
    if img_bgr is None:
        return img_bgr, {}
    h,w = img_bgr.shape[:2]
    short_side = min(h,w)
    max_ball_r = max(int(short_side * MAX_BALL_RADIUS_RATIO), 20)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    circles = _detect_circles(blur, max_radius=max_ball_r)
    white = _choose_white(img_bgr, circles)
    target = _choose_target(img_bgr, circles, white)
    cue = _detect_cue_line(blur, white)

    angle_rad = None
    if white is not None and target is not None and cue is not None:
        x1,y1,x2,y2 = cue
        d1 = (x1-white[0])**2 + (y1-white[1])**2
        d2 = (x2-white[0])**2 + (y2-white[1])**2
        if d2 < d1:
            x1,y1,x2,y2 = x2,y2,x1,y1
        vcue = np.array([x2-x1, y2-y1], dtype=float)
        vwt = np.array([target[0]-white[0], target[1]-white[1]], dtype=float)
        if np.linalg.norm(vcue) > 1e-6 and np.linalg.norm(vwt) > 1e-6:
            angle_rad = _angle_between(vcue, vwt)

    # prepare visual lines list and numpy array in overlay format
    visual_list = []
    if cue is not None:
        visual_list.append((cue[0], cue[1], cue[2], cue[3]))
    visual_np = _to_overlay_lines_np(visual_list)

    # annotated image (for debug)
    out = img_bgr.copy()
    if white is not None:
        cv2.circle(out, (white[0], white[1]), white[2], (0,255,255), 2)
        cv2.circle(out, (white[0], white[1]), 3, (0,255,255), -1)
    if target is not None:
        cv2.circle(out, (target[0], target[1]), target[2], (0,0,255), 2)
        cv2.circle(out, (target[0], target[1]), 3, (0,0,255), -1)
    if cue is not None:
        x1,y1,x2,y2 = cue
        cv2.line(out, (x1,y1), (x2,y2), (255,0,0), 2)
        vx,vy = x2-x1, y2-y1
        n = math.hypot(vx,vy)
        if n > 1e-6:
            ex1 = int(x1 - vx/n * 100)
            ey1 = int(y1 - vy/n * 100)
            ex2 = int(x2 + vx/n * 100)
            ey2 = int(y2 + vy/n * 100)
            cv2.line(out, (ex1,ey1), (ex2,ey2), (255,0,0), 1)
    if angle_rad is not None and white is not None:
        deg = math.degrees(angle_rad)
        txt = f"{deg:+.1f}°"
        cv2.putText(out, txt, (white[0]+10, white[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)

    metadata = {
        'white_ball': white,
        'target_ball': target,
        'cue_line': cue,
        'angle_deg': None if angle_rad is None else math.degrees(angle_rad),
        'visual_lines_np': visual_np
    }
    return out, metadata

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        ann, meta = process_frame_for_aim(img)
        print(meta)
        cv2.imshow("aim", ann)
        cv2.waitKey(0)
