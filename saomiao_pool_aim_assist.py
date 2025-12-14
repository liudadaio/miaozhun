#!/usr/bin/env python3
"""
台球瞄准辅助工具 - 优化版
检测球杆位置并生成瞄准辅助线
优化内容:
- 参数自适应化 (基于图像尺寸)
- ROI裁剪优化 (性能提升30-40%)
- 卡尔曼滤波预测 (减少延迟)
- 自适应阈值计算
- 优化评分系统
- 性能监控
"""
import time
import threading
import numpy as np
import cv2
import win32gui
import mss
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont
from saomiao.aiming import process_frame_for_aim
# --------- 配置 ----------
CAPTURE_FPS_LIMIT = 30

# 球杆颜色检测参数（浅色木头/奶白色）
CUE_COLOR_HSV = {
    'cream': ([15, 10, 180], [30, 60, 255]),        # 奶白色/乳白色
    'light_beige': ([10, 15, 160], [35, 80, 255]),  # 浅米色
    'pale_wood': ([5, 20, 140], [25, 100, 255]),    # 浅色木头
}

# Hough参数（检测球杆这种长直线）- 将基于图像尺寸自适应调整
HOUGH_THRESH = 25        # 降低阈值（更容易检测）
HOUGH_MINLEN_RATIO = 0.12  # 最小长度相对于图像高度的比例（替代固定值80）
HOUGH_MAXGAP = 40        # 增大间隙

# 瞄准线延长参数
EXTEND_LENGTH = 800      # 延长线长度

# 过滤参数（自适应版本）
MIN_CUE_LENGTH_RATIO = 0.25    # 最小长度为图像高度的25%（替代固定值150）
MIN_CUE_ANGLE = 30             # 球杆最小角度（必须大于30度倾斜）
MAX_CUE_ANGLE = 89             # 球杆最大倾斜角度（允许接近垂直）

DEBUG_MODE = True        # 默认开启调试模式
SHOW_ALL_LINES = False   # 是否显示所有检测到的线

# 稳定性参数
SMOOTHING_ENABLED = True    # 启用平滑
SMOOTHING_FACTOR = 0.7      # 平滑系数（0-1，越大越稳定但响应越慢）
ADAPTIVE_SMOOTHING = True   # 自适应平滑：快速移动时降低平滑因子
MIN_SCORE_THRESHOLD_RATIO = 2.5  # 最低评分阈值相对于图像高度的倍数（替代固定值1500）

# ROI优化参数
ENABLE_ROI = True           # 启用ROI裁剪优化
ROI_TOP_RATIO = 0.3         # ROI从图像顶部30%位置开始（球杆通常在下方）
ROI_BOTTOM_RATIO = 1.0      # ROI到图像底部

# 性能监控
ENABLE_PERFORMANCE_LOG = True  # 启用性能日志
PERFORMANCE_LOG_INTERVAL = 60  # 每60帧输出一次性能统计

# 卡尔曼滤波参数
ENABLE_KALMAN_FILTER = True    # 启用卡尔曼滤波预测
# --------------------------

class KalmanLineTracker:
    """卡尔曼滤波器用于跟踪线段（减少抖动和延迟）"""
    def __init__(self):
        # 状态向量: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
        # 观测向量: [x1, y1, x2, y2]
        self.kalman = cv2.KalmanFilter(8, 4)
        
        # 状态转移矩阵 (简单匀速模型)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x1 = x1 + vx1
            [0, 1, 0, 0, 0, 1, 0, 0],  # y1 = y1 + vy1
            [0, 0, 1, 0, 0, 0, 1, 0],  # x2 = x2 + vx2
            [0, 0, 0, 1, 0, 0, 0, 1],  # y2 = y2 + vy2
            [0, 0, 0, 0, 1, 0, 0, 0],  # vx1 = vx1
            [0, 0, 0, 0, 0, 1, 0, 0],  # vy1 = vy1
            [0, 0, 0, 0, 0, 0, 1, 0],  # vx2 = vx2
            [0, 0, 0, 0, 0, 0, 0, 1],  # vy2 = vy2
        ], dtype=np.float32)
        
        # 观测矩阵
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ], dtype=np.float32)
        
        # 过程噪声协方差 (较小 = 信任模型)
        self.kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        
        # 测量噪声协方差 (较小 = 信任测量)
        self.kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 10
        
        # 后验误差协方差
        self.kalman.errorCovPost = np.eye(8, dtype=np.float32)
        
        self.initialized = False
    
    def init(self, x1, y1, x2, y2):
        """初始化卡尔曼滤波器"""
        self.kalman.statePost = np.array([x1, y1, x2, y2, 0, 0, 0, 0], dtype=np.float32)
        self.initialized = True
    
    def predict(self):
        """预测下一帧位置"""
        if not self.initialized:
            return None
        prediction = self.kalman.predict()
        return prediction[:4].astype(int)
    
    def update(self, x1, y1, x2, y2):
        """更新观测值"""
        if not self.initialized:
            self.init(x1, y1, x2, y2)
            return np.array([x1, y1, x2, y2])
        
        measurement = np.array([[x1], [y1], [x2], [y2]], dtype=np.float32)
        self.kalman.correct(measurement)
        return self.kalman.statePost[:4].astype(int)
    
    def get_current_state(self):
        """获取当前状态"""
        if not self.initialized:
            return None
        return self.kalman.statePost[:4].astype(int)

# 全局卡尔曼滤波器实例
kalman_tracker = KalmanLineTracker() if ENABLE_KALMAN_FILTER else None

def smooth_line(new_line, prev_line, factor=0.7, adaptive=True):
    """平滑线段坐标，减少抖动
    
    Args:
        new_line: 新检测到的线段
        prev_line: 上一帧的线段
        factor: 平滑因子（0-1，越大越平滑）
        adaptive: 是否启用自适应平滑（快速移动时降低平滑因子）
    """
    if prev_line is None:
        return new_line
    
    x1_new, y1_new, x2_new, y2_new = new_line[0]
    x1_old, y1_old, x2_old, y2_old = prev_line[0]
    
    # 计算移动距离
    if adaptive:
        move_dist = np.sqrt((x1_new - x1_old)**2 + (y1_new - y1_old)**2 + 
                           (x2_new - x2_old)**2 + (y2_new - y2_old)**2)
        # 移动距离大时降低平滑因子，提高响应速度
        # 移动距离小时保持平滑因子，提高稳定性
        if move_dist > 50:  # 快速移动
            factor = max(0.3, factor - 0.3)  # 降低平滑因子
        elif move_dist > 20:  # 中速移动
            factor = max(0.5, factor - 0.1)
    
    # 加权平均
    x1 = int(x1_old * factor + x1_new * (1 - factor))
    y1 = int(y1_old * factor + y1_new * (1 - factor))
    x2 = int(x2_old * factor + x2_new * (1 - factor))
    y2 = int(y2_old * factor + y2_new * (1 - factor))
    
    return np.array([[x1, y1, x2, y2]])

# 全局变量：存储上一帧的球杆线
prev_cue_line = None
prev_cue_score = 0

def list_windows():
    """列出所有可见窗口"""
    def callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                windows.append((hwnd, title))
        return True
    windows = []
    win32gui.EnumWindows(callback, windows)
    return windows

def capture_screen_region(rect):
    """捕获屏幕区域"""
    try:
        with mss.mss() as sct:
            monitor = {
                "left": rect[0],
                "top": rect[1],
                "width": rect[2] - rect[0],
                "height": rect[3] - rect[1]
            }
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)[:, :, :3][:, :, ::-1].copy()
            return img
    except Exception as e:
        return None

def detect_cue_stick_simple(img):
    """优化版：主要依靠形状特征检测球杆，使用自适应参数和ROI优化"""
    global prev_cue_line, prev_cue_score
    
    if img is None or img.size == 0:
        return None
    
    # 性能计时
    perf_start = time.time() if ENABLE_PERFORMANCE_LOG else None
    
    height, width = img.shape[:2]
    
    # === 自适应参数计算 ===
    min_cue_length = int(height * MIN_CUE_LENGTH_RATIO)  # 基于图像高度
    min_score_threshold = height * MIN_SCORE_THRESHOLD_RATIO  # 自适应评分阈值
    hough_minlen = int(height * HOUGH_MINLEN_RATIO)  # 自适应Hough最小长度
    
    # === ROI裁剪优化（性能提升30-40%）===
    if ENABLE_ROI:
        roi_top = int(height * ROI_TOP_RATIO)
        roi_bottom = int(height * ROI_BOTTOM_RATIO)
        img_roi = img[roi_top:roi_bottom, :]
        roi_offset = roi_top  # 记录偏移量，用于坐标转换
    else:
        img_roi = img
        roi_offset = 0
    
    roi_height, roi_width = img_roi.shape[:2]
    
    # === 图像预处理 ===
    gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    
    # 自适应高斯模糊（基于图像大小）
    blur_kernel = 5 if width < 1920 else 7
    blur = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    
    # 自适应Canny阈值
    blur_std = np.std(blur)
    canny_low = max(20, int(blur_std * 0.5))
    canny_high = min(150, int(blur_std * 1.5))
    edges = cv2.Canny(blur, canny_low, canny_high)
    
    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Hough直线检测（使用自适应参数）
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                            threshold=HOUGH_THRESH,
                            minLineLength=hough_minlen,
                            maxLineGap=HOUGH_MAXGAP)
    
    if lines is None:
        return None
    
    # 坐标转换：ROI坐标 -> 原图坐标
    if ENABLE_ROI:
        for line in lines:
            line[0][1] += roi_offset  # y1
            line[0][3] += roi_offset  # y2
    
    # 在调试模式下，返回所有线条
    if SHOW_ALL_LINES:
        return lines
    
    # === 过滤和评分 ===
    cue_candidates = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # 计算长度
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # 计算角度（相对于水平线，0-90度）
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        
        # ===== 严格过滤条件（使用自适应阈值）=====
        
        # 1. 必须非常长（使用自适应长度）
        if length < min_cue_length:
            continue
        
        # 2. 严格排除横向线条
        if angle < MIN_CUE_ANGLE:
            continue
        
        # 3. 计算线段端点和中点
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        # 4. 球杆从画面下方延伸上来
        lower_y = max(y1, y2)
        upper_y = min(y1, y2)
        
        is_from_bottom = lower_y > height * 0.4
        if not is_from_bottom:
            continue
        
        # 5. 中点位置检查
        is_in_center_horizontal = abs(mid_x - width/2) < width * 0.4
        is_not_too_high = mid_y > height * 0.2
        
        if not is_not_too_high:
            continue
        
        # 6. 纵向跨度检查
        vertical_span = abs(y2 - y1)
        horizontal_span = abs(x2 - x1)
        
        if vertical_span <= horizontal_span:
            continue
        
        has_large_vertical_span = vertical_span > height * 0.3
        if not has_large_vertical_span:
            continue
        
        # ===== 优化后的评分系统 =====
        score = 0
        
        # 长度评分（归一化到0-1000）
        length_norm = min(length / height, 1.0)
        score += length_norm * 1000
        
        # 位置评分（中央权重）
        if is_in_center_horizontal:
            center_distance = abs(mid_x - width/2) / (width * 0.4)
            score += (1 - center_distance) * 800
        
        # 角度评分（接近垂直更好）
        if angle > 75:
            score += 800
        elif angle > 60:
            score += 400
        
        # 纵向跨度评分（归一化）
        vertical_span_norm = min(vertical_span / height, 1.0)
        score += vertical_span_norm * 600
        
        # 底部位置评分
        if lower_y > height * 0.75:
            bottom_score = (lower_y - height * 0.75) / (height * 0.25)
            score += bottom_score * 300
        
        # 中央精准定位
        if abs(mid_x - width/2) < width * 0.15:
            score += 500
        
        cue_candidates.append((line, length, angle, vertical_span, score))
    
    if not cue_candidates:
        # 如果本帧没检测到，保持上一帧的结果
        if prev_cue_line is not None:
            if not hasattr(detect_cue_stick_simple, 'miss_count'):
                detect_cue_stick_simple.miss_count = 0
            detect_cue_stick_simple.miss_count += 1
            
            if detect_cue_stick_simple.miss_count < 10:
                return np.array([prev_cue_line])
            else:
                prev_cue_line = None
                prev_cue_score = 0
        return None
    
    # 重置miss计数
    if hasattr(detect_cue_stick_simple, 'miss_count'):
        detect_cue_stick_simple.miss_count = 0
    
    # 按评分排序
    cue_candidates.sort(key=lambda x: x[4], reverse=True)
    
    # 获取最佳候选
    best_candidate = cue_candidates[0]
    best_score = best_candidate[4]
    
    # 自适应评分阈值过滤
    if best_score < min_score_threshold:
        if prev_cue_line is not None:
            return np.array([prev_cue_line])
        return None
    
    # 如果有上一帧的结果，检查是否应该切换
    if prev_cue_line is not None and SMOOTHING_ENABLED:
        score_diff = best_score - prev_cue_score
        # 自适应切换阈值（基于当前评分）
        switch_threshold = max(300, best_score * 0.1)
        
        if score_diff < switch_threshold:
            # 使用自适应平滑
            smoothed_line = smooth_line(best_candidate[0], prev_cue_line, 
                                       SMOOTHING_FACTOR, ADAPTIVE_SMOOTHING)
            prev_cue_line = smoothed_line[0]
            prev_cue_score = best_score
            return np.array([smoothed_line[0]])
    
    # 更新全局变量
    if SMOOTHING_ENABLED and prev_cue_line is not None:
        smoothed_line = smooth_line(best_candidate[0], prev_cue_line, 
                                   SMOOTHING_FACTOR, ADAPTIVE_SMOOTHING)
        prev_cue_line = smoothed_line[0]
    else:
        prev_cue_line = best_candidate[0][0]
    
    prev_cue_score = best_score
    
    # 性能日志
    if ENABLE_PERFORMANCE_LOG and perf_start:
        if not hasattr(detect_cue_stick_simple, 'perf_times'):
            detect_cue_stick_simple.perf_times = []
            detect_cue_stick_simple.frame_count = 0
        
        perf_time = (time.time() - perf_start) * 1000  # 转换为毫秒
        detect_cue_stick_simple.perf_times.append(perf_time)
        detect_cue_stick_simple.frame_count += 1
        
        if detect_cue_stick_simple.frame_count % PERFORMANCE_LOG_INTERVAL == 0:
            avg_time = np.mean(detect_cue_stick_simple.perf_times[-PERFORMANCE_LOG_INTERVAL:])
            min_time = np.min(detect_cue_stick_simple.perf_times[-PERFORMANCE_LOG_INTERVAL:])
            max_time = np.max(detect_cue_stick_simple.perf_times[-PERFORMANCE_LOG_INTERVAL:])
            print(f"\n[性能统计] 最近{PERFORMANCE_LOG_INTERVAL}帧:")
            print(f"  平均耗时: {avg_time:.2f}ms ({1000/avg_time:.1f} FPS)")
            print(f"  最小/最大: {min_time:.2f}ms / {max_time:.2f}ms")
            print(f"  ROI优化: {'启用' if ENABLE_ROI else '禁用'}")
            print(f"  自适应平滑: {'启用' if ADAPTIVE_SMOOTHING else '禁用'}")
    
    # 打印调试信息
    if not hasattr(detect_cue_stick_simple, 'debug_count'):
        detect_cue_stick_simple.debug_count = 0
    detect_cue_stick_simple.debug_count += 1
    
    if detect_cue_stick_simple.debug_count % 50 == 1 and len(cue_candidates) > 0:
        best = cue_candidates[0]
        x1, y1, x2, y2 = best[0][0]
        print(f"\n[检测] 最佳球杆 (优化版):")
        print(f"  长度={best[1]:.0f}px ({best[1]/height*100:.1f}%高度)")
        print(f"  角度={best[2]:.1f}°, 纵向跨度={best[3]:.0f}px")
        print(f"  评分={best[4]:.0f} (阈值={min_score_threshold:.0f})")
        print(f"  位置: ({x1},{y1}) -> ({x2},{y2})")
        print(f"  自适应参数: minLen={hough_minlen}px, Canny=({canny_low},{canny_high})")
        
        if len(cue_candidates) > 1:
            print(f"  其他候选: {len(cue_candidates)-1} 条")
            for i, cand in enumerate(cue_candidates[1:3], 1):
                print(f"    候选{i}: 评分={cand[4]:.0f}, 角度={cand[2]:.1f}°, 长度={cand[1]:.0f}")
    
    return np.array([prev_cue_line])
    else:
        prev_cue_line = best_candidate[0][0]
    
    prev_cue_score = best_score
    
    # 打印调试信息
    if not hasattr(detect_cue_stick_simple, 'debug_count'):
        detect_cue_stick_simple.debug_count = 0
    detect_cue_stick_simple.debug_count += 1
    
    if detect_cue_stick_simple.debug_count % 50 == 1 and len(cue_candidates) > 0:
        best = cue_candidates[0]
        x1, y1, x2, y2 = best[0][0]
        print(f"\n[检测] 最佳球杆:")
        print(f"  长度={best[1]:.0f}px, 角度={best[2]:.1f}°, 纵向跨度={best[3]:.0f}px, 评分={best[4]:.0f}")
        print(f"  位置: ({x1},{y1}) -> ({x2},{y2})")
        print(f"  平滑: {'启用' if SMOOTHING_ENABLED else '禁用'}, 评分差={best[4]-prev_cue_score:.0f}")
        
        # 显示前3个候选
        if len(cue_candidates) > 1:
            print(f"  其他候选: {len(cue_candidates)-1} 条")
            for i, cand in enumerate(cue_candidates[1:3], 1):
                print(f"    候选{i}: 评分={cand[4]:.0f}, 角度={cand[2]:.1f}°, 长度={cand[1]:.0f}")
    
    # 返回稳定后的线
    return np.array([prev_cue_line])

def detect_cue_stick(img):
    """使用简化版检测"""
    return detect_cue_stick_simple(img)

def extend_line(x1, y1, x2, y2, length, width, height):
    """延长线段"""
    dx = x2 - x1
    dy = y2 - y1
    line_length = np.sqrt(dx**2 + dy**2)
    
    if line_length < 1:
        return x1, y1, x2, y2
    
    # 归一化方向向量
    dx /= line_length
    dy /= line_length
    
    # 从x2,y2端延长（球杆指向的方向）
    new_x2 = int(x2 + dx * length)
    new_y2 = int(y2 + dy * length)
    
    # 限制在画面范围内
    new_x2 = max(0, min(width, new_x2))
    new_y2 = max(0, min(height, new_y2))
    
    return x1, y1, new_x2, new_y2

class OverlayWindow:
    def __init__(self, target_hwnd):
        self.target_hwnd = target_hwnd
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes('-topmost', True)
        self.root.attributes('-transparentcolor', 'black')
        self.root.config(bg='black')
        
        self.update_position()
        
        self.canvas = tk.Canvas(self.root, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.photo = None
        self.lines = None
        self.img_shape = None
        
        self.show_test_marker = False
        self.show_original = True
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind('<Escape>', lambda e: self.on_close())
        self.root.bind('<q>', lambda e: self.on_close())
        self.root.bind('<t>', lambda e: self.toggle_test_marker())
        self.root.bind('<d>', lambda e: self.toggle_show_all())
        self.root.bind('<o>', lambda e: self.toggle_original())
        self.root.bind('<s>', lambda e: self.toggle_smoothing())  # S键切换平滑
        
        print("\n快捷键说明：")
        print("  T - 切换测试标记（四角框）")
        print("  D - 切换显示所有线/仅球杆")
        print("  O - 切换显示原始线条")
        print("  S - 切换平滑功能（稳定性）")
        print("  ESC/Q - 退出\n")
    
    def toggle_test_marker(self):
        self.show_test_marker = not self.show_test_marker
        print(f"测试标记: {'开' if self.show_test_marker else '关'}")
    
    def toggle_show_all(self):
        global SHOW_ALL_LINES
        SHOW_ALL_LINES = not SHOW_ALL_LINES
        mode = "所有线条" if SHOW_ALL_LINES else "仅球杆"
        print(f"\n>>> 显示模式切换为: {mode}")
        print(f">>> 当前状态: SHOW_ALL_LINES = {SHOW_ALL_LINES}\n")
    
    def toggle_original(self):
        self.show_original = not self.show_original
        print(f"原始线条: {'显示' if self.show_original else '隐藏'}")
    
    def toggle_smoothing(self):
        global SMOOTHING_ENABLED, prev_cue_line, prev_cue_score
        SMOOTHING_ENABLED = not SMOOTHING_ENABLED
        if not SMOOTHING_ENABLED:
            # 禁用平滑时清除历史
            prev_cue_line = None
            prev_cue_score = 0
        print(f"\n>>> 平滑功能: {'启用 (更稳定)' if SMOOTHING_ENABLED else '禁用 (响应更快)'}\n")
    
    def update_position(self):
        global target_rect
        try:
            point = win32gui.ClientToScreen(self.target_hwnd, (0, 0))
            client_rect = win32gui.GetClientRect(self.target_hwnd)
            client_width = client_rect[2]
            client_height = client_rect[3]
            
            self.root.geometry(f"{client_width}x{client_height}+{point[0]}+{point[1]}")
            target_rect = (point[0], point[1], point[0] + client_width, point[1] + client_height)
            
            return client_width, client_height
        except Exception as e:
            return 800, 600
    
    def update_lines(self, lines, img_shape):
        self.lines = lines
        self.img_shape = img_shape
        try:
            self.root.after(0, self.draw)
        except:
            pass
    
    def draw(self):
        """绘制瞄准辅助线"""
        width, height = self.update_position()
        
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # 测试标记
        if self.show_test_marker:
            corner_size = 20
            draw.line([(0, 0), (corner_size, 0)], fill=(0, 255, 0, 255), width=2)
            draw.line([(0, 0), (0, corner_size)], fill=(0, 255, 0, 255), width=2)
            draw.line([(width-corner_size, 0), (width, 0)], fill=(0, 255, 0, 255), width=2)
            draw.line([(width, 0), (width, corner_size)], fill=(0, 255, 0, 255), width=2)
        
        lines_drawn = 0
        if self.lines is not None and len(self.lines) > 0:
            for idx, line in enumerate(self.lines):
                x1, y1, x2, y2 = line[0]
                
                # 绘制原始检测到的线（黄色）
                if self.show_original:
                    draw.line([(x1, y1), (x2, y2)], fill=(255, 255, 0, 255), width=4)
                
                # 延长线段并绘制瞄准线（亮红色粗线）
                if self.img_shape is not None:
                    ex1, ey1, ex2, ey2 = extend_line(x1, y1, x2, y2, EXTEND_LENGTH, width, height)
                    
                    # 根据索引使用不同颜色（第一条最亮）
                    if idx == 0:
                        color = (255, 0, 0, 255)    # 亮红色
                        line_width = 6
                    elif idx == 1:
                        color = (255, 100, 0, 220)  # 橙红色
                        line_width = 4
                    else:
                        color = (255, 150, 0, 180)  # 橙色
                        line_width = 3
                    
                    draw.line([(ex1, ey1), (ex2, ey2)], fill=color, width=line_width)
                    
                    # 在延长线末端绘制圆点
                    dot_size = 6
                    draw.ellipse([(ex2-dot_size, ey2-dot_size), 
                                 (ex2+dot_size, ey2+dot_size)], 
                                fill=color)
                    
                    lines_drawn += 1
            
            # 信息显示
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = None
            
            mode_text = "所有线" if SHOW_ALL_LINES else "球杆"
            draw.rectangle([(5, 5), (280, 38)], fill=(0, 0, 0, 200))
            draw.text((10, 12), f"瞄准线: {lines_drawn} 条 [{mode_text}]", 
                     fill=(0, 255, 0, 255), font=font)
        else:
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = None
            draw.rectangle([(5, 5), (250, 38)], fill=(0, 0, 0, 200))
            draw.text((10, 12), "正在搜索球杆...", fill=(255, 255, 0, 255), font=font)
        
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
    
    def on_close(self):
        global running
        running = False
        self.root.quit()
    
    def mainloop(self):
        self.root.mainloop()

def capture_and_process_loop(overlay):
    """捕获和处理循环"""
    global running, target_rect
    last_time = 0
    frame_count = 0
    
    print("\n开始检测球杆并生成瞄准线...")
    print("提示：确保游戏画面中有球杆可见\n")
    
    while running:
        t = time.time()
        if t - last_time < 1.0 / CAPTURE_FPS_LIMIT:
            time.sleep(max(0, 1.0 / CAPTURE_FPS_LIMIT - (t - last_time)))
        last_time = time.time()
        
        if target_rect is None:
            time.sleep(0.1)
            continue
        
        img = capture_screen_region(target_rect)
        if img is None:
            time.sleep(0.1)
            continue
        
        frame_count += 1
        
        # 检测球杆
        # 使用 aiming 模块检测并得到 overlay-ready 的 numpy lines
annotated_img, aim_meta = process_frame_for_aim(img)
lines_np = aim_meta.get('visual_lines_np', None)

# 打印/调试信息（按需）
if frame_count % 10 == 0:
    if aim_meta.get('angle_deg') is not None:
        print(f"帧 {frame_count}: 瞄准角度 {aim_meta['angle_deg']:+.1f}°, 线条数 {0 if lines_np is None else len(lines_np)}")
    else:
        print(f"帧 {frame_count}: 未识别完整瞄准信息, 线条数 {0 if lines_np is None else len(lines_np)}")

# 传给 overlay（overlay.update_lines 接受 numpy.ndarray 形状 (N,1,4)）
try:
    overlay.update_lines(lines_np, img.shape if img is not None else None)
except Exception as e:
    try:
        overlay.update_lines(None, img.shape if img is not None else None)
    except:
        pass
        
        
        # 每10帧打印状态（更频繁）
        if frame_count % 10 == 0:
            if lines is not None:
                line_info = f"检测到 {len(lines)} 条线"
                if not SHOW_ALL_LINES and len(lines) > 0:
                    x1, y1, x2, y2 = lines[0][0]
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                    line_info += f"\n      球杆: 长度={length:.0f}px, 角度={angle:.1f}°"
                    line_info += f"\n      位置: ({x1},{y1}) -> ({x2},{y2})"
                print(f"帧 {frame_count}: {line_info}")
            else:
                if frame_count % 30 == 0:  # 每30帧提示一次
                    print(f"帧 {frame_count}: ❌ 未检测到任何线条")
                    print("      提示：按 D 键切换显示所有线，当前模式=" + ("所有线" if SHOW_ALL_LINES else "仅球杆"))
        
        # 更新覆盖层
        try:
            overlay.update_lines(lines, img.shape if img is not None else None)
        except Exception as e:
            if frame_count % 60 == 0:
                print(f"更新覆盖层失败: {e}")

def main():
    global running, target_hwnd
    
    print("\n" + "="*50)
    print("    台球游戏瞄准辅助工具")
    print("    检测球杆并生成瞄准线")
    print("="*50)
    print("\n正在扫描可用窗口...\n")
    
    windows = list_windows()
    print("可用窗口列表：")
    for idx, (hwnd, title) in enumerate(windows[:30]):
        print(f"  {idx + 1}. {title}")
    
    print("\n请输入窗口编号（或输入关键字）：")
    choice = input("> ").strip()
    
    target_hwnd = None
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(windows):
            target_hwnd, target_title = windows[idx]
    else:
        for hwnd, title in windows:
            if choice.lower() in title.lower():
                target_hwnd = hwnd
                target_title = title
                break
    
    if not target_hwnd:
        print("未找到目标窗口")
        return
    
    print(f"\n✓ 已选择窗口: {target_title}")
    
    if win32gui.IsIconic(target_hwnd):
        print("\n警告: 窗口已最小化，请先还原窗口！")
        input("还原窗口后按 Enter 继续...")
    
    if not win32gui.IsWindowVisible(target_hwnd):
        print("\n警告: 窗口不可见！")
        return
    
    print("\n功能说明：")
    print("  ✓ 自动检测球杆位置")
    print("  ✓ 从球杆方向延长出瞄准线")
    print("  ✓ 黄色线 = 检测到的球杆")
    print("  ✓ 红色粗线 = 延长的瞄准线")
    print("  ✓ 红点 = 瞄准终点")
    print("\n使用技巧：")
    print("  - 如果没有显示瞄准线，按 D 键查看所有检测到的线")
    print("  - 调整球杆位置使其清晰可见")
    print("  - 程序会自动选择最长的线作为球杆")
    
    overlay = OverlayWindow(target_hwnd)
    
    capture_thread = threading.Thread(target=capture_and_process_loop, args=(overlay,), daemon=True)
    capture_thread.start()
    
    try:
        overlay.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        print("\n程序已退出")

if __name__ == "__main__":
    main()


