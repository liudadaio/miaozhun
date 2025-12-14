#!/usr/bin/env python3
"""
台球游戏瞄准线检测工具
专门用于识别和延长台球游戏中的瞄准辅助线
"""
import time
import threading
import numpy as np
import cv2
import win32gui
import win32con
import mss
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont

# --------- 配置 ----------
CAPTURE_FPS_LIMIT = 30

# 颜色检测参数（台球瞄准线通常是白色、黄色或亮色）
TARGET_COLORS = [
    # HSV范围：白色
    ([0, 0, 200], [180, 30, 255]),
    # 黄色
    ([20, 100, 100], [30, 255, 255]),
    # 青色
    ([80, 100, 100], [100, 255, 255]),
    # 红色
    ([0, 100, 100], [10, 255, 255]),
]

# Hough参数（检测长直线）
HOUGH_THRESH = 30        # 降低阈值
HOUGH_MINLEN = 50        # 较短的最小长度（瞄准线可能不长）
HOUGH_MAXGAP = 20        # 允许间隙（虚线）

# Canny边缘检测
CANNY_LOW = 40
CANNY_HIGH = 120

# 延长线参数
EXTEND_LENGTH = 500      # 延长线的长度（像素）

DEBUG_MODE = False
SHOW_ORIGINAL_LINES = True  # 是否显示原始检测到的线
# --------------------------

running = True
overlay_window = None
target_hwnd = None
target_rect = None

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
    """使用mss捕获屏幕指定区域"""
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
        print(f"屏幕捕获失败: {e}")
        return None

def detect_colored_lines(img):
    """检测特定颜色的线条（台球瞄准线）"""
    if img is None or img.size == 0:
        return None
    
    # 转换到HSV色彩空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 合并所有目标颜色的mask
    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in TARGET_COLORS:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # 形态学操作：连接虚线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
    combined_mask = cv2.erode(combined_mask, kernel, iterations=1)
    
    # 边缘检测
    edges = cv2.Canny(combined_mask, 50, 150)
    
    # Hough直线检测
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                            threshold=HOUGH_THRESH,
                            minLineLength=HOUGH_MINLEN,
                            maxLineGap=HOUGH_MAXGAP)
    
    if lines is None:
        # 如果颜色检测失败，使用标准边缘检测
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                                threshold=HOUGH_THRESH,
                                minLineLength=HOUGH_MINLEN,
                                maxLineGap=HOUGH_MAXGAP)
    
    if lines is not None and not DEBUG_MODE:
        # 过滤：只保留较长的线，且不是完全水平/垂直的
        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # 只保留足够长的线
            if length < 40:
                continue
            
            # 计算角度
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # 排除完全水平和完全垂直的线（台边）
            if angle < 3 or angle > 87:
                continue
            
            filtered_lines.append(line)
        
        if len(filtered_lines) > 0:
            return np.array(filtered_lines)
    
    return lines

def extend_line(x1, y1, x2, y2, length, img_shape):
    """延长线段"""
    # 计算方向向量
    dx = x2 - x1
    dy = y2 - y1
    line_length = np.sqrt(dx**2 + dy**2)
    
    if line_length < 1:
        return x1, y1, x2, y2
    
    # 归一化方向向量
    dx /= line_length
    dy /= line_length
    
    # 向两个方向延长
    new_x1 = int(x1 - dx * length)
    new_y1 = int(y1 - dy * length)
    new_x2 = int(x2 + dx * length)
    new_y2 = int(y2 + dy * length)
    
    # 限制在图像范围内
    height, width = img_shape[:2]
    new_x1 = max(0, min(width, new_x1))
    new_y1 = max(0, min(height, new_y1))
    new_x2 = max(0, min(width, new_x2))
    new_y2 = max(0, min(height, new_y2))
    
    return new_x1, new_y1, new_x2, new_y2

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
        
        self.show_test_marker = True
        self.show_original = SHOW_ORIGINAL_LINES
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind('<Escape>', lambda e: self.on_close())
        self.root.bind('<q>', lambda e: self.on_close())
        self.root.bind('<t>', lambda e: self.toggle_test_marker())
        self.root.bind('<d>', lambda e: self.toggle_debug_mode())
        self.root.bind('<o>', lambda e: self.toggle_original())
        
        print("\n快捷键说明：")
        print("  T - 切换测试标记（四角绿框）")
        print("  D - 切换调试模式（显示所有线）")
        print("  O - 切换原始线条显示")
        print("  ESC/Q - 退出\n")
    
    def toggle_test_marker(self):
        self.show_test_marker = not self.show_test_marker
        print(f"测试标记: {'开' if self.show_test_marker else '关'}")
    
    def toggle_debug_mode(self):
        global DEBUG_MODE
        DEBUG_MODE = not DEBUG_MODE
        print(f"调试模式: {'开 (显示所有线)' if DEBUG_MODE else '关 (过滤)'}")
    
    def toggle_original(self):
        self.show_original = not self.show_original
        print(f"原始线条: {'显示' if self.show_original else '隐藏'}")
    
    def update_position(self):
        global target_rect
        try:
            rect = win32gui.GetWindowRect(self.target_hwnd)
            point = win32gui.ClientToScreen(self.target_hwnd, (0, 0))
            client_rect = win32gui.GetClientRect(self.target_hwnd)
            client_width = client_rect[2]
            client_height = client_rect[3]
            
            self.root.geometry(f"{client_width}x{client_height}+{point[0]}+{point[1]}")
            target_rect = (point[0], point[1], point[0] + client_width, point[1] + client_height)
            
            return client_width, client_height
        except Exception as e:
            print(f"更新位置失败: {e}")
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
            corner_size = 30
            # 四角
            draw.line([(0, 0), (corner_size, 0)], fill=(0, 255, 0, 255), width=2)
            draw.line([(0, 0), (0, corner_size)], fill=(0, 255, 0, 255), width=2)
            draw.line([(width-corner_size, 0), (width, 0)], fill=(0, 255, 0, 255), width=2)
            draw.line([(width, 0), (width, corner_size)], fill=(0, 255, 0, 255), width=2)
            draw.line([(0, height), (corner_size, height)], fill=(0, 255, 0, 255), width=2)
            draw.line([(0, height-corner_size), (0, height)], fill=(0, 255, 0, 255), width=2)
            draw.line([(width-corner_size, height), (width, height)], fill=(0, 255, 0, 255), width=2)
            draw.line([(width, height-corner_size), (width, height)], fill=(0, 255, 0, 255), width=2)
        
        lines_drawn = 0
        if self.lines is not None and len(self.lines) > 0:
            for line in self.lines:
                x1, y1, x2, y2 = line[0]
                
                # 绘制原始检测到的线（黄色细线）
                if self.show_original:
                    draw.line([(x1, y1), (x2, y2)], fill=(255, 255, 0, 200), width=3)
                
                # 延长线段并绘制（亮红色粗线）
                if self.img_shape is not None:
                    ex1, ey1, ex2, ey2 = extend_line(x1, y1, x2, y2, EXTEND_LENGTH, self.img_shape)
                    draw.line([(ex1, ey1), (ex2, ey2)], fill=(255, 0, 0, 255), width=5)
                    lines_drawn += 1
            
            # 信息显示
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except:
                font = None
            
            draw.rectangle([(5, 5), (250, 35)], fill=(0, 0, 0, 200))
            draw.text((10, 12), f"瞄准线: {lines_drawn} 条", fill=(0, 255, 0, 255), font=font)
        else:
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except:
                font = None
            draw.rectangle([(5, 5), (200, 32)], fill=(0, 0, 0, 200))
            draw.text((10, 12), "正在搜索瞄准线...", fill=(255, 255, 0, 255), font=font)
        
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
    
    print("\n开始检测台球瞄准线...")
    
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
        
        # 检测瞄准线
        lines = detect_colored_lines(img)
        
        # 每30帧打印状态
        if frame_count % 30 == 0:
            line_info = f"检测到 {len(lines)} 条瞄准线" if lines is not None else "未检测到瞄准线"
            debug_status = " [调试模式]" if DEBUG_MODE else ""
            print(f"帧 {frame_count}: {line_info}{debug_status}")
        
        # 更新覆盖层
        try:
            overlay.update_lines(lines, img.shape)
        except Exception as e:
            if frame_count % 60 == 0:
                print(f"更新覆盖层失败: {e}")

def main():
    global running, target_hwnd
    
    print("\n=== 台球游戏瞄准线检测工具 ===")
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
    print(f"窗口句柄: {target_hwnd}")
    
    if win32gui.IsIconic(target_hwnd):
        print("\n警告: 窗口已最小化，请先还原窗口！")
        input("还原窗口后按 Enter 继续...")
    
    if not win32gui.IsWindowVisible(target_hwnd):
        print("\n警告: 窗口不可见！")
        return
    
    print("\n功能特点：")
    print("  - 专门检测台球游戏中的瞄准线")
    print("  - 自动延长瞄准线以辅助瞄准")
    print("  - 黄色细线 = 原始检测到的线")
    print("  - 红色粗线 = 延长后的辅助线")
    print("  - 支持虚线瞄准线检测")
    
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
