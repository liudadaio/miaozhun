#!/usr/bin/env python3
"""
透明覆盖层模式 V2：使用屏幕截图方式，支持硬件加速窗口
"""
import time
import threading
from queue import Queue, Empty
import numpy as np
import cv2
import win32gui
import win32con
import win32api
import mss
from skimage.morphology import skeletonize
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw

# --------- 配置 ----------
CAPTURE_FPS_LIMIT = 30
# Hough 参数（针对球杆优化：检测长直线）
HOUGH_THRESH = 40        # 降低阈值，检测更多线
HOUGH_MINLEN = 80        # 降低最小长度
HOUGH_MAXGAP = 30        # 允许更大的间隙
# Canny 参数
CANNY_LOW = 30           
CANNY_HIGH = 100
# 文字过滤参数
MIN_LINE_LENGTH = 80     # 降低最小长度（从100降到80）
EXCLUDE_HORIZONTAL = 5   # 降低水平排除角度（从8度降到5度）
EXCLUDE_VERTICAL = 5     # 降低垂直排除角度（从8度降到5度）
DEBUG_MODE = False       # 调试模式：显示所有检测到的线
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
    """使用 mss 捕获屏幕指定区域（支持硬件加速窗口）"""
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

def detect_lines(img):
    """检测图像中的直线（强力过滤文字）"""
    if img is None or img.size == 0:
        return None
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 使用高斯模糊减少噪声
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # 边缘检测
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH, apertureSize=3)
    
    # 形态学操作，连接断裂的边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Hough 直线检测
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, 
                            threshold=HOUGH_THRESH,
                            minLineLength=HOUGH_MINLEN, 
                            maxLineGap=HOUGH_MAXGAP)
    
    if lines is not None:
        # 调试模式：显示所有线
        if DEBUG_MODE:
            return lines
        
        # 正常模式：过滤文字线条
        filtered_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 计算线段长度
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # 计算角度（相对于水平线，0-90度）
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # 过滤条件（已放宽）：
            # 1. 必须足够长（文字笔画很短）
            if length < MIN_LINE_LENGTH:
                continue
            
            # 2. 排除水平和垂直线（文字的主要特征）
            if angle < EXCLUDE_HORIZONTAL or angle > (90 - EXCLUDE_VERTICAL):
                # 例外：如果线非常长（>180），即使是水平/垂直也保留
                if length < 180:
                    continue
            
            # 3. 排除常见的文字角度范围
            is_near_45 = (42 < angle < 48)
            if is_near_45 and length < 140:
                continue
            
            # 通过所有过滤条件，保留这条线
            filtered_lines.append(line)
        
        if len(filtered_lines) > 0:
            return np.array(filtered_lines)
    
    return None

class OverlayWindow:
    def __init__(self, target_hwnd):
        self.target_hwnd = target_hwnd
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes('-topmost', True)
        self.root.attributes('-transparentcolor', 'black')  # 黑色完全透明
        self.root.config(bg='black')
        
        # 获取目标窗口位置和大小
        self.update_position()
        
        self.canvas = tk.Canvas(self.root, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.photo = None
        self.lines = None
        self.line_count = 0
        
        # 添加测试标记确认覆盖层可见
        self.show_test_marker = True
        
        # 绑定关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind('<Escape>', lambda e: self.on_close())
        self.root.bind('<q>', lambda e: self.on_close())
        self.root.bind('<t>', lambda e: self.toggle_test_marker())  # 按 T 切换测试标记
        self.root.bind('<d>', lambda e: self.toggle_debug_mode())   # 按 D 切换调试模式
    
    def toggle_test_marker(self):
        """切换测试标记显示"""
        self.show_test_marker = not self.show_test_marker
        print(f"测试标记: {'开' if self.show_test_marker else '关'}")
    
    def toggle_debug_mode(self):
        """切换调试模式（显示所有线）"""
        global DEBUG_MODE
        DEBUG_MODE = not DEBUG_MODE
        print(f"调试模式: {'开 (显示所有线)' if DEBUG_MODE else '关 (过滤文字)'}")
        
    def update_position(self):
        """更新覆盖层位置以匹配目标窗口"""
        global target_rect
        try:
            rect = win32gui.GetWindowRect(self.target_hwnd)
            target_rect = rect
            x, y, right, bottom = rect
            width = right - x
            height = bottom - y
            
            # 获取客户区位置（去掉标题栏和边框）
            point = win32gui.ClientToScreen(self.target_hwnd, (0, 0))
            client_rect = win32gui.GetClientRect(self.target_hwnd)
            client_width = client_rect[2]
            client_height = client_rect[3]
            
            self.root.geometry(f"{client_width}x{client_height}+{point[0]}+{point[1]}")
            
            # 更新目标矩形为客户区
            target_rect = (point[0], point[1], point[0] + client_width, point[1] + client_height)
            
            return client_width, client_height
        except Exception as e:
            print(f"更新位置失败: {e}")
            return 800, 600
    
    def update_lines(self, lines):
        """更新显示的辅助线（线程安全）"""
        self.lines = lines
        if lines is not None:
            self.line_count = len(lines)
        # 使用after在主线程中更新GUI
        try:
            self.root.after(0, self.draw)
        except:
            pass
    
    def draw(self):
        """绘制辅助线"""
        width, height = self.update_position()
        
        # 创建透明图像
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # 如果启用测试标记，绘制四角标记和中心十字
        if self.show_test_marker:
            # 四角标记
            corner_size = 30
            draw.line([(0, 0), (corner_size, 0)], fill=(0, 255, 0, 255), width=3)
            draw.line([(0, 0), (0, corner_size)], fill=(0, 255, 0, 255), width=3)
            
            draw.line([(width-corner_size, 0), (width, 0)], fill=(0, 255, 0, 255), width=3)
            draw.line([(width, 0), (width, corner_size)], fill=(0, 255, 0, 255), width=3)
            
            draw.line([(0, height), (corner_size, height)], fill=(0, 255, 0, 255), width=3)
            draw.line([(0, height-corner_size), (0, height)], fill=(0, 255, 0, 255), width=3)
            
            draw.line([(width-corner_size, height), (width, height)], fill=(0, 255, 0, 255), width=3)
            draw.line([(width, height-corner_size), (width, height)], fill=(0, 255, 0, 255), width=3)
            
            # 中心十字
            center_x, center_y = width // 2, height // 2
            cross_size = 40
            draw.line([(center_x - cross_size, center_y), (center_x + cross_size, center_y)], 
                     fill=(0, 255, 255, 255), width=2)
            draw.line([(center_x, center_y - cross_size), (center_x, center_y + cross_size)], 
                     fill=(0, 255, 255, 255), width=2)
        
        # 绘制检测到的线条
        lines_drawn = 0
        if self.lines is not None and len(self.lines) > 0:
            for line in self.lines:
                x1, y1, x2, y2 = line[0]
                # 计算线段长度
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # 根据长度调整颜色和粗细
                if length > 150:
                    # 很长的线（可能是球杆）- 亮红色，更粗
                    color = (255, 0, 0, 255)
                    width_line = 6
                elif length > 100:
                    # 中等长度 - 橙色
                    color = (255, 128, 0, 255)
                    width_line = 5
                else:
                    # 短线 - 黄色，较细
                    color = (255, 255, 0, 255)
                    width_line = 4
                
                # 绘制线条（加粗更明显）
                draw.line([(x1, y1), (x2, y2)], fill=color, width=width_line)
                lines_drawn += 1
            
            # 显示检测到的线条数量（加大字体）
            from PIL import ImageFont
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = None
            
            draw.rectangle([(5, 5), (220, 35)], fill=(0, 0, 0, 200))
            draw.text((10, 12), f"检测到 {lines_drawn} 条线", fill=(0, 255, 0, 255), font=font)
            
            # 打印调试信息（仅首次）
            if not hasattr(self, '_first_draw_done'):
                print(f"[绘制] 成功绘制 {lines_drawn} 条线，画布大小: {width}x{height}")
                if lines_drawn > 0:
                    x1, y1, x2, y2 = self.lines[0][0]
                    print(f"[绘制] 第一条线坐标: ({x1},{y1}) -> ({x2},{y2})")
                self._first_draw_done = True
        else:
            # 显示提示信息
            from PIL import ImageFont
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except:
                font = None
            
            draw.rectangle([(5, 5), (180, 32)], fill=(0, 0, 0, 200))
            draw.text((10, 12), "等待检测...", fill=(255, 255, 0, 255), font=font)
        
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
    
    print("\n开始捕获循环...")
    
    while running:
        t = time.time()
        if t - last_time < 1.0 / CAPTURE_FPS_LIMIT:
            time.sleep(max(0, 1.0 / CAPTURE_FPS_LIMIT - (t - last_time)))
        last_time = time.time()
        
        if target_rect is None:
            time.sleep(0.1)
            continue
        
        # 使用屏幕截图方式捕获（支持硬件加速）
        img = capture_screen_region(target_rect)
        if img is None:
            time.sleep(0.1)
            continue
        
        frame_count += 1
        
        # 检测线条
        lines = detect_lines(img)
        
        # 每 30 帧打印一次状态
        if frame_count % 30 == 0:
            line_info = f"检测到 {len(lines)} 条线" if lines is not None else "未检测到线条"
            debug_status = " [调试模式]" if DEBUG_MODE else ""
            print(f"帧 {frame_count}: {line_info}, 图像: {img.shape}{debug_status}")
        
        # 更新覆盖层
        try:
            overlay.update_lines(lines)
        except Exception as e:
            if frame_count % 60 == 0:
                print(f"更新覆盖层失败: {e}")

def main():
    global running, target_hwnd
    
    print("\n=== 透明覆盖层辅助线工具 V2 (支持硬件加速) ===")
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
    
    # 检查窗口状态
    if win32gui.IsIconic(target_hwnd):
        print("\n警告: 窗口已最小化，请先还原窗口！")
        input("还原窗口后按 Enter 继续...")
    
    if not win32gui.IsWindowVisible(target_hwnd):
        print("\n警告: 窗口不可见！")
        return
    
    print("\n使用屏幕截图模式（支持硬件加速渲染）")
    print("正在创建透明覆盖层...")
    print("\n功能特点：")
    print("  - 检测倾斜的长直线（如球杆）")
    print("  - 自动过滤文字区域（水平/垂直短线）")
    print("  - 红色粗线 = 很长的线（可能是球杆）")
    print("  - 橙色中线 = 中等长度")
    print("  - 黄色细线 = 短线段")
    print("\n操作提示：")
    print("  - 按 T 键切换测试标记（四角绿色框+中心十字）")
    print("  - 按 ESC 或 Q 键退出")
    print("  - 覆盖层会自动跟随窗口位置")
    print("  - 如果看不到线条，按 T 检查覆盖层是否对齐\n")
    
    # 创建覆盖层窗口
    overlay = OverlayWindow(target_hwnd)
    
    # 启动捕获线程
    capture_thread = threading.Thread(target=capture_and_process_loop, args=(overlay,), daemon=True)
    capture_thread.start()
    
    # 运行GUI主循环
    try:
        overlay.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        print("\n程序已退出")

if __name__ == "__main__":
    main()
