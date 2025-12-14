#!/usr/bin/env python3
"""
透明覆盖层模式：直接在目标程序窗口上叠加辅助线
"""
import time
import threading
from queue import Queue, Empty
import numpy as np
import cv2
import win32gui
import win32ui
import win32con
import win32api
from ctypes import windll, byref, c_int
from skimage.morphology import skeletonize
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw

# --------- 配置 ----------
CAPTURE_FPS_LIMIT = 30
# Hough 参数
HOUGH_THRESH = 80
HOUGH_MINLEN = 50
HOUGH_MAXGAP = 10
# Canny 参数
CANNY_LOW = 50
CANNY_HIGH = 150
# --------------------------

running = True
overlay_window = None
target_hwnd = None

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

def capture_window(hwnd):
    """捕获指定窗口的画面"""
    try:
        # 检查窗口是否最小化
        if win32gui.IsIconic(hwnd):
            print("窗口已最小化，无法捕获")
            return None
        
        # 尝试获取窗口矩形
        try:
            rect = win32gui.GetWindowRect(hwnd)
            left, top, right, bottom = rect
            width = right - left
            height = bottom - top
            print(f"窗口大小: {width}x{height}")
        except Exception as e:
            print(f"获取窗口矩形失败: {e}")
            return None
        
        if width <= 0 or height <= 0:
            print(f"窗口大小无效: {width}x{height}")
            return None
        
        # 使用 GetWindowDC 而不是 GetClientRect
        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()
        
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
        saveDC.SelectObject(saveBitMap)
        
        # 尝试 PrintWindow
        result = win32gui.PrintWindow(hwnd, saveDC.GetSafeHdc(), 3)
        print(f"PrintWindow 结果: {result}")
        
        if result == 0:
            # PrintWindow 失败，尝试 BitBlt
            print("PrintWindow 失败，尝试 BitBlt")
            result = saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)
            print(f"BitBlt 结果: {result}")
        
        bmpstr = saveBitMap.GetBitmapBits(True)
        img = np.frombuffer(bmpstr, dtype=np.uint8).reshape(height, width, 4)
        
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)
        
        # 转换 BGRA 到 BGR
        img = img[:, :, :3][:, :, ::-1].copy()
        
        # 检查图像是否有效
        mean_val = np.mean(img)
        print(f"图像平均值: {mean_val}")
        
        if mean_val < 1:
            print("图像全黑，可能捕获失败")
            return None
            
        return img
    except Exception as e:
        print(f"捕获窗口异常: {e}")
        import traceback
        traceback.print_exc()
        return None

def detect_lines(img):
    """检测图像中的直线"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    # 使用 Canny 直接检测边缘
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    
    # 降低 Hough 阈值以检测更多线条
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50,
                            minLineLength=30, maxLineGap=15)
    
    if lines is not None:
        print(f"检测到 {len(lines)} 条线")
    else:
        print("未检测到线条")
    
    return lines

class OverlayWindow:
    def __init__(self, target_hwnd):
        self.target_hwnd = target_hwnd
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes('-topmost', True)
        self.root.attributes('-transparentcolor', 'black')
        self.root.config(bg='black')
        
        # 获取目标窗口位置和大小
        self.update_position()
        
        self.canvas = tk.Canvas(self.root, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.photo = None
        self.lines = None
        
        # 绑定关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind('<Escape>', lambda e: self.on_close())
        
    def update_position(self):
        """更新覆盖层位置以匹配目标窗口"""
        try:
            rect = win32gui.GetWindowRect(self.target_hwnd)
            x, y, right, bottom = rect
            width = right - x
            height = bottom - y
            
            # 调整为客户区坐标
            client_rect = win32gui.GetClientRect(self.target_hwnd)
            client_width = client_rect[2]
            client_height = client_rect[3]
            
            # 计算边框大小
            point = win32gui.ClientToScreen(self.target_hwnd, (0, 0))
            
            self.root.geometry(f"{client_width}x{client_height}+{point[0]}+{point[1]}")
            return client_width, client_height
        except:
            return 800, 600
    
    def update_lines(self, lines):
        """更新显示的辅助线"""
        self.lines = lines
        self.draw()
    
    def draw(self):
        """绘制辅助线"""
        width, height = self.update_position()
        
        # 创建透明图像
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # 绘制检测到的线条
        if self.lines is not None and len(self.lines) > 0:
            for line in self.lines:
                x1, y1, x2, y2 = line[0]
                # 使用更粗更明显的红色线条
                draw.line([(x1, y1), (x2, y2)], fill=(255, 0, 0, 255), width=3)
        else:
            # 如果没有检测到线条，绘制一个测试线条验证覆盖层是否工作
            draw.line([(50, 50), (width-50, height-50)], fill=(0, 255, 0, 255), width=3)
            draw.text((100, 100), "Overlay Active - No lines detected", fill=(255, 255, 0, 255))
        
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
    global running
    last_time = 0
    frame_count = 0
    
    while running:
        t = time.time()
        if t - last_time < 1.0 / CAPTURE_FPS_LIMIT:
            time.sleep(max(0, 1.0 / CAPTURE_FPS_LIMIT - (t - last_time)))
        last_time = time.time()
        
        # 捕获窗口
        img = capture_window(target_hwnd)
        if img is None:
            print("捕获失败")
            time.sleep(0.5)
            continue
        
        frame_count += 1
        if frame_count % 30 == 0:  # 每30帧打印一次
            print(f"已捕获 {frame_count} 帧, 图像大小: {img.shape}")
        
        # 检测线条
        lines = detect_lines(img)
        
        # 更新覆盖层
        try:
            overlay.update_lines(lines)
        except Exception as e:
            print(f"更新覆盖层失败: {e}")

def main():
    global running, target_hwnd
    
    print("\n=== 透明覆盖层辅助线工具 ===")
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
    
    print("正在创建透明覆盖层...")
    print("\n进行首次捕获测试...")
    test_img = capture_window(target_hwnd)
    if test_img is None:
        print("\n✗ 无法捕获该窗口！")
        print("可能原因:")
        print("  1. 窗口使用了硬件加速渲染（如游戏）")
        print("  2. 窗口受系统保护")
        print("  3. 窗口已最小化")
        print("\n建议:")
        print("  - 尝试其他窗口（如记事本、资源管理器）")
        print("  - 确保窗口处于正常显示状态")
        return
    
    print(f"✓ 首次捕获成功！图像大小: {test_img.shape}")
    print("\n提示：")
    print("  - 红色线条会直接显示在目标窗口上")
    print("  - 按 ESC 键退出")
    print("  - 覆盖层会自动跟随窗口位置\n")
    
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
