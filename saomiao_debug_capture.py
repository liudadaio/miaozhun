#!/usr/bin/env python3
"""
调试工具：捕获游戏画面并显示检测过程
帮助分析瞄准线特征
"""
import cv2
import numpy as np
import win32gui
import mss
import time

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
        print(f"捕获失败: {e}")
        return None

def analyze_frame(img):
    """分析一帧图像，显示多种检测结果"""
    if img is None:
        return
    
    # 缩小显示（如果太大）
    height, width = img.shape[:2]
    if width > 1280:
        scale = 1280 / width
        img = cv2.resize(img, None, fx=scale, fy=scale)
        height, width = img.shape[:2]
    
    # 1. 原始图像
    display_original = img.copy()
    
    # 2. 灰度 + Canny边缘
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # 3. HSV颜色检测（多种颜色）
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 白色
    mask_white = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
    # 黄色
    mask_yellow = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([30, 255, 255]))
    # 青色
    mask_cyan = cv2.inRange(hsv, np.array([80, 100, 100]), np.array([100, 255, 255]))
    # 红色
    mask_red1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    # 合并所有颜色mask
    mask_all = cv2.bitwise_or(mask_white, mask_yellow)
    mask_all = cv2.bitwise_or(mask_all, mask_cyan)
    mask_all = cv2.bitwise_or(mask_all, mask_red)
    
    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_all = cv2.dilate(mask_all, kernel, iterations=2)
    
    mask_colored = cv2.cvtColor(mask_all, cv2.COLOR_GRAY2BGR)
    
    # 4. Hough直线检测
    lines_img = img.copy()
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, 
                            threshold=30, minLineLength=50, maxLineGap=20)
    
    if lines is not None:
        print(f"检测到 {len(lines)} 条线")
        for i, line in enumerate(lines[:20]):  # 只显示前20条
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            
            # 根据长度和角度着色
            if length > 100:
                color = (0, 0, 255)  # 红色 - 长线
            elif length > 50:
                color = (0, 255, 255)  # 黄色 - 中等
            else:
                color = (0, 255, 0)  # 绿色 - 短线
            
            cv2.line(lines_img, (x1, y1), (x2, y2), color, 2)
            
            # 显示前5条线的信息
            if i < 5:
                print(f"  线{i+1}: 长度={length:.1f}, 角度={angle:.1f}°, 坐标=({x1},{y1})->({x2},{y2})")
    else:
        print("未检测到线条")
    
    # 拼接显示
    top_row = np.hstack([display_original, edges_colored])
    bottom_row = np.hstack([mask_colored, lines_img])
    combined = np.vstack([top_row, bottom_row])
    
    # 添加标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "Original", (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(combined, "Edges", (width + 10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(combined, "Color Mask", (10, height + 30), font, 1, (0, 255, 0), 2)
    cv2.putText(combined, "Lines Detected", (width + 10, height + 30), font, 1, (0, 255, 0), 2)
    
    cv2.imshow("Debug - Press SPACE to capture, Q to quit", combined)

def main():
    print("\n=== 台球瞄准线调试工具 ===\n")
    
    windows = list_windows()
    print("可用窗口列表：")
    for idx, (hwnd, title) in enumerate(windows[:30]):
        print(f"  {idx + 1}. {title}")
    
    print("\n请输入窗口编号：")
    choice = input("> ").strip()
    
    if not choice.isdigit():
        print("输入无效")
        return
    
    idx = int(choice) - 1
    if idx < 0 or idx >= len(windows):
        print("窗口编号超出范围")
        return
    
    target_hwnd, target_title = windows[idx]
    print(f"\n✓ 已选择: {target_title}")
    
    # 获取窗口位置
    try:
        point = win32gui.ClientToScreen(target_hwnd, (0, 0))
        client_rect = win32gui.GetClientRect(target_hwnd)
        target_rect = (point[0], point[1], 
                      point[0] + client_rect[2], 
                      point[1] + client_rect[3])
        print(f"窗口区域: {target_rect}")
    except Exception as e:
        print(f"获取窗口位置失败: {e}")
        return
    
    print("\n开始分析...")
    print("操作说明：")
    print("  - 画面会实时显示4个窗口的检测结果")
    print("  - 按 SPACE 暂停并保存当前帧分析")
    print("  - 按 Q 退出")
    print("\n左上：原始画面")
    print("右上：边缘检测")
    print("左下：颜色过滤（白/黄/青/红）")
    print("右下：检测到的线条\n")
    
    frame_count = 0
    while True:
        img = capture_screen_region(target_rect)
        if img is None:
            time.sleep(0.1)
            continue
        
        frame_count += 1
        
        analyze_frame(img)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            # 保存截图
            filename = f"e:/saomiao/debug_frame_{int(time.time())}.jpg"
            cv2.imwrite(filename, img)
            print(f"\n已保存截图: {filename}")
            print("按任意键继续...")
            cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    print("\n调试结束")

if __name__ == "__main__":
    main()
