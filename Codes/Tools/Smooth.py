import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import measure

def smooth_contour_from_csv(file_path, rows=800, cols=700, sigma=2.0):
    # 1. 读取 CSV 数据
    df = pd.read_csv(file_path)
    x_coords = df['x'].values
    y_coords = df['y'].values

    # 2. 构建二值网格矩阵 (重建图像)
    # 假设图像尺寸与你模拟的 ROWS, COLS 一致
    grid = np.zeros((rows, cols))
    
    # 将坐标点映射到矩阵中
    # 注意：轮廓点通常是实体的边界，我们需要填充实体内部或对边界进行膨胀
    for x, y in zip(x_coords, y_coords):
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < rows and 0 <= iy < cols:
            grid[ix, iy] = 1

    # 3. 图像处理平滑化
    # 使用高斯滤波平滑离散的点阵
    smoothed_grid = gaussian_filter(grid.astype(float), sigma=sigma)
    
    # 二值化处理：设定阈值重新获得清晰边界
    binary_grid = smoothed_grid > smoothed_grid.mean()

    # 4. 重新提取平滑后的轮廓
    # 使用 marching squares 算法提取等值线
    contours = measure.find_contours(binary_grid, 0.5)

    # 5. 可视化对比
    plt.figure(figsize=(10, 6))
    
    # 原始离散点
    plt.scatter(x_coords, y_coords, s=1, c='red', alpha=0.5, label='Original (Noisy)')
    
    # 平滑后的轮廓
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], c='blue', lw=2, label='Smoothed (Gaussian)')
            
    plt.title(f"Contour Smoothing (Sigma={sigma})")
    plt.gca().invert_yaxis() # 匹配刻蚀模拟坐标系
    plt.legend()
    plt.show()

    return contours

# 执行平滑
# 如果你的 CSV 路径不同，请修改路径
# contours = smooth_contour_from_csv('contour_300000.csv', sigma=2.5)