'''
改进版：引入邻域平滑滤波算法，消除轮廓毛刺
同时保留：物理溅射逻辑 (0.1概率) 和 移除入射角限制
'''
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
import os
import csv
import glob
from scipy.ndimage import uniform_filter  # 用于矩阵平滑

matplotlib.use('TkAgg')

# ================= 配置路径 =================
SAVE_DIR = r"E:\MachineLearning\data\py\Monte_Carlo\Monte_Carlo\Csv\Test27_Smooth"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
# ===========================================

class Si_Class:
    def __init__(self, existflag = True, CountCl = 0, material_type = 'Si', reflect_count = 0):
        self.existflag = existflag
        self.CountCl = CountCl
        self.material_type = material_type 
        self.reflect_count = reflect_count

# ---------------- 核心修改：平滑化的轮廓提取函数 ----------------
def extract_and_transform_contour(Si_array, rows, cols, smoothing_size=3):
    """
    提取轮廓并进行空间平滑处理
    smoothing_size: 平滑窗口大小，3代表3x3，数值越大越平滑但会丢失细节
    """
    # 1. 将对象数组转换为 0-1 浮点矩阵 (1表示存在，0表示真空)
    grid = np.zeros((rows, cols), dtype=float)
    for x in range(rows):
        for y in range(cols):
            if Si_array[x, y].existflag:
                grid[x, y] = 1.0

    # 2. 执行均值滤波平滑处理
    # 这步会消除孤立的像素点（噪声）
    smoothed_grid = uniform_filter(grid, size=smoothing_size)
    
    # 3. 重新二值化（判定：邻域内超过一半的格点存在，则认为该处有物质）
    refined_grid = smoothed_grid > 0.5

    contour_points = []
    # 4. 在平滑后的矩阵上探测边界
    # 垂直探测
    for y in range(cols):
        for x in range(rows - 1):
            if refined_grid[x, y] != refined_grid[x + 1, y]:
                contour_points.append((x + 0.5, y))
    # 水平探测
    for x in range(rows):
        for y in range(cols - 1):
            if refined_grid[x, y] != refined_grid[x, y + 1]:
                contour_points.append((x, y + 0.5))

    # 5. 坐标转换 (保持 rows - 1 - x 逻辑)
    transformed_points = []
    for x, y in contour_points:
        new_x = rows - 1 - x
        transformed_points.append((new_x, y))
    
    return transformed_points

# ---------------- 其余物理逻辑函数保持不变 ----------------

def return_next(emission_x, emission_y, emission_k, px, py, is_reflect, direction=1, rows=None, cols=None):
    if is_reflect:
        if direction >= 0:
            if emission_k > 0: nx, ny = px + 1, py + 1
            elif emission_k < 0: nx, ny = px - 1, py + 1
            else: nx, ny = px, py + 1
        else:
            if emission_k > 0:
                y = emission_y + emission_k * ((px - 0.5) - emission_x)
                if y <= py - 0.5: nx, ny = px, py - 1
                else: nx, ny = px - 1, py
            elif emission_k < 0:
                y = emission_y + emission_k * ((px + 0.5) - emission_x)
                if y <= py - 0.5: nx, ny = px, py - 1
                else: nx, ny = px + 1, py
            else: nx, ny = px, py - 1
    else:
        if direction >= 0:
            if emission_k > 0:
                y = emission_y + emission_k * ((px + 0.5) - emission_x)
                if y <= py + 0.5: nx, ny = px + 1, py
                else: nx, ny = px, py + 1
            elif emission_k < 0:
                y = emission_y + emission_k * ((px - 0.5) - emission_x)
                if y <= py + 0.5: nx, ny = px - 1, py
                else: nx, ny = px, py + 1
            else: nx, ny = px, py + 1
        else:
            if emission_k > 0:
                y = emission_y + emission_k * ((px - 0.5) - emission_x)
                if y <= py - 0.5: nx, ny = px, py - 1
                else: nx, ny = px - 1, py
            elif emission_k < 0:
                y = emission_y + emission_k * ((px + 0.5) - emission_x)
                if y <= py - 0.5: nx, ny = px, py - 1
                else: nx, ny = px + 1, py
            else: nx, ny = px, py - 1
    if rows is not None:
        nx = int(nx) % int(rows)
    return [nx, ny]

def calculate_Ysicl(abs_angle, Ei=50.0, Eth=20.0, C=0.77):
    energy_term = math.sqrt(Ei) - math.sqrt(Eth)
    acr = math.radians(45)
    f_alpha = 1.0 if abs_angle <= acr else math.cos(abs_angle) / math.cos(acr)
    return C * energy_term * f_alpha

def calculate_Ychem():
    v, Ne, r, Ncl, Ts = 4.04e-28/60, 1e20, 0.39, 1e11, 300
    E, k = 4.7 * 4186.8 / 6.02e23, 1.38e-23
    ERchem = v * (Ne ** r) * Ncl * math.sqrt(Ts) * math.exp(-E / (k * Ts))
    Tg, mcl = 400, 5.88e-26
    u = math.sqrt(8 * k * Tg / (math.pi * mcl))
    Tcl = (1 / 4) * Ncl * u
    return ERchem / Tcl

def reflect_prob(theta, material):
    if material in ['Hardmask', 'Si']:
        angle_else = max(0, 1 * (theta - math.pi/3) / (math.pi/2 - math.pi/3))
        return min(1, angle_else)
    return 0.0

def reflector_face(Si_array, center_i, center_j, n=4):
    rows, cols = Si_array.shape
    i_min, j_min = max(0, center_i-n), max(0, center_j-n)
    i_max, j_max = min(rows-1, center_i+n), min(cols-1, center_j+n)
    x_list, y_list = [], []
    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            if Si_array[i, j].existflag:
                if (i>0 and not Si_array[i-1, j].existflag) or (j>0 and not Si_array[i, j-1].existflag) or \
                   (i<rows-1 and not Si_array[i+1, j].existflag) or (j<cols-1 and not Si_array[i, j+1].existflag):
                    x_list.append(i); y_list.append(j)
    if len(x_list) < 2: return 0, np.array([0, -1])
    x_list, y_list = np.array(x_list), np.array(y_list)
    A = np.vstack((np.ones_like(x_list), x_list)).T
    theta_fit = np.linalg.lstsq(A, y_list, rcond=None)[0]
    k = theta_fit[1]
    n_vector = np.array([-k, 1])
    test_i = int(round(center_i + n_vector[0] * 0.1))
    test_j = int(round(center_j + n_vector[1] * 0.1))
    if 0 <= test_i < rows and 0 <= test_j < cols and Si_array[test_i, test_j].existflag:
        n_vector = -n_vector
    norm = np.linalg.norm(n_vector)
    return k, n_vector / norm if norm > 1e-9 else np.array([0, -1])

def calculate_acute_angle(vec1, vec2):
    norm_p = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if norm_p == 0: return 0.0
    return np.arccos(np.clip(abs(np.dot(vec1, vec2) / norm_p), -1.0, 1.0))

def reflect_angle(Si_array, px, py, k, species, direction = 1):
    if not Si_array[px, py].existflag: return 0, k, np.array([0,0])
    V_in = np.array([direction/k if abs(k)>1e-9 else direction*1e9, direction])
    V_in /= np.linalg.norm(V_in)
    _, N = reflector_face(Si_array, px, py)
    abs_angle = calculate_acute_angle(V_in, N)
    if random.random() < reflect_prob(abs_angle, Si_array[px, py].material_type):
        V_out = V_in - 2 * np.dot(V_in, N) * N
        ref_k = V_out[1]/V_out[0] if abs(V_out[0])>1e-10 else 999.0
        if species == 0: # 中性漫反射
            new_angle = math.atan2(N[1], N[0]) + (random.random()-0.5)*math.pi/2
            ref_k = math.tan(new_angle) if abs(abs(new_angle)-math.pi/2)>1e-9 else 999.0
            V_out = np.array([-direction/ref_k, -direction])
            V_out /= np.linalg.norm(V_out)
        return 1, ref_k, V_out
    return 0, k, np.array([0,0])

def collisionprocess(Si_array, px, py, species, reaction_probabilities, abs_angle, s_image):
    if not Si_array[px, py].existflag: return False
    Ysicl = calculate_Ysicl(abs_angle) if species == 1 else 0.0
    count = Si_array[px, py].CountCl
    prob_che = reaction_probabilities[count]['Cl+' if species==1 else 'Cl*']
    if Si_array[px, py].material_type == 'Hardmask': prob_che *= 0.1

    # 物理溅射(0.1) + 化学刻蚀逻辑
    is_physical = (species == 1 and Si_array[px, py].material_type == 'Si' and random.random() < 0.1)
    is_chemical = (random.random() < (prob_che * Ysicl if species==1 else prob_che))
    
    clear = is_physical or is_chemical
    if clear:
        Si_array[px, py].existflag = False
        s_image[px, py] = 25
        if Ysicl > 1.0: # 链式反应
            for nx, ny in [(px+1,py),(px-1,py),(px,py+1),(px,py-1)]:
                if 0<=nx<Si_array.shape[0] and 0<=ny<Si_array.shape[1] and Si_array[nx,ny].CountCl==4 and random.random()<Ysicl:
                    Si_array[nx,ny].existflag = False; s_image[nx,ny] = 25
    elif not species and count < 4:
        Si_array[px, py].CountCl += 1
    return clear

def save_contour_to_csv(points, filepath):
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y'])
        writer.writerows(points)

# ---------------- 主程序 ----------------
def main():
    rows, cols, vacuum, deep_border = 800, 700, 50, 200
    left_border, right_border = 300, 500
    total_particles = 200000
    ratio = (10/11) / 1.0
    
    Si_array = np.empty((rows, cols), dtype=object)
    for i in range(rows):
        for j in range(cols):
            Si_array[i, j] = Si_Class(existflag=(j>=vacuum))

    k_mask = abs(math.tan(30 * math.pi/90))
    s_image = np.ones((rows, cols))
    for y in range(cols):
        for x in range(rows):
            if y < vacuum: Si_array[x,y].existflag = False; s_image[x,y]=25
            elif y < deep_border:
                offset = int((deep_border - y) * k_mask)
                if (left_border - offset) < x < (right_border + offset):
                    Si_array[x,y].existflag = False; s_image[x,y]=25
                else:
                    Si_array[x,y].material_type = 'Hardmask'; s_image[x,y]=40

    probs = {0:{'Cl*':0.0,'Cl+':0.1}, 1:{'Cl*':0.1,'Cl+':0.3}, 2:{'Cl*':0.2,'Cl+':0.3}, 3:{'Cl*':0.3,'Cl+':0.3}, 4:{'Cl*':1.0,'Cl+':0.3}}

    print("开始模拟...")
    for count_num in range(1, total_particles + 1):
        if count_num % 100000 == 0:
            print(f"进度: {count_num}/{total_particles}, 提取平滑轮廓并保存...")
            # 提取平滑后的轮廓 (平滑窗口设为3)
            current_contour = extract_and_transform_contour(Si_array, rows, cols, smoothing_size=3)
            save_contour_to_csv(current_contour, os.path.join(SAVE_DIR, f"contour_{count_num}.csv"))

        emission_x = random.random() * (rows - 1)
        species = random.random() > ratio
        sigma = np.radians(1.91 if species==1 else 7.64)
        angle_rad = random.gauss(0, sigma)
        emission_k = 1.0 / math.tan(np.clip(angle_rad, -1.5, 1.5))
        
        px, py = int(emission_x), 1
        direction = 1
        for _ in range(4000):
            if not (0 <= px < rows and 0 <= py < cols): break
            if Si_array[px, py].existflag:
                is_ref, new_k, V_out = reflect_angle(Si_array, px, py, emission_k, species, direction)
                if is_ref:
                    direction = 1 if V_out[1] >= 0 else -1
                    emission_k = new_k
                    px, py = return_next(px, py, emission_k, px, py, 1, direction, rows, cols)
                else:
                    collisionprocess(Si_array, px, py, species, probs, abs(angle_rad), s_image)
                    break
            else:
                px, py = return_next(px, py, emission_k, px, py, 0, direction, rows, cols)

    # ==================== 绘图优化：更加平滑的视觉呈现 ====================
    plt.figure(figsize=(12, 8))
    rotated_img = np.rot90(s_image, -1)
    plt.imshow(rotated_img, cmap='jet', vmin=0, vmax=100)
    
    # 绘制历史轮廓（用极小的点和透明度，模拟“包络线”效果）
    csv_files = glob.glob(os.path.join(SAVE_DIR, "contour_*.csv"))
    csv_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    for filepath in csv_files:
        data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
        if data.size > 0:
            # 使用 s=0.5 (极小) 和 alpha=0.3 (半透明) 来减少视觉毛刺感
            plt.scatter(data[:,0], data[:,1], s=0.5, color='red', alpha=0.3)

    # 绘制最终平滑轮廓 (白色)
    final_contour = extract_and_transform_contour(Si_array, rows, cols, smoothing_size=5) # 最终轮廓可以再加大平滑度
    if final_contour:
        pts = np.array(final_contour)
        plt.scatter(pts[:, 0], pts[:, 1], s=1, color='white', alpha=0.9, label='Final Smooth')

    plt.title('Smoothed Etch Profile Monte Carlo')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()