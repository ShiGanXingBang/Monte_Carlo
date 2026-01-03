import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
import os
import csv
import glob

matplotlib.use('TkAgg')  # 或者 'Qt5Agg', 'Agg' 等

# ================= 配置路径 =================
# CSV 保存路径 (使用 r'' 原始字符串防止转义问题)
SAVE_DIR = r"E:\MachineLearning\data\py\Monte_Carlo\Monte_Carlo\Csv"

# 确保文件夹存在
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"已创建目录: {SAVE_DIR}")
# ===========================================

# 定义硅原子类属性
class Si_Class:
    def __init__(self, existflag = True, CountCl = 0, material_type = 'Si',reflect_count = 0):
        self.existflag = existflag
        self.CountCl = CountCl
        self.material_type = material_type # 'Si' 'Hardmask'
        self.reflect_count = reflect_count

# 运动方向判断
def return_next(emission_x, emission_y, emission_k, px, py, is_reflect, direction=1, rows=None, cols=None):
    if is_reflect:
        if direction >= 0:
            if emission_k > 0:
                nx, ny = px + 1, py + 1
            elif emission_k < 0:
                nx, ny = px - 1, py + 1
            else:
                nx, ny = px, py + 1
        else:
            if emission_k > 0:
                y = emission_y + emission_k * ((px - 0.5) - emission_x)
                if y <= py - 0.5:
                    nx, ny = px, py - 1
                else:
                    nx, ny = px - 1, py
            elif emission_k < 0:
                y = emission_y + emission_k * ((px + 0.5) - emission_x)
                if y <= py - 0.5:
                    nx, ny = px, py - 1
                else:
                    nx, ny = px + 1, py
            else:
                nx, ny = px, py - 1
    else:
        if direction >= 0:
            if emission_k > 0:
                y = emission_y + emission_k * ((px + 0.5) - emission_x)
                if y <= py + 0.5:
                    nx, ny = px + 1, py
                else:
                    nx, ny = px, py + 1
            elif emission_k < 0:
                y = emission_y + emission_k * ((px - 0.5) - emission_x)
                if y <= py + 0.5:
                    nx, ny = px - 1, py
                else:
                    nx, ny = px, py + 1
            else:
                nx, ny = px, py + 1
        else:
            if emission_k > 0:
                y = emission_y + emission_k * ((px - 0.5) - emission_x)
                if y <= py - 0.5:
                    nx, ny = px, py - 1
                else:
                    nx, ny = px - 1, py
            elif emission_k < 0:
                y = emission_y + emission_k * ((px + 0.5) - emission_x)
                if y <= py - 0.5:
                    nx, ny = px, py - 1
                else:
                    nx, ny = px + 1, py
            else:
                nx, ny = px, py - 1

    if rows is not None and cols is not None:
        nx = int(nx) % int(rows) 
    return [nx, ny]

def calculate_Ysicl(abs_angle, Ei=50.0, Eth=20.0, C=0.77):
    energy_term = math.sqrt(Ei) - math.sqrt(Eth)
    acr = math.radians(45)
    if abs_angle <= acr:
        f_alpha = 1.0
    else:
        f_alpha = math.cos(abs_angle) / math.cos(acr)
    Ysicl = C * energy_term * f_alpha
    return Ysicl

def calculate_Ychem():
    v = (4.04e-28) / 60 
    Ne = 1e20 
    r = 0.39 
    Ncl = 1e11 
    Ts = 300 
    E = 4.7 * 4186.8 / (6.02e23) 
    k = 1.3806503e-23 
    ERchem = v * (Ne ** r) * Ncl * math.sqrt(Ts) * math.exp(-E / (k * Ts))
    Tg = 400 
    mcl = 5.8877e-26 
    u = math.sqrt(8 * k * Tg / (math.pi * mcl))
    Tcl = (1 / 4) * Ncl * u 
    Ychem = ERchem / Tcl
    return Ychem

def chain_reaction(Si_array, px, py, Ysicl, s_image):
    neighbor = [(px + 1, py),(px - 1, py),(px, py + 1),(px, py - 1)]
    for nx, ny in neighbor:
        if 0 <= nx < Si_array.shape[0] and 0 <= ny < Si_array.shape[1]:
            if Si_array[nx, ny].existflag and Si_array[nx, ny].CountCl == 4:
                if random.random() < Ysicl:
                    Si_array[nx, ny].existflag = False
                    s_image[nx, ny] = 25

def reflect_prob(theta, material, species):
    threshold = math.pi/3
    if material == 'Hardmask':
        if species == 1:           
            base_prob = 0
            angle_else = max(0, 1 * (theta - threshold) / (math.pi/2 - threshold))
            angle_else = min(1, angle_else)
            return base_prob + angle_else
        elif species == 0:
            # 漫反射
            # angle_else = random.random()
            # return angle_else
            base_prob = 0
            angle_else = max(0, 1 * (theta - threshold) / (math.pi/2 - threshold))
            angle_else = min(1, angle_else)
            return base_prob + angle_else
        else:
            return 0.0
    elif material == 'Si':
        if species == 1:           
            base_prob = 0
            angle_else = max(0, 1 * (theta - threshold) / (math.pi/2 - threshold))
            angle_else = min(1, angle_else)
            return base_prob + angle_else
        elif species == 0:          
            # 漫反射
            # angle_else = random.random()
            # return angle_else
            base_prob = 0
            angle_else = max(0, 1 * (theta - threshold) / (math.pi/2 - threshold))
            angle_else = min(1, angle_else)
            return base_prob + angle_else
        else:
            return 0.0
    else:
        return 0.0

def reflector_face(Si_array, center_i, center_j, n=4, left_border = 350):
    rows, cols = Si_array.shape
    i_min = max(0, center_i - n)
    j_min = max(0, center_j - n)
    i_max = min(rows - 1, center_i + n)
    j_max = min(cols - 1, center_j + n)
    x_list = []
    y_list = []
    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            if not Si_array[i, j].existflag:
                continue
            is_boundary = False
            if (i > 0 and not Si_array[i - 1, j].existflag) or \
               (j > 0 and not Si_array[i, j - 1].existflag) or \
               (i < rows - 1 and not Si_array[i + 1, j].existflag) or \
               (j < cols - 1 and not Si_array[i, j + 1].existflag):
                is_boundary = True
            if is_boundary:
                x_list = np.append(x_list, i)   
                y_list = np.append(y_list, j)
    x_list = np.array(x_list)
    y_list = np.array(y_list)
    
    if len(x_list) < 2:
        return 0, np.array([0, -1])

    x_std = np.std(x_list) if len(x_list) > 1 else 0
    y_std = np.std(y_list) if len(y_list) > 1 else 0
    threshold = 0.5 
    
    if y_std < threshold and len(y_list) >= 2:
        y_mean = np.mean(y_list)
        test_up = int(round(y_mean - 1))
        test_down = int(round(y_mean + 1))
        
        if 0 <= center_i < rows and 0 <= test_up < cols:
            if not Si_array[center_i, test_up].existflag:
                n_vector = np.array([0, -1])
                k = 0.0 
                n_normal = n_vector / np.linalg.norm(n_vector)
                return k, n_normal
        if 0 <= center_i < rows and 0 <= test_down < cols:
            if not Si_array[center_i, test_down].existflag:
                n_vector = np.array([0, 1])
                k = 0.0 
                n_normal = n_vector / np.linalg.norm(n_vector)
                return k, n_normal
        n_vector = np.array([0, -1])
        k = 0.0
        n_normal = n_vector / np.linalg.norm(n_vector)
        return k, n_normal

    A = np.vstack((np.ones_like(x_list), x_list)).T
    try:
        theta = np.linalg.lstsq(A, y_list, rcond=None)[0]
    except np.linalg.LinAlgError:
        if center_i <= left_border:
            return 999, np.array([1, 0]) 
        else:
            return 999, np.array([-1, 0]) 

    k = theta[1]
    if abs(k) > 1000: 
        if x_std < y_std * 0.3: 
            x_mean = np.mean(x_list)
            test_left = int(round(x_mean - 1))
            test_right = int(round(x_mean + 1))
            
            if 0 <= test_left < rows and 0 <= center_j < cols:
                if not Si_array[test_left, center_j].existflag:
                    n_vector = np.array([-1, 0])
                    k = 999.0
                    n_normal = n_vector / np.linalg.norm(n_vector)
                    return k, n_normal
            if 0 <= test_right < rows and 0 <= center_j < cols:
                if not Si_array[test_right, center_j].existflag:
                    n_vector = np.array([1, 0])
                    k = 999.0
                    n_normal = n_vector / np.linalg.norm(n_vector)
                    return k, n_normal
    
    n_vector = np.array([-k, 1])

    epsilon = 0.1
    test_i = int(round(center_i + n_vector[0] * epsilon))
    test_j = int(round(center_j + n_vector[1] * epsilon))

    if 0 <= test_i < rows and 0 <= test_j < cols:
        if Si_array[test_i, test_j].existflag:
            n_vector = -n_vector 
    
    norm = np.linalg.norm(n_vector)
    if norm < 1e-9:
        return 0, np.array([0, -1]) 

    n_normal = n_vector / norm
    return k, n_normal

def calculate_acute_angle(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if norm_product == 0:
        return 0.0 
    cos_theta = dot_product / norm_product
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    acute_angle_rad = np.arccos(abs(cos_theta))
    return acute_angle_rad

def function_angle(Si_array, px, py, k, direction = 1, left_border = 350):
    if Si_array[px, py].existflag == True:
        incident_angle = math.atan(abs(k))
        if incident_angle > math.radians(10):
            if abs(k) < 1e-9:  
                V_in = np.array([direction * float('inf'), direction])
            else:
                V_in = np.array([direction / k, direction])

            V_in = V_in / np.linalg.norm(V_in)
            result = reflector_face(Si_array, px, py, n=4)
            if result is None or result[1] is None:
                if px <= left_border:
                    N = np.array([1, 0]) 
                else:
                    N = np.array([-1, 0]) 
            else:
                _, N = result
            abs_angle = calculate_acute_angle(V_in, N)
            return abs_angle
# 反射角度        
def reflect_angle(Si_array, px, py, k, species, direction = 1, left_border = 350):
    is_reflect_flag = 0
    reflect_k = k
    V_out = np.array([0, 0])

    if Si_array[px, py].existflag == True:
        incident_angle = math.atan(abs(k))
        if incident_angle > math.radians(10):
            if abs(k) < 1e-9: 
                V_in = np.array([direction * float('inf'), direction])
            else:
                V_in = np.array([direction / k, direction])

            V_in = V_in / np.linalg.norm(V_in)
            result = reflector_face(Si_array, px, py, n=4)
            if result is None or result[1] is None:
                if px <= left_border:
                    N = np.array([1, 0]) 
                else:
                    N = np.array([-1, 0]) 
            else:
                _, N = result
            abs_angle = calculate_acute_angle(V_in, N)
            reflext_prob = reflect_prob(abs_angle, Si_array[px, py].material_type, species)

            if random.random() < reflext_prob:
                V_out = V_in - 2 * (np.dot(V_in, N)) * N
                if abs(V_out[0]) < 1e-10: 
                    reflect_k = float('inf') 
                else:
                    reflect_k = V_out[1] / V_out[0]
                
                # 中性粒子漫反射机制
                # if species == 0:
                #     current_reflect_angle = math.atan2(N[1], N[0]) 
                #     angle_increment = (random.random()-0.5) * (math.pi / 2) 
                #     new_reflect_angle = current_reflect_angle + angle_increment
                #     if abs(abs(new_reflect_angle) - math.pi/2) < 1e-9:
                #         reflect_k = 999.0
                #     else:
                #         reflect_k = math.tan(new_reflect_angle)
                #     V_out = np.array([-direction / reflect_k, -direction])   
                #     V_out = V_out / np.linalg.norm(V_out) 

                is_reflect_flag = 1
                return is_reflect_flag, reflect_k, V_out

        return is_reflect_flag, reflect_k, V_out
    else:
        return is_reflect_flag, reflect_k, V_out

def collisionprocess(Si_array, px, py, species, reaction_probabilities, abs_angle, s_image):
    if not  Si_array[px, py].existflag:
        return False
    if species == 1:
        particle_type = 'Cl+'
        Ysicl = calculate_Ysicl(abs_angle)
    else:
        particle_type = 'Cl*'
        Ysicl = 0.0 

    Count = Si_array[px, py].CountCl
    if Si_array[px, py].material_type == 'Si':
        prob = reaction_probabilities[Count][particle_type]
    elif Si_array[px, py].material_type == 'Hardmask':
        prob = reaction_probabilities[Count][particle_type] * 0.02

    clearflag = False
    if species == 1 and random.random() < prob * Ysicl:
        clearflag = True
    elif species == 0 and random.random() < prob:
        clearflag = True

    if Ysicl > 1.0 and clearflag:
        chain_reaction(Si_array, px, py, Ysicl, s_image)

    if species == 0 and Si_array[px, py].CountCl == 4:
        Ychem = calculate_Ychem()
        if random.random() < Ychem:
            clearflag = True
    if not species and Si_array[px, py].CountCl < 4:
        Si_array[px, py].CountCl +=1

    Si_array[px, py].existflag = not clearflag
    return clearflag

# ==================== 新增功能：提取轮廓并转换坐标 ====================
def extract_and_transform_contour(Si_array, rows, cols):
    """提取当前的轮廓点，并转换为绘图坐标系 (rows-1-x, y)"""
    contour_points = []
    for y in range(cols):  # 遍历每一列
        for x in range(rows - 1):  # 从上到下扫描
            if Si_array[x, y].existflag != Si_array[x + 1, y].existflag:
                contour_points.append((x + 0.5, y))  # 记录边界点
    for x in range(rows):  # 遍历每一列
        for y in range(cols - 1):  # 从上到下扫描
            if Si_array[x, y].existflag != Si_array[x, y + 1].existflag:
                contour_points.append((x, y + 0.5))  # 记录边界点

    # 坐标转换
    transformed_points = []
    for x, y in contour_points:
        new_x = rows - 1 -x
        transformed_points.append((new_x, y))
    
    return transformed_points

def save_contour_to_csv(points, filepath):
    """将转换后的轮廓点写入 CSV"""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y']) # 写入表头
        writer.writerows(points)
# ===================================================================

def main():
    vacuum = 50
    rows = 800
    cols = 1000
    left_border = 200
    right_border = 600
    deep_border = 200
    count_num = 0
    # 粒子总数
    C1 = 1
    P = 1500000
    Total_nums = C1 * P
    C2 = 1
    PW = 21 / 20
    Ratio = C2 / PW
    C3 = 1
    V_bias = 1
    E = C3 * V_bias

    start_time = time.perf_counter()
    angle_img = abs(30 * math.pi/90)
    Si_array = np.empty(shape=(rows,cols), dtype=object)
    for i in range(rows):
        for j in range(vacuum):
            Si_array[i, j] = Si_Class(existflag = False)
        for j in range(vacuum, cols):
            Si_array[i, j] = Si_Class()

    k_img = abs(math.tan(angle_img))
    if deep_border * k_img > rows / 6:
        k_img = (rows - right_border - 1) / (deep_border * 3)
    s_image = np.ones((rows, cols))
    for y in range(vacuum):
        for x in range(rows):
            if Si_array[x, y].existflag == False:
                s_image[x, y] = 25 
    for y in range(vacuum, deep_border):
        offset = int((deep_border - y) * k_img) 
        left_current = left_border - offset
        right_current = right_border + offset
        left_current = max(0, min(left_current, rows - 1))
        right_current = max(0, min(right_current, rows - 1))
        for x in range(rows):
            if  left_current < x < right_current:
                s_image[x, y] = 25  
                Si_array[x, y].existflag = False
            else:
                s_image[x, y] = 40  
                Si_array[x, y].material_type = 'Hardmask'
                Si_array[x, y].existflag = True
    
    fig = plt.figure(figsize=(12, 8))
    
    reaction_probabilities = {
        0: {'Cl*': 0.0, 'Cl+': 0.1},  
        1: {'Cl*': 0.1, 'Cl+': 0.3},  
        2: {'Cl*': 0.2, 'Cl+': 0.3},  
        3: {'Cl*': 0.3, 'Cl+': 0.3},  
        4: {'Cl*': 1.0, 'Cl+': 0.3}  
    }

    # 清空之前的 CSV (可选，防止混淆)
    # for f in glob.glob(os.path.join(SAVE_DIR, "*.csv")):
    #    os.remove(f)

    # 模拟粒子入射
    for cl in range(Total_nums):
        count_num += 1
        
        # ==================== 修改：每10000次保存一次CSV ====================
        if count_num % 250000 == 0:
            print(f"当前循环次数为{count_num}，正在保存轮廓...")
            # 提取转换后的轮廓点
            current_contour = extract_and_transform_contour(Si_array, rows, cols)
            # 构造文件名
            filename = f"contour_{count_num}.csv"
            filepath = os.path.join(SAVE_DIR, filename)
            # 保存
            save_contour_to_csv(current_contour, filepath)
        # ===================================================================

        emission_x = random.random() * (rows - 1)
        emission_y_Ion = 1
        emission_y_neutral =  1
        
        species = random.random() > Ratio
             
        if species == 1:
            sigma_degrees = 1.91
            sigma = np.radians(sigma_degrees) 
            angle_rad = random.gauss(0, sigma)
            angle_rad = max(min(angle_rad, math.pi/2), -math.pi/2)
            abs_angle = abs(angle_rad)
            emission_k = 1.0 / math.tan(angle_rad)
            emission_y = emission_y_Ion
        else:
            sigma_degrees = 7.64
            sigma = np.radians(sigma_degrees)  
            angle_rad = random.gauss(0, sigma)
            angle_rad = max(min(angle_rad, math.pi/2), -math.pi/2)
            abs_angle = abs(angle_rad)
            emission_k = 1.0 / math.tan(angle_rad)
            emission_y = emission_y_neutral
      
        abs_k = np.abs(emission_k)

        if abs_k <= 0.1:
            continue
        elif abs_k >= 200:
            px = math.ceil(emission_x) 
            if left_border < px < right_border:
                for py in range(deep_border + 1, cols):
                    if 0 <= px < rows and Si_array[px, py].existflag:
                        clearflag = collisionprocess(Si_array, px, py, species, reaction_probabilities, abs_angle, s_image) 
                        if clearflag:
                            s_image[px, py] = 25  
                        break
            else:
                for py in range(vacuum + 1, cols):
                    if 0 <= px < rows and Si_array[px, py].existflag:
                        clearflag = collisionprocess(Si_array, px, py, species, reaction_probabilities, abs_angle, s_image) 
                        if clearflag:
                            s_image[px, py] = 25 
                        break
        else:
            px = int(emission_x)
            py = emission_y
                
        max_steps = 4000 
        ref_count = 0
        count = 1
        particle_direction = 1 
        for step in range(max_steps):
            if not (0 <= px < rows  and 0 <= py < cols):
                break
            if Si_array[px, py].existflag:
                is_reflect, new_k, V_out= reflect_angle(Si_array, px, py, emission_k, species, particle_direction)
                if is_reflect and ref_count < count:
                    ref_count = ref_count + 1
                    particle_direction = 1 if V_out[1] >= 0 else -1
                    emission_k = new_k
                    emission_x, emission_y = px, py 
                    next_pos = return_next(emission_x, emission_y, emission_k, px, py, is_reflect, particle_direction, rows, cols)
                    px, py = next_pos
                    continue  
                else: 
                    function_angle(Si_array, px, py, emission_k, particle_direction)
                    clearflag = collisionprocess(Si_array, px, py, species, reaction_probabilities, abs_angle, s_image) 
                    if clearflag:
                        s_image[px, py] = 25
                    break
            else:
                next_pos = return_next(emission_x, emission_y, emission_k, px, py, 0, particle_direction, rows, cols)
                px, py = next_pos


    # ==================== 绘制历史轮廓（点）+ 最终轮廓（着色） ====================
    # 用最终的 s_image 着色（仿真结果）
    rotated_image = np.rot90(s_image, -1)
    plt.imshow(rotated_image, cmap='jet', vmin=0, vmax=100)
    
    # 读取并绘制所有 CSV 轮廓（历史轨迹）
    csv_files = glob.glob(os.path.join(SAVE_DIR, "contour_*.csv"))
    
    # 按数字大小排序
    try:
        csv_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    except:
        csv_files.sort()

    # 使用颜色映射表，让不同时间的轮廓显示不同颜色
    # colors = plt.cm.spring(np.linspace(0, 1, len(csv_files)))

    print(f"开始绘制 {len(csv_files)} 条历史轮廓线...")
    
    for idx, filepath in enumerate(csv_files):
        try:
            # 读取 CSV
            x_data = []
            y_data = []
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                next(reader) # 跳过表头
                for row in reader:
                    if row:
                        x_data.append(float(row[0]))
                        y_data.append(float(row[1]))
            
            # 绘制历史轮廓的点
            # plt.scatter(x_data, y_data, s=2, color=colors[idx], alpha=0.8)
            plt.scatter(x_data, y_data, s=2, color='red', alpha=0.8)

        except Exception as e:
            print(f"读取文件 {filepath} 出错: {e}")

    # 绘制最终的轮廓点（边界线，白色）
    final_contour = extract_and_transform_contour(Si_array, rows, cols)
    if final_contour:
        points_array = np.array(final_contour)
        plt.scatter(points_array[:, 0], points_array[:, 1], s=2, color='white', alpha=0.8)

    plt.title('Final Contour with Area Coloring and History Contours')
    plt.axis('equal')
    plt.axis('on')
    plt.tight_layout()
    plt.show()
    # =============================================================
    
    end_time = time.perf_counter()
    print(f"程序总运行时间: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()