import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time

matplotlib.use('TkAgg')  # 或者 'Qt5Agg', 'Agg' 等

# Si_class = {'existflag': True, 'CountCl': 0} #字典
#定义硅原子类属性
class Si_Class:
    def __init__(self, existflag = True, CountCl = 0, material_type = 'Si',reflect_count = 0):
        self.existflag = existflag
        self.CountCl = CountCl
        self.material_type = material_type # 'Si' 'Hardmask'
        self.reflect_count = reflect_count

#运动方向判断
def return_next(emission_x, emission_y, emission_k, px, py, direction=1):
    if direction >= 0:
            #向右下运动
            if emission_k > 0:
                    return [px + 1, py+1]
            #向左下运动
            elif emission_k < 0:
                    return [px - 1, py + 1]
            #垂直向下 k=0
            else:
                return [px, py + 1]
    else:
        if emission_k > 0:
            y = emission_y + emission_k * ((px - 0.5) - emission_x)
            #向左上运动
            if y <= py - 0.5:
                return [px, py - 1]
            else:
                return [px - 1, py]
        #向右上运动
        elif emission_k < 0:
            y = emission_y + emission_k * ((px + 0.5) - emission_x)
            if y <= py - 0.5:
                return [px, py - 1]
            else:
                return [px + 1, py]
        #垂直向下 k=0
        else:
            return [px, py - 1]


# 计算YSi/Cl+,刻蚀产额
def calculate_Ysicl(abs_angle, Ei=50.0, Eth=20.0, C=0.77):
    # 能量项：sqrt(Ei) - sqrt(Eth)
    energy_term = math.sqrt(Ei) - math.sqrt(Eth)

    acr = math.radians(45)
    if abs_angle <= acr:
        f_alpha = 1.0
    else:
        f_alpha = math.cos(abs_angle) / math.cos(acr)
    # 计算刻蚀产额
    Ysicl = C * energy_term * f_alpha
    return Ysicl

#自发反应脱离概率
def calculate_Ychem():
    # 参数设置
    v = (4.04e-28) / 60  # m/s, 刻蚀速度
    Ne = 1e20  # cm⁻³, Si掺杂浓度
    r = 0.39  # 指数参数
    Ncl = 1e11  # cm⁻³, 气相Cl原子密度
    Ts = 300  # K, 衬底温度
    E = 4.7 * 4186.8 / (6.02e23)  # J, 活化能（单位转换）
    k = 1.3806503e-23  # J/K, 玻尔兹曼常数
    # 计算刻蚀速度
    ERchem = v * (Ne ** r) * Ncl * math.sqrt(Ts) * math.exp(-E / (k * Ts))
    #计算cl原子平均速度
    Tg = 400  # K, 气体温度
    mcl = 5.8877e-26  # kg, Cl原子质量
    u = math.sqrt(8 * k * Tg / (math.pi * mcl))
    # 计算粒子通量+Ychem
    Tcl = (1 / 4) * Ncl * u  # 粒子通量
    Ychem = ERchem / Tcl
    return Ychem


# 链式反应概率, Ysicl > 1时影响周边原子
def chain_reaction(Si_array, px, py, Ysicl, s_image):
    neighbor = [
        (px + 1, py),(px - 1, py),(px, py + 1),(px, py - 1)
    ]
    for nx, ny in neighbor:
        if 0 <= nx < Si_array.shape[0] and 0 <= ny < Si_array.shape[1]:
            if Si_array[nx, ny].existflag and Si_array[nx, ny].CountCl == 4:
                if random.random() < Ysicl:
                    Si_array[nx, ny].existflag = False
                    s_image[nx, ny] = 25


# 掩膜和衬底反射概率
def reflect_prob(theta, material):
    if material == 'Hardmask':
        base_prob = 0.4
        angle_else = min(0.6, 0.6 * (theta / math.pi) / 2)
        return base_prob + angle_else
        # 测试
        # return 1.0
    elif material == 'Si':
        base_prob = 0
        angle_else = min(1, 1 * (theta - math.pi/3) / math.pi/6)
        return base_prob + angle_else
        # 测试
        # return 1.0
    else:
        return 0.0


# 生成反射面
def reflector_face(Si_array, center_i, center_j, n=4, left_border = 350):
    #初始化所需条件
    rows, cols = Si_array.shape
    # 边界限制
    i_min = max(0, center_i - n)
    j_min = max(0, center_j - n)
    i_max = min(rows - 1, center_i + n)
    j_max = min(cols - 1, center_j + n)
    x_list = []
    y_list = []
    # 检查是否为边界点
    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            if not Si_array[i, j].existflag:
                continue
            is_boundary = False
            # 简化并加强边界判断逻辑
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
    
    # 两个点以上才能生成反射面
    if len(x_list) < 2:
        # 默认法线垂直向上（指向y轴负方向，即真空）
        return 0, np.array([0, -1])

    # ==================== 检测垂直或水平边界 START ====================
    # 计算边界点坐标的标准差，判断边界方向
    x_std = np.std(x_list) if len(x_list) > 1 else 0
    y_std = np.std(y_list) if len(y_list) > 1 else 0
    
    # 判断阈值（可以根据网格大小调整）
    threshold = 0.5  # 如果标准差小于这个值，认为是垂直或水平边界
    
    # 情况1：垂直边界（x坐标变化很小，几乎垂直的线）
    if x_std < threshold and len(x_list) >= 2:
        # 垂直边界：x = constant，法线应该是水平的
        # 判断材料在哪一侧，法线指向真空
        x_mean = np.mean(x_list)
        # 检查左侧和右侧，看哪边是真空
        test_left = int(round(x_mean - 1))
        test_right = int(round(x_mean + 1))
        
        if 0 <= test_left < rows and 0 <= center_j < cols:
            if not Si_array[test_left, center_j].existflag:
                # 左侧是真空，法线指向左
                n_vector = np.array([-1, 0])
                k = 999.0  # 垂直边界，斜率很大
                n_normal = n_vector / np.linalg.norm(n_vector)
                return k, n_normal
        if 0 <= test_right < rows and 0 <= center_j < cols:
            if not Si_array[test_right, center_j].existflag:
                # 右侧是真空，法线指向右
                n_vector = np.array([1, 0])
                k = 999.0  # 垂直边界，斜率很大
                n_normal = n_vector / np.linalg.norm(n_vector)
                return k, n_normal
        # 如果无法判断，使用基于位置的经验值
        if center_i <= left_border:
            n_vector = np.array([1, 0])  # 右侧可能是真空
        else:
            n_vector = np.array([-1, 0])  # 左侧可能是真空
        k = 999.0
        n_normal = n_vector / np.linalg.norm(n_vector)
        return k, n_normal
    
    # 情况2：水平边界（y坐标变化很小）
    if y_std < threshold and len(y_list) >= 2:
        # 水平边界：y = constant，法线应该是垂直的
        y_mean = np.mean(y_list)
        # 检查上方和下方，看哪边是真空
        test_up = int(round(y_mean - 1))
        test_down = int(round(y_mean + 1))
        
        if 0 <= center_i < rows and 0 <= test_up < cols:
            if not Si_array[center_i, test_up].existflag:
                # 上方是真空，法线指向上
                n_vector = np.array([0, -1])
                k = 0.0  # 水平边界，斜率=0
                n_normal = n_vector / np.linalg.norm(n_vector)
                return k, n_normal
        if 0 <= center_i < rows and 0 <= test_down < cols:
            if not Si_array[center_i, test_down].existflag:
                # 下方是真空，法线指向下
                n_vector = np.array([0, 1])
                k = 0.0  # 水平边界，斜率=0
                n_normal = n_vector / np.linalg.norm(n_vector)
                return k, n_normal
        # 默认指向上方（通常真空在上）
        n_vector = np.array([0, -1])
        k = 0.0
        n_normal = n_vector / np.linalg.norm(n_vector)
        return k, n_normal
    # ==================== 检测垂直或水平边界 END ====================

    # 情况3：一般倾斜边界，使用线性拟合
    # (A.T @ A) @ theta =  A.T @ y_list.T 公式表示
    #求出k和法向量单位向量  theta = [b, k] # y = b + kx
    A = np.vstack((np.ones_like(x_list), x_list)).T
    try:
        theta = np.linalg.lstsq(A, y_list, rcond=None)[0]
    except np.linalg.LinAlgError:
        if center_i <= left_border:
            return 999, np.array([1, 0]) # 拟合失败则返回默认值
        else:
            return 999, np.array([-1, 0]) # 拟合失败则返回默认值

    k = theta[1]
    # 检查拟合出的k是否过大（可能是垂直边界的误判）
    if abs(k) > 1000:  # 如果k非常大，可能是垂直边界
        # 重新判断：检查x坐标的变化
        if x_std < y_std * 0.3:  # x变化远小于y变化，很可能是垂直边界
            # 使用垂直边界处理
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

    # ==================== 代码修正处 START ====================
    # 检查法向量方向是否正确（是否指向真空）
    # 我们沿着法向量方向移动一个微小的距离，看落点是否在材料内部
    # 如果落点仍在材料内，说明法向量指向了材料，需要反向
    epsilon = 0.1
    test_i = int(round(center_i + n_vector[0] * epsilon))
    test_j = int(round(center_j + n_vector[1] * epsilon))

    # 确保测试点在边界内
    if 0 <= test_i < rows and 0 <= test_j < cols:
        if Si_array[test_i, test_j].existflag:
            n_vector = -n_vector  # 方向反了，进行翻转
    # 如果真空在y轴正方向（下方），则需要这句判断
    # else:
    #     if n_vector[1] > 0:
    #         n_vector = -n_vector

    # ==================== 代码修正处 END ====================
    
    # 对最终正确的法向量进行归一化
    norm = np.linalg.norm(n_vector)
    if norm < 1e-9:
        return 0, np.array([0, -1]) # 避免除以零

    n_normal = n_vector / norm
    return k, n_normal
# # 生成反射面
# def reflector_face(Si_array, center_i, center_j, n=4):
#     #初始化所需条件
#     rows, cols = Si_array.shape
#     # 边界限制
#     i_min = max(0, center_i - n)
#     j_min = max(0, center_j - n)
#     i_max = min(rows - 1, center_i + n)
#     j_max = min(cols - 1, center_j + n)
#     x_list = []
#     y_list = []
#     # 检查是否为边界点
#     for i in range(i_min, i_max + 1):
#         for j in range(j_min, j_max + 1):
#             if not Si_array[i, j].existflag:
#                 continue
#             is_boundary = False
#             if (i > i_min and not Si_array[i - 1, j].existflag) or \
#                     (j > j_min and not Si_array[i, j - 1].existflag) or \
#                     (i < i_max and not Si_array[i + 1, j].existflag) or \
#                     (j < j_max and not Si_array[i, j + 1].existflag):
#                 is_boundary = True
#             if is_boundary:
#                 x_list = np.append(x_list, i + 0.5)
#                 y_list = np.append(y_list, j + 0.5)
#     x_list = np.array(x_list)
#     y_list = np.array(y_list)
#     # 两个点以上才能生成反射面
#     if len(x_list) < 2:
#         return 0, np.array([0, 1])  # 默认斜率为0，法线垂直向上
#     # (A.T @ A) @ theta =  A.T @ y_list.T 公式表示
#     #求出k和法向量单位向量  theta = [a, b]
#     A = np.vstack((np.ones_like(x_list), x_list)).T
#     theta = np.linalg.lstsq(A, y_list, rcond=None)[0]  # 使用lstsq更稳定
#     k = theta[1]
#     n = np.array([-k ,1])
#     n_normal = n / np.linalg.norm(n)
#     return k, n_normal



# 生成反射角度
def reflect_angle(Si_array, px, py, k, reflext_prob, direction = 1, left_border = 350):
    is_reflect_flag = 0
    reflect_k = k
    V_out = np.array([0, 0])

    if Si_array[px, py].existflag == True:
        incident_angle = math.atan(abs(k))
        if incident_angle > math.radians(10):
            if random.random() < reflext_prob:
                # 计算入射向量,k是斜率
                # direction 是方向
                if abs(k) < 1e-9:  # 避免除零
                    # 近似水平运动
                    V_in = np.array([direction * float('inf'), direction])
                else:
                    V_in = np.array([direction / k, direction])

                V_in = V_in / np.linalg.norm(V_in)

                # 获取法线向量和反射向量- 添加安全检查
                result = reflector_face(Si_array, px, py, n=4)
                if result is None or result[1] is None:
                    # 无法获取法线，使用默认垂直法线
                    if px <= left_border:
                        N = np.array([1, 0]) # 拟合失败则返回默认值
                    else:
                        N = np.array([-1, 0]) # 拟合失败则返回默认值
                else:
                    _, N = result

                # 计算反射向量
                V_out = V_in - 2 * (np.dot(V_in, N)) * N

                # 计算反射斜率
                if abs(V_out[0]) < 1e-10:  # 避免除零错误
                    reflect_k = float('inf')  # 垂直运动
                else:
                    reflect_k = V_out[1] / V_out[0]
                    # print(V_out[1],V_out[0])

                # 引入文献中的 cos^n θ 分布：用高斯随机扰动模拟反射方向的分散性
                # 文献：Analytical modeling of silicon etch process in high density plasma
                # sigma_degrees = 4     #主要分布±5，10 els hardly equal zero
                # sigma_degrees = 15    #主要分布±10，20 els hardly equal zero
                sigma_degrees = 17
                # sigma_degrees = 20    #主要分布±20，40 els hardly equal zero
                sigma = np.radians(sigma_degrees)   # 这个sigma值对应分布宽度，相当于文献中的 `n`
                angle_increment = abs(random.gauss(0, sigma)) # 生成一个随机偏转角
                # deviation_angle = math.pi/2 *(5/90)
                # # 将偏转角加到由 reflect_k 计算出的角度上，得到最终方向
                # new_angle = math.atan(reflect_k) + deviation_angle
                # reflect_k = math.tan(new_angle)
                # 从V_out计算当前反射角（由物理反射定律计算出的反射角）
                current_reflect_angle = math.atan2(V_out[1], V_out[0])  # 反射角，范围[-π, π]
                
                # 固定反射角偏转角度
                # angle_increment = math.radians(5)  # θ角度转弧度
                
                # 根据象限决定角度调整方向和边界约束
                if current_reflect_angle > 0 and current_reflect_angle < math.pi / 2:
                    # 第一象限(0, 90°): 增大角度
                    new_reflect_angle = current_reflect_angle + angle_increment
                    # 不能越过象限边界
                    if new_reflect_angle >= math.pi / 2:
                        new_reflect_angle = math.pi / 2 - 1e-9
                        
                elif current_reflect_angle > math.pi / 2 and current_reflect_angle <= math.pi:
                    # 第二象限(90°, 180°): 减小角度
                    new_reflect_angle = current_reflect_angle - angle_increment
                    # 不能越过象限边界
                    if new_reflect_angle <= math.pi / 2:
                        new_reflect_angle = math.pi / 2 + 1e-9
                        
                elif current_reflect_angle < -math.pi / 2 and current_reflect_angle >= -math.pi:
                    # 第三象限(-180°, -90°): 增大角度
                    new_reflect_angle = current_reflect_angle + angle_increment
                    # 如果偏移后大于-90°，设置为-90°（垂直），k=999
                    if new_reflect_angle > -math.pi / 2:
                        new_reflect_angle = -math.pi / 2
                        
                elif current_reflect_angle > -math.pi / 2 and current_reflect_angle < 0:
                    # 第四象限(-90°, 0): 减小角度
                    new_reflect_angle = current_reflect_angle - angle_increment
                    # 如果偏移后小于-90°，设置为-90°（垂直），k=999
                    if new_reflect_angle < -math.pi / 2:
                        new_reflect_angle = -math.pi / 2
                else:
                    # 边界情况（0°, 90°, -90°, 180°），保持原角度
                    new_reflect_angle = current_reflect_angle
                
                # 从新反射角计算反射斜率
                if abs(abs(new_reflect_angle) - math.pi/2) < 1e-9:
                    # 垂直运动，k=999表示接近垂直的大斜率
                    reflect_k = 999.0
                else:
                    reflect_k = math.tan(new_reflect_angle)
                    
                # 更新状态
                is_reflect_flag = 1
                
                return is_reflect_flag, reflect_k, V_out

        return is_reflect_flag, reflect_k, V_out
    else:
        return is_reflect_flag, reflect_k, V_out

#撞击函数
def collisionprocess(Si_array, px, py, species, reaction_probabilities, abs_angle, s_image):
    if not  Si_array[px, py].existflag:
        return False
    #获得粒子类型和俘获活性种数目，确定反应概率；计算Ysicl
    if species == 1:
        particle_type = 'Cl+'
        Ysicl = calculate_Ysicl(abs_angle)
    else:
        particle_type = 'Cl*'
        Ysicl = 0.0  # 活性种不计算Yield

    Count = Si_array[px, py].CountCl
    if Si_array[px, py].material_type == 'Si':
        prob = reaction_probabilities[Count][particle_type]
    elif Si_array[px, py].material_type == 'Hardmask':
        prob = reaction_probabilities[Count][particle_type] * 0.1

    #4+1的逻辑
    # if Si_array[px, py].CountCl == 4 and species == 1:#和活性种复合过四次且被氯离子冲击过
        # Si_array[px, py].existflag = False
    clearflag = False
    if species == 1 and random.random() < prob * Ysicl:
        clearflag = True
    elif species == 0 and random.random() < prob:
        clearflag = True

    # 处理链式反应（Ysicl > 1时）
    if Ysicl > 1.0 and clearflag:
        chain_reaction(Si_array, px, py, Ysicl, s_image)

    # 自反应，化学反应
    if species == 0 and Si_array[px, py].CountCl == 4:
        Ychem = calculate_Ychem()
        if random.random() < Ychem:
            clearflag = True
    # 活性粒子撞击且复合数目小于四
    if not species and Si_array[px, py].CountCl < 4:
        Si_array[px, py].CountCl +=1


    Si_array[px, py].existflag = not clearflag
    return clearflag


# 加速函数：使用大步长+回退+小步长策略快速到达材料表面附近
def find_nearest_material_fast(emission_x, emission_y, emission_k, px, py, Si_array, rows, cols, s_image, direction=1):
    """
    使用大步长跳跃快速逼近材料表面，检测到碰撞后回退，最后用小步长逐行精确定位
    参考return_next的运动逻辑，支持向上/向下运动
    
    参数：
        emission_x, emission_y: 粒子发射点坐标
        emission_k: 粒子运动斜率
        px, py: 当前粒子位置
        Si_array: 材料数组
        rows, cols: 网格大小
        direction: 运动方向，1表示向下，-1表示向上
    
    返回：
        [px, py] 如果找到材料，或 None 如果未找到
    """
    big_step = 5  # 大步长大小
    rows, cols = Si_array.shape
    current_px, current_py = px, py
    
    # 第一阶段：粗查 - 用大步长快速跳跃，参考return_next的运动逻辑
    while True:
        # 根据direction计算下一个检测点（大步长）
        if direction >= 0:  # 向下运动
            next_py = min(current_py + big_step, cols - 1)
        else:  # 向上运动
            next_py = max(current_py - big_step, 0)
        
        # 参考return_next逻辑计算px变化（未反射情况，直接向下/上）
        if direction >= 0:  # 向下运动
            # 类似return_next中is_reflect=False, direction>=0的情况
            if emission_k > 0:
                # 主要向右下，需要根据轨迹判断
                next_px = current_px + big_step
            elif emission_k < 0:
                # 主要向左下
                next_px = current_px - big_step
            else:
                # 垂直向下
                next_px = current_px
        else:  # 向上运动
            # 类似return_next中direction<0的情况
            if emission_k > 0:
                y_test = emission_y + emission_k * ((current_px - 0.5) - emission_x)
                if y_test <= next_py - 0.5:
                    next_px = current_px  # 垂直向上
                else:
                    next_px = current_px - big_step  # 向左上
            elif emission_k < 0:
                y_test = emission_y + emission_k * ((current_px + 0.5) - emission_x)
                if y_test <= next_py - 0.5:
                    next_px = current_px  # 垂直向上
                else:
                    next_px = current_px + big_step  # 向右上
            else:
                next_px = current_px  # 垂直向上
        
        # 边界检查
        if not (0 <= next_px < rows and 0 <= next_py < cols):
            # 超出边界，返回 None
            return None
        
        # 检查这个点是否有材料
        if Si_array[next_px, next_py].existflag:
            # 检测到材料！进行回退和精查
            break

        # 画线
        if 0 < next_px < rows and 0 < next_py < cols:
            s_image[next_px, next_py] = 60

        current_px, current_py = next_px, next_py
        
        # 检查是否到达边界
        if (direction >= 0 and current_py >= cols - 1) or \
           (direction < 0 and current_py <= 0):
            return None
    
    # 第二阶段：回退到前一个安全位置
    if direction >= 0:
        current_py = max(current_py - big_step, py)
    else:
        current_py = min(current_py + big_step, py)
    
    # 重新计算current_px（保持逻辑一致）
    if direction >= 0:
        if emission_k > 0:
            current_px = px - big_step
        elif emission_k < 0:
            current_px = px + big_step
        else:
            current_px = px
    else:
        if emission_k > 0:
                current_px = px + big_step
        elif emission_k < 0:
                current_px = px - big_step
        else:
            current_px = px

    # # 画线
    # if 0 < next_px < rows and 0 < next_py < cols:
    #     s_image[next_px, next_py] = 80
    
    # 第三阶段：精查 - 用return_next进行精确逐步移动
    while 0 <= current_px < rows and 0 <= current_py < cols:
        # 检查是否有材料
        if Si_array[current_px, current_py].existflag:
            # 找到第一个有材料的点
            return [current_px, current_py]
        
        # 使用return_next进行精确移动（一步一步）
        next_pos = return_next(emission_x, emission_y, emission_k, current_px, current_py, direction)
        current_px, current_py = next_pos
    # 画线
    if 0 < current_px < rows and 0 < current_py < cols:
        s_image[current_px, current_py] = 80
        
    # 未找到材料
    return None


# 最简单的实现方式：
# def simple_angle_emission():
#     """最简单的角度生成方案"""
#     sigma = 0.0704  # 对于R=100
#     angle_rad = random.gauss(0, sigma)
#     emission_k = 1.0 / math.tan(angle_rad)
#     return emission_k


#一些用以处理图像的函数
# 基本样条插值函数

def main():
    # 数据层面上初始化仿真界面
    vacuum = 50
    rows = 700
    cols = 1000
    left_border = 300
    right_border = 400
    deep_border = 300
    start_time = time.perf_counter()
    # 掩膜角度fa
    angle_img = abs(0)
    Si_array = np.empty(shape=(rows,cols), dtype=object)
    # 数据初始化整合到下面的图形初始化里面了，一块初始化
    for i in range(rows):
        for j in range(vacuum):
            Si_array[i, j] = Si_Class(existflag = False)
        for j in range(vacuum, cols):
            Si_array[i, j] = Si_Class()

    # 将上面一半的si原子去除，不清除，下面一起清除
    # for i in range(rows):
    #     for j in range(deep_border):
    #         Si_array[i, j].existflag = False
    # 在图像和数据层面初始化界面
    # angle_img = abs(math.asin(2 * random.random() - 1))
    # angle_img = abs(math.asin(0))
    k_img = abs(math.tan(angle_img))
    # 入射开口限幅
    if deep_border * k_img > rows / 6:
        k_img = (rows - right_border - 1) / (deep_border * 3)
    #初始化图像数组
    s_image = np.ones((rows, cols))
    # 初始化真空界面
    for y in range(vacuum):
        for x in range(rows):
            if Si_array[x, y].existflag == False:
                s_image[x, y] = 25  # 真空
    # 初始化掩膜界面
    for y in range(vacuum, deep_border):
        offset = int((deep_border - y) * k_img) # 偏移量
        left_current = left_border - offset
        right_current = right_border + offset
        # 确保不出界
        left_current = max(0, min(left_current, rows - 1))
        right_current = max(0, min(right_current, rows - 1))
        # 遍历当前y行的所有x坐标
        for x in range(rows):
            if  left_current < x < right_current:
                s_image[x, y] = 25  # 真空
                Si_array[x, y].existflag = False
            else:
                s_image[x, y] = 40  # 光刻胶
                Si_array[x, y].material_type = 'Hardmask'
                Si_array[x, y].existflag = True
    # 创建初始图形
    fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(111)

    # 顺序三：撞击时的反应概率
    reaction_probabilities = {
        0: {'Cl*': 0.0, 'Cl+': 0.1},  # 未俘获Cl
        1: {'Cl*': 0.1, 'Cl+': 0.3},  # 俘获1个Cl
        2: {'Cl*': 0.2, 'Cl+': 0.3},  # 俘获2个Cl
        3: {'Cl*': 0.3, 'Cl+': 0.3},  # 俘获3个Cl
        4: {'Cl*': 1.0, 'Cl+': 0.3}  # 俘获4个Cl
    }

    #模拟粒子入射
    for cl in range(10):
        # 考虑openCD对形貌影响
        #粒子初始位置
        # emission_x = left_border + random.random() * (right_border - left_border)
        emission_y_Ion = 1
        emission_y_neutral = 1
        # emission_y_neutral = vacuum - 1
        
        # 测试入射角度45度的时候改了一下入射范围
        emission_x = left_border + random.random() * (right_border - left_border) / 2
        species = random.random() > (10/11)
        # emission_theta = (random.random()-0.5) * math.pi
        # emission_k = np.tan(emission_theta)
        #一种正态分布

                
        # 粒子入射概率判定
        if species == 1:
            # 离子入射概率
            # 方差 角度分布大部分在（-5,5）几乎所有都分布在（-10,10）
            sigma_degrees = 4
            sigma = np.radians(sigma_degrees)  # 对于R=100
            angle_rad = random.gauss(0, sigma)
            angle_rad = max(min(angle_rad, math.pi/2), -math.pi/2)
            # 测试45度入射角
            # angle_rad = -math.pi / 4
            abs_angle = abs(angle_rad)
            # 测试入射角度范围，下面同理
            # print("离子")
            # print(angle_rad)
            emission_k = 1.0 / math.tan(angle_rad)
            emission_y = emission_y_Ion
        else:
            #中性粒子入射概率
            # angle_rad = math.asin(random.random() * 2 - 1)
            angle_rad = math.pi / 4
            abs_angle = abs(angle_rad)
            # print("中性")
            # print(angle_rad)
            emission_k =  1.0 / math.tan(angle_rad)
            emission_y =  emission_y_neutral

        abs_k = np.abs(emission_k)


        #粒子初始角度确认，处理不同斜率情况
        if abs_k <= 0.1:#近似水平
            continue
        elif abs_k >= 200:#近似垂直
            px = math.ceil(emission_x)
            for py in range(deep_border + 1, cols):
                if 0 <= px < rows and Si_array[px, py].existflag:
                    clearflag = collisionprocess(Si_array, px, py, species, reaction_probabilities, abs_angle, s_image) # 碰撞函数
                    if clearflag:
                         s_image[px, py] = 25  #真空
                    break
        else:
            # 判定是否超出入射界限，后面需要改一下
            # x_intersect = (deep_border + 0.5 - emission_y)/emission_k + emission_x
            # if x_intersect < left_border or x_intersect > right_border:
            #     continue

            # x_intersect = (deep_border + 0.5 - emission_y) / emission_k + emission_x
            # px = int(x_intersect)
            # py = deep_border
            px = int(emission_x)
            py = emission_y
                
            
        #运动轨迹追踪
        max_steps = 4000  # 防止无限循环
        ref_count = 0
        count = 2
        particle_direction = 1 # 初始方向向下
        for step in range(max_steps):
            # next_pos  = return_next(emission_x, 1, emission_k, px, py)
            # px, py = next_pos
            if not (0 <= px < rows  and 0 <= py < cols):
                break
            if Si_array[px, py].existflag:
                # 检查反射
                ref_prob = reflect_prob(abs_angle, Si_array[px, py].material_type)
                is_reflect, new_k, V_out= reflect_angle(Si_array, px, py, emission_k, ref_prob, particle_direction)
                #反射次数计数
                if is_reflect and ref_count < count:
                    # print(f"反射次数: {ref_count + 1}, 最大允许次数: {count}")
                    ref_count = ref_count + 1
                    
                    # 重新初始化入射点（根据反射向量的y分量）
                    particle_direction = 1 if V_out[1] >= 0 else -1

                    # 更新粒子状态
                    emission_k = new_k
                    emission_x, emission_y = px, py  # 从反射点继续运动

                    # 显示当前状态
                    # ax.clear()
                    # rotated_image = np.rot90(s_image, -1)
                    # ax.imshow(rotated_image, cmap='jet', vmin=0, vmax=100)
                    # plt.pause(0.1)  # 暂停一小段时间以便观察

                    # 使用正确的方向参数调用return_next
                    next_pos = return_next(emission_x, emission_y, emission_k, px, py, particle_direction)
                    px, py = next_pos
                    # 标记粒子轨迹,画线
                    # if px < rows and py < cols:
                    #     s_image[px, py] = 60
                    continue  # 继续外部循环
                    # for step in range(max_steps):
                    #     next_pos = return_next(emission_x, emission_y, new_k, px, py, V_out[1])
                    #     px, py = next_pos
                    #     # 边界约束
                    #     if not (0 <= px < rows and 0 <= py < cols):
                    #         break
                    #     if Si_array[px, py].existflag == 1:
                    #             clearflag = collisionprocess(Si_array, px, py, species, reaction_probabilities, abs_angle,s_image)  # 碰撞函数
                    #             if clearflag:
                    #                 s_image[px, py] = 25  #真空
                    #     elif Si_array[px, py].existflag == 0:
                    #         break  # 跳出循环
                    # break
                else:  # 处理碰撞反应
                    clearflag = collisionprocess(Si_array, px, py, species, reaction_probabilities, abs_angle, s_image) # 碰撞函数
                    if clearflag:
                        s_image[px, py] = 25
                    break
            else:
                # 使用加速函数快速到达材料表面附近，传入direction参数
                result = find_nearest_material_fast(emission_x, emission_y, emission_k, px, py, Si_array, rows, cols, s_image, direction=particle_direction)
                if result is not None:
                    px, py = result
                else:
                    # 如果未找到材料，粒子逃逸
                    break




    # 提取轮廓线：从上往下扫描,再从左往右扫描
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

    # 绘制轮廓点
    if transformed_points:
        points_array = np.array(transformed_points)
        plt.plot(points_array[:, 0], points_array[:, 1], 'ro', markersize=1, alpha=0.6, label='point')

    # 添加图例
    plt.legend(loc='upper right')
    # 顺时针旋转90度（符合常规视角：x→横向，y→纵向）
    rotated_image = np.rot90(s_image, -1)
    plt.imshow(rotated_image, cmap='jet', vmin=0, vmax=100)



    plt.axis('equal')
    plt.axis('on')  # 隐藏坐标轴
    plt.tight_layout()
    plt.show()
    end_time = time.perf_counter()
    print(f"程序总运行时间: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()