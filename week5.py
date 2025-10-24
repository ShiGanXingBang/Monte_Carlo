import matplotlib
import numpy as np
import cv2

matplotlib.use('TkAgg')  # 或者 'Qt5Agg', 'Agg' 等
import matplotlib.pyplot as plt
import random
import math

# Si_class = {'existflag': True, 'CountCl': 0} #字典
#定义硅原子类属性
class Si_Class:
    def __init__(self, existflag = True, CountCl = 0, material_type = 'Si',reflect_count = 0):
        self.existflag = existflag
        self.CountCl = CountCl
        self.material_type = material_type # 'Si' 'Hardmask'
        self.reflect_count = reflect_count

#运动方向判断
def return_next(emission_x, emission_y, emission_k, px, py, s_image, direction=1):
    # 向下方运动
    if direction >= 0:
        #向右下运动
        if emission_k > 0:
            y = emission_y + emission_k * ((px + 0.5) - emission_x)
            #向右运动
            if y <= py +0.5:
                return [px + 1, py]
            else:
                return [px, py + 1]
        #向左下运动
        elif emission_k < 0:
            y = emission_y + emission_k * ((px - 0.5) - emission_x)
            if y <= py + 0.5:
                return [px - 1, py]
            else:
                return [px, py + 1]
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
    elif material == 'Si':
        base_prob = 0.2
        angle_else = min(0.5, 0.5 * (theta / math.pi) / 2)
        return base_prob + angle_else
    else:
        return 0.0

# 生成反射面
def reflector_face(Si_array, center_i, center_j, n=4):
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
            if (i > i_min and not Si_array[i - 1, j].existflag) or \
                    (j > j_min and not Si_array[i, j - 1].existflag) or \
                    (i < i_max and not Si_array[i + 1, j].existflag) or \
                    (j < j_max and not Si_array[i, j + 1].existflag):
                is_boundary = True
            if is_boundary:
                x_list = np.append(x_list, i + 0.5)
                y_list = np.append(y_list, j + 0.5)
    x_list = np.array(x_list)
    y_list = np.array(y_list)
    # 两个点以上才能生成反射面
    if len(x_list) < 2:
        return 0, np.array([0, 1])  # 默认斜率为0，法线垂直向上
    # (A.T @ A) @ theta =  A.T @ y_list.T 公式表示
    #求出k和法向量单位向量  theta = [a, b]
    A = np.vstack((np.ones_like(x_list), x_list)).T
    theta = np.linalg.lstsq(A, y_list, rcond=None)[0]  # 使用lstsq更稳定
    k = theta[1]
    n = np.array([-k ,1])
    n_normal = n / np.linalg.norm(n)
    return k, n_normal



# 生成反射角度
def reflect_angle(Si_array, px, py, k, reflext_prob, direction = 1):
    is_reflect_flag = 0
    reflect_k = k
    V_out = np.array([0, 0])

    if Si_array[px, py].existflag == True:
        count = 1 # 反射次数,只反射一次
        if Si_array[px, py].reflect_count < count:
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
                        N = np.array([0, 1])
                    else:
                        k_val, N = result

                    # 计算反射向量
                    V_out = V_in - 2 * (np.dot(V_in, N)) * N

                    # 计算反射斜率
                    if abs(V_out[0]) < 1e-10:  # 避免除零错误
                        reflect_k = float('inf')  # 垂直运动
                    else:
                        reflect_k = V_out[1] / V_out[0]

                    Si_array[px, py].reflect_count += 1
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

# 最简单的实现方式：
# def simple_angle_emission():
#     """最简单的角度生成方案"""
#     sigma = 0.0704  # 对于R=100
#     angle_rad = random.gauss(0, sigma)
#     emission_k = 1.0 / math.tan(angle_rad)
#     return emission_k

def main():
    # 数据层面上初始化仿真界面
    vacuum = 50
    rows = 700
    cols = 1000
    left_border = 300
    right_border = 400
    deep_border = 300
    angle_img = abs(85)
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


    # 顺序三：撞击时的反应概率
    reaction_probabilities = {
        0: {'Cl*': 0.0, 'Cl+': 0.1},  # 未俘获Cl
        1: {'Cl*': 0.1, 'Cl+': 0.3},  # 俘获1个Cl
        2: {'Cl*': 0.2, 'Cl+': 0.3},  # 俘获2个Cl
        3: {'Cl*': 0.3, 'Cl+': 0.3},  # 俘获3个Cl
        4: {'Cl*': 1.0, 'Cl+': 0.3}  # 俘获4个Cl
    }

    #模拟粒子入射
    for cl in range(20000):
        # 考虑openCD对形貌影响
        emission_x = left_border + random.random() * (right_border - left_border)
        species = random.random() > (10/11)
        # emission_theta = (random.random()-0.5) * math.pi
        # emission_k = np.tan(emission_theta)
        #一种正态分布

        # 粒子入射概率判定
        if species == 1:
            # 离子入射概率
            sigma = 0.0704  # 对于R=100
            angle_rad = random.gauss(0, sigma)
            abs_angle = abs(angle_rad)
            emission_k = 1.0 / math.tan(angle_rad)
        else:
            #中性粒子入射概率
            angle_rad = math.asin(random.random() * 2 - 1)
            abs_angle = abs(angle_rad)
            emission_k =  1.0 / math.tan(angle_rad)
        abs_k = np.abs(emission_k)

        #粒子初始位置
        emission_y = 1

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
        max_steps = 2000  # 防止无限循环
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

                if is_reflect:
                    # 这里有问题，明天改。重新初始化入射点（根据反射向量的y分量）
                    particle_direction = 1 if V_out[1] >= 0 else -1

                    # 更新粒子状态
                    emission_k = new_k
                    emission_x, emission_y = px, py  # 从反射点继续运动

                    # 使用正确的方向参数调用return_next
                    next_pos = return_next(emission_x, emission_y, emission_k, px, py, s_image, particle_direction)
                    px, py = next_pos
                    # 标记粒子轨迹
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
                next_pos = return_next(emission_x, emission_y, emission_k, px, py, s_image, particle_direction)
                px, py = next_pos
                # 标记粒子轨迹
                # if px < rows and py < cols:
                #     s_image[px, py] = 60


    plt.figure(figsize=(12, 8))

    # 在plt.show()之前添加以下代码

    # 提取轮廓线：从上往下扫描
    contour_points = []
    for y in range(cols):  # 遍历每一列
        for x in range(rows - 1):  # 从上到下扫描
            if Si_array[x, y].existflag != Si_array[x + 1, y].existflag:
                contour_points.append((x + 0.5, y))  # 记录边界点

    # 坐标转换
    transformed_points = []
    for x, y in contour_points:
        new_x = y
        new_y = rows - 1 - x
        transformed_points.append((new_x, new_y))

    if len(transformed_points) > 1:
        # 按x坐标排序以便连接
        sorted_points = sorted(transformed_points, key=lambda p: p[0])
        sorted_array = np.array(sorted_points)
        plt.plot(sorted_array[:, 0], sorted_array[:, 1], 'c-', linewidth=1, alpha=0.3, label='轮廓线')

    # 绘制轮廓点
    if transformed_points:
        points_array = np.array(transformed_points)
        plt.plot(points_array[:, 0], points_array[:, 1], 'ro', markersize=1, alpha=0.6, label='轮廓点')

    # 添加图例
    plt.legend(loc='upper right')

    # 顺时针旋转90度（符合常规视角：x→横向，y→纵向）
    rotated_image = np.rot90(s_image, -1)
    plt.imshow(rotated_image, cmap='jet', vmin=0, vmax=100)



    plt.axis('equal')
    plt.axis('on')  # 隐藏坐标轴
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()