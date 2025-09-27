import numpy as np
import matplotlib.pyplot as plt
import random
import math

# Si_class = {'existflag': True, 'ConuCl': 0} #字典
#定义硅原子类属性
class Si_Class:
    def __init__(self, existflag = True, ConuCl = 0):
        self.existflag = existflag
        self.ConuCl = ConuCl

#运动方向判断
def return_next(emission_x, emission_y, emission_k, px, py):
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


#撞击函数
def collisionprocess(Si_array, px, py, species, reaction_probabilities):
    if not  Si_array[px, py].existflag:
        return False
    #获得粒子类型和俘获活性种数目，确定反应概率
    if species == 1:
        particle_type = 'Cl+'
    else:
        particle_type = 'Cl*'
    clearflag = False
    Count = Si_array[px, py].ConuCl
    prob = reaction_probabilities[Count][particle_type]

    #4+1的逻辑
    # if Si_array[px, py].ConuCl == 4 and species == 1:#和活性种复合过四次且被氯离子冲击过
        # Si_array[px, py].existflag = False
    if random.random() < prob:
        clearflag = True
    elif not species and Si_array[px, py].ConuCl < 4:
        Si_array[px, py].ConuCl +=1

    Si_array[px, py].existflag = not clearflag
    return clearflag

# 最简单的实现方式：
def simple_angle_emission():
    """最简单的角度生成方案"""
    sigma = 0.0704  # 对于R=100
    angle_rad = random.gauss(0, sigma)
    emission_k = 1.0 / math.tan(angle_rad)
    return emission_k

def main():
    # 数据层面上初始化仿真界面
    Si_array = np.empty(shape=(70,100), dtype=object)
    for i in range(70):
        for j in range(100):
            Si_array[i, j] = Si_Class()
    # 将上面一半的si原子去除
    for i in range(70):
        for j in range(30):
            Si_array[i, j].existflag = False


    # 在图像和数据层面初始化界面
    s_image = np.ones((70, 100))
    s_image[0:30, 0:30] = 40
    s_image[30:40, 0:30] = 25
    s_image[40:70, 0:30] = 40
    # 顺序三：撞击时的反应概率
    reaction_probabilities = {
        0: {'Cl*': 0.0, 'Cl+': 0.1},  # 未俘获Cl
        1: {'Cl*': 0.1, 'Cl+': 0.3},  # 俘获1个Cl
        2: {'Cl*': 0.2, 'Cl+': 0.3},  # 俘获2个Cl
        3: {'Cl*': 0.3, 'Cl+': 0.3},  # 俘获3个Cl
        4: {'Cl*': 1.0, 'Cl+': 0.3}  # 俘获4个Cl
    }

    #模拟粒子入射
    for cl in range(3000):
        emission_x = 30 + random.random() * 10
        species = random.random() > (10/11)
        # emission_theta = (random.random()-0.5) * math.pi
        # emission_k = np.tan(emission_theta)
        emission_k = simple_angle_emission()
        abs_k = np.abs(emission_k)
        #粒子初始位置
        emission_y = 1
        rows, cols = 70, 100

        #粒子初始角度确认，处理不同斜率情况
        if abs_k <= 0.1:#近似水平
            continue
        elif abs_k >= 200:#近似垂直
            px = math.ceil(emission_x)
            for py in range(31, 100):
                if 0 <= px < rows and Si_array[px, py].existflag:
                    clearflag = collisionprocess(Si_array, px, py, species, reaction_probabilities) # 碰撞函数
                    if clearflag:
                        s_image[px, py] = 25
                    break
        else:
            x_intersect = (30.5 - emission_y)/emission_k + emission_x
            if x_intersect < 30 or x_intersect > 40:
                continue

            px = int(x_intersect)
            py = 31

        #运动轨迹追踪
        max_steps = 1000  # 防止无限循环
        for step in range(max_steps):
            # next_pos  = return_next(emission_x, 1, emission_k, px, py)
            # px, py = next_pos
            if not (0 <= px < rows  and 0 <= py < cols):
                break

            if Si_array[px, py].existflag:
                clearflag = collisionprocess(Si_array, px, py, species, reaction_probabilities) # 碰撞函数
                if clearflag:
                    s_image[px, py] = 25
                break
            else:
                next_pos = return_next(emission_x, emission_y, emission_k, px, py)
                px, py = next_pos



    plt.figure(figsize=(10, 8))
    plt.imshow(np.rot90(s_image, -1), cmap='jet', vmin=0, vmax=100)
    plt.colorbar(label='表面状态')
    plt.title('刻蚀效果模拟')
    plt.axis('equal')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()