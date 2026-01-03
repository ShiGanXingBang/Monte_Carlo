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
def collisionprocess(Si_array, px, py ,species):
    if not  Si_array[px, py].existflag:
        return False

    clearflag = False
    #4+1的逻辑
    if Si_array[px, py].ConuCl == 4 and species == 1:#和活性种复合过四次且被氯离子冲击过
        # Si_array[px, py].existflag = False
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
    rows = 700
    cols = 1000
    left_border = 300
    right_border = 400
    Si_array = np.empty(shape=(rows,cols), dtype = object)
    for i in range(rows):
        for j in range(cols):
            Si_array[i, j] = Si_Class()
    # 将上面一半的si原子去除
    for i in range(rows):
        for j in range(left_border):
            Si_array[i, j].existflag = False


    # 在图像和数据层面初始化界面
    s_image = np.ones((rows,cols))
    s_image[0:left_border, 0:left_border] = 40  #光刻胶
    s_image[left_border + 1:right_border, 0:left_border] = 25 #真空
    s_image[right_border + 1:rows, 0:left_border] = 40 #光刻胶
    #模拟粒子入射
    for cl in range(200000):
        emission_x = left_border + random.random() * 100
        species = random.random() > (10/11)
        # emission_theta = (random.random()-0.5) * math.pi
        # emission_k = np.tan(emission_theta)
        emission_k = simple_angle_emission()
        abs_k = np.abs(emission_k)
        #粒子初始位置
        emission_y = 1

        #粒子初始角度确认，处理不同斜率情况
        if abs_k <= 0.1:#近似水平
            continue
        elif abs_k >= 200:#近似垂直
            px = math.ceil(emission_x)
            for py in range(left_border + 1, cols):
                if 0 <= px < rows and Si_array[px, py].existflag:
                    clearflag = collisionprocess(Si_array, px, py, species) # 碰撞函数
                    if clearflag:
                        s_image[px, py] = 25
                    break
        else:
            x_intersect = (left_border + 0.5 - emission_y)/emission_k + emission_x
            if x_intersect < left_border or x_intersect > right_border:
                continue

            px = int(x_intersect)
            py = left_border + 1

        #运动轨迹追踪
        max_steps = 1500  # 防止无限循环
        for step in range(max_steps):
            # next_pos  = return_next(emission_x, 1, emission_k, px, py)
            # px, py = next_pos
            if not (0 <= px < rows  and 0 <= py < cols):
                break

            if Si_array[px, py].existflag:
                clearflag = collisionprocess(Si_array, px, py, species) # 碰撞函数
                if clearflag:
                    s_image[px, py] = 25
                break
            else:
                next_pos = return_next(emission_x, emission_y, emission_k, px, py)
                px, py = next_pos



    plt.figure(figsize=(12, 8))
    # 顺时针旋转90度（符合常规视角：x→横向，y→纵向）
    rotated_image = np.rot90(s_image, -1)
    plt.imshow(rotated_image, cmap='jet', vmin=0, vmax=100)
    # plt.colorbar(label='表面状态（1=Si衬底，25=真空，40=光刻胶）')
    # plt.title('Simulation of silicon wafer etching effect')

    # 添加标注（根据旋转后的坐标）
    # 左边掩膜（旋转后：x=0-299，y=400-699）
    plt.text(500, 150, 'MASK', fontsize=14, color='white', fontweight='bold')
    # 右边掩膜（旋转后：x=0-299，y=0-298）
    plt.text(100, 150, 'MASK', fontsize=14, color='white', fontweight='bold')
    # 衬底（旋转后：x≥300，y=0-699）
    plt.text(100, 400, 'Substrate', fontsize=14, color='yellow', fontweight='bold')

    plt.axis('equal')
    plt.axis('off')  # 隐藏坐标轴
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()