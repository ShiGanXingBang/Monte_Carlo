import numpy as np
import matplotlib.pyplot as plt
import random

class Si_atom():
    def __init__(self, existflag = True, Countcl = 0):
        self.existflag = existflag
        self.Countcl = Countcl

def collisionprocess(si_array, px, py):
    if not  si_array[px, py].existflag:
        return False
    clearflag =False
    if si_array[px, py].Countcl == 4:
        clearflag = True
    else:
        si_array[px, py].Countcl += 1

    si_array[px, py].existflag = not clearflag
    return clearflag

def main():
    # 数据层面上初始化仿真界面
    rows = 700
    cols = 1000
    left_border = 300
    right_border = 400
    si_array = np.empty(shape=(rows,cols), dtype = object)
    for i in range(rows):
        for j in range(cols):
            si_array[i,j] = Si_atom()

    for i in range(0, rows):
        for j in range(0, left_border):
            si_array[i,j].existflag = False

    s_image = np.ones((rows,cols))
    s_image[0:left_border, 0:left_border] = 40  #光刻胶
    s_image[left_border + 1:right_border, 0:left_border] = 25 #真空
    s_image[right_border + 1:rows, 0:left_border] = 40 #光刻胶
    #粒子入射角度初始化
    for cl in range(200000):
        emission_x = left_border + random.random() * 100
        px = int(emission_x)
        for py in range(left_border + 1, cols):
            if si_array[px, py].existflag :
                clearflag = collisionprocess(si_array, px, py)
                if clearflag == True:
                    s_image[px, py] = 25
                break

    # plt.imshow(np.rot90(s_image, -1), cmap='jet')
    # plt.colorbar()
    # plt.title("刻蚀效果图")
    # plt.show()
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

if __name__ == '__main__':
    main()