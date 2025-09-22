import numpy as np
import matplotlib.pyplot as plt
import random

class Si_atom():
    def __init__(self, existflag = True, Countcl = 0):
        self.existflag = existflag
        self.Countcl = Countcl

def collisionprocess(si_array, px, py):
    if not  si_array[px, py].existflag:
        return
    clearflag =False
    if si_array[px, py].Countcl == 4:
        clearflag = True
    else:
        si_array[px, py].Countcl += 1

    si_array[px, py].existflag = not clearflag
    return clearflag

def main():
    si_array = np.empty([70,100], dtype = object)
    for i in range(70):
        for j in range(100):
            si_array[i,j] = Si_atom()

    for i in range(0, 70):
        for j in range(0, 30):
            si_array[i,j].existflag = False

    s_image = np.ones((70,100))
    s_image[0:30, 0:30] = 40  #光刻胶
    s_image[30:40, 0:30] = 25 #真空
    s_image[40:70, 0:30] = 40 #光刻胶
    #粒子入射角度初始化
    for cl in range(1000):
        emission_x = 30 + random.random() * 10
        px = int(emission_x)
        for py in range(31 ,100):
            if si_array[px, py].existflag :
                clearflag = collisionprocess(si_array, px, py)
                if clearflag == True:
                    s_image[px, py] = 25
                break

    plt.imshow(np.rot90(s_image, -1), cmap='jet')
    plt.colorbar()
    plt.title("刻蚀效果图")
    plt.show()

if __name__ == '__main__':
    main()