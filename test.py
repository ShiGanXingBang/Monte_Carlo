import numpy as np
import matplotlib.pyplot as plt
import math
import random

for i in range(100000):
    angle_rad = math.cos((random.random() - 0.5) * math.pi) / 2
    angle_abs = abs(angle_rad)
    if angle_abs == 0.5:
        print(angle_rad)