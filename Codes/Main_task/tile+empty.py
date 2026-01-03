import numpy as np

# si = {'exit' : True, 'CountCL' : 0}
# Si_array = np.tile(si,(2, 5))
# # Si_array = np.tile(si, 2,)
# print(Si_array)  #tile 平铺法


class Si_single():
    def __init__(self, exit = True, CountCL = 0):
        self.exit = exit
        self.CountCL = CountCL


Si_array = np.empty((7, 10), dtype=object)
for i in range(7):
    for j in range(10):
        Si_array[i, j] = Si_single()

print(Si_array)