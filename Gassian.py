import numpy as np
import math

def generate_ion_angles(num_samples=1000, sigma_degrees=10):
    """
    生成遵循高斯分布的离子入射角度
    
    参数:
        num_samples: 生成的角度样本数量
        sigma_degrees: 标准差（以度为单位），控制分布宽度
    
    返回:
        angles_rad: 弧度制的角度数组
    """
    # 将标准差转换为弧度
    sigma_rad = np.radians(sigma_degrees)
    
    # 生成高斯分布的随机角度（均值0，标准差sigma_rad）
    angles_rad = np.random.normal(loc=0.0, scale=sigma_rad, size=num_samples)
    
    # 截断角度到[-π/2, π/2]范围，避免无效值
    angles_rad = np.clip(angles_rad, -math.pi/2, math.pi/2)
    return angles_rad

# 使用示例
angles = generate_ion_angles(10000, sigma_degrees=10)
print(angles)
# 可视化分布
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# 第一张图：角度（度）
axs[0].hist(np.degrees(angles), bins=50, density=True)
axs[0].set_xlabel('角度 (度)')
axs[0].set_ylabel('概率密度')
axs[0].set_title('离子入射角度高斯分布（度）')

# 第二张图：角度（弧度）
axs[1].hist(angles, bins=50, density=True)
axs[1].set_xlabel('角度 (弧度)')
axs[1].set_ylabel('概率密度')
axs[1].set_title('离子入射角度高斯分布（弧度）')

plt.tight_layout()
plt.show()