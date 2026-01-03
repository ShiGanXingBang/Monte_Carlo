import numpy as np
import matplotlib.pyplot as plt

def g_i_IICE(theta):
    """
    计算离子增强化学刻蚀函数值
    
    公式: g_i^IICE(θ) = 0.9 * (1.1 - 0.31*θ + 1.61*θ^2 - 2.13*θ^3 + 0.6*θ^4)
    
    参数:
    theta: 角度值（可以是一个数，也可以是numpy数组）
    
    返回:
    函数计算结果
    """
    cos_theta = np.cos(theta)
    cos2 = cos_theta**2
    cos3 = cos_theta**3
    cos4 = cos_theta**4
    cos5 = cos_theta**5
    cos6 = cos_theta**6
    
    # 计算物理溅射函数
    result = 0.4 * (18.7*cos_theta - 64.7*cos2 + 145.2*cos3 - 206*cos4 + 147.3*cos5 - 39.9*cos6)
    return result
    # return 0.9 * (1.1 - 0.31*theta + 1.61*theta**2 - 2.13*theta**3 + 0.6*theta**4)

def plot_IICE_function(theta_start=0, theta_end=2, num_points=200):
    """
    绘制离子增强化学刻蚀函数图像
    
    参数:
    theta_start: θ的起始值 (默认: 0)
    theta_end: θ的结束值 (默认: 2)
    num_points: 绘制的点数 (默认: 200)
    """
    # 1. 生成θ值
    theta_values = np.linspace(theta_start, theta_end, num_points)
    
    # 2. 计算对应的函数值
    g_values = g_i_IICE(theta_values)
    
    # 3. 创建图形
    plt.figure(figsize=(10, 6))
    
    # 4. 绘制函数曲线
    plt.plot(theta_values, g_values, 'b-', linewidth=2, label=r'$g_i^{IICE}(\theta)$')
    
    # 5. 添加标签和标题
    plt.xlabel(r'$\theta$ (角度)', fontsize=12)
    plt.ylabel(r'$g_i^{IICE}(\theta)$', fontsize=12)
    plt.title('离子增强化学刻蚀函数', fontsize=14, fontweight='bold')
    
    # 6. 显示公式
    equation_text = r'$g_i^{IICE}(\theta) = 0.9(1.1 - 0.31\cdot\theta + 1.61\cdot\theta^2 - 2.13\cdot\theta^3 + 0.6\cdot\theta^4)$'
    plt.text(0.5, 0.95, equation_text, 
             transform=plt.gca().transAxes,  # 使用相对坐标
             fontsize=11, 
             ha='center',  # 水平居中
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    # 7. 添加网格和网格线
    plt.grid(True, alpha=0.3)
    
    # 8. 添加图例
    plt.legend(fontsize=12)
    
    # 9. 显示图形
    plt.tight_layout()
    plt.show()
    
    # 10. 返回数据供进一步分析
    return theta_values, g_values

# 高级版本：添加更多功能
def plot_IICE_function_enhanced():
    """
    增强版的绘图函数，显示更多信息
    """
    # 1. 生成数据
    theta = np.linspace(0, 2, 400)
    g = g_i_IICE(theta)
    
    # 2. 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 3. 主图
    ax1.plot(theta, g, 'r-', linewidth=3, label='函数曲线')
    
    # 标记特殊点
    # 找最大值
    max_idx = np.argmax(g)
    ax1.plot(theta[max_idx], g[max_idx], 'ro', markersize=10, label=f'最大值: θ={theta[max_idx]:.2f}, g={g[max_idx]:.2f}')
    
    # 找零点（如果有）
    zero_crossings = np.where(np.diff(np.sign(g)))[0]
    for idx in zero_crossings:
        ax1.plot(theta[idx], g[idx], 'go', markersize=8, label=f'零点附近')
    
    # 4. 添加各种标记
    ax1.set_xlabel(r'$\theta$', fontsize=12)
    ax1.set_ylabel(r'$g_i^{IICE}(\theta)$', fontsize=12)
    ax1.set_title('离子增强化学刻蚀函数详细分析', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 5. 在图表中显示公式
    equation = r'$g_i^{IICE}(\theta) = 0.9(1.1 - 0.31\theta + 1.61\theta^2 - 2.13\theta^3 + 0.6\theta^4)$'
    ax1.text(0.5, 0.9, equation, transform=ax1.transAxes, fontsize=10, ha='center', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
    
    # 6. 子图：导数（变化率）
    # 计算数值导数
    derivative = np.gradient(g, theta)
    ax2.plot(theta, derivative, 'g-', linewidth=2, label='导数（变化率）')
    ax2.set_xlabel(r'$\theta$', fontsize=12)
    ax2.set_ylabel(r"$dg_i^{IICE}/d\theta$", fontsize=12)
    ax2.set_title('函数的变化率（导数）', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 标记导数为0的点（极值点）
    zero_derivative = np.where(np.diff(np.sign(derivative)))[0]
    for idx in zero_derivative:
        ax2.plot(theta[idx], derivative[idx], 'ro', markersize=6)
    
    # 7. 调整布局
    plt.tight_layout()
    plt.show()
    
    # 8. 打印统计信息
    print("=== 函数分析结果 ===")
    print(f"θ范围: 0 到 2")
    print(f"函数值范围: {g.min():.4f} 到 {g.max():.4f}")
    print(f"最大值: θ = {theta[max_idx]:.4f}, g = {g[max_idx]:.4f}")
    print(f"θ=0时的值: {g[0]:.4f}")
    print(f"θ=1时的值: {g_i_IICE(1):.4f}")
    print(f"θ=2时的值: {g_i_IICE(2):.4f}")

# 使用示例1：基本绘图
if __name__ == "__main__":
    print("正在绘制离子增强化学刻蚀函数...")
    
    # 方法1：使用基本绘图函数
    theta_vals, g_vals = plot_IICE_function(theta_start=0, theta_end=1.57)
    
    # 方法2：使用增强版
    # plot_IICE_function_enhanced()
    
    # 方法3：手动控制绘图
    # 创建θ值数组
    # theta_range = np.linspace(0, 2, 100)
    # 计算函数值
    # values = g_i_IICE(theta_range)
    # 手动绘图
    # plt.plot(theta_range, values, 'b-')
    # plt.xlabel('θ')
    # plt.ylabel('g_i^IICE(θ)')
    # plt.title('离子增强化学刻蚀')
    # plt.grid(True)
    # plt.show()

# 额外工具函数
def calculate_specific_values():
    """计算特定θ值的函数值"""
    test_points = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.57]
    
    print("\n=== 特定θ值的计算结果 ===")
    for theta in test_points:
        result = g_i_IICE(theta)
        print(f"θ = {theta:4.2f} 时, g_i^IICE(θ) = {result:.6f}")
    
    return {theta: g_i_IICE(theta) for theta in test_points}