#!/usr/bin/env python3
"""
plot_contours.py

读取工作区 `Csv/` 目录下的 `contour_*.csv` 文件，并把 x,y 数据绘制为叠加的散点图/线图，最后保存到 `Picture/contours_plot.png`。

用法示例:
  python plot_contours.py -d Csv -o Picture/contours_plot.png

如果缺少依赖 (pandas/matplotlib)，脚本会提示如何安装。
"""

# 画单个文件
# 如何使用（PowerShell，可直接复制粘贴）

# 绘制单个 CSV 并保存（指定输出）：
# python plot_contours.py -d Csv -f contour_350000.csv -o Picture/contour_350000.png

# 绘制单个 CSV（不指定输出，脚本会自动命名为 Picture/<csv_basename>.png）：
# python plot_contours.py -d Csv -f contour_350000.csv

# 绘制目录中所有匹配文件（默认行为）并保存为 Picture/contours_plot.png：
# python plot_contours.py -d Csv -f contour_350000.csv

# 同时弹出交互窗口查看：
# python plot_contours.py -d Csv -f contour_350000.csv --show

import argparse
import glob
import os
import sys

def main():
    # 这块的整体流程还不是很懂
    parser = argparse.ArgumentParser(description="Plot contour CSV files from a folder")
    parser.add_argument('-d', '--dir', default='Csv', help='directory containing contour_*.csv files')
    parser.add_argument('-o', '--out', default=os.path.join('Picture','contours_plot.png'), help='output image path')
    parser.add_argument('-f', '--file', help='single CSV file name or path to plot (overrides pattern)')
    parser.add_argument('--show', action='store_true', help='also show the plot interactively')
    parser.add_argument('--pattern', default='contour_*.csv', help='glob pattern for files')
    args = parser.parse_args()

    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except Exception as e:
        print('缺少依赖: pandas 或 matplotlib。请先执行：')
        print('\n    pip install pandas matplotlib\n')
        raise

    # 支持单文件绘制：若提供了 --file，则优先使用该文件（可为绝对路径或相对于 args.dir 的文件名）
    if args.file:
        candidate = args.file
        if not os.path.isabs(candidate):
            # 先在指定目录下查找
            candidate_in_dir = os.path.join(args.dir, candidate)
            if os.path.isfile(candidate_in_dir):
                files = [candidate_in_dir]
            elif os.path.isfile(candidate):
                files = [candidate]
            else:
                print(f'指定的文件 `{args.file}` 在当前目录或 `{args.dir}` 中未找到。')
                sys.exit(1)
        else:
            if os.path.isfile(candidate):
                files = [candidate]
            else:
                print(f'指定的文件 `{args.file}` 未找到。')
                sys.exit(1)
    else:
        search_path = os.path.join(args.dir, args.pattern)
        files = sorted(glob.glob(search_path))
        if not files:
            print(f'在目录 `{args.dir}` 中未找到匹配 `{args.pattern}` 的文件。')
            sys.exit(1)

    # 如果只绘制单个文件且用户没有更改输出名，则使用该 CSV 名称作为输出文件名
    default_out = os.path.join('Picture','contours_plot.png')
    if len(files) == 1 and args.out == default_out:
        base = os.path.splitext(os.path.basename(files[0]))[0]
        args.out = os.path.join('Picture', f'{base}.png')

    plt.figure(figsize=(10,8), dpi=150)
    ax = plt.gca()

    # 用颜色映射区分不同文件
    cmap = plt.get_cmap('tab10')

    for idx, f in enumerate(files):
        try:
            df = pd.read_csv(f)
        except Exception:
            # 尝试没有头的 CSV
            df = pd.read_csv(f, header=None, names=['x','y'])

        if 'x' not in df.columns or 'y' not in df.columns:
            # 取前两列作为 x,y
            df = df.iloc[:, :2]
            df.columns = ['x','y']

        x = df['x'].astype(float)
        y = df['y'].astype(float)

        # 绘制：线 + 半透明点，便于观察重叠
        color = cmap(idx % 10)
        ax.plot(x, y, '-', color=color, linewidth=0.8, alpha=0.9, label=os.path.basename(f))
        ax.scatter(x, y, s=4, color=color, alpha=0.4)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Overlay of contour_*.csv')
    ax.invert_yaxis()  # 如果需要倒置 y 轴（图像像素坐标）可以取消或保留
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(fontsize='small', loc='upper right', ncol=1)
    ax.grid(True, linestyle=':', linewidth=0.5)

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f'已保存图片到 `{args.out}`，包含 {len(files)} 个文件的曲线叠加。')

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
