#!/usr/bin/env python3
"""
Battery Fitting Calculator & 3D Visualizer
============================================
电池适配程序：给定电池仓尺寸和单节电池参数（长宽高、电压），
计算最优排列方案，判断能否满足目标电压需求，并提供3D可视化。

用法:
    python battery_fitter.py                  # 交互式输入
    python battery_fitter.py --example        # 运行内置示例
    python battery_fitter.py --help           # 查看帮助
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from itertools import permutations
import argparse
import sys


# ============================================================
#  核心计算
# ============================================================

def get_orientations(l, w, h):
    """返回电池所有6种正交旋转方向（去重）。"""
    return list(set(permutations((l, w, h))))


def calc_counts(compartment, batt_dims):
    """计算给定电池方向下每个维度能放几个。"""
    return tuple(int(compartment[i] // batt_dims[i]) for i in range(3))


def effective_dims(orient, gap):
    """返回计入电池间间隙后的有效尺寸 (pitch)。"""
    return tuple(d + gap for d in orient)


def find_best_arrangement(compartment, batt_l, batt_w, batt_h, batt_gap=0.0):
    """
    寻找最优排列方案。
    策略1: 单一方向填满整个仓
    策略2: 沿某个轴分两段，每段使用不同电池方向（混合排列）

    返回: (total_count, arrangement_dict)
    """
    orientations = get_orientations(batt_l, batt_w, batt_h)

    best_total = 0
    best_arr = None

    # 策略1: 单一方向
    for orient in orientations:
        eff = effective_dims(orient, batt_gap)
        counts = calc_counts(compartment, eff)
        total = counts[0] * counts[1] * counts[2]
        if total > best_total:
            best_total = total
            best_arr = {
                'type': 'single',
                'orient': orient,
                'eff_orient': eff,
                'counts': counts,
            }

    # 策略2: 混合方向 (沿某个轴分两段)
    min_eff = min(batt_l, batt_w, batt_h) + batt_gap
    for split_axis in range(3):
        for orient1 in orientations:
            eff1 = effective_dims(orient1, batt_gap)
            counts1 = calc_counts(compartment, eff1)
            used_len = counts1[split_axis] * eff1[split_axis]
            remaining = compartment[split_axis] - used_len

            if remaining < min_eff:
                continue

            sub_comp = list(compartment)
            sub_comp[split_axis] = remaining
            sub_comp = tuple(sub_comp)

            for orient2 in orientations:
                eff2 = effective_dims(orient2, batt_gap)
                counts2 = calc_counts(sub_comp, eff2)
                total = (counts1[0] * counts1[1] * counts1[2] +
                         counts2[0] * counts2[1] * counts2[2])

                if total > best_total:
                    best_total = total
                    best_arr = {
                        'type': 'mixed',
                        'split_axis': split_axis,
                        'offset': used_len,
                        'orient1': orient1,
                        'eff1': eff1,
                        'counts1': counts1,
                        'orient2': orient2,
                        'eff2': eff2,
                        'counts2': counts2,
                    }

    return best_total, best_arr


def try_pcb_placements(compartment, pcb_dims, batt_l, batt_w, batt_h, batt_gap):
    """
    尝试保护板(PCB)的所有可能放置方式，返回最优方案。

    PCB 有三个维度，其中厚度方向决定减小仓的哪个轴，
    PCB 面方向（face）决定贴在哪个内壁上。

    返回: (best_total, best_arrangement, effective_compartment, pcb_info)
    """
    pcb_l, pcb_w, pcb_h = pcb_dims
    comp_l, comp_w, comp_h = compartment

    best_total = 0
    best_arrangement = None
    best_compartment = compartment
    best_pcb_info = None

    # 6 种 PCB 摆放: 3 个厚度方向 × 2 种面旋转
    axis_names = ['Length', 'Width', 'Height']

    configs = [
        # (thickness, face_d1, face_d2, reduce_axis, cross_d1, cross_d2, cross_limit1, cross_limit2)
        (pcb_l, pcb_w, pcb_h, 0, comp_w, comp_h),  # 厚度沿 L, 面在 W×H
        (pcb_l, pcb_h, pcb_w, 0, comp_h, comp_w),  # 同上, 面旋转
        (pcb_w, pcb_l, pcb_h, 1, comp_l, comp_h),  # 厚度沿 W, 面在 L×H
        (pcb_w, pcb_h, pcb_l, 1, comp_h, comp_l),  # 同上, 面旋转
        (pcb_h, pcb_l, pcb_w, 2, comp_l, comp_w),  # 厚度沿 H, 面在 L×W
        (pcb_h, pcb_w, pcb_l, 2, comp_w, comp_l),  # 同上, 面旋转
    ]

    for thickness, fd1, fd2, red_axis, cl1, cl2 in configs:
        # 检查 PCB 面是否在对应截面上放得下
        if thickness <= 0 or fd1 > cl1 or fd2 > cl2:
            continue

        eff_comp = list(compartment)
        eff_comp[red_axis] = compartment[red_axis] - thickness
        if eff_comp[red_axis] <= 0:
            continue
        eff_comp = tuple(eff_comp)

        total, arr = find_best_arrangement(eff_comp, batt_l, batt_w, batt_h, batt_gap)

        if total > best_total:
            best_total = total
            best_arrangement = arr
            best_compartment = eff_comp
            best_pcb_info = {
                'dims': pcb_dims,
                'face': (fd1, fd2),
                'thickness': thickness,
                'axis': red_axis,
                'axis_name': axis_names[red_axis],
            }

    return best_total, best_arrangement, best_compartment, best_pcb_info


def generate_positions(compartment, arrangement, pcb_info=None):
    """根据排列方案生成所有电池的位置和方向尺寸。
    位置使用有效 pitch (含间隙) 计算间距，但 dims 存实际电池尺寸用于绘制。
    如果提供了 pcb_info，会在合适位置插入保护板信息。"""
    positions = []

    if arrangement['type'] == 'single':
        orient = arrangement['orient']
        eff = arrangement['eff_orient']
        counts = arrangement['counts']
        for i in range(counts[0]):
            for j in range(counts[1]):
                for k in range(counts[2]):
                    positions.append({
                        'pos': (i * eff[0], j * eff[1], k * eff[2]),
                        'dims': orient,
                        'kind': 'battery'
                    })

    elif arrangement['type'] == 'mixed':
        o1 = arrangement['orient1']
        e1 = arrangement['eff1']
        c1 = arrangement['counts1']
        for i in range(c1[0]):
            for j in range(c1[1]):
                for k in range(c1[2]):
                    positions.append({
                        'pos': (i * e1[0], j * e1[1], k * e1[2]),
                        'dims': o1,
                        'kind': 'battery'
                    })

        o2 = arrangement['orient2']
        e2 = arrangement['eff2']
        c2 = arrangement['counts2']
        split = arrangement['split_axis']
        offset = arrangement['offset']
        for i in range(c2[0]):
            for j in range(c2[1]):
                for k in range(c2[2]):
                    base = [i * e2[0], j * e2[1], k * e2[2]]
                    base[split] += offset
                    positions.append({
                        'pos': tuple(base),
                        'dims': o2,
                        'kind': 'battery'
                    })

    # 添加保护板位置 (紧贴电池堆末端)
    pcb_block = None
    if pcb_info:
        cl, cw, ch = compartment
        face_d1, face_d2 = pcb_info['face']
        thickness = pcb_info['thickness']
        axis = pcb_info['axis']

        # 计算电池在 PCB 轴方向上的最大延伸位置
        max_extent = 0
        for batt in positions:
            end = batt['pos'][axis] + batt['dims'][axis]
            if end > max_extent:
                max_extent = end

        # PCB 紧贴电池堆顶端，不超出仓壁
        boundary = ch if axis == 2 else (cw if axis == 1 else cl)
        pcb_base = min(max_extent, boundary - thickness)

        if axis == 0:  # 减小 Length, PCB 在 X 远端
            pcb_pos = (pcb_base, 0, 0)
            pcb_dims = (thickness, face_d1, face_d2)
        elif axis == 1:  # 减小 Width, PCB 在 Y 远端
            pcb_pos = (0, pcb_base, 0)
            pcb_dims = (face_d1, thickness, face_d2)
        else:  # 减小 Height, PCB 在 Z 远端 (顶部)
            pcb_pos = (0, 0, pcb_base)
            pcb_dims = (face_d1, face_d2, thickness)

        pcb_block = {
            'pos': pcb_pos,
            'dims': pcb_dims,
            'kind': 'pcb'
        }

    return positions, pcb_block


# ============================================================
#  可视化
# ============================================================

def draw_box(ax, x, y, z, dx, dy, dz, color, alpha=0.6):
    """在3D坐标系中画一个长方体。"""
    v = np.array([
        [x, y, z], [x + dx, y, z], [x + dx, y + dy, z], [x, y + dy, z],
        [x, y, z + dz], [x + dx, y, z + dz],
        [x + dx, y + dy, z + dz], [x, y + dy, z + dz]
    ])
    faces = [
        [v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]],
        [v[0], v[1], v[5], v[4]], [v[2], v[3], v[7], v[6]],
        [v[1], v[2], v[6], v[5]], [v[0], v[3], v[7], v[4]]
    ]
    ax.add_collection3d(
        Poly3DCollection(faces, alpha=alpha, facecolor=color,
                         edgecolor='black', linewidth=0.3)
    )


def draw_wireframe(ax, l, w, h):
    """画电池仓虚线框。"""
    v = [[0, 0, 0], [l, 0, 0], [l, w, 0], [0, w, 0],
         [0, 0, h], [l, 0, h], [l, w, h], [0, w, h]]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for e in edges:
        ax.plot3D(*zip(v[e[0]], v[e[1]]),
                  color='gray', linewidth=1.5, linestyle='--')


def visualize_3d(compartment, positions, info_text, pcb_block=None):
    """3D 视图 —— 电池仓线框 + 所有电池 + 保护板。"""
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection='3d')

    cl, cw, ch = compartment
    draw_wireframe(ax, cl, cw, ch)

    n = len(positions)
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, max(1, n)))

    for i, batt in enumerate(positions):
        x, y, z = batt['pos']
        dx, dy, dz = batt['dims']
        draw_box(ax, x, y, z, dx, dy, dz, colors[i], alpha=0.65)

    # 绘制保护板 (红色半透明)
    if pcb_block:
        x, y, z = pcb_block['pos']
        dx, dy, dz = pcb_block['dims']
        draw_box(ax, x, y, z, dx, dy, dz, color='crimson', alpha=0.55)

    max_extent = max(compartment) * 1.15
    ax.set_xlim(0, max_extent)
    ax.set_ylim(0, max_extent)
    ax.set_zlim(0, max_extent)

    ax.set_xlabel('X / Length (mm)', fontsize=10)
    ax.set_ylabel('Y / Width (mm)', fontsize=10)
    ax.set_zlabel('Z / Height (mm)', fontsize=10)
    ax.set_title('Battery Arrangement — 3D View', fontsize=13, fontweight='bold')

    fig.text(0.02, 0.02, info_text, fontsize=8.5, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
             verticalalignment='bottom')

    plt.tight_layout()
    return fig


def visualize_2d(compartment, positions, pcb_block=None):
    """三个正交投影：俯视(X-Y)、正视(X-Z)、侧视(Y-Z)。"""
    cl, cw, ch = compartment
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    views = [
        (0, 1, 'Top View (X-Y)', cl, cw, 'Length (mm)', 'Width (mm)'),
        (0, 2, 'Front View (X-Z)', cl, ch, 'Length (mm)', 'Height (mm)'),
        (1, 2, 'Side View (Y-Z)', cw, ch, 'Width (mm)', 'Height (mm)'),
    ]

    cmap = plt.cm.Set2
    n_colors = 8

    for idx, (di, dj, title, lim_i, lim_j, xlab, ylab) in enumerate(views):
        ax = axes[idx]
        # 电池
        for ci, batt in enumerate(positions):
            p = batt['pos']
            d = batt['dims']
            rect = plt.Rectangle(
                (p[di], p[dj]), d[di], d[dj],
                linewidth=0.8, edgecolor='black',
                facecolor=cmap(ci % n_colors / n_colors), alpha=0.5
            )
            ax.add_patch(rect)

        # 保护板
        if pcb_block:
            p = pcb_block['pos']
            d = pcb_block['dims']
            pcb_rect = plt.Rectangle(
                (p[di], p[dj]), d[di], d[dj],
                linewidth=1.2, edgecolor='darkred',
                facecolor='crimson', alpha=0.45, hatch='//'
            )
            ax.add_patch(pcb_rect)

        ax.set_xlim(0, lim_i * 1.08)
        ax.set_ylim(0, lim_j * 1.08)
        ax.set_aspect('equal')
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ============================================================
#  报告输出
# ============================================================

def print_report(compartment, margin_info, batt_params, target_v,
                 total, arrangement, can_fit, batt_gap, pcb_info=None):
    """打印计算结果报告。"""
    batt_l, batt_w, batt_h, batt_v = batt_params
    batteries_needed = int(np.ceil(target_v / batt_v))
    max_voltage = total * batt_v

    print("\n" + "=" * 62)
    print("   计 算 结 果")
    print("=" * 62)
    print(f"   电池仓有效空间: {compartment[0]:.1f} × {compartment[1]:.1f} × {compartment[2]:.1f} mm")
    if margin_info:
        print(f"   {margin_info}")
    print(f"   单节电池: {batt_l} × {batt_w} × {batt_h} mm  |  {batt_v} V")
    if batt_gap > 0:
        print(f"   电池间间隙: {batt_gap:.1f} mm")
        eff_l, eff_w, eff_h = batt_l + batt_gap, batt_w + batt_gap, batt_h + batt_gap
        print(f"   电池有效占位 (含间隙): {eff_l:.1f} × {eff_w:.1f} × {eff_h:.1f} mm")
    if pcb_info:
        pcb_l, pcb_w, pcb_h = pcb_info['dims']
        print(f"   保护板尺寸: {pcb_l:.1f} × {pcb_w:.1f} × {pcb_h:.1f} mm")
        print(f"   保护板安装: 贴在 {pcb_info['axis_name']} 方向内壁, "
              f"厚度 {pcb_info['thickness']:.1f} mm, "
              f"面 {pcb_info['face'][0]:.1f}×{pcb_info['face'][1]:.1f} mm")
    print(f"   目标电压: {target_v} V")
    print(f"   " + "-" * 54)
    print(f"   最多容纳: {total} 节电池")
    print(f"   至少需要: {batteries_needed} 节  (串联 ≈ {batteries_needed * batt_v:.1f} V)")
    print(f"   最大电压: {max_voltage:.1f} V  ({total} 节串联)")

    if arrangement['type'] == 'single':
        o = arrangement['orient']
        c = arrangement['counts']
        print(f"\n   最优排列方向: {o[0]:.1f} × {o[1]:.1f} × {o[2]:.1f} mm")
        print(f"   排列数量: {c[0]} × {c[1]} × {c[2]} = {total} 节")
    else:
        print(f"\n   采用混合方向排列以最大化空间利用")

    if can_fit:
        print(f"\n   ✓  可以满足电压需求!")
        ns = batteries_needed
        np_max = total // ns
        print(f"   建议: {ns}S 串联 → {ns * batt_v:.1f} V", end="")
        if np_max > 1:
            print(f", 剩余 {total - ns} 节可做 {np_max}P 并联扩容")
        else:
            print()
    else:
        shortfall = batteries_needed - total
        print(f"\n   ✗  无法满足! 还差 {shortfall} 节 (≈ {shortfall * batt_v:.1f} V)")
        print(f"   建议: 使用更大电池仓或更高电压的单节电池")

    # 空间利用率
    batt_vol = batt_l * batt_w * batt_h
    used_vol = total * batt_vol
    comp_vol = compartment[0] * compartment[1] * compartment[2]
    if pcb_info:
        pcb_l, pcb_w, pcb_h = pcb_info['dims']
        comp_vol += pcb_info['thickness'] * pcb_info['face'][0] * pcb_info['face'][1]
    util = used_vol / comp_vol * 100 if comp_vol > 0 else 0
    print(f"\n   空间利用率: {util:.1f}%  ({used_vol:.0f} / {comp_vol:.0f} mm³)")
    print("=" * 62)


# ============================================================
#  交互式输入 / 命令行入口
# ============================================================

def interactive_input():
    """交互式获取用户输入。"""
    print("╔" + "═" * 60 + "╗")
    print("║" + "     电池适配计算器 / Battery Fitting Calculator".center(52) + "║")
    print("╚" + "═" * 60 + "╝")
    print("  给定电池仓和电池参数，计算最优排列并可视化")

    try:
        print("\n  ── 电池仓参数 (Compartment) ──")
        comp_l = float(input("    长度 Length (mm): "))
        comp_w = float(input("    宽度 Width  (mm): "))
        comp_h = float(input("    高度 Height (mm): "))
        margin = input("    仓壁间隙 Margin (mm) [0]: ").strip()
        margin = float(margin) if margin else 0.0

        print("\n  ── 单节电池参数 (Single Battery) ──")
        batt_l = float(input("    长度 Length (mm): "))
        batt_w = float(input("    宽度 Width  (mm): "))
        batt_h = float(input("    高度 Height (mm): "))
        batt_v = float(input("    标称电压 Voltage (V): "))
        batt_gap = input("    电池间间隙 Gap (mm) [0]: ").strip()
        batt_gap = float(batt_gap) if batt_gap else 0.0

        print("\n  ── 电压需求 ──")
        target_v = float(input("    目标总电压 Target (V): "))

        print("\n  ── 保护板 (可选, 回车跳过) ──")
        pcb_str = input("    保护板 长×宽×厚 (mm), 如 50,30,3: ").strip()
        if pcb_str:
            parts = [float(x) for x in pcb_str.replace('×', ',').replace('x', ',').split(',')]
            if len(parts) != 3:
                print("\n  ❌ 保护板参数格式错误，请用逗号或×分隔三个数字。")
                sys.exit(1)
            pcb_l, pcb_w, pcb_h = parts
        else:
            pcb_l = pcb_w = pcb_h = 0.0

    except (ValueError, EOFError):
        print("\n  ❌ 输入无效，请使用数字。")
        sys.exit(1)

    return (comp_l, comp_w, comp_h, margin,
            batt_l, batt_w, batt_h, batt_v, batt_gap, target_v,
            pcb_l, pcb_w, pcb_h)


def run(comp_l, comp_w, comp_h, margin,
        batt_l, batt_w, batt_h, batt_v, batt_gap, target_v,
        pcb_l=0, pcb_w=0, pcb_h=0):
    """执行计算与可视化。"""

    # 有效性检查
    params = [comp_l, comp_w, comp_h, batt_l, batt_w, batt_h, batt_v, target_v]
    if any(d <= 0 for d in params):
        print("\n  ❌ 所有参数必须大于 0 !")
        sys.exit(1)
    if batt_gap < 0:
        print("\n  ❌ 电池间间隙不能为负!")
        sys.exit(1)

    raw_comp = (comp_l, comp_w, comp_h)
    compartment = (
        max(0, comp_l - 2 * margin),
        max(0, comp_w - 2 * margin),
        max(0, comp_h - 2 * margin),
    )

    if min(compartment) <= 0:
        print("\n  ❌ 间隙过大，有效空间为 0 !")
        sys.exit(1)

    margin_info = (f"(原始: {raw_comp[0]}×{raw_comp[1]}×{raw_comp[2]} mm, 仓壁间隙={margin} mm)"
                   if margin > 0 else "")

    # 处理保护板
    has_pcb = pcb_l > 0 and pcb_w > 0 and pcb_h > 0
    pcb_dims = (pcb_l, pcb_w, pcb_h) if has_pcb else None
    pcb_info = None
    pcb_block = None

    # 核心计算
    if has_pcb:
        total, arrangement, eff_compartment, pcb_info = try_pcb_placements(
            compartment, pcb_dims, batt_l, batt_w, batt_h, batt_gap)
        if pcb_info is None:
            print("\n  ❌ 保护板太大，无论怎么放置都放不下!")
            sys.exit(1)
    else:
        eff_compartment = compartment
        total, arrangement = find_best_arrangement(compartment, batt_l, batt_w, batt_h, batt_gap)

    if total == 0:
        print("\n  ❌ 电池太大，一节也放不下!")
        print(f"     最小电池边长 = {min(batt_l, batt_w, batt_h):.1f} mm")
        print(f"     有效间距      = {batt_gap:.1f} mm")
        print(f"     仓最小内径    = {min(compartment):.1f} mm")
        sys.exit(1)

    batteries_needed = int(np.ceil(target_v / batt_v))
    can_fit = total >= batteries_needed
    max_voltage = total * batt_v
    status = "✓ PASS" if can_fit else "✗ FAIL"

    # 输出报告
    print_report(eff_compartment, margin_info,
                 (batt_l, batt_w, batt_h, batt_v),
                 target_v, total, arrangement, can_fit, batt_gap, pcb_info)

    # 生成位置 (包含保护板)
    positions, pcb_block = generate_positions(compartment, arrangement, pcb_info)

    # 信息条
    gap_str = f", gap={batt_gap:.1f}" if batt_gap > 0 else ""
    pcb_str = f", PCB={pcb_l}×{pcb_w}×{pcb_h}" if has_pcb else ""
    info = (
        f"Compartment: {raw_comp[0]}×{raw_comp[1]}×{raw_comp[2]} mm | "
        f"Battery: {batt_l}×{batt_w}×{batt_h} mm, {batt_v} V{gap_str}{pcb_str}\n"
        f"Target: {target_v} V | Fits: {total} | Max V: {max_voltage:.1f} V | "
        f"Result: {status}"
    )

    # 可视化
    print("\n  正在生成可视化图表...")
    visualize_3d(compartment, positions, info, pcb_block)

    # 电池太多时 2D 视图更有用
    if total <= 200:
        visualize_2d(compartment, positions, pcb_block)
    else:
        print("  (电池数量 > 200，仅显示 3D 视图)")

    print("  完成! 关闭图表窗口退出。")
    plt.show()


# ============================================================
#  示例
# ============================================================

def run_example():
    """运行一个典型示例：4S 锂电池组 + 保护板。"""
    print("运行示例: 18650 锂电池 4S 配置 + 保护板\n")
    # 电池仓: 80×60×80mm, 电池: 18650 (≈18×18×65mm, 3.7V), 目标: 14.8V
    # 保护板: 50×30×3mm
    run(comp_l=80, comp_w=60, comp_h=80, margin=1.0,
        batt_l=18, batt_w=18, batt_h=65, batt_v=3.7, batt_gap=1.0, target_v=14.8,
        pcb_l=50, pcb_w=30, pcb_h=3)


# ============================================================
#  main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="电池适配计算器 — 计算电池仓能否装下足够电池满足电压需求，并提供3D可视化"
    )
    parser.add_argument('--example', action='store_true',
                        help='运行内置示例 (18650 4S 配置)')
    parser.add_argument('--no-gui', action='store_true',
                        help='仅文本输出，不显示图表')
    args = parser.parse_args()

    if args.example:
        run_example()
        return

    params = interactive_input()
    (comp_l, comp_w, comp_h, margin,
     batt_l, batt_w, batt_h, batt_v, batt_gap, target_v,
     pcb_l, pcb_w, pcb_h) = params

    if args.no_gui:
        # 仅计算，不画图
        import matplotlib
        matplotlib.use('Agg')

    run(comp_l, comp_w, comp_h, margin,
        batt_l, batt_w, batt_h, batt_v, batt_gap, target_v,
        pcb_l, pcb_w, pcb_h)


if __name__ == '__main__':
    main()
