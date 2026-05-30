#!/usr/bin/env python3
"""
Battery Fitting Calculator — Interactive Dash App
==================================================
浏览器中打开，拖动滑块实时调整参数，3D 视图即时更新。

用法:
    python battery_fitter_app.py
    然后打开 http://127.0.0.1:8050
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import permutations
from functools import lru_cache
import json
import os
import re
import dash
from dash import dcc, html, Input, Output, State, ctx

# ============================================================
#  计算核心 (从 battery_fitter.py 提取, 保持独立)
# ============================================================

def get_orientations(l, w, h):
    return list(set(permutations((l, w, h))))

def calc_counts(compartment, batt_dims):
    return tuple(int(compartment[i] // batt_dims[i]) for i in range(3))

def effective_dims(orient, gap):
    return tuple(d + gap for d in orient)

AXIS_NAMES = ['L', 'W', 'H']

def _geom_fingerprint(compartment, arrangement):
    """几何指纹: 所有电池 (pos, dims) 排序后哈希, 直接比较最终摆放。"""
    items, _ = generate_items(compartment, arrangement, center=False)
    return tuple(sorted((it['pos'], it['dims']) for it in items if it['kind'] == 'battery'))

@lru_cache(maxsize=256)
def find_all_arrangements(compartment, batt_l, batt_w, batt_h, batt_gap=0.0):
    """返回所有可行排列方案, 按电池数降序排列。"""
    orientations = get_orientations(batt_l, batt_w, batt_h)
    results = []

    # 单一方向
    for orient in orientations:
        eff = effective_dims(orient, batt_gap)
        counts = calc_counts(compartment, eff)
        total = counts[0] * counts[1] * counts[2]
        if total == 0:
            continue
        label = (f"单一 {orient[0]:.0f}×{orient[1]:.0f}×{orient[2]:.0f} mm  "
                 f"→ {counts[0]}×{counts[1]}×{counts[2]} = {total} 节")
        results.append({
            'total': total, 'label': label,
            'type': 'single', 'orient': orient, 'eff_orient': eff, 'counts': counts,
        })

    # 混合方向 (沿单一轴切分)
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
                if orient1 == orient2:
                    continue  # 两区同向等价于单一, 跳过
                eff2 = effective_dims(orient2, batt_gap)
                counts2 = calc_counts(sub_comp, eff2)
                total = (counts1[0] * counts1[1] * counts1[2] +
                         counts2[0] * counts2[1] * counts2[2])
                if total == 0:
                    continue
                label = (f"混合|{AXIS_NAMES[split_axis]}轴 "
                         f"({orient1[0]:.0f}×{orient1[1]:.0f}×{orient1[2]:.0f})+"
                         f"({orient2[0]:.0f}×{orient2[1]:.0f}×{orient2[2]:.0f})"
                         f" → {total} 节")
                results.append({
                    'total': total, 'label': label,
                    'type': 'mixed', 'split_axis': split_axis, 'offset': used_len,
                    'orient1': orient1, 'eff1': eff1, 'counts1': counts1,
                    'orient2': orient2, 'eff2': eff2, 'counts2': counts2,
                })

    # 分层铺设 (沿每个轴叠层, DP 求最优层组合)
    scale = 10  # 精度 0.1mm
    for stack_axis in range(3):
        face_axes = [i for i in range(3) if i != stack_axis]
        stack_limit = compartment[stack_axis]

        # 构建层类型: 每种朝向 => 一种层
        layer_types = []
        for orient in orientations:
            thick = orient[stack_axis] + batt_gap
            fc = tuple(int(compartment[fa] // (orient[fa] + batt_gap)) for fa in face_axes)
            per = fc[0] * fc[1]
            if per == 0 or thick > stack_limit:
                continue
            eff = effective_dims(orient, batt_gap)
            c3d = [fc[0] if i == face_axes[0] else fc[1] if i == face_axes[1] else 1
                   for i in range(3)]
            layer_types.append({
                'orient': orient, 'thick': thick, 'eff': eff,
                'face_counts': fc, 'per_layer': per, 'counts': tuple(c3d),
            })

        if not layer_types:
            continue
        # 去重: 相同(厚度,每层电池数,朝向)只保留面利用率最高的
        seen = {}
        for lt in layer_types:
            key = (lt['thick'], lt['per_layer'], lt['orient'])
            if key not in seen or lt['face_counts'] > seen[key]['face_counts']:
                seen[key] = lt
        lts = list(seen.values())

        max_h = int(stack_limit * scale)
        dp = [{'total': -1, 'prev': 0, 'lt': -1, 'n': 0} for _ in range(max_h + 1)]
        dp[0]['total'] = 0

        for h in range(max_h + 1):
            if dp[h]['total'] < 0:
                continue
            for li, lt in enumerate(lts):
                ts = int(lt['thick'] * scale)
                max_n = (max_h - h) // ts
                for n in range(1, max_n + 1):
                    nh = h + n * ts
                    cand = dp[h]['total'] + n * lt['per_layer']
                    if cand > dp[nh]['total']:
                        dp[nh] = {'total': cand, 'prev': h, 'lt': li, 'n': n}

        # 取该轴前3优结果
        ranked = sorted(
            [(dp[h]['total'], h) for h in range(max_h + 1) if dp[h]['total'] > 0],
            key=lambda x: x[0], reverse=True
        )[:3]
        for total, end_h in ranked:
            # 回溯
            segments = []
            h = end_h
            while h > 0:
                e = dp[h]
                lt = lts[e['lt']]
                seg_start = (e['prev']) / scale
                segments.append({
                    'start': seg_start,
                    'n': e['n'], 'orient': lt['orient'], 'eff': lt['eff'],
                    'face_counts': lt['face_counts'], 'counts': lt['counts'],
                })
                h = e['prev']
            segments.reverse()

            if len(segments) == 1:
                continue  # 单层等价于单一, 跳过

            # 标签
            parts = [f"({s['orient'][0]:.0f}×{s['orient'][1]:.0f}×{s['orient'][2]:.0f})×{s['n']}"
                     for s in segments]
            label = f"分层|{AXIS_NAMES[stack_axis]}轴 " + "+".join(parts) + f" → {total} 节"

            results.append({
                'total': total, 'label': label,
                'type': 'layered', 'stack_axis': stack_axis, 'segments': segments,
            })

    # 几何指纹去重: 直接比较最终电池摆放, 而非生成过程的参数
    seen = set()
    unique = []
    for r in sorted(results, key=lambda r: r['total'], reverse=True):
        arr = {k: v for k, v in r.items() if k not in ('total', 'label')}
        fp = _geom_fingerprint(compartment, arr)
        if fp not in seen:
            seen.add(fp)
            unique.append(r)
    return unique

def find_best_arrangement(compartment, batt_l, batt_w, batt_h, batt_gap=0.0):
    """返回最佳排列方案 (电池数最多)。兼容旧接口。"""
    all_results = find_all_arrangements(compartment, batt_l, batt_w, batt_h, batt_gap)
    if not all_results:
        return 0, None
    best = all_results[0]
    return best['total'], {k: v for k, v in best.items() if k not in ('total', 'label')}

def find_pcb_on_surface(compartment, items, pcb_l, pcb_w, pcb_h,
                         pcb_gap=0.0, pcb_axis=None, pcb_near=False, wall_pos=None,
                         resolution=1.0):
    """体素化 + 积分图: 在电池暴露面上扫描 PCB 可放置区域。

    对指定仓壁构建 2D 占据网格 (0=可用, 1=障碍), 用积分图快速检测矩形。
    wall_pos: 远端扫描时可指定具体位置, 不指定则取电池最大延伸。
    """
    batteries = [it for it in items if it['kind'] == 'battery']
    if not batteries:
        return None

    cl, cw, ch = compartment
    axis_names = ['Length', 'Width', 'Height']
    pcb_dims_list = [pcb_l, pcb_w, pcb_h]
    comp_sizes = [cl, cw, ch]

    # 收集需要扫描的面: (axis, near, u_axis, v_axis, u_len, v_len, wall_pos)
    faces_to_scan = []
    for axis in range(3):
        if pcb_axis is not None and axis != pcb_axis:
            continue
        a1, a2 = [i for i in range(3) if i != axis]
        if pcb_axis is None:
            faces_to_scan.append((axis, False, a1, a2, comp_sizes[a1], comp_sizes[a2], wall_pos))
            faces_to_scan.append((axis, True, a1, a2, comp_sizes[a1], comp_sizes[a2], 0.0))
        else:
            wp = wall_pos if not pcb_near else 0.0
            faces_to_scan.append((axis, pcb_near, a1, a2, comp_sizes[a1], comp_sizes[a2], wp))

    for axis, near, u_axis, v_axis, u_len, v_len, wp in faces_to_scan:
        # 构建 2D 占据网格: 0=可用, 1=障碍 (初始全障碍)
        grid_u = int(u_len / resolution) + 2
        grid_v = int(v_len / resolution) + 2
        occ = np.ones((grid_v, grid_u), dtype=np.uint8)

        # 仓壁位置: 近端=0, 远端=指定值或电池最大延伸
        if near:
            wall_pos = 0.0
        elif wp is not None:
            wall_pos = wp
        else:
            wall_pos = max(b['pos'][axis] + b['dims'][axis] for b in batteries)

        # 分类电池: 面电池(终止于此) → 可用, 阻挡电池(起始于此) → 障碍
        # block_margin: PCB 厚度方向需要的安全距离, 用 PCB 最大边长保证覆盖所有朝向
        block_margin = max(pcb_dims_list) + pcb_gap
        on_surf = []
        blocking_cells = []
        for b in batteries:
            u0 = int(b['pos'][u_axis] / resolution) + 1
            v0 = int(b['pos'][v_axis] / resolution) + 1
            u1 = int((b['pos'][u_axis] + b['dims'][u_axis]) / resolution) + 1
            v1 = int((b['pos'][v_axis] + b['dims'][v_axis]) / resolution) + 1

            b_start = b['pos'][axis]
            b_end = b_start + b['dims'][axis]

            if near:
                on_surface = abs(b_start - wall_pos) < resolution
                # 近端 PCB 会整体 shift 电池, 不存在轴向重叠, 无需 blocking
                blocking = False
            else:
                on_surface = abs(b_end - wall_pos) < resolution
                # 远端 PCB 不移动电池, 起始位置在 PCB+gap 区域内的电池视为阻挡
                blocking = not on_surface and b_start > wall_pos and b_start - wall_pos < block_margin

            if on_surface:
                occ[v0:v1, u0:u1] = 0
                on_surf.append((u0, v0, u1, v1))
            elif blocking:
                blocking_cells.append((u0, v0, u1, v1))

        # 填充表面电池包围盒内的缝隙 (PCB 可跨接)
        if on_surf:
            rmin = min(v0 for _, v0, _, _ in on_surf)
            rmax = max(v1 for _, _, _, v1 in on_surf)
            cmin = min(u0 for u0, _, _, _ in on_surf)
            cmax = max(u1 for _, _, u1, _ in on_surf)
            occ[rmin:rmax, cmin:cmax] = 0

        # 阻挡电池标记在填充之后, 确保不被覆盖 (坐在面上的电池优先)
        for u0, v0, u1, v1 in blocking_cells:
            occ[v0:v1, u0:u1] = 1

        # 用积分图搜索能容纳 PCB 面的矩形
        # 按贴附面积降序, 优先用 PCB 最大面贴合电池表面
        integral = occ.cumsum(axis=0).cumsum(axis=1).astype(np.int64)

        thick_order = sorted(range(3), key=lambda i:
            pcb_dims_list[(i+1)%3] * pcb_dims_list[(i+2)%3], reverse=True)
        for thick_idx in thick_order:
            thickness = pcb_dims_list[thick_idx]
            face_dims = [pcb_dims_list[i] for i in range(3) if i != thick_idx]
            fw, fh = face_dims

            for rot in [False, True]:
                w, h = (fh, fw) if rot else (fw, fh)
                w_cells = max(1, int((w + 2 * pcb_gap) / resolution))
                h_cells = max(1, int((h + 2 * pcb_gap) / resolution))
                if w_cells >= grid_u or h_cells >= grid_v:
                    continue

                # 检查厚度方向是否放得下 (电池表面到仓壁的距离)
                if near:
                    # PCB 贴壁 → 电池整体偏移, 需确保偏移后不超出仓
                    max_ext = max(b['pos'][axis] + b['dims'][axis] for b in batteries)
                    if max_ext + thickness + pcb_gap > compartment[axis]:
                        continue
                else:
                    if thickness + pcb_gap > compartment[axis] - wall_pos:
                        continue

                # 积分图矩形查询: O(1) 检查每个窗口
                for i in range(grid_v - h_cells):
                    for j in range(grid_u - w_cells):
                        i2, j2 = i + h_cells - 1, j + w_cells - 1
                        s = integral[i2, j2]
                        if i > 0: s -= integral[i - 1, j2]
                        if j > 0: s -= integral[i2, j - 1]
                        if i > 0 and j > 0: s += integral[i - 1, j - 1]
                        if s == 0:
                            u_pos = max(0.0, (j - 1) * resolution + pcb_gap)
                            v_pos = max(0.0, (i - 1) * resolution + pcb_gap)
                            return {
                                'dims': (pcb_l, pcb_w, pcb_h),
                                'face': (w, h),
                                'thickness': thickness,
                                'axis': axis,
                                'axis_name': axis_names[axis],
                                'near': near,
                                'wall_pos': wall_pos,
                                'gap': pcb_gap,
                                'u_pos': u_pos,
                                'v_pos': v_pos,
                            }
    return None


AXIS_LABELS_ZH = {0: 'L轴', 1: 'W轴', 2: 'H轴'}
AXIS_COORD_ZH = {0: 'x', 1: 'y', 2: 'z'}


def find_all_pcb_placements(compartment, items, pcb_l, pcb_w, pcb_h, pcb_gap=0.0):
    """扫描所有电池暴露面, 返回所有有效 PCB 贴附位置, 按贴附面积降序。

    对每个轴的每个不同电池终止位置 (阶梯排列产生多层暴露面),
    构建占据网格并搜索 PCB 放置位置。仅扫描电池终止面, 不扫描仓壁。
    """
    batteries = [it for it in items if it['kind'] == 'battery']
    if not batteries:
        return []

    all_found = []
    for axis in range(3):
        axis_zh = AXIS_LABELS_ZH[axis]
        coord_name = AXIS_COORD_ZH[axis]

        # 远端面: 扫描所有不同的电池终止位置 (电池暴露面)
        end_positions = sorted(set(
            b['pos'][axis] + b['dims'][axis] for b in batteries
        ), reverse=True)
        for wp in end_positions:
            info = find_pcb_on_surface(compartment, items, pcb_l, pcb_w, pcb_h,
                                       pcb_gap, pcb_axis=axis, pcb_near=False,
                                       wall_pos=wp)
            if info:
                info['label'] = f"{axis_zh} ({coord_name}={wp:.0f}) — {info['axis_name']}面"
                info['face_area'] = info['face'][0] * info['face'][1]
                all_found.append(info)

        # 近端面: 扫描电池起始面 (底面/后面/左面)
        info = find_pcb_on_surface(compartment, items, pcb_l, pcb_w, pcb_h,
                                   pcb_gap, pcb_axis=axis, pcb_near=True,
                                   wall_pos=0.0)
        if info:
            info['label'] = f"{axis_zh} ({coord_name}=0) — {info['axis_name']}面 (近端)"
            info['face_area'] = info['face'][0] * info['face'][1]
            all_found.append(info)

    all_found.sort(key=lambda x: x['face_area'], reverse=True)
    return all_found


# ══════════════════════════════════════════════════════════════════
#  暴露面统一分析框架 (绝缘材料 + PCB 放置)
# ══════════════════════════════════════════════════════════════════

def _merge_rects(rects_mm, max_gap=1.0):
    """合并重叠或邻近的矩形 (Union-Find), 用于提取绝缘材料连通区域。

    rects_mm: [(u0, v0, u1, v1), ...]  面电池在贴附面上的投影矩形 (mm)
    max_gap: 间距 ≤ 此值的矩形合并为一组
    返回: [(u0, v0, u1, v1), ...]  合并后的包围盒 (mm)
    """
    n = len(rects_mm)
    if n <= 1:
        return list(rects_mm)

    half = max_gap / 2.0
    grown = [(u0 - half, v0 - half, u1 + half, v1 + half) for u0, v0, u1, v1 in rects_mm]

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        gi0, gv0, gi1, gv1 = grown[i]
        for j in range(i + 1, n):
            gj0, gv0_j, gj1, gv1_j = grown[j]
            if gi0 <= gj1 and gi1 >= gj0 and gv0 <= gv1_j and gv1 >= gv0_j:
                union(i, j)

    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    result = []
    for indices in groups.values():
        u0 = min(rects_mm[i][0] for i in indices)
        v0 = min(rects_mm[i][1] for i in indices)
        u1 = max(rects_mm[i][2] for i in indices)
        v1 = max(rects_mm[i][3] for i in indices)
        result.append((u0, v0, u1, v1))

    return result


def _analyze_one_face(compartment, batteries, axis, near, wall_pos,
                      pcb_l, pcb_w, pcb_h, pcb_gap=0.0, resolution=1.0):
    """分析单个暴露面: 提取绝缘材料组件 并 搜索 PCB 放置位置。

    返回 face dict 或 None (当该面无表面电池时)。
    PCB 搜索采用 每厚度方向独立阻挡阈值 (thickness + pcb_gap),
    避免全局 max(pcb_dims) 的过度保守问题。
    """
    a1, a2 = [i for i in range(3) if i != axis]
    u_len = compartment[a1]
    v_len = compartment[a2]

    grid_u = int(u_len / resolution) + 2
    grid_v = int(v_len / resolution) + 2

    # ── 构建基础占据网格 (面电池 → 0) ──
    occ = np.ones((grid_v, grid_u), dtype=np.uint8)
    on_surf = []
    surf_rects_mm = []
    batt_coords = []

    for b in batteries:
        u0 = int(b['pos'][a1] / resolution) + 1
        v0 = int(b['pos'][a2] / resolution) + 1
        u1 = int((b['pos'][a1] + b['dims'][a1]) / resolution) + 1
        v1 = int((b['pos'][a2] + b['dims'][a2]) / resolution) + 1

        b_start = b['pos'][axis]
        b_end = b_start + b['dims'][axis]

        if near:
            on_surface = abs(b_start - wall_pos) < resolution
        else:
            on_surface = abs(b_end - wall_pos) < resolution

        batt_coords.append((u0, v0, u1, v1, b_start, on_surface))

        if on_surface:
            occ[v0:v1, u0:u1] = 0
            on_surf.append((u0, v0, u1, v1))
            umm0 = b['pos'][a1]
            vmm0 = b['pos'][a2]
            umm1 = umm0 + b['dims'][a1]
            vmm1 = vmm0 + b['dims'][a2]
            surf_rects_mm.append((umm0, vmm0, umm1, vmm1))

    if not on_surf:
        return None

    # ── 绝缘材料组件: 合并邻近面电池投影矩形 ──
    merge_gap = max(pcb_gap, 1.0)
    merged = _merge_rects(surf_rects_mm, merge_gap)
    components = []
    for u0, v0, u1, v1 in merged:
        w = u1 - u0
        h = v1 - v0
        components.append({'u': u0, 'v': v0, 'w': w, 'h': h, 'area': w * h})

    # ── 全局包围盒填充 (PCB 可跨接缝隙) ──
    rmin = min(v0 for _, v0, _, _ in on_surf)
    rmax = max(v1 for _, _, _, v1 in on_surf)
    cmin = min(u0 for u0, _, _, _ in on_surf)
    cmax = max(u1 for _, _, u1, _ in on_surf)
    occ[rmin:rmax, cmin:cmax] = 0

    # ── 暴露性判断: 远端面是否被后排电池完全遮挡? ──
    exposed = True
    if not near:
        occ_check = occ.copy()
        block_rects = []
        for u0, v0, u1, v1, b_start, on_surface in batt_coords:
            if not on_surface and b_start > wall_pos:
                occ_check[v0:v1, u0:u1] = 1
                block_rects.append((u0, v0, u1, v1))
        # 阻挡电池也需要 bbox fill, 覆盖电池间隙
        if block_rects:
            brmin = min(v0 for _, v0, _, _ in block_rects)
            brmax = max(v1 for _, _, _, v1 in block_rects)
            bcmin = min(u0 for u0, _, _, _ in block_rects)
            bcmax = max(u1 for _, _, u1, _ in block_rects)
            occ_check[brmin:brmax, bcmin:bcmax] = 1
        # 填充区内无可暴露像素 → 纯内部面
        if not np.any(occ_check[rmin:rmax, cmin:cmax] == 0):
            exposed = False

    # ── PCB 搜索: 每厚度方向独立阻挡阈值 ──
    axis_names = {0: 'Length', 1: 'Width', 2: 'Height'}
    has_pcb = pcb_l > 0 and pcb_w > 0 and pcb_h > 0
    pcb_info = None

    if has_pcb:
        pcb_dims_list = [pcb_l, pcb_w, pcb_h]
        thick_order = sorted(range(3), key=lambda i:
            pcb_dims_list[(i + 1) % 3] * pcb_dims_list[(i + 2) % 3], reverse=True)

        for thick_idx in thick_order:
            thickness = pcb_dims_list[thick_idx]
            face_dims = [pcb_dims_list[i] for i in range(3) if i != thick_idx]
            fw, fh = face_dims

            # 为当前厚度标记阻挡电池
            occ_thick = occ.copy()
            if not near:
                for u0, v0, u1, v1, b_start, on_surface in batt_coords:
                    if not on_surface and b_start > wall_pos and b_start - wall_pos < thickness + pcb_gap:
                        occ_thick[v0:v1, u0:u1] = 1

            integral = occ_thick.cumsum(axis=0).cumsum(axis=1).astype(np.int64)

            for rot in [False, True]:
                w, h = (fh, fw) if rot else (fw, fh)
                w_cells = max(1, int((w + 2 * pcb_gap) / resolution))
                h_cells = max(1, int((h + 2 * pcb_gap) / resolution))
                if w_cells >= grid_u or h_cells >= grid_v:
                    continue

                if near:
                    max_ext = max(b['pos'][axis] + b['dims'][axis] for b in batteries)
                    if max_ext + thickness + pcb_gap > compartment[axis]:
                        continue
                else:
                    if thickness + pcb_gap > compartment[axis] - wall_pos:
                        continue

                for i in range(grid_v - h_cells):
                    for j in range(grid_u - w_cells):
                        i2, j2 = i + h_cells - 1, j + w_cells - 1
                        s = integral[i2, j2]
                        if i > 0:
                            s -= integral[i - 1, j2]
                        if j > 0:
                            s -= integral[i2, j - 1]
                        if i > 0 and j > 0:
                            s += integral[i - 1, j - 1]
                        if s == 0:
                            pcb_info = {
                                'dims': (pcb_l, pcb_w, pcb_h),
                                'face': (w, h),
                                'thickness': thickness,
                                'axis': axis,
                                'axis_name': axis_names[axis],
                                'near': near,
                                'wall_pos': wall_pos,
                                'gap': pcb_gap,
                                'u_pos': max(0.0, (j - 1) * resolution + pcb_gap),
                                'v_pos': max(0.0, (i - 1) * resolution + pcb_gap),
                            }
                            break
                    if pcb_info:
                        break
                if pcb_info:
                    break
            if pcb_info:
                break

    face_area = sum(c['area'] for c in components)
    return {
        'axis': axis,
        'axis_name': axis_names[axis],
        'near': near,
        'wall_pos': wall_pos,
        'exposed': exposed,
        'components': components,
        'face_area': face_area,
        'pcb_info': pcb_info,
    }


def find_all_exposed_faces(compartment, items, pcb_l, pcb_w, pcb_h, pcb_gap=0.0, resolution=1.0):
    """扫描所有电池暴露面, 返回绝缘材料 + PCB 放置的统一信息。

    对每个轴的每个不同电池终止位置 (阶梯排列产生多层暴露面),
    调用 _analyze_one_face 提取绝缘组件和 PCB 可放置位置。
    """
    batteries = [it for it in items if it['kind'] == 'battery']
    if not batteries:
        return []

    all_faces = []
    for axis in range(3):
        axis_zh = AXIS_LABELS_ZH[axis]
        coord_name = AXIS_COORD_ZH[axis]

        # 远端面: 扫描所有终止位置, 由 _analyze_one_face 过滤内部面
        end_positions = sorted(set(
            b['pos'][axis] + b['dims'][axis] for b in batteries
        ), reverse=True)
        for wp in end_positions:
            face = _analyze_one_face(compartment, batteries, axis, False, wp,
                                     pcb_l, pcb_w, pcb_h, pcb_gap, resolution)
            if face:
                face['label'] = f"{axis_zh} ({coord_name}={wp:.0f}) — {face['axis_name']}面"
                all_faces.append(face)

        # 近端面: 电池起始面
        face = _analyze_one_face(compartment, batteries, axis, True, 0.0,
                                 pcb_l, pcb_w, pcb_h, pcb_gap, resolution)
        if face:
            face['label'] = f"{axis_zh} ({coord_name}=0) — {face['axis_name']}面 (近端)"
            all_faces.append(face)

    all_faces.sort(key=lambda x: x['face_area'], reverse=True)
    return all_faces


# ── 序列化辅助函数 (模块级, 供回调和惰性计算共用) ──

def _tuplify(v):
    if isinstance(v, (int, float, str, bool, type(None), list, dict)):
        return v
    return list(v)


def _serialize_face(face):
    """序列化暴露面 dict 为 JSON-safe 格式。"""
    result = {}
    for k, v in face.items():
        if k == 'components':
            result['components'] = [{kk: _tuplify(vv) for kk, vv in c.items()} for c in v]
        elif k == 'pcb_info':
            result['pcb_info'] = {kk: _tuplify(vv) for kk, vv in v.items()} if v else None
        else:
            result[k] = _tuplify(v)
    return result


def generate_items(compartment, arrangement, pcb_info=None, center=True):
    """生成电池和 PCB 的位置列表。

    center=False 时跳过居中，用于 PCB 扫描阶段保持原始坐标。"""
    items = []  # each: {'pos': (x,y,z), 'dims': (dx,dy,dz), 'kind': 'battery'|'pcb'}

    if arrangement['type'] == 'single':
        orient = arrangement['orient']
        eff = arrangement['eff_orient']
        counts = arrangement['counts']
        for i in range(counts[0]):
            for j in range(counts[1]):
                for k in range(counts[2]):
                    items.append({
                        'pos': (i * eff[0], j * eff[1], k * eff[2]),
                        'dims': orient, 'kind': 'battery'
                    })
    elif arrangement['type'] == 'mixed':
        for sec in [('orient1', 'eff1', 'counts1'), ('orient2', 'eff2', 'counts2')]:
            orient = arrangement[sec[0]]
            eff = arrangement[sec[1]]
            counts = arrangement[sec[2]]
            for i in range(counts[0]):
                for j in range(counts[1]):
                    for k in range(counts[2]):
                        pos = [i * eff[0], j * eff[1], k * eff[2]]
                        if sec[0] == 'orient2':
                            pos[arrangement['split_axis']] += arrangement['offset']
                        items.append({'pos': tuple(pos), 'dims': orient, 'kind': 'battery'})

    elif arrangement['type'] == 'layered':
        axis = arrangement['stack_axis']
        face_axes = [i for i in range(3) if i != axis]
        for seg in arrangement['segments']:
            orient = seg['orient']
            eff = seg['eff']
            fc = seg['face_counts']
            for layer in range(seg['n']):
                base = [0, 0, 0]
                base[axis] = seg['start'] + layer * eff[axis]
                for fi in range(fc[0]):
                    for fj in range(fc[1]):
                        pos = list(base)
                        pos[face_axes[0]] = fi * eff[face_axes[0]]
                        pos[face_axes[1]] = fj * eff[face_axes[1]]
                        items.append({'pos': tuple(pos), 'dims': orient, 'kind': 'battery'})

    pcb_block = None
    if pcb_info:
        cl, cw, ch = compartment
        face_d1, face_d2 = pcb_info['face']
        thickness = pcb_info['thickness']
        axis = pcb_info['axis']

        pcb_gap = pcb_info.get('gap', 0.0)
        u_off = pcb_info.get('u_pos', 0.0)
        v_off = pcb_info.get('v_pos', 0.0)

        if pcb_info['near']:
            pcb_base = 0
            shift = thickness + pcb_gap
            for it in items:
                pos = list(it['pos'])
                pos[axis] += shift
                it['pos'] = tuple(pos)
        else:
            wall_pos = pcb_info.get('wall_pos', 0)
            boundary = [cl, cw, ch][axis]
            pcb_base = min(wall_pos + pcb_gap, boundary - thickness)

        if axis == 0:
            pcb_pos = (pcb_base, u_off, v_off)
            pcb_dims = (thickness, face_d1, face_d2)
        elif axis == 1:
            pcb_pos = (u_off, pcb_base, v_off)
            pcb_dims = (face_d1, thickness, face_d2)
        else:
            pcb_pos = (u_off, v_off, pcb_base)
            pcb_dims = (face_d1, face_d2, thickness)

        pcb_block = {'pos': pcb_pos, 'dims': pcb_dims, 'kind': 'pcb'}

    if center:
        all_placed = list(items)
        if pcb_block:
            all_placed.append(pcb_block)
        if all_placed:
            max_extents = [0, 0, 0]
            for it in all_placed:
                for a in range(3):
                    end = it['pos'][a] + it['dims'][a]
                    if end > max_extents[a]:
                        max_extents[a] = end
            offsets = tuple(max(0, (compartment[a] - max_extents[a]) / 2) for a in range(3))
            for it in items:
                it['pos'] = tuple(it['pos'][a] + offsets[a] for a in range(3))
            if pcb_block:
                pcb_block['pos'] = tuple(pcb_block['pos'][a] + offsets[a] for a in range(3))

    return items, pcb_block


# ============================================================
#  Plotly 3D 构建 (Mesh3d 实体面 + Scatter3d 线框边框)
# ============================================================

_CUBE_TRIS = np.array([
    [0,1,2], [0,2,3], [4,5,6], [4,6,7],
    [0,5,1], [0,4,5], [2,7,3], [2,6,7],
    [1,5,6], [1,6,2], [0,3,7], [0,7,4],
], dtype=int)

def box_vertices(x, y, z, dx, dy, dz):
    return np.array([
        [x, y, z], [x+dx, y, z], [x+dx, y+dy, z], [x, y+dy, z],
        [x, y, z+dz], [x+dx, y, z+dz], [x+dx, y+dy, z+dz], [x, y+dy, z+dz],
    ])

def box_wireframe(x, y, z, dx, dy, dz):
    v = box_vertices(x, y, z, dx, dy, dz)
    pairs = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    xs, ys, zs = [], [], []
    for i, j in pairs:
        xs.extend([v[i,0], v[j,0], None])
        ys.extend([v[i,1], v[j,1], None])
        zs.extend([v[i,2], v[j,2], None])
    return xs, ys, zs

def add_box(fig, x, y, z, dx, dy, dz, facecolor, opacity, name, showlegend=False):
    """Mesh3d 实体面 + Scatter3d 线框边框。"""
    v = box_vertices(x, y, z, dx, dy, dz)
    # 实体面
    fig.add_trace(go.Mesh3d(
        x=v[:,0], y=v[:,1], z=v[:,2],
        i=_CUBE_TRIS[:,0], j=_CUBE_TRIS[:,1], k=_CUBE_TRIS[:,2],
        facecolor=[facecolor] * 12, opacity=opacity,
        name=name, showlegend=showlegend, hoverinfo='name',
    ))
    # 线框边框
    xs, ys, zs = box_wireframe(x, y, z, dx, dy, dz)
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs, mode='lines',
        line=dict(color='black', width=0.8),
        showlegend=False, hoverinfo='skip',
    ))

def add_compartment_wireframe(fig, l, w, h, color='gray'):
    xs, ys, zs = box_wireframe(0, 0, 0, l, w, h)
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs, mode='lines',
        line=dict(color=color, width=2.5, dash='dash'),
        showlegend=False, hoverinfo='skip', name='Compartment'
    ))

def build_3d_figure(compartment, items, pcb_block, margin_info, results_text,
                     raw_comp=(0,0,0), comp_color='gray', can_fit=True,
                     highlight_face=None, insulation_faces=None,
                     center_offset=(0, 0, 0)):
    cl, cw, ch = compartment
    fig = go.Figure()
    add_compartment_wireframe(fig, cl, cw, ch, color=comp_color)

    viridis = [
        '#440154','#482878','#3e4989','#31688e','#26828e',
        '#1f9e89','#35b779','#6ece58','#b5de2b','#fde725',
    ]
    n_batt = sum(1 for it in items if it['kind'] == 'battery')

    for i, it in enumerate(items):
        x, y, z = it['pos']
        dx, dy, dz = it['dims']
        color = viridis[i * len(viridis) // max(1, n_batt)]
        add_box(fig, x, y, z, dx, dy, dz, color, 0.55, f'Batt #{i+1}')

    if pcb_block:
        x, y, z = pcb_block['pos']
        dx, dy, dz = pcb_block['dims']
        # PCB: 更明显的红色实体 + 粗边框
        add_box(fig, x, y, z, dx, dy, dz, 'crimson', 0.55, 'Protection Board', showlegend=True)
        # 额外加粗红色边框 (覆盖默认黑框)
        xs, ys, zs = box_wireframe(x, y, z, dx, dy, dz)
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode='lines',
            line=dict(color='darkred', width=4),
            showlegend=False, hoverinfo='skip',
        ))

    # 高亮当前选中暴露面: 半透明金色矩形
    if highlight_face and highlight_face.get('components'):
        comps = highlight_face['components']
        axis = highlight_face['axis']
        wall_pos = highlight_face['wall_pos']
        u_axis, v_axis = [i for i in range(3) if i != axis]

        u_min = min(c['u'] for c in comps)
        u_max = max(c['u'] + c['w'] for c in comps)
        v_min = min(c['v'] for c in comps)
        v_max = max(c['v'] + c['h'] for c in comps)
        u_size = u_max - u_min
        v_size = v_max - v_min

        # 应用居中偏移
        wall_pos += center_offset[axis]
        u_min += center_offset[u_axis]
        v_min += center_offset[v_axis]

        hl_thin = 0.3  # 高亮面厚度 (纯视觉)
        x_hl = y_hl = z_hl = 0.0
        dx_hl = dy_hl = dz_hl = 0.0

        if axis == 0:
            x_hl, y_hl, z_hl = wall_pos - hl_thin / 2, u_min, v_min
            dx_hl, dy_hl, dz_hl = hl_thin, u_size, v_size
        elif axis == 1:
            x_hl, y_hl, z_hl = u_min, wall_pos - hl_thin / 2, v_min
            dx_hl, dy_hl, dz_hl = u_size, hl_thin, v_size
        else:
            x_hl, y_hl, z_hl = u_min, v_min, wall_pos - hl_thin / 2
            dx_hl, dy_hl, dz_hl = u_size, v_size, hl_thin

        v_hl = box_vertices(x_hl, y_hl, z_hl, dx_hl, dy_hl, dz_hl)
        fig.add_trace(go.Mesh3d(
            x=v_hl[:, 0], y=v_hl[:, 1], z=v_hl[:, 2],
            i=_CUBE_TRIS[:, 0], j=_CUBE_TRIS[:, 1], k=_CUBE_TRIS[:, 2],
            facecolor=['rgba(255,215,0,0.25)'] * 12, opacity=0.35,
            name='Selected Face', showlegend=True, hoverinfo='name',
        ))
        xs_hl, ys_hl, zs_hl = box_wireframe(x_hl, y_hl, z_hl, dx_hl, dy_hl, dz_hl)
        fig.add_trace(go.Scatter3d(
            x=xs_hl, y=ys_hl, z=zs_hl, mode='lines',
            line=dict(color='gold', width=3),
            showlegend=False, hoverinfo='skip',
        ))

    # 绝缘片可视化: 每个面一个 trace (图例中点击可独立开关)
    if insulation_faces:
        insul_colors = [
            'rgba(255,140,0,0.4)', 'rgba(0,180,200,0.4)', 'rgba(180,0,220,0.4)',
            'rgba(50,200,50,0.4)', 'rgba(220,80,80,0.4)', 'rgba(80,80,220,0.4)',
            'rgba(200,180,0,0.4)', 'rgba(200,100,180,0.4)', 'rgba(0,150,100,0.4)',
            'rgba(150,100,50,0.4)', 'rgba(100,100,100,0.4)', 'rgba(200,150,100,0.4)',
        ]
        border_colors = [
            'darkorange', 'darkcyan', 'darkorchid',
            'darkgreen', 'firebrick', 'darkblue',
            'darkgoldenrod', 'deeppink', 'teal',
            'saddlebrown', 'dimgray', 'chocolate',
        ]
        for fi, face in enumerate(insulation_faces):
            comps = face.get('components', [])
            if not comps:
                continue
            axis = face['axis']
            wall_pos = face['wall_pos'] + center_offset[axis]
            ua, va = [i for i in range(3) if i != axis]
            fc = insul_colors[fi % len(insul_colors)]
            bc = border_colors[fi % len(border_colors)]
            label = face.get('label', f'Face {fi+1}')
            for ci, comp in enumerate(comps):
                u = comp['u'] + center_offset[ua]
                v = comp['v'] + center_offset[va]
                w, h = comp['w'], comp['h']
                x_i = y_i = z_i = 0.0
                dx_i = dy_i = dz_i = 0.0
                thick = 0.5
                if axis == 0:
                    x_i, y_i, z_i = wall_pos - thick / 2, u, v
                    dx_i, dy_i, dz_i = thick, w, h
                elif axis == 1:
                    x_i, y_i, z_i = u, wall_pos - thick / 2, v
                    dx_i, dy_i, dz_i = w, thick, h
                else:
                    x_i, y_i, z_i = u, v, wall_pos - thick / 2
                    dx_i, dy_i, dz_i = w, h, thick

                vi = box_vertices(x_i, y_i, z_i, dx_i, dy_i, dz_i)
                fig.add_trace(go.Mesh3d(
                    x=vi[:, 0], y=vi[:, 1], z=vi[:, 2],
                    i=_CUBE_TRIS[:, 0], j=_CUBE_TRIS[:, 1], k=_CUBE_TRIS[:, 2],
                    facecolor=[fc] * 12, opacity=0.5,
                    name=label, showlegend=(ci == 0),
                    hoverinfo='name', legendgroup=label,
                ))
                xs_i, ys_i, zs_i = box_wireframe(x_i, y_i, z_i, dx_i, dy_i, dz_i)
                fig.add_trace(go.Scatter3d(
                    x=xs_i, y=ys_i, z=zs_i, mode='lines',
                    line=dict(color=bc, width=2.5),
                    showlegend=False, hoverinfo='skip',
                    legendgroup=label,
                ))

    max_extent = max(cl, cw, ch)
    # 如果有原始尺寸(含 margin), 用原始尺寸当边界
    if raw_comp != (0,0,0):
        max_extent = max(max_extent, max(raw_comp))

    annotations = []
    if not can_fit:
        annotations.append(dict(
            text='⚠ 不满足', showarrow=False,
            x=0.5, y=0.5, z=0.5,
            xanchor='center', yanchor='middle',
            font=dict(size=48, color='rgba(220,38,38,0.35)'),
            xshift=0, yshift=0,
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X / Length (mm)', range=[0, max_extent * 1.1]),
            yaxis=dict(title='Y / Width (mm)',  range=[0, max_extent * 1.1]),
            zaxis=dict(title='Z / Height (mm)', range=[0, max_extent * 1.1]),
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2),
                       projection=dict(type='orthographic')),
            annotations=annotations,
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text=f"Battery Arrangement — {results_text}", font=dict(size=14)),
        showlegend=True, legend=dict(x=0.01, y=0.99),
    )
    return fig

def build_2d_figure(compartment, items, pcb_block, comp_color='gray'):
    """构建 2D 三视图 (俯视/正视/侧视)。"""
    cl, cw, ch = compartment
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=('Top View (X-Y)', 'Front View (X-Z)', 'Side View (Y-Z)'))

    views = [
        (0, 1, cl, cw, 'Length', 'Width', 1, 1),
        (0, 2, cl, ch, 'Length', 'Height', 1, 2),
        (1, 2, cw, ch, 'Width', 'Height', 1, 3),
    ]
    colors_2d = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
                 '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']

    for di, dj, lim_i, lim_j, xlab, ylab, row, col in views:
        # 电池
        for ci, it in enumerate(items):
            p = it['pos']; d = it['dims']
            fig.add_trace(go.Scatter(
                x=[p[di], p[di]+d[di], p[di]+d[di], p[di], p[di]],
                y=[p[dj], p[dj], p[dj]+d[dj], p[dj]+d[dj], p[dj]],
                mode='lines', fill='toself',
                line=dict(color='black', width=0.8),
                fillcolor=colors_2d[ci % len(colors_2d)], opacity=0.5,
                showlegend=False, hoverinfo='skip',
            ), row=row, col=col)

        # 保护板 — 红色加粗边框 + 填充
        if pcb_block:
            p = pcb_block['pos']; d = pcb_block['dims']
            fig.add_trace(go.Scatter(
                x=[p[di], p[di]+d[di], p[di]+d[di], p[di], p[di]],
                y=[p[dj], p[dj], p[dj]+d[dj], p[dj]+d[dj], p[dj]],
                mode='lines', fill='toself',
                line=dict(color='darkred', width=2.5),
                fillcolor='rgba(220,20,60,0.45)',
                showlegend=False, hoverinfo='name', name='PCB',
            ), row=row, col=col)

        # 仓边界
        fig.add_trace(go.Scatter(
            x=[0, lim_i, lim_i, 0, 0], y=[0, 0, lim_j, lim_j, 0],
            mode='lines', line=dict(color=comp_color, width=2, dash='dash'),
            showlegend=False, hoverinfo='skip',
        ), row=row, col=col)

        fig.update_xaxes(title_text=xlab, range=[0, lim_i * 1.08], row=row, col=col)
        fig.update_yaxes(title_text=ylab, range=[0, lim_j * 1.08], row=row, col=col)

    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=40, b=10),
        title=dict(text="2D Projections", font=dict(size=13)),
    )
    return fig


# ============================================================
#  Dash App
# ============================================================

# ============================================================
#  参数预设
# ============================================================

PRESET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'presets.json')

def load_presets():
    if not os.path.exists(PRESET_FILE):
        return {}
    try:
        with open(PRESET_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

def save_presets(presets):
    with open(PRESET_FILE, 'w') as f:
        json.dump(presets, f, indent=2, ensure_ascii=False)

app = dash.Dash(__name__, title="Battery Fitting Calculator")
app._favicon = None

# --- 样式常量 ---
INPUT_STYLE = {
    'width': '100%', 'marginBottom': '2px', 'fontSize': '13px',
}
LABEL_STYLE = {
    'fontSize': '12px', 'fontWeight': 'bold', 'marginTop': '6px', 'color': '#333',
}
SECTION_STYLE = {
    'background': '#f8f9fa', 'padding': '10px 14px', 'borderRadius': '8px',
    'marginBottom': '10px', 'border': '1px solid #dee2e6',
}


def labeled_input(label, id_, value, min_val=0, max_val=500, step=0.5):
    return html.Div([
        html.Label(label, style=LABEL_STYLE),
        dcc.Input(id=id_, type='number', value=value, min=min_val, max=max_val,
                  step=step, style=INPUT_STYLE),
    ])

def labeled_slider(label, id_, value, min_val, max_val, step, unit=''):
    return html.Div([
        html.Label(f"{label}: {value}{unit}", id=f"{id_}-label", style=LABEL_STYLE),
        dcc.Slider(id=id_, min=min_val, max=max_val, step=step, value=value,
                   marks=None,
                   tooltip={"placement": "bottom", "always_visible": True}),
    ])

app.layout = html.Div([
    # 标题栏
    html.Div([
        html.H2("电池适配计算器 / Battery Fitting Calculator",
                style={'margin': '0', 'color': '#1a1a2e'}),
        html.Span("拖动滑块实时调整参数，3D 视图即时更新",
                  style={'color': '#666', 'fontSize': '13px'}),
    ], style={'padding': '12px 20px', 'borderBottom': '2px solid #4a90d9',
              'background': 'linear-gradient(135deg, #e8f0fe, #f0f4ff)'}),

    html.Div([
        # --- 隐藏存储 ---
        dcc.Store(id='preset-data', storage_type='memory'),
        dcc.Store(id='fit-cache', storage_type='memory'),

        # --- 左侧控制面板 ---
        html.Div([
            # 参数预设
            html.Div([
                html.Div([
                    html.Label("参数预设 Presets", style={'fontSize': '12px', 'fontWeight': 'bold',
                                                         'color': '#2c3e50'}),
                ]),
                html.Div([
                    dcc.Input(id='preset-name', type='text', placeholder='预设名称...',
                              style={'flex': '1', 'fontSize': '12px', 'padding': '4px 6px',
                                     'height': '28px', 'marginRight': '4px'}),
                    html.Button('💾 保存', id='btn-save', n_clicks=0,
                                style={'fontSize': '11px', 'padding': '2px 8px', 'height': '28px',
                                       'background': '#28a745', 'color': '#fff', 'border': 'none',
                                       'borderRadius': '4px', 'cursor': 'pointer'}),
                ], style={'display': 'flex', 'marginBottom': '4px'}),
                html.Div([
                    dcc.Dropdown(id='preset-select', options=[], placeholder='选择预设...',
                                 clearable=False, style={'flex': '1', 'fontSize': '12px'},
                                 optionHeight=30),
                    html.Button('📂 加载', id='btn-load', n_clicks=0,
                                style={'fontSize': '11px', 'padding': '2px 6px', 'height': '28px',
                                       'marginLeft': '4px', 'background': '#007bff', 'color': '#fff',
                                       'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer'}),
                    html.Button('🗑', id='btn-delete', n_clicks=0,
                                style={'fontSize': '11px', 'padding': '2px 6px', 'height': '28px',
                                       'marginLeft': '2px', 'background': '#dc3545', 'color': '#fff',
                                       'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer'}),
                ], style={'display': 'flex', 'alignItems': 'center'}),
                html.Div(id='preset-msg', style={'fontSize': '11px', 'marginTop': '2px',
                                                 'color': '#28a745'}),
            ], style={'background': '#e8f5e9', 'padding': '8px 10px', 'borderRadius': '8px',
                      'border': '1px solid #a5d6a7', 'marginBottom': '10px'}),

            # 电池仓
            html.Div([
                html.H4("电池仓 Compartment", style={'margin': '0 0 6px 0', 'color': '#2c3e50'}),
                html.Div([
                    html.Div([labeled_input("长 L (mm)", "comp-l", 80, 1, 2000, 0.5)],
                             style={'flex': '1', 'marginRight': '6px'}),
                    html.Div([labeled_input("宽 W (mm)", "comp-w", 60, 1, 2000, 0.5)],
                             style={'flex': '1', 'marginRight': '6px'}),
                    html.Div([labeled_input("高 H (mm)", "comp-h", 80, 1, 2000, 0.5)],
                             style={'flex': '1'}),
                ], style={'display': 'flex'}),
                html.Div([
                    html.Div([labeled_input("壁间隙 Margin (mm)", "margin", 1, 0, 100, 0.5)],
                             style={'flex': '1', 'marginRight': '6px'}),
                    html.Div([labeled_input("电池间隙 Gap (mm)", "batt-gap", 1, 0, 100, 0.5)],
                             style={'flex': '1'}),
                ], style={'display': 'flex', 'marginTop': '2px'}),
            ], style=SECTION_STYLE),

            # 电池参数
            html.Div([
                html.H4("单节电池 Battery", style={'margin': '0 0 6px 0', 'color': '#2c3e50'}),
                html.Div([
                    html.Div([labeled_input("长 L (mm)", "batt-l", 18, 0.5, 500, 0.5)],
                             style={'flex': '1', 'marginRight': '6px'}),
                    html.Div([labeled_input("宽 W (mm)", "batt-w", 18, 0.5, 500, 0.5)],
                             style={'flex': '1', 'marginRight': '6px'}),
                    html.Div([labeled_input("高 H (mm)", "batt-h", 65, 0.5, 500, 0.5)],
                             style={'flex': '1'}),
                ], style={'display': 'flex'}),
                html.Div([
                    html.Div([labeled_input("电压 V (V)", "batt-v", 3.7, 0, 100, 0.1)],
                             style={'flex': '1', 'marginRight': '6px'}),
                    html.Div([labeled_input("容量 Ah (Ah)", "batt-ah", 2.5, 0, 1000, 0.1)],
                             style={'flex': '1'}),
                ], style={'display': 'flex', 'marginTop': '2px'}),
            ], style=SECTION_STYLE),

            # 目标需求
            html.Div([
                html.H4("目标需求 Target", style={'margin': '0 0 6px 0', 'color': '#2c3e50'}),
                html.Div([
                    html.Div([labeled_input("目标电压 V (V)", "target-v", 14.8, 0, 1000, 0.1)],
                             style={'flex': '1', 'marginRight': '6px'}),
                    html.Div([labeled_input("目标容量 Ah (Ah)", "target-ah", 5.0, 0, 10000, 0.1)],
                             style={'flex': '1'}),
                ], style={'display': 'flex'}),
            ], style=SECTION_STYLE),

            # 保护板
            html.Div([
                html.H4("保护板 PCB", style={'margin': '0 0 6px 0', 'color': '#2c3e50'}),
                html.Div([
                    html.Div([labeled_input("长 L (mm)", "pcb-l", 50, 0, 500, 0.5)],
                             style={'flex': '1', 'marginRight': '6px'}),
                    html.Div([labeled_input("宽 W (mm)", "pcb-w", 30, 0, 500, 0.5)],
                             style={'flex': '1', 'marginRight': '6px'}),
                    html.Div([labeled_input("厚 H (mm)", "pcb-h", 3, 0, 100, 0.5)],
                             style={'flex': '1'}),
                ], style={'display': 'flex'}),
                html.Div([
                    html.Label("PCB 贴附面 (电池哪个面)", style={'fontSize': '11px', 'fontWeight': 'bold',
                                                              'marginTop': '4px', 'color': '#333'}),
                    dcc.Dropdown(
                        id='pcb-face', options=[
                            {'label': '自动 (最佳)', 'value': 'auto'},
                            {'label': '电池顶面 (↑ H轴远端)', 'value': '2-0'},
                            {'label': '电池底面 (↓ H轴近端)', 'value': '2-1'},
                            {'label': '电池前面 (→ W轴远端)', 'value': '1-0'},
                            {'label': '电池后面 (← W轴近端)', 'value': '1-1'},
                            {'label': '电池右面 (→ L轴远端)', 'value': '0-0'},
                            {'label': '电池左面 (← L轴近端)', 'value': '0-1'},
                        ], value='auto', clearable=False, style={'fontSize': '11px'},
                    ),
                ], style={'marginTop': '4px'}),
                html.Div([labeled_input("PCB 间隙 (mm)", "pcb-gap", 0, 0, 50, 0.5)],
                         style={'marginTop': '4px'}),
            ], style=SECTION_STYLE),

            # 绝缘材料 / BOM
            html.Div([
                html.H4("材料清单 BOM", style={'margin': '0 0 6px 0', 'color': '#2c3e50'}),
                html.Div([
                    labeled_input("绝缘 (元/cm²)", "material-cost", 0.05, 0, 10, 0.01),
                    labeled_input("PCB (元/片)", "pcb-price", 15, 0, 1000, 1),
                    labeled_input("电池 (元/节)", "batt-price", 8, 0, 1000, 1),
                ], style={'display': 'flex', 'gap': '4px'}),
                dcc.Checklist(
                    id='show-insulation',
                    options=[{'label': ' 可视化绝缘片', 'value': 'show'}],
                    value=[],
                    style={'fontSize': '12px', 'marginTop': '4px'},
                ),
                html.Div(id='insulation-panel', style={
                    'fontSize': '11px', 'fontFamily': 'monospace', 'whiteSpace': 'pre-line',
                    'minHeight': '20px', 'marginTop': '4px', 'color': '#555',
                }),
                html.Button('导出 BOM (CSV)', id='btn-export-bom',
                           style={'marginTop': '6px', 'fontSize': '11px', 'padding': '4px 10px'}),
                dcc.Download(id='download-bom'),
            ], style=SECTION_STYLE),

            # 排列方案选择
            html.Div([
                html.Label("排列方案 (点击切换)", style={'fontSize': '12px', 'fontWeight': 'bold',
                                                       'marginBottom': '4px', 'color': '#2c3e50'}),
                dcc.Dropdown(
                    id='arrangement-select',
                    options=[],
                    value=0,
                    clearable=False,
                    style={'fontSize': '12px'},
                    optionHeight=50,
                ),
            ], style={'background': '#fff3cd', 'padding': '8px 10px', 'borderRadius': '8px',
                      'border': '1px solid #ffc107', 'marginBottom': '10px'}),

            # 结果摘要
            html.Div(id='results-panel', style={
                'background': '#fff', 'padding': '12px 14px', 'borderRadius': '8px',
                'border': '1px solid #dee2e6', 'minHeight': '100px', 'fontSize': '13px',
                'fontFamily': 'monospace', 'whiteSpace': 'pre-line',
            }),
        ], style={'width': '340px', 'padding': '10px', 'overflowY': 'auto',
                  'maxHeight': 'calc(100vh - 70px)', 'background': '#f5f6fa'}),

        # --- 右侧可视化 ---
        html.Div([
            dcc.Loading([
                dcc.Graph(id='3d-graph', style={'height': '55vh'},
                          config={'scrollZoom': True, 'displayModeBar': True}),
                dcc.Graph(id='2d-graph', style={'height': '30vh'},
                          config={'displayModeBar': False}),
            ], type='circle'),
        ], style={'flex': '1', 'padding': '8px'}),
    ], style={'display': 'flex'}),
])


# ============================================================
#  回调
# ============================================================

def _error_fig(message):
    """生成带错误提示的空白 3D 图。"""
    fig = go.Figure()
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            annotations=[dict(
                text=message, showarrow=False,
                x=0.5, y=0.5, z=0.5,
                font=dict(size=22, color='red'),
            )]
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig

def _result_div(text, kind='info'):
    """生成带样式的结果面板。kind: 'ok', 'warn', 'error', 'info'"""
    colors = {
        'ok':    ('#d4edda', '#155724', '#c3e6cb'),
        'warn':  ('#fff3cd', '#856404', '#ffeeba'),
        'error': ('#f8d7da', '#721c24', '#f5c6cb'),
        'info':  ('#fff',    '#333',    '#dee2e6'),
    }
    bg, fg, border = colors.get(kind, colors['info'])
    font_size = '15px' if kind in ('error', 'warn') else '13px'
    font_weight = '600' if kind in ('error', 'warn') else '400'
    return html.Div(text, style={
        'background': bg, 'color': fg,
        'padding': '12px 14px', 'borderRadius': '8px',
        'border': f'2px solid {border}', 'minHeight': '100px',
        'fontSize': font_size, 'fontWeight': font_weight,
        'fontFamily': 'monospace', 'whiteSpace': 'pre-line',
    })

# ═══ 回调 A: 参数变化 → 重算排列方案 + 更新摘要文字 + 下拉菜单 ═══
@app.callback(
    Output('fit-cache', 'data'),
    Output('arrangement-select', 'options'),
    Output('arrangement-select', 'value'),
    Output('results-panel', 'children'),
    Output('pcb-face', 'options'),
    Output('pcb-face', 'value'),
    Input('comp-l', 'value'), Input('comp-w', 'value'), Input('comp-h', 'value'),
    Input('margin', 'value'),
    Input('batt-l', 'value'), Input('batt-w', 'value'), Input('batt-h', 'value'),
    Input('batt-gap', 'value'),
    Input('batt-v', 'value'), Input('batt-ah', 'value'),
    Input('target-v', 'value'), Input('target-ah', 'value'),
    Input('pcb-l', 'value'), Input('pcb-w', 'value'), Input('pcb-h', 'value'),
    Input('pcb-face', 'value'), Input('pcb-gap', 'value'),
    State('fit-cache', 'data'),
)
def compute_and_summarize(cl, cw, ch, margin, bl, bw, bh, bgap,
                           bv, bah, tv, tah,
                           pcb_l, pcb_w, pcb_h, pcb_face, pcb_gap,
                           old_cache):
    """物理拟合 + 电参数计算 → 更新 fit-cache、下拉菜单、结果摘要。"""

    noop_6 = [dash.no_update] * 6

    def fail(msg):
        return {'error': msg}, [], 0, _result_div('⚠ ' + msg, 'error'), [], 0

    try:
        cl, cw, ch = float(cl), float(cw), float(ch)
        margin = float(margin)
        bl, bw, bh = float(bl), float(bw), float(bh)
        bgap = float(bgap)
        bv, bah, tv, tah = float(bv), float(bah), float(tv), float(tah)
        pcb_l = float(pcb_l); pcb_w = float(pcb_w); pcb_h = float(pcb_h)
        pcb_gap_f = float(pcb_gap)
    except (TypeError, ValueError):
        return fail('请输入有效数字')

    if any(d <= 0 for d in [cl, cw, ch, bl, bw, bh, bv, bah, tv, tah]):
        return fail('参数必须 > 0')

    has_pcb = pcb_l > 0 and pcb_w > 0 and pcb_h > 0

    # ── 仅切换贴附面: 从缓存更新选中项 ──
    # 必须确认仅 pcb-face 变化 (加载预设时多个参数同时变化应走完整重算)
    triggered_ids = [t['prop_id'].split('.')[0] for t in ctx.triggered]
    only_face = (len(triggered_ids) == 1 and triggered_ids[0] == 'pcb-face')
    if (only_face and old_cache is not None
            and 'exposed_faces' in old_cache):
        c = old_cache
        faces = c['exposed_faces']  # 仅外表面
        idx = pcb_face if isinstance(pcb_face, (int, float)) else 0
        idx = max(0, min(int(idx), len(faces) - 1)) if faces else 0
        selected_face = faces[idx] if faces else None
        pcb_info = selected_face['pcb_info'] if selected_face else None
        new_cache = dict(c)
        new_cache['selected_face_idx'] = idx
        new_cache['pcb_info'] = pcb_info
        new_cache['pcb_near'] = pcb_info['near'] if pcb_info else False

        # 更新缓存并重建结果文本
        n_needed = c['n_needed']
        can_fit = c['all_arrangements'][0]['total'] >= n_needed
        face_opts = [{'label': f.get('label', f'面 {i+1}'), 'value': i} for i, f in enumerate(faces)]

        # 从旧缓存重建结果文本, 更新 PCB 面信息
        old_text = c.get('_results_text', '')
        if old_text and pcb_info:
            gap_str = f" 间隙{pcb_gap_f:.1f}mm" if pcb_gap_f > 0 else ""
            fd1, fd2 = pcb_info['face']
            new_pcb_line = (f"🛡 PCB: {pcb_l}×{pcb_w}×{pcb_h} mm"
                           f"  → 贴 {pcb_info['axis_name']} 面"
                           f"  厚度{pcb_info['thickness']:.1f}mm"
                           f"  贴面{fd1:.0f}×{fd2:.0f}mm{gap_str}")
            old_text = re.sub(r'🛡 PCB:.*', new_pcb_line, old_text)
        results_text = old_text
        new_cache['_results_text'] = results_text
        panel_kind = 'ok' if can_fit else 'error'
        return new_cache, dash.no_update, dash.no_update, \
               _result_div(results_text, panel_kind), face_opts, idx

    # ── 完整重算 ──
    raw_comp = (cl, cw, ch)
    compartment = (max(0.01, cl - 2*margin), max(0.01, cw - 2*margin), max(0.01, ch - 2*margin))

    batt_vol = bl * bw * bh
    comp_vol = compartment[0] * compartment[1] * compartment[2]
    batt_min, comp_min = min(bl, bw, bh), min(compartment)
    if batt_min > comp_min:
        return fail(f'电池最小边长({batt_min:.1f}) > 仓最小内径({comp_min:.1f})')

    all_arrangements = find_all_arrangements(compartment, bl, bw, bh, bgap)
    if not all_arrangements:
        return fail(f'电池({bl}×{bw}×{bh})太大, 一节也放不进')

    eff_compartment = compartment

    # ═══ 电参数 S×P (提前计算, 暴露面扫描需截断到实际用量) ═══
    s_needed = max(1, int(np.ceil(tv / bv)))
    p_needed = max(1, int(np.ceil(tah / bah)))
    n_needed = s_needed * p_needed

    # 暴露面扫描: 惰性计算 — 仅算最佳排列, 其余切换时按需计算
    pcb_pcb_l = pcb_l if has_pcb else 0
    pcb_pcb_w = pcb_w if has_pcb else 0
    pcb_pcb_h = pcb_h if has_pcb else 0

    def _compute_faces_for_arr(arr):
        arr_dict = {k: v for k, v in arr.items() if k not in ('total', 'label')}
        items_a, _ = generate_items(compartment, arr_dict, center=False)
        items_a = items_a[:n_needed]
        faces_a = find_all_exposed_faces(compartment, items_a,
                                         pcb_pcb_l, pcb_pcb_w, pcb_pcb_h, pcb_gap_f)
        exposed_a = [f for f in faces_a if f.get('exposed', True)
                     and f.get('pcb_info') is not None]
        return faces_a, exposed_a

    # 初始化 map (只算 index 0, 其余为 None 表示未计算)
    all_arr_faces_list = [None] * len(all_arrangements)
    all_arr_exposed_list = [None] * len(all_arrangements)
    all_faces, exposed_faces = _compute_faces_for_arr(all_arrangements[0])
    all_arr_faces_list[0] = all_faces
    all_arr_exposed_list[0] = exposed_faces

    face_idx = 0
    pcb_info = None

    if has_pcb:
        if not exposed_faces:
            return fail(f'保护板({pcb_l}×{pcb_w}×{pcb_h})在所有面上都放不下')

        if old_cache and 'exposed_faces' in old_cache:
            old_exposed = old_cache.get('exposed_faces', [])
            old_idx = old_cache.get('selected_face_idx', 0)
            if old_exposed and 0 <= old_idx < len(old_exposed):
                old_face = old_exposed[old_idx]
                old_key = (old_face.get('axis'), old_face.get('near'))
                for i, f in enumerate(exposed_faces):
                    if (f['axis'], f['near']) == old_key:
                        face_idx = i
                        break

        pcb_info = exposed_faces[face_idx].get('pcb_info')
        if pcb_info is None:
            for i, f in enumerate(exposed_faces):
                if f.get('pcb_info'):
                    pcb_info = f['pcb_info']
                    face_idx = i
                    break
    pcb_near = pcb_info['near'] if pcb_info else False

    # 过滤可行方案
    feasible = [a for a in all_arrangements if a['total'] >= n_needed]
    if feasible:
        all_arrangements = feasible

    best = all_arrangements[0]
    total = best['total']
    arrangement = {k: v for k, v in best.items() if k not in ('total', 'label')}

    can_fit = total >= n_needed
    vol_exceeded = batt_vol > comp_vol

    # --- 构建结果文本 (状态横幅置顶) ---
    if can_fit:
        p_actual = total // s_needed
        extra = f" (最大 {s_needed}S{p_actual}P)" if p_actual > p_needed else ""
        status_banner = (f"✅ 满足要求 — 需 {s_needed}S{p_needed}P = {n_needed} 节"
                         f"  |  可容纳 {total} 节{extra}")
    else:
        short = n_needed - total
        status_banner = (f"❌ 不满足 — 差 {short} 节  "
                         f"(需 {n_needed} 节达 {s_needed*bv:.1f}V {p_needed*bah:.1f}Ah)"
                         f"  |  仅可容纳 {total} 节")

    lines = [status_banner, "═" * 40]
    lines.append(
        f"📦 仓: {compartment[0]:.1f}×{compartment[1]:.1f}×{compartment[2]:.1f} mm"
        f"  (原始 {raw_comp[0]}×{raw_comp[1]}×{raw_comp[2]}, margin={margin})"
        f"  |  容积 {comp_vol:.0f} mm³"
    )
    lines.append(
        f"🔋 电池: {bl}×{bw}×{bh} mm  {bv}V {bah}Ah  "
        f"体积 {batt_vol:.0f} mm³  gap={bgap}mm"
    )
    if has_pcb and pcb_info:
        gap_str = f" 间隙{pcb_gap_f:.1f}mm" if pcb_gap_f > 0 else ""
        fd1, fd2 = pcb_info['face']
        lines.append(f"🛡 PCB: {pcb_l}×{pcb_w}×{pcb_h} mm"
                     f"  → 贴 {pcb_info['axis_name']} 面"
                     f"  厚度{pcb_info['thickness']:.1f}mm"
                     f"  贴面{fd1:.0f}×{fd2:.0f}mm{gap_str}")
    lines.append(
        f"⚡ 目标: {tv}V {tah}Ah  →  {s_needed}S{p_needed}P = {n_needed} 节  "
        f"|  可容纳: {total} 节"
    )
    if arrangement['type'] == 'single':
        o = arrangement['orient']; c = arrangement['counts']
        lines.append(f"排列: {o[0]:.1f}×{o[1]:.1f}×{o[2]:.1f} mm"
                     f"  →  {c[0]}×{c[1]}×{c[2]} = {total} 节")
    else:
        lines.append(f"排列: {best['label']}")
    if vol_exceeded:
        lines.append(f"⚠ 单节电池体积 ({batt_vol:.0f} mm³) > 仓容积 ({comp_vol:.0f} mm³)")
    eff_vol = eff_compartment[0] * eff_compartment[1] * eff_compartment[2]
    util = batt_vol * total / eff_vol * 100 if eff_vol > 0 else 0
    lines.append(f"空间利用率: {util:.1f}%")
    results_text = "\n".join(lines)

    # 序列化缓存 (不含 items — 由回调 B 按需生成)

    cache = {
        'all_arrangements': [{k: _tuplify(v) for k, v in a.items()} for a in all_arrangements],
        'compartment': list(compartment), 'raw_comp': list(raw_comp),
        'eff_compartment': list(eff_compartment),
        # 每个排列的暴露面 (切换排列时直接索引)
        'all_exposed_faces_map': [[_serialize_face(f) for f in fa] if fa is not None else None
                                  for fa in all_arr_faces_list],
        'exposed_faces_map': [[_serialize_face(f) for f in fa] if fa is not None else None
                               for fa in all_arr_exposed_list],
        # 当前选中排列的面 (默认 index 0)
        'all_exposed_faces': [_serialize_face(f) for f in all_faces],
        'exposed_faces': [_serialize_face(f) for f in exposed_faces],
        'selected_face_idx': face_idx if has_pcb else 0,
        'pcb_info': {k: _tuplify(v) for k, v in pcb_info.items()} if pcb_info else None,
        'margin': margin, 'bgap': bgap, 'has_pcb': has_pcb,
        'pcb_near': pcb_near,
        'pcb_dims_lazy': [pcb_l, pcb_w, pcb_h, pcb_gap_f],  # 惰性计算用
        'n_needed': n_needed, 'can_fit': can_fit, 'vol_exceeded': vol_exceeded,
        '_results_text': results_text,
    }

    dropdown_opts = []
    for i, a in enumerate(all_arrangements):
        label = a['label']
        if a['total'] > n_needed:
            label += f"  → 用 {n_needed} 节"
        dropdown_opts.append({'label': label, 'value': i})

    face_opts = []
    for i, f in enumerate(exposed_faces):
        face_opts.append({'label': f.get('label', f'面 {i+1}'), 'value': i})

    panel_kind = 'ok' if can_fit else ('error' if total < n_needed else 'warn')
    return (cache, dropdown_opts, 0, _result_div(results_text, panel_kind),
            face_opts, face_idx if has_pcb else 0)


# ═══ 回调 B: 排列方案切换 → 仅重建 3D/2D 图形 ═══
@app.callback(
    Output('3d-graph', 'figure'),
    Output('2d-graph', 'figure'),
    Input('fit-cache', 'data'),
    Input('arrangement-select', 'value'),
    Input('show-insulation', 'value'),
)
def render_selected(cache, arr_idx, show_insulation):
    """仅重建 3D/2D 图形 — 参数变化或切换排列方案时触发。"""
    empty = go.Figure()

    if cache is None:
        return empty, empty
    if 'error' in cache:
        return _error_fig(cache['error']), empty

    compartment = tuple(cache['compartment'])
    raw_comp = tuple(cache['raw_comp'])
    all_arrangements = cache['all_arrangements']
    pcb_info = cache['pcb_info']
    margin = cache['margin']
    has_pcb = cache['has_pcb']
    pcb_near = cache.get('pcb_near', False)
    n_needed = cache['n_needed']
    vol_exceeded = cache['vol_exceeded']

    if arr_idx is None or arr_idx >= len(all_arrangements):
        arr_idx = 0
    selected = all_arrangements[arr_idx]
    total = selected['total']
    arrangement = {k: v for k, v in selected.items() if k not in ('total', 'label')}

    items, pcb_block = generate_items(compartment, arrangement, pcb_info)

    # 计算居中偏移 (绝缘片可视化需要)
    items_raw, _ = generate_items(compartment, arrangement, center=False)
    center_offset = [0, 0, 0]
    if items_raw:
        max_ext = [0, 0, 0]
        for it in items_raw:
            for a in range(3):
                e = it['pos'][a] + it['dims'][a]
                if e > max_ext[a]:
                    max_ext[a] = e
        if pcb_block:
            for a in range(3):
                e = pcb_block['pos'][a] + pcb_block['dims'][a]
                if e > max_ext[a]:
                    max_ext[a] = e
        center_offset = tuple(max(0, (compartment[a] - max_ext[a]) / 2) for a in range(3))

    # 提取当前选中暴露面用于 3D 高亮 (按排列索引切换)
    highlight_face = None
    exposed_maps = cache.get('exposed_faces_map', [])
    if exposed_maps and 0 <= arr_idx < len(exposed_maps):
        exposed_faces_cache = exposed_maps[arr_idx]
        face_idx = cache.get('selected_face_idx', 0)
        if exposed_faces_cache and 0 <= face_idx < len(exposed_faces_cache):
            highlight_face = exposed_faces_cache[face_idx]

    can_fit = total >= n_needed
    status = "✓ 满足" if can_fit else "✗ 不满足"
    display_items = items[:n_needed] if can_fit else items

    if can_fit and total > n_needed:
        title = f"Showing {len(display_items)} / {total} batteries  |  {status}"
    elif can_fit:
        title = f"All {total} batteries  |  {status}"
    else:
        title = f"Fits: {total} (need {n_needed})  |  {status}"

    comp_color = 'red' if (not can_fit or vol_exceeded) else 'gray'
    # 绝缘片可视化 (总开关, 每个面在 3D 图例中独立开关)
    insulation_faces = None
    if show_insulation and 'show' in (show_insulation or []):
        insulation_faces = cache.get('all_exposed_faces', [])

    fig_3d = build_3d_figure(compartment, display_items, pcb_block,
                             f"(margin={margin})" if margin > 0 else "",
                             title, raw_comp,
                             comp_color=comp_color, can_fit=can_fit,
                             highlight_face=highlight_face,
                             insulation_faces=insulation_faces,
                             center_offset=center_offset)
    fig_2d = build_2d_figure(compartment, display_items, pcb_block, comp_color=comp_color)
    return fig_3d, fig_2d


# ═══ 回调 B2: 切换排列方案 → 更新面数据和下拉菜单 ═══
@app.callback(
    Output('fit-cache', 'data', allow_duplicate=True),
    Output('pcb-face', 'options', allow_duplicate=True),
    Output('pcb-face', 'value', allow_duplicate=True),
    Input('arrangement-select', 'value'),
    State('fit-cache', 'data'),
    prevent_initial_call=True,
)
def switch_arrangement_faces(arr_idx, cache):
    if cache is None or 'all_exposed_faces_map' not in cache:
        raise dash.exceptions.PreventUpdate
    arr_idx = arr_idx or 0
    all_maps = list(cache.get('all_exposed_faces_map', []))
    exposed_maps = list(cache.get('exposed_faces_map', []))
    if arr_idx >= len(all_maps) or arr_idx >= len(exposed_maps):
        raise dash.exceptions.PreventUpdate

    # 惰性计算: 如果该排列未计算过, 现在计算
    if all_maps[arr_idx] is None:
        compartment = tuple(cache['compartment'])
        arr = cache['all_arrangements'][arr_idx]
        pcb_p = cache.get('pcb_dims_lazy', [0, 0, 0, 0])
        pcb_lz, pcb_wz, pcb_hz, pcb_gz = pcb_p
        n = cache.get('n_needed', 0)

        arr_dict = {k: v for k, v in arr.items() if k not in ('total', 'label')}
        items_a, _ = generate_items(compartment, arr_dict, center=False)
        items_a = items_a[:n]
        faces_a = find_all_exposed_faces(compartment, items_a,
                                         pcb_lz, pcb_wz, pcb_hz, pcb_gz)
        exposed_a = [f for f in faces_a if f.get('exposed', True)
                     and f.get('pcb_info') is not None]

        all_maps[arr_idx] = [_serialize_face(f) for f in faces_a]
        exposed_maps[arr_idx] = [_serialize_face(f) for f in exposed_a]
        # 同时更新 cache 中的 map
        cache['all_exposed_faces_map'] = all_maps
        cache['exposed_faces_map'] = exposed_maps

    new_cache = dict(cache)
    new_cache['all_exposed_faces'] = all_maps[arr_idx]
    new_cache['exposed_faces'] = exposed_maps[arr_idx]

    new_exposed = exposed_maps[arr_idx]
    face_opts = [{'label': f.get('label', f'面 {i+1}'), 'value': i}
                 for i, f in enumerate(new_exposed)]

    old_idx = cache.get('selected_face_idx', 0)
    if old_idx >= len(new_exposed):
        old_idx = 0
    new_cache['selected_face_idx'] = old_idx

    if new_exposed and old_idx < len(new_exposed):
        new_cache['pcb_info'] = new_exposed[old_idx].get('pcb_info')
        if new_cache['pcb_info']:
            new_cache['pcb_near'] = new_cache['pcb_info'].get('near', False)

    return new_cache, face_opts, old_idx


# ═══ 回调 C: 绝缘材料面板更新 ═══
@app.callback(
    Output('insulation-panel', 'children'),
    Input('fit-cache', 'data'),
    Input('material-cost', 'value'),
)
def update_insulation_panel(cache, cost_per_cm2):
    """绝缘材料摘要。"""
    if cache is None or 'all_exposed_faces' not in cache:
        return "无绝缘信息"

    all_faces_cache = cache.get('all_exposed_faces', [])

    try:
        cost_per_cm2 = float(cost_per_cm2) if cost_per_cm2 else 0.05
    except (TypeError, ValueError):
        cost_per_cm2 = 0.05

    all_pieces = []
    for f in all_faces_cache:
        for c in f.get('components', []):
            all_pieces.append(c)

    total_area = sum(c['area'] for c in all_pieces)
    total_cm2 = total_area / 100.0

    return (f"绝缘片: {len(all_pieces)} 片  总面积: {total_area:.0f} mm² ({total_cm2:.2f} cm²)"
            f"  成本: ¥{total_cm2 * cost_per_cm2:.2f}")


# ═══ 回调 D: 导出 BOM (CSV) ═══
@app.callback(
    Output('download-bom', 'data'),
    Input('btn-export-bom', 'n_clicks'),
    State('fit-cache', 'data'),
    State('material-cost', 'value'),
    State('pcb-price', 'value'),
    State('batt-price', 'value'),
    prevent_initial_call=True,
)
def export_bom(n_clicks, cache, cost_per_cm2, pcb_price, batt_price):
    if cache is None:
        raise dash.exceptions.PreventUpdate
    try:
        cost_per_cm2 = float(cost_per_cm2) if cost_per_cm2 else 0.05
        pcb_price = float(pcb_price) if pcb_price else 15
        batt_price = float(batt_price) if batt_price else 8
    except (TypeError, ValueError):
        cost_per_cm2, pcb_price, batt_price = 0.05, 15, 8

    import io, csv
    from collections import defaultdict

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['类型', '尺寸 (mm)', '面积 (mm²)', '所属面', '数量', '单价', '小计'])

    total = 0
    n = cache.get('n_needed', 0)
    pcb_info = cache.get('pcb_info')
    arr = cache.get('all_arrangements', [{}])[0]
    orient = arr.get('orient') or arr.get('orient1') or (0, 0, 0)

    # 电池
    bsize = f'{orient[0]:.0f}×{orient[1]:.0f}×{orient[2]:.0f}'
    writer.writerow(['电池', bsize, '', '', n, f'{batt_price:.2f}', f'{n * batt_price:.2f}'])
    total += n * batt_price

    # PCB
    if pcb_info and cache.get('has_pcb'):
        d = pcb_info.get('dims', (0, 0, 0))
        psize = f'{d[0]:.0f}×{d[1]:.0f}×{d[2]:.0f}'
        writer.writerow(['保护板', psize, '', pcb_info.get('axis_name', ''), 1, f'{pcb_price:.2f}', f'{pcb_price:.2f}'])
        total += pcb_price

    # 绝缘片: 相同尺寸合并
    all_faces = cache.get('all_exposed_faces', [])
    insul_groups = defaultdict(lambda: {'count': 0, 'faces': set(), 'area': 0})
    for face in all_faces:
        for c in face.get('components', []):
            key = f'{c["w"]:.0f}×{c["h"]:.0f}'
            insul_groups[key]['count'] += 1
            insul_groups[key]['faces'].add(face.get('label', '?'))
            insul_groups[key]['area'] = c['area']

    for size, g in sorted(insul_groups.items()):
        area = g['area']
        piece_cost = (area / 100.0) * cost_per_cm2
        faces_str = ', '.join(sorted(g['faces']))
        writer.writerow(['绝缘片', size, f'{area:.0f}', faces_str,
                        g['count'], f'{piece_cost:.2f}', f'{piece_cost * g["count"]:.2f}'])
        total += piece_cost * g['count']

    writer.writerow([])
    writer.writerow(['', '', '', '', '', '总计', f'{total:.2f}'])

    content = output.getvalue().encode('utf-8')
    return dcc.send_bytes(b'\xef\xbb\xbf' + content, filename='battery_bom.csv')


# ============================================================
#  预设回调 (Save / Load / Delete)
# ============================================================

_PARAM_IDS = [
    'comp-l', 'comp-w', 'comp-h', 'margin',
    'batt-l', 'batt-w', 'batt-h', 'batt-v', 'batt-ah', 'batt-gap',
    'target-v', 'target-ah',
    'pcb-l', 'pcb-w', 'pcb-h', 'pcb-gap', 'pcb-face',
    'material-cost', 'pcb-price', 'batt-price',
]

_NUM_PARAM_IDS = len(_PARAM_IDS)

def _make_preset_outputs():
    """构建预设回调的 Output 列表."""
    return [Output('preset-select', 'options'),
            Output('preset-msg', 'children')] + \
           [Output(pid, 'value') for pid in _PARAM_IDS]

def _make_preset_states():
    """构建预设回调的 State 列表."""
    return [State(pid, 'value') for pid in _PARAM_IDS]

@app.callback(
    *_make_preset_outputs(),
    Input('btn-save', 'n_clicks'),
    Input('btn-load', 'n_clicks'),
    Input('btn-delete', 'n_clicks'),
    State('preset-name', 'value'),
    State('preset-select', 'value'),
    *_make_preset_states(),
)
def handle_presets(btn_save, btn_load, btn_delete, preset_name, preset_select, *args):
    triggered = ctx.triggered_id
    presets = load_presets()
    input_values = args  # all the input State values
    noop = [dash.no_update] * (2 + len(_PARAM_IDS))  # options + msg + all inputs

    # 构建当前选项列表
    def build_opts(plist):
        return [{'label': f'{k} ({v.get("batt-v","?")}V {v.get("batt-ah","?")}Ah '
                          f'→ {v.get("target-v","?")}V {v.get("target-ah","?")}Ah)',
                 'value': k}
                for k, v in plist.items()]

    # --- 保存 ---
    if triggered == 'btn-save':
        if not preset_name or not preset_name.strip():
            opts = build_opts(presets)
            return [opts, html.Span('⚠ 请输入预设名称', style={'color': '#dc3545'})] + noop[2:]
        name = preset_name.strip()
        presets[name] = {_PARAM_IDS[i]: input_values[i]
                         for i in range(len(_PARAM_IDS))}
        save_presets(presets)
        opts = build_opts(presets)
        return [opts, html.Span(f'✅ 已保存 "{name}"', style={'color': '#28a745'})] + noop[2:]

    # --- 加载 ---
    if triggered == 'btn-load':
        if not preset_select or preset_select not in presets:
            opts = build_opts(presets)
            return [opts, html.Span('⚠ 请选择预设', style={'color': '#dc3545'})] + noop[2:]
        p = presets[preset_select]
        vals = [p.get(pid, dash.no_update) for pid in _PARAM_IDS]
        opts = build_opts(presets)
        return [opts, html.Span(f'✅ 已加载 "{preset_select}"', style={'color': '#28a745'})] + vals

    # --- 删除 ---
    if triggered == 'btn-delete':
        if not preset_select or preset_select not in presets:
            opts = build_opts(presets)
            return [opts, html.Span('⚠ 请选择预设', style={'color': '#dc3545'})] + noop[2:]
        del presets[preset_select]
        save_presets(presets)
        opts = build_opts(presets)
        return [opts, html.Span(f'🗑 已删除 "{preset_select}"', style={'color': '#dc3545'})] + noop[2:]

    # --- 初始化 ---
    opts = build_opts(presets)
    return [opts, ''] + noop[2:]


if __name__ == '__main__':
    print("=" * 60)
    print("  Battery Fitting Calculator — Interactive Dash App")
    print("=" * 60)
    print()
    print("  浏览器打开 →  http://127.0.0.1:8050")
    print("  拖动左侧滑块实时调整参数，3D/2D 视图即时更新。")
    print()
    app.run(debug=False, host='127.0.0.1', port=8050)
