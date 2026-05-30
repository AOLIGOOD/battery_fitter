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
                     raw_comp=(0,0,0), comp_color='gray', can_fit=True):
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
    if (ctx.triggered_id == 'pcb-face' and old_cache is not None
            and 'all_pcb_placements' in old_cache):
        c = old_cache
        placements = c['all_pcb_placements']
        idx = pcb_face if isinstance(pcb_face, (int, float)) else 0
        idx = max(0, min(int(idx), len(placements) - 1)) if placements else 0
        pcb_info = placements[idx] if placements else None
        new_cache = dict(c)
        new_cache['selected_pcb_idx'] = idx
        new_cache['pcb_info'] = pcb_info
        new_cache['pcb_near'] = pcb_info['near'] if pcb_info else False

        # 更新缓存并重建结果文本
        n_needed = c['n_needed']
        can_fit = c['all_arrangements'][0]['total'] >= n_needed
        face_opts = [{'label': p['label'], 'value': i} for i, p in enumerate(placements)]

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
    all_placements = []
    pcb_info = None
    if has_pcb:
        best_arr = {k: v for k, v in all_arrangements[0].items()
                    if k not in ('total', 'label')}
        items_for_pcb, _ = generate_items(compartment, best_arr, center=False)
        all_placements = find_all_pcb_placements(compartment, items_for_pcb,
                                                  pcb_l, pcb_w, pcb_h, pcb_gap_f)
        if not all_placements:
            return fail(f'保护板({pcb_l}×{pcb_w}×{pcb_h})在所有面上都放不下')
        pcb_info = all_placements[0]
        pcb_near = pcb_info['near']
    else:
        pcb_near = False

    # ═══ 电参数: S×P ═══
    s_needed = max(1, int(np.ceil(tv / bv)))
    p_needed = max(1, int(np.ceil(tah / bah)))
    n_needed = s_needed * p_needed

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
    def tuplify(v):
        if isinstance(v, (int, float, str, bool, type(None), list, dict)):
            return v
        return list(v)

    cache = {
        'all_arrangements': [{k: tuplify(v) for k, v in a.items()} for a in all_arrangements],
        'compartment': list(compartment), 'raw_comp': list(raw_comp),
        'eff_compartment': list(eff_compartment),
        'all_pcb_placements': [{k: tuplify(v) for k, v in p.items()} for p in all_placements],
        'selected_pcb_idx': 0,
        'pcb_info': {k: tuplify(v) for k, v in pcb_info.items()} if pcb_info else None,
        'margin': margin, 'bgap': bgap, 'has_pcb': has_pcb,
        'pcb_near': pcb_near,
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
    for i, p in enumerate(all_placements):
        face_opts.append({'label': p.get('label', f'面 {i+1}'), 'value': i})

    panel_kind = 'ok' if can_fit else ('error' if total < n_needed else 'warn')
    return (cache, dropdown_opts, 0, _result_div(results_text, panel_kind),
            face_opts, 0)


# ═══ 回调 B: 排列方案切换 → 仅重建 3D/2D 图形 ═══
@app.callback(
    Output('3d-graph', 'figure'),
    Output('2d-graph', 'figure'),
    Input('fit-cache', 'data'),
    Input('arrangement-select', 'value'),
)
def render_selected(cache, arr_idx):
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
    fig_3d = build_3d_figure(compartment, display_items, pcb_block,
                             f"(margin={margin})" if margin > 0 else "",
                             title, raw_comp,
                             comp_color=comp_color, can_fit=can_fit)
    fig_2d = build_2d_figure(compartment, display_items, pcb_block, comp_color=comp_color)
    return fig_3d, fig_2d


# ============================================================
#  预设回调 (Save / Load / Delete)
# ============================================================

_PARAM_IDS = [
    'comp-l', 'comp-w', 'comp-h', 'margin',
    'batt-l', 'batt-w', 'batt-h', 'batt-v', 'batt-ah', 'batt-gap',
    'target-v', 'target-ah',
    'pcb-l', 'pcb-w', 'pcb-h', 'pcb-gap', 'pcb-face',
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
        presets[name] = {
            'comp-l': input_values[0], 'comp-w': input_values[1], 'comp-h': input_values[2],
            'margin': input_values[3],
            'batt-l': input_values[4], 'batt-w': input_values[5], 'batt-h': input_values[6],
            'batt-v': input_values[7], 'batt-ah': input_values[8], 'batt-gap': input_values[9],
            'target-v': input_values[10], 'target-ah': input_values[11],
            'pcb-l': input_values[12], 'pcb-w': input_values[13], 'pcb-h': input_values[14],
            'pcb-gap': input_values[15], 'pcb-face': input_values[16],
        }
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
