#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ZMX 解析器验证脚本

验证 one_mirror_up_45deg.zmx 文件的解析功能：
1. 验证反射镜被正确识别
2. 验证坐标断点被正确提取
3. 验证 45 度倾斜角被正确解析

作者：混合光学仿真项目
"""

import sys
import os
import numpy as np

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sequential_system.zmx_parser import ZmxParser, ZmxDataModel


def verify_one_mirror_up_45deg():
    """验证 one_mirror_up_45deg.zmx 文件的解析"""
    
    print("=" * 70)
    print("ZMX 解析器验证脚本")
    print("=" * 70)
    print()
    
    # 文件路径
    zmx_file = "optiland-master/tests/zemax_files/one_mirror_up_45deg.zmx"
    
    print(f"解析文件: {zmx_file}")
    print("-" * 70)
    
    # 创建解析器并解析文件
    parser = ZmxParser(zmx_file)
    data_model = parser.parse()
    
    # 打印基本信息
    print(f"\n【基本信息】")
    print(f"  表面总数: {data_model.get_surface_count()}")
    print(f"  入瞳直径: {data_model.entrance_pupil_diameter} mm")
    print(f"  波长数量: {len(data_model.wavelengths)}")
    if data_model.wavelengths:
        print(f"  主波长: {data_model.wavelengths[0]} μm")
    
    # 打印所有表面
    print(f"\n【所有表面】")
    for idx in sorted(data_model.surfaces.keys()):
        surface = data_model.surfaces[idx]
        print(f"  表面 {idx}: {surface.surface_type}")
        if surface.comment:
            print(f"    注释: {surface.comment}")
        if surface.radius != np.inf:
            print(f"    曲率半径: {surface.radius:.4f} mm")
        if surface.thickness != 0.0:
            print(f"    厚度: {surface.thickness:.4f} mm")
        if surface.is_mirror:
            print(f"    ★ 反射镜")
        if surface.is_stop:
            print(f"    ★ 光阑")
        if surface.semi_diameter > 0:
            print(f"    半口径: {surface.semi_diameter:.4f} mm")
    
    # 验证反射镜
    print(f"\n【反射镜验证】")
    mirrors = data_model.get_mirror_surfaces()
    print(f"  反射镜数量: {len(mirrors)}")
    
    if len(mirrors) == 1:
        print("  ✓ 正确识别了 1 个反射镜")
        mirror = mirrors[0]
        print(f"    表面索引: {mirror.index}")
        print(f"    注释: {mirror.comment}")
        print(f"    曲率半径: {mirror.radius}")
        print(f"    半口径: {mirror.semi_diameter:.4f} mm")
    else:
        print(f"  ✗ 预期 1 个反射镜，实际 {len(mirrors)} 个")
    
    # 验证坐标断点
    print(f"\n【坐标断点验证】")
    coord_breaks = data_model.get_coordinate_break_surfaces()
    print(f"  坐标断点数量: {len(coord_breaks)}")
    
    if len(coord_breaks) == 2:
        print("  ✓ 正确识别了 2 个坐标断点")
    else:
        print(f"  ✗ 预期 2 个坐标断点，实际 {len(coord_breaks)} 个")
    
    # 验证 45 度倾斜角
    print(f"\n【45 度倾斜角验证】")
    has_45_deg_tilt = False
    
    for cb in coord_breaks:
        print(f"  坐标断点 {cb.index}:")
        print(f"    decenter_x: {cb.decenter_x:.4f} mm")
        print(f"    decenter_y: {cb.decenter_y:.4f} mm")
        print(f"    tilt_x_deg: {cb.tilt_x_deg:.4f}°")
        print(f"    tilt_y_deg: {cb.tilt_y_deg:.4f}°")
        print(f"    tilt_z_deg: {cb.tilt_z_deg:.4f}°")
        print(f"    thickness: {cb.thickness:.4f} mm")
        
        # 检查是否有 45 度倾斜
        if abs(cb.tilt_x_deg - 45.0) < 0.001 or abs(cb.tilt_y_deg - 45.0) < 0.001:
            has_45_deg_tilt = True
            print(f"    ★ 检测到 45° 倾斜")
    
    if has_45_deg_tilt:
        print("  ✓ 正确解析了 45 度倾斜角")
    else:
        print("  ✗ 未检测到 45 度倾斜角")
    
    # 验证折叠镜序列
    print(f"\n【折叠镜序列验证】")
    # 预期序列：COORDBRK (45°) -> MIRROR -> COORDBRK (45°)
    
    # 找到反射镜前后的坐标断点
    if mirrors and coord_breaks:
        mirror_idx = mirrors[0].index
        pre_cb = None
        post_cb = None
        
        for cb in coord_breaks:
            if cb.index == mirror_idx - 1:
                pre_cb = cb
            elif cb.index == mirror_idx + 1:
                post_cb = cb
        
        if pre_cb and post_cb:
            print(f"  ✓ 检测到折叠镜序列:")
            print(f"    前坐标断点 (表面 {pre_cb.index}): tilt_x = {pre_cb.tilt_x_deg}°")
            print(f"    反射镜 (表面 {mirror_idx})")
            print(f"    后坐标断点 (表面 {post_cb.index}): tilt_x = {post_cb.tilt_x_deg}°, thickness = {post_cb.thickness} mm")
            
            # 验证后坐标断点的负厚度（表示反射方向传播）
            if post_cb.thickness < 0:
                print(f"    ✓ 后坐标断点厚度为负值 ({post_cb.thickness} mm)，表示反射方向传播")
            else:
                print(f"    注意: 后坐标断点厚度为正值或零")
        else:
            print("  ✗ 未检测到完整的折叠镜序列")
    
    # 总结
    print(f"\n{'=' * 70}")
    print("验证总结")
    print("=" * 70)
    
    all_passed = True
    
    # 检查 1: 反射镜数量
    if len(mirrors) == 1:
        print("✓ 反射镜识别: 通过")
    else:
        print("✗ 反射镜识别: 失败")
        all_passed = False
    
    # 检查 2: 坐标断点数量
    if len(coord_breaks) == 2:
        print("✓ 坐标断点识别: 通过")
    else:
        print("✗ 坐标断点识别: 失败")
        all_passed = False
    
    # 检查 3: 45 度倾斜角
    if has_45_deg_tilt:
        print("✓ 45 度倾斜角解析: 通过")
    else:
        print("✗ 45 度倾斜角解析: 失败")
        all_passed = False
    
    # 检查 4: 入瞳直径
    if data_model.entrance_pupil_diameter == 20.0:
        print("✓ 入瞳直径解析: 通过")
    else:
        print(f"✗ 入瞳直径解析: 失败 (预期 20.0, 实际 {data_model.entrance_pupil_diameter})")
        all_passed = False
    
    print()
    if all_passed:
        print("🎉 所有验证通过！ZMX 解析器功能正常。")
        return 0
    else:
        print("❌ 部分验证失败，请检查解析器实现。")
        return 1


if __name__ == "__main__":
    sys.exit(verify_one_mirror_up_45deg())
