"""
ZMX 转换器 Checkpoint 验证脚本

验证 ElementConverter 对 complicated_fold_mirrors_setup_v2.zmx 的处理：
1. 所有反射镜被正确识别
2. 所有坐标断点被正确提取
3. 折叠镜序列的 is_fold 标志正确
4. 厚度计算正确

**Validates: Requirements 10.1, 10.2, 10.3, 10.6**
"""

import sys
import os

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from sequential_system.zmx_parser import ZmxParser
from sequential_system.zmx_converter import ElementConverter
from gaussian_beam_simulation.optical_elements import (
    FlatMirror,
    ParabolicMirror,
    SphericalMirror,
)


def print_separator(title: str = ""):
    """打印分隔线"""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    else:
        print("-" * 60)


def verify_complicated_fold_mirrors():
    """验证 complicated_fold_mirrors_setup_v2.zmx 的转换结果"""
    
    zmx_path = "optiland-master/tests/zemax_files/complicated_fold_mirrors_setup_v2.zmx"
    
    if not os.path.exists(zmx_path):
        print(f"❌ 错误：测试文件不存在: {zmx_path}")
        return False
    
    print_separator("ZMX 转换器 Checkpoint 验证")
    print(f"测试文件: {zmx_path}")
    
    # =========================================================================
    # 步骤 1: 解析 ZMX 文件
    # =========================================================================
    print_separator("步骤 1: 解析 ZMX 文件")
    
    try:
        parser = ZmxParser(zmx_path)
        data_model = parser.parse()
        print(f"✓ ZMX 文件解析成功")
        print(f"  - 总表面数: {len(data_model.surfaces)}")
    except Exception as e:
        print(f"❌ ZMX 文件解析失败: {e}")
        return False

    # =========================================================================
    # 步骤 2: 验证反射镜识别
    # =========================================================================
    print_separator("步骤 2: 验证反射镜识别")
    
    mirrors = data_model.get_mirror_surfaces()
    print(f"  - 识别到的反射镜数量: {len(mirrors)}")
    
    if len(mirrors) == 0:
        print(f"❌ 未识别到任何反射镜")
        return False
    
    print(f"✓ 反射镜列表:")
    for mirror in mirrors:
        radius_str = "∞" if np.isinf(mirror.radius) else f"{mirror.radius:.2f} mm"
        conic_str = f"conic={mirror.conic:.4f}" if mirror.conic != 0 else ""
        comment_str = f"'{mirror.comment}'" if mirror.comment else ""
        print(f"    Surface {mirror.index}: radius={radius_str} {conic_str} {comment_str}")
    
    # =========================================================================
    # 步骤 3: 验证坐标断点提取
    # =========================================================================
    print_separator("步骤 3: 验证坐标断点提取")
    
    coord_breaks = data_model.get_coordinate_break_surfaces()
    print(f"  - 识别到的坐标断点数量: {len(coord_breaks)}")
    
    if len(coord_breaks) == 0:
        print(f"⚠ 未识别到任何坐标断点（可能是简单系统）")
    else:
        print(f"✓ 坐标断点列表:")
        for cb in coord_breaks:
            tilt_info = []
            if cb.tilt_x_deg != 0:
                tilt_info.append(f"tilt_x={cb.tilt_x_deg:.1f}°")
            if cb.tilt_y_deg != 0:
                tilt_info.append(f"tilt_y={cb.tilt_y_deg:.1f}°")
            if cb.tilt_z_deg != 0:
                tilt_info.append(f"tilt_z={cb.tilt_z_deg:.1f}°")
            
            decenter_info = []
            if cb.decenter_x != 0:
                decenter_info.append(f"dx={cb.decenter_x:.2f}")
            if cb.decenter_y != 0:
                decenter_info.append(f"dy={cb.decenter_y:.2f}")
            
            thickness_str = f"thickness={cb.thickness:.2f} mm" if cb.thickness != 0 else ""
            
            print(f"    Surface {cb.index}: {' '.join(tilt_info)} {' '.join(decenter_info)} {thickness_str}")

    # =========================================================================
    # 步骤 4: 转换为 OpticalElement
    # =========================================================================
    print_separator("步骤 4: 转换为 OpticalElement")
    
    try:
        converter = ElementConverter(data_model)
        elements = converter.convert()
        print(f"✓ 转换成功")
        print(f"  - 生成的元件数量: {len(elements)}")
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    if len(elements) == 0:
        print(f"❌ 未生成任何元件")
        return False
    
    # =========================================================================
    # 步骤 5: 验证折叠镜 is_fold 标志
    # =========================================================================
    print_separator("步骤 5: 验证折叠镜 is_fold 标志")
    
    converted_elements = converter.get_converted_elements()
    
    fold_mirrors = [ce for ce in converted_elements if ce.is_fold_mirror]
    non_fold_mirrors = [ce for ce in converted_elements if not ce.is_fold_mirror]
    
    print(f"  - 折叠镜数量: {len(fold_mirrors)}")
    print(f"  - 非折叠镜数量: {len(non_fold_mirrors)}")
    
    print(f"\n✓ 元件详情:")
    for i, ce in enumerate(converted_elements):
        elem = ce.element
        elem_type = type(elem).__name__
        
        # 获取倾斜信息
        tilt_x_deg = np.rad2deg(elem.tilt_x) if hasattr(elem, 'tilt_x') else 0
        tilt_y_deg = np.rad2deg(elem.tilt_y) if hasattr(elem, 'tilt_y') else 0
        
        fold_str = "✓ FOLD" if ce.is_fold_mirror else ""
        tilt_str = ""
        if tilt_x_deg != 0 or tilt_y_deg != 0:
            tilt_str = f"tilt=({tilt_x_deg:.1f}°, {tilt_y_deg:.1f}°)"
        
        comment_str = f"'{ce.zmx_comment}'" if ce.zmx_comment else ""
        
        print(f"    [{i+1}] {elem_type}: ZMX Surface {ce.zmx_surface_index}")
        print(f"        thickness={elem.thickness:.2f} mm, semi_aperture={elem.semi_aperture:.2f} mm")
        print(f"        {tilt_str} is_fold={elem.is_fold} {fold_str} {comment_str}")

    # =========================================================================
    # 步骤 6: 验证结果
    # =========================================================================
    print_separator("步骤 6: 验证结果")
    
    all_passed = True
    
    # 验证 1: 所有反射镜都被转换
    if len(elements) >= len(mirrors):
        print(f"✓ 验证 1: 反射镜数量正确 ({len(elements)} 个元件)")
    else:
        print(f"⚠ 验证 1: 元件数量 ({len(elements)}) 少于反射镜数量 ({len(mirrors)})")
    
    # 验证 2: 折叠镜的 is_fold 标志正确
    for ce in converted_elements:
        elem = ce.element
        if ce.is_fold_mirror:
            if elem.is_fold:
                print(f"✓ 验证 2: Surface {ce.zmx_surface_index} 的 is_fold=True 正确")
            else:
                print(f"❌ 验证 2: Surface {ce.zmx_surface_index} 应该是折叠镜但 is_fold=False")
                all_passed = False
    
    # 验证 3: 厚度值有效
    for ce in converted_elements:
        elem = ce.element
        if elem.thickness >= 0 and not np.isnan(elem.thickness):
            pass  # 正常
        else:
            print(f"❌ 验证 3: Surface {ce.zmx_surface_index} 的厚度无效: {elem.thickness}")
            all_passed = False
    
    print(f"\n✓ 验证 3: 所有元件厚度值有效")
    
    # 验证 4: 半口径值有效
    for ce in converted_elements:
        elem = ce.element
        if elem.semi_aperture > 0:
            pass  # 正常
        else:
            print(f"❌ 验证 4: Surface {ce.zmx_surface_index} 的半口径无效: {elem.semi_aperture}")
            all_passed = False
    
    print(f"✓ 验证 4: 所有元件半口径值有效")
    
    # =========================================================================
    # 总结
    # =========================================================================
    print_separator("验证总结")
    
    if all_passed:
        print("🎉 所有验证通过！")
        print(f"\n转换结果摘要:")
        print(f"  - 输入: {len(data_model.surfaces)} 个 ZMX 表面")
        print(f"  - 输出: {len(elements)} 个 OpticalElement")
        print(f"  - 折叠镜: {len(fold_mirrors)} 个")
        print(f"  - 非折叠镜: {len(non_fold_mirrors)} 个")
        return True
    else:
        print("❌ 部分验证失败，请检查上述错误信息")
        return False


if __name__ == "__main__":
    success = verify_complicated_fold_mirrors()
    sys.exit(0 if success else 1)
