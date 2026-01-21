#!/usr/bin/env python
"""
坐标转换验证脚本

本脚本用于验证 Zemax 光轴追踪与坐标转换功能的正确性。
可以加载测试 ZMX 文件，输出转换后的全局坐标表面定义，
并与 Zemax 导出数据进行对比。

使用方法：
    python scripts/verify_coordinate_conversion.py [zmx_file_path]

如果不提供 ZMX 文件路径，将使用内置的测试数据进行验证。

作者：混合光学仿真项目
"""

import sys
import os

# 将 src 目录添加到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from sequential_system.zmx_parser import ZmxSurfaceData, ZmxDataModel
from sequential_system.coordinate_system import (
    CurrentCoordinateSystem,
    CoordinateBreakProcessor,
    GlobalSurfaceDefinition,
    SurfaceTraversalAlgorithm,
    ZemaxToOptilandConverter,
)


def print_separator(title: str = ""):
    """打印分隔线"""
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
    else:
        print("-" * 60)


def verify_45_degree_rotation():
    """验证 45 度旋转"""
    print_separator("验证 45 度旋转")
    
    cs = CurrentCoordinateSystem.identity()
    print(f"初始 Z 轴: {cs.z_axis}")
    
    # 绕 X 轴旋转 45 度
    angle = np.deg2rad(45.0)
    cs_rotated = cs.apply_rotation(angle, 0, 0)
    print(f"旋转 45° 后 Z 轴: {cs_rotated.z_axis}")
    
    # 预期 Z 轴
    expected_z = np.array([0, -np.sin(angle), np.cos(angle)])
    print(f"预期 Z 轴: {expected_z}")
    
    if np.allclose(cs_rotated.z_axis, expected_z, rtol=1e-10):
        print("✓ 45 度旋转验证通过！")
        return True
    else:
        print("✗ 45 度旋转验证失败！")
        return False


def verify_coordinate_break_order():
    """验证坐标断点 Order 参数"""
    print_separator("验证坐标断点 Order 参数")
    
    cs = CurrentCoordinateSystem.identity()
    dx, dy = 10.0, 0.0
    tilt_y = np.deg2rad(45.0)
    
    # Order=0: 先平移后旋转
    cs_order_0 = CoordinateBreakProcessor.process(
        cs, dx, dy, 0, tilt_y, 0, order=0, thickness=0
    )
    print(f"Order=0 原点: {cs_order_0.origin}")
    
    # Order=1: 先旋转后平移
    cs_order_1 = CoordinateBreakProcessor.process(
        cs, dx, dy, 0, tilt_y, 0, order=1, thickness=0
    )
    print(f"Order=1 原点: {cs_order_1.origin}")
    
    # 验证两者不同
    if not np.allclose(cs_order_0.origin, cs_order_1.origin):
        print("✓ Order=0 和 Order=1 产生不同结果，验证通过！")
        return True
    else:
        print("✗ Order=0 和 Order=1 结果相同，验证失败！")
        return False


def verify_z_shape_system():
    """验证 Z 形双镜系统"""
    print_separator("验证 Z 形双镜系统")
    
    # 创建测试数据模型
    zmx_data = ZmxDataModel()
    
    # 第一个坐标断点：绕 X 轴旋转 45 度
    zmx_data.surfaces[0] = ZmxSurfaceData(
        index=0,
        surface_type='coordinate_break',
        tilt_x_deg=45.0,
        thickness=0.0
    )
    
    # 第一个平面反射镜
    zmx_data.surfaces[1] = ZmxSurfaceData(
        index=1,
        surface_type='standard',
        radius=np.inf,
        thickness=100.0,
        is_mirror=True,
        semi_diameter=25.0,
        comment='M1'
    )
    
    # 第二个坐标断点：再绕 X 轴旋转 45 度
    zmx_data.surfaces[2] = ZmxSurfaceData(
        index=2,
        surface_type='coordinate_break',
        tilt_x_deg=45.0,
        thickness=0.0
    )
    
    # 第二个平面反射镜
    zmx_data.surfaces[3] = ZmxSurfaceData(
        index=3,
        surface_type='standard',
        radius=np.inf,
        thickness=50.0,
        is_mirror=True,
        semi_diameter=25.0,
        comment='M2'
    )
    
    # 遍历
    traversal = SurfaceTraversalAlgorithm(zmx_data)
    global_surfaces = traversal.traverse()
    
    print(f"生成 {len(global_surfaces)} 个全局坐标表面定义:")
    for surface in global_surfaces:
        print(f"  表面 {surface.index} ({surface.comment}):")
        print(f"    顶点: {surface.vertex_position}")
        print(f"    Z轴: {surface.orientation[:, 2]}")
    
    # 验证最终 Z 轴方向
    final_cs = traversal.current_coordinate_system
    expected_final_z = np.array([0, -1, 0])
    print(f"\n最终坐标系 Z 轴: {final_cs.z_axis}")
    print(f"预期 Z 轴: {expected_final_z}")
    
    if np.allclose(final_cs.z_axis, expected_final_z, atol=1e-10):
        print("✓ Z 形双镜系统验证通过！")
        return True
    else:
        print("✗ Z 形双镜系统验证失败！")
        return False


def verify_curvature_center():
    """验证曲率中心计算"""
    print_separator("验证曲率中心计算")
    
    # 测试正半径
    surface_pos = GlobalSurfaceDefinition(
        index=1,
        surface_type='standard',
        vertex_position=np.array([0, 0, 0]),
        orientation=np.eye(3),
        radius=100.0
    )
    print(f"正半径 (R=100): 曲率中心 = {surface_pos.curvature_center}")
    
    # 测试负半径
    surface_neg = GlobalSurfaceDefinition(
        index=2,
        surface_type='standard',
        vertex_position=np.array([0, 0, 0]),
        orientation=np.eye(3),
        radius=-100.0
    )
    print(f"负半径 (R=-100): 曲率中心 = {surface_neg.curvature_center}")
    
    # 测试旋转后的曲率中心
    angle = np.deg2rad(45.0)
    R = CoordinateBreakProcessor.rotation_matrix_x(angle)
    orientation = np.eye(3) @ R
    
    surface_rotated = GlobalSurfaceDefinition(
        index=3,
        surface_type='standard',
        vertex_position=np.array([0, 0, 0]),
        orientation=orientation,
        radius=100.0
    )
    print(f"旋转 45° 后 (R=100): 曲率中心 = {surface_rotated.curvature_center}")
    
    # 验证
    expected_center_rotated = 100.0 * np.array([0, -np.sin(angle), np.cos(angle)])
    if np.allclose(surface_rotated.curvature_center, expected_center_rotated, atol=1e-10):
        print("✓ 曲率中心计算验证通过！")
        return True
    else:
        print("✗ 曲率中心计算验证失败！")
        return False


def verify_mirror_behavior():
    """验证反射镜不改变坐标系"""
    print_separator("验证反射镜不改变坐标系")
    
    zmx_data = ZmxDataModel()
    
    # 反射镜（无坐标断点）
    zmx_data.surfaces[0] = ZmxSurfaceData(
        index=0,
        surface_type='standard',
        radius=100.0,
        thickness=50.0,
        is_mirror=True
    )
    
    # 遍历
    traversal = SurfaceTraversalAlgorithm(zmx_data)
    global_surfaces = traversal.traverse()
    
    # 验证坐标系轴方向不变
    final_cs = traversal.current_coordinate_system
    print(f"最终坐标系轴:\n{final_cs.axes}")
    print(f"最终原点: {final_cs.origin}")
    
    if np.allclose(final_cs.axes, np.eye(3), atol=1e-10):
        print("✓ 反射镜不改变坐标系轴方向，验证通过！")
        return True
    else:
        print("✗ 反射镜改变了坐标系轴方向，验证失败！")
        return False


def load_and_verify_zmx_file(zmx_file_path: str):
    """加载并验证 ZMX 文件"""
    print_separator(f"加载 ZMX 文件: {zmx_file_path}")
    
    try:
        from sequential_system.zmx_parser import ZmxParser
        from sequential_system.coordinate_system import convert_zmx_to_global_surfaces
        
        # 解析 ZMX 文件
        parser = ZmxParser(zmx_file_path)
        zmx_data = parser.parse()
        
        print(f"解析成功！")
        print(f"  表面数量: {len(zmx_data.surfaces)}")
        print(f"  波长: {zmx_data.wavelengths} μm")
        print(f"  入瞳直径: {zmx_data.entrance_pupil_diameter} mm")
        
        # 转换为全局坐标表面定义
        global_surfaces = convert_zmx_to_global_surfaces(zmx_file_path)
        
        print(f"\n转换后的全局坐标表面定义:")
        for surface in global_surfaces:
            print(f"\n  表面 {surface.index}:")
            print(f"    类型: {surface.surface_type}")
            print(f"    顶点: {surface.vertex_position}")
            print(f"    Z轴: {surface.orientation[:, 2]}")
            if not np.isinf(surface.radius):
                print(f"    曲率半径: {surface.radius}")
                print(f"    曲率中心: {surface.curvature_center}")
            if surface.is_mirror:
                print(f"    反射镜: 是")
            if surface.comment:
                print(f"    注释: {surface.comment}")
        
        return True
        
    except FileNotFoundError as e:
        print(f"✗ 文件不存在: {e}")
        return False
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("  Zemax 光轴追踪与坐标转换验证脚本")
    print("=" * 60)
    
    results = []
    
    # 运行内置验证测试
    results.append(("45 度旋转", verify_45_degree_rotation()))
    results.append(("坐标断点 Order 参数", verify_coordinate_break_order()))
    results.append(("Z 形双镜系统", verify_z_shape_system()))
    results.append(("曲率中心计算", verify_curvature_center()))
    results.append(("反射镜行为", verify_mirror_behavior()))
    
    # 如果提供了 ZMX 文件路径，加载并验证
    if len(sys.argv) > 1:
        zmx_file_path = sys.argv[1]
        results.append(("ZMX 文件加载", load_and_verify_zmx_file(zmx_file_path)))
    
    # 打印总结
    print_separator("验证结果总结")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {name}: {status}")
    
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n所有验证测试通过！")
        return 0
    else:
        print("\n部分验证测试失败，请检查。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
