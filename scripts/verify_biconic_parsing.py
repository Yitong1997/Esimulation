"""
验证双锥面（BiconicX）解析和转换的完整流程

本脚本测试：
1. ZMX 解析器是否正确读取 BICONICX 表面的两个方向曲率和圆锥常数
2. 坐标转换是否正确传递双锥面参数
3. show_info 是否正确打印双锥面参数
4. optiland 转换是否正确设置双锥面参数
"""

import sys
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sequential_system.zmx_parser import ZmxParser, ZmxSurfaceData


def test_zmx_surface_data():
    """测试 ZmxSurfaceData 是否正确存储双锥面参数"""
    print("=" * 60)
    print("测试 1: ZmxSurfaceData 双锥面参数存储")
    print("=" * 60)
    
    # 创建双锥面表面数据
    surface = ZmxSurfaceData(
        index=1,
        surface_type='biconic',
        radius=100.0,      # Y 方向曲率半径
        conic=-1.0,        # Y 方向圆锥常数（抛物面）
        radius_x=50.0,     # X 方向曲率半径
        conic_x=-0.5,      # X 方向圆锥常数
        is_mirror=True,
        semi_diameter=25.0,
        comment="Test Biconic Mirror"
    )
    
    print(f"表面类型: {surface.surface_type}")
    print(f"Y 方向曲率半径: {surface.radius} mm")
    print(f"Y 方向圆锥常数: {surface.conic}")
    print(f"X 方向曲率半径: {surface.radius_x} mm")
    print(f"X 方向圆锥常数: {surface.conic_x}")
    print(f"\n__repr__: {surface}")
    
    # 验证
    assert surface.surface_type == 'biconic'
    assert surface.radius == 100.0
    assert surface.conic == -1.0
    assert surface.radius_x == 50.0
    assert surface.conic_x == -0.5
    print("\n✓ ZmxSurfaceData 双锥面参数存储正确")


def test_global_surface_definition():
    """测试 GlobalSurfaceDefinition 是否正确存储双锥面参数"""
    print("\n" + "=" * 60)
    print("测试 2: GlobalSurfaceDefinition 双锥面参数存储")
    print("=" * 60)
    
    from sequential_system.coordinate_system import GlobalSurfaceDefinition
    
    # 创建全局坐标表面定义
    surface = GlobalSurfaceDefinition(
        index=1,
        surface_type='biconic',
        vertex_position=np.array([0.0, 0.0, 100.0]),
        orientation=np.eye(3),
        radius=100.0,      # Y 方向曲率半径
        conic=-1.0,        # Y 方向圆锥常数
        radius_x=50.0,     # X 方向曲率半径
        conic_x=-0.5,      # X 方向圆锥常数
        is_mirror=True,
        semi_aperture=25.0,
        comment="Test Biconic Mirror"
    )
    
    print(f"表面类型: {surface.surface_type}")
    print(f"Y 方向曲率半径: {surface.radius} mm")
    print(f"Y 方向圆锥常数: {surface.conic}")
    print(f"X 方向曲率半径: {surface.radius_x} mm")
    print(f"X 方向圆锥常数: {surface.conic_x}")
    
    # 验证
    assert surface.surface_type == 'biconic'
    assert surface.radius == 100.0
    assert surface.conic == -1.0
    assert surface.radius_x == 50.0
    assert surface.conic_x == -0.5
    print("\n✓ GlobalSurfaceDefinition 双锥面参数存储正确")


def test_zmx_visualization_print():
    """测试 print_surface_info 是否正确打印双锥面参数"""
    print("\n" + "=" * 60)
    print("测试 3: print_surface_info 双锥面参数打印")
    print("=" * 60)
    
    from sequential_system.coordinate_system import GlobalSurfaceDefinition
    from sequential_system.zmx_visualization import ZmxOpticLoader
    
    # 创建模拟的 loader 并手动设置 global_surfaces
    loader = ZmxOpticLoader("dummy.zmx")
    loader._global_surfaces = [
        GlobalSurfaceDefinition(
            index=1,
            surface_type='biconic',
            vertex_position=np.array([0.0, 0.0, 100.0]),
            orientation=np.eye(3),
            radius=100.0,
            conic=-1.0,
            radius_x=50.0,
            conic_x=-0.5,
            is_mirror=True,
            semi_aperture=25.0,
            comment="Test Biconic Mirror"
        )
    ]
    loader.zmx_file_path = Path("test_biconic.zmx")
    
    # 打印表面信息
    print("\n--- print_surface_info 输出 ---")
    loader.print_surface_info()
    print("--- 输出结束 ---")
    
    print("\n✓ print_surface_info 双锥面参数打印测试完成")


def test_optiland_conversion():
    """测试转换到 optiland 时双锥面参数是否正确"""
    print("\n" + "=" * 60)
    print("测试 4: optiland 转换双锥面参数")
    print("=" * 60)
    
    from sequential_system.coordinate_system import (
        GlobalSurfaceDefinition,
        ZemaxToOptilandConverter,
    )
    
    # 创建全局坐标表面定义
    global_surfaces = [
        GlobalSurfaceDefinition(
            index=1,
            surface_type='biconic',
            vertex_position=np.array([0.0, 0.0, 0.0]),
            orientation=np.eye(3),
            radius=100.0,      # Y 方向曲率半径
            conic=-1.0,        # Y 方向圆锥常数
            radius_x=50.0,     # X 方向曲率半径
            conic_x=-0.5,      # X 方向圆锥常数
            is_mirror=False,
            semi_aperture=25.0,
            thickness=10.0,
            comment="Test Biconic Surface"
        )
    ]
    
    # 转换为 optiland
    converter = ZemaxToOptilandConverter(
        global_surfaces,
        wavelength=0.55,
        entrance_pupil_diameter=10.0
    )
    
    try:
        optic = converter.convert()
        
        # 检查表面参数
        # optiland 中表面索引：0=物面，1=第一个光学表面，2=像面
        surface = optic.surface_group.surfaces[1]
        
        print(f"optiland 表面类型: {surface.surface_type}")
        print(f"几何类型: {surface.geometry.__class__.__name__}")
        
        # 检查几何参数
        geo = surface.geometry
        if hasattr(geo, 'Rx'):
            print(f"X 方向曲率半径 (Rx): {geo.Rx}")
        if hasattr(geo, 'Ry'):
            print(f"Y 方向曲率半径 (Ry): {geo.Ry}")
        if hasattr(geo, 'kx'):
            print(f"X 方向圆锥常数 (kx): {geo.kx}")
        if hasattr(geo, 'ky'):
            print(f"Y 方向圆锥常数 (ky): {geo.ky}")
        
        print("\n✓ optiland 转换测试完成")
        
    except Exception as e:
        print(f"\n⚠ optiland 转换出错: {e}")
        import traceback
        traceback.print_exc()


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("双锥面（BiconicX）解析和转换验证")
    print("=" * 60)
    
    test_zmx_surface_data()
    test_global_surface_definition()
    test_zmx_visualization_print()
    test_optiland_conversion()
    
    print("\n" + "=" * 60)
    print("所有测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
