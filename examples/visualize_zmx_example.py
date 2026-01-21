"""
ZMX 文件可视化示例

本示例演示如何使用 zmx_visualization 模块加载和可视化 ZMX 文件。
主要功能：
1. 加载 ZMX 文件并转换为 optiland Optic 对象
2. 使用 optiland OpticViewer 进行 2D 可视化
3. 使用 optiland OpticViewer3D 进行 3D 可视化（可选，需要 VTK）
4. 打印表面信息摘要

使用方法：
    python examples/visualize_zmx_example.py

作者：混合光学仿真项目
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'optiland-master'))

import matplotlib.pyplot as plt
from sequential_system.zmx_visualization import (
    ZmxOpticLoader,
    visualize_zmx,
    view_2d,
)


def main():
    """主函数：演示 ZMX 文件可视化"""
    
    # ZMX 测试文件路径
    zmx_dir = project_root / 'optiland-master' / 'tests' / 'zemax_files'
    
    # 主要测试文件：复杂折叠镜系统
    main_zmx_file = zmx_dir / 'complicated_fold_mirrors_setup_v2.zmx'
    
    print("=" * 60)
    print("ZMX 文件可视化示例")
    print("=" * 60)
    
    # =========================================================================
    # 示例 1：使用 ZmxOpticLoader 类（完整控制）
    # =========================================================================
    print("\n[示例 1] 使用 ZmxOpticLoader 类加载 ZMX 文件")
    print("-" * 40)
    
    loader = ZmxOpticLoader(main_zmx_file)
    optic = loader.load()
    
    # 打印表面信息
    loader.print_surface_info()
    
    # 使用 optiland OpticViewer 进行 2D 可视化
    print("\n正在生成 2D 可视化...")
    fig1, ax1, _ = view_2d(optic, projection='YZ', num_rays=5)
    ax1.set_title(f'Complicated Fold Mirrors System (YZ)\n{main_zmx_file.name}')
    
    # 保存图片
    output_file = project_root / 'zmx_visualization_demo.png'
    fig1.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"图片已保存到: {output_file}")
    
    # =========================================================================
    # 示例 2：使用 visualize_zmx() 便捷函数
    # =========================================================================
    print("\n[示例 2] 使用 visualize_zmx() 便捷函数")
    print("-" * 40)
    
    # 测试其他 ZMX 文件
    other_zmx_files = [
        'simple_fold_mirror_up.zmx',
        'one_mirror_up_45deg.zmx',
    ]
    
    for zmx_name in other_zmx_files:
        zmx_path = zmx_dir / zmx_name
        if zmx_path.exists():
            print(f"\n加载: {zmx_name}")
            try:
                fig, ax, _ = visualize_zmx(
                    zmx_path,
                    mode='2d',
                    projection='YZ',
                    num_rays=3,
                    title=zmx_name,
                    show_info=True
                )
            except Exception as e:
                print(f"  警告: 可视化失败 - {e}")
    
    # =========================================================================
    # 示例 3：测试透镜系统
    # =========================================================================
    print("\n[示例 3] 测试透镜系统")
    print("-" * 40)
    
    lens_files = ['lens1.zmx', 'lens2.zmx']
    for lens_name in lens_files:
        lens_path = zmx_dir / lens_name
        if lens_path.exists():
            print(f"\n加载: {lens_name}")
            try:
                fig, ax, _ = visualize_zmx(
                    lens_path,
                    mode='2d',
                    projection='YZ',
                    num_rays=5,
                    title=lens_name
                )
            except Exception as e:
                print(f"  警告: 可视化失败 - {e}")
    
    # =========================================================================
    # 显示所有图形
    # =========================================================================
    print("\n" + "=" * 60)
    print("可视化完成！")
    print("=" * 60)
    
    # 如果在交互模式下运行，显示图形
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--show':
        print("关闭图形窗口以退出程序。")
        plt.show()
    else:
        print(f"主图片已保存到: {output_file}")
        print("运行 'python examples/visualize_zmx_example.py --show' 以显示交互式图形窗口")


def demo_3d_visualization():
    """演示 3D 可视化（需要 VTK）"""
    
    zmx_dir = project_root / 'optiland-master' / 'tests' / 'zemax_files'
    main_zmx_file = zmx_dir / 'complicated_fold_mirrors_setup_v2.zmx'
    
    print("\n[3D 可视化演示]")
    print("-" * 40)
    print("注意: 3D 可视化需要安装 VTK 库")
    print("如果未安装，请运行: pip install vtk")
    
    try:
        visualize_zmx(main_zmx_file, mode='3d')
    except Exception as e:
        print(f"3D 可视化失败: {e}")


if __name__ == '__main__':
    # 运行主示例
    main()
    
    # 如果需要 3D 可视化，取消下面的注释
    demo_3d_visualization()
