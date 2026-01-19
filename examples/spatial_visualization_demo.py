"""
空间坐标可视化演示

展示如何用简洁的代码定义光学系统并自动绘制空间光路图。
对比展开模式和空间模式的可视化效果。

作者：混合光学仿真项目
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sequential_system import (
    SequentialOpticalSystem,
    GaussianBeamSource,
    ParabolicMirror,
    FlatMirror,
)


def main():
    print("=" * 60)
    print("空间坐标可视化演示")
    print("=" * 60)
    
    # 1. 定义光源
    source = GaussianBeamSource(
        wavelength=1.064,  # μm
        w0=5.0,            # mm
    )
    
    # 2. 创建系统
    system = SequentialOpticalSystem(source, grid_size=256, beam_ratio=0.3)
    
    # 3. 添加采样面和元件
    system.add_sampling_plane(distance=0.0, name="Input")
    
    system.add_surface(ParabolicMirror(
        parent_focal_length=-30.0,
        thickness=40.0,
        semi_aperture=15.0,
        tilt_x=np.pi/4,
        name="OAP1",
    ))
    
    system.add_surface(FlatMirror(
        thickness=40.0,
        semi_aperture=20.0,
        tilt_x=np.pi/4,
        name="Fold",
    ))
    
    system.add_surface(ParabolicMirror(
        parent_focal_length=90.0,
        thickness=60.0,
        semi_aperture=30.0,
        tilt_x=np.pi/4,
        name="OAP2",
    ))
    
    system.add_sampling_plane(distance=140.0, name="Output")
    
    # 4. 打印系统配置
    print("\n系统配置：")
    print(system.summary())
    
    # 5. 运行仿真并自动绘制空间光路图
    print("\n运行仿真...")
    results = system.run(
        plot=True,
        plot_mode="spatial",
        save_plot="spatial_visualization_demo.png",
        show_plot=False,
    )
    print("✓ 保存: spatial_visualization_demo.png")
    
    # 6. 创建对比图（展开 vs 空间）
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 左图：展开模式
    from sequential_system.visualization import LayoutVisualizer
    visualizer = LayoutVisualizer(system)
    
    # 手动绘制到指定 axes
    ax1 = axes[0]
    from gaussian_beam_simulation.abcd_calculator import ABCDCalculator
    beam = system.source.to_gaussian_beam()
    calculator = ABCDCalculator(beam, system.elements)
    
    total_path = system.total_path_length
    distances = np.linspace(0, total_path * 1.1, 200)
    w_values = np.array([calculator.propagate_distance(d).w for d in distances])
    
    ax1.fill_between(distances, w_values, -w_values, color='blue', alpha=0.2)
    ax1.plot(distances, w_values, 'b-', linewidth=1.5)
    ax1.plot(distances, -w_values, 'b-', linewidth=1.5)
    
    colors = ['purple', 'gray', 'teal']
    for i, elem in enumerate(system.elements):
        ax1.axvline(x=elem.path_length, color=colors[i], linewidth=3, label=elem.name)
    
    for plane in system.sampling_planes:
        ax1.axvline(x=plane.distance, color='red', linestyle=':', linewidth=2, alpha=0.7)
    
    ax1.set_xlabel("Path Length (mm)")
    ax1.set_ylabel("Y Position (mm)")
    ax1.set_title("Unfolded View")
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_aspect('equal', adjustable='datalim')
    
    # 右图：空间模式（使用内置方法）
    ax2 = axes[1]
    tracker = system.axis_tracker
    
    # 绘制光束路径
    z_coords, y_coords = tracker.calculate_beam_path_2d(num_points=200, projection="yz")
    ax2.plot(z_coords, y_coords, 'b-', linewidth=2.5, alpha=0.8, label='Beam Path')
    
    # 使用 visualizer 的内部方法绘制元件
    visualizer._draw_elements_spatial(ax2, tracker)
    visualizer._draw_sampling_planes_spatial(ax2, tracker)
    
    ax2.set_xlabel("Z Position (mm)")
    ax2.set_ylabel("Y Position (mm)")
    ax2.set_title("Spatial View")
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='datalim')
    
    plt.suptitle("Folded Beam Expander: Unfolded vs Spatial View", fontsize=14)
    plt.tight_layout()
    fig.savefig("comparison_view.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("✓ 保存: comparison_view.png")
    
    # 7. 打印光轴跟踪信息
    print("\n" + "=" * 60)
    print("光轴跟踪信息")
    print("=" * 60)
    
    for elem, pos, dir_before, dir_after in tracker.get_element_global_positions():
        name = elem.name if elem.name else elem.element_type
        print(f"\n{name}:")
        print(f"  位置: ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f}) mm")
        print(f"  入射方向: ({dir_before.L:.3f}, {dir_before.M:.3f}, {dir_before.N:.3f})")
        print(f"  出射方向: ({dir_after.L:.3f}, {dir_after.M:.3f}, {dir_after.N:.3f})")
    
    print("\n采样面位置:")
    for plane in system.sampling_planes:
        state = tracker.get_state_at_distance(plane.distance)
        print(f"  {plane.name}: ({state.position.x:.2f}, {state.position.y:.2f}, {state.position.z:.2f}) mm")
    
    # 8. 打印仿真结果
    print("\n" + "=" * 60)
    print("仿真结果")
    print("=" * 60)
    for result in results:
        print(f"  {result.name}: 光束半径 = {result.beam_radius:.3f} mm")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
