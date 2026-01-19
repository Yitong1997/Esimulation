"""
简化演示：自动光路绘制

展示如何用最少的代码定义光学系统并自动绘制光路图。

作者：混合光学仿真项目
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
from sequential_system import (
    SequentialOpticalSystem,
    GaussianBeamSource,
    ParabolicMirror,
    FlatMirror,
)


def main():
    # 1. 定义光源
    source = GaussianBeamSource(
        wavelength=1.064,  # μm
        w0=5.0,            # mm
    )
    
    # 2. 创建系统并添加元件
    system = SequentialOpticalSystem(source, grid_size=256, beam_ratio=0.3)
    
    # 添加采样面：输入
    system.add_sampling_plane(distance=0.0, name="Input")
    
    # OAP1：凸面（发散）
    system.add_surface(ParabolicMirror(
        parent_focal_length=-30.0,
        thickness=40.0,
        semi_aperture=15.0,
        tilt_x=np.pi/4,
        name="OAP1",
    ))
    
    # 折叠镜
    system.add_surface(FlatMirror(
        thickness=40.0,
        semi_aperture=20.0,
        tilt_x=np.pi/4,
        name="Fold",
    ))
    
    # OAP2：凹面（准直）
    system.add_surface(ParabolicMirror(
        parent_focal_length=90.0,
        thickness=60.0,
        semi_aperture=30.0,
        tilt_x=np.pi/4,
        name="OAP2",
    ))
    
    # 添加采样面：输出
    system.add_sampling_plane(distance=140.0, name="Output")
    
    # 3. 运行仿真并自动绘制光路图
    print("运行仿真...")
    results = system.run(
        plot=True,                      # 自动绘制光路图
        plot_mode="spatial",            # 空间坐标模式
        save_plot="simple_demo.png",    # 保存图像
        show_plot=False,                # 不显示（适合脚本运行）
    )
    
    print(f"✓ 保存: simple_demo.png")
    
    # 4. 查看结果
    print("\n仿真结果:")
    for result in results:
        print(f"  {result.name}: 光束半径 = {result.beam_radius:.3f} mm")


if __name__ == "__main__":
    main()
