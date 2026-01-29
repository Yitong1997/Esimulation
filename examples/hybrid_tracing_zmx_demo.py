"""
ZMX 混合光路追迹主程序
=======================

本程序演示如何利用 BTS API 读取 ZMX 文件并进行物理光学混合追迹。
参考：tests/integration/伽利略式离轴抛物面扩束镜传输误差标准测试文件.py

功能：
1. 加载 Zemax (.zmx) 光学系统文件 (系统会自动解析为 bts.OpticalSystem)
2. 定义高斯光束源 (bts.GaussianSource)
3. 执行混合物理光学追迹 (bts.simulate)
4. 可视化光路和最终波前

注意：
- 本示例使用 'simple_fold_mirror_up.zmx'。
- 确保 ZMX 文件中的光路布局与光源定义的入射位置 (z0_mm) 匹配。

作者: 混合光学仿真项目
"""

import sys
from pathlib import Path

# ==============================================================================
# 1. 环境配置 (Environment Setup)
# ==============================================================================

# 获取项目根目录
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

# 添加源代码和依赖库到 Python 路径
sys.path.insert(0, str(project_root / 'src'))                # BTS 源码
sys.path.insert(0, str(project_root / 'optiland-master'))    # Optiland 几何光线追迹库
sys.path.insert(0, str(project_root / 'proper_v3.3.4_python')) # PROPER 物理光学传播库

import bts
import matplotlib.pyplot as plt

def main():
    print("=" * 60)
    print("BTS 混合光路追迹演示程序")
    print("=" * 60)

    # ==========================================================================
    # 2. 加载光学系统 (Load Optical System)
    # ==========================================================================
    
    # 指定 ZMX 文件路径
    # 可以在此处修改为您想要测试的 ZMX 文件
    zmx_dir = project_root / 'optiland-master' / 'tests' / 'zemax_files'
    # zmx_file = zmx_dir / 'complicated_fold_mirrors_setup_v2.zmx'
    zmx_file = zmx_dir / 'simple_fold_mirror_up.zmx'
    
    if not zmx_file.exists():
        print(f"错误: 找不到 ZMX 文件: {zmx_file}")
        return

    print(f"\n[1] 正在加载 ZMX 文件: {zmx_file.name}")
    try:
        # 使用 bts.load_zmx 直接加载系统
        # 该函数会自动处理坐标转换，将 ZMX 表面转换为 BTS 全局表面定义
        system = bts.load_zmx(str(zmx_file))
        
        # 打印系统基本信息
        system.print_info()
        
    except Exception as e:
        print(f"加载失败: {e}")
        return

    # ==========================================================================
    # 3. 定义光源 (Define Light Source)
    # ==========================================================================
    
    print("\n[2] 定义高斯光源")
    
    # 设置光源参数
    wavelength = 0.6328  # 波长 (微米), HeNe 激光
    waist_radius = 2.0   # 束腰半径 w0 (毫米)
    grid_size = 512      # 网格点数 (建议 2 的幂次)
    
    source = bts.GaussianSource(
        wavelength_um=wavelength,
        w0_mm=waist_radius,
        grid_size=grid_size,
        z0_mm=0.0,              # 光束起始位置 (假设从系统入口 z=0 处入射)
        physical_size_mm=40.0,  # 物理网格尺寸 (需大于光束直径，例如 > 4*w0)
        beam_diam_fraction=0.25 # 光束直径占网格的比例
    )
    
    print(f"  波长: {source.wavelength_um} um")
    print(f"  束腰半径: {source.w0_mm} mm")
    print(f"  网格大小: {source.grid_size} x {source.grid_size}")

    # ==========================================================================
    # 4. 执行仿真 (Run Simulation)
    # ==========================================================================
    
    print("\n[3] 开始混合光路追迹仿真...")
    print("  (使用 'local_raytracing' 方法计算元件相位调制)")
    
    try:
        # 执行仿真
        # propagation_method="local_raytracing": 
        #   使用光线追迹 (Optiland) 计算每个网格点的 OPD，然后应用到物理光学波前 (PROPER)
        result = bts.simulate(
            system, 
            source, 
            propagation_method="local_raytracing", 
            debug=False  # 设置为 True 可查看详细调试图
        )
        print("仿真完成！")
        
    except Exception as e:
        print(f"仿真过程中发生错误: {e}")
        # 如果仿真失败，可能是光线未击中元件或 ZMX 布局问题
        print("提示: 请检查光源位置 (z0_mm) 是否正确，以及光线是否能够顺利穿过系统。")
        return

    # ==========================================================================
    # 5. 结果分析与可视化 (Analysis & Visualization)
    # ==========================================================================
    
    print("\n[4] 结果可视化")
    
    # 5.1 获取最终波前
    final_wf = result.get_final_wavefront()
    
    # 打印简单的结果统计
    rms = final_wf.get_residual_rms_waves()
    print(f"  最终波前 RMS 误差: {rms:.4f} waves")
    
    # 5.2 绘制 2D 光路布局图
    print("  绘制光路布局...")
    try:
        layout_fig, _ = system.plot_layout(
            mode='2d', 
            projection='YZ', 
            save_path='zmx_trace_layout.png',
            show=False
        )
        print("  光路布局已保存: zmx_trace_layout.png")
    except Exception as e:
        print(f"  无法绘制光路布局: {e}")
    
    # 5.3 绘制最终波前 (相位和强度)
    print("  绘制最终波前...")
    try:
        final_wf.plot(
            title="Final Wavefront at Exit Surface", 
            save_path='zmx_trace_wavefront.png',
            show=False
        )
        print("  波前图已保存: zmx_trace_wavefront.png")
    except Exception as e:
        print(f"  无法绘制波前: {e}")
    
    print("\n程序执行完毕。")
    print("=" * 60)

    # 如果需要显示交互式窗口，请取消下面的注释
    # plt.show()

if __name__ == "__main__":
    main()
