"""
ZMX 文件混合光学仿真示例

演示如何从 ZMX 文件加载光学系统并执行仿真。

本示例展示了 MATLAB 风格的代码块结构：
1. 导入与初始化
2. 加载 ZMX 文件
3. 系统信息展示
4. 定义光源
5. 执行仿真
6. 结果展示与保存

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6
"""

import sys
import os

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# ============================================================
# 1. 导入与初始化
# ============================================================
import bts

# ============================================================
# 2. 加载 ZMX 文件
# ============================================================
# 使用工作区中存在的 ZMX 文件
# 该文件定义了一个复杂的折叠镜光学系统
zmx_path = os.path.join(
    os.path.dirname(__file__), 
    '..', 
    'optiland-master', 
    'tests', 
    'zemax_files', 
    'complicated_fold_mirrors_setup_v2.zmx'
)

# 从 ZMX 文件加载光学系统
system = bts.load_zmx(zmx_path)

# ============================================================
# 3. 系统信息展示
# ============================================================
# 打印系统参数摘要
print("=" * 60)
print("光学系统信息")
print("=" * 60)
system.print_info()

# 绘制光路图（可选，需要 matplotlib）
try:
    system.plot_layout(
        projection='YZ', 
        num_rays=5, 
        save_path="output/zmx_layout.png",
        show=False
    )
    print("\n已保存光路图: output/zmx_layout.png")
except Exception as e:
    print(f"\n绘制光路图时出错: {e}")

# ============================================================
# 4. 定义光源
# ============================================================
# 定义 He-Ne 激光高斯光源
source = bts.GaussianSource(
    wavelength_um=0.633,    # He-Ne 激光波长 (μm)
    w0_mm=5.0,              # 束腰半径 (mm)
    grid_size=256,          # 网格大小
)

# 打印光源信息
print("\n" + "=" * 60)
print("光源信息")
print("=" * 60)
source.print_info()

# ============================================================
# 5. 执行仿真
# ============================================================
print("\n" + "=" * 60)
print("执行仿真")
print("=" * 60)
result = bts.simulate(system, source)

# ============================================================
# 6. 结果展示与保存
# ============================================================
print("\n" + "=" * 60)
print("仿真结果")
print("=" * 60)

# 打印结果摘要
result.summary()

# 绘制所有表面的概览图
result.plot_all(save_path="output/zmx_simulation_overview.png", show=False)
print("\n已保存概览图: output/zmx_simulation_overview.png")

# 保存完整结果到目录
result.save("output/zmx_result_data")
print("已保存完整结果到: output/zmx_result_data/")


# ============================================================
# 可选：验证加载功能
# ============================================================
def verify_load_functionality():
    """验证结果保存/加载功能"""
    print("\n" + "=" * 60)
    print("验证结果加载功能")
    print("=" * 60)
    
    # 从保存的目录加载结果
    loaded = bts.SimulationResult.load("output/zmx_result_data")
    
    # 验证加载的结果
    print(f"加载验证:")
    print(f"  - 仿真成功: {loaded.success}")
    print(f"  - 表面数量: {len(loaded.surfaces)}")
    print(f"  - 波长: {loaded.config.wavelength_um} μm")
    print(f"  - 网格大小: {loaded.config.grid_size}")
    
    # 验证数据一致性
    if loaded.success == result.success and len(loaded.surfaces) == len(result.surfaces):
        print("\n✓ 结果加载验证通过！")
    else:
        print("\n✗ 结果加载验证失败！")


def analyze_wavefront():
    """分析最终波前数据"""
    print("\n" + "=" * 60)
    print("波前分析")
    print("=" * 60)
    
    # 获取最终波前数据
    final_wf = result.get_final_wavefront()
    
    if final_wf is not None:
        print(f"最终波前参数:")
        print(f"  - 网格大小: {final_wf.grid.grid_size}")
        print(f"  - 物理尺寸: {final_wf.grid.physical_size_mm:.3f} mm")
        print(f"  - 采样间距: {final_wf.grid.sampling_mm:.6f} mm")
        
        # 计算波前误差
        rms_waves = final_wf.get_residual_rms_waves()
        pv_waves = final_wf.get_residual_pv_waves()
        
        print(f"\n波前误差:")
        print(f"  - 残差相位 RMS: {rms_waves*1000:.3f} milli-waves")
        print(f"  - 残差相位 PV:  {pv_waves:.4f} waves")
    else:
        print("无法获取最终波前数据")


if __name__ == '__main__':
    # 运行验证和分析
    verify_load_functionality()
    analyze_wavefront()
    
    print("\n" + "=" * 60)
    print("ZMX 文件仿真示例完成！")
    print("=" * 60)
