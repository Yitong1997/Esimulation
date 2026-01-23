"""
简单折叠镜测试示例

演示 bts API 的基本用法，代码极简（< 10 行核心代码）。

本示例展示了 MATLAB 风格的代码块结构：
1. 导入与初始化
2. 定义光学系统
3. 定义光源
4. 执行仿真
5. 查看结果

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
# 2. 定义光学系统
# ============================================================
# 创建一个简单的光学系统，包含一个 45° 折叠镜
system = bts.OpticalSystem("Simple Fold Mirror")
system.add_flat_mirror(z=50.0, tilt_x=45.0)  # 45° 折叠镜

# ============================================================
# 3. 定义光源
# ============================================================
# 定义 He-Ne 激光高斯光源
source = bts.GaussianSource(wavelength_um=0.633, w0_mm=5.0, grid_size=256)

# ============================================================
# 4. 执行仿真
# ============================================================
result = bts.simulate(system, source)

# ============================================================
# 5. 查看结果
# ============================================================
# 打印结果摘要
result.summary()

# 绘制所有表面的概览图并保存
result.plot_all(save_path='fold_mirror_overview.png', show=False)
print("已保存概览图: fold_mirror_overview.png")

# 保存完整结果到目录
result.save('output/fold_mirror_result')
print("已保存完整结果到: output/fold_mirror_result/")


# ============================================================
# 可选：验证 API 功能
# ============================================================
def verify_api():
    """验证 bts API 的各项功能"""
    print("\n" + "=" * 60)
    print("验证 bts API 功能")
    print("=" * 60)
    
    # 验证系统信息展示
    print("\n--- 系统信息 ---")
    system.print_info()
    
    # 验证光源信息展示
    print("\n--- 光源信息 ---")
    source.print_info()
    
    # 验证结果加载功能
    print("\n--- 验证结果加载 ---")
    loaded = bts.SimulationResult.load('output/fold_mirror_result')
    print(f"加载验证: success={loaded.success}, 表面数量={len(loaded.surfaces)}")
    
    # 验证波前数据访问
    print("\n--- 波前数据访问 ---")
    final_wf = result.get_final_wavefront()
    if final_wf is not None:
        print(f"最终波前网格大小: {final_wf.grid.grid_size}")
        print(f"最终波前物理尺寸: {final_wf.grid.physical_size_mm:.3f} mm")
        print(f"残差相位 RMS: {final_wf.get_residual_rms_waves()*1000:.3f} milli-waves")
        print(f"残差相位 PV: {final_wf.get_residual_pv_waves():.4f} waves")
    
    print("\n" + "=" * 60)
    print("API 验证完成！")
    print("=" * 60)


if __name__ == '__main__':
    # 运行 API 验证
    verify_api()
