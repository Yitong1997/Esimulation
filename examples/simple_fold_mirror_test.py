"""
简单折叠镜测试示例

演示 HybridSimulator 的基本用法：
1. 创建仿真器
2. 添加 45° 平面镜
3. 设置光源
4. 执行仿真
5. 查看结果

这是主程序的典型用法，代码极简。
"""

import sys
import os

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hybrid_simulation import HybridSimulator


def main():
    """主程序 - 极简示例"""
    
    # ========== 主程序代码（< 10 行）==========
    
    # 步骤 1-3：创建仿真器，添加元件，设置光源
    sim = HybridSimulator(verbose=True)
    sim.add_flat_mirror(z=50.0, tilt_x=45.0)  # 45° 折叠镜
    sim.set_source(wavelength_um=0.633, w0_mm=5.0, grid_size=256)
    
    # 步骤 4：执行仿真
    result = sim.run()
    
    # 步骤 5：查看结果
    result.summary()
    
    # ========== 可选：详细分析 ==========
    
    # 绘制所有表面
    result.plot_all(save_path='fold_mirror_overview.png', show=False)
    print("已保存概览图: fold_mirror_overview.png")
    
    # 绘制特定表面详情
    if len(result.surfaces) > 1:
        result.plot_surface(0, save_path='fold_mirror_surface0.png', show=False)
        print("已保存表面详情图: fold_mirror_surface0.png")
    
    # 保存完整结果
    result.save('output/fold_mirror_result')
    print("已保存完整结果到: output/fold_mirror_result/")
    
    # 验证加载
    from hybrid_simulation import SimulationResult
    loaded = SimulationResult.load('output/fold_mirror_result')
    print(f"加载验证: {loaded.success}, {len(loaded.surfaces)} 个表面")
    
    return result


if __name__ == '__main__':
    result = main()
