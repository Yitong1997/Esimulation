"""
调试 NaN 错误

定位 "Points cannot contain NaN" 错误的来源
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'optiland-master'))

import numpy as np
import warnings

# 忽略 scipy 的废弃警告
warnings.filterwarnings('ignore', category=DeprecationWarning)

from hybrid_optical_propagation import (
    SourceDefinition,
    HybridOpticalPropagator,
    load_optical_system_from_zmx,
)


def main():
    """主函数"""
    
    zmx_file = "optiland-master/tests/zemax_files/complicated_fold_mirrors_setup_v2.zmx"
    
    print("加载光学系统...")
    optical_system = load_optical_system_from_zmx(zmx_file)
    print(f"  表面数量: {len(optical_system)}")
    
    for surface in optical_system:
        mirror_str = " [MIRROR]" if surface.is_mirror else ""
        radius_str = f"R={surface.radius:.2f}" if not np.isinf(surface.radius) else "R=∞"
        print(f"  - 表面 {surface.index}: {surface.surface_type}, "
              f"{radius_str}{mirror_str}, comment='{surface.comment}'")
    
    # 创建光源定义
    source = SourceDefinition(
        wavelength_um=0.55,
        w0_mm=5.0,
        z0_mm=0.0,
        grid_size=256,
        physical_size_mm=40.0,
    )
    
    # 创建传播器
    print("\n创建传播器...")
    propagator = HybridOpticalPropagator(
        optical_system=optical_system,
        source=source,
        wavelength_um=0.55,
        grid_size=256,
        num_rays=150,
    )
    
    # 逐步传播，找出问题表面
    print("\n逐步传播...")
    
    from hybrid_optical_propagation.material_detection import (
        is_coordinate_break,
        detect_material_change,
    )
    
    # 初始化
    propagator._current_state = propagator._initialize_propagation()
    propagator._surface_states = [propagator._current_state]
    
    print(f"\n初始状态:")
    print(f"  振幅范围: [{np.min(propagator._current_state.amplitude):.6f}, {np.max(propagator._current_state.amplitude):.6f}]")
    print(f"  相位范围: [{np.min(propagator._current_state.phase):.6f}, {np.max(propagator._current_state.phase):.6f}]")
    print(f"  振幅 NaN: {np.any(np.isnan(propagator._current_state.amplitude))}")
    print(f"  相位 NaN: {np.any(np.isnan(propagator._current_state.phase))}")
    
    for i, surface in enumerate(optical_system):
        print(f"\n处理表面 {i}: {surface.surface_type}, is_mirror={surface.is_mirror}")
        
        # 跳过坐标断点
        if is_coordinate_break(surface):
            print("  跳过坐标断点")
            continue
        
        try:
            # 传播到当前表面
            propagator._propagate_to_surface(i)
            
            # 检查状态
            state = propagator._current_state
            print(f"  传播成功!")
            print(f"  振幅范围: [{np.min(state.amplitude):.6f}, {np.max(state.amplitude):.6f}]")
            print(f"  相位范围: [{np.min(state.phase):.6f}, {np.max(state.phase):.6f}]")
            print(f"  振幅 NaN: {np.any(np.isnan(state.amplitude))}")
            print(f"  相位 NaN: {np.any(np.isnan(state.phase))}")
            
            if np.any(np.isnan(state.amplitude)) or np.any(np.isnan(state.phase)):
                print("  [ERROR] 检测到 NaN!")
                break
                
        except Exception as e:
            print(f"  [ERROR] 传播失败: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("\n完成")


if __name__ == '__main__':
    main()
