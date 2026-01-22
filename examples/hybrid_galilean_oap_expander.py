"""
使用 HybridOpticalPropagator 的伽利略 OAP 扩束镜示例

本示例展示如何使用新的 HybridOpticalPropagator API 进行混合光学传播仿真。

系统配置：
- OAP1: f=-300mm 凸面镜（发散光束），倾斜 45°
- 折叠镜: 平面镜，倾斜 45°
- OAP2: f=900mm 凹面镜（准直发散光束），倾斜 45°
- 放大倍率: M = -f2/f1 = 3x

作者：混合光学仿真项目
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List

from hybrid_optical_propagation import (
    HybridOpticalPropagator,
    SourceDefinition,
)


# ============================================================================
# 辅助类：模拟 GlobalSurfaceDefinition
# ============================================================================

@dataclass
class MockSurface:
    """模拟的 GlobalSurfaceDefinition 对象"""
    index: int
    surface_type: str
    vertex_position: np.ndarray
    orientation: np.ndarray
    radius: float = np.inf
    conic: float = 0.0
    is_mirror: bool = False
    semi_aperture: float = 25.0
    material: str = "air"
    asphere_coeffs: List[float] = field(default_factory=list)
    comment: str = ""
    thickness: float = 0.0
    radius_x: float = np.inf
    conic_x: float = 0.0
    focal_length: float = np.inf
    
    @property
    def surface_normal(self) -> np.ndarray:
        return -self.orientation[:, 2]


def create_flat_mirror(
    index: int,
    position: np.ndarray,
    tilt_x_rad: float = 0.0,
) -> MockSurface:
    """创建平面镜"""
    c, s = np.cos(tilt_x_rad), np.sin(tilt_x_rad)
    orientation = np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ])
    
    return MockSurface(
        index=index,
        surface_type='flat',
        vertex_position=np.asarray(position),
        orientation=orientation,
        is_mirror=True,
        material='mirror',
    )


def create_parabolic_mirror(
    index: int,
    position: np.ndarray,
    focal_length: float,
    tilt_x_rad: float = 0.0,
) -> MockSurface:
    """创建抛物面镜"""
    c, s = np.cos(tilt_x_rad), np.sin(tilt_x_rad)
    orientation = np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ])
    
    # 抛物面：R = 2f, k = -1
    radius = 2 * focal_length
    
    return MockSurface(
        index=index,
        surface_type='standard',
        vertex_position=np.asarray(position),
        orientation=orientation,
        radius=radius,
        conic=-1.0,  # 抛物面
        is_mirror=True,
        material='mirror',
    )


# ============================================================================
# 主程序
# ============================================================================

def main():
    print("=" * 60)
    print("HybridOpticalPropagator 伽利略 OAP 扩束镜示例")
    print("=" * 60)
    
    # 设计参数
    f1 = -300.0  # mm, OAP1 焦距（负值 = 凸面）
    f2 = 900.0   # mm, OAP2 焦距（正值 = 凹面）
    magnification = -f2 / f1  # 3x
    
    # 几何参数
    d_oap1_to_fold = 300.0   # mm
    d_fold_to_oap2 = 300.0   # mm
    
    print(f"""
设计参数:
  OAP1 焦距: f1 = {f1} mm (凸面)
  OAP2 焦距: f2 = {f2} mm (凹面)
  放大倍率: M = {magnification:.1f}x
  
几何参数:
  OAP1 -> 折叠镜: {d_oap1_to_fold} mm
  折叠镜 -> OAP2: {d_fold_to_oap2} mm
""")
    
    # 创建光学系统
    print("创建光学系统...")
    
    oap1 = create_parabolic_mirror(
        index=0,
        position=np.array([0.0, 0.0, 100.0]),
        focal_length=f1,
        tilt_x_rad=-np.pi/4,
    )
    
    fold = create_flat_mirror(
        index=1,
        position=np.array([0.0, -d_oap1_to_fold, 100.0]),
        tilt_x_rad=-np.pi/4,
    )
    
    oap2 = create_parabolic_mirror(
        index=2,
        position=np.array([0.0, -d_oap1_to_fold, 100.0 - d_fold_to_oap2]),
        focal_length=f2,
        tilt_x_rad=-np.pi/4,
    )
    
    optical_system = [oap1, fold, oap2]
    
    # 创建入射波面
    print("创建入射波面...")
    source = SourceDefinition(
        wavelength_um=10.64,  # CO2 激光
        w0_mm=10.0,
        z0_mm=0.0,
        grid_size=128,
        physical_size_mm=100.0,
    )
    
    # 创建传播器
    print("创建 HybridOpticalPropagator...")
    propagator = HybridOpticalPropagator(
        optical_system=optical_system,
        source=source,
        wavelength_um=10.64,
        grid_size=128,
        num_rays=200,
    )
    
    # 执行传播
    print("执行传播...")
    result = propagator.propagate()
    
    if result.success:
        print("传播成功！")
    else:
        print(f"传播失败: {result.error_message}")
        return
    
    # 分析结果
    print("\n" + "=" * 60)
    print("传播结果分析")
    print("=" * 60)
    
    print(f"""
表面状态数量: {len(result.surface_states)}
总光程: {result.total_path_length:.2f} mm

初始状态:
  网格大小: {result.surface_states[0].grid_sampling.grid_size}
  物理尺寸: {result.surface_states[0].grid_sampling.physical_size_mm:.2f} mm
  采样间隔: {result.surface_states[0].grid_sampling.sampling_mm:.4f} mm

最终状态:
  网格大小: {result.final_state.grid_sampling.grid_size}
  物理尺寸: {result.final_state.grid_sampling.physical_size_mm:.2f} mm
  采样间隔: {result.final_state.grid_sampling.sampling_mm:.4f} mm
""")
    
    # 计算能量
    initial_energy = result.surface_states[0].get_total_energy()
    final_energy = result.final_state.get_total_energy()
    
    print(f"""
能量分析:
  初始能量: {initial_energy:.4f}
  最终能量: {final_energy:.4f}
  能量比: {final_energy/initial_energy:.4f}
""")
    
    # 可视化
    print("生成可视化图像...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 初始振幅（使用新的 amplitude 属性）
    initial_amp = result.surface_states[0].amplitude
    im1 = axes[0, 0].imshow(initial_amp, cmap='hot')
    axes[0, 0].set_title('Initial Amplitude')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 初始相位
    initial_phase = result.surface_states[0].get_phase()
    im2 = axes[0, 1].imshow(initial_phase, cmap='twilight')
    axes[0, 1].set_title('Initial Phase')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 最终振幅（使用新的 amplitude 属性）
    final_amp = result.final_state.amplitude
    im3 = axes[1, 0].imshow(final_amp, cmap='hot')
    axes[1, 0].set_title('Final Amplitude')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 最终相位
    final_phase = result.final_state.get_phase()
    im4 = axes[1, 1].imshow(final_phase, cmap='twilight')
    axes[1, 1].set_title('Final Phase')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.suptitle('Hybrid Galilean OAP Beam Expander', fontsize=14)
    plt.tight_layout()
    plt.savefig('hybrid_galilean_oap_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ 保存: hybrid_galilean_oap_results.png")
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
