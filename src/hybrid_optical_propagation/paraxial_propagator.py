"""
PARAXIAL 表面处理器模块

本模块实现 Zemax ZMX 文件中 PARAXIAL 表面类型（理想薄透镜）的处理。

核心功能：
1. 使用 PROPER 的 prop_lens 应用薄透镜效果
2. 更新 Pilot Beam 参数
3. 同步更新仿真复振幅和 PROPER 对象

注意：仅用于 ZMX 文件中明确标记为 PARAXIAL 的表面，
严禁对其他表面类型擅自使用傍轴近似。

**Validates: Requirements 19.1-19.7**
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from .data_models import PilotBeamParams, GridSampling, PropagationState
from .state_converter import StateConverter

if TYPE_CHECKING:
    from sequential_system.coordinate_tracking import OpticalAxisState
    from sequential_system.coordinate_system import GlobalSurfaceDefinition


class ParaxialPhasePropagator:
    """PARAXIAL 表面处理器
    
    处理 Zemax ZMX 文件中的 PARAXIAL 表面类型（理想薄透镜），
    使用 PROPER 的 prop_lens 进行处理。
    
    注意：仅用于 ZMX 文件中明确标记为 PARAXIAL 的表面，
    严禁对其他表面类型擅自使用傍轴近似。
    
    属性:
        wavelength_um: 波长 (μm)
    
    **Validates: Requirements 19.1-19.7**
    """
    
    def __init__(self, wavelength_um: float) -> None:
        """初始化 PARAXIAL 表面处理器
        
        参数:
            wavelength_um: 波长 (μm)
        """
        self._wavelength_um = wavelength_um
        self._state_converter = StateConverter(wavelength_um)
    
    @property
    def wavelength_um(self) -> float:
        """波长 (μm)"""
        return self._wavelength_um
    
    def propagate(
        self,
        state: PropagationState,
        surface: "GlobalSurfaceDefinition",
        target_surface_index: int,
    ) -> PropagationState:
        """处理 PARAXIAL 表面
        
        使用 PROPER 的 prop_lens 应用薄透镜效果，
        这会正确更新 PROPER 的高斯光束参数和参考面。
        
        参数:
            state: 当前传播状态
            surface: PARAXIAL 表面定义（包含焦距）
            target_surface_index: 目标表面索引
        
        返回:
            处理后的传播状态
        
        **Validates: Requirements 19.4, 19.5, 19.6, 19.7**
        """
        import proper
        
        # 获取焦距
        focal_length_mm = surface.focal_length
        
        # 无穷焦距，无效果
        if np.isinf(focal_length_mm):
            return PropagationState(
                surface_index=target_surface_index,
                position='exit',
                amplitude=state.amplitude.copy(),
                phase=state.phase.copy(),
                pilot_beam_params=state.pilot_beam_params,
                proper_wfo=state.proper_wfo,
                optical_axis_state=state.optical_axis_state,
                grid_sampling=state.grid_sampling,
            )
        
        # 使用 PROPER 的 prop_lens 应用薄透镜效果
        # prop_lens 会：
        # 1. 应用正确的相位修正（考虑参考面变化）
        # 2. 更新高斯光束参数（w0, z_w0, z_Rayleigh）
        # 3. 更新参考面类型（PLANAR 或 SPHERI）
        focal_length_m = focal_length_mm * 1e-3
        proper.prop_lens(state.proper_wfo, focal_length_m)
        
        # 更新 Pilot Beam 参数（使用 ABCD 法则）
        new_pilot_params = state.pilot_beam_params.apply_lens(focal_length_mm)
        
        # 从 PROPER 提取新的振幅和相位
        new_amplitude, new_phase = self._state_converter.proper_to_amplitude_phase(
            state.proper_wfo,
            state.grid_sampling,
            pilot_beam_params=new_pilot_params,
        )
        
        return PropagationState(
            surface_index=target_surface_index,
            position='exit',
            amplitude=new_amplitude,
            phase=new_phase,
            pilot_beam_params=new_pilot_params,
            proper_wfo=state.proper_wfo,
            optical_axis_state=state.optical_axis_state,
            grid_sampling=state.grid_sampling,
        )


def compute_paraxial_phase_correction(
    focal_length_mm: float,
    grid_size: int,
    physical_size_mm: float,
    wavelength_um: float,
) -> NDArray[np.floating]:
    """计算薄透镜相位修正（独立函数版本）
    
    公式: φ(r) = -k × r² / (2f)
    
    注意：此函数仅用于测试和验证。
    实际使用时应通过 PROPER 的 prop_lens 应用薄透镜效果。
    
    参数:
        focal_length_mm: 焦距 (mm)
        grid_size: 网格大小
        physical_size_mm: 物理尺寸 (mm)
        wavelength_um: 波长 (μm)
    
    返回:
        相位修正数组 (弧度)
    
    **Validates: Requirements 19.4, 19.5**
    """
    # 创建坐标网格
    half_size = physical_size_mm / 2
    coords = np.linspace(-half_size, half_size, grid_size)
    X, Y = np.meshgrid(coords, coords)
    r_sq = X**2 + Y**2  # mm²
    
    # 计算波数
    wavelength_mm = wavelength_um * 1e-3
    k = 2 * np.pi / wavelength_mm
    
    # 相位修正: φ = -k × r² / (2f)
    phase_correction = -k * r_sq / (2 * focal_length_mm)
    
    return phase_correction
