"""
自由空间传播器模块

本模块实现面与面之间的自由空间衍射传播。

核心功能：
1. 使用 PROPER 执行衍射传播
2. 支持正向和逆向传播
3. 同步更新仿真复振幅、Pilot Beam 参数和 PROPER 对象

**Validates: Requirements 4.1-4.8**
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from .data_models import PilotBeamParams, GridSampling, PropagationState
from .state_converter import StateConverter

if TYPE_CHECKING:
    from sequential_system.coordinate_tracking import OpticalAxisState


class FreeSpacePropagator:
    """自由空间传播器
    
    使用 PROPER 执行面与面之间的衍射传播，支持正向和逆向传播。
    
    属性:
        wavelength_um: 波长 (μm)
    
    **Validates: Requirements 4.1, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8**
    """
    
    def __init__(self, wavelength_um: float) -> None:
        """初始化自由空间传播器
        
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
        target_axis_state: "OpticalAxisState",
        target_surface_index: int,
        target_position: str = "entrance",
    ) -> PropagationState:
        """执行自由空间传播
        
        参数:
            state: 当前传播状态
            target_axis_state: 目标位置的光轴状态
            target_surface_index: 目标表面索引
            target_position: 目标位置类型 ('entrance' 或 'exit')
        
        返回:
            传播后的状态
        
        说明:
            - 传播距离由主光线交点连线计算
            - 如果方向与主光线相同，距离为正（正向传播）
            - 如果方向与主光线相反，距离为负（逆向传播）
        
        **Validates: Requirements 4.1, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8**
        """
        import proper
        
        # 计算传播距离
        current_position = state.optical_axis_state.position.to_array()
        target_position_arr = target_axis_state.position.to_array()
        current_direction = state.optical_axis_state.direction.to_array()
        
        distance_mm = self._compute_propagation_distance(
            current_position,
            target_position_arr,
            current_direction,
        )
        
        # 如果距离为零，直接返回更新后的状态
        if abs(distance_mm) < 1e-10:
            return PropagationState(
                surface_index=target_surface_index,
                position=target_position,
                amplitude=state.amplitude.copy(),
                phase=state.phase.copy(),
                pilot_beam_params=state.pilot_beam_params,
                proper_wfo=state.proper_wfo,
                optical_axis_state=target_axis_state,
                grid_sampling=state.grid_sampling,
            )
        
        # 使用 PROPER 执行传播
        distance_m = distance_mm * 1e-3
        proper.prop_propagate(state.proper_wfo, distance_m)
        
        # --------------------------------------------------------
        # 关键修正：从 PROPER 获取新的网格参数
        # PROPER 可能在传播过程中调整网格大小 (dx)，需要同步更新
        # --------------------------------------------------------
        wfo = state.proper_wfo
        new_dx_mm = wfo.dx * 1e3
        new_physical_size_mm = new_dx_mm * wfo.ngrid
        
        new_grid_sampling = GridSampling(
            grid_size=wfo.ngrid,
            physical_size_mm=new_physical_size_mm,
            sampling_mm=new_physical_size_mm / wfo.ngrid
        )
        
        # 更新 Pilot Beam 参数
        new_pilot_params = state.pilot_beam_params.propagate(distance_mm)
        
        # 从 PROPER 提取新的振幅和相位（使用新的网格采样）
        new_amplitude, new_phase = self._state_converter.proper_to_amplitude_phase(
            state.proper_wfo,
            new_grid_sampling,
            pilot_beam_params=new_pilot_params,
        )
        
        return PropagationState(
            surface_index=target_surface_index,
            position=target_position,
            amplitude=new_amplitude,
            phase=new_phase,
            pilot_beam_params=new_pilot_params,
            proper_wfo=state.proper_wfo,
            optical_axis_state=target_axis_state,
            grid_sampling=new_grid_sampling,
        )
    
    def propagate_distance(
        self,
        state: PropagationState,
        distance_mm: float,
        target_surface_index: int,
        target_position: str = "entrance",
        target_axis_state: "OpticalAxisState" = None,
    ) -> PropagationState:
        """执行指定距离的自由空间传播
        
        参数:
            state: 当前传播状态
            distance_mm: 传播距离 (mm)，正值为正向，负值为逆向
            target_surface_index: 目标表面索引
            target_position: 目标位置类型 ('entrance' 或 'exit')
            target_axis_state: 目标光轴状态（可选，如果不提供则从当前状态计算）
        
        返回:
            传播后的状态
        
        **Validates: Requirements 4.3, 4.4**
        """
        import proper
        
        # 如果距离为零，直接返回
        if abs(distance_mm) < 1e-10:
            return PropagationState(
                surface_index=target_surface_index,
                position=target_position,
                amplitude=state.amplitude.copy(),
                phase=state.phase.copy(),
                pilot_beam_params=state.pilot_beam_params,
                proper_wfo=state.proper_wfo,
                optical_axis_state=target_axis_state or state.optical_axis_state,
                grid_sampling=state.grid_sampling,
            )
        
        # 使用 PROPER 执行传播
        distance_m = distance_mm * 1e-3
        proper.prop_propagate(state.proper_wfo, distance_m)
        
        # 更新 Pilot Beam 参数
        new_pilot_params = state.pilot_beam_params.propagate(distance_mm)
        
        # 计算新的光轴状态（如果未提供）
        if target_axis_state is None:
            target_axis_state = state.optical_axis_state.propagate(distance_mm)
        
        # 从 PROPER 提取新的振幅和相位
        new_amplitude, new_phase = self._state_converter.proper_to_amplitude_phase(
            state.proper_wfo,
            state.grid_sampling,
            pilot_beam_params=new_pilot_params,
        )
        
        return PropagationState(
            surface_index=target_surface_index,
            position=target_position,
            amplitude=new_amplitude,
            phase=new_phase,
            pilot_beam_params=new_pilot_params,
            proper_wfo=state.proper_wfo,
            optical_axis_state=target_axis_state,
            grid_sampling=state.grid_sampling,
        )

    
    def _compute_propagation_distance(
        self,
        current_position: NDArray,
        target_position: NDArray,
        current_direction: NDArray,
    ) -> float:
        """计算传播距离（带符号）
        
        规则:
        1. 计算位移向量: displacement = target - current
        2. 计算距离绝对值: distance = |displacement|
        3. 判断方向:
           - 如果 displacement · current_direction > 0，距离为正（正向传播）
           - 如果 displacement · current_direction < 0，距离为负（逆向传播）
        
        参数:
            current_position: 当前位置 (mm)
            target_position: 目标位置 (mm)
            current_direction: 当前光轴方向（单位向量）
        
        返回:
            传播距离 (mm)，正值表示正向，负值表示逆向
        
        **Validates: Requirements 4.5, 4.6, 4.7**
        """
        displacement = target_position - current_position
        distance = np.linalg.norm(displacement)
        
        if distance < 1e-10:
            return 0.0
        
        # 判断方向
        direction = displacement / distance
        dot_product = np.dot(direction, current_direction)
        
        if dot_product < 0:
            distance = -distance
        
        return float(distance)


def compute_propagation_distance(
    current_position: NDArray,
    target_position: NDArray,
    current_direction: NDArray,
) -> float:
    """计算传播距离（带符号）- 独立函数版本
    
    规则:
    1. 计算位移向量: displacement = target - current
    2. 计算距离绝对值: distance = |displacement|
    3. 判断方向:
       - 如果 displacement · current_direction > 0，距离为正（正向传播）
       - 如果 displacement · current_direction < 0，距离为负（逆向传播）
    
    参数:
        current_position: 当前位置 (mm)
        target_position: 目标位置 (mm)
        current_direction: 当前光轴方向（单位向量）
    
    返回:
        传播距离 (mm)，正值表示正向，负值表示逆向
    
    **Validates: Requirements 4.5, 4.6, 4.7**
    """
    displacement = target_position - current_position
    distance = np.linalg.norm(displacement)
    
    if distance < 1e-10:
        return 0.0
    
    # 判断方向
    direction = displacement / distance
    dot_product = np.dot(direction, current_direction)
    
    if dot_product < 0:
        distance = -distance
    
    return float(distance)
