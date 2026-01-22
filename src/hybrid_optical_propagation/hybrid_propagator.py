"""
混合光学传播器主类

本模块实现 HybridOpticalPropagator 主类，协调 PROPER 物理光学传输
和 optiland 几何光线追迹，实现完整的混合光学传播仿真。

主要功能：
- 光轴追踪集成
- 入射面/出射面定义
- 传播流程协调
- 状态管理

**Validates: Requirements 2.1-2.7, 3.1-3.6, 16.1-16.6**
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
import warnings

from .data_models import (
    PilotBeamParams,
    GridSampling,
    PropagationState,
    SourceDefinition,
)
from .state_converter import StateConverter
from .free_space_propagator import FreeSpacePropagator, compute_propagation_distance
from .hybrid_element_propagator import HybridElementPropagator
from .paraxial_propagator import ParaxialPhasePropagator
from .material_detection import (
    detect_material_change,
    is_paraxial_surface,
    is_coordinate_break,
    classify_surface_interaction,
)
from .exceptions import (
    HybridPropagationError,
    RayTracingError,
    GridSamplingError,
)

if TYPE_CHECKING:
    from sequential_system.coordinate_system import GlobalSurfaceDefinition
    from sequential_system.coordinate_tracking import OpticalAxisState


@dataclass
class PropagationResult:
    """传播结果
    
    存储完整传播过程的结果，包括最终波前和中间状态。
    
    属性:
        final_state: 最终传播状态
        surface_states: 各表面处的传播状态列表
        total_path_length: 总光程 (mm)
        success: 传播是否成功
        error_message: 错误信息（如果失败）
    """
    final_state: PropagationState
    surface_states: List[PropagationState]
    total_path_length: float
    success: bool = True
    error_message: str = ""
    
    def get_final_wavefront(self) -> NDArray[np.complexfloating]:
        """获取最终波前复振幅
        
        注意：返回的复振幅会有相位折叠。如需非折叠相位，
        请使用 get_final_phase() 或 final_state.phase。
        """
        return self.final_state.get_complex_amplitude()
    
    def get_final_amplitude(self) -> NDArray[np.floating]:
        """获取最终振幅分布"""
        return self.final_state.amplitude
    
    def get_final_intensity(self) -> NDArray[np.floating]:
        """获取最终光强分布"""
        return self.final_state.get_intensity()
    
    def get_final_phase(self) -> NDArray[np.floating]:
        """获取最终相位分布（非折叠）"""
        return self.final_state.get_phase()



class HybridOpticalPropagator:
    """混合光学传播器
    
    协调 PROPER 物理光学传输和 optiland 几何光线追迹，
    实现完整的混合光学传播仿真。
    
    属性:
        optical_system: 光学系统定义（GlobalSurfaceDefinition 列表）
        source: 入射波面定义
        wavelength_um: 波长 (μm)
        grid_size: 网格大小
        num_rays: 光线采样数量
        propagation_method: 元件传播方法
    
    使用示例:
        >>> from hybrid_optical_propagation import (
        ...     HybridOpticalPropagator,
        ...     SourceDefinition,
        ... )
        >>> 
        >>> # 定义入射波面
        >>> source = SourceDefinition(
        ...     wavelength_um=0.633,
        ...     w0_mm=5.0,
        ...     z0_mm=0.0,
        ...     grid_size=512,
        ...     physical_size_mm=50.0,
        ... )
        >>> 
        >>> # 创建传播器
        >>> propagator = HybridOpticalPropagator(
        ...     optical_system=surfaces,
        ...     source=source,
        ...     wavelength_um=0.633,
        ... )
        >>> 
        >>> # 执行传播
        >>> result = propagator.propagate()
    
    **Validates: Requirements 16.1, 16.2, 16.3, 16.4**
    """
    
    def __init__(
        self,
        optical_system: List["GlobalSurfaceDefinition"],
        source: SourceDefinition,
        wavelength_um: float,
        grid_size: int = 512,
        num_rays: int = 200,
        propagation_method: str = "local_raytracing",
    ) -> None:
        """初始化混合光学传播器
        
        参数:
            optical_system: 光学系统定义（GlobalSurfaceDefinition 列表）
            source: 入射波面定义
            wavelength_um: 波长 (μm)
            grid_size: 网格大小，默认 512
            num_rays: 光线采样数量，默认 200
            propagation_method: 元件传播方法
                - 'local_raytracing': 局部光线追迹方法（默认）
                - 'pure_diffraction': 纯衍射方法
        
        **Validates: Requirements 16.1**
        """
        self._optical_system = optical_system
        self._source = source
        self._wavelength_um = wavelength_um
        self._grid_size = grid_size
        self._num_rays = num_rays
        self._propagation_method = propagation_method
        
        # 内部状态
        self._current_state: Optional[PropagationState] = None
        self._surface_states: List[PropagationState] = []
        self._optical_axis_states: Dict[int, "OpticalAxisState"] = {}
        
        # 子组件
        self._state_converter = StateConverter(wavelength_um)
        self._free_space_propagator = FreeSpacePropagator(wavelength_um)
        self._hybrid_element_propagator = HybridElementPropagator(
            wavelength_um=wavelength_um,
            num_rays=num_rays,
            method=propagation_method,
        )
        self._paraxial_propagator = ParaxialPhasePropagator(wavelength_um)
        
        # 预计算光轴状态
        self._precompute_optical_axis()
    
    def _precompute_optical_axis(self) -> None:
        """预计算所有表面处的光轴状态
        
        遍历光学系统，计算每个表面处的入射和出射光轴状态。
        
        **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7**
        """
        from sequential_system.coordinate_tracking import (
            OpticalAxisState,
            Position3D,
            RayDirection,
        )
        
        # 初始光轴状态：原点，沿 +Z 方向
        current_position = np.array([0.0, 0.0, 0.0])
        current_direction = np.array([0.0, 0.0, 1.0])
        path_length = 0.0
        
        for i, surface in enumerate(self._optical_system):
            # 计算到当前表面的距离
            vertex = surface.vertex_position
            displacement = vertex - current_position
            distance = np.linalg.norm(displacement)
            
            # 更新光程
            path_length += distance
            
            # 入射光轴状态
            entrance_state = OpticalAxisState(
                position=Position3D.from_array(vertex),
                direction=RayDirection.from_array(current_direction),
                path_length=path_length,
            )
            
            # 计算出射方向（反射镜改变方向）
            if surface.is_mirror:
                # 表面法向量
                normal = surface.surface_normal
                # 反射公式: r = d - 2(d·n)n
                d = current_direction
                n = normal
                exit_direction = d - 2 * np.dot(d, n) * n
                exit_direction = exit_direction / np.linalg.norm(exit_direction)
            else:
                exit_direction = current_direction.copy()
            
            # 出射光轴状态
            exit_state = OpticalAxisState(
                position=Position3D.from_array(vertex),
                direction=RayDirection.from_array(exit_direction),
                path_length=path_length,
            )
            
            # 存储状态
            self._optical_axis_states[i] = {
                'entrance': entrance_state,
                'exit': exit_state,
            }
            
            # 更新当前位置和方向
            current_position = vertex.copy()
            current_direction = exit_direction.copy()
    
    def get_optical_axis_at_surface(
        self,
        surface_index: int,
        position: str = 'entrance',
    ) -> "OpticalAxisState":
        """获取指定表面处的光轴状态
        
        参数:
            surface_index: 表面索引
            position: 'entrance' 或 'exit'
        
        返回:
            光轴状态
        
        **Validates: Requirements 2.1, 2.2**
        """
        if surface_index not in self._optical_axis_states:
            raise ValueError(f"无效的表面索引: {surface_index}")
        
        return self._optical_axis_states[surface_index][position]


    def _create_entrance_plane(
        self,
        surface_index: int,
    ) -> Dict[str, Any]:
        """创建入射面定义
        
        入射面垂直于入射光轴，原点为主光线与表面的交点。
        
        参数:
            surface_index: 表面索引
        
        返回:
            入射面定义字典
        
        **Validates: Requirements 3.1, 3.2**
        """
        surface = self._optical_system[surface_index]
        axis_state = self.get_optical_axis_at_surface(surface_index, 'entrance')
        
        return {
            'origin': surface.vertex_position.copy(),
            'normal': axis_state.direction.to_array(),
            'surface_index': surface_index,
            'position': 'entrance',
        }
    
    def _create_exit_plane(
        self,
        surface_index: int,
    ) -> Dict[str, Any]:
        """创建出射面定义
        
        出射面垂直于出射光轴，原点为主光线与表面的交点。
        
        参数:
            surface_index: 表面索引
        
        返回:
            出射面定义字典
        
        **Validates: Requirements 3.3, 3.4**
        """
        surface = self._optical_system[surface_index]
        axis_state = self.get_optical_axis_at_surface(surface_index, 'exit')
        
        return {
            'origin': surface.vertex_position.copy(),
            'normal': axis_state.direction.to_array(),
            'surface_index': surface_index,
            'position': 'exit',
        }
    
    def _initialize_propagation(self) -> PropagationState:
        """初始化传播状态
        
        创建初始波前并设置初始传播状态。
        
        返回:
            初始传播状态
        
        **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6**
        """
        from sequential_system.coordinate_tracking import (
            OpticalAxisState,
            Position3D,
            RayDirection,
        )
        
        # 创建初始波前（振幅和相位分离）
        amplitude, phase, pilot_beam_params, proper_wfo = (
            self._source.create_initial_wavefront()
        )
        
        # 初始光轴状态
        initial_axis_state = OpticalAxisState(
            position=Position3D(0.0, 0.0, 0.0),
            direction=RayDirection(0.0, 0.0, 1.0),
            path_length=0.0,
        )
        
        # 网格采样信息
        grid_sampling = self._source.get_grid_sampling()
        
        # 创建初始状态
        initial_state = PropagationState(
            surface_index=-1,  # 表示初始位置
            position='source',
            amplitude=amplitude,
            phase=phase,
            pilot_beam_params=pilot_beam_params,
            proper_wfo=proper_wfo,
            optical_axis_state=initial_axis_state,
            grid_sampling=grid_sampling,
        )
        
        return initial_state
    
    def propagate(self) -> PropagationResult:
        """执行完整传播
        
        遍历所有光学表面，根据材质变化选择传播方法，
        更新状态并记录中间结果。
        
        返回:
            PropagationResult 包含最终波前和中间结果
        
        **Validates: Requirements 16.2**
        """
        try:
            # 初始化
            self._current_state = self._initialize_propagation()
            self._surface_states = [self._current_state]
            
            # 遍历所有表面
            for i, surface in enumerate(self._optical_system):
                # 跳过坐标断点
                if is_coordinate_break(surface):
                    continue
                
                # 传播到当前表面
                self._propagate_to_surface(i)
            
            # 计算总光程
            total_path = 0.0
            if self._surface_states:
                last_state = self._surface_states[-1]
                if last_state.optical_axis_state:
                    total_path = last_state.optical_axis_state.path_length
            
            return PropagationResult(
                final_state=self._current_state,
                surface_states=self._surface_states.copy(),
                total_path_length=total_path,
                success=True,
            )
            
        except Exception as e:
            return PropagationResult(
                final_state=self._current_state,
                surface_states=self._surface_states.copy(),
                total_path_length=0.0,
                success=False,
                error_message=str(e),
            )
    
    def _propagate_to_surface(self, surface_index: int) -> None:
        """传播到指定表面
        
        根据材质变化选择传播方法：
        - 同材质：自由空间传播
        - 材质变化：混合元件传播
        - PARAXIAL 表面：薄相位元件处理
        
        参数:
            surface_index: 目标表面索引
        
        **Validates: Requirements 11.1, 11.2, 11.3, 11.4, 11.5**
        """
        surface = self._optical_system[surface_index]
        
        # 获取前一个表面（用于材质变化检测）
        prev_surface = None
        if surface_index > 0:
            prev_surface = self._optical_system[surface_index - 1]
        
        # 获取光轴状态
        entrance_axis = self.get_optical_axis_at_surface(surface_index, 'entrance')
        exit_axis = self.get_optical_axis_at_surface(surface_index, 'exit')
        
        # 1. 自由空间传播到入射面
        self._propagate_free_space_to_entrance(surface_index, entrance_axis)
        
        # 2. 根据表面类型选择处理方法
        if is_paraxial_surface(surface):
            # PARAXIAL 表面：薄相位元件处理
            self._process_paraxial_surface(surface_index, surface)
        elif detect_material_change(surface, prev_surface):
            # 材质变化：混合元件传播
            self._process_hybrid_element(surface_index, surface, entrance_axis, exit_axis)
        else:
            # 同材质：只更新光轴状态
            self._update_state_for_same_material(surface_index, exit_axis)


    def _propagate_free_space_to_entrance(
        self,
        surface_index: int,
        entrance_axis: "OpticalAxisState",
    ) -> None:
        """自由空间传播到入射面
        
        参数:
            surface_index: 表面索引
            entrance_axis: 入射光轴状态
        """
        # 计算传播距离
        current_pos = self._current_state.optical_axis_state.position.to_array()
        target_pos = entrance_axis.position.to_array()
        current_dir = self._current_state.optical_axis_state.direction.to_array()
        
        distance = compute_propagation_distance(current_pos, target_pos, current_dir)
        
        # 如果距离很小，跳过传播
        if abs(distance) < 1e-10:
            return
        
        # 执行自由空间传播
        new_state = self._free_space_propagator.propagate(
            self._current_state,
            entrance_axis,
            surface_index,
            'entrance',
        )
        
        # 更新状态
        new_state = PropagationState(
            surface_index=surface_index,
            position='entrance',
            amplitude=new_state.amplitude,
            phase=new_state.phase,
            pilot_beam_params=new_state.pilot_beam_params,
            proper_wfo=new_state.proper_wfo,
            optical_axis_state=entrance_axis,
            grid_sampling=new_state.grid_sampling,
        )
        
        self._current_state = new_state
        self._surface_states.append(new_state)
    
    def _process_paraxial_surface(
        self,
        surface_index: int,
        surface: "GlobalSurfaceDefinition",
    ) -> None:
        """处理 PARAXIAL 表面
        
        参数:
            surface_index: 表面索引
            surface: 表面定义
        """
        new_state = self._paraxial_propagator.propagate(
            self._current_state,
            surface,
            surface_index,
        )
        
        # 更新状态
        new_state = PropagationState(
            surface_index=surface_index,
            position='exit',
            amplitude=new_state.amplitude,
            phase=new_state.phase,
            pilot_beam_params=new_state.pilot_beam_params,
            proper_wfo=new_state.proper_wfo,
            optical_axis_state=self._current_state.optical_axis_state,
            grid_sampling=new_state.grid_sampling,
        )
        
        self._current_state = new_state
        self._surface_states.append(new_state)
    
    def _process_hybrid_element(
        self,
        surface_index: int,
        surface: "GlobalSurfaceDefinition",
        entrance_axis: "OpticalAxisState",
        exit_axis: "OpticalAxisState",
    ) -> None:
        """处理混合元件传播
        
        参数:
            surface_index: 表面索引
            surface: 表面定义
            entrance_axis: 入射光轴状态
            exit_axis: 出射光轴状态
        """
        new_state = self._hybrid_element_propagator.propagate(
            self._current_state,
            surface,
            entrance_axis,
            exit_axis,
            surface_index,
        )
        
        # 更新状态
        new_state = PropagationState(
            surface_index=surface_index,
            position='exit',
            amplitude=new_state.amplitude,
            phase=new_state.phase,
            pilot_beam_params=new_state.pilot_beam_params,
            proper_wfo=new_state.proper_wfo,
            optical_axis_state=exit_axis,
            grid_sampling=new_state.grid_sampling,
        )
        
        self._current_state = new_state
        self._surface_states.append(new_state)
    
    def _update_state_for_same_material(
        self,
        surface_index: int,
        exit_axis: "OpticalAxisState",
    ) -> None:
        """更新同材质表面的状态
        
        参数:
            surface_index: 表面索引
            exit_axis: 出射光轴状态
        """
        # 只更新光轴状态，波前不变
        new_state = PropagationState(
            surface_index=surface_index,
            position='exit',
            amplitude=self._current_state.amplitude.copy(),
            phase=self._current_state.phase.copy(),
            pilot_beam_params=self._current_state.pilot_beam_params,
            proper_wfo=self._current_state.proper_wfo,
            optical_axis_state=exit_axis,
            grid_sampling=self._current_state.grid_sampling,
        )
        
        self._current_state = new_state
        self._surface_states.append(new_state)
    
    def propagate_to_surface(self, surface_index: int) -> PropagationState:
        """传播到指定表面
        
        参数:
            surface_index: 目标表面索引
        
        返回:
            该表面处的传播状态
        
        **Validates: Requirements 16.3**
        """
        if surface_index < 0 or surface_index >= len(self._optical_system):
            raise ValueError(f"无效的表面索引: {surface_index}")
        
        # 如果已经传播过，直接返回
        for state in self._surface_states:
            if state.surface_index == surface_index:
                return state
        
        # 否则执行传播
        self._current_state = self._initialize_propagation()
        self._surface_states = [self._current_state]
        
        for i in range(surface_index + 1):
            surface = self._optical_system[i]
            if is_coordinate_break(surface):
                continue
            self._propagate_to_surface(i)
        
        return self._current_state
    
    def get_wavefront_at_surface(self, surface_index: int) -> NDArray[np.complexfloating]:
        """获取指定表面的波前复振幅
        
        注意：返回的复振幅会有相位折叠。如需非折叠相位，
        请使用 get_state_at_surface(index).phase。
        
        参数:
            surface_index: 表面索引
        
        返回:
            复振幅数组 (grid_size × grid_size)
        
        **Validates: Requirements 16.4**
        """
        state = self.get_state_at_surface(surface_index)
        return state.get_complex_amplitude()
    
    def get_state_at_surface(self, surface_index: int) -> PropagationState:
        """获取指定表面的完整传播状态
        
        参数:
            surface_index: 表面索引
        
        返回:
            PropagationState 对象
        
        **Validates: Requirements 10.3**
        """
        # 查找已有状态
        for state in self._surface_states:
            if state.surface_index == surface_index:
                return state
        
        # 如果没有，执行传播
        return self.propagate_to_surface(surface_index)
    
    def get_grid_sampling(self, surface_index: int) -> GridSampling:
        """获取指定表面的网格采样信息
        
        参数:
            surface_index: 表面索引
        
        返回:
            GridSampling 对象
        
        **Validates: Requirements 17.5, 17.6, 17.7**
        """
        state = self.get_state_at_surface(surface_index)
        return state.grid_sampling
    
    @property
    def optical_system(self) -> List["GlobalSurfaceDefinition"]:
        """光学系统定义"""
        return self._optical_system
    
    @property
    def source(self) -> SourceDefinition:
        """入射波面定义"""
        return self._source
    
    @property
    def wavelength_um(self) -> float:
        """波长 (μm)"""
        return self._wavelength_um
    
    @property
    def grid_size(self) -> int:
        """网格大小"""
        return self._grid_size
    
    @property
    def num_rays(self) -> int:
        """光线采样数量"""
        return self._num_rays

