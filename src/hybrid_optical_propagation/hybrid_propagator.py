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
from .hybrid_element_propagator_global import HybridElementPropagatorGlobal
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
    from sequential_system.coordinate_system import GlobalSurfaceDefinition, ZemaxToOptilandConverter
    from optiland.rays import RealRays
    from sequential_system.coordinate_tracking import OpticalAxisState
    from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition


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
        num_rays: int = 1000,
        propagation_method: str = "local_raytracing",
        use_global_raytracer: bool = False,
        debug: bool = False,
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
            use_global_raytracer: 是否使用全局坐标系光线追迹器
                - False: 使用 HybridElementPropagator（默认）
                - True: 使用 HybridElementPropagatorGlobal
        
        **Validates: Requirements 16.1**
        """
        self._optical_system = optical_system
        self._source = source
        self._wavelength_um = wavelength_um
        self._grid_size = grid_size
        self._num_rays = num_rays
        self._propagation_method = propagation_method
        self._use_global_raytracer = use_global_raytracer
        
        # 内部状态
        self._current_state: Optional[PropagationState] = None
        self._surface_states: List[PropagationState] = []
        self._optical_axis_states: Dict[int, "OpticalAxisState"] = {}
        
        # 子组件
        self._state_converter = StateConverter(wavelength_um)
        self._free_space_propagator = FreeSpacePropagator(wavelength_um)
        
        # 选择元件传播器
        if use_global_raytracer:
            self._hybrid_element_propagator = HybridElementPropagatorGlobal(
                wavelength_um=wavelength_um,
                num_rays=num_rays,
            )
        else:
            self._hybrid_element_propagator = HybridElementPropagator(
                wavelength_um=wavelength_um,
                num_rays=num_rays,
                method=propagation_method,
                debug=debug,
            )
        
        self._paraxial_propagator = ParaxialPhasePropagator(wavelength_um)
        
        # 预计算光轴状态
        self._precompute_optical_axis()
    
    def _precompute_optical_axis(self) -> None:
        """预计算所有表面处的光轴状态
        
        通过追迹主光线穿过整个系统，获取主光线与每个表面的实际交点位置和出射方向。
        
        关键改进：
        - 使用实际主光线交点位置，而不是表面顶点位置
        - 对于离轴系统（如 OAP），主光线不经过顶点
        - 使用 optiland 的光线追迹获取准确的交点和方向
        
        **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7**
        """
        from sequential_system.coordinate_tracking import (
            OpticalAxisState,
            Position3D,
            RayDirection,
        )
        
        # 追迹主光线穿过整个系统，获取每个表面的交点和方向
        chief_ray_data = self._trace_chief_ray_through_system()
        
        # 初始光轴状态：原点，沿 +Z 方向
        current_direction = np.array([0.0, 0.0, 1.0])
        path_length = 0.0
        previous_position = np.array([0.0, 0.0, 0.0])
        
        for i, surface in enumerate(self._optical_system):
            # 获取主光线在该表面的交点和出射方向
            if i < len(chief_ray_data):
                intersection = chief_ray_data[i]['intersection']
                exit_direction = chief_ray_data[i]['exit_direction']
            else:
                #不应该出现这种情况，如果出现，报警
                print(f"Warning: No chief ray data for surface {i}")
                # 如果没有追迹数据，回退到顶点位置
                intersection = surface.vertex_position.copy()
                if surface.is_mirror:
                    # Fallback
                    normal = surface.surface_normal
                    d = current_direction
                    n = normal
                    exit_direction = d - 2 * np.dot(d, n) * n
                    exit_direction = exit_direction / np.linalg.norm(exit_direction)
                else:
                    exit_direction = current_direction.copy()
            
            # 计算到当前表面的距离
            displacement = intersection - previous_position
            distance = np.linalg.norm(displacement)
            
            # 更新光程
            path_length += distance
            
            # 入射光轴状态（使用实际交点位置）
            entrance_state = OpticalAxisState(
                position=Position3D.from_array(intersection),
                direction=RayDirection.from_array(current_direction),
                path_length=path_length,
            )
            
            # 出射光轴状态（使用实际交点位置和出射方向）
            exit_state = OpticalAxisState(
                position=Position3D.from_array(intersection),
                direction=RayDirection.from_array(exit_direction),
                path_length=path_length,
            )
            
            # 存储状态
            self._optical_axis_states[i] = {
                'entrance': entrance_state,
                'exit': exit_state,
            }
            
            # 更新当前位置和方向
            previous_position = intersection.copy()
            current_direction = exit_direction.copy()

    def _trace_chief_ray_through_system(self) -> list:
        """追迹主光线穿过整个系统
        
        使用 ElementRaytracer (Optiland) 追迹主光线，
        获取精确的主光线与每个表面的交点位置和出射方向。
        
        Delegates to the new robust system-level tracer.
        Kept for backward internal compatibility.
        """
        return self._trace_chief_ray_optiland()

    def _trace_chief_ray_optiland(self) -> List[Dict[str, np.ndarray]]:
        """
        Perform robust system-level tracing using optiland and extract global coordinate data.
        
        Returns:
            List of dictionaries containing 'intersection' (x, y, z) and 
            'exit_direction' (L, M, N) for each surface (excluding the dummy object surface).
        """
        from optiland.optic import Optic
        from optiland.rays import RealRays
        from sequential_system.coordinate_system import ZemaxToOptilandConverter
        
        # 1. Convert to Optic using existing logic
        # This handles creating the full optiland model from our surfaces
        converter = ZemaxToOptilandConverter(
            self._optical_system,
            wavelength=self._wavelength_um,
            entrance_pupil_diameter=10.0 # Default/Placeholder, doesn't affect single ray trace
        )
        optic = converter.convert()
        
        # 2. Define Chief Ray
        # Start at global origin (0,0,0) pointing along Z (0,0,1) in Pupil/Object space
        # Optiland handles the object->first surface transfer
        rays = RealRays(
            x=[0], y=[0], z=[0], 
            L=[0], M=[0], N=[1], 
            intensity=[1.0],
            wavelength=self._wavelength_um
        )
        
        # 3. Trace
        # This propagates rays through ALL surfaces.
        # optiland.surfaces.Surface._record() stores GLOBAL coordinates of:
        # - Intersection (x, y, z)
        # - Exit Direction (L, M, N) -> AFTER interaction and globalization
        optic.surface_group.trace(rays)
        
        # 4. Extract Data & Map Indices
        chief_ray_data = []
        
        # Use optic.surface_group.surfaces
        # Mapping: self._optical_system[i] corresponds to optic.surface_group.surfaces[i + 1]
        # (Because ZemaxToOptilandConverter adds a dummy Object Surface at index 0)
        
        for i in range(len(self._optical_system)):
            # Access corresponding optiland surface (skip object surface 0)
            opt_surface_index = i + 1
            if opt_surface_index >= len(optic.surface_group.surfaces):
                 break 

            opt_surface = optic.surface_group.surfaces[opt_surface_index]
            
            # Extract Global Intersection (x, y, z)
            # opt_surface.x is an array (size 1 for 1 ray), take item 0
            if len(opt_surface.x) > 0:
                intersection = np.array([
                    opt_surface.x[0],
                    opt_surface.y[0],
                    opt_surface.z[0]
                ])
                
                # Extract Global Exit Direction (L, M, N)
                exit_direction = np.array([
                    opt_surface.L[0],
                    opt_surface.M[0],
                    opt_surface.N[0]
                ])
                
                print(f"[DEBUG_CHIEF] Map User Surf {i} -> Optiland Surf {opt_surface_index} ({type(opt_surface).__name__})")
                print(f"    Intersection: {intersection}")
                print(f"    Exit Dir    : {exit_direction}")

            else:
                # Fallback if ray was clipped/stopped (shouldn't happen for chief ray usually)
                print(f"Warning: Chief ray stopped at surface {i}")
                intersection = np.zeros(3)
                exit_direction = np.array([0., 0., 1.])

            chief_ray_data.append({
                'intersection': intersection,
                'exit_direction': exit_direction,
            })
            
        return chief_ray_data



    def propagate(self) -> PropagationResult:
        """执行混合传播
        
        按照光学系统的顺序，依次进行自由空间传播和元件传播。
        
        返回:
            传播结果对象，包含最终状态和所有中间状态
        
        **Validates: Requirements 16.3**
        """
        from sequential_system.coordinate_tracking import (
            OpticalAxisState,
            Position3D,
            RayDirection,
        )

        # 1. 初始化传播状态
        amplitude, phase, pilot_beam, proper_wfo = self._source.create_initial_wavefront()
        grid_sampling = self._source.get_grid_sampling()
        
        # 初始光轴状态：原点，沿 +Z 方向
        # 注意：这里假设光源位于 (0,0,0) 并沿 +Z 传播
        # 如果 self._source 有位置信息，应使用它
        source_axis_state = OpticalAxisState(
            position=Position3D(0.0, 0.0, 0.0),
            direction=RayDirection(0.0, 0.0, 1.0),
            path_length=0.0
        )
        
        current_state = PropagationState(
            surface_index=-1,
            position='source',
            amplitude=amplitude,
            phase=phase,
            pilot_beam_params=pilot_beam,
            proper_wfo=proper_wfo,
            optical_axis_state=source_axis_state,
            grid_sampling=grid_sampling,
        )
        
        self._current_state = current_state
        self._surface_states = []
        
        total_path_length = 0.0
        current_path_length = 0.0 # 当前累积光程
        
        # 2. 依次经过每个表面
        for i, surface in enumerate(self._optical_system):
            # 获取当前表面的光轴状态
            axis_state = self._optical_axis_states[i]
            entrance_axis = axis_state['entrance']
            exit_axis = axis_state['exit']
            
            # 计算到该表面的传播距离
            # distance = surface_path_length - current_total_path_length
            distance = entrance_axis.path_length - current_path_length
            
            # A. 自由空间传播 (如果距离 > 0)

            ##这里很大的不足，需要考虑传播距离负向（反向倒退光束）的情况。
            if distance > 1e-6:
                current_state = self._free_space_propagator.propagate(
                    state=current_state,
                    target_axis_state=entrance_axis, # 传入当前表面的入口光轴状态
                    target_surface_index=i, # 传入当前表面的索引
                    target_position="entrance", # 传入当前表面的入口位置
                )
                total_path_length += distance
                current_path_length += distance
            
            # B. 元件传播 / 表面交互
            # 优化：智能检测是否需要执行光线追迹
            prev_surface = self._optical_system[i-1] if i > 0 else None
            interaction_type = classify_surface_interaction(surface, prev_surface)
            
            # 对于单纯的自由空间传播（如无光焦度的空气面）或坐标断点，跳过光线追迹
            if interaction_type in ['free_space', 'coordinate_break']:
                # 直接通过 (Pass-through)
                # 使用 FreeSpacePropagator 将状态更新到 exit_axis (距离通常为 0)
                # 这会正确更新 optical_axis_state, position, index 等元数据
                current_state = self._free_space_propagator.propagate(
                    state=current_state,
                    target_axis_state=exit_axis,
                    target_surface_index=i,
                    target_position="exit",
                )
            else:
                # 执行混合元件传播 (光线追迹 + 波前调制)
                current_state = self._hybrid_element_propagator.propagate(
                    current_state,
                    surface,
                    entrance_axis,
                    exit_axis,
                    target_surface_index=i,
                )
            
            # 存储中间状态
            self._surface_states.append(current_state)
            
            # 更新当前光程
            # 注意：元件本身可能也贡献了 path length? 通常默认为 0 (薄元件)
            # 光轴追踪系统已经考虑了厚度
            # 更新为 exit_axis 的 path_length
            current_path_length = exit_axis.path_length
        
        # 确保最终状态的光程是正确的
        total_path_length = current_path_length
        
        return PropagationResult(
            final_state=current_state,
            surface_states=self._surface_states,
            total_path_length=total_path_length,
            success=True
        )

    def get_optical_axis_at_surface(
        self,
        surface_index: int,
        position: str = 'entrance',
    ) -> Optional["OpticalAxisState"]:
        """获取指定表面的光轴状态
        
        参数:
            surface_index: 表面索引
            position: 'entrance' 或 'exit'
        
        返回:
            光轴状态对象，如果索引无效则返回 None
        """
        if surface_index not in self._optical_axis_states:
            return None
        
        state_dict = self._optical_axis_states[surface_index]
        return state_dict.get(position)
