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
    from sequential_system.coordinate_system import GlobalSurfaceDefinition
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
        这确保了"参考光轴"与物理光线追迹的"实际光轴"完全一致。
        
        返回:
            列表，每个元素是字典，包含：
            - 'intersection': 主光线与表面的交点位置 (mm)，全局坐标系
            - 'exit_direction': 主光线离开表面的方向（归一化），全局坐标系
        """
        from wavefront_to_rays.element_raytracer import ElementRaytracer
        
        # 如果没有表面，返回空列表
        if not self._optical_system:
            return []
        
        chief_ray_data = []
        
        # 初始主光线：从原点沿 +Z 方向
        current_pos = np.array([0.0, 0.0, 0.0])
        current_dir = np.array([0.0, 0.0, 1.0])
        
        for i, surface in enumerate(self._optical_system):
            # 1. 创建 SurfaceDefinition
            # 需要将 GlobalSurfaceDefinition 转换为 ElementRaytracer 可用的格式
            # 特别是正确处理姿态角
            # surface从全局坐标系转换到入射面坐标系
            surface_def = self._create_surface_definition_for_tracing(
                surface, current_dir
            )
            
            # 2. 创建 ElementRaytracer，将全局光线角度转换至入射面坐标系。
            # 仅包含当前这一个表面，且表面已经位于入射面坐标系。
            # 注意：ElementRaytracer 会自动处理光线追迹和坐标变换
            raytracer = ElementRaytracer(
                surfaces=[surface_def],
                wavelength=self._wavelength_um,
                chief_ray_direction=tuple(current_pos * 0 + current_dir), # Ensure copy/type
                entrance_position=tuple(current_pos),
            )
            
            # 3. 追迹主光线
            # 这会计算出射方向和交点
            raytracer.trace_chief_ray()
            
            # 4. 获取结果
            # 获取出射方向（全局）
            exit_dir = np.array(raytracer.exit_chief_direction)
            
            # 获取交点（全局）
            # 新增的 API
            intersection = np.array(raytracer.get_global_chief_ray_intersection())
            
            chief_ray_data.append({
                'intersection': intersection,
                'exit_direction': exit_dir,
            })
            #打印此处光线状态
            print(f"Surface {i}: Intersection {intersection}, Exit Direction {exit_dir}")
            # 5. 更新当前位置和方向
            # 下一个表面的入射点是当前表面的交点
            current_pos = intersection.copy()
            current_dir = exit_dir.copy()
        
        return chief_ray_data

    def _create_surface_definition_for_tracing(
        self,
        surface: "GlobalSurfaceDefinition",
        incident_dir: np.ndarray,
    ) -> "SurfaceDefinition":
        """创建用于光线追迹的 SurfaceDefinition
        
        参数:
            surface: 全局表面定义
            incident_dir: 入射光方向（归一化）
            
        返回:
            SurfaceDefinition 对象
        """
        from wavefront_to_rays.element_raytracer import (
            SurfaceDefinition,
            compute_rotation_matrix,
        )
        from scipy.spatial.transform import Rotation
        
        # 确定表面类型
        if surface.is_mirror:
            surface_type = 'mirror'
            material = 'mirror'
        else:
            surface_type = 'refract'
            material = surface.material
        
        # 1. 获取入射光坐标系的旋转矩阵 (R_beam)
        R_beam = compute_rotation_matrix(tuple(incident_dir))
        
        # 2. 获取表面全局姿态矩阵 (R_surf)
        R_surf = surface.orientation
        
        # 3. 计算相对旋转矩阵 (R_rel)
        # R_surf = R_beam @ R_rel  =>  R_rel = R_beam.T @ R_surf
        R_rel = R_beam.T @ R_surf
        
        # 4. 提取欧拉角 (rx, ry)
        # ⚠️ 关键修正：必须使用 'yxz' 顺序以匹配 optiland 的旋转约定 (Ry @ Rx)
        # optiland/ElementRaytracer 使用: Direction -> (rx, ry) -> Ry(ry) @ Rx(rx) @ [0,0,1]
        # 这是 intrinsic rotation sequence: y -> x
        euler_angles = Rotation.from_matrix(R_rel).as_euler('yxz', degrees=False)
        tilt_y = euler_angles[0]
        tilt_x = euler_angles[1]
        # tilt_z = euler_angles[2]  # 通常应接近 0
        
        return SurfaceDefinition(
            surface_type=surface_type,
            radius=surface.radius,
            thickness=surface.thickness,
            material=material,
            conic=surface.conic,
            tilt_x=tilt_x,
            tilt_y=tilt_y,
            vertex_position=tuple(surface.vertex_position),
        )

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
            if distance > 1e-6:
                current_state = self._free_space_propagator.propagate(
                    state=current_state,
                    target_axis_state=entrance_axis, # 传入当前表面的入口光轴状态
                    target_surface_index=i, # 传入当前表面的索引
                    target_position="entrance", # 传入当前表面的入口位置
                )
                total_path_length += distance
                current_path_length += distance
            
            # B. 元件传播
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
