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
<<<<<<< Updated upstream
=======
        use_global_raytracer: bool = False,
        debug: bool = False,
        debug_dir: Optional[str] = None,
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
=======
            use_global_raytracer: 是否使用全局坐标系光线追迹器
                - False: 使用 HybridElementPropagator（默认）
                - True: 使用 HybridElementPropagatorGlobal
            debug: 是否开启调试模式
            debug_dir: 调试输出目录（可选）
>>>>>>> Stashed changes
        
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
<<<<<<< Updated upstream
        self._hybrid_element_propagator = HybridElementPropagator(
            wavelength_um=wavelength_um,
            num_rays=num_rays,
            method=propagation_method,
        )
=======
        
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
                debug_dir=debug_dir,
            )
        
>>>>>>> Stashed changes
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
                # 如果没有追迹数据，回退到顶点位置
                intersection = surface.vertex_position.copy()
                if surface.is_mirror:
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
        
        手动计算主光线与每个表面的交点位置和出射方向。
        
        返回:
            列表，每个元素是字典，包含：
            - 'intersection': 主光线与表面的交点位置 (mm)，全局坐标系
            - 'exit_direction': 主光线离开表面的方向（归一化），全局坐标系
        
        说明:
            由于 optiland 在使用绝对坐标定位时存在坐标系问题，
            这里手动计算主光线的传播路径。
        """
        # 如果没有表面，返回空列表
        if not self._optical_system:
            return []
        
        chief_ray_data = []
        
<<<<<<< Updated upstream
        # 初始主光线：从原点沿 +Z 方向
        current_pos = np.array([0.0, 0.0, 0.0])
        current_dir = np.array([0.0, 0.0, 1.0])
=======
        current_state = PropagationState(
            surface_index=-1,
            position='source',
            amplitude=amplitude,
            phase=phase,
            pilot_beam_params=pilot_beam,
            proper_wfo=proper_wfo,
            optical_axis_state=source_axis_state,
            grid_sampling=grid_sampling,
            current_refractive_index=1.0, # 初始假设为真空/空气
        )
>>>>>>> Stashed changes
        
        for surface in self._optical_system:
            # 计算主光线与表面的交点
            intersection = self._compute_ray_surface_intersection(
                current_pos, current_dir, surface
            )
            
            if intersection is None:
                # 如果无法计算交点，使用顶点位置作为回退
                intersection = surface.vertex_position.copy()
            
            # 计算出射方向
            if surface.is_mirror:
                # 反射镜：计算反射方向
                exit_dir = self._compute_reflection_direction(
                    current_dir, intersection, surface
                )
            else:
                # 透射元件：方向不变（简化处理）
                exit_dir = current_dir.copy()
            
            chief_ray_data.append({
                'intersection': intersection,
                'exit_direction': exit_dir,
            })
            
            # 更新当前位置和方向
            current_pos = intersection.copy()
            current_dir = exit_dir.copy()
        
        return chief_ray_data
    
    def _compute_ray_surface_intersection(
        self,
        ray_origin: np.ndarray,
        ray_direction: np.ndarray,
        surface: "GlobalSurfaceDefinition",
    ) -> np.ndarray:
        """计算光线与表面的交点
        
        参数:
            ray_origin: 光线起点 (mm)
            ray_direction: 光线方向（归一化）
            surface: 表面定义
        
        返回:
            交点位置 (mm)，如果无法计算则返回 None
        """
        vertex = surface.vertex_position
        radius = surface.radius
        conic = surface.conic
        
        # 对于平面（radius = inf）
        if radius is None or abs(radius) > 1e10:
            # 平面方程：(P - vertex) · normal = 0
            # 其中 normal 是表面法向量（沿 Z 轴）
            normal = surface.orientation[:, 2]  # Z 轴方向
            denom = np.dot(ray_direction, normal)
            if abs(denom) < 1e-10:
                return None
            t = np.dot(vertex - ray_origin, normal) / denom
            if t < 0:
                return None
            return ray_origin + t * ray_direction
        
        # 对于二次曲面（球面、抛物面等）
        # 使用迭代方法求解
        return self._solve_conic_intersection(
            ray_origin, ray_direction, vertex, radius, conic, surface.orientation
        )
    
    def _solve_conic_intersection(
        self,
        ray_origin: np.ndarray,
        ray_direction: np.ndarray,
        vertex: np.ndarray,
        radius: float,
        conic: float,
        orientation: np.ndarray,
    ) -> np.ndarray:
        """求解光线与二次曲面的交点
        
        二次曲面方程（局部坐标系，顶点在原点）：
        z = (x² + y²) / (R * (1 + sqrt(1 - (1+k)(x² + y²)/R²)))
        
        对于抛物面 (k = -1)：z = (x² + y²) / (2R)
        """
        # 将光线转换到表面局部坐标系
        # 局部坐标系：原点在顶点，Z 轴沿表面法向
        ray_origin_local = orientation.T @ (ray_origin - vertex)
        ray_dir_local = orientation.T @ ray_direction
        
        # 对于抛物面 (conic = -1)
        if abs(conic + 1) < 1e-10:
            # 抛物面方程：z = (x² + y²) / (2R)
            # 光线：P(t) = O + t*D
            # 代入：O_z + t*D_z = ((O_x + t*D_x)² + (O_y + t*D_y)²) / (2R)
            
            ox, oy, oz = ray_origin_local
            dx, dy, dz = ray_dir_local
            R = radius
            
            # 展开：2R*(oz + t*dz) = (ox + t*dx)² + (oy + t*dy)²
            # 2R*oz + 2R*t*dz = ox² + 2*ox*t*dx + t²*dx² + oy² + 2*oy*t*dy + t²*dy²
            # t²*(dx² + dy²) + t*(2*ox*dx + 2*oy*dy - 2R*dz) + (ox² + oy² - 2R*oz) = 0
            
            a = dx**2 + dy**2
            b = 2*ox*dx + 2*oy*dy - 2*R*dz
            c = ox**2 + oy**2 - 2*R*oz
            
            if abs(a) < 1e-10:
                # 线性方程
                if abs(b) < 1e-10:
                    return None
                t = -c / b
            else:
                # 二次方程
                discriminant = b**2 - 4*a*c
                if discriminant < 0:
                    return None
                sqrt_disc = np.sqrt(discriminant)
                t1 = (-b - sqrt_disc) / (2*a)
                t2 = (-b + sqrt_disc) / (2*a)
                
                # 选择正的、较小的 t
                if t1 > 1e-10 and t2 > 1e-10:
                    t = min(t1, t2)
                elif t1 > 1e-10:
                    t = t1
                elif t2 > 1e-10:
                    t = t2
                else:
                    return None
            
            # 计算交点（全局坐标系）
            intersection_local = ray_origin_local + t * ray_dir_local
            intersection_global = orientation @ intersection_local + vertex
            return intersection_global
        
        # 对于一般二次曲面，使用迭代方法
        # 简化处理：使用球面近似
        c = 1.0 / radius  # 曲率
        
        # 球面方程：(x² + y² + z²) - 2*R*z = 0（顶点在原点）
        # 简化为：z = c*(x² + y²) / (1 + sqrt(1 - c²*(x² + y²)))
        
        ox, oy, oz = ray_origin_local
        dx, dy, dz = ray_dir_local
        
        # 使用牛顿迭代
        t = 0.0
        for _ in range(20):
            x = ox + t * dx
            y = oy + t * dy
            z = oz + t * dz
            
            r2 = x**2 + y**2
            
            # 球面 sag
            if abs(c) < 1e-10:
                sag = 0.0
                dsag_dr2 = 0.0
            else:
                denom = 1 + np.sqrt(max(0, 1 - (1 + conic) * c**2 * r2))
                if abs(denom) < 1e-10:
                    break
                sag = c * r2 / denom
                # 导数
                if 1 - (1 + conic) * c**2 * r2 > 0:
                    sqrt_term = np.sqrt(1 - (1 + conic) * c**2 * r2)
                    dsag_dr2 = c / denom + c * r2 * (1 + conic) * c**2 / (2 * sqrt_term * denom**2)
                else:
                    dsag_dr2 = c / denom
            
            # 残差
            f = z - sag
            
            # 导数
            df_dt = dz - dsag_dr2 * 2 * (x * dx + y * dy)
            
            if abs(df_dt) < 1e-10:
                break
            
            dt = -f / df_dt
            t += dt
            
            if abs(dt) < 1e-10:
                break
        
        if t < 0:
            return None
        
        intersection_local = ray_origin_local + t * ray_dir_local
        intersection_global = orientation @ intersection_local + vertex
        return intersection_global
    
    def _compute_reflection_direction(
        self,
        incident_dir: np.ndarray,
        intersection: np.ndarray,
        surface: "GlobalSurfaceDefinition",
    ) -> np.ndarray:
        """计算反射方向
        
        参数:
            incident_dir: 入射方向（归一化）
            intersection: 交点位置 (mm)
            surface: 表面定义
        
        返回:
            反射方向（归一化）
        
        注意:
            对于抛物面（conic = -1），使用抛物面的光学性质直接计算反射方向，
            而不是使用几何法向量。这是因为几何法向量在离轴配置下会给出错误的结果。
        """
        vertex = surface.vertex_position
        radius = surface.radius
        conic = surface.conic
        orientation = surface.orientation
        
        # 检查是否是抛物面
        is_parabola = (radius is not None and 
                       abs(radius) < 1e10 and 
                       conic is not None and 
                       abs(conic + 1) < 1e-6)
        
        if is_parabola:
            # 对于抛物面，使用光学性质计算反射方向
            # 
            # ⚠️ 关键修复：正确计算焦点位置
            # 
            # 抛物面方程（局部坐标系，顶点在原点）：z = (x² + y²) / (2R)
            # 焦点在局部坐标系中的位置：(0, 0, f)，其中 f = R/2
            # 
            # 当抛物面顶点在全局坐标 vertex_position 时：
            # 焦点在全局坐标系中的位置 = vertex_position + f × 光学轴方向
            # 
            # 对于离轴抛物面（OAP）：
            # - 主光线从 (0, 0, 0) 沿 +Z 方向出发
            # - 抛物面顶点在 (0, y_offset, 0)
            # - 焦点在 (0, y_offset, f)
            # - 主光线与抛物面的交点在 (0, 0, y_offset²/(2R))
            # - 反射方向应该指向焦点
            
            # 抛物面的光学轴方向（局部坐标系中是 Z 轴）
            optical_axis_local = np.array([0.0, 0.0, 1.0])
            optical_axis_global = orientation @ optical_axis_local
            
            # 焦距
            f = radius / 2
            
            # ⚠️ 关键修复：焦点位置 = 顶点位置 + f × 光学轴方向
            focus_global = vertex + f * optical_axis_global
            
            # 检查入射光线是否平行于光学轴
            dot_product = abs(np.dot(incident_dir, optical_axis_global))
            
            if dot_product > 0.999:  # 几乎平行于光学轴
                # 反射方向指向焦点
                to_focus = focus_global - intersection
                norm = np.linalg.norm(to_focus)
                if norm > 1e-10:
                    reflected_dir = to_focus / norm
                else:
                    # 交点就在焦点，使用光学轴方向
                    reflected_dir = optical_axis_global.copy()
                
                # 如果入射方向与光学轴反向，反射方向也要调整
                if np.dot(incident_dir, optical_axis_global) < 0:
                    # 从焦点发出的光线反射后平行于光轴
                    reflected_dir = optical_axis_global.copy()
                
                return reflected_dir
            else:
                # 入射光线不平行于光学轴，使用正确的法向量公式
                # 对于抛物面，正确的法向量是：n = (u + d) / |u + d|
                # 其中 u = (P - F) / |P - F| 是从焦点到交点的单位向量
                # d 是入射方向
                
                # 从焦点到交点的单位向量
                P_minus_F = intersection - focus_global
                u = P_minus_F / np.linalg.norm(P_minus_F)
                
                # 计算法向量
                n_sum = u + incident_dir
                normal = n_sum / np.linalg.norm(n_sum)
                
                # 确保法向量指向入射光来的方向
                if np.dot(normal, incident_dir) > 0:
                    normal = -normal
                
                # 反射公式
                dot_dn = np.dot(incident_dir, normal)
                reflected_dir = incident_dir - 2 * dot_dn * normal
                
                norm = np.linalg.norm(reflected_dir)
                if norm > 1e-10:
                    reflected_dir = reflected_dir / norm
                
                return reflected_dir
        
        # 对于非抛物面，使用标准的几何法向量方法
        normal = self._compute_surface_normal(intersection, surface)
        
        # 确保法向量指向入射光来的方向
        if np.dot(normal, incident_dir) > 0:
            normal = -normal
        
        # 反射公式：r = d - 2(d·n)n
        dot_dn = np.dot(incident_dir, normal)
        reflected_dir = incident_dir - 2 * dot_dn * normal
        
        # 归一化
        norm = np.linalg.norm(reflected_dir)
        if norm > 1e-10:
            reflected_dir = reflected_dir / norm
        
        return reflected_dir
    
    def _compute_surface_normal(
        self,
        point: np.ndarray,
        surface: "GlobalSurfaceDefinition",
    ) -> np.ndarray:
        """计算表面在指定点处的法向量
        
        参数:
            point: 表面上的点 (mm)，全局坐标系
            surface: 表面定义
        
        返回:
            法向量（归一化），指向 +Z 方向（表面外侧）
        """
        vertex = surface.vertex_position
        radius = surface.radius
        conic = surface.conic
        orientation = surface.orientation
        
        # 将点转换到局部坐标系
        point_local = orientation.T @ (point - vertex)
        x, y, z = point_local
        
        # 对于平面
        if radius is None or abs(radius) > 1e10:
            # 法向量沿 Z 轴
            normal_local = np.array([0.0, 0.0, 1.0])
        else:
            # 对于二次曲面
            # 曲面方程：z = f(x, y)
            # 法向量：(-∂f/∂x, -∂f/∂y, 1) / |...|
            
            R = radius
            k = conic
            r2 = x**2 + y**2
            
            if abs(k + 1) < 1e-10:
                # 抛物面：z = (x² + y²) / (2R)
                # ∂z/∂x = x/R, ∂z/∂y = y/R
                dz_dx = x / R
                dz_dy = y / R
            else:
                # 一般二次曲面
                c = 1.0 / R
                denom_sq = 1 - (1 + k) * c**2 * r2
                if denom_sq > 0:
                    sqrt_denom = np.sqrt(denom_sq)
                    denom = 1 + sqrt_denom
                    # ∂z/∂x = c*2x/denom + c*r2 * (1+k)*c²*2x / (2*sqrt_denom*denom²)
                    dz_dx = c * 2 * x / denom
                    dz_dy = c * 2 * y / denom
                    if sqrt_denom > 1e-10:
                        factor = (1 + k) * c**2 / (sqrt_denom * denom**2)
                        dz_dx += c * r2 * factor * x
                        dz_dy += c * r2 * factor * y
                else:
                    dz_dx = x / R
                    dz_dy = y / R
            
            normal_local = np.array([-dz_dx, -dz_dy, 1.0])
            normal_local = normal_local / np.linalg.norm(normal_local)
        
        # 转换到全局坐标系
        normal_global = orientation @ normal_local
        
        return normal_global
    
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
        
        # 如果距离很小，跳过传播但仍记录入射状态
        if abs(distance) < 1e-10:
            # 即使不传播，也记录入射状态以便调试接口可以获取光轴信息
            entrance_state = PropagationState(
                surface_index=surface_index,
                position='entrance',
                amplitude=self._current_state.amplitude.copy(),
                phase=self._current_state.phase.copy(),
                pilot_beam_params=self._current_state.pilot_beam_params,
                proper_wfo=self._current_state.proper_wfo,
                optical_axis_state=entrance_axis,
                grid_sampling=self._current_state.grid_sampling,
            )
            self._current_state = entrance_state
            self._surface_states.append(entrance_state)
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

