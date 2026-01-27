"""
全局坐标系混合元件传播器模块

本模块实现基于全局坐标系的混合传播仿真，使用 GlobalElementRaytracer 进行光线追迹。

核心功能：
1. 根据入射光轴方向定义入射面（垂直于光轴）
2. 将波前采样的光线转换到全局坐标系
3. 使用 GlobalElementRaytracer 进行光线追迹
4. 根据出射光轴方向定义出射面（垂直于光轴）
5. 将出射光线转换回局部坐标系进行重建

与 HybridElementPropagator 的主要区别：
- 使用全局坐标系进行光线追迹，避免局部坐标系转换
- 入射面和出射面显式定义（点+法向量）
- 支持复杂的折叠镜系统

**Validates: Requirements 5.1-5.4, 6.1-6.4, 7.1-7.3, 8.1-8.4, 9.1-9.4**
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from .data_models import PilotBeamParams, GridSampling, PropagationState
from .state_converter import StateConverter

if TYPE_CHECKING:
    from sequential_system.coordinate_tracking import OpticalAxisState
    from sequential_system.coordinate_system import GlobalSurfaceDefinition
    from wavefront_to_rays.global_element_raytracer import (
        GlobalElementRaytracer,
        GlobalSurfaceDefinition as RaytracerSurfaceDefinition,
        PlaneDef,
    )


class HybridElementPropagatorGlobal:
    """全局坐标系混合元件传播器
    
    在材质变化处执行波前-光线-波前重建流程，使用全局坐标系进行光线追迹。
    
    与 HybridElementPropagator 的主要区别：
    
    | 方面 | HybridElementPropagator | HybridElementPropagatorGlobal |
    |------|-------------------------|-------------------------------|
    | 坐标系 | 入射面局部坐标系 | 全局坐标系 |
    | 光线追迹器 | ElementRaytracer | GlobalElementRaytracer |
    | 入射面定义 | 隐式（原点） | 显式（点+法向量） |
    | 出射面定义 | 通过旋转矩阵 | 显式（点+法向量） |
    | 适用场景 | 简单系统 | 复杂折叠镜系统 |
    
    属性:
        wavelength_um: 波长 (μm)
        num_rays: 光线采样数量
    
    **Validates: Requirements 8.1, 8.4**
    """

    def __init__(
        self,
        wavelength_um: float,
        num_rays: int = 200,
    ) -> None:
        """初始化全局坐标系混合元件传播器
        
        参数:
            wavelength_um: 波长 (μm)
            num_rays: 光线采样数量，默认 200
        
        **Validates: Requirements 8.1**
        """
        self._wavelength_um = wavelength_um
        self._num_rays = num_rays
        self._state_converter = StateConverter(wavelength_um)
    
    @property
    def wavelength_um(self) -> float:
        """波长 (μm)"""
        return self._wavelength_um
    
    @property
    def num_rays(self) -> int:
        """光线采样数量"""
        return self._num_rays
    
    def propagate(
        self,
        state: PropagationState,
        surface: "GlobalSurfaceDefinition",
        entrance_axis: "OpticalAxisState",
        exit_axis: "OpticalAxisState",
        target_surface_index: int,
    ) -> PropagationState:
        """执行全局坐标系混合元件传播
        
        流程:
        1. 定义入射面（垂直于入射光轴，原点为主光线与表面交点）
        2. 从振幅/相位采样光线（入射面局部坐标系）
        3. 将光线转换到全局坐标系
        4. 使用 GlobalElementRaytracer 进行光线追迹
        5. 定义出射面（垂直于出射光轴，原点为主光线与表面交点）
        6. 将出射光线转换到出射面局部坐标系
        7. 计算残差 OPD 并重建波前
        8. 更新 Pilot Beam 参数
        
        参数:
            state: 入射面传播状态
            surface: 表面定义
            entrance_axis: 入射光轴状态
            exit_axis: 出射光轴状态
            target_surface_index: 目标表面索引
        
        返回:
            出射面传播状态
        
        **Validates: Requirements 8.2, 8.3**
        """
        from wavefront_to_rays.global_element_raytracer import (
            GlobalElementRaytracer,
            GlobalSurfaceDefinition as RaytracerSurfaceDefinition,
            PlaneDef,
        )
        from wavefront_to_rays.reconstructor import RayToWavefrontReconstructor
        from optiland.rays import RealRays
        
        # =====================================================================
        # 1. 定义入射面（垂直于入射光轴）
        # =====================================================================
        entrance_plane = self._define_entrance_plane(entrance_axis, surface)
        
        # =====================================================================
        # 2. 从振幅/相位采样光线（入射面局部坐标系）
        # =====================================================================
        local_rays = self._sample_rays_from_wavefront(
            state.amplitude,
            state.phase,
            state.grid_sampling,
        )
        
        # =====================================================================
        # 3. 将光线转换到全局坐标系
        # =====================================================================
        global_rays = self._local_to_global_rays(
            local_rays,
            entrance_plane,
            entrance_axis,
        )
        
        # =====================================================================
        # 4. 创建表面定义并使用 GlobalElementRaytracer 进行光线追迹
        # =====================================================================
        raytracer_surface = self._create_raytracer_surface(surface)
        
        raytracer = GlobalElementRaytracer(
            surfaces=[raytracer_surface],
            wavelength=self._wavelength_um,
            entrance_plane=entrance_plane,
        )
        
        # 追迹主光线以确定出射面
        raytracer.trace_chief_ray()
        
        # 追迹所有光线
        output_rays = raytracer.trace(global_rays)
        
        # =====================================================================
        # 5. 定义出射面（垂直于出射光轴）
        # =====================================================================
        exit_plane = self._define_exit_plane(exit_axis, raytracer)
        
        # =====================================================================
        # 6. 将出射光线转换到出射面局部坐标系
        # =====================================================================
        local_output_rays = self._global_to_local_rays(
            output_rays,
            exit_plane,
            exit_axis,
        )
        
        # =====================================================================
        # 7. 计算残差 OPD 并重建波前
        # =====================================================================
        
        # 更新 Pilot Beam 参数（在计算残差 OPD 之前）
        new_pilot_params = self._update_pilot_beam(
            state.pilot_beam_params, surface
        )
        
        # 计算绝对 OPD（相对于主光线）
        absolute_opd_waves = self._compute_absolute_opd(
            local_rays, local_output_rays
        )
        
        # 计算出射面 Pilot Beam 理论 OPD
        pilot_opd_waves = self._compute_pilot_opd(
            local_output_rays, new_pilot_params
        )
        
        # 残差 OPD = 绝对 OPD + Pilot Beam OPD
        residual_opd_waves = absolute_opd_waves + pilot_opd_waves
        
        # 去除低阶项
        residual_opd_waves = self._remove_low_order_terms(
            local_output_rays, residual_opd_waves
        )
        
        # 在光线位置处插值输入振幅
        input_amplitude_at_rays = self._interpolate_amplitude(
            state.amplitude, state.grid_sampling, local_rays
        )
        
        # 重建振幅/残差相位
        reconstructor = RayToWavefrontReconstructor(
            grid_size=state.grid_sampling.grid_size,
            sampling_mm=state.grid_sampling.sampling_mm,
            wavelength_um=self._wavelength_um,
        )
        
        x_in = np.asarray(local_rays.x)
        y_in = np.asarray(local_rays.y)
        x_out = np.asarray(local_output_rays.x)
        y_out = np.asarray(local_output_rays.y)
        valid_mask = np.ones(len(x_in), dtype=bool)
        
        exit_amplitude, residual_phase = reconstructor.reconstruct_amplitude_phase(
            ray_x_in=x_in,
            ray_y_in=y_in,
            ray_x_out=x_out,
            ray_y_out=y_out,
            opd_waves=residual_opd_waves,
            valid_mask=valid_mask,
            input_amplitude=input_amplitude_at_rays,
        )
        
        # =====================================================================
        # 8. 加回 Pilot Beam 相位，得到完整相位
        # =====================================================================
        pilot_phase_grid = new_pilot_params.compute_phase_grid(
            state.grid_sampling.grid_size,
            state.grid_sampling.physical_size_mm,
        )
        
        exit_phase = residual_phase + pilot_phase_grid
        
        # 转换为 PROPER 形式
        new_wfo = self._state_converter.amplitude_phase_to_proper(
            exit_amplitude,
            exit_phase,
            state.grid_sampling,
            pilot_beam_params=new_pilot_params,
        )
        
        return PropagationState(
            surface_index=target_surface_index,
            position='exit',
            amplitude=exit_amplitude,
            phase=exit_phase,
            pilot_beam_params=new_pilot_params,
            proper_wfo=new_wfo,
            optical_axis_state=exit_axis,
            grid_sampling=state.grid_sampling,
        )

    # =========================================================================
    # 入射面和出射面定义
    # =========================================================================
    
    def _define_entrance_plane(
        self,
        entrance_axis: "OpticalAxisState",
        surface: "GlobalSurfaceDefinition",
    ) -> "PlaneDef":
        """定义入射面（垂直于入射光轴）
        
        入射面定义：
        - 法向量：入射光轴方向（归一化）
        - 原点：主光线与表面的交点
        
        对于简化实现，使用入射光轴位置作为入射面原点。
        实际的交点将由 GlobalElementRaytracer.trace_chief_ray() 计算。
        
        参数:
            entrance_axis: 入射光轴状态
            surface: 表面定义
        
        返回:
            入射面定义 (PlaneDef)
        
        **Validates: Requirements 1.4, 1.5, 5.1**
        """
        from wavefront_to_rays.global_element_raytracer import PlaneDef
        
        # 入射面法向量 = 入射光轴方向（归一化）
        entrance_dir = entrance_axis.direction.to_array()
        entrance_dir = entrance_dir / np.linalg.norm(entrance_dir)
        
        # 入射面原点 = 入射光轴位置
        # 注意：这是一个近似，实际原点应该是主光线与表面的交点
        # GlobalElementRaytracer 会在 trace_chief_ray() 中计算精确位置
        entrance_pos = entrance_axis.position.to_array()
        
        return PlaneDef(
            position=tuple(entrance_pos),
            normal=tuple(entrance_dir),
        )
    
    def _define_exit_plane(
        self,
        exit_axis: "OpticalAxisState",
        raytracer: "GlobalElementRaytracer",
    ) -> "PlaneDef":
        """定义出射面（垂直于出射光轴）
        
        出射面定义：
        - 法向量：出射光轴方向（归一化）
        - 原点：主光线与表面的交点（由 raytracer 计算）
        
        参数:
            exit_axis: 出射光轴状态
            raytracer: 已追迹主光线的光线追迹器
        
        返回:
            出射面定义 (PlaneDef)
        
        **Validates: Requirements 2.1, 2.2**
        """
        # 使用 raytracer 计算的出射面
        # trace_chief_ray() 已经设置了 exit_plane
        if raytracer.exit_plane is not None:
            return raytracer.exit_plane
        
        # 如果 raytracer 没有设置出射面，使用出射光轴信息
        from wavefront_to_rays.global_element_raytracer import PlaneDef
        
        exit_dir = exit_axis.direction.to_array()
        exit_dir = exit_dir / np.linalg.norm(exit_dir)
        exit_pos = exit_axis.position.to_array()
        
        return PlaneDef(
            position=tuple(exit_pos),
            normal=tuple(exit_dir),
        )

    # =========================================================================
    # 坐标转换
    # =========================================================================
    
    def _local_to_global_rays(
        self,
        local_rays: "RealRays",
        entrance_plane: "PlaneDef",
        entrance_axis: "OpticalAxisState",
    ) -> "RealRays":
        """将入射面局部坐标系的光线转换到全局坐标系
        
        局部坐标系定义：
        - 原点：入射面原点
        - Z 轴：入射面法向量（入射光轴方向）
        - X, Y 轴：由 Z 轴确定的正交基
        
        转换公式：
        - P_global = P_local @ R.T + origin
        - D_global = D_local @ R.T
        
        其中 R 是从全局坐标系到局部坐标系的旋转矩阵。
        
        参数:
            local_rays: 局部坐标系中的光线
            entrance_plane: 入射面定义
            entrance_axis: 入射光轴状态
        
        返回:
            全局坐标系中的光线
        
        **Validates: Requirements 5.1-5.4**
        """
        from optiland.rays import RealRays
        
        # 获取局部坐标系的基向量（全局坐标系表示）
        z_axis = np.array(entrance_plane.normal, dtype=np.float64)
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        # 选择一个与 z_axis 不平行的向量来构建 x_axis
        if abs(z_axis[2]) < 0.9:
            ref = np.array([0.0, 0.0, 1.0])
        else:
            ref = np.array([1.0, 0.0, 0.0])
        
        x_axis = np.cross(ref, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # 旋转矩阵：局部 -> 全局
        # R 的列是局部坐标系的基向量在全局坐标系中的表示
        R = np.column_stack([x_axis, y_axis, z_axis])
        
        # 入射面原点（全局坐标系）
        origin = np.array(entrance_plane.position, dtype=np.float64)
        
        # 获取光线数据
        x_local = np.asarray(local_rays.x)
        y_local = np.asarray(local_rays.y)
        z_local = np.asarray(local_rays.z)
        L_local = np.asarray(local_rays.L)
        M_local = np.asarray(local_rays.M)
        N_local = np.asarray(local_rays.N)
        
        n_rays = len(x_local)
        
        # 转换位置：P_global = R @ P_local + origin
        P_local = np.stack([x_local, y_local, z_local], axis=0)  # (3, n_rays)
        P_global = R @ P_local + origin.reshape(3, 1)  # (3, n_rays)
        
        # 转换方向：D_global = R @ D_local
        D_local = np.stack([L_local, M_local, N_local], axis=0)  # (3, n_rays)
        D_global = R @ D_local  # (3, n_rays)
        
        # 创建全局坐标系光线
        global_rays = RealRays(
            x=P_global[0],
            y=P_global[1],
            z=P_global[2],
            L=D_global[0],
            M=D_global[1],
            N=D_global[2],
            intensity=np.asarray(local_rays.i).copy(),
            wavelength=np.asarray(local_rays.w).copy(),
        )
        global_rays.opd = np.asarray(local_rays.opd).copy()
        
        return global_rays
    
    def _global_to_local_rays(
        self,
        global_rays: "RealRays",
        exit_plane: "PlaneDef",
        exit_axis: "OpticalAxisState",
    ) -> "RealRays":
        """将全局坐标系的光线转换到出射面局部坐标系
        
        局部坐标系定义：
        - 原点：出射面原点
        - Z 轴：出射面法向量（出射光轴方向）
        - X, Y 轴：由 Z 轴确定的正交基
        
        转换公式：
        - P_local = R.T @ (P_global - origin)
        - D_local = R.T @ D_global
        
        其中 R 是从全局坐标系到局部坐标系的旋转矩阵。
        
        参数:
            global_rays: 全局坐标系中的光线
            exit_plane: 出射面定义
            exit_axis: 出射光轴状态
        
        返回:
            出射面局部坐标系中的光线
        
        **Validates: Requirements 6.1-6.4**
        """
        from optiland.rays import RealRays
        
        # 获取局部坐标系的基向量（全局坐标系表示）
        z_axis = np.array(exit_plane.normal, dtype=np.float64)
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        # 选择一个与 z_axis 不平行的向量来构建 x_axis
        if abs(z_axis[2]) < 0.9:
            ref = np.array([0.0, 0.0, 1.0])
        else:
            ref = np.array([1.0, 0.0, 0.0])
        
        x_axis = np.cross(ref, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # 旋转矩阵：局部 -> 全局
        R = np.column_stack([x_axis, y_axis, z_axis])
        
        # 出射面原点（全局坐标系）
        origin = np.array(exit_plane.position, dtype=np.float64)
        
        # 获取光线数据
        x_global = np.asarray(global_rays.x)
        y_global = np.asarray(global_rays.y)
        z_global = np.asarray(global_rays.z)
        L_global = np.asarray(global_rays.L)
        M_global = np.asarray(global_rays.M)
        N_global = np.asarray(global_rays.N)
        
        # 转换位置：P_local = R.T @ (P_global - origin)
        P_global = np.stack([x_global, y_global, z_global], axis=0)  # (3, n_rays)
        P_local = R.T @ (P_global - origin.reshape(3, 1))  # (3, n_rays)
        
        # 转换方向：D_local = R.T @ D_global
        D_global = np.stack([L_global, M_global, N_global], axis=0)  # (3, n_rays)
        D_local = R.T @ D_global  # (3, n_rays)
        
        # 创建局部坐标系光线
        local_rays = RealRays(
            x=P_local[0],
            y=P_local[1],
            z=P_local[2],
            L=D_local[0],
            M=D_local[1],
            N=D_local[2],
            intensity=np.asarray(global_rays.i).copy(),
            wavelength=np.asarray(global_rays.w).copy(),
        )
        local_rays.opd = np.asarray(global_rays.opd).copy()
        
        return local_rays

    # =========================================================================
    # 光线采样和重建辅助方法
    # =========================================================================
    
    def _sample_rays_from_wavefront(
        self,
        amplitude: NDArray[np.floating],
        phase: NDArray[np.floating],
        grid_sampling: GridSampling,
    ) -> "RealRays":
        """从振幅/相位采样光线（入射面局部坐标系）
        
        使用 WavefrontToRaysSampler 的方法：入射面上均匀网格光线透过
        由相位定义的纯相位元件，利用相位梯度生成实际方向的光线。
        
        参数:
            amplitude: 振幅网格（实数，非负）
            phase: 相位网格（实数，非折叠，弧度）
            grid_sampling: 网格采样信息
        
        返回:
            采样的光线（入射面局部坐标系）
        
        **Validates: Requirements 5.1**
        """
        from wavefront_to_rays import WavefrontToRaysSampler
        
        sampler = WavefrontToRaysSampler(
            amplitude=amplitude,
            phase=phase,
            physical_size=grid_sampling.physical_size_mm,
            wavelength=self._wavelength_um,
            num_rays=self._num_rays,
            distribution="hexapolar",
        )
        
        output_rays = sampler.get_output_rays()
        
        # 入射光线 OPD 初始化为 0（相对于主光线）
        output_rays.opd = np.zeros(len(output_rays.x))
        
        return output_rays
    
    def _create_raytracer_surface(
        self,
        surface: "GlobalSurfaceDefinition",
    ) -> "RaytracerSurfaceDefinition":
        """从 GlobalSurfaceDefinition 创建 GlobalElementRaytracer 使用的表面定义
        
        参数:
            surface: 全局表面定义（来自 sequential_system.coordinate_system）
        
        返回:
            光线追迹器使用的表面定义
        """
        from wavefront_to_rays.global_element_raytracer import (
            GlobalSurfaceDefinition as RaytracerSurfaceDefinition,
        )
        
        # 从 orientation 矩阵提取倾斜角度
        # orientation 的列向量是表面局部坐标系的 X, Y, Z 轴
        # Z 轴方向是表面法向量的反方向
        orientation = surface.orientation
        
        # 提取 Z 轴（表面法向量的反方向）
        z_axis = orientation[:, 2]
        
        # 计算倾斜角度（从 Z 轴方向推导）
        # 假设旋转顺序为 X -> Y -> Z
        # 这里使用简化的方法：直接从 Z 轴方向计算 tilt_x 和 tilt_y
        tilt_y = np.arcsin(-z_axis[0])  # 绕 Y 轴旋转
        cos_tilt_y = np.cos(tilt_y)
        if abs(cos_tilt_y) > 1e-10:
            tilt_x = np.arctan2(z_axis[1], z_axis[2])  # 绕 X 轴旋转
        else:
            tilt_x = 0.0
        tilt_z = 0.0  # 简化处理，忽略绕 Z 轴旋转
        
        return RaytracerSurfaceDefinition(
            surface_type='mirror' if surface.is_mirror else 'refract',
            radius=surface.radius,
            conic=surface.conic,
            material=surface.material,
            vertex_position=tuple(surface.vertex_position),
            surface_normal=tuple(surface.surface_normal),
            tilt_x=tilt_x,
            tilt_y=tilt_y,
            tilt_z=tilt_z,
        )
    
    def _compute_absolute_opd(
        self,
        input_rays: "RealRays",
        output_rays: "RealRays",
    ) -> NDArray[np.floating]:
        """计算绝对 OPD（相对于主光线）
        
        参数:
            input_rays: 输入光线
            output_rays: 输出光线
        
        返回:
            OPD 数组（波长数），相对于主光线
        
        **Validates: Requirements 7.1**
        """
        wavelength_mm = self._wavelength_um * 1e-3
        opd_mm = np.asarray(output_rays.opd)
        
        # 找到主光线（最接近原点的光线）
        x_out = np.asarray(output_rays.x)
        y_out = np.asarray(output_rays.y)
        r_sq = x_out**2 + y_out**2
        chief_idx = np.argmin(r_sq)
        
        # 计算相对于主光线的 OPD
        chief_opd_mm = opd_mm[chief_idx]
        relative_opd_mm = opd_mm - chief_opd_mm
        
        # 转换为波长数
        opd_waves = relative_opd_mm / wavelength_mm
        
        return opd_waves
    
    def _compute_pilot_opd(
        self,
        output_rays: "RealRays",
        pilot_params: PilotBeamParams,
    ) -> NDArray[np.floating]:
        """计算出射面 Pilot Beam 理论 OPD
        
        Pilot Beam OPD 公式：opd = r² / (2R)
        其中 R 是曲率半径（带符号）
        
        参数:
            output_rays: 出射光线
            pilot_params: Pilot Beam 参数
        
        返回:
            Pilot Beam OPD 数组（波长数）
        
        **Validates: Requirements 7.2**
        """
        x_out = np.asarray(output_rays.x)
        y_out = np.asarray(output_rays.y)
        r_sq = x_out**2 + y_out**2
        
        R = pilot_params.curvature_radius_mm
        wavelength_mm = self._wavelength_um * 1e-3
        
        if np.isinf(R):
            pilot_opd_mm = np.zeros_like(r_sq)
        else:
            pilot_opd_mm = r_sq / (2 * R)
        
        # 转换为波长数（相对于主光线，主光线处 OPD = 0）
        chief_idx = np.argmin(r_sq)
        pilot_opd_waves = (pilot_opd_mm - pilot_opd_mm[chief_idx]) / wavelength_mm
        
        return pilot_opd_waves
    
    def _remove_low_order_terms(
        self,
        output_rays: "RealRays",
        residual_opd_waves: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """去除残差 OPD 中的低阶项（倾斜、二次、三次）
        
        物理原理：
        - 入射面垂直于入射光轴，出射面垂直于出射光轴
        - OPD 相对于主光线计算，倾斜相位应该被自动补偿
        - 结果中不应包含整体倾斜，只包含真实像差
        
        参数:
            output_rays: 出射光线
            residual_opd_waves: 残差 OPD（波长数）
        
        返回:
            去除低阶项后的残差 OPD（波长数）
        
        **Validates: Requirements 7.3**
        """
        x_out = np.asarray(output_rays.x)
        y_out = np.asarray(output_rays.y)
        
        # 拟合项：1, x, y, x², y², xy, x³, y³, x²y, xy²
        A_fit = np.column_stack([
            np.ones_like(x_out),      # 常数项
            x_out,                     # x 倾斜
            y_out,                     # y 倾斜
            x_out**2,                  # x²
            y_out**2,                  # y²
            x_out * y_out,             # xy
            x_out**3,                  # x³
            y_out**3,                  # y³
            x_out**2 * y_out,          # x²y
            x_out * y_out**2,          # xy²
        ])
        
        fit_coeffs, _, _, _ = np.linalg.lstsq(A_fit, residual_opd_waves, rcond=None)
        
        fit_component = (
            fit_coeffs[0] +
            fit_coeffs[1] * x_out +
            fit_coeffs[2] * y_out +
            fit_coeffs[3] * x_out**2 +
            fit_coeffs[4] * y_out**2 +
            fit_coeffs[5] * x_out * y_out +
            fit_coeffs[6] * x_out**3 +
            fit_coeffs[7] * y_out**3 +
            fit_coeffs[8] * x_out**2 * y_out +
            fit_coeffs[9] * x_out * y_out**2
        )
        
        return residual_opd_waves - fit_component
    
    def _interpolate_amplitude(
        self,
        amplitude: NDArray[np.floating],
        grid_sampling: GridSampling,
        rays: "RealRays",
    ) -> NDArray[np.floating]:
        """在光线位置处插值输入振幅
        
        参数:
            amplitude: 振幅网格
            grid_sampling: 网格采样信息
            rays: 光线
        
        返回:
            光线位置处的振幅值
        """
        from scipy.interpolate import RegularGridInterpolator
        
        x_in = np.asarray(rays.x)
        y_in = np.asarray(rays.y)
        
        half_size = grid_sampling.physical_size_mm / 2
        n = grid_sampling.grid_size
        coords = np.linspace(-half_size, half_size, n)
        
        amp_interp = RegularGridInterpolator(
            (coords, coords),  # (y, x) 坐标
            amplitude,
            method='linear',
            bounds_error=False,
            fill_value=0.0,
        )
        
        ray_points = np.column_stack([y_in, x_in])
        return amp_interp(ray_points)

    # =========================================================================
    # Pilot Beam 参数更新
    # =========================================================================
    
    def _update_pilot_beam(
        self,
        pilot_params: PilotBeamParams,
        surface: "GlobalSurfaceDefinition",
    ) -> PilotBeamParams:
        """更新 Pilot Beam 参数
        
        根据表面类型应用相应的 ABCD 变换。
        
        物理原理：
        - 球面镜：等效于焦距 f = R/2 的薄透镜，使用 apply_mirror(R)
        - 离轴抛物面镜（OAP）：出射波前几乎是平面波，曲率半径接近无穷大
        - 折射面：使用折射面的 ABCD 矩阵
        
        对于 OAP 的特殊处理：
        - 对于离轴抛物面镜，出射波前的实际曲率半径接近无穷大
        - 光线追迹 OPD 主要是线性的（倾斜项），二次项系数几乎为 0
        - 这意味着出射波前几乎是平面波
        
        参数:
            pilot_params: 当前 Pilot Beam 参数
            surface: 表面定义
        
        返回:
            更新后的 Pilot Beam 参数
        
        **Validates: Requirements 9.1-9.4**
        """
        if surface.is_mirror:
            conic = surface.conic
            R = surface.radius
            
            if abs(conic + 1.0) < 1e-10 and not np.isinf(R):
                # 离轴抛物面镜（OAP）
                vertex = surface.vertex_position
                x = vertex[0]
                y = vertex[1]
                d = np.sqrt(x**2 + y**2)
                
                if d < 1e-10:
                    # 轴上情况：等效于普通球面镜
                    return pilot_params.apply_mirror(R)
                
                # 离轴抛物面：出射波前几乎是平面波
                return pilot_params.apply_mirror(np.inf)
            else:
                # 球面镜或平面镜
                return pilot_params.apply_mirror(R)
        else:
            # 折射面变换
            n1 = 1.0  # 假设入射介质为空气
            n2 = self._get_refractive_index(surface.material)
            return pilot_params.apply_refraction(surface.radius, n1, n2)
    
    def _get_refractive_index(self, material: str) -> float:
        """获取材料的折射率
        
        参数:
            material: 材料名称
        
        返回:
            折射率
        """
        material_lower = material.lower()
        
        if material_lower in ('air', ''):
            return 1.0
        elif material_lower == 'n-bk7':
            return 1.5168
        elif material_lower == 'fused_silica':
            return 1.4585
        else:
            return 1.5  # 默认玻璃折射率
