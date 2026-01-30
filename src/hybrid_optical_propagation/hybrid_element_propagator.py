"""
混合元件传播器模块

本模块实现元件处的混合传播仿真，将波前-光线-波前重建流程封装为统一接口。

核心功能：
1. 从入射面采样光线（使用非折叠相位）
2. 使用 ElementRaytracer 进行光线追迹
3. 计算 OPD 和雅可比矩阵振幅
4. 使用 RayToWavefrontReconstructor 重建复振幅

**Validates: Requirements 6.1-6.9, 7.1-7.7**
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Any
import numpy as np
from numpy.typing import NDArray
import os

from .data_models import PilotBeamParams, GridSampling, PropagationState
from .state_converter import StateConverter
from .material_detection import classify_surface_interaction

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from sequential_system.coordinate_tracking import OpticalAxisState
    from sequential_system.coordinate_system import GlobalSurfaceDefinition

    # For rotation decomposition
    from scipy.spatial.transform import Rotation


class HybridElementPropagator:
    """混合元件传播器
    
    在材质变化处执行波前-光线-波前重建流程。
    支持局部光线追迹方法和纯衍射方法。
    
    属性:
        wavelength_um: 波长 (μm)
        num_rays: 光线采样数量
        method: 传播方法 ('local_raytracing' 或 'pure_diffraction')
    
    **Validates: Requirements 6.1-6.9, 7.1-7.7**
    """
    
    def __init__(
        self,
        wavelength_um: float,
        num_rays: int = 200,
        method: str = "local_raytracing",
<<<<<<< Updated upstream
=======
        debug: bool = False,
        debug_dir: Optional[str] = None,
>>>>>>> Stashed changes
    ) -> None:
        """初始化混合元件传播器
        
        参数:
            wavelength_um: 波长 (μm)
            num_rays: 光线采样数量，默认 200
            method: 传播方法
                - 'local_raytracing': 局部光线追迹方法（默认）
                - 'pure_diffraction': 纯衍射方法
            debug: 是否开启调试模式
            debug_dir: 调试输出目录（可选）
        """
        self._wavelength_um = wavelength_um
        self._num_rays = num_rays
        self._method = method
<<<<<<< Updated upstream
=======
        self._debug = debug
        print(f"[HybridElementPropagator] Initialized with debug={self._debug}")
        
        if debug_dir:
            self._debug_dir = debug_dir
        else:
            self._debug_dir = os.path.join("debug", "hybrid")
            
        if self._debug:
             os.makedirs(self._debug_dir, exist_ok=True)
             print(f"[HybridElementPropagator] Debug output directory: {os.path.abspath(self._debug_dir)}")
             
>>>>>>> Stashed changes
        self._state_converter = StateConverter(wavelength_um)
    
    @property
    def wavelength_um(self) -> float:
        """波长 (μm)"""
        return self._wavelength_um
    
    @property
    def num_rays(self) -> int:
        """光线采样数量"""
        return self._num_rays
    
    @property
    def method(self) -> str:
        """传播方法"""
        return self._method
    
    def propagate(
        self,
        state: PropagationState,
        surface: "GlobalSurfaceDefinition",
        entrance_axis: "OpticalAxisState",
        exit_axis: "OpticalAxisState",
        target_surface_index: int,
    ) -> PropagationState:
        """执行混合元件传播
        
        参数:
            state: 入射面传播状态
            surface: 表面定义
            entrance_axis: 入射光轴状态
            exit_axis: 出射光轴状态
            target_surface_index: 目标表面索引
        
        返回:
            出射面传播状态
        
        **Validates: Requirements 6.1, 7.7**
        """
        if self._method == "local_raytracing":
            return self._propagate_local_raytracing(
                state, surface, entrance_axis, exit_axis, target_surface_index
            )
        else:
            return self._propagate_pure_diffraction(
                state, surface, entrance_axis, exit_axis, target_surface_index
            )
    
    def _propagate_local_raytracing(
        self,
        state: PropagationState,
        surface: "GlobalSurfaceDefinition",
        entrance_axis: "OpticalAxisState",
        exit_axis: "OpticalAxisState",
        target_surface_index: int,
    ) -> PropagationState:
        """局部光线追迹方法
        
        流程:
        1. 从振幅/相位采样光线（相位是非折叠实数）
        2. 使用 ElementRaytracer 进行光线追迹
        3. 计算出射 Pilot Beam 参数
        4. 计算残差 OPD = 绝对 OPD - Pilot Beam 理论 OPD
        5. 使用雅可比矩阵方法计算振幅变化
        6. 使用 RayToWavefrontReconstructor 重建振幅/残差相位
        7. 加回 Pilot Beam 相位，得到完整相位
        8. 转换为 PROPER 形式
        
        物理说明:
        - 入射面：垂直于入射光轴，原点为主光线与表面交点
        - 出射面：垂直于出射光轴，原点为主光线与表面交点
        - 光线追迹计算的是绝对 OPD（从入射面到出射面的总光程差）
        - 残差 OPD = 绝对 OPD - Pilot Beam 理论 OPD
        - 残差 OPD 应该很小（理想系统为 0），可以安全地进行网格重采样
        
        **Validates: Requirements 6.2-6.9**
        """
        from wavefront_to_rays.element_raytracer import (
            ElementRaytracer,
            SurfaceDefinition,
        )
        from wavefront_to_rays.reconstructor import RayToWavefrontReconstructor
        
        # 1. 从振幅/相位采样光线
        # 注意：相位是非折叠实数，不需要解包裹
        input_rays = self._sample_rays_from_wavefront(
            state.amplitude,
            state.phase,
            state.grid_sampling,
            entrance_axis,
        )
<<<<<<< Updated upstream
        
        # 2. 创建表面定义并进行光线追迹
=======

        debug_info_text = ""
        if self._debug:
            opd_in = np.asarray(input_rays.opd)
            debug_info_text = self._log_debug_info(
                surface, entrance_axis, exit_axis, target_surface_index,
                input_rays_opd_stats=(float(np.mean(opd_in)), float(np.std(opd_in)))
            )

        # 2. 创建单个表面定义并进行光线追迹 (Local Raytracing)
>>>>>>> Stashed changes
        surface_def = self._create_surface_definition(surface, entrance_axis, exit_axis)
        
        raytracer = ElementRaytracer(
            surfaces=[surface_def],
            wavelength=self._wavelength_um,
            chief_ray_direction=tuple(entrance_axis.direction.to_array()),
            entrance_position=tuple(entrance_axis.position.to_array()),
            exit_chief_direction=tuple(exit_axis.direction.to_array()),
        )
<<<<<<< Updated upstream
        
        output_rays = raytracer.trace(input_rays)
=======

        # Plot Input Wavefront (if debug)
        if self._debug:
             self._plot_debug_phase(
                state, target_surface_index, is_pltshow=False,
                figure_title=f"Surface {target_surface_index} - 入射面波前分析"
            )
        #核心函数：光学追迹，输入全局坐标的input_rays
        output_rays = raytracer.trace(input_rays)
        # 返回出射面局部坐标系 (Exit Surface Local Frame) 下的 output_rays
        # 坐标原点位于出射面中心 (主光线交点)，Z 轴沿出射法向
        # DEBUG: Output Ray Analysis
        x_out = np.asarray(output_rays.x)
        y_out = np.asarray(output_rays.y)
        L_out = np.asarray(output_rays.L)
        M_out = np.asarray(output_rays.M)
        
>>>>>>> Stashed changes
        
        # 3. 计算绝对 OPD（相对于主光线）
        absolute_opd_waves = self._compute_opd(
            input_rays, output_rays, raytracer, surface
        )
        
<<<<<<< Updated upstream
        # 4. 更新 Pilot Beam 参数（在计算残差 OPD 之前）
        new_pilot_params = self._update_pilot_beam(
            state.pilot_beam_params, surface
=======
        
        # 4. 更新 Pilot Beam 参数（在计算残差 OPD 之前）
        # 确定折射率
        n1 = state.current_refractive_index
        if surface.is_mirror:
            n2 = n1
        else:
            n2 = self._get_refractive_index(surface.material)
            
        # 传递主光线交点位置（即出射面原点）
        new_pilot_params = self._update_pilot_beam(
            state.pilot_beam_params, 
            surface,
            n1=n1,
            n2=n2,
            interaction_point=exit_axis.position.to_array(),
            entrance_axis=entrance_axis
>>>>>>> Stashed changes
        )
        
        # 5. 计算出射面 Pilot Beam 理论 OPD
        # 
        # 关键符号约定：
        # - 光线追迹 OPD（absolute_opd_waves）：几何光程增加量，正值表示边缘光线走更长路径
        # - Pilot Beam OPD（pilot_opd_waves）：相位延迟对应的等效光程，= r²/(2R)
        #   当 R < 0（会聚波）时，pilot_opd_waves < 0
        # 
        # 在 RayToWavefrontReconstructor 中，相位 = -2π × OPD
        # 所以：
        # - 光线追迹相位 = -2π × absolute_opd_waves（负值）
        # - Pilot Beam 相位 = 2π × pilot_opd_waves = k × r²/(2R)（负值，当 R < 0）
        # 
        # 对于理想球面镜：
        # - 光线追迹相位 ≈ Pilot Beam 相位
        # - 即 -2π × absolute_opd_waves ≈ 2π × pilot_opd_waves
        # - 即 absolute_opd_waves ≈ -pilot_opd_waves
        # 
        # 残差相位 = 光线追迹相位 - Pilot Beam 相位
        #          = -2π × absolute_opd_waves - 2π × pilot_opd_waves
        #          = -2π × (absolute_opd_waves + pilot_opd_waves)
        # 
        # 所以残差 OPD = absolute_opd_waves + pilot_opd_waves
        # 对于理想球面镜，残差 OPD ≈ 0
        
        # 确定出射介质折射率
        n_exit = n2
        
        x_out = np.asarray(output_rays.x)
        y_out = np.asarray(output_rays.y)
        r_sq_out = x_out**2 + y_out**2
        
        R_out = new_pilot_params.curvature_radius_mm
        wavelength_mm = self._wavelength_um * 1e-3
        
        if np.isinf(R_out):
            pilot_opd_mm = np.zeros_like(r_sq_out)
        else:
            # 修正: Phase = k * n * z = (2pi/lam) * n * (r^2/2R)
            # OPD_waves = n * (r^2/2R) / lam
            # 所以 pilot_opd_mm 需要乘 n_exit (OPD = n * geometric_sag)
            pilot_opd_mm = n_exit * r_sq_out / (2 * R_out)
        
        # 转换为波长数（相对于主光线，主光线处 OPD = 0）
        chief_idx = np.argmin(r_sq_out)
        pilot_opd_waves = (pilot_opd_mm - pilot_opd_mm[chief_idx]) / wavelength_mm
        
<<<<<<< Updated upstream
=======
        
>>>>>>> Stashed changes
        # 6. 计算残差 OPD
        # 注意：是加法，不是减法！
        # 因为 absolute_opd_waves > 0，pilot_opd_waves < 0（当 R < 0）
        # 对于理想球面镜，两者大小相等符号相反，残差 ≈ 0
        residual_opd_waves = absolute_opd_waves + pilot_opd_waves
        
        # ⚠️ 关键步骤：去除残差 OPD 中的低阶项（倾斜、二次、三次）
        # 
        # 物理原理：
        # - 入射面垂直于入射光轴，出射面垂直于出射光轴
        # - OPD 相对于主光线计算，倾斜相位应该被自动补偿
        # - 结果中不应包含整体倾斜，只包含真实像差
        # 
        # 对于离轴抛物面镜（OAP）：
        # - 光线追迹 OPD 包含大量的线性项（倾斜）和二次项（离焦）
        # - 还包含显著的三次项（y³ 和 x²y），这是 OAP 的固有几何特性
        # - 这些都不是真正的像差，应该被去除
        # 
        # 对于球面镜：
        # - 二次项应该与 Pilot Beam 的二次项抵消
        # - 三次项系数很小，去除不会影响结果
        # 
        # 使用最小二乘法拟合并去除到三次项
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
        residual_opd_waves = residual_opd_waves - fit_component
        
        # 7. 在光线位置处插值输入振幅
        # 这是为了在重建时保留输入振幅分布（高斯分布等）
        from scipy.interpolate import RegularGridInterpolator
        
        x_in = np.asarray(input_rays.x)
        y_in = np.asarray(input_rays.y)
        
        # 创建输入振幅的插值器
        half_size = state.grid_sampling.physical_size_mm / 2
        n = state.grid_sampling.grid_size
        coords = np.linspace(-half_size, half_size, n)
        
        # 注意：RegularGridInterpolator 期望 (y, x) 顺序的坐标
        amp_interp = RegularGridInterpolator(
            (coords, coords),  # (y, x) 坐标
            state.amplitude,
            method='linear',
            bounds_error=False,
            fill_value=0.0,
        )
        
        # 在光线位置处插值输入振幅
        # 注意：插值点格式为 (y, x)
        ray_points = np.column_stack([y_in, x_in])
        input_amplitude_at_rays = amp_interp(ray_points)
        
        # 8. 重建振幅/残差相位
        reconstructor = RayToWavefrontReconstructor(
            grid_size=state.grid_sampling.grid_size,
            sampling_mm=state.grid_sampling.sampling_mm,
            wavelength_um=self._wavelength_um,
        )
        
        # 创建有效光线掩模（所有光线都有效）
        valid_mask = np.ones(len(x_in), dtype=bool)
        
        # 使用残差 OPD 进行重建
        # 重建得到的是残差相位（相对于 Pilot Beam 的偏差）
        # 传递输入振幅以保留振幅分布
        exit_amplitude, residual_phase = reconstructor.reconstruct_amplitude_phase(
            ray_x_in=x_in,
            ray_y_in=y_in,
            ray_x_out=x_out,
            ray_y_out=y_out,
            opd_waves=residual_opd_waves,
            valid_mask=valid_mask,
            input_amplitude=input_amplitude_at_rays,
        )
        
        # 9. 加回 Pilot Beam 相位，得到完整相位
        # 计算网格上的 Pilot Beam 相位
        pilot_phase_grid = new_pilot_params.compute_phase_grid(
            state.grid_sampling.grid_size,
            state.grid_sampling.physical_size_mm,
        )
<<<<<<< Updated upstream
        
        # 完整相位 = 残差相位 + Pilot Beam 相位
        exit_phase = residual_phase + pilot_phase_grid
=======
        # 修正: Pilot Phase grid 也需要乘 n_exit
        if abs(n_exit - 1.0) > 1e-6:
             pilot_phase_grid *= n_exit

        # 使用残差 OPD 进行重建
        # 重建得到的是残差相位（相对于 Pilot Beam 的偏差）
        # 传递输入振幅以保留振幅分布
        # 传递 debug_pilot_phase 以便在报警时绘制完整相位
        print(f"[DEBUG TRACE] Calling reconstructor with {len(x_in)} rays. x_in range: [{x_in.min():.4e}, {x_in.max():.4e}]. x_out range: [{x_out_local.min():.4e}, {x_out_local.max():.4e}]")
        exit_amplitude, residual_phase = reconstructor.reconstruct_amplitude_phase(
            ray_x_in=x_in, # Local
            ray_y_in=y_in, # Local
            ray_x_out=x_out_local, # Local
            ray_y_out=y_out_local, # Local
            opd_waves=residual_opd_waves,
            valid_mask=valid_mask,
            input_amplitude=input_amplitude_at_rays,
            debug_pilot_phase=pilot_phase_grid,
        )
        print("[DEBUG TRACE] After reconstruct_amplitude_phase")
        
        # 完整相位 = 残差相位 + Pilot Beam 相位
        exit_phase = residual_phase + pilot_phase_grid

>>>>>>> Stashed changes
        
        # 10. 转换为 PROPER 形式
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
            current_refractive_index=n2,
        )
<<<<<<< Updated upstream
=======
        
        if self._debug:
             self._plot_debug_results(
                target_surface_index,
                input_rays, output_rays,
                absolute_opd_waves, pilot_opd_waves, residual_opd_waves,
                residual_phase, pilot_phase_grid, exit_phase,
                x_out_local, y_out_local,
                debug_info_text=debug_info_text
            )

        self._plot_debug_phase(
            new_state, target_surface_index, is_pltshow=False,
            figure_title=f"Surface {target_surface_index} - 出射面波前分析",
            filename_suffix="exit"
        )



        return new_state
>>>>>>> Stashed changes

    
    def _propagate_pure_diffraction(
        self,
        state: PropagationState,
        surface: "GlobalSurfaceDefinition",
        entrance_axis: "OpticalAxisState",
        exit_axis: "OpticalAxisState",
        target_surface_index: int,
    ) -> PropagationState:
        """纯衍射方法
        
        流程:
        1. 使用 tilted_asm 从入射面传播到元件切平面
        2. 在切平面计算表面矢高并应用相位延迟
        3. 使用 tilted_asm 从切平面传播到出射面
        
        **Validates: Requirements 7.1-7.6**
        """
        from angular_spectrum_method.tilted_asm import tilted_asm
        from scipy.spatial.transform import Rotation
        
        # 获取复振幅形式（tilted_asm 需要复数）
        u = state.get_complex_amplitude()
        
        # 计算采样间隔（转换为米，因为 tilted_asm 使用一致的单位）
        dx_mm = state.grid_sampling.sampling_mm
        dy_mm = state.grid_sampling.sampling_mm
        wavelength_mm = self._wavelength_um * 1e-3
        
        # 1. 计算入射面到切平面的旋转矩阵
        # 入射光轴方向
        entrance_dir = entrance_axis.direction.to_array()
        
        # 表面法向量（切平面的法向量）
        surface_normal = surface.surface_normal
        
<<<<<<< Updated upstream
        # 计算旋转矩阵：从入射面（垂直于入射光轴）到切平面（垂直于表面法向量）
        T_to_tangent = self._compute_rotation_matrix(entrance_dir, surface_normal)
        
        # 使用 tilted_asm 传播到切平面
        # 注意：tilted_asm 假设传播距离为 0（只是坐标变换）
        u_tangent = tilted_asm(
            u, wavelength_mm, dx_mm, dy_mm, T_to_tangent, expand=True
=======
        # 特殊情况：入射和出射方向相同（无光轴折叠）
        # 这种情况下不需要使用 tilted_asm，直接应用表面相位延迟即可
        if total_rotation_deg < 1e-6:
            # 直接应用表面相位延迟
            n = state.grid_sampling.grid_size
            half_size = state.grid_sampling.physical_size_mm / 2
            coords = np.linspace(-half_size, half_size, n)
            X, Y = np.meshgrid(coords, coords)
            
            sag = self._compute_surface_sag_in_tangent_plane(
                X, Y, surface, entrance_axis
            )
            
            k = 2 * np.pi / wavelength_mm
            if surface.is_mirror:
                phase_delay = 2 * k * sag
                n1 = state.current_refractive_index
                n2 = n1
            else:
                n1 = state.current_refractive_index
                n2 = self._get_refractive_index(surface.material)
                phase_delay = k * sag * (n2 - n1)
            
            u_exit = u * np.exp(1j * phase_delay)
            
            # 更新 Pilot Beam 参数
            new_pilot_params = self._update_pilot_beam(
                state.pilot_beam_params, 
                surface,
                n1=n1,
                n2=n2
            )
            
            # 分离振幅和相位
            exit_amplitude = np.abs(u_exit)
            exit_phase = np.angle(u_exit)
            
            # 使用 Pilot Beam 解包裹出射相位
            exit_phase = self._state_converter.unwrap_with_pilot_beam(
                exit_phase,
                new_pilot_params,
                state.grid_sampling,
            )
            
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
                current_refractive_index=n2,
            )
        
        # 特殊情况：180° 旋转（光线原路返回，如正入射反射镜）
        # 180° 旋转等价于坐标翻转，不需要使用 tilted_asm
        if total_rotation_deg > 179.9:
            # 应用表面相位延迟
            n = state.grid_sampling.grid_size
            half_size = state.grid_sampling.physical_size_mm / 2
            coords = np.linspace(-half_size, half_size, n)
            X, Y = np.meshgrid(coords, coords)
            
            sag = self._compute_surface_sag_in_tangent_plane(
                X, Y, surface, entrance_axis
            )
            
            k = 2 * np.pi / wavelength_mm
            if surface.is_mirror:
                phase_delay = 2 * k * sag
                n1 = state.current_refractive_index
                n2 = n1
            else:
                n1 = state.current_refractive_index
                n2 = self._get_refractive_index(surface.material)
                phase_delay = k * sag * (n2 - n1)
            
            u_after_surface = u * np.exp(1j * phase_delay)
            
            # 180° 旋转：翻转坐标（等价于 np.flip）
            # 对于反射镜，还需要取共轭
            if surface.is_mirror:
                u_exit = np.conj(np.flip(np.flip(u_after_surface, axis=0), axis=1))
            else:
                u_exit = np.flip(np.flip(u_after_surface, axis=0), axis=1)
            
            # 更新 Pilot Beam 参数
            new_pilot_params = self._update_pilot_beam(
                state.pilot_beam_params, 
                surface,
                n1=n1,
                n2=n2,
                entrance_axis=entrance_axis
            )
            
            # 分离振幅和相位
            exit_amplitude = np.abs(u_exit)
            exit_phase = np.angle(u_exit)
            
            # 使用 Pilot Beam 解包裹出射相位
            exit_phase = self._state_converter.unwrap_with_pilot_beam(
                exit_phase,
                new_pilot_params,
                state.grid_sampling,
            )
            
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
                current_refractive_index=n2,
            )
        
        # 计算"和"角平分线作为中间平面法向量
        # 这样可以将总旋转角度平均分配到两步
        sum_dir = entrance_dir + exit_dir
        sum_norm = np.linalg.norm(sum_dir)
        
        if sum_norm < 1e-10:
            # 入射和出射方向相反（180° 旋转），使用任意垂直方向
            perp = np.array([1, 0, 0]) if abs(entrance_dir[0]) < 0.9 else np.array([0, 1, 0])
            mid_plane_normal = np.cross(entrance_dir, perp)
            mid_plane_normal = mid_plane_normal / np.linalg.norm(mid_plane_normal)
        else:
            mid_plane_normal = sum_dir / sum_norm
        
        # 计算两步旋转的角度
        angle_to_mid = np.rad2deg(np.arccos(np.clip(np.dot(entrance_dir, mid_plane_normal), -1, 1)))
        angle_from_mid = np.rad2deg(np.arccos(np.clip(np.dot(mid_plane_normal, exit_dir), -1, 1)))
        
        # 1. 计算入射面到中间平面的旋转矩阵
        T_to_mid = self._compute_rotation_matrix(entrance_dir, mid_plane_normal)
        
        # 使用 tilted_asm 传播到中间平面
        u_mid = tilted_asm(
            u, wavelength_mm, dx_mm, dy_mm, T_to_mid, expand=True
>>>>>>> Stashed changes
        )
        
        # 2. 在切平面计算表面矢高并应用相位延迟
        # 计算网格坐标
        n = state.grid_sampling.grid_size
        half_size = state.grid_sampling.physical_size_mm / 2
        coords = np.linspace(-half_size, half_size, n)
        X, Y = np.meshgrid(coords, coords)
        
        # 计算表面矢高（sag）
        sag = self._compute_surface_sag(X, Y, surface)
        
        # 计算相位延迟
        # 对于反射：相位延迟 = 2 * k * sag
        # 对于折射：相位延迟 = k * sag * (n2 - n1)
        k = 2 * np.pi / wavelength_mm
        
        if surface.is_mirror:
            phase_delay = 2 * k * sag
            n1 = state.current_refractive_index
            n2 = n1
        else:
<<<<<<< Updated upstream
            # 折射面：需要折射率差
            n1 = 1.0  # 假设入射介质为空气
=======
            n1 = state.current_refractive_index
>>>>>>> Stashed changes
            n2 = self._get_refractive_index(surface.material)
            phase_delay = k * sag * (n2 - n1)
        
        # 应用相位延迟
        u_after_surface = u_tangent * np.exp(1j * phase_delay)
        
        # 3. 使用 tilted_asm 从切平面传播到出射面
        # 出射光轴方向
        exit_dir = exit_axis.direction.to_array()
        
        # 计算旋转矩阵：从切平面到出射面（垂直于出射光轴）
        T_to_exit = self._compute_rotation_matrix(surface_normal, exit_dir)
        
        # 使用 tilted_asm 传播到出射面
        u_exit = tilted_asm(
            u_after_surface, wavelength_mm, dx_mm, dy_mm, T_to_exit, expand=True
        )
        
        # 4. 更新 Pilot Beam 参数
        new_pilot_params = self._update_pilot_beam(
<<<<<<< Updated upstream
            state.pilot_beam_params, surface
=======
            state.pilot_beam_params, 
            surface,
            n1=n1,
            n2=n2,
            entrance_axis=entrance_axis
>>>>>>> Stashed changes
        )
        
        # 5. 分离振幅和相位
        exit_amplitude = np.abs(u_exit)
        exit_phase = np.angle(u_exit)
        
        # 使用 Pilot Beam 解包裹出射相位
        exit_phase = self._state_converter.unwrap_with_pilot_beam(
            exit_phase,
            new_pilot_params,
            state.grid_sampling,
        )
        
        # 6. 转换为 PROPER 形式
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
            current_refractive_index=n2,
        )
    
    def _compute_rotation_matrix(
        self,
        from_dir: NDArray,
        to_dir: NDArray,
    ) -> NDArray:
        """计算从一个方向到另一个方向的旋转矩阵
        
        参数:
            from_dir: 起始方向（单位向量）
            to_dir: 目标方向（单位向量）
        
        返回:
            3x3 旋转矩阵
        """
        from scipy.spatial.transform import Rotation
        
        # 归一化
        from_dir = from_dir / np.linalg.norm(from_dir)
        to_dir = to_dir / np.linalg.norm(to_dir)
        
        # 计算旋转轴和角度
        cross = np.cross(from_dir, to_dir)
        dot = np.dot(from_dir, to_dir)
        
        # 处理平行或反平行的情况
        if np.linalg.norm(cross) < 1e-10:
            if dot > 0:
                return np.eye(3)  # 同向，无旋转
            else:
                # 反向，绕任意垂直轴旋转 180°
                perp = np.array([1, 0, 0]) if abs(from_dir[0]) < 0.9 else np.array([0, 1, 0])
                axis = np.cross(from_dir, perp)
                axis = axis / np.linalg.norm(axis)
                return Rotation.from_rotvec(np.pi * axis).as_matrix()
        
        # 使用 Rodrigues 公式
        axis = cross / np.linalg.norm(cross)
        angle = np.arccos(np.clip(dot, -1, 1))
        
        return Rotation.from_rotvec(angle * axis).as_matrix()
    
    def _compute_surface_sag(
        self,
        X: NDArray,
        Y: NDArray,
        surface: "GlobalSurfaceDefinition",
    ) -> NDArray:
        """计算表面矢高（sag）
        
        对于球面/非球面：
        sag = c*r² / (1 + sqrt(1 - (1+k)*c²*r²))
        
        其中 c = 1/R 是曲率，k 是圆锥常数，r² = x² + y²
        
        参数:
            X: X 坐标网格 (mm)
            Y: Y 坐标网格 (mm)
            surface: 表面定义
        
        返回:
            矢高网格 (mm)
        """
        r_sq = X**2 + Y**2
        
        if np.isinf(surface.radius):
            return np.zeros_like(r_sq)
        
        c = 1.0 / surface.radius
        k = surface.conic
        
        # 标准非球面公式
        # sag = c*r² / (1 + sqrt(1 - (1+k)*c²*r²))
        discriminant = 1 - (1 + k) * c**2 * r_sq
        
        # 处理可能的负值（超出有效区域）
        discriminant = np.maximum(discriminant, 0)
        
        sag = c * r_sq / (1 + np.sqrt(discriminant))
        
        return sag
    
    def _get_refractive_index(self, material: str) -> float:
        """获取材料的折射率
        
        参数:
            material: 材料名称
        
        返回:
            折射率
        """
        # 简化实现：常见材料的折射率
        material_lower = material.lower()
        
        if material_lower in ('air', ''):
            return 1.0
        elif material_lower == 'n-bk7':
            return 1.5168
        elif material_lower == 'fused_silica':
            return 1.4585
        else:
            # 默认返回 1.5（典型玻璃）
            return 1.5
    
    def _sample_rays_from_wavefront(
        self,
        amplitude: NDArray[np.floating],
        phase: NDArray[np.floating],
        grid_sampling: GridSampling,
        entrance_axis: "OpticalAxisState",
    ) -> "RealRays":
        """从振幅/相位采样光线
        
        使用 WavefrontToRaysSampler 的方法：入射面上均匀网格光线透过
        由相位定义的纯相位元件，利用相位梯度生成实际方向的光线。
        
        流程：
        1. 使用输入的振幅和相位（相位是非折叠实数）
        2. 创建一个相位面（薄元件），其相位分布与输入相位匹配
        3. 产生平面波光束入射到相位面
        4. 通过 optiland 进行光线追迹
        5. 输出出射光束的光线数据（光线方向由相位梯度决定）
        
        参数:
            amplitude: 振幅网格（实数，非负）
            phase: 相位网格（实数，非折叠，弧度）
            grid_sampling: 网格采样信息
            entrance_axis: 入射光轴状态
        
        返回:
            采样的光线（方向由相位梯度决定）
        
        **Validates: Requirements 6.2, 6.3**
        """
        from wavefront_to_rays import WavefrontToRaysSampler
        
        # 使用新的振幅/相位分离接口
        sampler = WavefrontToRaysSampler(
            amplitude=amplitude,
            phase=phase,
            physical_size=grid_sampling.physical_size_mm,
            wavelength=self._wavelength_um,
            num_rays=self._num_rays,
            distribution="hexapolar",
        )
        
        # 获取出射光线（方向已由相位梯度决定）
        output_rays = sampler.get_output_rays()
        
        # ⚠️ 关键：入射光线 OPD 应该设置为 0
        # 
        # 根据 OPD 定义规范：
        # - 入射光线 OPD 初始化为 0（相对于主光线）
        # - 光线追迹会累加 OPD 增量
        # - 出射光线 OPD = 入射光线 OPD + 光线追迹 OPD 增量
        # - 残差 OPD = 出射光线 OPD + 出射面 Pilot Beam OPD
        #
        # 不使用 sampler.get_ray_opd()，因为那是入射波前的 OPD，
        # 而我们需要的是相对于主光线的 OPD（初始为 0）。
        output_rays.opd = np.zeros(len(output_rays.x))
        
        return output_rays
    
    def _create_surface_definition(
        self,
        surface: "GlobalSurfaceDefinition",
        entrance_axis: "OpticalAxisState",
        exit_axis: "OpticalAxisState",
    ) -> "SurfaceDefinition":
        """从 GlobalSurfaceDefinition 创建 SurfaceDefinition
        
        计算表面在入射面局部坐标系中的倾斜角度。
        
        对于反射镜，使用入射和出射方向来计算倾斜角度：
        - 表面法向量 = (入射方向 + 出射方向) / 2 的归一化
        - 倾斜角度 = 入射方向与出射方向夹角的一半
        
        这种方法对于离轴系统（如 OAP）是正确的，因为它使用的是
        主光线交点处的实际法向量，而不是表面顶点处的法向量。
        
        参数:
            surface: 全局表面定义
            entrance_axis: 入射光轴状态
            exit_axis: 出射光轴状态
        
        返回:
            ElementRaytracer 使用的表面定义
        
        **Validates: Requirements 6.4**
        """
        from wavefront_to_rays.element_raytracer import SurfaceDefinition
        
        # 确定表面类型
        if surface.is_mirror:
            surface_type = 'mirror'
            material = 'mirror'
        else:
            surface_type = 'refract'
            material = surface.material
        
        # 计算表面在入射面局部坐标系中的倾斜角度
        # 入射面局部坐标系：Z 轴为入射方向
        
        from wavefront_to_rays.element_raytracer import compute_rotation_matrix
        entrance_dir = entrance_axis.direction.to_array()
        exit_dir = exit_axis.direction.to_array()
        R_entrance = compute_rotation_matrix(tuple(entrance_dir))
        
        if surface.is_mirror:
            # 对于反射镜，使用入射和出射方向计算表面法向量
            # 反射定律：n = (d_in + d_out) / |d_in + d_out|
            # 其中 d_in 是入射方向，d_out 是出射方向
            # 注意：这里的法向量指向入射光来的方向
            n_sum = entrance_dir + exit_dir
            n_norm = np.linalg.norm(n_sum)
            if n_norm > 1e-10:
                surface_normal_global = n_sum / n_norm
            else:
                # 入射和出射方向相反（正入射），法向量沿入射方向
                surface_normal_global = entrance_dir.copy()
            
            # 确保法向量指向入射光来的方向（与入射方向相反）
            if np.dot(surface_normal_global, entrance_dir) > 0:
                surface_normal_global = -surface_normal_global
        else:
            # 对于透射元件，使用表面定义的法向量
            surface_normal_global = surface.surface_normal
        
        # 将表面法向量转换到入射面局部坐标系
        # v_local = R.T @ v_global
        surface_normal_local = R_entrance.T @ surface_normal_global
        
        # 从表面法向量计算倾斜角度
        # 在入射面局部坐标系中，未倾斜的表面法向量为 (0, 0, -1)
        # 倾斜后的法向量为 surface_normal_local
        # 
        # 旋转顺序：X → Y
        # 绕 X 轴旋转 rx 后：n = (0, sin(rx), -cos(rx))
        # 绕 Y 轴旋转 ry 后：n = (sin(ry)*cos(rx), sin(rx), -cos(ry)*cos(rx))
        #
        # 从法向量反推角度：
        # sin(rx) = n_y
        # sin(ry) = -n_x / cos(rx)  (当 cos(rx) != 0)
        
        nx, ny, nz = surface_normal_local
        
        # 计算 rx（绕 X 轴旋转）
        # sin(rx) = ny
        tilt_x = np.arcsin(np.clip(ny, -1, 1))
        
        # 计算 ry（绕 Y 轴旋转）
        cos_rx = np.cos(tilt_x)
        if abs(cos_rx) > 1e-10:
            # sin(ry) = -nx / cos(rx)
            sin_ry = -nx / cos_rx
            tilt_y = np.arcsin(np.clip(sin_ry, -1, 1))
        else:
            # cos(rx) ≈ 0，即 rx ≈ ±90°，此时 ry 不确定
            tilt_y = 0.0
        
        return SurfaceDefinition(
            surface_type=surface_type,
            radius=surface.radius,
            thickness=surface.thickness,
            material=material,
            conic=surface.conic,
            tilt_x=tilt_x,
            tilt_y=tilt_y,
            # ⚠️ 关键：传递表面顶点位置
            # 对于离轴系统（如 OAP），这是表面顶点的实际位置
            vertex_position=tuple(surface.vertex_position),
        )
    
    def _compute_opd(
        self,
        input_rays: "RealRays",
        output_rays: "RealRays",
        raytracer: "ElementRaytracer",
        surface: "GlobalSurfaceDefinition",
    ) -> NDArray[np.floating]:
        """计算 OPD（相对于主光线）
        
        参数:
            input_rays: 输入光线
            output_rays: 输出光线
            raytracer: 光线追迹器
            surface: 表面定义
        
        返回:
            OPD 数组（波长数），相对于主光线
        
        **Validates: Requirements 6.5**
        """
        # OPD 已经由 ElementRaytracer 计算并存储在 output_rays.opd 中
        # 转换为波长数
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
    
    def _compute_jacobian_amplitude(
        self,
        input_rays: "RealRays",
        output_rays: "RealRays",
        raytracer: "ElementRaytracer",
    ) -> NDArray[np.floating]:
        """计算雅可比矩阵振幅
        
        基于能量守恒原理，使用输入/输出光线位置计算局部面积放大率。
        
        物理原理：
        - 能量守恒：I_in × dA_in = I_out × dA_out
        - 振幅比：A_out / A_in = sqrt(I_out / I_in) = 1 / sqrt(|J|)
        - 其中 |J| 是雅可比行列式（局部面积放大率）
        
        算法：
        1. 使用 RBF 插值创建输入→输出坐标映射
        2. 使用数值微分计算雅可比矩阵
        3. 振幅 = 1 / sqrt(|J|)
        
        参数:
            input_rays: 输入光线
            output_rays: 输出光线
            raytracer: 光线追迹器
        
        返回:
            雅可比矩阵振幅数组
        
        **Validates: Requirements 6.6**
        """
        from scipy.interpolate import RBFInterpolator
        
        # 获取光线坐标
        x_in = np.asarray(input_rays.x)
        y_in = np.asarray(input_rays.y)
        x_out = np.asarray(output_rays.x)
        y_out = np.asarray(output_rays.y)
        
        n_rays = len(x_in)
        
        # 如果光线数量太少，无法计算雅可比矩阵
        if n_rays < 4:
            return np.ones(n_rays)
        
        # 创建从输入坐标到输出坐标的映射函数
        # 使用 RBF 插值创建平滑的映射函数
        points_in = np.column_stack([x_in, y_in])
        
        # 使用 thin_plate_spline 核函数，适合平滑的坐标映射
        try:
            interp_x = RBFInterpolator(points_in, x_out, kernel='thin_plate_spline')
            interp_y = RBFInterpolator(points_in, y_out, kernel='thin_plate_spline')
        except Exception:
            # 如果插值失败，返回单位振幅
            return np.ones(n_rays)
        
        # 计算雅可比矩阵的各分量（使用数值微分）
        # 微分步长选择：足够小以保证精度，但不能太小以避免数值误差
        eps = 1e-6  # 微分步长 (mm)
        
        # 在每个光线位置计算雅可比矩阵
        jacobian_det = np.zeros(n_rays)
        
        for i in range(n_rays):
            x0, y0 = x_in[i], y_in[i]
            
            # 计算 ∂x_out/∂x_in（x 方向输出对 x 方向输入的偏导数）
            dx_out_dx_in = (
                interp_x([[x0 + eps, y0]])[0] - interp_x([[x0 - eps, y0]])[0]
            ) / (2 * eps)
            
            # 计算 ∂x_out/∂y_in（x 方向输出对 y 方向输入的偏导数）
            dx_out_dy_in = (
                interp_x([[x0, y0 + eps]])[0] - interp_x([[x0, y0 - eps]])[0]
            ) / (2 * eps)
            
            # 计算 ∂y_out/∂x_in（y 方向输出对 x 方向输入的偏导数）
            dy_out_dx_in = (
                interp_y([[x0 + eps, y0]])[0] - interp_y([[x0 - eps, y0]])[0]
            ) / (2 * eps)
            
            # 计算 ∂y_out/∂y_in（y 方向输出对 y 方向输入的偏导数）
            dy_out_dy_in = (
                interp_y([[x0, y0 + eps]])[0] - interp_y([[x0, y0 - eps]])[0]
            ) / (2 * eps)
            
            # 雅可比行列式 = ∂x_out/∂x_in × ∂y_out/∂y_in - ∂x_out/∂y_in × ∂y_out/∂x_in
            # 取绝对值，因为我们关心的是面积变化的大小
            jacobian_det[i] = abs(
                dx_out_dx_in * dy_out_dy_in - dx_out_dy_in * dy_out_dx_in
            )
        
        # 避免除零
        # 使用最小值限制，确保数值稳定性
        min_jacobian = 1e-10
        jacobian_det = np.maximum(jacobian_det, min_jacobian)
        
        # 振幅 = 1 / sqrt(|J|)（能量守恒）
        # 当 |J| > 1 时，光束扩展，振幅减小
        # 当 |J| < 1 时，光束收缩，振幅增大
        amplitude = 1.0 / np.sqrt(jacobian_det)
        
        # 归一化振幅
        # 使归一化后的平均振幅为 1，保持相对变化
        mean_amplitude = np.mean(amplitude)
        if mean_amplitude > 0:
            amplitude = amplitude / mean_amplitude
        
        return amplitude
    
    def _update_pilot_beam(
        self,
        pilot_params: PilotBeamParams,
        surface: "GlobalSurfaceDefinition",
<<<<<<< Updated upstream
=======
        n1: float,
        n2: float,
        interaction_point: Optional[NDArray] = None,
        entrance_axis: "OpticalAxisState" = None,
>>>>>>> Stashed changes
    ) -> PilotBeamParams:
        """更新 Pilot Beam 参数
        
        根据表面类型应用相应的 ABCD 变换。
        
        物理原理：
        - 球面镜：等效于焦距 f = R/2 的薄透镜，使用 apply_mirror(R)
        - 离轴抛物面镜（OAP）：出射波前几乎是平面波，曲率半径接近无穷大
        - 折射面：使用折射面的 ABCD 矩阵
        
        对于 OAP 的特殊处理：
        ⚠️ 关键发现：对于离轴抛物面镜，出射波前的实际曲率半径接近无穷大！
        
        原因分析：
        - 抛物面将平行光聚焦到焦点
        - 但在出射面（垂直于出射光轴）上，波前的曲率半径不是简单的到焦点距离
        - 光线追迹 OPD 主要是线性的（倾斜项），二次项系数几乎为 0
        - 这意味着出射波前几乎是平面波
        
        解决方案：
        - 对于离轴抛物面，直接返回平面波参数（R = ∞）
        - 这与光线追迹的实际结果一致
        
        参数:
            pilot_params: 当前 Pilot Beam 参数
            surface: 表面定义
        
        返回:
            更新后的 Pilot Beam 参数
        
        **Validates: Requirements 6.7**
        """
        if surface.is_mirror:
            # 检查是否是抛物面（conic = -1）
            conic = surface.conic
            R = surface.radius
            
            if abs(conic + 1.0) < 1e-10 and not np.isinf(R):
                # 离轴抛物面镜（OAP）
                # 离轴距离从表面顶点位置获取
                vertex = surface.vertex_position
                x = vertex[0]
                y = vertex[1]
                d = np.sqrt(x**2 + y**2)
                
                if d < 1e-10:
                    # 轴上情况：等效于普通球面镜
                    return pilot_params.apply_mirror(R)
                
                # ⚠️ 关键修复：对于离轴抛物面，出射波前几乎是平面波
                # 
                # 物理解释：
                # 1. 抛物面将平行光聚焦到焦点
                # 2. 在出射面（垂直于出射光轴）上，光线追迹 OPD 主要是线性的
                # 3. 二次项系数几乎为 0，意味着出射波前几乎是平面波
                # 
                # 数学验证（来自 debug_correct_radius.py）：
                # - 光线追迹 OPD 的 y² 系数 ≈ 0.006 waves/mm²
                # - 对应的曲率半径 ≈ -134,415 mm（几乎是平面）
                # - 而 ABCD 计算的曲率半径是 -500 mm（错误！）
                #
                # 解决方案：直接返回平面波参数
                # 使用 apply_mirror(∞) 等效于不改变曲率半径
                return pilot_params.apply_mirror(np.inf)
            else:
                # 球面镜或平面镜：使用 apply_mirror
                return pilot_params.apply_mirror(R)
        else:
            # 折射面变换
            # n1, n2 已传入
            
            # 使用折射面 ABCD 变换
<<<<<<< Updated upstream
            return pilot_params.apply_refraction(surface.radius, n1, n2)
=======
            
            R_abcd = surface.radius * sign_factor
            
            return pilot_params.apply_refraction(R_abcd, n1, n2)

>>>>>>> Stashed changes
    
    def _compute_effective_radius(
        self,
        surface: "GlobalSurfaceDefinition",
    ) -> float:
        """计算反射镜的等效曲率半径
        
        对于球面镜（conic = 0）：等效曲率半径 = 名义曲率半径
        对于离轴抛物面镜（conic = -1）：
            - 出射波前是会聚球面波，曲率中心在焦点
            - 等效曲率半径 = 从出射面到焦点的距离（负值表示会聚）
        
        OAP 几何计算：
        - 焦距 f = R/2
        - 主光线交点 z = d²/(2R)，其中 d 是离轴距离
        - 焦点位置：(0, 0, f)
        - 主光线交点位置：(0, d, z_intersection)
        - 从交点到焦点的距离 = sqrt(d² + (f - z)²)
        - 出射波前曲率半径 = -distance（负值表示会聚）
        
        注意：这里返回的是用于 ABCD 变换的等效曲率半径，
        apply_mirror 方法会使用 -2/R 作为 C 参数。
        
        参数:
            surface: 表面定义
        
        返回:
            等效曲率半径 (mm)，负值表示会聚镜
        """
        R = surface.radius
        
        # 如果是平面镜，直接返回无穷大
        if np.isinf(R):
            return R
        
        # 检查是否是抛物面（conic = -1）
        conic = surface.conic
        
        if abs(conic + 1.0) < 1e-10:
            # 抛物面：计算等效曲率半径
            # 离轴距离从表面顶点位置获取
            vertex = surface.vertex_position
            x = vertex[0]
            y = vertex[1]
            d = np.sqrt(x**2 + y**2)
            
            if d < 1e-10:
                # 轴上情况：等效曲率半径 = 名义曲率半径
                return R
            
            # 计算从主光线交点到焦点的距离
            f = R / 2  # 焦距
            z_intersection = d**2 / (2 * R)  # 主光线交点 z 坐标
            distance_to_focus = np.sqrt(d**2 + (f - z_intersection)**2)
            
            # 等效曲率半径 = 2 × 距离（用于 ABCD 变换）
            # 因为 apply_mirror 使用 f = R/2，所以 R_eff = 2 × distance
            # 负值表示会聚（焦点在出射方向）
            R_eff = -2 * distance_to_focus
            
            return R_eff
        else:
            # 非抛物面（球面镜等）：使用名义曲率半径
            return R
<<<<<<< Updated upstream
=======

    def _plot_debug_phase(
        self,
        state: PropagationState,
        target_surface_index: int,
        is_pltshow: bool = False,
        figure_title: Optional[str] = None,
        filename_suffix: str = "entrance"
    ) -> None:
        """[Debug] 绘制详细的入射波前信息（振幅、相位、Pilot Beam、残差）
        
        参数:
            state: 传播状态
            target_surface_index: 目标表面索引
            is_pltshow: 是否显示图形（True）还是保存到文件（False）
            figure_title: 可选的大标题，用于描述这张信息图（例如 "OAP1 表面处、入射"）
            filename_suffix: 文件名后缀，用于区分入射/出射（默认 "entrance"，可设为 "exit"）
        """  
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # 准备数据
        grid_size = state.grid_sampling.grid_size
        physical_size = state.grid_sampling.physical_size_mm
        half_size = physical_size / 2
        
        # 1. 仿真波前
        sim_amp = state.amplitude
        sim_phase = state.phase
        
        # 2. Pilot Beam 理论值
        pilot_amp = state.pilot_beam_params.compute_amplitude_grid(grid_size, physical_size)
        pilot_phase = state.pilot_beam_params.compute_phase_grid(grid_size, physical_size)
        
        # 3. 相位残差
        phase_residual = sim_phase - pilot_phase
        
        # 计算截取范围 (80%) - 仍然截取以聚焦中心细节
        # 但用户可能也想看整体，这里保持 80% 或 90%
        margin = int(grid_size * 0.1)
        sl = slice(margin, grid_size - margin)
        
        # 更新 extent
        crop_half_size = half_size * 0.8
        extent = [-crop_half_size, crop_half_size, -crop_half_size, crop_half_size]
        
        # 绘图配置 (2行3列)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        # Info String
        pilot_info = (
            f"Pilot Beam Params (Surf {target_surface_index}): "
            f"w(z)={state.pilot_beam_params.spot_size_mm:.4f}mm, "
            f"R(z)={state.pilot_beam_params.curvature_radius_mm:.4e}mm, "
            f"z_waist={state.pilot_beam_params.waist_position_mm:.4f}mm"
        )
        
        # 构建完整标题：如果有 figure_title 则作为主标题，原信息作为副标题
        if figure_title:
            full_title = f"{figure_title}\n{pilot_info}"
        else:
            full_title = f"Debug Analysis - Surface {target_surface_index} (Entrance)\n{pilot_info}"
        fig.suptitle(full_title, fontsize=12)
        
        # Helper for colorbar
        def plot_im(ax, data, title, cmap='viridis', has_cbar=True):
            im = ax.imshow(data[sl, sl], extent=extent, origin='lower', cmap=cmap)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')
            if has_cbar:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
            return im

        # --- Row 1: Amplitude ---
        
        # 1.1 Simulation Amplitude
        # 计算近似光斑大小 (1/e^2)
        max_val = np.max(sim_amp)
        if max_val > 0:
            # 简单估算：等效宽度
            # w_eff = sqrt(2 * Sum(I) * dA / (pi * I_max))
            intensity = sim_amp**2
            total_power = np.sum(intensity)
            dx = state.grid_sampling.sampling_mm
            w_eff = np.sqrt(2 * total_power * dx**2 / (np.pi * np.max(intensity)))
            title_amp = f"Sim Amplitude\nCalc w ≈ {w_eff:.4f} mm\n(Max={max_val:.2e})"
        else:
            title_amp = "Sim Amplitude (Zero)"
        
        plot_im(axes[0, 0], sim_amp, title_amp, cmap='inferno')
        
        # 1.2 Pilot Beam Amplitude
        plot_im(axes[0, 1], pilot_amp, "Pilot Beam Amplitude\n(Theoretical Gaussian)", cmap='inferno')
        
        # 1.3 Cross Section Comparison
        ax_slice = axes[0, 2]
        y_mid = grid_size // 2
        x_axis = np.linspace(-half_size, half_size, grid_size)
        x_crop = x_axis[sl]
        
        # Normalize Sim Amp for comparison if max > 0
        sim_slice = sim_amp[y_mid, sl]
        if max_val > 0:
            sim_slice_norm = sim_slice / max_val
            ax_slice.plot(x_crop, sim_slice_norm, label='Sim (Norm)', color='blue')
        else:
            ax_slice.plot(x_crop, sim_slice, label='Sim', color='blue')
            
        ax_slice.plot(x_crop, pilot_amp[y_mid, sl], '--', label='Pilot (Ref)', color='orange')
        ax_slice.set_title("Amplitude Cross-section (y=0)", fontsize=10)
        ax_slice.set_xlabel('x (mm)')
        ax_slice.legend()
        ax_slice.grid(True, alpha=0.3)
        ax_slice.set_xlim(extent[0], extent[1])
        
        # --- Row 2: Phase ---
        
        # 2.1 Simulation Phase
        plot_im(axes[1, 0], sim_phase, "Simulation Phase\n(Unwrapped)", cmap='RdBu')
        
        # 2.2 Pilot Phase
        plot_im(axes[1, 1], pilot_phase, "Pilot Beam Phase\n(Analytic)", cmap='RdBu')
        
        # 2.3 Phase Residual
        # 残差通常很小，最好去除均值或活塞项方便观察
        res_crop = phase_residual[sl, sl]
        rms = np.std(res_crop)
        pv = np.max(res_crop) - np.min(res_crop)
        plot_im(axes[1, 2], phase_residual, f"Phase Residual\n(Sim - Pilot)\nRMS={rms:.4f} rad, PV={pv:.4f} rad", cmap='RdBu_r')
        
        # Annotations
        fig.text(0.5, 0.02, 
                    "Global Note: Coordinates (x, y) are in the LOCAL TANGENT PLANE of the entrance surface (perpendicular to chief ray).", 
                    ha='center', fontsize=11, bbox=dict(facecolor='#f0f0f0', alpha=0.9, pad=5))
        
        if is_pltshow:
            plt.show()
        else:
            filename = f'debug_surf_{target_surface_index}_{filename_suffix}.png'
            filepath = os.path.join(self._debug_dir, filename)
            plt.savefig(filepath, dpi=100)
            print(f"[Debug] Saved plot to {filepath}")

    def _log_debug_info(
        self,
        surface: "GlobalSurfaceDefinition",
        entrance_axis: "OpticalAxisState",
        exit_axis: "OpticalAxisState",
        target_surface_index: int,
        input_rays_opd_stats: Optional[Tuple[float, float]] = None
    ) -> str:
        """记录详细的调试信息 (User Requested)
        
        对于每个入射、出射面，记录：
        - 几何信息：顶点(Global)，曲率，法向
        - 旋转角：Local Rotation (Euler Angles by decomposing Global Orientation)
        - 标注：Comment
        - 光线信息：主光线交点(Global), 入射方向, 出射方向
        
        Returns:
            The constructed debug info string.
        """
        lines = []
        lines.append(f"{'='*30} Surface {target_surface_index} Debug Info {'='*30}")
        
        # 1. 表面信息
        lines.append(f"[Surface Info]")
        lines.append(f"  ID: {surface.index}")
        lines.append(f"  Type: {surface.surface_type}")
        lines.append(f"  Comment: {surface.comment}")
        lines.append(f"  Material: {surface.material}")
        
        # 2. 几何信息 (Global)
        lines.append(f"[Geometry (Global Frame)]")
        lines.append(f"  Vertex Position: [{surface.vertex_position[0]:.6f}, {surface.vertex_position[1]:.6f}, {surface.vertex_position[2]:.6f}] mm")
        
        if np.isinf(surface.radius):
             lines.append(f"  Curvature: Infinity (Flat)")
        else:
             lines.append(f"  Curvature Radius: {surface.radius:.6f} mm")
        
        # Orientation & Rotation
        try:
            r = Rotation.from_matrix(surface.orientation)
            euler_angles = r.as_euler('xyz', degrees=True)
            lines.append(f"  Global Orientation (Euler XYZ): [X={euler_angles[0]:.4f}°, Y={euler_angles[1]:.4f}°, Z={euler_angles[2]:.4f}°]")
        except Exception as e:
            lines.append(f"  Orientation: [Matrix]\n{surface.orientation}")
        
        lines.append(f"  Normal Vector (at vertex): [{surface.surface_normal[0]:.6f}, {surface.surface_normal[1]:.6f}, {surface.surface_normal[2]:.6f}]")
        lines.append(f"  Local Z-axis (Global):     [{surface.local_z_axis[0]:.6f}, {surface.local_z_axis[1]:.6f}, {surface.local_z_axis[2]:.6f}]")

        # 3. 主光线交互
        lines.append(f"[Chief Ray Interaction]")
        
        intersect = entrance_axis.position.to_array()
        lines.append(f"  Intersection Point (Global): [{intersect[0]:.6f}, {intersect[1]:.6f}, {intersect[2]:.6f}] mm")
        
        inc_dir = entrance_axis.direction.to_array()
        lines.append(f"  Incident Direction: [{inc_dir[0]:.6f}, {inc_dir[1]:.6f}, {inc_dir[2]:.6f}]")
        
        exit_dir = exit_axis.direction.to_array()
        lines.append(f"  Exit Direction:     [{exit_dir[0]:.6f}, {exit_dir[1]:.6f}, {exit_dir[2]:.6f}]")

        # Angle of Incidence
        cos_theta = np.dot(inc_dir, surface.surface_normal)
        aoi = np.degrees(np.arccos(np.clip(abs(cos_theta), -1, 1)))
        lines.append(f"  Angle of Incidence: {aoi:.4f}°")

        if input_rays_opd_stats:
            lines.append(f"[Input Rays Stats]")
            lines.append(f"  OPD Mean: {input_rays_opd_stats[0]:.6e}, Std: {input_rays_opd_stats[1]:.6e}")

        lines.append(f"{'='*80}")
        
        # Print and return
        full_text = "\n".join(lines)
        print("\n" + full_text + "\n")
        return full_text

    def _plot_debug_results(
        self,
        target_surface_index: int,
        input_rays: Any,
        output_rays: Any,
        absolute_opd_waves: np.ndarray,
        pilot_opd_waves: np.ndarray,
        residual_opd_waves: np.ndarray,
        residual_phase: np.ndarray,
        pilot_phase_grid: np.ndarray,
        exit_phase: np.ndarray,
        x_out_local: np.ndarray, 
        y_out_local: np.ndarray,
        debug_info_text: str = ""
    ) -> None:
        """绘制调试图表"""
        from utils.debug_viz import plot_rays_2d, plot_phase
        
        print(f"[DEBUG] Plotting detailed results for Surface {target_surface_index}...")

        # 0. Save Debug Info Text Image
        if debug_info_text:
            fig_text = plt.figure(figsize=(10, 8))
            # Use a monospace font for alignment
            fig_text.text(0.05, 0.95, debug_info_text, fontsize=10, family='monospace', va='top')
            plt.axis('off')
            plt.title(f"Surface {target_surface_index} Info", fontsize=14)
            filename_text = f'debug_surf_{target_surface_index}_00_info.png'
            filepath_text = os.path.join(self._debug_dir, filename_text)
            plt.savefig(filepath_text, dpi=100)
            plt.close(fig_text)
            print(f"[Debug] Saved info plot to {filepath_text}")

        # Plot Input Rays
        x_in = np.asarray(input_rays.x)
        y_in = np.asarray(input_rays.y)
        L_in = np.asarray(input_rays.L)
        M_in = np.asarray(input_rays.M)
        plot_rays_2d(
            x_in, y_in, L_in, M_in, 
            title=f"Surface {target_surface_index} Input Rays (Global)"
        )
        plt.savefig(os.path.join(self._debug_dir, f'debug_surf_{target_surface_index}_01_input_rays.png'))
        plt.close()
        
        # Plot Output Rays
        L_out = np.asarray(output_rays.L)
        M_out = np.asarray(output_rays.M)
        plot_rays_2d(
            x_out_local, y_out_local, L_out, M_out,
            title=f"Surface {target_surface_index} Output Rays (Local)"
        )
        plt.savefig(os.path.join(self._debug_dir, f'debug_surf_{target_surface_index}_02_output_rays.png'))
        plt.close()
        
        # Plot Absolute OPD
        plot_phase(
            absolute_opd_waves, 
            x=x_out_local, y=y_out_local,
            title=f"Surface {target_surface_index} Absolute OPD (Waves)"
        )
        plt.savefig(os.path.join(self._debug_dir, f'debug_surf_{target_surface_index}_03_abs_opd.png'))
        plt.close()
        
        # Plot Theoretical Pilot OPD
        plot_phase(
            pilot_opd_waves,
            x=x_out_local, y=y_out_local,
            title=f"Surface {target_surface_index} Pilot Beam Theoretical OPD"
        )
        plt.savefig(os.path.join(self._debug_dir, f'debug_surf_{target_surface_index}_04_pilot_opd.png'))
        plt.close()
        
        # Plot Residual OPD (Scatter)
        plot_phase(
            residual_opd_waves,
            x=x_out_local, y=y_out_local,
            title=f"Surface {target_surface_index} Residual OPD (Waves) on Rays"
        )
        plt.savefig(os.path.join(self._debug_dir, f'debug_surf_{target_surface_index}_05_resid_opd.png'))
        plt.close()
        
        # Plot Reconstructed Residual Phase (Grid)
        plot_phase(
            residual_phase,
            title=f"Surface {target_surface_index} Reconstructed Residual Phase (Grid)"
        )
        plt.savefig(os.path.join(self._debug_dir, f'debug_surf_{target_surface_index}_06_resid_phase.png'))
        plt.close()
        
        # Plot Pilot Phase on Grid
        plot_phase(
            pilot_phase_grid,
            title=f"Surface {target_surface_index} Pilot Phase on Grid"
        )
        plt.savefig(os.path.join(self._debug_dir, f'debug_surf_{target_surface_index}_07_pilot_phase.png'))
        plt.close()
        
        # Plot Final Exit Phase
        plot_phase(
            exit_phase,
            title=f"Surface {target_surface_index} Final Exit Phase"
        )
        plt.savefig(os.path.join(self._debug_dir, f'debug_surf_{target_surface_index}_08_exit_phase.png'))
        plt.close()

>>>>>>> Stashed changes
