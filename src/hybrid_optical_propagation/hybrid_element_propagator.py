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

from typing import TYPE_CHECKING, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from .data_models import PilotBeamParams, GridSampling, PropagationState
from .state_converter import StateConverter
from .material_detection import classify_surface_interaction

if TYPE_CHECKING:
    from sequential_system.coordinate_tracking import OpticalAxisState
    from sequential_system.coordinate_system import GlobalSurfaceDefinition


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
        num_rays: int = 1000,
        method: str = "local_raytracing",
        debug: bool = False,
    ) -> None:
        """初始化混合元件传播器
        
        参数:
            wavelength_um: 波长 (μm)
            num_rays: 光线采样数量，默认 200
            method: 传播方法
                - 'local_raytracing': 局部光线追迹方法（默认）
                - 'pure_diffraction': 纯衍射方法
        """
        self._wavelength_um = wavelength_um
        self._num_rays = num_rays
        self._method = method
        self._debug = debug
        print(f"[HybridElementPropagator] Initialized with debug={self._debug}")
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
       

        # [Debug] Visualizing Incoming Wavefront vs Pilot Beam
        self._plot_debug_phase(state, target_surface_index)
        
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
        #绘制相位与振幅
        self._plot_debug_phase(state, target_surface_index)
        # 1. 从振幅/相位采样光线
        # 注意：相位是非折叠实数，不需要解包裹
        input_rays = self._sample_rays_from_wavefront(
            state.amplitude,
            state.phase,
            state.grid_sampling,
            entrance_axis,
            state.pilot_beam_params,  # 传入 Pilot Beam 参数
        )

        
        # Debug: 检查输入光线是否带有倾斜
        L_in = np.asarray(input_rays.L)
        M_in = np.asarray(input_rays.M)
        N_in = np.asarray(input_rays.N)
        opd_in = np.asarray(input_rays.opd)
        #为我换算为角度并用于打印debug信息
        L_in_angle = np.arctan(L_in)*180/np.pi
        M_in_angle = np.arctan(M_in)*180/np.pi
        #为我打印debug信息
        print(f"\n[DEBUG] Input Rays Analysis:")
        print(f"  L mean in degrees: {np.mean(L_in_angle):.6e}, std: {np.std(L_in_angle):.6e}")
        print(f"  M mean in degrees: {np.mean(M_in_angle):.6e}, std: {np.std(M_in_angle):.6e}")
        print(f"  N mean: {np.mean(N_in):.6e}, std: {np.std(N_in):.6e}")
        print(f"  OPD mean: {np.mean(opd_in):.6e}, std: {np.std(opd_in):.6e}")
        
        # 简单的 Zernike check for input OPD
        # ... (可以使用 reconstructor 里的逻辑，或者只是简单的平面拟合)
        if len(opd_in) > 10:
             A = np.column_stack([np.asarray(input_rays.x), np.asarray(input_rays.y), np.ones_like(input_rays.x)])
             coeffs, _, _, _ = np.linalg.lstsq(A, opd_in, rcond=None)
             print(f"  Input OPD Fit: {coeffs[0]:.6e}*x + {coeffs[1]:.6e}*y + C")
        

        
        # 2. 创建表面定义并进行光线追迹
        surface_def = self._create_surface_definition(surface, entrance_axis, exit_axis)
        
        raytracer = ElementRaytracer(
            surfaces=[surface_def],
            wavelength=self._wavelength_um,
            chief_ray_direction=tuple(entrance_axis.direction.to_array()),
            entrance_position=tuple(entrance_axis.position.to_array()),
            exit_chief_direction=tuple(exit_axis.direction.to_array()),
            debug=self._debug,
        )
        
        if self._debug:
            from utils.debug_viz import plot_rays_2d, plot_phase
            
            # Plot Input Phase
            plot_phase(
                state.phase, 
                title=f"Surface {target_surface_index} Input Phase"
            )
            
            # Plot Input Rays
            x_in = np.asarray(input_rays.x)
            y_in = np.asarray(input_rays.y)
            L_in = np.asarray(input_rays.L)
            M_in = np.asarray(input_rays.M)
            
            # Convert global to local for plotting cleanliness (optional, but requested by user to be clear)
            # Actually input_rays are in Global coords here?
            # ElementRaytracer converts them to local.
            # Let's plot what we have.
            plot_rays_2d(
                x_in, y_in, L_in, M_in, 
                title=f"Surface {target_surface_index} Input Rays (Global)"
            )
        #核心函数：光学追迹
        output_rays = raytracer.trace(input_rays)
        
        # DEBUG: Output Ray Analysis
        x_out = np.asarray(output_rays.x)
        y_out = np.asarray(output_rays.y)
        L_out = np.asarray(output_rays.L)
        M_out = np.asarray(output_rays.M)
        
        print(f"\n[DEBUG] Output Rays (Local Frame):")
        print(f"  Centroid: x={np.mean(x_out):.6e}, y={np.mean(y_out):.6e}")
        print(f"  Direction L: mean={np.mean(L_out):.6e}, std={np.std(L_out):.6e}, min={np.min(L_out):.6e}, max={np.max(L_out):.6e}")
        print(f"  Direction M: mean={np.mean(M_out):.6e}, std={np.std(M_out):.6e}, min={np.min(M_out):.6e}, max={np.max(M_out):.6e}")
        
        # 3. 计算绝对 OPD（相对于主光线）
        absolute_opd_waves = self._compute_opd(
            input_rays, output_rays, raytracer, surface
        )
        
        # OPD debug (after _compute_opd)
        print(f"  OPD: mean={np.mean(absolute_opd_waves):.6e}, std={np.std(absolute_opd_waves):.6e}, min={np.min(absolute_opd_waves):.6e}, max={np.max(absolute_opd_waves):.6e}")
        if len(absolute_opd_waves) > 10:
             A = np.column_stack([x_out, y_out, np.ones_like(x_out)])
             coeffs, _, _, _ = np.linalg.lstsq(A, absolute_opd_waves, rcond=None)
             print(f"  Output OPD Fit: {coeffs[0]:.6e}*x + {coeffs[1]:.6e}*y + C")
             print(f"  (Corresponds to Tilt X={coeffs[0]:.6e}, Tilt Y={coeffs[1]:.6e})")
        
        # 4. 更新 Pilot Beam 参数（在计算残差 OPD 之前）
        # 传递主光线交点位置（即出射面原点）
        new_pilot_params = self._update_pilot_beam(
            state.pilot_beam_params, 
            surface,
            exit_axis.position.to_array()
        )
        
        # DEBUG: Check Pilot Beam Phase match
        # Calculate theoretical Pilot Phase at output ray positions
        from .data_models import PilotBeamParams
        # Note: new_pilot_params defines curvature relative to the exit plane
        # φ_pilot = k * r^2 / (2 * R)
        wavelength_mm = self._wavelength_um * 1e-3
        k = 2 * np.pi / wavelength_mm
        r_sq = x_out**2 + y_out**2
        R_pilot_mm = new_pilot_params.curvature_radius_mm
        
        if np.isinf(R_pilot_mm):
            pilot_phase_theoretical = np.zeros_like(r_sq)
        else:
            pilot_phase_theoretical = k * r_sq / (2 * R_pilot_mm) # radians

        # 5. 计算出射面 Pilot Beam 理论 OPD
        
        x_out = np.asarray(output_rays.x)
        y_out = np.asarray(output_rays.y)
        r_sq_out = x_out**2 + y_out**2
        
        R_out = new_pilot_params.curvature_radius_mm
        wavelength_mm = self._wavelength_um * 1e-3
        
        if np.isinf(R_out): 
            pilot_opd_mm = np.zeros_like(r_sq_out)
        else:
            pilot_opd_mm = r_sq_out / (2 * R_out)
        
        chief_idx = np.argmin(r_sq_out)
        pilot_opd_waves = (pilot_opd_mm - pilot_opd_mm[chief_idx]) / wavelength_mm
        
        if self._debug:
            from utils.debug_viz import plot_rays_2d, plot_phase
            
            # Plot Output Rays
            plot_rays_2d(
                x_out, y_out, L_out, M_out,
                title=f"Surface {target_surface_index} Output Rays (Local)"
            )
            
            # Plot Absolute OPD
            plot_phase(
                absolute_opd_waves, 
                x=x_out, y=y_out,
                title=f"Surface {target_surface_index} Absolute OPD (Waves)"
            )
            
            # Plot Theoretical Pilot OPD
            plot_phase(
                pilot_opd_waves,
                x=x_out, y=y_out,
                title=f"Surface {target_surface_index} Pilot Beam Theoretical OPD"
            )
        
        # 6. 计算残差 OPD
        # 注意：是加法，不是减法！
        # 因为 absolute_opd_waves > 0，pilot_opd_waves < 0（当 R < 0）
        # 对于理想球面镜，两者大小相等符号相反，残差 ≈ 0
        # 6. 计算残差 OPD
        # Residual = Actual - Reference
        
        # 6. 计算残差 OPD
        # Residual = Actual - Reference
        residual_opd_waves = absolute_opd_waves - pilot_opd_waves
        
        # ⚠️ 关键修正：只去除 Piston
        r_sq_out = x_out**2 + y_out**2
        
        A_fit = np.column_stack([
            np.ones_like(x_out) 
        ])
        
        # 使用最小二乘法拟合
        fit_coeffs, _, _, _ = np.linalg.lstsq(A_fit, residual_opd_waves, rcond=None)
        
        # 构建拟合波前
        fit_component = (
            fit_coeffs[0]  
        )
        
        # 去除拟合分量
        residual_opd_waves = residual_opd_waves - fit_component
        
        # 7. 在光线位置处插值输入振幅
        # 先将全局光线坐标转换回局部坐标（相对于光轴中心）
        # 否则如果光轴不在原点（如 OAP2），插值会越界得到 0
        from scipy.interpolate import RegularGridInterpolator
        
        x_in_global = np.asarray(input_rays.x)
        y_in_global = np.asarray(input_rays.y)
        
        entrance_pos = entrance_axis.position.to_array()
        x_in_local = x_in_global - entrance_pos[0]
        y_in_local = y_in_global - entrance_pos[1]
        
        # 定义 x_in, y_in 供 Reconstructor 使用 (局部坐标)
        x_in = x_in_local
        y_in = y_in_local

        
        # 创建输入振幅的插值器
        n = state.grid_sampling.grid_size
        dx = state.grid_sampling.sampling_mm
        
        # 修正：使用 arange 生成精确对齐的坐标
        coords = (np.arange(n) - n // 2) * dx
        
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
        ray_points = np.column_stack([y_in_local, x_in_local])
        input_amplitude_at_rays = amp_interp(ray_points)
        
        # 8. 重建振幅/残差相位
        reconstructor = RayToWavefrontReconstructor(
            grid_size=state.grid_sampling.grid_size,
            sampling_mm=state.grid_sampling.sampling_mm,
            wavelength_um=self._wavelength_um,
        )
        
        # 创建有效光线掩模（所有光线都有效）
        valid_mask = np.ones(len(x_in), dtype=bool)
        
        # ⚠️ 关键修正：直接使用出射光线的局部坐标
        # raytracer.trace 返回的已经是相对于出射面中心的局部坐标
        # 不需要再进行任何全局坐标减法
        x_out_local = np.asarray(output_rays.x)
        y_out_local = np.asarray(output_rays.y)


        # 9. 加回 Pilot Beam 相位，得到完整相位
        # 计算网格上的 Pilot Beam 相位
        # 提前计算以便传递给 reconstructor 进行调试绘图
        pilot_phase_grid = new_pilot_params.compute_phase_grid(
            state.grid_sampling.grid_size,
            state.grid_sampling.physical_size_mm,
        )

        # 使用残差 OPD 进行重建
        # 重建得到的是残差相位（相对于 Pilot Beam 的偏差）
        # 传递输入振幅以保留振幅分布
        # 传递 debug_pilot_phase 以便在报警时绘制完整相位
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
        
        # 完整相位 = 残差相位 + Pilot Beam 相位
        exit_phase = residual_phase + pilot_phase_grid

        if self._debug:
            from utils.debug_viz import plot_phase, plot_comparison
            
            # Plot Residual OPD (Scatter)
            plot_phase(
                residual_opd_waves,
                x=x_out_local, y=y_out_local,
                title=f"Surface {target_surface_index} Residual OPD (Waves) on Rays"
            )
            
            # Plot Reconstructed Residual Phase (Grid)
            plot_phase(
                residual_phase,
                title=f"Surface {target_surface_index} Reconstructed Residual Phase (Grid)"
            )
            
            # Plot Pilot Phase on Grid
            plot_phase(
                pilot_phase_grid,
                title=f"Surface {target_surface_index} Pilot Phase on Grid"
            )
            
            # Plot Final Exit Phase
            plot_phase(
                exit_phase,
                title=f"Surface {target_surface_index} Final Exit Phase"
            )
        
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
        )

    
    def _propagate_pure_diffraction(
        self,
        state: PropagationState,
        surface: "GlobalSurfaceDefinition",
        entrance_axis: "OpticalAxisState",
        exit_axis: "OpticalAxisState",
        target_surface_index: int,
    ) -> PropagationState:
        """纯衍射方法（使用 tilted_asm 投影传输）
        
        流程:
        1. 使用 tilted_asm 从入射面传播到中间平面
        2. 在中间平面应用表面相位延迟（对于平面镜为 0）
        3. 使用 tilted_asm 从中间平面传播到出射面
        
        关键改进：
        使用入射光轴和出射光轴的"和"角平分线作为中间平面法向量，
        而不是表面法向量（切平面）。这样可以将总旋转角度平均分配，
        避免单步旋转角度超过 tilted_asm 的限制（~80°）。
        
        对于 45° 平面镜：
        - 入射光轴与出射光轴夹角 = 90°
        - 使用切平面：入射面→切平面 = 45°，切平面→出射面 = 135°（超限！）
        - 使用"和"角平分线：入射面→中间平面 = 45°，中间平面→出射面 = 45°（OK）
        
        注意：
        - 所有长度单位统一使用 mm
        - tilted_asm 对单位没有要求，只要求所有参数单位一致
        - 对于曲面镜，sag 相位需要投影到中间平面（当前仅支持平面镜）
        
        **Validates: Requirements 7.1-7.6**
        """
        # 动态添加 angular_spectrum_method 到路径
        import sys
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        asm_path = os.path.join(project_root, 'angular_spectrum_method')
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        from angular_spectrum_method.tilted_asm import tilted_asm, compute_carrier_frequency
        from scipy.spatial.transform import Rotation
        
        # 获取复振幅形式（tilted_asm 需要复数）
        u = state.get_complex_amplitude()
        
        # 计算采样间隔（单位：mm，与波长和物理尺寸一致）
        dx_mm = state.grid_sampling.sampling_mm
        dy_mm = state.grid_sampling.sampling_mm
        wavelength_mm = self._wavelength_um * 1e-3
        
        # 获取入射和出射光轴方向
        entrance_dir = entrance_axis.direction.to_array()
        exit_dir = exit_axis.direction.to_array()
        
        # 计算入射光轴和出射光轴的夹角
        total_rotation_angle = np.arccos(np.clip(np.dot(entrance_dir, exit_dir), -1, 1))
        total_rotation_deg = np.rad2deg(total_rotation_angle)
        
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
            else:
                n1 = 1.0
                n2 = self._get_refractive_index(surface.material)
                phase_delay = k * sag * (n2 - n1)
            
            u_exit = u * np.exp(1j * phase_delay)
            
            # 更新 Pilot Beam 参数
            new_pilot_params = self._update_pilot_beam(
                state.pilot_beam_params, surface
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
            else:
                n1 = 1.0
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
                state.pilot_beam_params, surface
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
        )
        
        # ⚠️ 关键：移除第一步引入的载波相位
        # 
        # tilted_asm 在输出时添加载波相位：exp(2πi(ν̂₀[1]*y + ν̂₀[0]*x))
        # 其中 ν̂₀ = T·[0, 0, 1/λ]
        # 
        # 正向和逆向旋转的载波频率不相等，所以必须手动移除
        #
        n = state.grid_sampling.grid_size
        
        # 计算第一步的载波频率
        nu_hat_0_1 = compute_carrier_frequency(T_to_mid, wavelength_mm)
        
        # 生成空间坐标（与 tilted_asm 内部一致）
        r_y = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / (n * dy_mm)))
        r_x = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / (n * (-dx_mm))))
        r_y_2d = r_y.reshape(-1, 1)
        r_x_2d = r_x.reshape(1, -1)
        
        # 移除第一步的载波相位
        carrier_phase_1 = np.exp(2j * np.pi * (nu_hat_0_1[1] * r_y_2d + nu_hat_0_1[0] * r_x_2d))
        u_mid = u_mid / carrier_phase_1
        
        # 2. 在中间平面应用表面相位延迟
        # 
        # 对于平面镜（sag = 0），不需要应用任何相位延迟
        # 对于曲面镜，需要将切平面上的 sag 投影到中间平面
        # 
        # 注意：中间平面与切平面的夹角可能很大（对于 45° 平面镜是 90°），
        # 所以 sag 的投影可能不准确。当前实现仅支持平面镜。
        
        n = state.grid_sampling.grid_size
        half_size = state.grid_sampling.physical_size_mm / 2
        coords = np.linspace(-half_size, half_size, n)
        X_mid, Y_mid = np.meshgrid(coords, coords)
        
        # 计算表面矢高（sag）
        # 对于平面镜，sag = 0
        sag = self._compute_surface_sag_in_tangent_plane(
            X_mid, Y_mid, surface, entrance_axis
        )
        
        # 计算相位延迟
        k = 2 * np.pi / wavelength_mm
        
        if surface.is_mirror:
            phase_delay = 2 * k * sag
        else:
            n1 = 1.0
            n2 = self._get_refractive_index(surface.material)
            phase_delay = k * sag * (n2 - n1)
        
        # 应用相位延迟
        u_after_surface = u_mid * np.exp(1j * phase_delay)
        
        # 反射时对相位取共轭（时间反演）
        if surface.is_mirror:
            u_after_surface = np.conj(u_after_surface)
        
        # 3. 使用 tilted_asm 从中间平面传播到出射面
        # 使用逆旋转矩阵
        T_to_exit = T_to_mid.T
        
        # 使用 tilted_asm 传播到出射面
        u_exit = tilted_asm(
            u_after_surface, wavelength_mm, dx_mm, dy_mm, T_to_exit, expand=True
        )
        
        # ⚠️ 关键：移除第二步引入的载波相位
        nu_hat_0_2 = compute_carrier_frequency(T_to_exit, wavelength_mm)
        carrier_phase_2 = np.exp(2j * np.pi * (nu_hat_0_2[1] * r_y_2d + nu_hat_0_2[0] * r_x_2d))
        u_exit = u_exit / carrier_phase_2
        
        # 4. 更新 Pilot Beam 参数
        new_pilot_params = self._update_pilot_beam(
            state.pilot_beam_params, surface
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
    
    def _compute_surface_sag_in_tangent_plane(
        self,
        X_tangent: NDArray,
        Y_tangent: NDArray,
        surface: "GlobalSurfaceDefinition",
        entrance_axis: "OpticalAxisState",
    ) -> NDArray:
        """在切平面局部坐标系中计算表面矢高
        
        切平面坐标系：
        - 原点：主光线与表面的交点
        - Z 轴：表面法向量方向
        - X, Y 轴：切平面内的正交方向
        
        对于平面镜：sag = 0（切平面与表面重合）
        对于球面/非球面：需要计算相对于切平面的矢高偏差
        
        参数:
            X_tangent: 切平面 X 坐标网格 (mm)
            Y_tangent: 切平面 Y 坐标网格 (mm)
            surface: 表面定义
            entrance_axis: 入射光轴状态
        
        返回:
            矢高网格 (mm)，相对于切平面
        """
        # 对于平面镜，矢高为 0
        if np.isinf(surface.radius):
            return np.zeros_like(X_tangent)
        
        # 对于曲面，需要计算相对于切平面的矢高
        # 
        # 物理解释：
        # 1. 切平面在主光线击中点处与表面相切
        # 2. 在切平面局部坐标系中，主光线击中点为原点
        # 3. 矢高是表面相对于切平面的高度差
        # 
        # 对于球面/非球面，在切平面局部坐标系中：
        # sag ≈ (x² + y²) / (2 * R_local)
        # 其中 R_local 是主光线击中点处的局部曲率半径
        
        # 获取表面参数
        R = surface.radius
        k = surface.conic
        
        # 计算主光线击中点处的局部曲率半径
        # 对于球面（k=0），局部曲率半径等于名义曲率半径
        # 对于非球面，局部曲率半径取决于击中点位置
        
        # 简化处理：使用名义曲率半径
        # 这对于小倾斜角和小光束尺寸是足够准确的
        R_local = R
        
        # 计算矢高（二次近似）
        r_sq = X_tangent**2 + Y_tangent**2
        
        if abs(k) < 1e-10:
            # 球面：使用精确公式
            c = 1.0 / R_local
            discriminant = 1 - c**2 * r_sq
            discriminant = np.maximum(discriminant, 0)
            sag = c * r_sq / (1 + np.sqrt(discriminant))
        else:
            # 非球面：使用标准公式
            c = 1.0 / R_local
            discriminant = 1 - (1 + k) * c**2 * r_sq
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
        pilot_beam_params: "PilotBeamParams" = None,
    ) -> "RealRays":
        """从振幅/相位采样光线
        
        改进版：使用网格对齐的笛卡尔采样 + Pilot Beam 解析叠加。
        这消除了中心像素的插值误差，并修正了坐标定义。
        """
        from optiland.rays import RealRays
        # Remove RegularGridInterpolator as we don't need it for grid-aligned sampling
        
        # 1. 准备采样用的相位
        if pilot_beam_params is not None:
            pilot_phase = pilot_beam_params.compute_phase_grid(
                grid_sampling.grid_size,
                grid_sampling.physical_size_mm
            )
            sampling_phase = phase - pilot_phase
        else:
            sampling_phase = phase
        
        #绘制采样相位
        self._plot_debug_phase(sampling_phase, 0,is_pltshow=True)

        # 2. 生成网格坐标 (Correct Method)
        n = grid_sampling.grid_size
        dx = grid_sampling.sampling_mm
        dy = grid_sampling.sampling_mm
        
        # 使用 arange 生成坐标，确保 0.0 在像素中心
        coords = (np.arange(n) - n // 2) * dx
        
        # 3. 笛卡尔网格采样
        # 目标是采样约 num_rays 个点
        # (N_sample)^2 ≈ num_rays => N_sample ≈ sqrt(num_rays)
        
        num_rays_target = self._num_rays
        stride = max(1, int(n / np.sqrt(num_rays_target)))
        
        # 确保中心点总是被包含
        center_idx = n // 2
        
        # 生成采样索引: 从中心向两边扩展
        half_n = n // 2
        offsets = np.arange(0, half_n, stride)
        
        # 合并正负偏移
        valid_offsets = np.unique(np.concatenate([offsets, -offsets]))
        sample_indices = center_idx + valid_offsets
        # 过滤越界索引
        sample_indices = np.sort(sample_indices[ (sample_indices >= 0) & (sample_indices < n) ])
        
        # 网格化
        ix_grid, iy_grid = np.meshgrid(sample_indices, sample_indices)
        
        # 展平索引
        ix_flat = ix_grid.flatten()
        iy_flat = iy_grid.flatten()
        
        # 提取数据 (直接读取，消除插值误差)
        sampled_phase = sampling_phase[iy_flat, ix_flat]
        sampled_amp = amplitude[iy_flat, ix_flat]
        
        # 提取坐标
        # 注意：numpy 数组 sampling_phase[y, x] 对应 coords[y], coords[x]
        x_rays = coords[ix_flat]
        y_rays = coords[iy_flat]
        
        # 4. 计算相位梯度得到方向 (L_res, M_res)
        # 先计算全分辨率的梯度，然后提取
        grad_y, grad_x = np.gradient(sampling_phase, dy, dx)
        
        wavelength_mm = self._wavelength_um * 1e-3
        k = 2 * np.pi / wavelength_mm
        
        L_grid_full = grad_x / k
        M_grid_full = grad_y / k
        
        L_rays = L_grid_full[iy_flat, ix_flat]
        M_rays = M_grid_full[iy_flat, ix_flat]
        
        # 5. 叠加 Pilot Beam 方向 (解析计算)
        if pilot_beam_params is not None:
            R = pilot_beam_params.curvature_radius_mm
            if np.isinf(R):
                L_pilot = 0.0
                M_pilot = 0.0
            else:
                L_pilot = x_rays / R
                M_pilot = y_rays / R
            
            L_rays += L_pilot
            M_rays += M_pilot
            
        # 6. 过滤无效光线
        threshold = 1e-3 * np.max(amplitude)  
        sin_sq = L_rays**2 + M_rays**2
        
        valid_mask = (sampled_amp > threshold) & (sin_sq < 1.0)
        
        # 6.1 只采样中心区域 95% 的光线，过滤边缘 5%
        half_size = grid_sampling.physical_size_mm / 2
        max_radius_sq = (0.95 * half_size) ** 2
        dist_sq = x_rays**2 + y_rays**2
        center_region_mask = dist_sq <= max_radius_sq
        valid_mask = valid_mask & center_region_mask
        
        # 强制保留中心光线 (x=0, y=0)
        center_ray_mask = (dist_sq < 1e-10)
        valid_mask = valid_mask | center_ray_mask

        if not np.any(valid_mask):
             raise ValueError("No valid rays remaining after manual sampling!")
             
        x_final = x_rays[valid_mask]
        y_final = y_rays[valid_mask]
        z_final = np.zeros_like(x_final)
        
        L_final = L_rays[valid_mask]
        M_final = M_rays[valid_mask]
        
        # Debug Output
        print(f"[DEBUG] Cartesian Sampling: {len(x_final)} rays (Target approx {num_rays_target})")
        # Find index of center ray in final arrays
        center_idx_final = np.argmin(x_final**2 + y_final**2)
        print(f"[DEBUG] Center Ray: x={x_final[center_idx_final]:.6e}, y={y_final[center_idx_final]:.6e}")
        print(f"[DEBUG] Center Ray Direction: L={L_final[center_idx_final]:.6e}, M={M_final[center_idx_final]:.6e}")
        
        # 转换到全局坐标
        axis_pos = entrance_axis.position.to_array()
        x_final += axis_pos[0]
        y_final += axis_pos[1]
        z_final += axis_pos[2]
        
        N_final = np.sqrt(1.0 - (L_final**2 + M_final**2))
        
        output_rays = RealRays(
            x=x_final,
            y=y_final,
            z=z_final,
            L=L_final,
            M=M_final,
            N=N_final,
            wavelength=self._wavelength_um,
            intensity=sampled_amp[valid_mask]**2
        )
        # 提取完整相位（用于计算初始 OPD）
        sampled_full_phase = phase[iy_flat, ix_flat]
        initial_phase = sampled_full_phase[valid_mask]
        initial_opd_mm = initial_phase * wavelength_mm / (2 * np.pi)
        output_rays.opd = initial_opd_mm
        output_rays.i = sampled_amp[valid_mask]**2
        
        return output_rays

    
    def _create_surface_definition(
        self,
        surface: "GlobalSurfaceDefinition",
        entrance_axis: "OpticalAxisState",
        exit_axis: "OpticalAxisState",
    ) -> "SurfaceDefinition":
        """从 GlobalSurfaceDefinition 创建 SurfaceDefinition
        
        修正说明：
        之前版本错误地使用光线击中点的法向量来计算 tilt，这对于 OAP 是错误的。
        现在的版本直接计算表面全局姿态（orientation）相对于入射光轴的旋转。
        这能正确传递机械倾角。对于对准良好的 OAP，此角度应接近 0。
        对于平面折叠镜，此角度应接近 45 度（或其他所需的折叠角）。
        """
        from wavefront_to_rays.element_raytracer import SurfaceDefinition, compute_rotation_matrix
        from scipy.spatial.transform import Rotation
        
        # 确定表面类型
        if surface.is_mirror:
            surface_type = 'mirror'
            material = 'mirror'
        else:
            surface_type = 'refract'
            material = surface.material
        
        # 1. 获取入射光坐标系的旋转矩阵 (R_beam)
        # Z轴沿入射光方向
        # compute_rotation_matrix构建的坐标系：Z轴对齐direction，X/Y轴正交
        entrance_dir = entrance_axis.direction.to_array()
        R_beam = compute_rotation_matrix(tuple(entrance_dir))
        
        # 2. 获取表面全局姿态矩阵 (R_surf)
        # 代表表面固有坐标系的朝向
        R_surf = surface.orientation
        
        # 3. 计算相对旋转矩阵 (R_rel)
        # R_rel 描述了：如果在光束坐标系中看，表面的旋转是多少
        # R_surf = R_beam @ R_rel  =>  R_rel = R_beam.T @ R_surf
        R_rel = R_beam.T @ R_surf
        
        # 4. 提取欧拉角 (rx, ry)
        # 使用 'xyz' 顺序提取。tile_x 对应 x 转角，tilt_y 对应 y 转角。
        # 忽略 z 转角 (clocking)，因为对于旋转对称的父曲面（如抛物面），
        # 绕轴旋转不改变曲面形状。
        # (注意：如果表面是非旋转对称的，如双曲面或像散面，忽略 tilt_z 可能有风险，
        # 但目前 SurfaceDefinition 不支持 tilt_z)
        euler_angles = Rotation.from_matrix(R_rel).as_euler('xyz', degrees=False)
        tilt_x = euler_angles[0]
        tilt_y = euler_angles[1]
        
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
            # 这个位置结合 tilt=0 (对于 OAP) 确保了正确的几何偏心关系
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
        interaction_point: Optional[NDArray] = None,
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
            interaction_point: 主光线与表面的交点位置 (mm)
        
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
                
                # 计算离轴距离 d
                # 按照用户要求：离轴量应当是光线交点向抛物面虚拟完整面的主轴方向做垂线的距离
                # 算法：
                # 1. 将交点转换到表面局部坐标系（原点在顶点，Z 轴为旋转轴）
                # 2. d = sqrt(x_local^2 + y_local^2)
                
                if interaction_point is not None:
                    # 获取表面参数
                    vertex = surface.vertex_position
                    orientation = surface.orientation
                    
                    # 转换到局部坐标系: P_local = R^T @ (P_global - V)
                    # interaction_point 是 (3,) 数组
                    delta = interaction_point - vertex
                    # orientation 是 (3,3) 矩阵，列向量为局部坐标轴
                    p_local = orientation.T @ delta
                    
                    # 提取局部 x, y
                    x_local = float(p_local[0])
                    y_local = float(p_local[1])
                    
                    d = np.sqrt(x_local**2 + y_local**2)
                else:
                    # 回退到旧逻辑（仅作为最后手段）
                    # 假设顶点位置本身就代表了离轴量（对于某些特定定义的系统）
                    vertex = surface.vertex_position
                    # 注意：如果表面有旋转，这里直接用 vertex xy 可能不准确
                    # 但如果没有 interaction_point，这是唯一能做的
                    d = np.sqrt(vertex[0]**2 + vertex[1]**2)
                
                # ⚠️ 关键修复：OAP 有效曲率半径修正
                # R_eff = R + d^2/R
                effective_R = R + d**2 / R
                return pilot_params.apply_mirror(effective_R)
            else:
                # 球面镜或平面镜：使用 apply_mirror
                return pilot_params.apply_mirror(R)
        else:
            # 折射面变换
            # 获取折射率
            n1 = 1.0  # 假设入射介质为空气
            n2 = self._get_refractive_index(surface.material)
            
            # 使用折射面 ABCD 变换
            return pilot_params.apply_refraction(surface.radius, n1, n2)
    
    def _compute_effective_radius(
        self,
        surface: "GlobalSurfaceDefinition"
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
            # 注意：_compute_effective_radius 仅用于内部计算，
            # 现在 _update_pilot_beam 已经改用 apply_mirror(inf) 处理 OAP，
            # 所以此方法对 OAP 的精确性要求降低，但为了完整性，我们也可以更新它。
            # 由于此方法接口未改变且调用者不多，目前保留原有逻辑，
            # 假设 interaction point 影响主要在外部处理。
            # 如果需要更精确，可以添加 interaction_point 参数。
            
            vertex = surface.vertex_position
            x = vertex[0]
            y = vertex[1]
            d = np.sqrt(x**2 + y**2) # 保持兼容性，暂不修改接口

            
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

    def _plot_debug_phase(
        self,
        state: PropagationState,
        target_surface_index: int,
        is_pltshow: bool = False
    ) -> None:
        """[Debug] 绘制入射波前相位分布与 Pilot Beam 理论相位分布（中央 80%）"""
        try:
            import matplotlib.pyplot as plt
            
            # 计算物理尺寸范围 (mm)
            half_size = state.grid_sampling.physical_size_mm / 2
            
            # 计算理论 Pilot Beam 相位
            pilot_phase = state.pilot_beam_params.compute_phase_grid(
                state.grid_sampling.grid_size,
                state.grid_sampling.physical_size_mm
            )
            
            # 截取中央 80% 区域
            N = state.grid_sampling.grid_size
            margin = int(N * 0.1)  # 10% margin on each side
            if margin > 0:
                s_ = slice(margin, N - margin)
                
                phase_crop = state.phase[s_, s_]
                pilot_crop = pilot_phase[s_, s_]
                
                # 更新 extent
                crop_half_size = half_size * 0.8
                extent = [-crop_half_size, crop_half_size, -crop_half_size, crop_half_size]
            else:
                s_ = slice(None)
                phase_crop = state.phase
                pilot_crop = pilot_phase
                extent = [-half_size, half_size, -half_size, half_size]
            
            plt.figure(figsize=(12, 5))
            
            # 1. 绘制实际入射相位 (Cropped)
            plt.subplot(1, 2, 1)
            im1 = plt.imshow(phase_crop, extent=extent, origin='lower', cmap='RdBu')
            plt.colorbar(im1, label='Phase (rad)')
            plt.title(f'Incoming Wavefront Phase (Central 80%, Surf {target_surface_index})')
            plt.xlabel('x (mm)'); plt.ylabel('y (mm)')
            
            # 2. 绘制 Pilot Beam 相位 (Cropped)
            plt.subplot(1, 2, 2)
            im2 = plt.imshow(pilot_crop, extent=extent, origin='lower', cmap='RdBu')
            plt.colorbar(im2, label='Phase (rad)')
            plt.title('Pilot Beam Phase (Theoretical, Central 80%)')
            plt.xlabel('x (mm)'); plt.ylabel('y (mm)')
            
            plt.tight_layout()
            if is_pltshow:
                plt.show()
            else:
                plt.savefig(f'wavefront_phase_{target_surface_index}.png')
                #加一个文件保存的提示
                print(f"Wavefront phase saved to wavefront_phase_{target_surface_index}.png")
            plt.close()
            
        except Exception as e:
            print(f"[Warning] Debug plotting failed: {e}")
