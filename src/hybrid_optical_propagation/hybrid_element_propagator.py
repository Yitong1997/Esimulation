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
        num_rays: int = 200,
        method: str = "local_raytracing",
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
        
        # 2. 创建表面定义并进行光线追迹
        surface_def = self._create_surface_definition(surface, entrance_axis, exit_axis)
        
        raytracer = ElementRaytracer(
            surfaces=[surface_def],
            wavelength=self._wavelength_um,
            chief_ray_direction=tuple(entrance_axis.direction.to_array()),
            entrance_position=tuple(entrance_axis.position.to_array()),
            exit_chief_direction=tuple(exit_axis.direction.to_array()),
        )
        
        output_rays = raytracer.trace(input_rays)
        
        # 3. 计算绝对 OPD（相对于主光线）
        absolute_opd_waves = self._compute_opd(
            input_rays, output_rays, raytracer, surface
        )
        
        # 4. 更新 Pilot Beam 参数（在计算残差 OPD 之前）
        new_pilot_params = self._update_pilot_beam(
            state.pilot_beam_params, surface
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
        
        x_out = np.asarray(output_rays.x)
        y_out = np.asarray(output_rays.y)
        r_sq_out = x_out**2 + y_out**2
        
        R_out = new_pilot_params.curvature_radius_mm
        wavelength_mm = self._wavelength_um * 1e-3
        
        if np.isinf(R_out):
            pilot_opd_mm = np.zeros_like(r_sq_out)
        else:
            pilot_opd_mm = r_sq_out / (2 * R_out)
        
        # 转换为波长数（相对于主光线，主光线处 OPD = 0）
        chief_idx = np.argmin(r_sq_out)
        pilot_opd_waves = (pilot_opd_mm - pilot_opd_mm[chief_idx]) / wavelength_mm
        
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
        
        # 完整相位 = 残差相位 + Pilot Beam 相位
        exit_phase = residual_phase + pilot_phase_grid
        
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
        
        # 计算旋转矩阵：从入射面（垂直于入射光轴）到切平面（垂直于表面法向量）
        T_to_tangent = self._compute_rotation_matrix(entrance_dir, surface_normal)
        
        # 使用 tilted_asm 传播到切平面
        # 注意：tilted_asm 假设传播距离为 0（只是坐标变换）
        u_tangent = tilted_asm(
            u, wavelength_mm, dx_mm, dy_mm, T_to_tangent, expand=True
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
        else:
            # 折射面：需要折射率差
            n1 = 1.0  # 假设入射介质为空气
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
            # 获取折射率
            n1 = 1.0  # 假设入射介质为空气
            n2 = self._get_refractive_index(surface.material)
            
            # 使用折射面 ABCD 变换
            return pilot_params.apply_refraction(surface.radius, n1, n2)
    
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
