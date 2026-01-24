"""
混合高斯光束传输仿真器

本模块实现基于 PROPER 物理光学传播和 optiland 几何光线追迹的
混合高斯光束传输仿真。

设计理念：Zemax 序列模式
- 元件按顺序排列，使用 thickness 定义间距
- 光束沿光路自动传播
- 使用 propagate_distance() 方法沿光路传播

工作流程：
1. 在初始位置使用 PROPER 定义高斯光束波前
2. 使用 PROPER 物理光学传播到元件位置
3. 在元件处使用 wavefront_sampler 将波前采样为光线
4. 使用 element_raytracer 进行光线追迹
5. 从出射光线重建波前（相位和振幅）
6. 继续使用 PROPER 传播到下一个元件或观察面

作者：混合光学仿真项目
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from numpy.typing import NDArray

# 导入 PROPER
import proper

# 导入项目模块
from .gaussian_beam import GaussianBeam
from .optical_elements import OpticalElement, ParabolicMirror, SphericalMirror, ThinLens


@dataclass
class SimulationResult:
    """仿真结果
    
    属性:
        z: 位置（mm）
        path_length: 从起点的光程距离（mm）
        amplitude: 振幅分布
        phase: 相位分布（rad）
        wavefront: 复振幅分布
        sampling: 采样间隔（mm）
        beam_radius: 光束半径（mm），从振幅分布估计
        wavefront_rms: 波前 RMS（waves）
        wavefront_pv: 波前 PV（waves）
    """
    z: float
    path_length: float
    amplitude: NDArray
    phase: NDArray
    wavefront: NDArray
    sampling: float
    beam_radius: float
    wavefront_rms: float
    wavefront_pv: float
    
    @property
    def grid_size(self) -> int:
        return self.amplitude.shape[0]
    
    @property
    def physical_size(self) -> float:
        return self.sampling * self.grid_size


@dataclass
class PropagationStep:
    """传播步骤记录"""
    step_type: str
    z_start: float
    z_end: float
    path_start: float
    path_end: float
    element: Optional[OpticalElement] = None
    result: Optional[SimulationResult] = None
    notes: str = ""


class HybridGaussianBeamSimulator:
    """混合高斯光束传输仿真器（Zemax 序列模式）
    
    结合 PROPER 物理光学传播和 optiland 几何光线追迹，
    实现高斯光束通过光学系统的传输仿真。
    
    设计理念：
    - 元件按顺序排列，使用 thickness 定义到下一元件的间距
    - 光束沿光路自动传播
    - 使用 propagate_distance() 方法沿光路传播指定距离
    
    参数:
        beam: 高斯光束对象
        elements: 光学元件列表（按顺序排列）
        initial_distance: 从光束初始面到第一个元件的距离（mm）
        grid_size: 网格大小（像素），默认 512
        beam_ratio: 光束直径与网格的比例，默认 0.3
        num_rays: 光线采样数量，默认 200
        use_hybrid: 是否使用混合方法，默认 True
    
    示例:
        >>> beam = GaussianBeam(wavelength=0.633, w0=1.0, z0=50.0, z_init=0.0)
        >>> mirror = ParabolicMirror(thickness=100.0, semi_aperture=15.0,
        ...                          parent_focal_length=100.0)
        >>> sim = HybridGaussianBeamSimulator(beam, [mirror], initial_distance=200.0)
        >>> result = sim.propagate_distance(250.0)
    """
    
    def __init__(
        self,
        beam: GaussianBeam,
        elements: List[OpticalElement],
        initial_distance: float = 0.0,
        grid_size: int = 512,
        beam_ratio: float = 0.5,
        num_rays: int = 200,
        use_hybrid: bool = True,
    ) -> None:
        self.beam = beam
        self.elements = elements
        self.initial_distance = initial_distance
        self.grid_size = grid_size
        self.beam_ratio = beam_ratio
        self.num_rays = num_rays
        self.use_hybrid = use_hybrid
        
        # 计算每个元件的位置
        self._compute_element_positions()
        
        # 计算初始光束直径
        # beam_diameter = 2 × w0（PROPER 固定用法）
        # 必须使用束腰半径 w0，而不是某位置的光斑半径 w_init
        self.beam_diameter = 2 * beam.w0
        
        # 波长（转换为米）
        self.wavelength_m = beam.wavelength * 1e-6
        
        # 初始化状态
        self.wfo = None
        self.current_z = beam.z_init
        self.current_path = 0.0
        self.direction = 1
        
        # 传播历史
        self.history: List[PropagationStep] = []
        
        # 初始化波前
        self._initialize_wavefront()
    
    def _compute_element_positions(self) -> None:
        """计算每个元件的 z 坐标位置和光程距离"""
        current_z = self.beam.z_init
        current_path = 0.0
        direction = 1
        
        # 传播到第一个元件
        current_z += direction * self.initial_distance
        current_path += self.initial_distance
        
        for element in self.elements:
            element.z_position = current_z
            element.path_length = current_path
            
            if element.is_reflective:
                direction = -direction
            
            current_z += direction * element.thickness
            current_path += element.thickness
    
    def _initialize_wavefront(self) -> None:
        """初始化 PROPER 波前
        
        注意：PROPER 的 prop_begin 会创建一个默认的高斯光束，
        但其参数（w0, zR）是基于 beam_diameter 计算的，
        与我们的 GaussianBeam 对象不匹配。
        
        我们需要：
        1. 使用 prop_begin 创建波前网格
        2. 更新 PROPER 内部的高斯光束参数
        3. 手动设置正确的高斯振幅分布
        4. 手动设置正确的球面波前相位
        """
        import warnings
        
        beam_diameter_m = self.beam_diameter * 1e-3
        
        if beam_diameter_m <= 0:
            raise RuntimeError(f"光束直径必须为正值，实际为 {beam_diameter_m} m")
        if self.wavelength_m <= 0:
            raise RuntimeError(f"波长必须为正值，实际为 {self.wavelength_m} m")
        if self.grid_size <= 0:
            raise RuntimeError(f"网格大小必须为正整数，实际为 {self.grid_size}")
        
        try:
            # beam_diam_fraction = 0.5（PROPER 固定用法）
            self.wfo = proper.prop_begin(
                beam_diameter_m,
                self.wavelength_m,
                self.grid_size,
                0.5,
            )
        except Exception as e:
            raise RuntimeError(f"PROPER 波前初始化失败: {e}") from e
        
        if self.wfo is None:
            raise RuntimeError("PROPER 波前初始化失败: prop_begin 返回 None")
        
        # 更新 PROPER 内部的高斯光束参数以匹配我们的 GaussianBeam
        # 这对于 PROPER 的传播算法很重要
        w0_m = self.beam.w0 * 1e-3  # 束腰半径（米）
        zR_m = self.beam.zR * 1e-3  # 瑞利距离（米）
        z0_m = self.beam.z0 * 1e-3  # 束腰位置（米）
        z_init_m = self.beam.z_init * 1e-3  # 初始位置（米）
        
        self.wfo.w0 = w0_m
        self.wfo.z_Rayleigh = zR_m
        self.wfo.z_w0 = z0_m
        self.wfo.z = z_init_m
        
        # 计算初始位置的曲率半径
        R_init = self.beam.R(self.beam.z_init)
        if np.isinf(R_init):
            self.wfo.R_beam_inf = 1
            self.wfo.R_beam = 0.0
        else:
            self.wfo.R_beam_inf = 0
            self.wfo.R_beam = R_init * 1e-3
        
        proper.prop_define_entrance(self.wfo)
        
        # 重要：prop_begin 会创建一个默认的均匀振幅波前
        # 我们需要用正确的高斯分布替换它
        self._apply_gaussian_amplitude()
        self._apply_initial_phase()
        
        self.current_z = self.beam.z_init
        self.current_path = 0.0
        self.direction = 1
    
    def _apply_gaussian_amplitude(self) -> None:
        """应用高斯振幅分布
        
        高斯光束振幅公式：A(r, z) = (w0/w(z)) * exp(-r²/w(z)²)
        
        注意：PROPER 的初始波前是均匀的（全1），
        我们需要用正确的高斯分布替换它。
        
        重要：PROPER 使用 FFT 居中的数组布局，需要使用 prop_shift_center
        将以数组中心为原点的数据转换为 FFT 布局。
        """
        sampling_m = proper.prop_get_sampling(self.wfo)
        n = self.grid_size
        
        # 创建坐标网格（以米为单位）
        # 坐标以数组中心 (n/2, n/2) 为原点
        coords = (np.arange(n) - n // 2) * sampling_m
        X, Y = np.meshgrid(coords, coords)
        R_sq = X**2 + Y**2
        
        # 计算 z_init 位置的光束半径（转换为米）
        w_z = self.beam.w(self.beam.z_init) * 1e-3
        
        # 高斯振幅分布
        # A(r) = exp(-r²/w²)
        # 注意：这里不包含 w0/w(z) 因子，因为我们只关心相对振幅分布
        amplitude = np.exp(-R_sq / w_z**2)
        
        # 使用 prop_shift_center 将数组从"以中心为原点"转换为 FFT 布局
        # 这是 PROPER 内部使用的数组布局
        amplitude_shifted = proper.prop_shift_center(amplitude)
        
        # 设置波前数组（替换而不是乘以）
        # 保持相位为零（后面会添加正确的相位）
        self.wfo.wfarr = amplitude_shifted.astype(np.complex128)
        
        # 重新归一化
        total_intensity = np.sum(np.abs(self.wfo.wfarr)**2)
        if total_intensity > 0:
            self.wfo.wfarr /= np.sqrt(total_intensity)
    
    def _apply_initial_phase(self) -> None:
        """应用初始相位
        
        高斯光束相位公式：φ(r, z) = -k * r² / (2R(z)) + ψ(z)
        其中：
        - k = 2π/λ 是波数
        - R(z) 是波前曲率半径
        - ψ(z) 是 Gouy 相位（这里忽略，因为它是全局相位）
        
        当 z = z0（束腰位置）时，R(z) = ∞，相位为平面
        
        重要：相位数组也需要使用 prop_shift_center 转换为 FFT 布局
        """
        sampling_m = proper.prop_get_sampling(self.wfo)
        n = self.grid_size
        
        # 创建坐标网格（以米为单位）
        # 坐标以数组中心 (n/2, n/2) 为原点
        coords = (np.arange(n) - n // 2) * sampling_m
        X, Y = np.meshgrid(coords, coords)
        R_sq = X**2 + Y**2
        
        # 计算 z_init 位置的波前曲率半径
        R_z = self.beam.R(self.beam.z_init)
        
        if not np.isinf(R_z):
            # 转换为米
            R_z_m = R_z * 1e-3
            # 波数
            k = 2 * np.pi / self.wavelength_m
            # 球面波前相位
            curvature_phase = -k * R_sq / (2 * R_z_m)
            # 使用 prop_shift_center 转换为 FFT 布局
            curvature_phase_shifted = proper.prop_shift_center(curvature_phase)
            # 应用相位
            self.wfo.wfarr = self.wfo.wfarr * np.exp(1j * curvature_phase_shifted)
        
        # 应用用户定义的波前误差
        if self.beam.wavefront_error is not None:
            X_mm = X * 1e3
            Y_mm = Y * 1e3
            error_phase = self.beam.wavefront_error(X_mm, Y_mm)
            # 使用 prop_shift_center 转换为 FFT 布局
            error_phase_shifted = proper.prop_shift_center(error_phase)
            self.wfo.wfarr = self.wfo.wfarr * np.exp(1j * error_phase_shifted)
    
    def _propagate_proper(self, distance_mm: float) -> None:
        """使用 PROPER 进行物理光学传播"""
        if abs(distance_mm) < 1e-10:
            return
        
        distance_m = distance_mm * 1e-3
        proper.prop_propagate(self.wfo, distance_m)
        self.current_z += self.direction * distance_mm
        self.current_path += distance_mm
    
    def _apply_element_hybrid(self, element: OpticalElement) -> None:
        """使用混合方法处理光学元件"""
        src_path = os.path.join(os.path.dirname(__file__), '..')
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from wavefront_to_rays import WavefrontToRaysSampler
        from wavefront_to_rays.element_raytracer import ElementRaytracer
        
        wavefront = proper.prop_get_wavefront(self.wfo)
        sampling_m = proper.prop_get_sampling(self.wfo)
        sampling_mm = sampling_m * 1e3
        physical_size_mm = sampling_mm * self.grid_size
        
        sampler = WavefrontToRaysSampler(
            wavefront_amplitude=wavefront,
            physical_size=physical_size_mm,
            wavelength=self.beam.wavelength,
            num_rays=self.num_rays,
            distribution='uniform',
        )
        
        input_rays = sampler.get_output_rays()
        
        surface_def = element.get_surface_definition()
        if surface_def is None:
            self._apply_thin_lens_proper(element)
            return
        
        raytracer = ElementRaytracer(
            surfaces=[surface_def],
            wavelength=self.beam.wavelength,
            chief_ray_direction=element.get_chief_ray_direction(),
            entrance_position=element.get_entrance_position(),
        )
        
        output_rays = raytracer.trace(input_rays)
        
        x_out = np.asarray(output_rays.x)
        y_out = np.asarray(output_rays.y)
        opd_waves = raytracer.get_relative_opd_waves()
        valid_mask = raytracer.get_valid_ray_mask()
        
        reconstructed_phase = self._reconstruct_phase_from_rays(
            x_out, y_out, opd_waves, valid_mask,
            physical_size_mm, self.grid_size
        )
        
        if element.is_reflective:
            phase_change = 2 * np.pi * reconstructed_phase
            proper.prop_add_phase(self.wfo, phase_change)
            self.direction = -self.direction
            
            if element.tilt_x != 0 or element.tilt_y != 0:
                import warnings
                warnings.warn(
                    f"元件 '{element.name or element.element_type}' 存在倾斜，"
                    f"当前版本对倾斜反射镜的支持有限。",
                    RuntimeWarning
                )
        else:
            phase_change = 2 * np.pi * reconstructed_phase
            proper.prop_add_phase(self.wfo, phase_change)
    
    def _apply_thin_lens_proper(self, element: ThinLens) -> None:
        """使用 PROPER 处理薄透镜"""
        focal_length_m = element.focal_length * 1e-3
        proper.prop_lens(self.wfo, focal_length_m)
    
    def _apply_element_proper_only(self, element: OpticalElement) -> None:
        """仅使用 PROPER 处理光学元件"""
        if isinstance(element, ThinLens):
            self._apply_thin_lens_proper(element)
        elif isinstance(element, (ParabolicMirror, SphericalMirror)):
            focal_length_m = element.focal_length * 1e-3
            proper.prop_lens(self.wfo, focal_length_m)
            self.direction = -self.direction
    
    def _reconstruct_phase_from_rays(
        self,
        x: NDArray,
        y: NDArray,
        opd_waves: NDArray,
        valid_mask: NDArray,
        physical_size: float,
        grid_size: int,
    ) -> NDArray:
        """从光线数据重建相位分布"""
        import warnings
        from scipy.interpolate import griddata
        
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        opd_valid = opd_waves[valid_mask]
        
        half_size = physical_size / 2
        coords = np.linspace(-half_size, half_size, grid_size)
        X_grid, Y_grid = np.meshgrid(coords, coords)
        
        min_rays_required = 4
        if len(x_valid) < min_rays_required:
            warnings.warn(
                f"有效光线数量不足（{len(x_valid)} < {min_rays_required}），"
                f"返回零相位分布",
                RuntimeWarning
            )
            return np.zeros((grid_size, grid_size))
        
        points = np.column_stack([x_valid, y_valid])
        phase_grid = griddata(
            points, opd_valid,
            (X_grid, Y_grid),
            method='linear',
            fill_value=0.0,
        )
        
        phase_grid = np.nan_to_num(phase_grid, nan=0.0)
        
        return phase_grid
    
    def propagate_distance(self, distance: float) -> SimulationResult:
        """沿光路传播指定距离（Zemax 序列模式）
        
        参数:
            distance: 从初始位置沿光路的累积传播距离（mm）
        
        返回:
            SimulationResult 对象
        """
        # 重置到初始状态
        self.reset()
        
        remaining_distance = distance
        
        # 按顺序处理元件
        for element in self.elements:
            d_to_element = element.path_length - self.current_path
            
            if d_to_element < 0:
                continue
            
            if remaining_distance < d_to_element:
                break
            
            # 传播到元件位置
            if d_to_element > 0:
                self._propagate_proper(d_to_element)
                remaining_distance -= d_to_element
                
                self.history.append(PropagationStep(
                    step_type='propagate',
                    z_start=self.current_z - self.direction * d_to_element,
                    z_end=self.current_z,
                    path_start=self.current_path - d_to_element,
                    path_end=self.current_path,
                ))
            
            # 处理元件
            if self.use_hybrid:
                self._apply_element_hybrid(element)
            else:
                self._apply_element_proper_only(element)
            
            self.history.append(PropagationStep(
                step_type='element',
                z_start=element.z_position,
                z_end=element.z_position,
                path_start=element.path_length,
                path_end=element.path_length,
                element=element,
            ))
        
        # 传播剩余距离
        if remaining_distance > 0:
            self._propagate_proper(remaining_distance)
            
            self.history.append(PropagationStep(
                step_type='propagate',
                z_start=self.current_z - self.direction * remaining_distance,
                z_end=self.current_z,
                path_start=self.current_path - remaining_distance,
                path_end=self.current_path,
            ))
        
        return self._get_current_result()
    
    def _get_current_result(self) -> SimulationResult:
        """获取当前波前状态"""
        wavefront = proper.prop_get_wavefront(self.wfo)
        amplitude = proper.prop_get_amplitude(self.wfo)
        phase = proper.prop_get_phase(self.wfo)
        sampling_m = proper.prop_get_sampling(self.wfo)
        sampling_mm = sampling_m * 1e3
        
        beam_radius = self._estimate_beam_radius(amplitude, sampling_mm)
        
        mask = amplitude > 0.1 * np.max(amplitude)
        if np.any(mask):
            phase_valid = phase[mask]
            phase_valid = phase_valid - np.mean(phase_valid)
            phase_waves = phase_valid / (2 * np.pi)
            wavefront_rms = np.std(phase_waves)
            wavefront_pv = np.max(phase_waves) - np.min(phase_waves)
        else:
            wavefront_rms = 0.0
            wavefront_pv = 0.0
        
        return SimulationResult(
            z=self.current_z,
            path_length=self.current_path,
            amplitude=amplitude,
            phase=phase,
            wavefront=wavefront,
            sampling=sampling_mm,
            beam_radius=beam_radius,
            wavefront_rms=wavefront_rms,
            wavefront_pv=wavefront_pv,
        )
    
    def _estimate_beam_radius(self, amplitude: NDArray, sampling: float) -> float:
        """从振幅分布估计光束半径（二阶矩方法）
        
        对于高斯振幅 A(r) = exp(-r²/w²)，光强 I(r) = exp(-2r²/w²)
        二阶矩 <r²> = w²/2，所以 w = sqrt(2 * <r²>)
        """
        n = amplitude.shape[0]
        coords = (np.arange(n) - n // 2) * sampling
        X, Y = np.meshgrid(coords, coords)
        R_sq = X**2 + Y**2
        
        intensity = amplitude**2
        total_intensity = np.sum(intensity)
        
        if total_intensity < 1e-10:
            return 0.0
        
        r_sq_mean = np.sum(R_sq * intensity) / total_intensity
        # 对于高斯光束：<r²> = w²/2，所以 w = sqrt(2 * <r²>)
        beam_radius = np.sqrt(2 * r_sq_mean)
        
        return beam_radius
    
    def reset(self) -> None:
        """重置仿真器到初始状态"""
        self.history.clear()
        self._initialize_wavefront()
    
    def get_psf(self) -> Tuple[NDArray, float]:
        """获取 PSF"""
        psf, sampling_m = proper.prop_end(self.wfo)
        sampling_mm = sampling_m * 1e3
        return psf, sampling_mm
