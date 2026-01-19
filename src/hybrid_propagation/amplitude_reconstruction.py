"""
复振幅重建模块

本模块实现从光线数据重建网格化复振幅场的功能。

主要功能：
- 将光线数据插值到规则网格
- 应用参考相位
- 处理无效区域

复用 optiland 的算法：
- 振幅计算：使用 optiland FFTPSF 的归一化方法
- 相位计算：使用 optiland 的 OPD 到相位转换
- 网格化：使用 scipy.interpolate.griddata（与 optiland 一致）

作者：混合光学仿真项目
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple
from scipy.interpolate import griddata
import warnings

# 尝试导入 optiland 的后端模块（用于兼容性）
try:
    import optiland.backend as be
    HAS_OPTILAND_BACKEND = True
except ImportError:
    HAS_OPTILAND_BACKEND = False


class AmplitudeReconstructor:
    """复振幅重建器
    
    从光线数据重建网格化的复振幅场。
    
    复用 optiland 的算法：
    - 振幅归一化：与 FFTPSF._generate_pupils() 一致
    - 相位计算：OPD(waves) → phase(rad) = -2π × OPD
    - 网格化插值：使用 scipy.interpolate.griddata
    
    参数:
        grid_size: 输出网格大小
        physical_size: 物理尺寸（直径），单位 mm
        wavelength: 波长，单位 μm
        use_optiland_normalization: 是否使用 optiland 的振幅归一化方法，默认 True
    
    示例:
        >>> import numpy as np
        >>> # 创建重建器
        >>> reconstructor = AmplitudeReconstructor(
        ...     grid_size=64,
        ...     physical_size=20.0,
        ...     wavelength=0.633,
        ... )
        >>> 
        >>> # 光线数据
        >>> ray_x = np.array([0.0, 1.0, 2.0])
        >>> ray_y = np.array([0.0, 1.0, 2.0])
        >>> ray_intensity = np.array([1.0, 1.0, 1.0])
        >>> ray_opd_waves = np.array([0.0, 0.1, 0.2])
        >>> reference_phase = np.zeros((64, 64))
        >>> valid_mask = np.array([True, True, True])
        >>> 
        >>> # 重建复振幅
        >>> amplitude = reconstructor.reconstruct(
        ...     ray_x, ray_y, ray_intensity, ray_opd_waves,
        ...     reference_phase, valid_mask
        ... )
    
    **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**
    """
    
    def __init__(
        self,
        grid_size: int,
        physical_size: float,
        wavelength: float,
        use_optiland_normalization: bool = True,
    ):
        """初始化重建器
        
        参数:
            grid_size: 输出网格大小
            physical_size: 物理尺寸（直径），单位 mm
            wavelength: 波长，单位 μm
            use_optiland_normalization: 是否使用 optiland 的振幅归一化方法
        """
        self.grid_size = grid_size
        self.physical_size = physical_size
        self.wavelength = wavelength
        self.use_optiland_normalization = use_optiland_normalization
        
        # 创建输出网格坐标
        half_size = physical_size / 2
        self.x_grid = np.linspace(-half_size, half_size, grid_size)
        self.y_grid = np.linspace(-half_size, half_size, grid_size)
        self.X_grid, self.Y_grid = np.meshgrid(self.x_grid, self.y_grid)
        
        # 创建归一化坐标网格（用于与 optiland 兼容）
        self.x_norm = np.linspace(-1, 1, grid_size)
        self.y_norm = np.linspace(-1, 1, grid_size)
        self.X_norm, self.Y_norm = np.meshgrid(self.x_norm, self.y_norm)
        self.R2_norm = self.X_norm**2 + self.Y_norm**2
    
    def reconstruct(
        self,
        ray_x: NDArray[np.floating],
        ray_y: NDArray[np.floating],
        ray_intensity: NDArray[np.floating],
        ray_opd_waves: NDArray[np.floating],
        reference_phase: NDArray[np.floating],
        valid_mask: NDArray[np.bool_],
    ) -> NDArray[np.complexfloating]:
        """重建复振幅
        
        从光线数据重建网格化的复振幅场。
        
        算法说明（复用 optiland FFTPSF._generate_pupils 的方法）：
        1. 振幅计算：intensity / mean(valid_intensity)，然后开方
        2. 相位计算：-2π × OPD（注意负号，与 optiland 一致）
        3. 复振幅：amplitude × exp(j × phase)
        
        参数:
            ray_x: 光线 x 坐标数组，单位 mm
            ray_y: 光线 y 坐标数组，单位 mm
            ray_intensity: 光线强度数组
            ray_opd_waves: 光线 OPD 数组（波长数）
            reference_phase: 参考相位网格（弧度）
            valid_mask: 有效光线掩模
        
        返回:
            复振幅数组，形状 (grid_size, grid_size)
        
        **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**
        """
        # 提取有效光线数据
        valid_x = ray_x[valid_mask]
        valid_y = ray_y[valid_mask]
        valid_intensity = ray_intensity[valid_mask]
        valid_opd = ray_opd_waves[valid_mask]
        
        # 检查是否有有效光线
        if len(valid_x) == 0:
            warnings.warn("没有有效光线，返回零复振幅", UserWarning)
            return np.zeros((self.grid_size, self.grid_size), dtype=np.complex128)
        
        # =====================================================================
        # 1. 计算振幅（复用 optiland FFTPSF 的归一化方法）
        # =====================================================================
        if self.use_optiland_normalization:
            # optiland 方法：amplitude = intensity / mean(valid_intensity)
            # 然后在构建复振幅时直接使用（不开方）
            # 参见 optiland/psf/fft.py: _generate_pupils()
            mean_valid_intensity = np.mean(valid_intensity[valid_intensity > 0])
            if mean_valid_intensity > 0:
                normalized_amplitude = valid_intensity / mean_valid_intensity
            else:
                normalized_amplitude = np.zeros_like(valid_intensity)
        else:
            # 传统方法：振幅 = sqrt(强度)
            normalized_amplitude = np.sqrt(valid_intensity)
        
        # 插值振幅到网格
        amplitude_grid = self._interpolate_to_grid(
            valid_x, valid_y, normalized_amplitude
        )
        
        # =====================================================================
        # 2. 计算相位（复用 optiland 的 OPD 到相位转换）
        # =====================================================================
        # optiland 使用：exp(-j * 2π * OPD)
        # 参见 optiland/psf/fft.py: P[R2 <= 1] = amplitude * exp(-1j * 2 * pi * opd)
        # 注意：optiland 使用负号！
        ray_phase = -2 * np.pi * valid_opd
        
        # 插值相位到网格
        phase_grid = self._interpolate_to_grid(
            valid_x, valid_y, ray_phase
        )
        
        # =====================================================================
        # 3. 应用参考相位
        # =====================================================================
        total_phase = self._apply_reference_phase(phase_grid, reference_phase)
        
        # =====================================================================
        # 4. 构建复振幅（与 optiland 一致）
        # =====================================================================
        complex_amplitude = amplitude_grid * np.exp(1j * total_phase)
        
        # =====================================================================
        # 5. 处理无效区域（NaN -> 0）
        # =====================================================================
        complex_amplitude = np.nan_to_num(complex_amplitude, nan=0.0)
        
        return complex_amplitude.astype(np.complex128)
    
    def reconstruct_optiland_style(
        self,
        ray_x: NDArray[np.floating],
        ray_y: NDArray[np.floating],
        ray_intensity: NDArray[np.floating],
        ray_opd_waves: NDArray[np.floating],
        valid_mask: NDArray[np.bool_],
        pupil_radius: Optional[float] = None,
    ) -> NDArray[np.complexfloating]:
        """使用完全复用 optiland 风格的方法重建复振幅
        
        这个方法更接近 optiland FFTPSF._generate_pupils() 的实现：
        1. 使用归一化光瞳坐标 [-1, 1]
        2. 在单位圆内填充复振幅
        3. 使用 optiland 的振幅归一化和相位符号约定
        
        参数:
            ray_x: 光线 x 坐标数组，单位 mm
            ray_y: 光线 y 坐标数组，单位 mm
            ray_intensity: 光线强度数组
            ray_opd_waves: 光线 OPD 数组（波长数）
            valid_mask: 有效光线掩模
            pupil_radius: 光瞳半径，单位 mm。如果为 None，使用 physical_size/2
        
        返回:
            复振幅数组，形状 (grid_size, grid_size)
        """
        # 提取有效光线数据
        valid_x = ray_x[valid_mask]
        valid_y = ray_y[valid_mask]
        valid_intensity = ray_intensity[valid_mask]
        valid_opd = ray_opd_waves[valid_mask]
        
        if len(valid_x) == 0:
            warnings.warn("没有有效光线，返回零复振幅", UserWarning)
            return np.zeros((self.grid_size, self.grid_size), dtype=np.complex128)
        
        # 确定光瞳半径
        if pupil_radius is None:
            pupil_radius = self.physical_size / 2
        
        # 将物理坐标转换为归一化坐标
        valid_x_norm = valid_x / pupil_radius
        valid_y_norm = valid_y / pupil_radius
        
        # =====================================================================
        # 复用 optiland FFTPSF._generate_pupils() 的算法
        # =====================================================================
        
        # 创建归一化坐标网格
        x = np.linspace(-1, 1, self.grid_size)
        x_grid, y_grid = np.meshgrid(x, x)
        x_flat = x_grid.ravel()
        y_flat = y_grid.ravel()
        R2 = x_flat**2 + y_flat**2
        
        # 初始化复振幅数组
        P = np.zeros_like(x_flat, dtype=np.complex128)
        
        # 计算归一化振幅（optiland 方法）
        valid_intensities = valid_intensity[valid_intensity > 0]
        if len(valid_intensities) > 0:
            mean_valid_intensity = np.mean(valid_intensities)
            amplitude = valid_intensity / mean_valid_intensity
        else:
            amplitude = np.zeros_like(valid_intensity)
        
        # 插值到网格点
        points = np.column_stack([valid_x_norm, valid_y_norm])
        
        # 插值振幅
        amplitude_interp = griddata(
            points, amplitude,
            (x_flat, y_flat),
            method='linear',
            fill_value=0.0
        )
        
        # 插值 OPD
        opd_interp = griddata(
            points, valid_opd,
            (x_flat, y_flat),
            method='linear',
            fill_value=0.0
        )
        
        # 在单位圆内填充复振幅（optiland 风格）
        # P[R2 <= 1] = amplitude * exp(-1j * 2 * pi * opd)
        mask = R2 <= 1
        P[mask] = amplitude_interp[mask] * np.exp(-1j * 2 * np.pi * opd_interp[mask])
        
        # 重塑为 2D 数组
        pupil = P.reshape((self.grid_size, self.grid_size))
        
        return pupil
    
    def _interpolate_to_grid(
        self,
        ray_x: NDArray[np.floating],
        ray_y: NDArray[np.floating],
        ray_values: NDArray[np.floating],
        method: str = 'linear',
    ) -> NDArray[np.floating]:
        """将光线数据插值到网格
        
        使用 scipy.interpolate.griddata 进行插值（与 optiland 一致）。
        
        参数:
            ray_x: 光线 x 坐标数组
            ray_y: 光线 y 坐标数组
            ray_values: 光线数据值数组
            method: 插值方法，'linear', 'nearest', 或 'cubic'
        
        返回:
            插值后的网格数据
        
        **Validates: Requirements 7.2**
        """
        # 构造插值点
        points = np.column_stack([ray_x, ray_y])
        
        # 执行插值（与 optiland OPD.generate_opd_map 一致）
        grid_values = griddata(
            points,
            ray_values,
            (self.X_grid, self.Y_grid),
            method=method,
            fill_value=np.nan,
        )
        
        return grid_values.astype(np.float64)
    
    def _apply_reference_phase(
        self,
        residual_phase: NDArray[np.floating],
        reference_phase: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """将参考相位加回到重建的相位中
        
        参数:
            residual_phase: 残差相位网格（弧度）
            reference_phase: 参考相位网格（弧度）
        
        返回:
            总相位网格（弧度）
        
        **Validates: Requirements 7.3**
        """
        # 确保参考相位与输出网格大小匹配
        if reference_phase.shape != (self.grid_size, self.grid_size):
            # 需要重采样参考相位
            from scipy.interpolate import RectBivariateSpline
            
            n_ref = reference_phase.shape[0]
            x_ref = np.linspace(-self.physical_size/2, self.physical_size/2, n_ref)
            y_ref = np.linspace(-self.physical_size/2, self.physical_size/2, n_ref)
            
            interp = RectBivariateSpline(y_ref, x_ref, reference_phase)
            reference_phase_resampled = interp(self.y_grid, self.x_grid)
        else:
            reference_phase_resampled = reference_phase
        
        # 加回参考相位
        total_phase = residual_phase + reference_phase_resampled
        
        return total_phase.astype(np.float64)
    
    def reconstruct_from_corrected_rays(
        self,
        ray_x: NDArray[np.floating],
        ray_y: NDArray[np.floating],
        ray_intensity: NDArray[np.floating],
        corrected_opd_waves: NDArray[np.floating],
        reference_phase: NDArray[np.floating],
        valid_mask: NDArray[np.bool_],
    ) -> NDArray[np.complexfloating]:
        """从修正后的光线数据重建复振幅
        
        这是 reconstruct 的别名，用于更清晰地表达意图。
        
        参数:
            ray_x: 光线 x 坐标数组，单位 mm
            ray_y: 光线 y 坐标数组，单位 mm
            ray_intensity: 光线强度数组
            corrected_opd_waves: 修正后的光线 OPD 数组（波长数）
            reference_phase: 参考相位网格（弧度）
            valid_mask: 有效光线掩模
        
        返回:
            复振幅数组，形状 (grid_size, grid_size)
        """
        return self.reconstruct(
            ray_x, ray_y, ray_intensity, corrected_opd_waves,
            reference_phase, valid_mask
        )
    
    def get_coverage_mask(
        self,
        ray_x: NDArray[np.floating],
        ray_y: NDArray[np.floating],
        valid_mask: NDArray[np.bool_],
        margin: float = 0.0,
    ) -> NDArray[np.bool_]:
        """获取光线覆盖区域掩模
        
        返回一个布尔数组，指示哪些网格点在有效光线覆盖范围内。
        
        参数:
            ray_x: 光线 x 坐标数组
            ray_y: 光线 y 坐标数组
            valid_mask: 有效光线掩模
            margin: 边缘余量，单位 mm
        
        返回:
            覆盖区域掩模，形状 (grid_size, grid_size)
        
        **Validates: Requirements 7.5**
        """
        valid_x = ray_x[valid_mask]
        valid_y = ray_y[valid_mask]
        
        if len(valid_x) == 0:
            return np.zeros((self.grid_size, self.grid_size), dtype=bool)
        
        # 计算有效光线的边界
        x_min = np.min(valid_x) - margin
        x_max = np.max(valid_x) + margin
        y_min = np.min(valid_y) - margin
        y_max = np.max(valid_y) + margin
        
        # 创建掩模
        mask = (
            (self.X_grid >= x_min) & (self.X_grid <= x_max) &
            (self.Y_grid >= y_min) & (self.Y_grid <= y_max)
        )
        
        return mask
    
    def get_circular_pupil_mask(
        self,
        pupil_radius: Optional[float] = None,
    ) -> NDArray[np.bool_]:
        """获取圆形光瞳掩模
        
        返回一个布尔数组，指示哪些网格点在圆形光瞳内。
        与 optiland 的 R2 <= 1 掩模一致。
        
        参数:
            pupil_radius: 光瞳半径，单位 mm。如果为 None，使用 physical_size/2
        
        返回:
            圆形光瞳掩模，形状 (grid_size, grid_size)
        """
        if pupil_radius is None:
            pupil_radius = self.physical_size / 2
        
        R2 = (self.X_grid / pupil_radius)**2 + (self.Y_grid / pupil_radius)**2
        return R2 <= 1
