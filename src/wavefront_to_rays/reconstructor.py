"""
光线到波前复振幅重建器模块

本模块实现将稀疏光线数据（位置、OPD）重建为规则网格上的复振幅。
使用雅可比矩阵方法（网格变形）计算振幅，基于能量守恒原理。

核心功能：
1. 使用雅可比矩阵方法计算振幅（基于能量守恒）
2. 将稀疏光线数据插值到 PROPER 网格
3. 正确处理无效光线区域（振幅为 0）
4. 检测相位突变并发出警告

物理原理：
- 能量守恒：I_in × dA_in = I_out × dA_out
- 振幅比：A_out / A_in = sqrt(I_out / I_in) = 1 / sqrt(|J|)
- 其中 |J| 是雅可比行列式（局部面积放大率）
- 复振幅公式：A × exp(-j × 2π × OPD)
"""

from typing import Tuple
import numpy as np

from .exceptions import InsufficientRaysError


class RayToWavefrontReconstructor:
    """光线到波前复振幅重建器
    
    使用雅可比矩阵方法（网格变形）计算振幅，基于能量守恒原理。
    将稀疏光线数据（位置、OPD）重建为规则网格上的复振幅。
    
    核心算法：
    1. 使用 RBF 插值创建输入→输出坐标映射
    2. 使用数值微分计算雅可比矩阵
    3. 基于能量守恒计算振幅：A = 1 / sqrt(|J|)
    4. 使用复振幅公式：A × exp(-j × 2π × OPD)
    5. 将稀疏数据插值到规则网格
    
    使用示例：
    ```python
    # 创建重建器
    reconstructor = RayToWavefrontReconstructor(
        grid_size=512,
        sampling_mm=0.01,
        wavelength_um=0.633
    )
    
    # 重建复振幅
    complex_amplitude = reconstructor.reconstruct(
        ray_x_in, ray_y_in,
        ray_x_out, ray_y_out,
        opd_waves, valid_mask
    )
    ```
    
    属性:
        grid_size (int): 输出网格大小（与 PROPER 网格一致）
        sampling_mm (float): 网格采样间隔 (mm/pixel)
        wavelength_um (float): 波长 (μm)
    """
    
    def __init__(
        self,
        grid_size: int,
        sampling_mm: float,
        wavelength_um: float,
    ):
        """初始化重建器
        
        参数:
            grid_size: 输出网格大小（与 PROPER 网格一致）
                      必须为正整数，通常为 2 的幂次（如 256, 512, 1024）
            sampling_mm: 网格采样间隔 (mm/pixel)
                        必须为正数
            wavelength_um: 波长 (μm)
                          必须为正数，通常在可见光范围 (0.38-0.78 μm)
        
        异常:
            ValueError: 如果参数不满足有效性要求
        
        示例:
            >>> reconstructor = RayToWavefrontReconstructor(
            ...     grid_size=512,
            ...     sampling_mm=0.01,
            ...     wavelength_um=0.633
            ... )
        """
        # 参数验证
        self._validate_grid_size(grid_size)
        self._validate_sampling_mm(sampling_mm)
        self._validate_wavelength_um(wavelength_um)
        
        # 保存参数
        self.grid_size = grid_size
        self.sampling_mm = sampling_mm
        self.wavelength_um = wavelength_um
    
    def _validate_grid_size(self, grid_size: int) -> None:
        """验证网格大小参数
        
        参数:
            grid_size: 输出网格大小
        
        异常:
            ValueError: 如果 grid_size 不是正整数
        """
        if not isinstance(grid_size, (int, np.integer)):
            raise ValueError(
                f"grid_size 必须为整数，但收到类型 {type(grid_size).__name__}"
            )
        if grid_size <= 0:
            raise ValueError(
                f"grid_size 必须为正整数，但收到 {grid_size}"
            )
    
    def _validate_sampling_mm(self, sampling_mm: float) -> None:
        """验证采样间隔参数
        
        参数:
            sampling_mm: 网格采样间隔 (mm/pixel)
        
        异常:
            ValueError: 如果 sampling_mm 不是正数
        """
        if not isinstance(sampling_mm, (int, float, np.integer, np.floating)):
            raise ValueError(
                f"sampling_mm 必须为数值类型，但收到类型 {type(sampling_mm).__name__}"
            )
        if sampling_mm <= 0:
            raise ValueError(
                f"sampling_mm 必须为正数，但收到 {sampling_mm}"
            )
        if np.isnan(sampling_mm) or np.isinf(sampling_mm):
            raise ValueError(
                f"sampling_mm 必须为有限正数，但收到 {sampling_mm}"
            )
    
    def _validate_wavelength_um(self, wavelength_um: float) -> None:
        """验证波长参数
        
        参数:
            wavelength_um: 波长 (μm)
        
        异常:
            ValueError: 如果 wavelength_um 不是正数
        """
        if not isinstance(wavelength_um, (int, float, np.integer, np.floating)):
            raise ValueError(
                f"wavelength_um 必须为数值类型，但收到类型 {type(wavelength_um).__name__}"
            )
        if wavelength_um <= 0:
            raise ValueError(
                f"wavelength_um 必须为正数，但收到 {wavelength_um}"
            )
        if np.isnan(wavelength_um) or np.isinf(wavelength_um):
            raise ValueError(
                f"wavelength_um 必须为有限正数，但收到 {wavelength_um}"
            )
    
    @property
    def grid_half_size_mm(self) -> float:
        """获取网格半尺寸 (mm)
        
        返回:
            网格半尺寸，即从中心到边缘的距离 (mm)
        """
        return self.sampling_mm * self.grid_size / 2
    
    @property
    def grid_extent_mm(self) -> Tuple[float, float]:
        """获取网格范围 (mm)
        
        返回:
            (min, max) 元组，表示网格坐标范围 (mm)
        """
        half_size = self.grid_half_size_mm
        return (-half_size, half_size)
    
    def __repr__(self) -> str:
        """返回对象的字符串表示"""
        return (
            f"RayToWavefrontReconstructor("
            f"grid_size={self.grid_size}, "
            f"sampling_mm={self.sampling_mm}, "
            f"wavelength_um={self.wavelength_um})"
        )
    
    def __str__(self) -> str:
        """返回对象的可读字符串表示"""
        return (
            f"光线到波前复振幅重建器:\n"
            f"  网格大小: {self.grid_size} × {self.grid_size}\n"
            f"  采样间隔: {self.sampling_mm} mm/pixel\n"
            f"  波长: {self.wavelength_um} μm\n"
            f"  网格范围: [{self.grid_extent_mm[0]:.3f}, {self.grid_extent_mm[1]:.3f}] mm"
        )
    
    def _compute_amplitude_phase_jacobian(
        self,
        ray_x_in: np.ndarray,
        ray_y_in: np.ndarray,
        ray_x_out: np.ndarray,
        ray_y_out: np.ndarray,
        opd_waves: np.ndarray,
        valid_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算复振幅的振幅和相位分量（基于雅可比矩阵）
        
        通过计算输入面到输出面的网格变形，使用雅可比矩阵计算光强变化。
        
        注意：此方法复用已有的光线追迹结果，不需要额外的追迹。
        输入/输出光线位置在 ElementRaytracer.trace() 过程中已经获得。
        
        物理原理：
        - 能量守恒：I_in × dA_in = I_out × dA_out
        - 振幅比：A_out / A_in = sqrt(I_out / I_in) = 1 / sqrt(|J|)
        - 其中 |J| 是雅可比行列式（局部面积放大率）
        
        参数:
            ray_x_in: 输入面光线 x 坐标 (mm)，来自采样光线
            ray_y_in: 输入面光线 y 坐标 (mm)，来自采样光线
            ray_x_out: 输出面光线 x 坐标 (mm)，来自追迹结果
            ray_y_out: 输出面光线 y 坐标 (mm)，来自追迹结果
            opd_waves: OPD (波长数)
            valid_mask: 有效光线掩模
        
        返回:
            (amplitude, phase) 元组
            - amplitude: 振幅数组，与输入光线数组同形状
            - phase: 相位数组（弧度），与输入光线数组同形状
        
        异常:
            ValueError: 如果有效光线数量不足（< 4）
        """
        from scipy.interpolate import RBFInterpolator
        
        # 只使用有效光线
        valid_x_in = ray_x_in[valid_mask]
        valid_y_in = ray_y_in[valid_mask]
        valid_x_out = ray_x_out[valid_mask]
        valid_y_out = ray_y_out[valid_mask]
        valid_opd = opd_waves[valid_mask]
        
        # 检查有效光线数量（需求 6.1）
        if len(valid_x_in) < 4:
            raise InsufficientRaysError(
                f"有效光线数量不足：{len(valid_x_in)} < 4。"
                f"无法进行复振幅重建，请检查光学系统配置或增加采样光线数量。"
            )
        
        # 创建从输入坐标到输出坐标的映射函数
        # 使用 RBF 插值创建平滑的映射函数
        points_in = np.column_stack([valid_x_in, valid_y_in])
        
        # 使用 thin_plate_spline 核函数，适合平滑的坐标映射
        interp_x = RBFInterpolator(points_in, valid_x_out, kernel='thin_plate_spline')
        interp_y = RBFInterpolator(points_in, valid_y_out, kernel='thin_plate_spline')
        
        # 计算雅可比矩阵的各分量（使用数值微分）
        # 微分步长选择：足够小以保证精度，但不能太小以避免数值误差
        eps = 1e-6  # 微分步长 (mm)
        
        # 在每个有效光线位置计算雅可比矩阵
        jacobian_det = np.zeros(len(valid_x_in))
        
        for i in range(len(valid_x_in)):
            x0, y0 = valid_x_in[i], valid_y_in[i]
            
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
        
        # 避免除零（需求 6.4）
        # 使用最小值限制，确保数值稳定性
        min_jacobian = 1e-10
        jacobian_det = np.maximum(jacobian_det, min_jacobian)
        
        # 振幅 = 1 / sqrt(|J|)（能量守恒）
        # 当 |J| > 1 时，光束扩展，振幅减小
        # 当 |J| < 1 时，光束收缩，振幅增大
        amplitude_valid = 1.0 / np.sqrt(jacobian_det)
        
        # 归一化振幅（需求 2.4）
        # 使归一化后的平均振幅为 1，保持相对变化
        mean_amplitude = np.mean(amplitude_valid)
        if mean_amplitude > 0:
            amplitude_valid = amplitude_valid / mean_amplitude
        
        # 构建完整的振幅数组
        # 无效光线区域振幅为 0（需求 2.6）
        amplitude = np.zeros_like(ray_x_in)
        amplitude[valid_mask] = amplitude_valid
        
        # 相位 = -2π × OPD（复振幅公式）
        # 负号是因为 OPD 增加对应相位滞后
        phase = np.zeros_like(ray_x_in)
        phase[valid_mask] = -2 * np.pi * valid_opd
        
        return amplitude, phase

    def _resample_to_grid_separate(
        self,
        ray_x: np.ndarray,
        ray_y: np.ndarray,
        amplitude: np.ndarray,
        phase: np.ndarray,
        valid_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """将光线数据重采样到规则网格（分别返回振幅和相位）
        
        关键设计决策：
        1. 分别插值振幅和相位
           - 避免直接插值复数导致的相位跳变问题
           - 振幅和相位有不同的物理特性，分别处理更合理
        2. 使用三次插值（cubic）提高精度
           - 三次插值比线性插值更平滑
           - 适合连续变化的物理量
        3. 采样范围外的区域设为 0（需求 5.4）
           - 使用 fill_value=0.0 处理外推区域
           - 确保无效区域不会引入伪信号
        4. 分别返回振幅和相位网格，便于后续相位突变检测
        
        参数:
            ray_x: 光线 x 坐标 (mm)，通常使用输出面位置
            ray_y: 光线 y 坐标 (mm)，通常使用输出面位置
            amplitude: 振幅数组，与光线数组同形状
            phase: 相位数组（弧度），与光线数组同形状
            valid_mask: 有效光线掩模（布尔数组）
        
        返回:
            (amp_grid, phase_grid) 元组
            - amp_grid: 振幅网格 (grid_size × grid_size)
            - phase_grid: 相位网格 (grid_size × grid_size)，单位弧度
        
        异常:
            ValueError: 如果有效光线数量不足（< 4）
        
        注意:
            - 输入的 ray_x, ray_y 应该是输出面的光线位置
            - 网格坐标范围由 self.sampling_mm 和 self.grid_size 决定
            - 插值使用 scipy.interpolate.griddata 的 'cubic' 方法
        """
        from scipy.interpolate import griddata
        
        # 只使用有效光线
        valid_x = ray_x[valid_mask]
        valid_y = ray_y[valid_mask]
        valid_amp = amplitude[valid_mask]
        valid_phase = phase[valid_mask]
        
        # 检查有效光线数量
        if len(valid_x) < 4:
            raise InsufficientRaysError(
                f"有效光线数量不足：{len(valid_x)} < 4。"
                f"无法进行网格重采样，请检查光学系统配置或增加采样光线数量。"
            )
        
        # 创建目标网格坐标
        # 网格范围：[-half_size, half_size]
        half_size = self.sampling_mm * self.grid_size / 2
        coords = np.linspace(-half_size, half_size, self.grid_size)
        X_grid, Y_grid = np.meshgrid(coords, coords)
        
        # 准备插值点（有效光线的坐标）
        points = np.column_stack([valid_x, valid_y])
        
        # 分别插值振幅和相位
        # 使用三次插值（cubic）提高精度
        # fill_value=0.0 确保采样范围外的区域设为 0（需求 5.4）
        amp_grid = griddata(
            points, 
            valid_amp, 
            (X_grid, Y_grid),
            method='cubic', 
            fill_value=0.0
        )
        
        phase_grid = griddata(
            points, 
            valid_phase, 
            (X_grid, Y_grid),
            method='cubic', 
            fill_value=0.0
        )
        
        # 处理可能的 NaN 值（三次插值在边界可能产生 NaN）
        # 将 NaN 替换为 0，确保数值稳定性
        amp_grid = np.nan_to_num(amp_grid, nan=0.0)
        phase_grid = np.nan_to_num(phase_grid, nan=0.0)
        
        # 确保振幅非负（物理约束）
        amp_grid = np.maximum(amp_grid, 0.0)
        
        return amp_grid, phase_grid

    def reconstruct(
        self,
        ray_x_in: np.ndarray,
        ray_y_in: np.ndarray,
        ray_x_out: np.ndarray,
        ray_y_out: np.ndarray,
        opd_waves: np.ndarray,
        valid_mask: np.ndarray,
        check_phase_discontinuity: bool = True,
    ) -> np.ndarray:
        """重建复振幅（使用雅可比矩阵方法）
        
        这是 RayToWavefrontReconstructor 的主要公共接口方法。
        将稀疏光线数据（输入/输出位置、OPD）重建为规则网格上的复振幅。
        
        处理流程：
        1. 使用雅可比矩阵计算振幅和相位
        2. 重采样到 PROPER 网格（使用输出位置）
        3. 检测相位突变（可选，重采样后、加回理想相位之前）
        4. 组合为复振幅
        
        物理原理：
        - 振幅：基于能量守恒，A = 1 / sqrt(|J|)，其中 |J| 是雅可比行列式
        - 相位：φ = -2π × OPD
        - 复振幅：A × exp(j × φ)
        
        参数:
            ray_x_in: 输入面光线 x 坐标 (mm)
                     来自采样光线，用于计算雅可比矩阵
            ray_y_in: 输入面光线 y 坐标 (mm)
                     来自采样光线，用于计算雅可比矩阵
            ray_x_out: 输出面光线 x 坐标 (mm)
                      来自光线追迹结果，用于计算雅可比矩阵和网格重采样
            ray_y_out: 输出面光线 y 坐标 (mm)
                      来自光线追迹结果，用于计算雅可比矩阵和网格重采样
            opd_waves: OPD (波长数)
                      光程差，用于计算相位
            valid_mask: 有效光线掩模（布尔数组）
                       True 表示有效光线，False 表示无效光线
            check_phase_discontinuity: 是否检测相位突变（默认 True）
                                      如果为 True，在重采样后检测相邻像素相位差
                                      如果检测到相位差 > π，发出 UserWarning
        
        返回:
            复振幅网格 (grid_size × grid_size 复数数组)
            - 数据类型：np.complex128
            - 形状：(self.grid_size, self.grid_size)
            - 有效区域包含振幅和相位信息
            - 无效区域（采样范围外）振幅为 0
        
        异常:
            ValueError: 如果有效光线数量不足（< 4）
        
        示例:
            >>> reconstructor = RayToWavefrontReconstructor(
            ...     grid_size=512,
            ...     sampling_mm=0.01,
            ...     wavelength_um=0.633
            ... )
            >>> complex_amplitude = reconstructor.reconstruct(
            ...     ray_x_in, ray_y_in,
            ...     ray_x_out, ray_y_out,
            ...     opd_waves, valid_mask
            ... )
            >>> print(complex_amplitude.shape)
            (512, 512)
            >>> print(complex_amplitude.dtype)
            complex128
        
        注意:
            - 此方法不包含理论曲率相位，需要在调用后单独加回
            - 相位突变检测在重采样后、加回理想相位之前执行
            - 如果 check_phase_discontinuity=True 且检测到相位突变，
              会发出 UserWarning 但不会中断执行
        
        参见:
            - _compute_amplitude_phase_jacobian: 振幅和相位计算
            - _resample_to_grid_separate: 网格重采样
            - _check_phase_discontinuity_on_grid: 相位突变检测（任务 6.1）
        """
        # 1. 使用雅可比矩阵计算振幅和相位
        # 振幅基于能量守恒：A = 1 / sqrt(|J|)
        # 相位基于 OPD：φ = -2π × OPD
        amplitude, phase = self._compute_amplitude_phase_jacobian(
            ray_x_in, ray_y_in, 
            ray_x_out, ray_y_out, 
            opd_waves, valid_mask
        )
        
        # 2. 重采样到 PROPER 网格（使用输出位置）
        # 分别插值振幅和相位，避免直接插值复数导致的问题
        amp_grid, phase_grid = self._resample_to_grid_separate(
            ray_x_out, ray_y_out, 
            amplitude, phase, 
            valid_mask
        )
        
        # 3. 检测相位突变（重采样后、加回理想相位之前）（需求 6.2）
        # 如果相邻像素相位差 > π，可能导致相位混叠
        if check_phase_discontinuity:
            self._check_phase_discontinuity_on_grid(phase_grid, amp_grid)
        
        # 4. 组合为复振幅
        # 复振幅 = 振幅 × exp(j × 相位)
        complex_amplitude = amp_grid * np.exp(1j * phase_grid)
        
        return complex_amplitude

    def _check_phase_discontinuity_on_grid(
        self,
        phase_grid: np.ndarray,
        amplitude_grid: np.ndarray,
        threshold_rad: float = np.pi,
    ) -> bool:
        """检测重采样后网格上的相位突变
        
        原理：
        - 相邻像素之间的相位差不应超过 π
        - 如果超过，可能导致相位混叠（aliasing）
        - 这是 Nyquist 采样准则的要求
        
        检测时机：
        - 在重采样完成后、加回理想相位之前
        - 只检测像差相位（已减除理想相位的部分）
        
        参数:
            phase_grid: 重采样后的相位网格 (弧度)
            amplitude_grid: 重采样后的振幅网格（用于确定有效区域）
            threshold_rad: 相位差阈值（弧度），默认 π
        
        返回:
            True 如果检测到相位突变，False 否则
        
        副作用:
            如果检测到相位突变，发出 UserWarning
        
        注意:
            此方法是占位符实现，完整实现将在任务 6.1 中完成。
            当前实现只进行基本检测，不发出警告。
        """
        import warnings
        
        # 创建有效区域掩模（振幅 > 0 的区域）
        valid_mask = amplitude_grid > 1e-10
        
        # 如果没有有效区域，直接返回
        if not np.any(valid_mask):
            return False
        
        # 计算 x 方向相邻像素的相位差
        phase_diff_x = np.abs(np.diff(phase_grid, axis=1))
        # 计算 y 方向相邻像素的相位差
        phase_diff_y = np.abs(np.diff(phase_grid, axis=0))
        
        # 只检查有效区域内的相位差
        # x 方向：两个相邻像素都必须有效
        valid_x = valid_mask[:, :-1] & valid_mask[:, 1:]
        # y 方向：两个相邻像素都必须有效
        valid_y = valid_mask[:-1, :] & valid_mask[1:, :]
        
        # 获取有效区域内的最大相位差
        max_diff_x = np.max(phase_diff_x[valid_x]) if np.any(valid_x) else 0
        max_diff_y = np.max(phase_diff_y[valid_y]) if np.any(valid_y) else 0
        max_phase_diff = max(max_diff_x, max_diff_y)
        
        # 检查是否超过阈值
        has_discontinuity = max_phase_diff > threshold_rad
        
        if has_discontinuity:
            max_opd_waves = max_phase_diff / (2 * np.pi)
            warnings.warn(
                f"检测到相位突变：重采样后网格上相邻像素最大相位差为 "
                f"{max_phase_diff:.3f} 弧度 ({max_opd_waves:.3f} 波长)，"
                f"超过阈值 {threshold_rad:.3f} 弧度（π）。"
                f"这可能导致相位混叠，建议增加采样密度。",
                UserWarning
            )
        
        return has_discontinuity
