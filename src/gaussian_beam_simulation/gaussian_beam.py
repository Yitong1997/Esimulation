"""
高斯光束定义模块

本模块定义高斯光束的参数和波前生成功能。

高斯光束参数：
- 束腰半径 w0
- 束腰位置 z0
- M² 因子（光束质量因子）
- 波长 λ
- 初始面位置 z_init
- 附加波前误差

理论基础：
高斯光束的复振幅分布：
    E(r, z) = E0 * (w0/w(z)) * exp(-r²/w(z)²) * exp(-i*k*r²/(2*R(z))) * exp(i*φ(z))

其中：
- w(z) = w0 * sqrt(1 + (z/zR)²)  光束半径
- R(z) = z * (1 + (zR/z)²)       波前曲率半径
- φ(z) = arctan(z/zR)            Gouy 相位
- zR = π * w0² / (M² * λ)        瑞利距离

作者：混合光学仿真项目
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple
import numpy as np
from numpy.typing import NDArray


@dataclass
class GaussianBeam:
    """高斯光束定义
    
    定义高斯光束的所有参数，并提供波前生成功能。
    
    参数:
        wavelength: 波长，单位：μm
        w0: 束腰半径，单位：mm
        z0: 束腰位置（相对于全局坐标系原点），单位：mm
            - 正值表示束腰在 +Z 方向
            - 负值表示束腰在 -Z 方向
        m2: M² 因子（光束质量因子），默认 1.0（理想高斯光束）
            - M² >= 1.0
            - M² = 1.0 表示理想高斯光束
            - M² > 1.0 表示实际光束（光束质量较差）
        z_init: 初始面位置，单位：mm
            - 波前将在此位置生成
        wavefront_error: 附加波前误差函数，可选
            - 函数签名：(X, Y) -> phase_error
            - X, Y 为网格坐标（mm）
            - 返回相位误差（弧度）
    
    属性:
        zR: 瑞利距离（mm）
        divergence: 远场发散角（rad）
    
    示例:
        >>> # 创建理想高斯光束
        >>> beam = GaussianBeam(
        ...     wavelength=0.5,  # 0.5 μm
        ...     w0=1.0,          # 束腰半径 1 mm
        ...     z0=-100.0,       # 束腰在 z=-100 mm
        ...     m2=1.0,          # 理想高斯光束
        ...     z_init=0.0,      # 初始面在 z=0
        ... )
        >>> print(f"瑞利距离: {beam.zR:.2f} mm")
        
        >>> # 创建带 M² 因子的高斯光束
        >>> beam = GaussianBeam(
        ...     wavelength=0.5,
        ...     w0=1.0,
        ...     z0=-100.0,
        ...     m2=1.3,  # M² = 1.3
        ...     z_init=0.0,
        ... )
    """
    
    wavelength: float  # μm
    w0: float  # mm
    z0: float  # mm
    m2: float = 1.0
    z_init: float = 0.0  # mm
    wavefront_error: Optional[Callable[[NDArray, NDArray], NDArray]] = None
    
    def __post_init__(self) -> None:
        """初始化后验证参数
        
        验证规则：
        - wavelength: 必须为正值（Requirements 1.5, 9.1）
        - w0: 必须为正值（Requirements 1.1, 9.2）
        - m2: 必须 >= 1.0（Requirements 1.3, 9.3）
        - z0: 必须为有限实数（Requirements 1.2）
        - z_init: 必须为有限实数（Requirements 1.4）
        """
        # 验证波长（必须为正值）
        if not isinstance(self.wavelength, (int, float)):
            raise TypeError(
                f"参数 'wavelength' 必须为数值类型，"
                f"实际类型为 {type(self.wavelength).__name__}"
            )
        if not np.isfinite(self.wavelength):
            raise ValueError(
                f"参数 'wavelength' 必须为有限值，"
                f"实际为 {self.wavelength} μm（无穷大或 NaN 不允许）"
            )
        if self.wavelength <= 0:
            raise ValueError(
                f"参数 'wavelength'（波长）必须为正值，"
                f"实际为 {self.wavelength} μm"
            )
        
        # 验证束腰半径（必须为正值）
        if not isinstance(self.w0, (int, float)):
            raise TypeError(
                f"参数 'w0' 必须为数值类型，"
                f"实际类型为 {type(self.w0).__name__}"
            )
        if not np.isfinite(self.w0):
            raise ValueError(
                f"参数 'w0'（束腰半径）必须为有限值，"
                f"实际为 {self.w0} mm（无穷大或 NaN 不允许）"
            )
        if self.w0 <= 0:
            raise ValueError(
                f"参数 'w0'（束腰半径）必须为正值，"
                f"实际为 {self.w0} mm"
            )
        
        # 验证 M² 因子（必须 >= 1.0）
        if not isinstance(self.m2, (int, float)):
            raise TypeError(
                f"参数 'm2' 必须为数值类型，"
                f"实际类型为 {type(self.m2).__name__}"
            )
        if not np.isfinite(self.m2):
            raise ValueError(
                f"参数 'm2'（M² 因子）必须为有限值，"
                f"实际为 {self.m2}（无穷大或 NaN 不允许）"
            )
        if self.m2 < 1.0:
            raise ValueError(
                f"参数 'm2'（M² 因子）必须 >= 1.0，"
                f"实际为 {self.m2}（物理上 M² 不能小于 1）"
            )
        
        # 验证束腰位置（必须为有限实数，可以为任意值）
        if not isinstance(self.z0, (int, float)):
            raise TypeError(
                f"参数 'z0' 必须为数值类型，"
                f"实际类型为 {type(self.z0).__name__}"
            )
        if not np.isfinite(self.z0):
            raise ValueError(
                f"参数 'z0'（束腰位置）必须为有限值，"
                f"实际为 {self.z0} mm（无穷大或 NaN 不允许）"
            )
        
        # 验证初始面位置（必须为有限实数，可以为任意值）
        if not isinstance(self.z_init, (int, float)):
            raise TypeError(
                f"参数 'z_init' 必须为数值类型，"
                f"实际类型为 {type(self.z_init).__name__}"
            )
        if not np.isfinite(self.z_init):
            raise ValueError(
                f"参数 'z_init'（初始面位置）必须为有限值，"
                f"实际为 {self.z_init} mm（无穷大或 NaN 不允许）"
            )
    
    @property
    def wavelength_mm(self) -> float:
        """波长（mm）"""
        return self.wavelength * 1e-3
    
    @property
    def k(self) -> float:
        """波数 k = 2π/λ（1/mm）"""
        return 2 * np.pi / self.wavelength_mm
    
    @property
    def zR(self) -> float:
        """瑞利距离（mm）
        
        zR = π * w0² / (M² * λ)
        """
        return np.pi * self.w0**2 / (self.m2 * self.wavelength_mm)
    
    @property
    def divergence(self) -> float:
        """远场发散角（rad）
        
        θ = M² * λ / (π * w0)
        """
        return self.m2 * self.wavelength_mm / (np.pi * self.w0)
    
    def w(self, z: float) -> float:
        """计算位置 z 处的光束半径（mm）
        
        w(z) = w0 * sqrt(1 + ((z - z0) / zR)²)
        
        参数:
            z: 位置（mm）
        
        返回:
            光束半径（mm）
        """
        dz = z - self.z0
        return self.w0 * np.sqrt(1 + (dz / self.zR)**2)
    
    def R(self, z: float) -> float:
        """计算位置 z 处的波前曲率半径（mm）
        
        R(z) = (z - z0) * (1 + (zR / (z - z0))²)
        
        在束腰处 R = ∞（平面波前）
        
        参数:
            z: 位置（mm）
        
        返回:
            波前曲率半径（mm），束腰处返回 np.inf
        """
        dz = z - self.z0
        if abs(dz) < 1e-10:
            return np.inf
        return dz * (1 + (self.zR / dz)**2)
    
    def gouy_phase(self, z: float) -> float:
        """计算位置 z 处的 Gouy 相位（rad）
        
        φ(z) = arctan((z - z0) / zR)
        
        参数:
            z: 位置（mm）
        
        返回:
            Gouy 相位（rad）
        """
        dz = z - self.z0
        return np.arctan(dz / self.zR)
    
    def generate_wavefront(
        self,
        grid_size: int,
        physical_size: float,
        z: Optional[float] = None,
        include_gouy_phase: bool = False,
        normalize: bool = False,
    ) -> NDArray:
        """生成指定位置的高斯光束波前复振幅
        
        根据高斯光束理论，生成包含振幅和相位信息的复振幅分布。
        
        参数:
            grid_size: 网格大小（像素）
            physical_size: 物理尺寸（直径），单位：mm
            z: 位置（mm），默认使用 z_init
            include_gouy_phase: 是否包含 Gouy 相位，默认 False
                - Gouy 相位是全局相位，不影响波前形状
                - 在混合仿真中通常可以忽略
            normalize: 是否归一化振幅（峰值为 1），默认 False
                - False: 使用物理振幅 (w0/w(z)) * exp(-r²/w(z)²)
                - True: 归一化为峰值 1
        
        返回:
            wavefront: 波前复振幅数组，形状为 (grid_size, grid_size)
                - 复数数组，包含振幅和相位信息
                - wavefront = amplitude * exp(i * phase)
        
        波前复振幅公式:
            E(r, z) = A(r, z) * exp(i * φ(r, z))
            
            其中：
            
            1. 振幅分布 A(r, z):
               A(r, z) = (w0/w(z)) * exp(-r²/w(z)²)
               
               - (w0/w(z)) 因子确保能量守恒
               - exp(-r²/w(z)²) 是高斯分布
               - 在 r = w(z) 处，振幅降至峰值的 1/e
               
            2. 相位分布 φ(r, z):
               φ(r, z) = -k * r² / (2 * R(z)) + φ_error(r)
               
               - 球面波前相位：-k * r² / (2 * R(z))
                 - k = 2π/λ 是波数
                 - R(z) 是波前曲率半径
                 - 在束腰处 R(z) = ∞，相位为零（平面波前）
               - φ_error(r) 是附加波前误差（如果指定）
               
            3. Gouy 相位（可选）:
               φ_gouy(z) = arctan((z - z0) / zR)
               
               - 这是全局相位，不影响波前形状
               - 在干涉测量中可能需要考虑
        
        示例:
            >>> beam = GaussianBeam(wavelength=0.5, w0=1.0, z0=-100.0)
            >>> wavefront = beam.generate_wavefront(
            ...     grid_size=256,
            ...     physical_size=10.0,  # 10 mm 直径
            ...     z=0.0,
            ... )
            >>> amplitude = np.abs(wavefront)
            >>> phase = np.angle(wavefront)
        
        注意:
            - 振幅分布符合高斯函数（Requirements 1.11, 3.2）
            - 相位分布包含正确的球面波前相位（Requirements 3.3）
            - 支持附加波前误差（Requirements 1.6, 1.7）
        """
        if z is None:
            z = self.z_init
        
        # 创建坐标网格
        half_size = physical_size / 2.0
        coords = np.linspace(-half_size, half_size, grid_size)
        X, Y = np.meshgrid(coords, coords)
        R_sq = X**2 + Y**2  # r²
        
        # 计算光束参数
        w_z = self.w(z)
        R_z = self.R(z)
        
        # ========== 振幅分布 ==========
        # A(r) = (w0/w(z)) * exp(-r²/w(z)²)
        #
        # 物理意义：
        # - (w0/w(z)) 因子：光束扩展时振幅降低，确保能量守恒
        # - exp(-r²/w(z)²)：高斯分布，在 r = w(z) 处降至 1/e
        #
        # 验证：在 r = 0 处，amplitude = w0/w(z)
        #       在 r = w(z) 处，amplitude = (w0/w(z)) * exp(-1) ≈ 0.368 * (w0/w(z))
        amplitude = (self.w0 / w_z) * np.exp(-R_sq / w_z**2)
        
        if normalize:
            # 归一化：峰值为 1
            amplitude = np.exp(-R_sq / w_z**2)
        
        # ========== 相位分布 ==========
        
        # 1. 球面波前相位：φ_curvature = -k * r² / (2 * R(z))
        #
        # 物理意义：
        # - 描述波前的曲率（发散或会聚）
        # - R(z) > 0：发散波前（曲率中心在 -z 方向）
        # - R(z) < 0：会聚波前（曲率中心在 +z 方向）
        # - R(z) = ∞：平面波前（束腰处）
        #
        # 符号约定：
        # - 负号确保发散波前（R > 0）在边缘处相位滞后
        # - 这与 PROPER 库的约定一致
        if np.isinf(R_z):
            curvature_phase = np.zeros_like(R_sq)
        else:
            curvature_phase = -self.k * R_sq / (2 * R_z)
        
        # 2. Gouy 相位（可选）
        #
        # φ_gouy(z) = arctan((z - z0) / zR)
        #
        # 物理意义：
        # - 高斯光束通过焦点时的额外相位延迟
        # - 从 z = -∞ 到 z = +∞，Gouy 相位变化 π
        # - 这是全局相位，不影响波前形状
        if include_gouy_phase:
            gouy = self.gouy_phase(z)
        else:
            gouy = 0.0
        
        # 3. 附加波前误差
        #
        # 支持用户自定义的波前误差函数，例如：
        # - Zernike 多项式形式的像差
        # - 表面形状误差
        # - 大气湍流引起的波前畸变
        #
        # 波前误差函数签名：(X, Y) -> phase_error (rad)
        # - X, Y：网格坐标（mm）
        # - 返回：相位误差（弧度）
        if self.wavefront_error is not None:
            error_phase = self.wavefront_error(X, Y)
        else:
            error_phase = np.zeros_like(R_sq)
        
        # ========== 总相位 ==========
        total_phase = curvature_phase + gouy + error_phase
        
        # ========== 复振幅 ==========
        # E = A * exp(i * φ)
        wavefront = amplitude * np.exp(1j * total_phase)
        
        return wavefront
    
    def get_beam_info_at(self, z: float) -> dict:
        """获取指定位置的光束信息
        
        参数:
            z: 位置（mm）
        
        返回:
            包含光束参数的字典
        """
        return {
            'z': z,
            'w': self.w(z),
            'R': self.R(z),
            'gouy_phase': self.gouy_phase(z),
            'distance_from_waist': z - self.z0,
            'normalized_distance': (z - self.z0) / self.zR,
        }
    
    def verify_wavefront(
        self,
        wavefront: NDArray,
        grid_size: int,
        physical_size: float,
        z: Optional[float] = None,
        rtol: float = 1e-6,
    ) -> dict:
        """验证波前生成的正确性
        
        检查生成的波前是否符合高斯光束理论。
        
        参数:
            wavefront: 波前复振幅数组
            grid_size: 网格大小
            physical_size: 物理尺寸（直径），单位：mm
            z: 位置（mm），默认使用 z_init
            rtol: 相对容差
        
        返回:
            验证结果字典，包含：
            - amplitude_gaussian: 振幅是否符合高斯分布
            - phase_spherical: 相位是否符合球面波前
            - peak_amplitude: 峰值振幅
            - expected_peak: 期望峰值
            - amplitude_error: 振幅误差
            - phase_error: 相位误差（如果适用）
        """
        if z is None:
            z = self.z_init
        
        # 提取振幅和相位
        amplitude = np.abs(wavefront)
        phase = np.angle(wavefront)
        
        # 创建坐标网格
        half_size = physical_size / 2.0
        coords = np.linspace(-half_size, half_size, grid_size)
        X, Y = np.meshgrid(coords, coords)
        R_sq = X**2 + Y**2
        
        # 计算期望值
        w_z = self.w(z)
        R_z = self.R(z)
        
        # 期望振幅
        expected_amplitude = (self.w0 / w_z) * np.exp(-R_sq / w_z**2)
        
        # 期望相位（球面波前）
        if np.isinf(R_z):
            expected_phase = np.zeros_like(R_sq)
        else:
            expected_phase = -self.k * R_sq / (2 * R_z)
        
        # 计算误差
        amplitude_error = np.max(np.abs(amplitude - expected_amplitude))
        
        # 相位误差需要考虑 2π 周期性
        phase_diff = phase - expected_phase
        # 将相位差归一化到 [-π, π]
        phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi
        phase_error = np.max(np.abs(phase_diff))
        
        # 验证结果
        amplitude_gaussian = amplitude_error < rtol * np.max(expected_amplitude)
        phase_spherical = phase_error < rtol * 2 * np.pi
        
        return {
            'amplitude_gaussian': amplitude_gaussian,
            'phase_spherical': phase_spherical,
            'peak_amplitude': np.max(amplitude),
            'expected_peak': self.w0 / w_z,
            'amplitude_error': amplitude_error,
            'phase_error': phase_error,
            'w_z': w_z,
            'R_z': R_z,
        }
    
    def __repr__(self) -> str:
        return (
            f"GaussianBeam(λ={self.wavelength}μm, w0={self.w0}mm, "
            f"z0={self.z0}mm, M²={self.m2}, zR={self.zR:.2f}mm)"
        )


def create_zernike_wavefront_error(
    coefficients: dict,
    pupil_radius: float,
) -> Callable[[NDArray, NDArray], NDArray]:
    """创建 Zernike 多项式波前误差函数
    
    参数:
        coefficients: Zernike 系数字典，键为 Noll 索引，值为系数（波长数）
            例如：{4: 0.1, 11: 0.05} 表示 0.1λ 离焦 + 0.05λ 球差
        pupil_radius: 光瞳半径（mm）
    
    返回:
        波前误差函数 (X, Y) -> phase_error (rad)
    
    常用 Zernike 项（Noll 索引）：
        1: Piston
        2: Tilt X
        3: Tilt Y
        4: Defocus（离焦）
        5: Astigmatism 45°
        6: Astigmatism 0°
        7: Coma X
        8: Coma Y
        11: Spherical（球差）
    """
    def wavefront_error(X: NDArray, Y: NDArray) -> NDArray:
        """计算波前误差"""
        # 归一化坐标
        rho = np.sqrt(X**2 + Y**2) / pupil_radius
        theta = np.arctan2(Y, X)
        
        # 光瞳外设为 0
        mask = rho <= 1.0
        
        # 计算 Zernike 多项式
        # 使用 float64 类型确保数值精度
        phase = np.zeros_like(X, dtype=np.float64)
        
        for noll_index, coeff in coefficients.items():
            if coeff == 0:
                continue
            
            # 计算 Zernike 多项式值
            z_value = _zernike_noll(noll_index, rho, theta)
            
            # 系数单位是波长数，转换为弧度
            phase = phase + coeff * 2 * np.pi * z_value
        
        # 光瞳外设为 0
        phase = np.where(mask, phase, 0.0)
        
        return phase
    
    return wavefront_error


def _zernike_noll(noll_index: int, rho: NDArray, theta: NDArray) -> NDArray:
    """计算 Noll 索引的 Zernike 多项式值
    
    简化实现，仅支持常用的几个 Zernike 项
    """
    if noll_index == 1:  # Piston
        return np.ones_like(rho)
    elif noll_index == 2:  # Tilt X
        return 2 * rho * np.cos(theta)
    elif noll_index == 3:  # Tilt Y
        return 2 * rho * np.sin(theta)
    elif noll_index == 4:  # Defocus
        return np.sqrt(3) * (2 * rho**2 - 1)
    elif noll_index == 5:  # Astigmatism 45°
        return np.sqrt(6) * rho**2 * np.sin(2 * theta)
    elif noll_index == 6:  # Astigmatism 0°
        return np.sqrt(6) * rho**2 * np.cos(2 * theta)
    elif noll_index == 7:  # Coma X
        return np.sqrt(8) * (3 * rho**3 - 2 * rho) * np.cos(theta)
    elif noll_index == 8:  # Coma Y
        return np.sqrt(8) * (3 * rho**3 - 2 * rho) * np.sin(theta)
    elif noll_index == 11:  # Spherical
        return np.sqrt(5) * (6 * rho**4 - 6 * rho**2 + 1)
    else:
        raise ValueError(f"不支持的 Noll 索引: {noll_index}")
