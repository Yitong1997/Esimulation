# -*- coding: utf-8 -*-
"""
M² 因子计算器

本模块实现通过多点光束直径拟合计算 M² 因子的功能。

M² 因子表征实际光束与理想高斯光束的偏离程度：
- M² = 1：理想高斯光束
- M² > 1：实际光束，值越大偏离越大

拟合公式：w(z)² = w₀² × [1 + (M² × λ × (z-z₀) / (π × w₀²))²]

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import Tuple, Optional
import warnings

from .data_models import M2Result
from .exceptions import InvalidInputError, InsufficientDataError


class M2Calculator:
    """M² 因子计算器
    
    通过多点光束直径拟合计算 M² 因子。
    
    拟合公式：w(z)² = w₀² × [1 + (M² × λ × (z-z₀) / (π × w₀²))²]
    
    其中：
    - w(z): 位置 z 处的光束半径
    - w₀: 束腰半径
    - z₀: 束腰位置
    - M²: M² 因子
    - λ: 波长
    
    使用方法：
    ```python
    calculator = M2Calculator(wavelength=633e-9)  # 波长 633nm
    result = calculator.calculate(z_positions, beam_diameters_x, beam_diameters_y)
    print(f"M² = {result.m2_mean}")
    ```
    
    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
    """
    
    # 最小推荐测量点数
    MIN_RECOMMENDED_POINTS = 5
    
    def __init__(self, wavelength: float):
        """初始化 M² 计算器
        
        参数:
            wavelength: 波长 (m)
        
        Raises:
            InvalidInputError: 如果波长为零或负值
        """
        if wavelength <= 0:
            raise InvalidInputError(f"波长必须为正值，当前值: {wavelength}")
        
        self.wavelength = wavelength

    def calculate(
        self,
        z_positions: np.ndarray,
        beam_diameters_x: np.ndarray,
        beam_diameters_y: np.ndarray,
    ) -> M2Result:
        """通过曲线拟合计算 M² 因子
        
        对 X 和 Y 方向分别进行光束因果曲线拟合，计算各自的 M² 因子。
        
        参数:
            z_positions: 传输位置数组 (m)
            beam_diameters_x: X 方向光束直径数组 (m)
            beam_diameters_y: Y 方向光束直径数组 (m)
        
        返回:
            M2Result 对象，包含：
            - m2_x, m2_y: X 和 Y 方向的 M² 因子
            - w0_x, w0_y: 拟合得到的束腰半径
            - z0_x, z0_y: 拟合得到的束腰位置
            - r_squared_x, r_squared_y: 拟合优度
            - warning: 如果测量点数不足则包含警告信息
        
        Raises:
            InvalidInputError: 如果输入数组为空或长度不一致
            InsufficientDataError: 如果测量点数少于 3 个（拟合最低要求）
        
        Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
        """
        # 转换为 numpy 数组
        z = np.asarray(z_positions, dtype=np.float64)
        dx = np.asarray(beam_diameters_x, dtype=np.float64)
        dy = np.asarray(beam_diameters_y, dtype=np.float64)
        
        # 输入验证
        self._validate_inputs(z, dx, dy)
        
        # 检查测量点数并生成警告
        warning = None
        if len(z) < self.MIN_RECOMMENDED_POINTS:
            warning = (
                f"测量点数 ({len(z)}) 少于推荐的最小值 ({self.MIN_RECOMMENDED_POINTS})，"
                f"拟合结果可能不可靠。建议增加测量点数以提高精度。"
            )
            warnings.warn(warning, UserWarning)
        
        # 分别拟合 X 和 Y 方向
        m2_x, w0_x, z0_x, r_squared_x = self._fit_beam_caustic(z, dx)
        m2_y, w0_y, z0_y, r_squared_y = self._fit_beam_caustic(z, dy)
        
        # 计算平均值
        m2_mean = (m2_x + m2_y) / 2.0
        
        return M2Result(
            m2_x=m2_x,
            m2_y=m2_y,
            m2_mean=m2_mean,
            w0_x=w0_x,
            w0_y=w0_y,
            z0_x=z0_x,
            z0_y=z0_y,
            r_squared_x=r_squared_x,
            r_squared_y=r_squared_y,
            wavelength=self.wavelength,
            warning=warning,
        )

    def _fit_beam_caustic(
        self,
        z: np.ndarray,
        d: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """拟合光束因果曲线
        
        使用非线性最小二乘法拟合光束直径随传输距离的变化曲线。
        
        拟合函数（直径形式）：
        d(z)² = d₀² × [1 + (M² × λ × (z-z₀) / (π × w₀²))²]
        
        其中 d₀ = 2×w₀（束腰直径 = 2 × 束腰半径）
        
        参数:
            z: 传输位置数组 (m)
            d: 光束直径数组 (m)
        
        返回:
            (m2, w0, z0, r_squared) 元组：
            - m2: M² 因子
            - w0: 束腰半径 (m)
            - z0: 束腰位置 (m)
            - r_squared: 拟合优度 R²
        
        Requirements: 3.2
        """
        # 保存波长的局部引用，供内部函数使用
        wavelength = self.wavelength
        
        # 定义拟合函数：d² = d0² × [1 + (M² × λ × (z-z0) / (π × w0²))²]
        # 其中 w0 = d0/2
        def beam_caustic_squared(z, d0, z0, m2):
            """光束因果曲线的平方形式
            
            使用标准 M² 光束传播公式：
            d(z)² = d₀² × [1 + (M² × λ × (z-z₀) / (π × w₀²))²]
            
            参数:
                z: 传输位置
                d0: 束腰直径 (2×w0)
                z0: 束腰位置
                m2: M² 因子
            
            返回:
                d² 值
            """
            w0 = d0 / 2.0  # 束腰半径
            # 直接使用 M² 公式中的因子
            # factor = M² × λ / (π × w0²)
            factor = m2 * wavelength / (np.pi * w0**2)
            # d² = d0² × [1 + (factor × (z-z0))²]
            return d0**2 * (1.0 + (factor * (z - z0))**2)
        
        # 计算 d² 用于拟合
        d_squared = d**2
        
        # 估计初始参数
        d0_init = np.min(d)  # 最小直径作为束腰直径的初始估计
        z0_init = z[np.argmin(d)]  # 最小直径位置作为束腰位置的初始估计
        
        # 估计初始 M² 值：使用远场数据点的斜率
        # 在远场，d ≈ d0 × M² × λ × |z-z0| / (π × w0²)
        # 所以 M² ≈ d × π × w0² / (d0 × λ × |z-z0|)
        w0_init = d0_init / 2.0
        z_range = np.max(np.abs(z - z0_init))
        if z_range > 0:
            d_max = np.max(d)
            # 使用远场近似估计 M²
            m2_estimate = d_max * np.pi * w0_init**2 / (d0_init * wavelength * z_range)
            # 确保 M² 在合理范围内 [1.0, 100.0]
            m2_init = max(1.0, min(99.0, m2_estimate))
        else:
            m2_init = 1.0
        
        # 参数边界
        # d0 > 0, z0 可以是任意值, m2 >= 1
        bounds = (
            [1e-12, -np.inf, 1.0],  # 下界
            [np.inf, np.inf, 100.0]  # 上界（M² 通常不超过 100）
        )
        
        try:
            # 执行非线性拟合
            popt, pcov = curve_fit(
                beam_caustic_squared,
                z,
                d_squared,
                p0=[d0_init, z0_init, m2_init],
                bounds=bounds,
                maxfev=10000,
            )
            
            d0_fit, z0_fit, m2_fit = popt
            w0_fit = d0_fit / 2.0  # 转换为束腰半径
            
            # 计算拟合优度 R²
            d_squared_fit = beam_caustic_squared(z, d0_fit, z0_fit, m2_fit)
            ss_res = np.sum((d_squared - d_squared_fit)**2)
            ss_tot = np.sum((d_squared - np.mean(d_squared))**2)
            
            if ss_tot > 0:
                r_squared = 1.0 - ss_res / ss_tot
            else:
                r_squared = 1.0  # 所有数据点相同
            
            # 确保 R² 在 [0, 1] 范围内
            r_squared = max(0.0, min(1.0, r_squared))
            
            return m2_fit, w0_fit, z0_fit, r_squared
            
        except RuntimeError as e:
            # 拟合失败时返回默认值
            warnings.warn(f"光束因果曲线拟合失败: {e}", UserWarning)
            return 1.0, np.min(d) / 2.0, z[np.argmin(d)], 0.0

    def _validate_inputs(
        self,
        z: np.ndarray,
        dx: np.ndarray,
        dy: np.ndarray,
    ) -> None:
        """验证输入数据
        
        参数:
            z: 传输位置数组
            dx: X 方向光束直径数组
            dy: Y 方向光束直径数组
        
        Raises:
            InvalidInputError: 如果输入数组为空或长度不一致
            InsufficientDataError: 如果测量点数少于 3 个
        """
        # 检查空数组
        if len(z) == 0:
            raise InvalidInputError("传输位置数组不能为空")
        if len(dx) == 0:
            raise InvalidInputError("X 方向光束直径数组不能为空")
        if len(dy) == 0:
            raise InvalidInputError("Y 方向光束直径数组不能为空")
        
        # 检查长度一致性
        if len(z) != len(dx) or len(z) != len(dy):
            raise InvalidInputError(
                f"输入数组长度不一致: z={len(z)}, dx={len(dx)}, dy={len(dy)}"
            )
        
        # 检查最小数据点数（拟合需要至少 3 个参数，所以至少需要 3 个数据点）
        if len(z) < 3:
            raise InsufficientDataError(
                f"测量点数 ({len(z)}) 不足，拟合至少需要 3 个数据点"
            )
        
        # 检查直径值是否为正
        if np.any(dx <= 0):
            raise InvalidInputError("X 方向光束直径必须全部为正值")
        if np.any(dy <= 0):
            raise InvalidInputError("Y 方向光束直径必须全部为正值")
        
        # 检查是否有 NaN 或 Inf
        if np.any(~np.isfinite(z)):
            raise InvalidInputError("传输位置数组包含 NaN 或 Inf 值")
        if np.any(~np.isfinite(dx)):
            raise InvalidInputError("X 方向光束直径数组包含 NaN 或 Inf 值")
        if np.any(~np.isfinite(dy)):
            raise InvalidInputError("Y 方向光束直径数组包含 NaN 或 Inf 值")
