# -*- coding: utf-8 -*-
"""
D4sigma 光束直径计算器（理想方法）

本模块实现使用二阶矩方法计算光束直径的 D4sigmaCalculator 类。
D4sigma 定义为光束强度分布二阶矩的 4 倍标准差。

公式：D4σ = 4 × √(∫∫ I(x,y) × (x-x̄)² dxdy / ∫∫ I(x,y) dxdy)

对于理想高斯光束，D4sigma = 2×w（1/e² 半径的 2 倍）。

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5
"""

from typing import Union, Optional
import numpy as np

from .data_models import D4sigmaResult
from .exceptions import InvalidInputError


class D4sigmaCalculator:
    """D4sigma 光束直径计算器（理想方法）
    
    使用二阶矩方法计算光束直径。
    D4σ = 4 × √(∫∫ I(x,y) × (x-x̄)² dxdy / ∫∫ I(x,y) dxdy)
    
    对于理想高斯光束，D4sigma = 2×w（1/e² 半径的 2 倍）。
    
    支持两种输入类型：
    1. numpy 复振幅数组（需要提供 sampling 参数）
    2. PROPER 波前对象（自动获取采样间隔）
    
    Example:
        >>> calculator = D4sigmaCalculator()
        >>> # 使用 numpy 数组
        >>> result = calculator.calculate(complex_amplitude, sampling=1e-6)
        >>> print(f"X 方向直径: {result.dx * 1e3:.3f} mm")
        >>> 
        >>> # 使用 PROPER 波前对象
        >>> result = calculator.calculate(wfo)
        >>> print(f"平均直径: {result.d_mean * 1e3:.3f} mm")
    
    Requirements: 1.1, 1.2, 1.3, 1.4, 1.5
    """
    
    def calculate(
        self,
        data: Union[np.ndarray, "proper.WaveFront"],
        sampling: Optional[float] = None,
    ) -> D4sigmaResult:
        """计算 D4sigma 光束直径
        
        使用二阶矩方法计算光束的 X 和 Y 方向直径。
        
        计算步骤：
        1. 从复振幅计算强度：I = |E|²
        2. 计算质心：x̄ = ∫∫ I(x,y) × x dxdy / ∫∫ I(x,y) dxdy
        3. 计算二阶矩：σ² = ∫∫ I(x,y) × (x-x̄)² dxdy / ∫∫ I(x,y) dxdy
        4. D4sigma = 4 × σ
        
        参数:
            data: 复振幅数组或 PROPER 波前对象
                - 如果是 numpy 数组，必须是 2D 复数数组
                - 如果是 PROPER 波前对象，将自动提取振幅和采样间隔
            sampling: 采样间隔 (m)
                - 如果 data 是 numpy 数组，此参数必须提供
                - 如果 data 是 PROPER 波前对象，此参数可选（自动获取）
        
        返回:
            D4sigmaResult 对象，包含：
            - dx: X 方向直径 (m)
            - dy: Y 方向直径 (m)
            - d_mean: 平均直径 (m)
            - centroid_x: 质心 X 坐标 (m)
            - centroid_y: 质心 Y 坐标 (m)
            - total_power: 总功率（归一化）
        
        异常:
            InvalidInputError: 输入数据无效时抛出
                - 空数组
                - 采样间隔为零或负
                - numpy 数组输入时未提供 sampling 参数
        
        Requirements: 1.1, 1.2, 1.3, 1.4, 1.5
        """
        # 提取复振幅数组和采样间隔
        amplitude, dx = self._extract_data(data, sampling)
        
        # 验证输入
        self._validate_input(amplitude, dx)
        
        # 计算强度分布
        intensity = np.abs(amplitude) ** 2
        
        # 计算总功率
        total_power = np.sum(intensity)
        
        # 处理零功率情况
        if total_power == 0:
            raise InvalidInputError("输入数据的总功率为零，无法计算 D4sigma")
        
        # 创建坐标网格
        ny, nx = intensity.shape
        x = (np.arange(nx) - nx / 2) * dx
        y = (np.arange(ny) - ny / 2) * dx
        X, Y = np.meshgrid(x, y)
        
        # 计算质心
        centroid_x = np.sum(intensity * X) / total_power
        centroid_y = np.sum(intensity * Y) / total_power
        
        # 计算二阶矩
        sigma_x_squared = np.sum(intensity * (X - centroid_x) ** 2) / total_power
        sigma_y_squared = np.sum(intensity * (Y - centroid_y) ** 2) / total_power
        
        # 计算 D4sigma
        # D4σ = 4 × σ = 4 × √(σ²)
        sigma_x = np.sqrt(sigma_x_squared)
        sigma_y = np.sqrt(sigma_y_squared)
        
        d4sigma_x = 4 * sigma_x
        d4sigma_y = 4 * sigma_y
        d4sigma_mean = (d4sigma_x + d4sigma_y) / 2
        
        return D4sigmaResult(
            dx=d4sigma_x,
            dy=d4sigma_y,
            d_mean=d4sigma_mean,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            total_power=total_power,
        )

    
    def _extract_data(
        self,
        data: Union[np.ndarray, "proper.WaveFront"],
        sampling: Optional[float],
    ) -> tuple:
        """从输入数据中提取复振幅数组和采样间隔
        
        参数:
            data: 复振幅数组或 PROPER 波前对象
            sampling: 采样间隔 (m)，可选
        
        返回:
            (amplitude, dx) 元组
            - amplitude: 复振幅数组
            - dx: 采样间隔 (m)
        
        异常:
            InvalidInputError: 无法提取数据时抛出
        """
        # 检查是否为 PROPER 波前对象
        # PROPER 波前对象具有 wfarr 属性
        if hasattr(data, 'wfarr'):
            # 这是 PROPER 波前对象
            try:
                import proper
                amplitude = proper.prop_get_amplitude(data)
                # 获取采样间隔
                if sampling is not None:
                    dx = sampling
                else:
                    dx = proper.prop_get_sampling(data)
                return amplitude, dx
            except ImportError:
                raise InvalidInputError(
                    "检测到 PROPER 波前对象，但无法导入 proper 库"
                )
            except Exception as e:
                raise InvalidInputError(
                    f"从 PROPER 波前对象提取数据失败: {e}"
                )
        
        # 假设是 numpy 数组
        if not isinstance(data, np.ndarray):
            raise InvalidInputError(
                f"不支持的输入类型: {type(data).__name__}。"
                "支持的类型: numpy.ndarray 或 PROPER WaveFront 对象"
            )
        
        # numpy 数组必须提供 sampling 参数
        if sampling is None:
            raise InvalidInputError(
                "使用 numpy 数组输入时，必须提供 sampling 参数"
            )
        
        return data, sampling
    
    def _validate_input(
        self,
        amplitude: np.ndarray,
        sampling: float,
    ) -> None:
        """验证输入数据的有效性
        
        参数:
            amplitude: 复振幅数组
            sampling: 采样间隔 (m)
        
        异常:
            InvalidInputError: 输入数据无效时抛出
        """
        # 检查数组是否为空
        if amplitude.size == 0:
            raise InvalidInputError("输入数组为空")
        
        # 检查数组维度
        if amplitude.ndim != 2:
            raise InvalidInputError(
                f"输入数组必须是 2D 数组，当前维度: {amplitude.ndim}"
            )
        
        # 检查采样间隔
        if sampling <= 0:
            raise InvalidInputError(
                f"采样间隔必须为正数，当前值: {sampling}"
            )
        
        # 检查是否包含 NaN 或 Inf
        if np.any(np.isnan(amplitude)):
            raise InvalidInputError("输入数组包含 NaN 值")
        
        if np.any(np.isinf(amplitude)):
            raise InvalidInputError("输入数组包含 Inf 值")
