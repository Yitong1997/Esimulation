# -*- coding: utf-8 -*-
"""
核心数学函数模块

本模块提供角谱法计算所需的核心数学函数，包括：
- rect: 矩形函数
- spatial_frequencies: 空间频率生成
- transfer_function: 传递函数计算

这些函数为所有角谱法变体提供基础计算支持。
"""

import numpy as np
from numpy.typing import NDArray
from typing import Union

# 类型别名
ComplexArray = NDArray[np.complexfloating]
RealArray = NDArray[np.floating]
Scalar = Union[float, np.floating]


def rect(x: Union[Scalar, RealArray]) -> Union[float, RealArray]:
    """
    矩形函数（rect function）
    
    数学定义：
        rect(x) = 1  当 |x| < 0.5
        rect(x) = 0  否则
    
    参数：
        x: 标量或 NumPy 数组输入
    
    返回：
        与输入相同形状的输出。标量输入返回 float，数组输入返回相同形状的数组。
    
    示例：
        >>> rect(0.0)
        1.0
        >>> rect(0.5)
        0.0
        >>> rect(np.array([-0.3, 0.0, 0.3, 0.6]))
        array([1., 1., 1., 0.])
    
    注意：
        - 边界值 |x| = 0.5 时返回 0.0
        - 支持标量和 NumPy 数组输入
    
    Validates: Requirements 1.1, 1.2
    """
    # 处理标量输入
    if np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0):
        # 将 0 维数组转换为标量
        x_val = float(x)
        return 1.0 if abs(x_val) < 0.5 else 0.0
    
    # 处理数组输入
    x_arr = np.asarray(x)
    result = np.where(np.abs(x_arr) < 0.5, 1.0, 0.0)
    return result


def spatial_frequencies(n: int, delta: float) -> RealArray:
    """
    生成空间频率数组（DC 在角落）
    
    等价于 numpy.fft.fftfreq(n, delta)，生成与 FFT 输出对应的空间频率。
    
    数学定义：
        对于 n 个采样点和采样间隔 delta：
        - 频率范围: [-1/(2*delta), 1/(2*delta)]
        - 频率间隔: 1/(n*delta)
        - DC 分量位于索引 0
    
    参数：
        n: 数组长度（采样点数）
        delta: 采样间隔（空间域）
    
    返回：
        空间频率数组，形状为 (n,)，DC 分量在索引 0
    
    示例：
        >>> spatial_frequencies(4, 1.0)
        array([ 0.  ,  0.25, -0.5 , -0.25])
        >>> spatial_frequencies(5, 0.1)
        array([ 0.,  2.,  4., -4., -2.])
    
    注意：
        - 输出与 numpy.fft.fftfreq(n, delta) 完全一致
        - DC 分量（零频率）位于数组的第一个位置
        - 正频率在前半部分，负频率在后半部分
    
    Validates: Requirements 1.3
    """
    return np.fft.fftfreq(n, delta)


def transfer_function(
    nu_x: RealArray,
    nu_y: RealArray,
    wavelength: float,
    z: float
) -> ComplexArray:
    """
    计算角谱传递函数
    
    数学公式：
        H(νx, νy) = exp(2πiz√(1/λ² - νx² - νy²))
    
    对于传播波（νx² + νy² < 1/λ²）：
        H = exp(2πiz * √(1/λ² - νx² - νy²))  （纯相位）
    
    对于倏逝波（νx² + νy² > 1/λ²）：
        H = exp(-2πz * √(νx² + νy² - 1/λ²))  （指数衰减）
    
    参数：
        nu_x: x 方向空间频率数组
        nu_y: y 方向空间频率数组
        wavelength: 波长 λ（单位与空间频率一致）
        z: 传播距离（单位与波长一致）
    
    返回：
        传递函数数组，形状由 nu_x 和 nu_y 的广播规则决定
    
    示例：
        >>> nu_x = np.array([0.0, 0.1, 0.2])
        >>> nu_y = np.array([0.0])
        >>> H = transfer_function(nu_x, nu_y, 633e-9, 0.01)
    
    注意：
        - 使用广播机制处理 2D 频率网格
        - 倏逝波通过指数衰减处理，不会被完全消除
        - 传播波保持能量守恒（|H| = 1）
    
    Validates: Requirements 1.4
    """
    # 计算 1/λ²
    inv_lambda_sq = 1.0 / (wavelength ** 2)
    
    # 计算 νx² + νy²（使用广播）
    nu_sq = nu_x ** 2 + nu_y ** 2
    
    # 计算根号内的值：1/λ² - νx² - νy²
    radicand = inv_lambda_sq - nu_sq
    
    # 使用复数平方根处理正负情况
    # 当 radicand >= 0 时：sqrt(radicand) 是实数，exp(2πiz*sqrt) 是纯相位
    # 当 radicand < 0 时：sqrt(radicand) 是纯虚数，exp(2πiz*sqrt) 是指数衰减
    # 
    # 具体来说：
    # sqrt(radicand + 0j) 当 radicand < 0 时返回 i*sqrt(|radicand|)
    # 所以 exp(2πiz * i*sqrt(|radicand|)) = exp(-2πz*sqrt(|radicand|))
    
    sqrt_term = np.sqrt(radicand.astype(np.complex128))
    
    # 计算传递函数 H = exp(2πiz * sqrt_term)
    H = np.exp(2j * np.pi * z * sqrt_term)
    
    return H
