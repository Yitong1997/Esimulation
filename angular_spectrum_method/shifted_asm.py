# -*- coding: utf-8 -*-
"""
偏移角谱法（Shifted ASM）模块

本模块实现偏移角谱法，用于计算目标平面中心偏移 (x₀, y₀) 的衍射场。

偏移角谱法的核心思想：
1. 根据偏移量自适应计算带宽限制 ν₊ 和 ν₋
2. 计算中心频率 ν₀ 和带宽 νw
3. 在传递函数中包含偏移相位因子 exp(2πi(x₀νx + y₀νy))
4. 应用自适应窗函数 W = rect((νx-ν₀x)/νwx) × rect((νy-ν₀y)/νwy)

主要特性：
- 支持目标平面中心偏移
- 自适应带宽限制防止混叠
- 支持零填充扩展（expand=True）
- 提供就地操作版本以节省内存

参考文献：
    Matsushima, Opt. Express 18, 18453-18463 (2010)

Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 8.5
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Union

from .core import rect, spatial_frequencies
from .utils import select_region

# 类型别名
ComplexArray = NDArray[np.complexfloating]
RealArray = NDArray[np.floating]
Scalar = Union[float, np.floating]


def _get_expanded_size(shape: Tuple[int, int]) -> Tuple[int, int]:
    """
    计算 4 倍零填充扩展后的尺寸
    
    参数：
        shape: 原始数组形状 (Ny, Nx)
    
    返回：
        扩展后的形状 (2*Ny, 2*Nx)
    """
    return (2 * shape[0], 2 * shape[1])


def center_frequency(
    x0: Scalar,
    S: Scalar,
    nu_plus: Scalar,
    nu_minus: Scalar
) -> float:
    """
    计算自适应中心频率
    
    根据偏移量 x₀ 和半宽度 S 的关系，计算自适应中心频率 ν₀。
    
    数学定义：
        - 当 S ≤ x₀ 时：ν₀ = (ν₊ + ν₋) / 2
        - 当 -S < x₀ < S 时：ν₀ = (ν₊ - ν₋) / 2
        - 当 x₀ ≤ -S 时：ν₀ = -(ν₊ + ν₋) / 2
    
    参数：
        x0: 偏移量
        S: 半宽度（N × Δ / 2）
        nu_plus: 正向带宽限制 ν₊
        nu_minus: 负向带宽限制 ν₋
    
    返回：
        中心频率 ν₀
    
    示例：
        >>> center_frequency(1.0, 0.5, 0.8, 0.6)  # S ≤ x₀
        0.7
        >>> center_frequency(0.0, 0.5, 0.8, 0.6)  # -S < x₀ < S
        0.1
        >>> center_frequency(-1.0, 0.5, 0.8, 0.6)  # x₀ ≤ -S
        -0.7
    
    Validates: Requirements 6.3
    """
    x0 = float(x0)
    S = float(S)
    nu_plus = float(nu_plus)
    nu_minus = float(nu_minus)
    
    if S <= x0:
        # 偏移量大于半宽度：目标区域完全在正方向
        return (nu_plus + nu_minus) / 2.0
    elif -S < x0 < S:
        # 偏移量在半宽度范围内：目标区域跨越中心
        return (nu_plus - nu_minus) / 2.0
    elif x0 <= -S:
        # 偏移量小于负半宽度：目标区域完全在负方向
        return -(nu_plus + nu_minus) / 2.0
    else:
        # 理论上不会到达这里
        return 0.0


def bandwidth(
    x0: Scalar,
    S: Scalar,
    nu_plus: Scalar,
    nu_minus: Scalar
) -> float:
    """
    计算自适应带宽
    
    根据偏移量 x₀ 和半宽度 S 的关系，计算自适应带宽 νw。
    
    数学定义：
        - 当 S ≤ x₀ 时：νw = ν₊ - ν₋
        - 当 -S < x₀ < S 时：νw = ν₊ + ν₋
        - 当 x₀ ≤ -S 时：νw = ν₋ - ν₊
    
    参数：
        x0: 偏移量
        S: 半宽度（N × Δ / 2）
        nu_plus: 正向带宽限制 ν₊
        nu_minus: 负向带宽限制 ν₋
    
    返回：
        带宽 νw
    
    示例：
        >>> bandwidth(1.0, 0.5, 0.8, 0.6)  # S ≤ x₀
        0.2
        >>> bandwidth(0.0, 0.5, 0.8, 0.6)  # -S < x₀ < S
        1.4
        >>> bandwidth(-1.0, 0.5, 0.8, 0.6)  # x₀ ≤ -S
        -0.2
    
    Validates: Requirements 6.3
    """
    x0 = float(x0)
    S = float(S)
    nu_plus = float(nu_plus)
    nu_minus = float(nu_minus)
    
    if S <= x0:
        # 偏移量大于半宽度：目标区域完全在正方向
        return nu_plus - nu_minus
    elif -S < x0 < S:
        # 偏移量在半宽度范围内：目标区域跨越中心
        return nu_plus + nu_minus
    elif x0 <= -S:
        # 偏移量小于负半宽度：目标区域完全在负方向
        return nu_minus - nu_plus
    else:
        # 理论上不会到达这里
        return 0.0



def shifted_asm(
    u: ComplexArray,
    wavelength: float,
    dx: float,
    dy: float,
    z: float,
    x0: float,
    y0: float,
    *,
    expand: bool = True
) -> ComplexArray:
    """
    偏移角谱法（Shifted ASM）
    
    计算目标平面中心偏移 (x₀, y₀) 的衍射场。
    
    数学原理：
        1. 计算半宽度：S = N × Δ / 2
        
        2. 计算自适应带宽限制：
           ν₊ = 1 / (λ × √(z²/(r₀ + S)² + 1))
           ν₋ = 1 / (λ × √(z²/(r₀ - S)² + 1))
        
        3. 计算中心频率和带宽：
           ν₀ = CenterFrequency(r₀, S, ν₊, ν₋)
           νw = BandWidth(r₀, S, ν₊, ν₋)
        
        4. 传递函数（包含偏移相位因子）：
           H = exp(2πi((x₀νx + y₀νy) + z√(1/λ² - νx² - νy²)))
        
        5. 自适应窗函数：
           W = rect((νx - ν₀x)/νwx) × rect((νy - ν₀y)/νwy)
        
        6. 传播计算：
           U_out = IFFT{ FFT{U_in} × H × W }
    
    参数：
        u: 输入光场，形状为 (Ny, Nx)，必须是复数类型
        wavelength: 波长 λ（单位与 dx, dy, z, x0, y0 一致）
        dx: x 方向采样间隔
        dy: y 方向采样间隔
        z: 传播距离（正值表示向前传播，负值表示向后传播）
        x0: 目标平面 x 方向中心偏移量
        y0: 目标平面 y 方向中心偏移量
        expand: 是否进行 4 倍零填充扩展（默认 True）
            - True: 将数组扩展到 (2Ny, 2Nx) 以抑制混叠
            - False: 使用原始尺寸进行计算
    
    返回：
        偏移后的传播光场，形状与输入相同 (Ny, Nx)
    
    示例：
        >>> import numpy as np
        >>> from angular_spectrum_method import shifted_asm
        >>> 
        >>> # 创建高斯光束
        >>> x = np.linspace(-1e-3, 1e-3, 64)
        >>> y = np.linspace(-1e-3, 1e-3, 64)
        >>> X, Y = np.meshgrid(x, y)
        >>> u = np.exp(-(X**2 + Y**2) / (0.5e-3)**2).astype(complex)
        >>> 
        >>> # 使用偏移角谱法传播，目标平面中心偏移 (0.5mm, 0.3mm)
        >>> result = shifted_asm(u, 633e-9, x[1]-x[0], y[1]-y[0], 0.01, 0.5e-3, 0.3e-3)
        >>> print(result.shape)
        (64, 64)
    
    注意：
        - x 轴为水平方向（列方向），y 轴为垂直方向（行方向）
        - 输入数组的形状约定为 (Ny, Nx)，即 (行数, 列数)
        - 偏移量 x0, y0 的单位应与采样间隔 dx, dy 一致
        - 自适应带宽限制可以有效防止偏移传播中的混叠
    
    参考文献：
        Matsushima, Opt. Express 18, 18453-18463 (2010)
    
    Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5
    """
    # 获取原始尺寸
    original_shape = u.shape
    ny, nx = original_shape
    
    # 确定工作尺寸
    # Julia: N = ifelse(expand, size(u).*2, size(u))
    if expand:
        # 4 倍零填充扩展（2x 在每个维度）
        work_shape = _get_expanded_size(original_shape)
        # 将输入数组扩展到工作尺寸（居中放置，周围填充零）
        u_work = select_region(u, work_shape, center=True, pad_value=0)
    else:
        # 使用原始尺寸
        work_shape = original_shape
        u_work = select_region(u, work_shape, center=True, pad_value=0)
    
    work_ny, work_nx = work_shape
    
    # Julia: r₀ = [y₀, x₀]
    # Julia: S = N.*[Δy, Δx]./2
    # 注意：Julia 中 r₀[1] 对应 y₀，r₀[2] 对应 x₀
    # S[1] = N[1] * Δy / 2，S[2] = N[2] * Δx / 2
    r0_y = y0
    r0_x = x0
    S_y = work_ny * dy / 2.0
    S_x = work_nx * dx / 2.0
    
    # 生成空间频率（DC 在角落，与 FFT 输出对应）
    # Julia: ν = fftfreq.(N, inv.([Δy, Δx]))
    nu_y = spatial_frequencies(work_ny, dy)  # 形状 (work_ny,)
    nu_x = spatial_frequencies(work_nx, dx)  # 形状 (work_nx,)
    
    # 计算自适应带宽限制 ν₊ 和 ν₋
    # Julia: ν₊ = @. inv(λ*√(z^2/(r₀ + S)^2 + 1))
    # Julia: ν₋ = @. inv(λ*√(z^2/(r₀ - S)^2 + 1))
    # 
    # ν₊[1] = 1 / (λ * √(z²/(y₀ + S_y)² + 1))
    # ν₋[1] = 1 / (λ * √(z²/(y₀ - S_y)² + 1))
    # ν₊[2] = 1 / (λ * √(z²/(x₀ + S_x)² + 1))
    # ν₋[2] = 1 / (λ * √(z²/(x₀ - S_x)² + 1))
    
    # y 方向
    nu_plus_y = 1.0 / (wavelength * np.sqrt(z**2 / (r0_y + S_y)**2 + 1))
    nu_minus_y = 1.0 / (wavelength * np.sqrt(z**2 / (r0_y - S_y)**2 + 1))
    
    # x 方向
    nu_plus_x = 1.0 / (wavelength * np.sqrt(z**2 / (r0_x + S_x)**2 + 1))
    nu_minus_x = 1.0 / (wavelength * np.sqrt(z**2 / (r0_x - S_x)**2 + 1))
    
    # 计算中心频率和带宽
    # Julia: ν₀ = CenterFrequency.(r₀, S, ν₊, ν₋)
    # Julia: νw = BandWidth.(r₀, S, ν₊, ν₋)
    nu0_y = center_frequency(r0_y, S_y, nu_plus_y, nu_minus_y)
    nu0_x = center_frequency(r0_x, S_x, nu_plus_x, nu_minus_x)
    nuw_y = bandwidth(r0_y, S_y, nu_plus_y, nu_minus_y)
    nuw_x = bandwidth(r0_x, S_x, nu_plus_x, nu_minus_x)
    
    # 创建 2D 频率网格（使用广播）
    nu_y_2d = nu_y.reshape(-1, 1)  # (work_ny, 1) - 列向量
    nu_x_2d = nu_x.reshape(1, -1)  # (1, work_nx) - 行向量
    
    # 计算传递函数（包含偏移相位因子）
    # Julia: H = @. exp(2π*im*((r₀[1]*ν[1] + r₀[2]*ν[2]') + z*√(1/λ^2 - ν[1]^2 - ν[2]'^2 + 0im)))
    # 
    # H = exp(2πi * ((y₀*νy + x₀*νx) + z*√(1/λ² - νy² - νx²)))
    # 
    # 注意：Julia 中 ν[1] 是列向量，ν[2]' 是行向量
    # r₀[1]*ν[1] + r₀[2]*ν[2]' 通过广播得到 2D 矩阵
    
    # 计算 1/λ²
    inv_lambda_sq = 1.0 / (wavelength ** 2)
    
    # 计算 νx² + νy²（使用广播）
    nu_sq = nu_y_2d ** 2 + nu_x_2d ** 2
    
    # 计算根号内的值：1/λ² - νx² - νy²
    radicand = inv_lambda_sq - nu_sq
    
    # 使用复数平方根处理正负情况
    sqrt_term = np.sqrt(radicand.astype(np.complex128))
    
    # 偏移相位因子：y₀*νy + x₀*νx
    shift_phase = r0_y * nu_y_2d + r0_x * nu_x_2d
    
    # 传递函数 H = exp(2πi * (shift_phase + z * sqrt_term))
    H = np.exp(2j * np.pi * (shift_phase + z * sqrt_term))
    
    # 计算自适应窗函数
    # Julia: W = @. rect((ν[1] - ν₀[1])/νw[1])*rect((ν[2]' - ν₀[2])/νw[2])
    # W = rect((νy - ν₀y)/νwy) × rect((νx - ν₀x)/νwx)
    W_y = rect((nu_y_2d - nu0_y) / nuw_y)  # (work_ny, 1)
    W_x = rect((nu_x_2d - nu0_x) / nuw_x)  # (1, work_nx)
    W = W_y * W_x  # (work_ny, work_nx) - 广播乘法
    
    # Julia 实现：û = fftshift(ifft(fft(ifftshift(ũ)).*H.*W))
    # 1. ifftshift: 将居中的数组移动到 DC 在角落的布局
    # 2. fft: 进行 FFT
    # 3. 乘以 H.*W: 频域乘法（传递函数 × 窗函数）
    # 4. ifft: 进行 IFFT
    # 5. fftshift: 将结果移回居中布局
    
    # ifftshift 将数组中心移到角落（为 FFT 准备）
    u_shifted = np.fft.ifftshift(u_work)
    
    # FFT 变换到频域
    U = np.fft.fft2(u_shifted)
    
    # 频域乘法：传递函数 × 窗函数
    U_propagated = U * H * W
    
    # IFFT 变换回空域
    u_propagated = np.fft.ifft2(U_propagated)
    
    # fftshift 将结果中心移回中间
    u_result = np.fft.fftshift(u_propagated)
    
    # 如果进行了扩展，裁剪回原始尺寸
    # Julia: return select_region_view(û, new_size=size(u))
    if expand:
        result = select_region(u_result, original_shape, center=True)
    else:
        result = u_result
    
    return result


def shifted_asm_(
    u: ComplexArray,
    wavelength: float,
    dx: float,
    dy: float,
    z: float,
    x0: float,
    y0: float,
    *,
    expand: bool = True
) -> None:
    """
    就地版本的 ShiftedASM，直接修改输入数组
    
    此函数与 shifted_asm() 功能相同，但直接修改输入数组而不创建新数组。
    适用于内存受限的场景。
    
    参数：
        u: 输入/输出光场，形状为 (Ny, Nx)，必须是复数类型
           函数执行后，此数组将包含传播后的光场
        wavelength: 波长 λ
        dx: x 方向采样间隔
        dy: y 方向采样间隔
        z: 传播距离
        x0: 目标平面 x 方向中心偏移量
        y0: 目标平面 y 方向中心偏移量
        expand: 是否进行 4 倍零填充扩展（默认 True）
    
    返回：
        None（结果直接写入输入数组 u）
    
    示例：
        >>> import numpy as np
        >>> from angular_spectrum_method import shifted_asm_
        >>> 
        >>> # 创建光场
        >>> u = np.ones((64, 64), dtype=complex)
        >>> 
        >>> # 就地传播，目标平面中心偏移 (0.5mm, 0.3mm)
        >>> shifted_asm_(u, 633e-9, 1e-6, 1e-6, 0.01, 0.5e-3, 0.3e-3)
        >>> # u 现在包含传播后的光场
    
    注意：
        - 输入数组必须是复数类型（如 np.complex64 或 np.complex128）
        - 如果输入数组不是复数类型，行为未定义
        - 虽然名为"就地"操作，但在 expand=True 时仍需要临时内存
    
    Validates: Requirements 8.5
    """
    # 计算传播结果
    result = shifted_asm(u, wavelength, dx, dy, z, x0, y0, expand=expand)
    
    # 将结果复制回输入数组
    u[:] = result
