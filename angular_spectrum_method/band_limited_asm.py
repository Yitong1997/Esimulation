# -*- coding: utf-8 -*-
"""
带限角谱法（Band-Limited ASM）模块

本模块实现带限角谱法，通过应用带宽限制窗函数防止远场传播中的混叠。

带限角谱法的核心思想：
1. 计算带宽限制 νₗ = 1/(λ√((2Δνz)² + 1))
2. 应用窗函数 W = rect(νx/(2νₗx)) × rect(νy/(2νₗy))
3. 在传递函数上乘以窗函数后进行传播计算

主要特性：
- 防止远场传播中的混叠
- 支持零填充扩展（expand=True）
- 提供就地操作版本以节省内存

参考文献：
    Matsushima & Shimobaba, Opt. Express 17, 19662-19673 (2009)

Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 8.2
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple

from .core import rect, spatial_frequencies, transfer_function
from .utils import select_region

# 类型别名
ComplexArray = NDArray[np.complexfloating]
RealArray = NDArray[np.floating]


def _get_expanded_size(shape: Tuple[int, int]) -> Tuple[int, int]:
    """
    计算 4 倍零填充扩展后的尺寸
    
    参数：
        shape: 原始数组形状 (Ny, Nx)
    
    返回：
        扩展后的形状 (2*Ny, 2*Nx)
    """
    return (2 * shape[0], 2 * shape[1])


def _compute_bandwidth_limit(
    wavelength: float,
    delta_nu: float,
    z: float
) -> float:
    """
    计算带宽限制 νₗ
    
    数学公式：
        νₗ = 1/(λ√((2Δνz)² + 1))
    
    参数：
        wavelength: 波长 λ
        delta_nu: 频率采样间隔 Δν = 1/(N×Δ)
        z: 传播距离
    
    返回：
        带宽限制 νₗ
    
    Validates: Requirements 3.2
    """
    # νₗ = 1/(λ√((2Δνz)² + 1))
    return 1.0 / (wavelength * np.sqrt((2 * delta_nu * z) ** 2 + 1))


def band_limited_asm(
    u: ComplexArray,
    wavelength: float,
    dx: float,
    dy: float,
    z: float,
    *,
    expand: bool = True
) -> ComplexArray:
    """
    带限角谱法（Band-Limited ASM）
    
    通过应用带宽限制窗函数防止远场传播中的混叠。
    
    数学原理：
        1. 带宽限制计算：
           νₗ = 1/(λ√((2Δνz)² + 1))
           
        2. 窗函数：
           W = rect(νx/(2νₗx)) × rect(νy/(2νₗy))
           
        3. 传播计算：
           U_out = IFFT{ FFT{U_in} × H × W }
           
           其中 H 是传递函数：
           H(νx, νy) = exp(2πiz√(1/λ² - νx² - νy²))
    
    参数：
        u: 输入光场，形状为 (Ny, Nx)，必须是复数类型
        wavelength: 波长 λ（单位与 dx, dy, z 一致）
        dx: x 方向采样间隔
        dy: y 方向采样间隔
        z: 传播距离（正值表示向前传播，负值表示向后传播）
        expand: 是否进行 4 倍零填充扩展（默认 True）
            - True: 将数组扩展到 (2Ny, 2Nx) 以抑制混叠
            - False: 使用原始尺寸进行计算
    
    返回：
        传播后的光场，形状与输入相同 (Ny, Nx)
    
    示例：
        >>> import numpy as np
        >>> from angular_spectrum_method import band_limited_asm
        >>> 
        >>> # 创建高斯光束
        >>> x = np.linspace(-1e-3, 1e-3, 64)
        >>> y = np.linspace(-1e-3, 1e-3, 64)
        >>> X, Y = np.meshgrid(x, y)
        >>> u = np.exp(-(X**2 + Y**2) / (0.5e-3)**2).astype(complex)
        >>> 
        >>> # 使用带限角谱法传播 10cm（远场）
        >>> result = band_limited_asm(u, 633e-9, x[1]-x[0], y[1]-y[0], 0.1)
        >>> print(result.shape)
        (64, 64)
    
    注意：
        - x 轴为水平方向（列方向），y 轴为垂直方向（行方向）
        - 输入数组的形状约定为 (Ny, Nx)，即 (行数, 列数)
        - 带宽限制窗函数可以有效防止远场传播中的混叠
        - 对于近场传播，可以使用基础 ASM
    
    参考文献：
        Matsushima & Shimobaba, Opt. Express 17, 19662-19673 (2009)
    
    Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5
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
    
    # 生成空间频率（DC 在角落，与 FFT 输出对应）
    # Julia: ν = fftfreq.(N, inv.([Δy, Δx]))
    # 注意：Julia 中 N 的顺序是 (ny, nx)，对应 inv.([Δy, Δx]) = [1/Δy, 1/Δx]
    # 所以 ν[1] 对应 y 方向，ν[2] 对应 x 方向
    nu_y = spatial_frequencies(work_ny, dy)  # 形状 (work_ny,)
    nu_x = spatial_frequencies(work_nx, dx)  # 形状 (work_nx,)
    
    # 计算频率采样间隔
    # Julia: Δν = inv.(N.*[Δy, Δx])
    # Δν[1] = 1/(N_y * Δy)，Δν[2] = 1/(N_x * Δx)
    delta_nu_y = 1.0 / (work_ny * dy)
    delta_nu_x = 1.0 / (work_nx * dx)
    
    # 计算带宽限制
    # Julia: νₗ = @. 1/(λ*√((2Δν*z)^2 + 1))
    # νₗ[1] 对应 y 方向，νₗ[2] 对应 x 方向
    nu_l_y = _compute_bandwidth_limit(wavelength, delta_nu_y, z)
    nu_l_x = _compute_bandwidth_limit(wavelength, delta_nu_x, z)
    
    # 创建 2D 频率网格（使用广播）
    # Julia: H = @. exp(2π*im*z*√(1/λ^2 - ν[1]^2 - ν[2]'^2 + 0im))
    # ν[1]^2 是列向量 (ny,)，ν[2]'^2 是行向量 (1, nx)
    # 广播后得到 (ny, nx) 的矩阵
    nu_y_2d = nu_y.reshape(-1, 1)  # (work_ny, 1) - 列向量
    nu_x_2d = nu_x.reshape(1, -1)  # (1, work_nx) - 行向量
    
    # 计算传递函数
    # Julia: H = @. exp(2π*im*z*√(1/λ^2 - ν[1]^2 - ν[2]'^2 + 0im))
    H = transfer_function(nu_x_2d, nu_y_2d, wavelength, z)
    
    # 计算窗函数
    # Julia: W = @. rect(ν[1]/(2*νₗ[1]))*rect(ν[2]'/(2*νₗ[2]))
    # W = rect(νy/(2νₗy)) × rect(νx/(2νₗx))
    # 注意：rect 函数在 |x| < 0.5 时返回 1，否则返回 0
    # 所以 rect(νy/(2νₗy)) = 1 当 |νy| < νₗy 时
    W_y = rect(nu_y_2d / (2 * nu_l_y))  # (work_ny, 1)
    W_x = rect(nu_x_2d / (2 * nu_l_x))  # (1, work_nx)
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


def band_limited_asm_(
    u: ComplexArray,
    wavelength: float,
    dx: float,
    dy: float,
    z: float,
    *,
    expand: bool = True
) -> None:
    """
    就地版本的 BandLimitedASM，直接修改输入数组
    
    此函数与 band_limited_asm() 功能相同，但直接修改输入数组而不创建新数组。
    适用于内存受限的场景。
    
    参数：
        u: 输入/输出光场，形状为 (Ny, Nx)，必须是复数类型
           函数执行后，此数组将包含传播后的光场
        wavelength: 波长 λ
        dx: x 方向采样间隔
        dy: y 方向采样间隔
        z: 传播距离
        expand: 是否进行 4 倍零填充扩展（默认 True）
    
    返回：
        None（结果直接写入输入数组 u）
    
    示例：
        >>> import numpy as np
        >>> from angular_spectrum_method import band_limited_asm_
        >>> 
        >>> # 创建光场
        >>> u = np.ones((64, 64), dtype=complex)
        >>> 
        >>> # 就地传播
        >>> band_limited_asm_(u, 633e-9, 1e-6, 1e-6, 0.1)
        >>> # u 现在包含传播后的光场
    
    注意：
        - 输入数组必须是复数类型（如 np.complex64 或 np.complex128）
        - 如果输入数组不是复数类型，行为未定义
        - 虽然名为"就地"操作，但在 expand=True 时仍需要临时内存
    
    Validates: Requirements 8.2
    """
    # 计算传播结果
    result = band_limited_asm(u, wavelength, dx, dy, z, expand=expand)
    
    # 将结果复制回输入数组
    u[:] = result
