# -*- coding: utf-8 -*-
"""
可缩放角谱法（Scalable ASM）模块

本模块实现可缩放角谱法，自动计算目标平面采样间距以获得最优采样。

可缩放角谱法的核心思想：
1. 自动计算目标平面采样间距 Δd = λz/(pNΔs)，其中 p=2 为填充因子
2. 应用 Fresnel 核进行相位校正
3. 计算并应用带限窗函数防止混叠
4. 在传播距离超过渐晕限制或小于最小距离时发出警告

主要特性：
- 自动缩放的角谱传播
- 包含 Fresnel 核相位校正
- 支持零填充扩展（expand=True）
- 提供就地操作版本以节省内存

参考文献：
    Heintzmann et al., Optica 10, 1407-1416 (2023)

Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 8.3
"""

import warnings
import numpy as np
from numpy.typing import NDArray
from typing import Tuple

from .core import spatial_frequencies
from .utils import select_region

# 类型别名
ComplexArray = NDArray[np.complexfloating]
RealArray = NDArray[np.floating]


class PropagationWarning(UserWarning):
    """传播计算警告"""
    pass


def _get_expanded_size(shape: Tuple[int, int]) -> Tuple[int, int]:
    """
    计算 4 倍零填充扩展后的尺寸
    
    参数：
        shape: 原始数组形状 (Ny, Nx)
    
    返回：
        扩展后的形状 (2*Ny, 2*Nx)
    """
    return (2 * shape[0], 2 * shape[1])


def band_limit(
    wavelength: float,
    z: float,
    L: float,
    nu: RealArray,
    nu_z: ComplexArray
) -> RealArray:
    """
    计算带限函数
    
    数学公式：
        BandLimit(λ, z, L, ν, νz) = 1  当 |ν/νz - λν| ≤ L/(2z)
        BandLimit(λ, z, L, ν, νz) = 0  否则
    
    Julia 实现：
        ifelse(abs(ν/νz - λ*ν) ≤ L/(2z), one(λ), zero(λ))
    
    参数：
        wavelength: 波长 λ
        z: 传播距离
        L: 计算域尺寸
        nu: 空间频率数组
        nu_z: z 方向空间频率数组（复数）
    
    返回：
        带限窗函数数组，值为 0 或 1
    
    Validates: Requirements 4.5
    """
    # 计算 |ν/νz - λν|
    # 注意：nu_z 是复数，需要处理
    term = np.abs(nu / nu_z - wavelength * nu)
    
    # 计算阈值 L/(2z)
    threshold = L / (2 * z)
    
    # 返回窗函数：满足条件时为 1，否则为 0
    return np.where(term <= threshold, 1.0, 0.0)


def distance_limit(wavelength: float, L: float, N: int) -> float:
    """
    计算渐晕距离限制
    
    当传播距离超过此限制时，传播场可能受到渐晕影响。
    
    数学公式：
        R = L/(N×λ)
        DistanceLimit = L / (2 × |1/(4R) - 1/√(16R² + 2)|)
    
    Julia 实现：
        R = L/N/λ
        return L*inv(2*abs(inv(4R) - inv(√(16R^2 + 2))))
    
    参数：
        wavelength: 波长 λ
        L: 计算域尺寸
        N: 采样点数
    
    返回：
        渐晕距离限制
    
    示例：
        >>> distance_limit(633e-9, 1e-3, 64)
        # 返回渐晕距离限制值
    
    Validates: Requirements 4.3
    """
    # R = L/(N×λ)
    R = L / (N * wavelength)
    
    # 计算 1/(4R) - 1/√(16R² + 2)
    term1 = 1.0 / (4 * R)
    term2 = 1.0 / np.sqrt(16 * R**2 + 2)
    diff = term1 - term2
    
    # DistanceLimit = L / (2 × |diff|)
    return L / (2 * np.abs(diff))


def minimum_distance(wavelength: float, L: float, N: int) -> float:
    """
    计算最小传播距离
    
    当传播距离小于此值时，放大倍数可能小于 1。
    
    数学公式：
        Δ = L/N
        MinimumDistance = L × Δ / λ = L² / (N × λ)
    
    Julia 实现：
        Δ = L/N
        return L*Δ/λ
    
    参数：
        wavelength: 波长 λ
        L: 计算域尺寸
        N: 采样点数
    
    返回：
        最小传播距离
    
    示例：
        >>> minimum_distance(633e-9, 1e-3, 64)
        # 返回最小传播距离值
    
    Validates: Requirements 4.4
    """
    # Δ = L/N
    delta = L / N
    
    # MinimumDistance = L × Δ / λ
    return L * delta / wavelength


def scalable_asm(
    u: ComplexArray,
    wavelength: float,
    dx: float,
    dy: float,
    z: float,
    *,
    expand: bool = True
) -> ComplexArray:
    """
    可缩放角谱法（Scalable ASM）
    
    自动缩放的角谱传播，目标平面采样间距：
        Δd = λz/(pNΔs)
    其中 p=2 为填充因子
    
    包含 Fresnel 核相位校正。
    
    数学原理：
        1. 计算传递函数：
           H = exp(2πiz(νz - (1/λ - λν²/2)))
           其中 νz = √(1/λ² - ν²)
           
        2. 计算带限窗函数：
           W = BandLimit(λ, z, Ly, νy, νz) × BandLimit(λ, z, Lx, νx, νz)
           
        3. 计算 Fresnel 核：
           Q₁ = exp(πi/(λz)(r₁y² + r₁x²))  （源平面）
           Q₂ = exp(2πiz/λ)/(iλz) × exp(πi/(λz)(r₂y² + r₂x²))  （目标平面）
           
        4. 传播计算：
           û = IFFT{ FFT{ifftshift(ũ)} × H × W }
           result = fftshift{ Q₂ × FFT{û × Q₁} }
    
    参数：
        u: 输入光场，形状为 (Ny, Nx)，必须是复数类型
        wavelength: 波长 λ（单位与 dx, dy, z 一致）
        dx: x 方向采样间隔
        dy: y 方向采样间隔
        z: 传播距离（正值表示向前传播）
        expand: 是否进行 4 倍零填充扩展（默认 True）
            - True: 将数组扩展到 (2Ny, 2Nx) 以抑制混叠
            - False: 使用原始尺寸进行计算
    
    返回：
        传播后的光场，形状与输入相同 (Ny, Nx)
        注意：目标平面的采样间距已改变为 Δd = λz/(pNΔs)
    
    警告：
        - 传播距离超过渐晕限制时发出警告
        - 传播距离小于最小距离时发出警告
    
    示例：
        >>> import numpy as np
        >>> from angular_spectrum_method import scalable_asm
        >>> 
        >>> # 创建高斯光束
        >>> x = np.linspace(-1e-3, 1e-3, 64)
        >>> y = np.linspace(-1e-3, 1e-3, 64)
        >>> X, Y = np.meshgrid(x, y)
        >>> u = np.exp(-(X**2 + Y**2) / (0.5e-3)**2).astype(complex)
        >>> 
        >>> # 使用可缩放角谱法传播
        >>> result = scalable_asm(u, 633e-9, x[1]-x[0], y[1]-y[0], 0.01)
        >>> print(result.shape)
        (64, 64)
    
    注意：
        - x 轴为水平方向（列方向），y 轴为垂直方向（行方向）
        - 输入数组的形状约定为 (Ny, Nx)，即 (行数, 列数)
        - 目标平面的采样间距与源平面不同
    
    参考文献：
        Heintzmann et al., Optica 10, 1407-1416 (2023)
    
    Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5
    """
    # 获取原始尺寸
    original_shape = u.shape
    ny_orig, nx_orig = original_shape
    
    # 确定工作尺寸
    # Julia: N = ifelse(expand, size(u).*2, size(u))
    if expand:
        work_shape = _get_expanded_size(original_shape)
    else:
        work_shape = original_shape
    
    work_ny, work_nx = work_shape
    
    # 计算计算域尺寸
    # Julia: L = N.*[Δy, Δx]
    # L[1] 对应 y 方向，L[2] 对应 x 方向
    L_y = work_ny * dy
    L_x = work_nx * dx
    
    # 检查传播距离并发出警告
    # Julia: z ≥ minimum(DistanceLimit.(λ, L, N)) && @warn "..."
    dist_limit_y = distance_limit(wavelength, L_y, work_ny)
    dist_limit_x = distance_limit(wavelength, L_x, work_nx)
    min_dist_limit = min(dist_limit_y, dist_limit_x)
    
    if z >= min_dist_limit:
        warnings.warn(
            "传播距离超过渐晕限制，传播场可能受到渐晕影响",
            PropagationWarning
        )
    
    # Julia: z ≤ maximum(MinimumDistance.(λ, L, N)) && @warn "..."
    min_dist_y = minimum_distance(wavelength, L_y, work_ny)
    min_dist_x = minimum_distance(wavelength, L_x, work_nx)
    max_min_dist = max(min_dist_y, min_dist_x)
    
    if z <= max_min_dist:
        warnings.warn(
            "传播距离小于最小距离，放大倍数可能小于 1",
            PropagationWarning
        )
    
    # 生成空间频率（DC 在角落）
    # Julia: ν = fftfreq.(N, inv.([Δy, Δx]))
    # ν[1] 对应 y 方向，ν[2] 对应 x 方向
    nu_y = spatial_frequencies(work_ny, dy)  # 形状 (work_ny,)
    nu_x = spatial_frequencies(work_nx, dx)  # 形状 (work_nx,)
    
    # 创建 2D 频率网格
    # Julia: ν² = @. ν[1]^2 + ν[2]'^2
    nu_y_2d = nu_y.reshape(-1, 1)  # (work_ny, 1) - 列向量
    nu_x_2d = nu_x.reshape(1, -1)  # (1, work_nx) - 行向量
    nu_sq = nu_y_2d**2 + nu_x_2d**2  # (work_ny, work_nx)
    
    # 计算 νz = √(1/λ² - ν² + 0im)
    # Julia: νz = @. √(1/λ^2 - ν² + 0im)
    inv_lambda_sq = 1.0 / (wavelength**2)
    nu_z = np.sqrt(inv_lambda_sq - nu_sq + 0j)  # 复数平方根
    
    # 计算传递函数
    # Julia: H = @. exp(2π*im*z*(νz - (1/λ - λ*ν²/2)))
    # H = exp(2πiz(νz - (1/λ - λν²/2)))
    inv_lambda = 1.0 / wavelength
    H = np.exp(2j * np.pi * z * (nu_z - (inv_lambda - wavelength * nu_sq / 2)))
    
    # 计算带限窗函数
    # Julia: W = @. BandLimit(λ, z, L[1], ν[1], νz)*BandLimit(λ, z, L[2], ν[2]', νz)
    # W = BandLimit(λ, z, Ly, νy, νz) × BandLimit(λ, z, Lx, νx, νz)
    W_y = band_limit(wavelength, z, L_y, nu_y_2d, nu_z)  # (work_ny, work_nx)
    W_x = band_limit(wavelength, z, L_x, nu_x_2d, nu_z)  # (work_ny, work_nx)
    W = W_y * W_x  # (work_ny, work_nx)
    
    # 将输入数组扩展到工作尺寸
    # Julia: ũ = select_region_view(u, new_size=N)
    u_work = select_region(u, work_shape, center=True, pad_value=0)
    
    # 第一阶段传播
    # Julia: û = ifft(fft(ifftshift(ũ)).*H.*W)
    u_shifted = np.fft.ifftshift(u_work)
    U = np.fft.fft2(u_shifted)
    U_propagated = U * H * W
    u_hat = np.fft.ifft2(U_propagated)
    
    # 计算源平面坐标
    # Julia: r₁ = fftfreq.(N, L)
    # r₁[1] 对应 y 方向，r₁[2] 对应 x 方向
    # fftfreq(N, L) 生成范围为 [-0.5, 0.5) 的归一化频率，乘以 L 得到坐标
    r1_y = np.fft.fftfreq(work_ny, 1.0 / L_y)  # 形状 (work_ny,)
    r1_x = np.fft.fftfreq(work_nx, 1.0 / L_x)  # 形状 (work_nx,)
    
    # 创建 2D 坐标网格
    r1_y_2d = r1_y.reshape(-1, 1)  # (work_ny, 1)
    r1_x_2d = r1_x.reshape(1, -1)  # (1, work_nx)
    
    # 计算目标平面坐标
    # Julia: r₂ = fftfreq.(N, λ*z*inv.([Δy, Δx]))
    # r₂[1] 对应 y 方向，r₂[2] 对应 x 方向
    # λ*z*inv.([Δy, Δx]) = [λz/Δy, λz/Δx]
    r2_y = np.fft.fftfreq(work_ny, 1.0 / (wavelength * z / dy))  # 形状 (work_ny,)
    r2_x = np.fft.fftfreq(work_nx, 1.0 / (wavelength * z / dx))  # 形状 (work_nx,)
    
    # 创建 2D 坐标网格
    r2_y_2d = r2_y.reshape(-1, 1)  # (work_ny, 1)
    r2_x_2d = r2_x.reshape(1, -1)  # (1, work_nx)
    
    # 计算 Fresnel 核 Q₁（源平面）
    # Julia: Q₁ = @. exp(π*im/(λ*z)*(r₁[1]^2 + r₁[2]'^2))
    Q1 = np.exp(1j * np.pi / (wavelength * z) * (r1_y_2d**2 + r1_x_2d**2))
    
    # 计算 Fresnel 核 Q₂（目标平面）
    # Julia: Q₂ = @. exp(2π*im*z/λ)/(im*λ*z)*exp(π*im/(λ*z)*(r₂[1]^2 + r₂[2]'^2))
    # Q₂ = exp(2πiz/λ) / (iλz) × exp(πi/(λz)(r₂y² + r₂x²))
    prefactor = np.exp(2j * np.pi * z / wavelength) / (1j * wavelength * z)
    Q2 = prefactor * np.exp(1j * np.pi / (wavelength * z) * (r2_y_2d**2 + r2_x_2d**2))
    
    # 第二阶段传播：应用 Fresnel 核
    # Julia: û = fftshift(Q₂.*fft(û.*Q₁))
    u_hat_Q1 = u_hat * Q1
    U_hat = np.fft.fft2(u_hat_Q1)
    u_result = np.fft.fftshift(Q2 * U_hat)
    
    # 裁剪回原始尺寸
    # Julia: return select_region_view(û, new_size=size(u))
    result = select_region(u_result, original_shape, center=True)
    
    return result


def scalable_asm_(
    u: ComplexArray,
    wavelength: float,
    dx: float,
    dy: float,
    z: float,
    *,
    expand: bool = True
) -> None:
    """
    就地版本的 ScalableASM，直接修改输入数组
    
    此函数与 scalable_asm() 功能相同，但直接修改输入数组而不创建新数组。
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
    
    警告：
        - 传播距离超过渐晕限制时发出警告
        - 传播距离小于最小距离时发出警告
    
    示例：
        >>> import numpy as np
        >>> from angular_spectrum_method import scalable_asm_
        >>> 
        >>> # 创建光场
        >>> u = np.ones((64, 64), dtype=complex)
        >>> 
        >>> # 就地传播
        >>> scalable_asm_(u, 633e-9, 1e-6, 1e-6, 0.01)
        >>> # u 现在包含传播后的光场
    
    注意：
        - 输入数组必须是复数类型（如 np.complex64 或 np.complex128）
        - 如果输入数组不是复数类型，行为未定义
        - 虽然名为"就地"操作，但在 expand=True 时仍需要临时内存
    
    Validates: Requirements 8.3
    """
    # 计算传播结果
    result = scalable_asm(u, wavelength, dx, dy, z, expand=expand)
    
    # 将结果复制回输入数组
    u[:] = result
