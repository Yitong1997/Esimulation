# -*- coding: utf-8 -*-
"""
缩放角谱法 (ScaledASM) 模块

本模块实现缩放角谱法，根据指定缩放因子进行传播，使用 NFFT 实现非均匀采样。

缩放角谱法的核心思想：
1. 根据缩放因子 R 选择缩小或放大模式
2. |R| ≤ 1 时使用 NFFT 进行缩小传播
3. |R| > 1 时使用 NFFT 伴随算子进行放大传播
4. 应用雅可比因子 J 保持能量守恒

主要特性：
- 支持任意缩放因子 R
- 使用 finufft 库进行高效 NFFT 计算
- 支持零填充扩展（expand=True）
- 提供就地操作版本以节省内存

参考文献：
    Shimobaba et al., Opt. Lett. 37, 4128-4130 (2012)

Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 8.4
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple

try:
    import finufft
    HAS_FINUFFT = True
except ImportError:
    HAS_FINUFFT = False

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



def scaled_asm(
    u: ComplexArray,
    wavelength: float,
    dx: float,
    dy: float,
    z: float,
    R: float,
    *,
    expand: bool = True
) -> ComplexArray:
    """
    缩放角谱法（Scaled ASM）
    
    根据缩放因子 R 进行缩放传播，使用 NFFT 实现非均匀采样。
    
    数学原理：
        1. 确定工作尺寸 N 和采样间隔 Δ：
           - |R| ≤ 1: Δ = [Δy, Δx]（保持原采样间隔）
           - |R| > 1: Δ = R × [Δy, Δx]（缩放采样间隔）
           
        2. 计算缩放因子 S：
           - |R| ≤ 1: S = R
           - |R| > 1: S = 1/R
           
        3. 计算 NFFT 节点：
           k = [f[1] × S, f[2] × S]
           其中 f 是 DC 居中的 DFT 采样频率
           
        4. 计算雅可比因子 J（能量守恒）：
           - |R| ≤ 1: J = |R| / (N[1] × N[2])
           - |R| > 1: J = 1 / |R|
           
        5. 计算传递函数：
           H = exp(2πiz√(1/λ² - (f[1]/Δ[1])² - (f[2]/Δ[2])²))
           
        6. 传播计算：
           - |R| ≤ 1（缩小）: 
             û = FFT{fftshift(ũ)}
             û = û × H × J
             û = NFFT(k, û)  # 均匀到非均匀
             
           - |R| > 1（放大）:
             û = NFFT_adjoint(k, ũ)  # 非均匀到均匀
             û = û × H × J
             û = IFFT{fftshift(û)}
    
    参数：
        u: 输入光场，形状为 (Ny, Nx)，必须是复数类型
        wavelength: 波长 λ（单位与 dx, dy, z 一致）
        dx: x 方向采样间隔
        dy: y 方向采样间隔
        z: 传播距离（正值表示向前传播）
        R: 缩放因子
            - |R| < 1: 缩小（目标平面尺寸变小）
            - |R| = 1: 无缩放（等效于普通 ASM）
            - |R| > 1: 放大（目标平面尺寸变大）
        expand: 是否进行 4 倍零填充扩展（默认 True）
            - True: 将数组扩展到 (2Ny, 2Nx) 以抑制混叠
            - False: 使用原始尺寸进行计算
    
    返回：
        传播后的光场，形状与输入相同 (Ny, Nx)
    
    异常：
        ImportError: 如果 finufft 库未安装
        ValueError: 如果 R = 0
    
    示例：
        >>> import numpy as np
        >>> from angular_spectrum_method import scaled_asm
        >>> 
        >>> # 创建高斯光束
        >>> x = np.linspace(-1e-3, 1e-3, 64)
        >>> y = np.linspace(-1e-3, 1e-3, 64)
        >>> X, Y = np.meshgrid(x, y)
        >>> u = np.exp(-(X**2 + Y**2) / (0.5e-3)**2).astype(complex)
        >>> 
        >>> # 使用缩放角谱法传播（缩小 0.5 倍）
        >>> result = scaled_asm(u, 633e-9, x[1]-x[0], y[1]-y[0], 0.01, R=0.5)
        >>> print(result.shape)
        (64, 64)
    
    注意：
        - x 轴为水平方向（列方向），y 轴为垂直方向（行方向）
        - 输入数组的形状约定为 (Ny, Nx)，即 (行数, 列数)
        - 需要安装 finufft 库：pip install finufft
    
    参考文献：
        Shimobaba et al., Opt. Lett. 37, 4128-4130 (2012)
    
    Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5
    """
    # 检查 finufft 是否可用
    if not HAS_FINUFFT:
        raise ImportError(
            "finufft 库未安装。请使用 'pip install finufft' 安装。"
        )
    
    # 检查 R 是否为零
    if R == 0:
        raise ValueError("缩放因子 R 不能为零")
    
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
    
    # 确定采样间隔
    # Julia: Δ = ifelse(abs(R) ≤ 1, [Δy, Δx], R.*[Δy, Δx])
    abs_R = abs(R)
    if abs_R <= 1:
        delta_y = dy
        delta_x = dx
    else:
        delta_y = R * dy
        delta_x = R * dx
    
    # 计算缩放因子 S
    # Julia: S = ifelse(abs(R) ≤ 1, R, 1/R)
    if abs_R <= 1:
        S = R
    else:
        S = 1.0 / R
    
    # 生成 DC 居中的 DFT 采样频率
    # Julia: f = @. fftshift(fftfreq(N))
    # fftfreq(N) 返回 [0, 1/N, 2/N, ..., (N/2-1)/N, -N/2/N, ..., -1/N]
    # fftshift 后变为 [-N/2/N, ..., -1/N, 0, 1/N, ..., (N/2-1)/N]
    f_y = np.fft.fftshift(np.fft.fftfreq(work_ny))  # 形状 (work_ny,)
    f_x = np.fft.fftshift(np.fft.fftfreq(work_nx))  # 形状 (work_nx,)
    
    # 计算 NFFT 节点
    # Julia: k = [repeat(f[1].*S, N[2])'; repeat(f[2].*S, inner=N[1])']
    # k 是 2×M 矩阵，M = N[1] × N[2]
    # k[1,:] = f[1] × S 重复 N[2] 次（每个 y 频率对应所有 x 位置）
    # k[2,:] = f[2] × S 每个元素重复 N[1] 次（每个 x 频率重复 N[1] 次）
    
    # finufft 的节点范围是 [-π, π)，而 Julia NFFT 的节点范围是 [-0.5, 0.5)
    # 所以需要乘以 2π
    k_y = np.tile(f_y * S, work_nx) * 2 * np.pi  # 形状 (work_ny * work_nx,)
    k_x = np.repeat(f_x * S, work_ny) * 2 * np.pi  # 形状 (work_ny * work_nx,)
    
    # 计算雅可比因子 J（能量守恒）
    # Julia: J = ifelse(abs(R) ≤ 1, abs(R)/(N[1]*N[2]), 1/abs(R))
    if abs_R <= 1:
        J = abs_R / (work_ny * work_nx)
    else:
        J = 1.0 / abs_R
    
    # 计算传递函数 H
    # Julia: H = @. exp(2π*im*z*√(1/λ^2 - (f[1]/Δ[1])^2 - (f[2]'/Δ[2])^2 + 0im))
    # 创建 2D 频率网格
    f_y_2d = f_y.reshape(-1, 1) / delta_y  # (work_ny, 1)
    f_x_2d = f_x.reshape(1, -1) / delta_x  # (1, work_nx)
    
    inv_lambda_sq = 1.0 / (wavelength ** 2)
    radicand = inv_lambda_sq - f_y_2d**2 - f_x_2d**2
    
    # 使用复数平方根处理倏逝波
    sqrt_term = np.sqrt(radicand.astype(np.complex128))
    H = np.exp(2j * np.pi * z * sqrt_term)
    
    # 将输入数组扩展到工作尺寸
    # Julia: û::Matrix{ComplexF64} = select_region_view(u, new_size=N)
    u_work = select_region(u, work_shape, center=True, pad_value=0).astype(np.complex128)
    
    if abs_R <= 1:
        # 缩小模式
        # Julia:
        # û = ifftshift(fft(fftshift(û)))
        # û = @. û*H*J
        # û = reshape(conj.(nfft(k, conj.(û))::Vector{ComplexF64}), N)
        
        # FFT 变换到频域（DC 居中）
        u_shifted = np.fft.fftshift(u_work)
        U = np.fft.fft2(u_shifted)
        U = np.fft.ifftshift(U)
        
        # 应用传递函数和雅可比因子
        U = U * H * J
        
        # 使用 NFFT 进行非均匀采样（均匀到非均匀）
        # finufft.nufft2d2: 均匀到非均匀
        # Julia 的 nfft 使用 conj 技巧来匹配符号约定
        # nfft(k, c) 计算 sum_j c_j exp(-2πi k·x_j)
        # finufft.nufft2d2 计算 sum_k f_k exp(isign * i * (k1*x + k2*y))
        # 默认 isign=-1，所以 finufft.nufft2d2 计算 sum_k f_k exp(-i * (k1*x + k2*y))
        
        # Julia NFFT 的约定：nfft(k, c) = sum_j c_j exp(-2πi k·j/N)
        # 其中 k 在 [-0.5, 0.5)，j 是数组索引
        # 
        # finufft 的约定：nufft2d2(x, y, f) = sum_{k1,k2} f_{k1,k2} exp(isign * i * (k1*x + k2*y))
        # 其中 x, y 在 [-π, π)，k1, k2 是模式索引
        #
        # 为了匹配 Julia 的行为，我们需要使用共轭技巧
        U_flat = U.flatten(order='C')  # 按行展平
        
        # 使用 nufft2d2（均匀到非均匀）
        # 注意：finufft 的输入数组形状是 (N1, N2)，输出是 (M,)
        # isign=1 表示 exp(+i*k*x)，isign=-1 表示 exp(-i*k*x)
        # Julia 使用 conj(nfft(k, conj(c)))，等效于改变符号
        u_hat_flat = finufft.nufft2d2(
            k_x, k_y,  # 非均匀点坐标
            np.conj(U),  # 输入的均匀数据（共轭）
            isign=-1,
            eps=1e-12
        )
        u_hat = np.conj(u_hat_flat).reshape(work_shape, order='C')
        
    else:
        # 放大模式
        # Julia:
        # û = conj.(nfft_adjoint(k, N, conj.(û[:]))::Matrix{ComplexF64})
        # û = @. û*H*J
        # û = ifftshift(ifft(fftshift(û)))
        
        # 使用 NFFT 伴随算子（非均匀到均匀）
        # finufft.nufft2d1: 非均匀到均匀
        u_flat = u_work.flatten(order='C')
        
        # 使用 nufft2d1（非均匀到均匀）
        # Julia 使用 conj(nfft_adjoint(k, N, conj(c)))
        U = finufft.nufft2d1(
            k_x, k_y,  # 非均匀点坐标
            np.conj(u_flat),  # 输入的非均匀数据（共轭）
            n_modes=(work_nx, work_ny),  # 输出尺寸 (N1, N2)
            isign=1,
            eps=1e-12
        )
        U = np.conj(U).T  # 转置以匹配 (work_ny, work_nx) 形状
        
        # 应用传递函数和雅可比因子
        U = U * H * J
        
        # IFFT 变换回空域
        U_shifted = np.fft.fftshift(U)
        u_hat = np.fft.ifft2(U_shifted)
        u_hat = np.fft.ifftshift(u_hat)
    
    # 裁剪回原始尺寸
    # Julia: return select_region_view(û, new_size=size(u))
    result = select_region(u_hat, original_shape, center=True)
    
    return result


def scaled_asm_(
    u: ComplexArray,
    wavelength: float,
    dx: float,
    dy: float,
    z: float,
    R: float,
    *,
    expand: bool = True
) -> None:
    """
    就地版本的 ScaledASM，直接修改输入数组
    
    此函数与 scaled_asm() 功能相同，但直接修改输入数组而不创建新数组。
    适用于内存受限的场景。
    
    参数：
        u: 输入/输出光场，形状为 (Ny, Nx)，必须是复数类型
           函数执行后，此数组将包含传播后的光场
        wavelength: 波长 λ
        dx: x 方向采样间隔
        dy: y 方向采样间隔
        z: 传播距离
        R: 缩放因子
        expand: 是否进行 4 倍零填充扩展（默认 True）
    
    返回：
        None（结果直接写入输入数组 u）
    
    异常：
        ImportError: 如果 finufft 库未安装
        ValueError: 如果 R = 0
    
    示例：
        >>> import numpy as np
        >>> from angular_spectrum_method import scaled_asm_
        >>> 
        >>> # 创建光场
        >>> u = np.ones((64, 64), dtype=complex)
        >>> 
        >>> # 就地传播（缩小 0.5 倍）
        >>> scaled_asm_(u, 633e-9, 1e-6, 1e-6, 0.01, R=0.5)
        >>> # u 现在包含传播后的光场
    
    注意：
        - 输入数组必须是复数类型（如 np.complex64 或 np.complex128）
        - 如果输入数组不是复数类型，行为未定义
        - 虽然名为"就地"操作，但在 expand=True 时仍需要临时内存
    
    Validates: Requirements 8.4
    """
    # 计算传播结果
    result = scaled_asm(u, wavelength, dx, dy, z, R, expand=expand)
    
    # 将结果复制回输入数组
    u[:] = result
