# -*- coding: utf-8 -*-
"""
倾斜角谱法 (TiltedASM) 模块

本模块实现倾斜角谱法，计算倾斜平面上的衍射场。

算法步骤：
    1. 计算源平面空间频率 (νx, νy, νz)
    2. 通过旋转矩阵变换: ν̂ = T·ν - ν̂₀
    3. 计算载波频率: ν̂₀ = T·[0, 0, 1/λ]ᵀ
    4. 使用 NFFT 处理非均匀采样
    5. 叠加载波相位

参考文献：
    1. Matsushima et al., J. Opt. Soc. Am. A 20, 1755-1762 (2003)
    2. Matsushima, Appl. Opt. 47, D110-D116 (2008)
    3. Pipe & Menon, Magn. Reson. Med. 41, 179-186 (1999)

主要函数：
- tilted_asm: 倾斜角谱法传播计算
- tilted_asm_: 就地版本
- compute_jacobian: 计算雅可比行列式

Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 7.10, 8.6
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple

try:
    import finufft
    HAS_FINUFFT = True
except ImportError:
    HAS_FINUFFT = False

from .core import spatial_frequencies
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


def compute_sdc(k_x: np.ndarray, k_y: np.ndarray, n_modes: Tuple[int, int], 
                iters: int = 10) -> np.ndarray:
    """
    计算采样密度补偿（SDC）权重
    
    使用迭代方法计算 NFFT 的采样密度补偿权重，用于改善能量守恒。
    
    算法参考：
        Pipe & Menon, Magn. Reson. Med. 41, 179-186 (1999)
    
    参数：
        k_x: x 方向频率节点，形状为 (M,)
        k_y: y 方向频率节点，形状为 (M,)
        n_modes: 输出模式数 (N1, N2)
        iters: 迭代次数（默认 10）
    
    返回：
        SDC 权重，形状为 (M,)
    
    Validates: Requirements 7.8
    """
    M = len(k_x)
    w = np.ones(M, dtype=np.float64)
    
    for _ in range(iters):
        # 计算 P^H * P * w，其中 P 是 NFFT 矩阵
        # 使用 NFFT 伴随算子和 NFFT 来近似
        
        # 步骤 1: NFFT 伴随（非均匀到均匀）
        # nufft2d1: 非均匀到均匀
        temp = finufft.nufft2d1(
            k_x, k_y,
            w.astype(np.complex128),
            n_modes=n_modes,
            isign=1,
            eps=1e-12
        )
        
        # 步骤 2: NFFT（均匀到非均匀）
        # nufft2d2: 均匀到非均匀
        Pw = finufft.nufft2d2(
            k_x, k_y,
            temp,
            isign=-1,
            eps=1e-12
        )
        
        # 更新权重
        # w = w / |P^H * P * w|
        Pw_abs = np.abs(Pw)
        # 避免除零
        Pw_abs = np.maximum(Pw_abs, 1e-15)
        w = w / Pw_abs
    
    return w


def compute_source_frequencies(
    N: Tuple[int, int],
    dx: float,
    dy: float,
    wavelength: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算源平面空间频率 (νx, νy, νz)
    
    数学公式：
        νx, νy: 由 fftfreq 生成的空间频率（DC 在角落）
        νz² = 1/λ² - νx² - νy²
        νz = √(νz²)  当 νz² > 0
    
    参数：
        N: 工作数组尺寸 (Nx, Ny)，注意是转置后的尺寸
        dx: x 方向采样间隔
        dy: y 方向采样间隔
        wavelength: 波长 λ
    
    返回：
        (nu_x, nu_y, nu_z_sq): 空间频率数组
        - nu_x: x 方向空间频率，形状为 (Nx,)
        - nu_y: y 方向空间频率，形状为 (Ny,)
        - nu_z_sq: νz² 的 2D 数组，形状为 (Nx, Ny)
    
    Validates: Requirements 7.2
    """
    Nx, Ny = N
    
    # 生成空间频率（DC 在角落）
    # Julia: ν = fftfreq.(N, inv.([Δx, Δy]))
    # 注意：Julia 中 N 的顺序是 (Nx, Ny)，对应 inv.([Δx, Δy]) = [1/Δx, 1/Δy]
    nu_x = np.fft.fftfreq(Nx, dx)  # 形状 (Nx,)
    nu_y = np.fft.fftfreq(Ny, dy)  # 形状 (Ny,)
    
    # 计算 νz² = 1/λ² - νx² - νy²
    # Julia: νz² = @. 1/λ^2 - ν[1]^2 - ν[2]'^2
    # ν[1]^2 是列向量 (Nx,)，ν[2]'^2 是行向量 (1, Ny)
    # 广播后得到 (Nx, Ny) 的矩阵
    inv_lambda_sq = 1.0 / (wavelength ** 2)
    nu_x_2d = nu_x.reshape(-1, 1)  # (Nx, 1)
    nu_y_2d = nu_y.reshape(1, -1)  # (1, Ny)
    nu_z_sq = inv_lambda_sq - nu_x_2d**2 - nu_y_2d**2  # (Nx, Ny)
    
    return nu_x, nu_y, nu_z_sq


def compute_carrier_frequency(
    T: np.ndarray,
    wavelength: float
) -> np.ndarray:
    """
    计算载波频率 ν̂₀ = T·[0, 0, 1/λ]ᵀ
    
    载波频率是旋转矩阵作用于光轴方向单位向量（乘以 1/λ）的结果。
    
    参数：
        T: 3×3 旋转矩阵
        wavelength: 波长 λ
    
    返回：
        载波频率向量 ν̂₀，形状为 (3,)
    
    Validates: Requirements 7.4
    """
    # Julia: ν̂₀ = T*[0, 0, 1/λ]
    carrier_input = np.array([0.0, 0.0, 1.0 / wavelength])
    nu_hat_0 = T @ carrier_input
    return nu_hat_0


def find_valid_indices(nu_z_sq: np.ndarray) -> np.ndarray:
    """
    筛选有效索引（νz² > 0 的位置）
    
    只有当 νz² > 0 时，空间频率才对应传播波（非倏逝波）。
    
    参数：
        nu_z_sq: νz² 的 2D 数组，形状为 (Nx, Ny)
    
    返回：
        有效索引的一维数组（列优先展平后的索引，与 Julia 一致）
    
    Validates: Requirements 7.2
    """
    # Julia: i = findall(==(true), (@view νz²[:]) .> 0)
    # Julia 使用列优先展平，所以我们需要用 Fortran 顺序
    valid_mask = nu_z_sq.flatten(order='F') > 0
    valid_indices = np.where(valid_mask)[0]
    return valid_indices


def extract_source_frequencies_at_indices(
    nu_x: np.ndarray,
    nu_y: np.ndarray,
    nu_z_sq: np.ndarray,
    valid_indices: np.ndarray,
    N: Tuple[int, int]
) -> np.ndarray:
    """
    提取有效索引处的源平面空间频率 (νx, νy, νz)
    
    根据有效索引从空间频率数组中提取对应的值，构建 3×M 的频率矩阵。
    
    参数：
        nu_x: x 方向空间频率，形状为 (Nx,)
        nu_y: y 方向空间频率，形状为 (Ny,)
        nu_z_sq: νz² 的 2D 数组，形状为 (Nx, Ny)
        valid_indices: 有效索引数组（列优先展平后的索引）
        N: 工作数组尺寸 (Nx, Ny)
    
    返回：
        源平面空间频率矩阵，形状为 (3, M)，其中 M 是有效索引数量
    
    Validates: Requirements 7.2
    """
    Nx, Ny = N
    M = len(valid_indices)
    
    # Julia 索引转换（列优先）：
    # Julia: ν̃ = @. [(@view ν[1][(i-1)%N[1]+1])';
    #               (@view ν[2][(i-1)÷N[1]+1])';
    #               √(@view νz²[i])']
    # 
    # Julia 使用 1-based 索引，列优先存储
    # 对于列优先存储，线性索引 i 对应：
    # x = i % Nx (0-based)
    # y = i // Nx (0-based)
    
    # 计算 x 和 y 索引（列优先）
    x_indices = valid_indices % Nx   # x 索引
    y_indices = valid_indices // Nx  # y 索引
    
    # 提取对应的空间频率
    nu_x_valid = nu_x[x_indices]  # 形状 (M,)
    nu_y_valid = nu_y[y_indices]  # 形状 (M,)
    nu_z_sq_valid = nu_z_sq.flatten(order='F')[valid_indices]  # 形状 (M,)
    nu_z_valid = np.sqrt(nu_z_sq_valid)  # 形状 (M,)
    
    # 构建 3×M 的频率矩阵
    # Julia: ν̃ 是 3×M 矩阵，每列是一个 (νx, νy, νz) 向量
    nu_tilde = np.vstack([nu_x_valid, nu_y_valid, nu_z_valid])  # (3, M)
    
    return nu_tilde


def transform_frequencies(
    nu_tilde: np.ndarray,
    T: np.ndarray,
    nu_hat_0: np.ndarray
) -> np.ndarray:
    """
    通过旋转矩阵变换空间频率 ν̂ = T·ν - ν̂₀
    
    将源平面的空间频率通过旋转矩阵变换到参考平面。
    
    参数：
        nu_tilde: 源平面空间频率矩阵，形状为 (3, M)
        T: 3×3 旋转矩阵
        nu_hat_0: 载波频率向量，形状为 (3,)
    
    返回：
        变换后的空间频率矩阵 ν̂，形状为 (3, M)
    
    Validates: Requirements 7.3
    """
    # Julia: ν̂ = T*ν̃ .- ν̂₀
    # T 是 3×3 矩阵，ν̃ 是 3×M 矩阵
    # T*ν̃ 得到 3×M 矩阵
    # 减去 ν̂₀（3×1 向量，广播到每一列）
    nu_hat = T @ nu_tilde - nu_hat_0.reshape(-1, 1)
    return nu_hat


def compute_frequency_nodes(
    nu_hat: np.ndarray,
    dx: float,
    dy: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 NFFT 频率节点并处理周期性边界
    
    将变换后的空间频率转换为 NFFT 节点，并应用周期性边界条件 [-1/2, 1/2)。
    
    参数：
        nu_hat: 变换后的空间频率矩阵，形状为 (3, M)
        dx: x 方向采样间隔
        dy: y 方向采样间隔
    
    返回：
        (k_x, k_y): NFFT 节点坐标
        - k_x: x 方向节点，形状为 (M,)，范围 [-π, π)
        - k_y: y 方向节点，形状为 (M,)，范围 [-π, π)
    
    Validates: Requirements 7.5, 7.6
    """
    # Julia: k̂ = (@view ν̂[1:2,:]).*Δ
    # 提取前两行（νx 和 νy），乘以采样间隔
    # 注意：Julia 中 Δ = [Δx, Δy]
    k_hat = np.vstack([nu_hat[0, :] * dx, nu_hat[1, :] * dy])  # (2, M)
    
    # Julia: k̂ = @. k̂ - floor(k̂ + 1/2)
    # 应用周期性边界条件，将节点映射到 [-1/2, 1/2)
    k_hat = k_hat - np.floor(k_hat + 0.5)
    
    # 转换为 finufft 的节点范围 [-π, π)
    k_x = k_hat[0, :] * 2 * np.pi  # 形状 (M,)
    k_y = k_hat[1, :] * 2 * np.pi  # 形状 (M,)
    
    return k_x, k_y


def compute_jacobian_factor(
    T: np.ndarray,
    nu_hat: np.ndarray,
    wavelength: float,
    M: int
) -> np.ndarray:
    """
    计算雅可比行列式因子用于能量校正
    
    雅可比因子用于补偿坐标变换导致的能量变化。
    
    数学公式：
        j = [T[1,0]*T[2,1] - T[2,0]*T[1,1],
             T[2,0]*T[0,1] - T[0,0]*T[2,1],
             T[0,0]*T[1,1] - T[1,0]*T[0,1]]
        
        ν̂z = √(1/λ² - ν̂x² - ν̂y²)
        
        J = √([ν̂x/ν̂z, ν̂y/ν̂z, 1] · j)
    
    参数：
        T: 3×3 旋转矩阵
        nu_hat: 变换后的空间频率矩阵，形状为 (3, M)
        wavelength: 波长 λ
        M: 有效点数量
    
    返回：
        雅可比因子数组，形状为 (M,)
    
    Validates: Requirements 7.7
    """
    # 计算变换后的 νz
    # Julia: ν̂[3,:] .= @. √(1/λ^2 - ν̂[1,:]^2 - ν̂[2,:]^2)
    inv_lambda_sq = 1.0 / (wavelength ** 2)
    nu_hat_z_sq = inv_lambda_sq - nu_hat[0, :]**2 - nu_hat[1, :]**2
    # 处理可能的负值（数值误差）
    nu_hat_z_sq = np.maximum(nu_hat_z_sq, 0)
    nu_hat_z = np.sqrt(nu_hat_z_sq)
    
    # 计算 j 向量
    # Julia: j = [T[4]*T[8] - T[7]*T[5],
    #             T[7]*T[2] - T[1]*T[8],
    #             T[1]*T[5] - T[4]*T[2]]
    # Julia 使用 1-based 线性索引（列优先）：
    # T[1]=T[1,1], T[2]=T[2,1], T[4]=T[1,2], T[5]=T[2,2], T[7]=T[1,3], T[8]=T[2,3]
    # 转换为 Python 0-based 索引：
    # T[1,1]->T[0,0], T[2,1]->T[1,0], T[1,2]->T[0,1], T[2,2]->T[1,1], T[1,3]->T[0,2], T[2,3]->T[1,2]
    # 
    # Julia 列优先线性索引：
    # T[1] = T[0,0], T[2] = T[1,0], T[3] = T[2,0]
    # T[4] = T[0,1], T[5] = T[1,1], T[6] = T[2,1]
    # T[7] = T[0,2], T[8] = T[1,2], T[9] = T[2,2]
    #
    # 所以：
    # j[0] = T[4]*T[8] - T[7]*T[5] = T[0,1]*T[1,2] - T[0,2]*T[1,1]
    # j[1] = T[7]*T[2] - T[1]*T[8] = T[0,2]*T[1,0] - T[0,0]*T[1,2]
    # j[2] = T[1]*T[5] - T[4]*T[2] = T[0,0]*T[1,1] - T[0,1]*T[1,0]
    j = np.array([
        T[0, 1] * T[1, 2] - T[0, 2] * T[1, 1],
        T[0, 2] * T[1, 0] - T[0, 0] * T[1, 2],
        T[0, 0] * T[1, 1] - T[0, 1] * T[1, 0]
    ])
    
    # 计算雅可比因子
    # Julia: J = .√([ν̂[1,:]./ν̂[3,:] ν̂[2,:]./ν̂[3,:] ones(eltype(ν̂), length(f̂), 1)]*j)
    # 构建 M×3 矩阵，每行是 [ν̂x/ν̂z, ν̂y/ν̂z, 1]
    # 避免除零
    nu_hat_z_safe = np.where(nu_hat_z > 1e-15, nu_hat_z, 1e-15)
    
    ratio_matrix = np.column_stack([
        nu_hat[0, :] / nu_hat_z_safe,  # ν̂x/ν̂z
        nu_hat[1, :] / nu_hat_z_safe,  # ν̂y/ν̂z
        np.ones(M)                      # 1
    ])  # (M, 3)
    
    # 计算点积并取平方根
    # Julia: J = .√(matrix * j)
    dot_product = ratio_matrix @ j  # (M,)
    # 处理可能的负值（数值误差）
    dot_product = np.maximum(dot_product, 0)
    J = np.sqrt(dot_product)
    
    return J





def tilted_asm(
    u: ComplexArray,
    wavelength: float,
    dx: float,
    dy: float,
    T: np.ndarray,
    *,
    expand: bool = True,
    weight: bool = False
) -> ComplexArray:
    """
    倾斜角谱法（Tilted ASM）
    
    计算倾斜平面上的衍射场，T 为 3×3 旋转矩阵。
    
    算法步骤：
        1. 计算源平面空间频率 (νx, νy, νz)
        2. 通过旋转矩阵变换: ν̂ = T·ν - ν̂₀
        3. 计算载波频率: ν̂₀ = T·[0, 0, 1/λ]ᵀ
        4. 使用 NFFT 处理非均匀采样
        5. 叠加载波相位
    
    数学原理：
        源平面空间频率：
            νz = √(1/λ² - νx² - νy²)
        
        旋转变换：
            ν̂ = T·[νx, νy, νz]ᵀ - ν̂₀
            其中 ν̂₀ = T·[0, 0, 1/λ]ᵀ
        
        载波相位叠加：
            u_out = u_nfft × exp(2πi(ν̂₀y·y + ν̂₀x·x))
    
    参数：
        u: 输入光场，形状为 (Ny, Nx)，必须是复数类型
        wavelength: 波长 λ（单位与 dx, dy 一致）
        dx: x 方向采样间隔
        dy: y 方向采样间隔
        T: 3×3 旋转矩阵，描述倾斜平面相对于源平面的旋转
        expand: 是否进行 4 倍零填充扩展（默认 True）
            - True: 将数组扩展到 (2Ny, 2Nx) 以抑制混叠
            - False: 使用原始尺寸进行计算
        weight: 是否使用 SDC 权重改善能量守恒（默认 False）
            - False: 使用雅可比行列式进行能量校正
            - True: 使用采样密度补偿（SDC）权重，计算成本较高
    
    返回：
        倾斜平面上的衍射场，形状与输入相同 (Ny, Nx)
    
    异常：
        ImportError: 如果 finufft 库未安装
    
    示例：
        >>> import numpy as np
        >>> from scipy.spatial.transform import Rotation
        >>> from angular_spectrum_method import tilted_asm
        >>> 
        >>> # 创建高斯光束
        >>> x = np.linspace(-1e-3, 1e-3, 64)
        >>> y = np.linspace(-1e-3, 1e-3, 64)
        >>> X, Y = np.meshgrid(x, y)
        >>> u = np.exp(-(X**2 + Y**2) / (0.5e-3)**2).astype(complex)
        >>> 
        >>> # 创建旋转矩阵（绕 x 轴旋转 5 度）
        >>> T = Rotation.from_euler('x', 5, degrees=True).as_matrix()
        >>> 
        >>> # 计算倾斜平面上的衍射场
        >>> result = tilted_asm(u, 633e-9, x[1]-x[0], y[1]-y[0], T)
        >>> print(result.shape)
        (64, 64)
    
    注意：
        - 使用 scipy.spatial.transform.Rotation 生成旋转矩阵
        - 数组会被转置以保持与旋转矩阵的一致性
        - 需要安装 finufft 库：pip install finufft
    
    参考文献：
        1. Matsushima et al., J. Opt. Soc. Am. A 20, 1755-1762 (2003)
        2. Matsushima, Appl. Opt. 47, D110-D116 (2008)
        3. Pipe & Menon, Magn. Reson. Med. 41, 179-186 (1999)
    
    Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 7.10
    """
    # 检查 finufft 是否可用
    if not HAS_FINUFFT:
        raise ImportError(
            "finufft 库未安装。请使用 'pip install finufft' 安装。"
        )
    
    # 获取原始尺寸
    original_shape = u.shape
    ny_orig, nx_orig = original_shape
    
    # Julia: N = ifelse(expand, size(u').*2, size(u'))
    # 注意：Julia 中数组被转置，所以 size(u') = (Nx, Ny)
    # 在 Python 中，u.T 的形状是 (Nx, Ny)
    if expand:
        # 转置后的扩展尺寸
        N = (2 * nx_orig, 2 * ny_orig)  # (Nx*2, Ny*2)
    else:
        N = (nx_orig, ny_orig)  # (Nx, Ny)
    
    Nx, Ny = N
    
    # 计算载波频率
    # Julia: ν̂₀ = T*[0, 0, 1/λ]
    nu_hat_0 = compute_carrier_frequency(T, wavelength)
    
    # 计算源平面空间频率
    # Julia: ν = fftfreq.(N, inv.([Δx, Δy]))
    # 注意：Julia 中 Δ = [Δx, Δy]，对应转置后的数组
    nu_x, nu_y, nu_z_sq = compute_source_frequencies(N, dx, dy, wavelength)
    
    # 准备输入数组（转置并扩展）
    # Julia: ũ = select_region(transpose(u), new_size=N)
    u_transposed = u.T  # (Nx_orig, Ny_orig)
    u_work = select_region(u_transposed, N, center=True, pad_value=0).astype(np.complex128)
    
    # FFT 变换
    # Julia: û = fft(ifftshift(ũ))
    u_shifted = np.fft.ifftshift(u_work)
    U = np.fft.fft2(u_shifted)
    
    # 找到有效索引（νz² > 0）
    # Julia: i = findall(==(true), (@view νz²[:]) .> 0)
    valid_indices = find_valid_indices(nu_z_sq)
    M = len(valid_indices)
    
    # 提取有效频谱数据
    # Julia: f̂ = @view û[i]
    # 注意：Julia 使用列优先展平，所以我们也需要用列优先
    f_hat = U.flatten(order='F')[valid_indices]  # (M,)
    
    # 提取有效索引处的源平面空间频率
    nu_tilde = extract_source_frequencies_at_indices(
        nu_x, nu_y, nu_z_sq, valid_indices, N
    )  # (3, M)
    
    # 旋转变换
    # Julia: ν̂ = T*ν̃ .- ν̂₀
    nu_hat = transform_frequencies(nu_tilde, T, nu_hat_0)  # (3, M)
    
    # 计算 NFFT 频率节点
    k_x, k_y = compute_frequency_nodes(nu_hat, dx, dy)
    
    # 使用 NFFT 伴随算子进行非均匀到均匀变换
    if weight:
        # 使用 SDC 权重
        # Julia: û = adjoint(p)*(f̂.*sqrt.(sdc(p, iters=10)./length(f̂)))
        sdc_weights = compute_sdc(k_x, k_y, (Nx, Ny), iters=10)
        weighted_f_hat = f_hat * np.sqrt(sdc_weights / M)
        
        # NFFT 伴随（非均匀到均匀）
        # finufft.nufft2d1: 非均匀到均匀
        # Julia NFFT 的 adjoint 对应 finufft 的 nufft2d1 with isign=1
        U_result = finufft.nufft2d1(
            k_x, k_y,
            weighted_f_hat,
            n_modes=(Nx, Ny),
            isign=1,
            eps=1e-12
        )
    else:
        # 使用雅可比行列式
        # Julia: 
        # ν̂[3,:] .= @. √(1/λ^2 - ν̂[1,:]^2 - ν̂[2,:]^2)
        # j = [T[4]*T[8] - T[7]*T[5], ...]
        # J = .√([ν̂[1,:]./ν̂[3,:] ν̂[2,:]./ν̂[3,:] ones(...)]*j)
        # û = adjoint(p)*(f̂.*J./length(f̂))
        
        J = compute_jacobian_factor(T, nu_hat, wavelength, M)
        weighted_f_hat = f_hat * J / M
        
        # NFFT 伴随（非均匀到均匀）
        U_result = finufft.nufft2d1(
            k_x, k_y,
            weighted_f_hat,
            n_modes=(Nx, Ny),
            isign=1,
            eps=1e-12
        )
    
    # 裁剪并转置回原始形状
    # Julia: f = select_region(transpose(û), new_size=size(u))
    # U_result 的形状是 (Nx, Ny)，转置后是 (Ny, Nx)
    f = select_region(U_result.T, original_shape, center=True)
    
    # 叠加载波相位
    # Julia: r = fftshift.(fftfreq.(size(u), size(u).*[Δy, -Δx]))
    # Julia: return @. f*exp(2π*im*(ν̂₀[2]*r[1] + ν̂₀[1]*r[2]'))
    #
    # 注意：Julia 中 size(u) = (Ny, Nx)
    # r[1] 对应 y 方向，r[2] 对应 x 方向
    #
    # 重要：Julia 的 fftfreq(n, fs) 中 fs 是采样率（samples per second）
    # Python 的 fftfreq(n, d) 中 d 是采样间隔（sample spacing）
    # Julia fftfreq(n, fs) = Python fftfreq(n, 1/fs)
    #
    # Julia: fftfreq(Ny, Ny*Δy) 
    # Python: fftfreq(Ny, 1/(Ny*Δy))
    
    # 生成空间坐标
    # Julia: r = fftshift.(fftfreq.(size(u), size(u).*[Δy, -Δx]))
    # 转换为 Python：
    r_y = np.fft.fftshift(np.fft.fftfreq(ny_orig, 1.0 / (ny_orig * dy)))  # 形状 (Ny,)
    r_x = np.fft.fftshift(np.fft.fftfreq(nx_orig, 1.0 / (nx_orig * (-dx))))  # 形状 (Nx,)
    
    # 创建 2D 网格
    r_y_2d = r_y.reshape(-1, 1)  # (Ny, 1)
    r_x_2d = r_x.reshape(1, -1)  # (1, Nx)
    
    # 计算载波相位
    # Julia: exp(2π*im*(ν̂₀[2]*r[1] + ν̂₀[1]*r[2]'))
    # ν̂₀[2] 是 y 分量，ν̂₀[1] 是 x 分量（Julia 1-based 索引）
    # Python 中 nu_hat_0[1] 是 y 分量，nu_hat_0[0] 是 x 分量
    carrier_phase = np.exp(2j * np.pi * (nu_hat_0[1] * r_y_2d + nu_hat_0[0] * r_x_2d))
    
    # 叠加载波
    result = f * carrier_phase
    
    return result


def tilted_asm_(
    u: ComplexArray,
    wavelength: float,
    dx: float,
    dy: float,
    T: np.ndarray,
    *,
    expand: bool = True,
    weight: bool = False
) -> None:
    """
    就地版本的 TiltedASM，直接修改输入数组
    
    此函数与 tilted_asm() 功能相同，但直接修改输入数组而不创建新数组。
    适用于内存受限的场景。
    
    参数：
        u: 输入/输出光场，形状为 (Ny, Nx)，必须是复数类型
           函数执行后，此数组将包含倾斜平面上的衍射场
        wavelength: 波长 λ
        dx: x 方向采样间隔
        dy: y 方向采样间隔
        T: 3×3 旋转矩阵
        expand: 是否进行 4 倍零填充扩展（默认 True）
        weight: 是否使用 SDC 权重改善能量守恒（默认 False）
    
    返回：
        None（结果直接写入输入数组 u）
    
    异常：
        ImportError: 如果 finufft 库未安装
    
    示例：
        >>> import numpy as np
        >>> from scipy.spatial.transform import Rotation
        >>> from angular_spectrum_method import tilted_asm_
        >>> 
        >>> # 创建光场
        >>> u = np.ones((64, 64), dtype=complex)
        >>> 
        >>> # 创建旋转矩阵
        >>> T = Rotation.from_euler('x', 5, degrees=True).as_matrix()
        >>> 
        >>> # 就地计算
        >>> tilted_asm_(u, 633e-9, 1e-6, 1e-6, T)
        >>> # u 现在包含倾斜平面上的衍射场
    
    注意：
        - 输入数组必须是复数类型（如 np.complex64 或 np.complex128）
        - 如果输入数组不是复数类型，行为未定义
        - 虽然名为"就地"操作，但在 expand=True 时仍需要临时内存
    
    Validates: Requirements 8.6
    """
    # 计算传播结果
    result = tilted_asm(u, wavelength, dx, dy, T, expand=expand, weight=weight)
    
    # 将结果复制回输入数组
    u[:] = result
