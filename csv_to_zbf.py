"""CSV 到 ZBF 文件转换器。

将 CSV 格式的光强（Irradiance）和相位（Phase）数据转换为 Zemax Beam File（ZBF）格式。
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes

# 配置 Matplotlib 以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from zbf_io import ZBFData, write_zbf, read_zbf


def read_beam_csv(filepath: str) -> np.ndarray:
    """读取 CSV 格式的光束数据文件。

    自动检测并跳过非数值表头行。

    参数：
        filepath: CSV 文件路径

    返回：
        二维 float64 numpy 数组

    异常：
        FileNotFoundError: 文件路径不存在
        ValueError: 文件内容无法解析为数值矩阵
    """
    # 检查文件是否存在
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{filepath}")

    # 尝试直接解析（无表头）
    try:
        data = np.loadtxt(filepath, delimiter=',')
        return data.astype(np.float64)
    except ValueError:
        pass

    # 尝试跳过第一行表头
    try:
        data = np.loadtxt(filepath, delimiter=',', skiprows=1)
        return data.astype(np.float64)
    except ValueError as e:
        raise ValueError(f"无法将文件解析为数值矩阵：{filepath}。原因：{e}") from e


def detect_aperture(
    data: np.ndarray,
    threshold: float = None,
    fill_holes: bool = False,
) -> np.ndarray:
    """检测二维数据中的有效孔径区域。

    通过背景零值检测识别有效信号区域。

    参数：
        data: 二维浮点数组（光强或相位）
        threshold: 背景判定阈值。默认为 data 最大绝对值的 1e-6 倍
        fill_holes: 是否填充内部空洞（从外围识别背景）。
            为 True 时，仅边缘连通的零值区域被视为背景，
            内部零值（如相位中心 = 0）保留为有效区域。

    返回：
        布尔掩膜数组，True 表示有效孔径区域，False 表示背景
    """
    # 计算默认阈值
    if threshold is None:
        max_abs = np.max(np.abs(data))
        # 全零数组：最大值为零，返回全 False 掩膜
        if max_abs == 0.0:
            return np.zeros(data.shape, dtype=bool)
        threshold = max_abs * 1e-6

    # 生成布尔掩膜：绝对值大于阈值的像素为有效区域
    mask = np.abs(data) > threshold

    # 填充内部空洞：仅边缘连通的 False 区域为背景，内部 False 填充为 True
    if fill_holes:
        mask = binary_fill_holes(mask)

    return mask


def _center_pad_to_shape(
    arr: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """将二维数组居中零填充到目标形状。

    较小数组被放置在目标网格中心，周围用零填充。
    这保证了空间位置对齐（假设两者共享同一中心和像素尺寸）。

    参数：
        arr: 源二维数组
        target_shape: (target_ny, target_nx)

    返回：
        零填充后的数组，形状为 target_shape
    """
    if arr.shape == target_shape:
        return arr.copy()

    result = np.zeros(target_shape, dtype=arr.dtype)
    # 计算居中偏移量
    pad_y = (target_shape[0] - arr.shape[0]) // 2
    pad_x = (target_shape[1] - arr.shape[1]) // 2
    result[pad_y:pad_y + arr.shape[0],
           pad_x:pad_x + arr.shape[1]] = arr
    return result


def _mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    """计算布尔掩膜的质心（像素坐标）。

    参数：
        mask: 二维布尔数组

    返回：
        (cy, cx) 质心像素坐标。若掩膜为空则返回网格几何中心。
    """
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return mask.shape[0] / 2.0, mask.shape[1] / 2.0
    return float(np.mean(ys)), float(np.mean(xs))


def _centroid_align_arrays(
    arr1: np.ndarray, centroid1: tuple[float, float],
    arr2: np.ndarray, centroid2: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """基于掩膜质心将两个数组对齐到通用网格上。

    计算允许两个数组质心重合的最小网格，
    将两者分别放置到该网格上，周围用零填充。

    参数：
        arr1: 第一个数组 (h1, w1)
        centroid1: arr1 掩膜的质心 (cy1, cx1)
        arr2: 第二个数组 (h2, w2)
        centroid2: arr2 掩膜的质心 (cy2, cx2)

    返回：
        (aligned_arr1, aligned_arr2)，形状相同的对齐后数组
    """
    cy1, cx1 = centroid1
    cy2, cx2 = centroid2
    h1, w1 = arr1.shape
    h2, w2 = arr2.shape

    # 将质心取整（像素对齐）
    icy1, icx1 = int(round(cy1)), int(round(cx1))
    icy2, icx2 = int(round(cy2)), int(round(cx2))

    # 计算对齐后质心周围所需空间
    above = max(icy1, icy2)
    below = max(h1 - icy1, h2 - icy2)
    left  = max(icx1, icx2)
    right = max(w1 - icx1, w2 - icx2)

    target_h = above + below
    target_w = left + right

    out1 = np.zeros((target_h, target_w), dtype=arr1.dtype)
    out2 = np.zeros((target_h, target_w), dtype=arr2.dtype)

    # arr1 的质心啩入目标网格的 (above, left) 位置
    y1 = above - icy1
    x1 = left - icx1
    out1[y1:y1 + h1, x1:x1 + w1] = arr1

    y2 = above - icy2
    x2 = left - icx2
    out2[y2:y2 + h2, x2:x2 + w2] = arr2

    return out1, out2


def match_apertures(
    irradiance: np.ndarray,
    phase: np.ndarray,
    threshold: float = None,
    align_mode: str = 'center',
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """匹配光强和相位数据的孔径区域。

    支持两种对齐模式：
      - 'center'：居中零填充，假设两者共享网格几何中心和像素尺寸。
      - 'centroid'：基于各自掩膜的质心对齐，从外围识别背景（fill_holes），
        适用于光斑偏心或相位中心为零的场景。

    参数：
        irradiance: 光强二维数组
        phase: 相位二维数组
        threshold: 孔径检测阈值（可选）
        align_mode: 对齐模式，'center'(默认) 或 'centroid'

    返回：
        (matched_irradiance, matched_phase, unified_mask) 元组
    """
    use_fill_holes = (align_mode == 'centroid')

    if align_mode == 'centroid':
        # 质心模式：先在各自原始网格上检测掩膜（填充内部空洞）
        irr_mask_raw = detect_aperture(irradiance, threshold, fill_holes=True)
        ph_mask_raw  = detect_aperture(phase, threshold, fill_holes=True)

        # 计算各自掩膜的质心
        cy_irr, cx_irr = _mask_centroid(irr_mask_raw)
        cy_ph, cx_ph   = _mask_centroid(ph_mask_raw)

        # 判断是否需要偏移对齐
        dy = int(round(cy_irr - cy_ph))
        dx = int(round(cx_irr - cx_ph))
        need_align = (irradiance.shape != phase.shape) or (dy != 0) or (dx != 0)

        if need_align:
            irr, ph = _centroid_align_arrays(
                irradiance, (cy_irr, cx_irr),
                phase, (cy_ph, cx_ph),
            )
        else:
            irr = irradiance.copy()
            ph = phase.copy()

    else:  # 'center'
        if irradiance.shape != phase.shape:
            target_ny = max(irradiance.shape[0], phase.shape[0])
            target_nx = max(irradiance.shape[1], phase.shape[1])
            target_shape = (target_ny, target_nx)
            irr = _center_pad_to_shape(irradiance, target_shape)
            ph  = _center_pad_to_shape(phase, target_shape)
        else:
            irr = irradiance.copy()
            ph = phase.copy()

    # 在对齐后的网格上检测掩膜
    irr_mask = detect_aperture(irr, threshold, fill_holes=use_fill_holes)
    phase_mask = detect_aperture(ph, threshold, fill_holes=use_fill_holes)

    # 统一掩膜：取两者的逻辑与（交集）
    unified_mask = irr_mask & phase_mask

    # 掩膜外置零
    irr[~unified_mask] = 0.0
    ph[~unified_mask] = 0.0

    return irr, ph, unified_mask


def _next_power_of_2(n: int) -> int:
    """返回 >= n 的最小 2 的幂。"""
    if n <= 0:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


def pad_to_power_of_2(
    Ex: np.ndarray,
) -> np.ndarray:
    """将复数电场数组居中零填充到 2 的幂次正方形尺寸。

    两个维度取较大的 power-of-2，确保输出始终为正方形。
    Zemax POP 模块要求 ZBF 网格为 2 的幂（32, 64, 128, ...）。

    参数：
        Ex: 复数电场 (ny, nx)

    返回：
        填充后的复数电场，形状为 (N, N)，N = max(pow2(ny), pow2(nx))
    """
    ny, nx = Ex.shape
    N = max(_next_power_of_2(ny), _next_power_of_2(nx))

    if N == ny and N == nx:
        return Ex.copy()

    result = np.zeros((N, N), dtype=Ex.dtype)
    pad_y = (N - ny) // 2
    pad_x = (N - nx) // 2
    result[pad_y:pad_y + ny, pad_x:pad_x + nx] = Ex
    return result


def _resize_to_square(
    Ex: np.ndarray,
    target_size: int,
) -> np.ndarray:
    """将复数电场居中裁剪或补零到 target_size × target_size。

    - 若输入尺寸 > target_size：从中心裁剪
    - 若输入尺寸 < target_size：居中零填充
    - 混合情况（一维裁剪一维补零）也正确处理

    参数：
        Ex: 复数电场 (ny, nx)
        target_size: 目标正方形尺寸

    返回：
        形状为 (target_size, target_size) 的复数电场
    """
    ny, nx = Ex.shape
    N = target_size

    if ny == N and nx == N:
        return Ex.copy()

    result = np.zeros((N, N), dtype=Ex.dtype)

    # 源区域（居中裁剪）
    src_y0 = max(0, (ny - N) // 2)
    src_x0 = max(0, (nx - N) // 2)
    src_y1 = src_y0 + min(ny, N)
    src_x1 = src_x0 + min(nx, N)

    # 目标区域（居中放置）
    dst_y0 = max(0, (N - ny) // 2)
    dst_x0 = max(0, (N - nx) // 2)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    result[dst_y0:dst_y1, dst_x0:dst_x1] = Ex[src_y0:src_y1, src_x0:src_x1]
    return result


def convert_csv_to_zbf(
    irradiance_csv: str,
    phase_csv: str,
    output_zbf: str,
    dx: float,
    dy: float,
    wavelength: float,
    units: int = 0,
    index: float = 1.0,
    wx: float = 0.0, wy: float = 0.0,
    Rx: float = 0.0, Ry: float = 0.0,
    zx: float = 0.0, zy: float = 0.0,
    threshold: float = None,
    verify: bool = True,
    force_power_of_2: bool = True,
    target_size: int = None,
    align_mode: str = 'center',
    plot: bool = False,
    output_dir: str = None,
) -> ZBFData:
    """CSV 到 ZBF 的端到端转换。

    参数：
        irradiance_csv: 光强 CSV 文件路径
        phase_csv: 相位 CSV 文件路径
        output_zbf: 输出 ZBF 文件路径
        dx, dy: X/Y 方向网格间距（透镜单位）
        wavelength: 波长（透镜单位）
        units: 长度单位 (0=mm, 1=cm, 2=in, 3=m)
        index: 折射率（默认 1.0）
        wx, wy, Rx, Ry, zx, zy: 导引光束参数（默认 0.0）
        threshold: 孔径检测阈值（可选）
        verify: 是否执行往返验证（默认 True）
        force_power_of_2: 是否将网格补零到 2 的幂次正方形尺寸（默认 True）
        target_size: 目标像素数（必须为 2 的幂）。指定时覆盖 force_power_of_2，
            通过居中裁剪或补零调整到 target_size × target_size。
        align_mode: 孔径对齐模式，'center'(默认) 或 'centroid'
        plot: 是否显示可视化图（默认 False）
        output_dir: 可视化图像输出目录（可选）

    返回：
        生成的 ZBFData 对象

    异常：
        ValueError: dx/dy <= 0 或 target_size 非 2 的幂
    """
    # 参数校验：网格间距须为正数
    if dx <= 0 or dy <= 0:
        raise ValueError(
            f"网格间距须为正数，当前 dx={dx}, dy={dy}"
        )

    # 参数校验：target_size 必须为 2 的幂
    if target_size is not None:
        if target_size <= 0 or (target_size & (target_size - 1)) != 0:
            raise ValueError(
                f"target_size 必须为 2 的幂（如 32, 64, 128, ...)，"
                f"当前 target_size={target_size}"
            )

    # 读取光强和相位 CSV 文件
    irradiance = read_beam_csv(irradiance_csv)
    phase = read_beam_csv(phase_csv)

    # 记录原始尺寸
    irr_shape_orig = irradiance.shape
    ph_shape_orig = phase.shape
    if irr_shape_orig != ph_shape_orig:
        print(
            f"注意：光强尺寸 {irr_shape_orig} "
            f"与相位尺寸 {ph_shape_orig} 不同，将执行对齐"
        )

    # 执行孔径匹配
    matched_irradiance, matched_phase, unified_mask = match_apertures(
        irradiance, phase, threshold, align_mode=align_mode
    )

    # 构造复数电场：振幅 = sqrt(光强)，Ex = 振幅 * exp(j * 相位)
    Ex = np.sqrt(matched_irradiance) * np.exp(1j * matched_phase)

    # 调整网格尺寸
    ny_orig, nx_orig = Ex.shape
    if target_size is not None:
        # 指定目标像素数：居中裁剪或补零到 target_size × target_size
        Ex = _resize_to_square(Ex, target_size)
        action = "裁剪" if target_size < max(ny_orig, nx_orig) else "补零"
        print(
            f"  目标尺寸{action}：{nx_orig}×{ny_orig} → "
            f"{target_size}×{target_size}"
        )
    elif force_power_of_2:
        # 自动补零到 2 的幂次正方形
        Ex = pad_to_power_of_2(Ex)
        if Ex.shape != (ny_orig, nx_orig):
            print(
                f"  2的幂次正方形补零：{nx_orig}×{ny_orig} → "
                f"{Ex.shape[1]}×{Ex.shape[0]}"
            )

    # 同步调整 unified_mask 形状
    if Ex.shape != unified_mask.shape:
        unified_mask = _resize_to_square(
            unified_mask.astype(np.uint8), Ex.shape[0]
        ).astype(bool)

    # 创建并填充 ZBFData 对象
    ny, nx = Ex.shape
    zbf_data = ZBFData()
    zbf_data.nx = nx
    zbf_data.ny = ny
    zbf_data.dx = dx
    zbf_data.dy = dy
    zbf_data.wavelength = wavelength
    zbf_data.units = units
    zbf_data.index = index
    zbf_data.wx = wx
    zbf_data.wy = wy
    zbf_data.Rx = Rx
    zbf_data.Ry = Ry
    zbf_data.zx = zx
    zbf_data.zy = zy
    zbf_data.Ex = Ex

    # 写入 ZBF 文件
    write_zbf(output_zbf, zbf_data)

    # 打印输出摘要
    total_power = np.sum(matched_irradiance)
    unit_names = {0: "mm", 1: "cm", 2: "in", 3: "m"}
    unit_str = unit_names.get(units, f"单位代码{units}")
    print(f"已写入 ZBF 文件：{output_zbf}")
    print(f"  网格尺寸：{nx} x {ny}")
    print(
        f"  物理范围：{nx * dx:.4f} x {ny * dy:.4f} {unit_str}"
    )
    print(f"  总功率：{total_power:.6e}")

    # 往返验证
    if verify:
        readback = read_zbf(output_zbf)
        # 振幅最大绝对误差
        amp_orig = np.abs(Ex)
        amp_read = np.abs(readback.Ex)
        amp_err = np.max(np.abs(amp_orig - amp_read))
        # 相位最大绝对误差（仅在孔径内比较）
        phase_orig = np.angle(Ex)
        phase_read = np.angle(readback.Ex)
        phase_diff = np.abs(phase_orig - phase_read)
        phase_diff[~unified_mask] = 0.0
        phase_err = np.max(phase_diff)
        print(f"  往返验证：振幅最大误差={amp_err:.2e}，"
              f"相位最大误差={phase_err:.2e}")

    # 可视化
    if plot:
        irr_mask = detect_aperture(irradiance, threshold)
        phase_mask = detect_aperture(phase, threshold)
        visualize_conversion(
            irradiance, phase,
            irr_mask, phase_mask, unified_mask,
            matched_irradiance, matched_phase,
            output_dir=output_dir,
        )

    return zbf_data


def visualize_conversion(
    raw_irradiance: np.ndarray,
    raw_phase: np.ndarray,
    irr_mask: np.ndarray,
    phase_mask: np.ndarray,
    unified_mask: np.ndarray,
    matched_irradiance: np.ndarray,
    matched_phase: np.ndarray,
    output_dir: str = None,
) -> None:
    """可视化转换流程的关键步骤。

    生成 2×4 多子图窗口，包含：原始数据、孔径检测、孔径匹配、最终结果。

    参数：
        raw_irradiance: 原始光强数组
        raw_phase: 原始相位数组
        irr_mask: 光强孔径掩膜
        phase_mask: 相位孔径掩膜
        unified_mask: 统一孔径掩膜
        matched_irradiance: 匹配后的光强
        matched_phase: 匹配后的相位
        output_dir: PNG 图像输出目录（可选）
    """
    # 创建 2×4 多子图布局
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # 第一行第一列：原始光强
    im00 = axes[0, 0].imshow(raw_irradiance, cmap='hot')
    axes[0, 0].set_title('原始光强')
    fig.colorbar(im00, ax=axes[0, 0], shrink=0.8)

    # 第一行第二列：原始相位
    im01 = axes[0, 1].imshow(raw_phase, cmap='twilight')
    axes[0, 1].set_title('原始相位')
    fig.colorbar(im01, ax=axes[0, 1], shrink=0.8)

    # 第一行第三列：光强掩膜
    im02 = axes[0, 2].imshow(irr_mask, cmap='gray')
    axes[0, 2].set_title('光强掩膜')
    fig.colorbar(im02, ax=axes[0, 2], shrink=0.8)

    # 第一行第四列：相位掩膜
    im03 = axes[0, 3].imshow(phase_mask, cmap='gray')
    axes[0, 3].set_title('相位掩膜')
    fig.colorbar(im03, ax=axes[0, 3], shrink=0.8)

    # 第二行第一列：统一掩膜叠加光强
    axes[1, 0].imshow(raw_irradiance, cmap='hot')
    # 半透明掩膜叠加：掩膜外区域用半透明灰色覆盖
    mask_overlay = np.ma.masked_where(unified_mask, unified_mask)
    axes[1, 0].imshow(
        mask_overlay, cmap='gray', alpha=0.5,
        vmin=0, vmax=1,
    )
    axes[1, 0].set_title('统一掩膜叠加光强')

    # 第二行第二列：匹配后光强
    im11 = axes[1, 1].imshow(matched_irradiance, cmap='hot')
    axes[1, 1].set_title('匹配后光强')
    fig.colorbar(im11, ax=axes[1, 1], shrink=0.8)

    # 第二行第三列：匹配后相位
    im12 = axes[1, 2].imshow(matched_phase, cmap='twilight')
    axes[1, 2].set_title('匹配后相位')
    fig.colorbar(im12, ax=axes[1, 2], shrink=0.8)

    # 第二行第四列：摘要信息
    axes[1, 3].axis('off')
    axes[1, 3].set_title('摘要')
    # 计算摘要统计信息
    total_power = np.sum(matched_irradiance)
    aperture_ratio = np.sum(unified_mask) / unified_mask.size * 100
    summary_text = (
        f"原始光强尺寸：{raw_irradiance.shape}\n"
        f"原始相位尺寸：{raw_phase.shape}\n"
        f"匹配后尺寸：{matched_irradiance.shape}\n"
        f"有效孔径占比：{aperture_ratio:.1f}%\n"
        f"总功率：{total_power:.4e}"
    )
    axes[1, 3].text(
        0.5, 0.5, summary_text,
        transform=axes[1, 3].transAxes,
        fontsize=12, verticalalignment='center',
        horizontalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )

    plt.tight_layout()

    # 若指定输出目录，保存为 PNG 文件
    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        save_path = out_path / 'conversion_overview.png'
        fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
        print(f"可视化图像已保存：{save_path}")

    plt.show()


def main():
    """命令行入口函数。

    使用 argparse 解析命令行参数，执行 CSV 到 ZBF 的转换，
    并根据选项决定是否生成可视化图像。
    """
    parser = argparse.ArgumentParser(
        description='CSV 到 ZBF 文件转换器'
    )

    # 位置参数
    parser.add_argument(
        'irradiance_csv',
        help='光强 CSV 文件路径',
    )
    parser.add_argument(
        'phase_csv',
        help='相位 CSV 文件路径',
    )
    parser.add_argument(
        'output_zbf',
        help='输出 ZBF 文件路径',
    )

    # 必需选项
    parser.add_argument(
        '--dx', type=float, required=True,
        help='X 方向网格间距（透镜单位）',
    )
    parser.add_argument(
        '--wavelength', type=float, required=True,
        help='波长（透镜单位）',
    )

    # 可选选项
    parser.add_argument(
        '--dy', type=float, default=None,
        help='Y 方向网格间距（默认等于 dx）',
    )
    parser.add_argument(
        '--units', type=int, default=0,
        help='长度单位：0=mm, 1=cm, 2=in, 3=m（默认 0）',
    )
    parser.add_argument(
        '--threshold', type=float, default=None,
        help='孔径检测阈值（默认自动计算）',
    )
    parser.add_argument(
        '--no-plot', action='store_true',
        help='禁用可视化',
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='可视化图像输出目录',
    )
    parser.add_argument(
        '--align-mode', type=str, default='center',
        choices=['center', 'centroid'],
        help='孔径对齐模式：center=几何中心, centroid=掩膜质心（默认 center）',
    )

    # 可选导引光束参数
    parser.add_argument(
        '--index', type=float, default=1.0,
        help='折射率（默认 1.0）',
    )
    parser.add_argument(
        '--wx', type=float, default=0.0,
        help='导引光束参数 wx（默认 0.0）',
    )
    parser.add_argument(
        '--wy', type=float, default=0.0,
        help='导引光束参数 wy（默认 0.0）',
    )
    parser.add_argument(
        '--Rx', type=float, default=0.0,
        help='导引光束参数 Rx（默认 0.0）',
    )
    parser.add_argument(
        '--Ry', type=float, default=0.0,
        help='导引光束参数 Ry（默认 0.0）',
    )
    parser.add_argument(
        '--zx', type=float, default=0.0,
        help='导引光束参数 zx（默认 0.0）',
    )
    parser.add_argument(
        '--zy', type=float, default=0.0,
        help='导引光束参数 zy（默认 0.0）',
    )

    # 解析参数
    args = parser.parse_args()

    # 若未提供 dy，默认等于 dx
    dy = args.dy if args.dy is not None else args.dx

    # 读取原始数据（用于可视化）
    raw_irradiance = read_beam_csv(args.irradiance_csv)
    raw_phase = read_beam_csv(args.phase_csv)

    # 执行转换
    zbf_data = convert_csv_to_zbf(
        args.irradiance_csv, args.phase_csv, args.output_zbf,
        dx=args.dx, dy=dy, wavelength=args.wavelength,
        units=args.units, index=args.index,
        wx=args.wx, wy=args.wy,
        Rx=args.Rx, Ry=args.Ry,
        zx=args.zx, zy=args.zy,
        threshold=args.threshold,
        align_mode=args.align_mode,
    )

    # 可视化
    if not args.no_plot:
        # 计算孔径掩膜用于可视化
        irr_mask = detect_aperture(raw_irradiance, args.threshold)
        phase_mask = detect_aperture(raw_phase, args.threshold)
        matched_irr, matched_ph, unified_mask = match_apertures(
            raw_irradiance, raw_phase, args.threshold,
            align_mode=args.align_mode,
        )
        visualize_conversion(
            raw_irradiance, raw_phase,
            irr_mask, phase_mask, unified_mask,
            matched_irr, matched_ph,
            output_dir=args.output_dir,
        )


if __name__ == '__main__':
    main()
