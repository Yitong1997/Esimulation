"""
相位面创建辅助模块

本模块提供创建相位面光学系统的辅助函数。

作者：混合光学仿真项目
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from optiland.optic import Optic
from optiland.phase import GridPhaseProfile

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _extend_phase_grid(
    phase_grid: NDArray,
    x_coords: NDArray,
    y_coords: NDArray,
    extend_pixels: int = 2,
) -> tuple[NDArray, NDArray, NDArray]:
    """扩展相位网格的边缘区域
    
    为了避免边缘像素的插值失败，将相位网格向外扩展若干像素。
    扩展时使用边缘值外推（保持边缘像素的值不变），不改变相位的计算逻辑。
    
    参数:
        phase_grid: 原始相位网格，形状为 (N, N)
        x_coords: 原始 X 坐标数组
        y_coords: 原始 Y 坐标数组
        extend_pixels: 向外扩展的像素数，默认 2
    
    返回:
        (extended_phase, extended_x, extended_y) 元组
        - extended_phase: 扩展后的相位网格
        - extended_x: 扩展后的 X 坐标数组
        - extended_y: 扩展后的 Y 坐标数组
    """
    if extend_pixels <= 0:
        return phase_grid, x_coords, y_coords
    
    n_rows = phase_grid.shape[0]
    n_cols = phase_grid.shape[1]
    
    # 计算坐标间距
    dx = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 1.0
    dy = y_coords[1] - y_coords[0] if len(y_coords) > 1 else 1.0
    
    # 创建扩展后的坐标数组
    # 向左/下扩展
    x_left = x_coords[0] - dx * np.arange(extend_pixels, 0, -1)
    y_bottom = y_coords[0] - dy * np.arange(extend_pixels, 0, -1)
    # 向右/上扩展
    x_right = x_coords[-1] + dx * np.arange(1, extend_pixels + 1)
    y_top = y_coords[-1] + dy * np.arange(1, extend_pixels + 1)
    
    # 拼接坐标
    extended_x = np.concatenate([x_left, x_coords, x_right])
    extended_y = np.concatenate([y_bottom, y_coords, y_top])
    
    # 创建扩展后的相位网格
    new_rows = n_rows + 2 * extend_pixels
    new_cols = n_cols + 2 * extend_pixels
    extended_phase = np.zeros((new_rows, new_cols), dtype=phase_grid.dtype)
    
    # 复制原始数据到中心区域
    extended_phase[extend_pixels:extend_pixels+n_rows, extend_pixels:extend_pixels+n_cols] = phase_grid
    
    # 使用边缘值填充扩展区域
    # 上边缘（复制第一行）
    for i in range(extend_pixels):
        extended_phase[i, extend_pixels:extend_pixels+n_cols] = phase_grid[0, :]
    # 下边缘（复制最后一行）
    for i in range(extend_pixels):
        extended_phase[extend_pixels+n_rows+i, extend_pixels:extend_pixels+n_cols] = phase_grid[-1, :]
    # 左边缘（复制第一列）
    for j in range(extend_pixels):
        extended_phase[extend_pixels:extend_pixels+n_rows, j] = phase_grid[:, 0]
    # 右边缘（复制最后一列）
    for j in range(extend_pixels):
        extended_phase[extend_pixels:extend_pixels+n_rows, extend_pixels+n_cols+j] = phase_grid[:, -1]
    
    # 四个角落（使用对应角的值）
    # 左上角
    extended_phase[:extend_pixels, :extend_pixels] = phase_grid[0, 0]
    # 右上角
    extended_phase[:extend_pixels, extend_pixels+n_cols:] = phase_grid[0, -1]
    # 左下角
    extended_phase[extend_pixels+n_rows:, :extend_pixels] = phase_grid[-1, 0]
    # 右下角
    extended_phase[extend_pixels+n_rows:, extend_pixels+n_cols:] = phase_grid[-1, -1]
    
    return extended_phase, extended_x, extended_y


def create_phase_surface_optic(
    phase_grid: NDArray,
    x_coords: NDArray,
    y_coords: NDArray,
    wavelength: float,
    aperture_diameter: float | None = None,
    edge_extend_pixels: int = 2,
) -> Optic:
    """创建包含相位面的光学系统
    
    创建一个简单的光学系统，仅包含一个相位面（薄元件）。
    相位面位于入射面位置，厚度为 0。
    
    为了避免边缘像素的插值失败，相位网格会被扩展。
    扩展使用边缘值外推，不改变相位的计算逻辑。
    
    参数:
        phase_grid: 相位网格数据（弧度），形状为 (N, N)
        x_coords: X 坐标数组，单位：mm
        y_coords: Y 坐标数组，单位：mm
        wavelength: 波长，单位：μm
        aperture_diameter: 孔径直径，单位：mm。如果为 None，则使用坐标范围
        edge_extend_pixels: 相位网格边缘扩展像素数，默认 2。
            用于避免边缘像素的插值失败。设为 0 禁用扩展。
    
    返回:
        配置好的 Optic 对象
    
    示例:
        >>> import numpy as np
        >>> # 创建一个简单的球面波前相位
        >>> n = 64
        >>> x = np.linspace(-5, 5, n)
        >>> y = np.linspace(-5, 5, n)
        >>> X, Y = np.meshgrid(x, y)
        >>> R = np.sqrt(X**2 + Y**2)
        >>> phase = 0.5 * R**2  # 简单的二次相位
        >>> optic = create_phase_surface_optic(phase, x, y, wavelength=0.55)
    """
    # 验证输入
    if phase_grid.ndim != 2:
        raise ValueError(f"相位网格必须是二维的，当前维度: {phase_grid.ndim}")
    
    if len(x_coords) != phase_grid.shape[1]:
        raise ValueError(
            f"X 坐标长度 ({len(x_coords)}) 与相位网格列数 ({phase_grid.shape[1]}) 不匹配"
        )
    
    if len(y_coords) != phase_grid.shape[0]:
        raise ValueError(
            f"Y 坐标长度 ({len(y_coords)}) 与相位网格行数 ({phase_grid.shape[0]}) 不匹配"
        )
    
    # 计算孔径直径
    if aperture_diameter is None:
        x_range = x_coords[-1] - x_coords[0]
        y_range = y_coords[-1] - y_coords[0]
        aperture_diameter = min(x_range, y_range)
    
    # 创建光学系统
    optic = Optic()
    
    # 设置孔径
    optic.set_aperture(aperture_type='EPD', value=aperture_diameter)
    
    # 设置视场类型为角度，轴上视场
    optic.set_field_type(field_type='angle')
    optic.add_field(y=0, x=0)
    
    # 设置波长
    optic.add_wavelength(value=wavelength, is_primary=True)
    
    # 扩展相位网格以避免边缘插值失败
    extended_phase, extended_x, extended_y = _extend_phase_grid(
        phase_grid,
        x_coords,
        y_coords,
        extend_pixels=edge_extend_pixels,
    )
    
    # 创建相位分布（使用扩展后的网格）
    phase_profile = GridPhaseProfile(
        x_coords=extended_x,
        y_coords=extended_y,
        phase_grid=extended_phase,
    )
    
    # 添加物面（index=0）
    optic.add_surface(index=0, radius=np.inf, thickness=np.inf)
    
    # 添加相位面（index=1）
    optic.add_surface(
        index=1,
        surface_type='standard',
        radius=np.inf,
        thickness=0.0,
        material='air',
        is_stop=True,
        phase_profile=phase_profile,
    )
    
    # 添加像面（index=2）
    optic.add_surface(
        index=2,
        surface_type='standard',
        radius=np.inf,
        thickness=0.0,
        material='air',
    )
    
    return optic


def wavefront_to_phase(
    wavefront: NDArray,
) -> NDArray:
    """从波前复振幅中提取相位
    
    参数:
        wavefront: 波前复振幅数组（复数）
    
    返回:
        相位数组（弧度），范围 [-π, π]
    """
    return np.angle(wavefront)


def phase_to_opd_waves(phase: NDArray) -> NDArray:
    """将相位（弧度）转换为 OPD（波长数）
    
    参数:
        phase: 相位数组（弧度）
    
    返回:
        OPD 数组（波长数）
    """
    return phase / (2 * np.pi)


def opd_waves_to_phase(opd_waves: NDArray) -> NDArray:
    """将 OPD（波长数）转换为相位（弧度）
    
    参数:
        opd_waves: OPD 数组（波长数）
    
    返回:
        相位数组（弧度）
    """
    return 2 * np.pi * opd_waves
