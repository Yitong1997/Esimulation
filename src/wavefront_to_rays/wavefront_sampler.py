"""
波前采样为几何光线的核心模块

本模块实现将入射波前复振幅采样为几何光线的功能。
输入为入射面的波面复振幅，输出为出射光束的光线数据。

工作原理：
1. 从波前复振幅中提取相位信息
2. 创建一个相位面（薄元件），其相位分布与输入波前相位匹配
3. 产生平面波光束入射到相位面
4. 通过 optiland 进行光线追迹
5. 输出出射光束的光线数据

作者：混合光学仿真项目
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from optiland.optic import Optic
from optiland.phase import GridPhaseProfile
from optiland.rays import RealRays
from optiland.distribution import create_distribution

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
    
    n = phase_grid.shape[0]
    
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
    new_n = n + 2 * extend_pixels
    extended_phase = np.zeros((new_n, new_n), dtype=phase_grid.dtype)
    
    # 复制原始数据到中心区域
    extended_phase[extend_pixels:extend_pixels+n, extend_pixels:extend_pixels+n] = phase_grid
    
    # 使用边缘值填充扩展区域
    # 上边缘（复制第一行）
    for i in range(extend_pixels):
        extended_phase[i, extend_pixels:extend_pixels+n] = phase_grid[0, :]
    # 下边缘（复制最后一行）
    for i in range(extend_pixels):
        extended_phase[extend_pixels+n+i, extend_pixels:extend_pixels+n] = phase_grid[-1, :]
    # 左边缘（复制第一列）
    for j in range(extend_pixels):
        extended_phase[extend_pixels:extend_pixels+n, j] = phase_grid[:, 0]
    # 右边缘（复制最后一列）
    for j in range(extend_pixels):
        extended_phase[extend_pixels:extend_pixels+n, extend_pixels+n+j] = phase_grid[:, -1]
    
    # 四个角落（使用对应角的值）
    # 左上角
    extended_phase[:extend_pixels, :extend_pixels] = phase_grid[0, 0]
    # 右上角
    extended_phase[:extend_pixels, extend_pixels+n:] = phase_grid[0, -1]
    # 左下角
    extended_phase[extend_pixels+n:, :extend_pixels] = phase_grid[-1, 0]
    # 右下角
    extended_phase[extend_pixels+n:, extend_pixels+n:] = phase_grid[-1, -1]
    
    return extended_phase, extended_x, extended_y


class WavefrontToRaysSampler:
    """将波前复振幅采样为几何光线的采样器
    
    本类实现将入射波前（复振幅形式）转换为几何光线的功能。
    通过创建一个相位面来模拟波前的相位分布，然后使用平面波入射
    进行光线追迹，得到出射光线。
    
    参数:
        wavefront_amplitude (NDArray): 波前复振幅数组，形状为 (N, N)
        physical_size (float): 波前的物理尺寸（直径），单位：mm
        wavelength (float): 波长，单位：μm
        num_rays (int): 期望的采样光线数量，默认 100。
            注意：对于 hexapolar 分布，实际光线数会略有不同，
            因为 hexapolar 分布的光线数由环数决定。
            实际光线数 = 1 + 3*n*(n+1)，其中 n 是环数。
        distribution (str): 光线分布类型，默认 'hexapolar'
        edge_extend_pixels (int): 相位网格边缘扩展像素数，默认 2。
            用于避免边缘像素的插值失败。设为 0 禁用扩展。
    
    属性:
        phase_grid (NDArray): 从波前提取的相位网格（弧度）
        amplitude_grid (NDArray): 从波前提取的振幅网格
        optic (Optic): 包含相位面的光学系统
        output_rays (RealRays): 出射光线数据
    """
    
    def __init__(
        self,
        wavefront_amplitude: NDArray,
        physical_size: float,
        wavelength: float,
        num_rays: int = 100,
        distribution: str = "hexapolar",
        edge_extend_pixels: int = 2,
    ):
        """初始化波前采样器
        
        参数:
            wavefront_amplitude: 波前复振幅数组，形状为 (N, N)，复数类型
            physical_size: 波前的物理尺寸（直径），单位：mm
            wavelength: 波长，单位：μm
            num_rays: 采样光线数量
            distribution: 光线分布类型
            edge_extend_pixels: 相位网格边缘扩展像素数，默认 2
        """
        self.wavefront_amplitude = np.asarray(wavefront_amplitude)
        self.physical_size = physical_size
        self.wavelength = wavelength
        self.num_rays = num_rays
        self.distribution_type = distribution
        self.edge_extend_pixels = edge_extend_pixels
        
        # 验证输入
        self._validate_input()
        
        # 提取相位和振幅
        self.phase_grid = self._extract_phase()
        self.amplitude_grid = self._extract_amplitude()
        
        # 创建坐标网格
        self._create_coordinate_grids()
        
        # 创建光学系统
        self.optic = self._create_optic()
        
        # 执行光线追迹
        self.output_rays = self._trace_rays()
    
    def _validate_input(self):
        """验证输入参数"""
        if self.wavefront_amplitude.ndim != 2:
            raise ValueError(
                f"波前数组必须是二维的，当前维度: {self.wavefront_amplitude.ndim}"
            )
        
        if self.wavefront_amplitude.shape[0] != self.wavefront_amplitude.shape[1]:
            raise ValueError(
                f"波前数组必须是正方形，当前形状: {self.wavefront_amplitude.shape}"
            )
        
        if self.physical_size <= 0:
            raise ValueError(f"物理尺寸必须为正数，当前值: {self.physical_size}")
        
        if self.wavelength <= 0:
            raise ValueError(f"波长必须为正数，当前值: {self.wavelength}")
    
    def _extract_phase(self) -> NDArray:
        """从波前复振幅中提取相位
        
        返回:
            相位数组（弧度），形状与输入波前相同
        """
        # 使用 np.angle 提取相位，范围 [-π, π]
        phase = np.angle(self.wavefront_amplitude)
        return phase
    
    def _extract_amplitude(self) -> NDArray:
        """从波前复振幅中提取振幅
        
        返回:
            振幅数组，形状与输入波前相同
        """
        amplitude = np.abs(self.wavefront_amplitude)
        return amplitude
    
    def _create_coordinate_grids(self):
        """创建物理坐标网格"""
        n = self.wavefront_amplitude.shape[0]
        half_size = self.physical_size / 2.0
        
        # 创建物理坐标（单位：mm）
        coords = np.linspace(-half_size, half_size, n)
        self.x_coords = coords
        self.y_coords = coords
        
        # 创建网格
        self.x_grid, self.y_grid = np.meshgrid(coords, coords)
    
    def _create_optic(self) -> Optic:
        """创建包含相位面的光学系统
        
        创建一个简单的光学系统，包含：
        1. 物面（无穷远）
        2. 相位面（位于入射面，厚度为 0）
        3. 像面
        
        注意：为了避免边缘像素的插值失败，相位网格会被扩展。
        扩展使用边缘值外推，不改变相位的计算逻辑。
        
        重要：optiland 的相位面存在单位不一致问题：
        - 相位梯度单位是 rad/mm（因为坐标单位是 mm）
        - 但 k0 = 2π/λ 的单位是 rad/μm = 1000 rad/mm
        - 这导致相位梯度对光线方向的影响被放大了 1000 倍
        - 为了修正这个问题，我们将相位值缩小 1000 倍
        
        返回:
            配置好的 Optic 对象
        """
        optic = Optic()
        
        # 设置孔径（入瞳直径等于波前物理尺寸）
        optic.set_aperture(aperture_type='EPD', value=self.physical_size)
        
        # 设置视场类型为角度，轴上视场
        optic.set_field_type(field_type='angle')
        optic.add_field(y=0, x=0)
        
        # 设置波长
        optic.add_wavelength(value=self.wavelength, is_primary=True)
        
        # 扩展相位网格以避免边缘插值失败
        extended_phase, extended_x, extended_y = _extend_phase_grid(
            self.phase_grid,
            self.x_coords,
            self.y_coords,
            extend_pixels=self.edge_extend_pixels,
        )
        
        # 修正 optiland 的单位问题：将相位值缩小 1000 倍
        # 这样相位梯度也会缩小 1000 倍，从而得到正确的光线方向
        corrected_phase = extended_phase / 1000.0
        
        # 创建相位分布（使用扩展后的网格）
        phase_profile = GridPhaseProfile(
            x_coords=extended_x,
            y_coords=extended_y,
            phase_grid=corrected_phase,
        )
        
        # 添加物面（index=0）
        optic.add_surface(index=0, radius=np.inf, thickness=np.inf)
        
        # 添加相位面（平面）
        # index=1 是第一个光学面
        optic.add_surface(
            index=1,
            surface_type='standard',
            radius=np.inf,  # 平面
            thickness=0.0,  # 厚度
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
    
    def _trace_rays(self) -> RealRays:
        """执行光线追迹
        
        使用平面波入射到相位面，追迹光线并返回出射光线数据。
        
        注意：optiland 的 hexapolar 分布的 num_rays 参数实际上是环数（num_rings），
        不是光线数。实际光线数 = 1 + 3*num_rings*(num_rings+1)。
        例如：num_rays=6 产生 127 条光线，num_rays=10 产生 331 条光线。
        
        为了让 num_rays 参数更直观，我们将其转换为合适的环数。
        
        返回:
            出射光线数据 (RealRays 对象)
        """
        # 将期望的光线数转换为 hexapolar 环数
        # 实际光线数 = 1 + 3*n*(n+1)，解方程得 n ≈ sqrt(num_rays/3) - 0.5
        if self.distribution_type == "hexapolar":
            # 计算需要的环数
            num_rings = max(1, int(np.sqrt(self.num_rays / 3.0)))
            actual_rays = 1 + 3 * num_rings * (num_rings + 1)
            # 如果实际光线数太少，增加一环
            while actual_rays < self.num_rays and num_rings < 100:
                num_rings += 1
                actual_rays = 1 + 3 * num_rings * (num_rings + 1)
        else:
            num_rings = self.num_rays
        
        # 使用 optic.trace 进行光线追迹
        # Hx=0, Hy=0 表示轴上视场（平面波入射）
        rays = self.optic.trace(
            Hx=0,
            Hy=0,
            wavelength=self.wavelength,
            num_rays=num_rings,  # 对于 hexapolar，这是环数
            distribution=self.distribution_type,
        )
        
        return rays
    
    def get_output_rays(self) -> RealRays:
        """获取出射光线数据
        
        返回:
            出射光线数据 (RealRays 对象)
        """
        return self.output_rays
    
    def get_ray_positions(self) -> tuple[NDArray, NDArray]:
        """获取出射光线的位置坐标
        
        返回:
            (x, y) 坐标数组的元组，单位：mm
        """
        x = np.asarray(self.output_rays.x)
        y = np.asarray(self.output_rays.y)
        return x, y
    
    def get_ray_directions(self) -> tuple[NDArray, NDArray, NDArray]:
        """获取出射光线的方向余弦
        
        返回:
            (L, M, N) 方向余弦数组的元组
        """
        L = np.asarray(self.output_rays.L)
        M = np.asarray(self.output_rays.M)
        N = np.asarray(self.output_rays.N)
        return L, M, N
    
    def get_ray_opd(self) -> NDArray:
        """获取出射光线的 OPD（相对于主光线，单位：波长数）
        
        注意：optiland 的 rays.opd 是从物面到像面的总光程（单位：mm）。
        本方法返回相对于主光线的 OPD，并转换为波长数。
        
        重要：由于我们在 _create_optic 中将相位缩小了 1000 倍来修正
        光线方向的单位问题，相位引入的 OPD 也被缩小了 1000 倍。
        但 optiland 的 OPD 计算本身就有 1000 倍放大问题，所以两者抵消了。
        
        返回:
            OPD 数组，单位：波长数（相对于主光线）
        """
        # 获取主光线 OPD 作为参考
        chief_ray = self.optic.trace_generic(
            Hx=0, Hy=0, Px=0, Py=0, wavelength=self.wavelength
        )
        chief_opd_mm = float(np.asarray(chief_ray.opd).item())
        
        # 计算相对于主光线的 OPD（单位：mm）
        opd_mm = np.asarray(self.output_rays.opd)
        relative_opd_mm = opd_mm - chief_opd_mm
        
        # 由于相位被缩小了 1000 倍，OPD 也被缩小了 1000 倍
        # 但 optiland 的 OPD 计算有 1000 倍放大问题，两者抵消
        # 所以不需要额外的修正
        
        # 转换为波长数：OPD(mm) / wavelength(mm)
        wavelength_mm = self.wavelength * 1e-3  # μm -> mm
        opd_waves = relative_opd_mm / wavelength_mm
        
        return opd_waves
    
    def get_ray_opd_raw(self) -> NDArray:
        """获取出射光线的原始 OPD（单位：mm）
        
        返回 optiland 计算的原始 OPD 值，不做任何转换。
        
        返回:
            OPD 数组，单位：mm
        """
        return np.asarray(self.output_rays.opd)
    
    def get_ray_intensity(self) -> NDArray:
        """获取出射光线的强度
        
        返回:
            强度数组
        """
        return np.asarray(self.output_rays.i)
    
    def phase_to_opd_waves(self) -> NDArray:
        """将相位转换为 OPD（波长数）
        
        返回:
            OPD 网格，单位：波长数
        """
        # 相位（弧度）转 OPD（波长数）：OPD = phase / (2π)
        return self.phase_grid / (2 * np.pi)
