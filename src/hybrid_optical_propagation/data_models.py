"""
混合光学传播系统数据模型

本模块定义混合光学传播系统中使用的核心数据类。

主要数据类：
- PilotBeamParams: Pilot Beam 参数（基于 ABCD 法则）
- GridSampling: 网格采样信息
- PropagationState: 传播状态
- SourceDefinition: 入射波面定义

重要：仿真波前使用振幅和相位分离存储，相位为非折叠实数。
这避免了复数形式 exp(1j*φ) 隐含的相位折叠问题。

**Validates: Requirements 1.1-1.6, 8.1-8.7, 10.1-10.5, 17.1-17.7**
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Any, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from sequential_system.coordinate_tracking import OpticalAxisState


@dataclass
class PilotBeamParams:
    """Pilot Beam 参数
    
    基于 ABCD 法则追踪的理想高斯光束参数。用于相位解包裹和参考相位计算。
    
    属性:
        wavelength_um: 波长 (μm)
        waist_radius_mm: 束腰半径 (mm)
        waist_position_mm: 束腰位置（相对于当前位置）(mm)
        curvature_radius_mm: 当前曲率半径 (mm)，正值表示发散，负值表示会聚
        spot_size_mm: 当前光斑大小（1/e² 半径）(mm)
        q_parameter: 复参数 q (mm)
    
    复参数 q 的定义：
        1/q = 1/R - j*λ/(π*w²)
        其中 R 是曲率半径，w 是光斑大小
    
    **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6**
    """
    wavelength_um: float
    waist_radius_mm: float
    waist_position_mm: float
    curvature_radius_mm: float
    spot_size_mm: float
    q_parameter: complex
    
    @classmethod
    def from_gaussian_source(
        cls,
        wavelength_um: float,
        w0_mm: float,
        z0_mm: float,
    ) -> "PilotBeamParams":
        """从高斯光源参数创建
        
        参数:
            wavelength_um: 波长 (μm)
            w0_mm: 束腰半径 (mm)
            z0_mm: 束腰位置（负值表示束腰在当前位置之前）(mm)
        
        返回:
            PilotBeamParams 对象
        
        **Validates: Requirements 8.1, 8.2**
        """
        wavelength_mm = wavelength_um * 1e-3
        
        # 瑞利长度: z_R = π * w0² / λ
        z_R = np.pi * w0_mm**2 / wavelength_mm
        
        # 复参数 q = z + j*z_R（在束腰处 q = j*z_R）
        # 当前位置相对于束腰的距离是 -z0_mm（因为 z0_mm 是束腰相对于当前位置）
        z = -z0_mm
        q = z + 1j * z_R
        
        # 从 q 计算曲率半径和光斑大小
        if abs(z) < 1e-15:
            # 在束腰处
            R = np.inf
            w = w0_mm
        else:
            # 曲率半径: R = z * (1 + (z_R/z)²)
            R = z * (1 + (z_R / z)**2)
            # 光斑大小: w = w0 * sqrt(1 + (z/z_R)²)
            w = w0_mm * np.sqrt(1 + (z / z_R)**2)
        
        return cls(
            wavelength_um=wavelength_um,
            waist_radius_mm=w0_mm,
            waist_position_mm=z0_mm,
            curvature_radius_mm=R,
            spot_size_mm=w,
            q_parameter=q,
        )
    
    @classmethod
    def from_q_parameter(
        cls,
        q: complex,
        wavelength_um: float,
    ) -> "PilotBeamParams":
        """从复参数 q 创建
        
        参数:
            q: 复参数 q (mm)
            wavelength_um: 波长 (μm)
        
        返回:
            PilotBeamParams 对象
        
        **Validates: Requirements 8.1, 8.2**
        """
        wavelength_mm = wavelength_um * 1e-3
        
        # 从 1/q 提取参数
        # 1/q = 1/R - j*λ/(π*w²)
        inv_q = 1.0 / q
        
        # 曲率半径
        real_part = np.real(inv_q)
        if abs(real_part) < 1e-15:
            R = np.inf
        else:
            R = 1.0 / real_part
        
        # 光斑大小
        imag_part = np.imag(inv_q)
        w_sq = -wavelength_mm / (np.pi * imag_part)
        w = np.sqrt(w_sq) if w_sq > 0 else 0.0
        
        # 瑞利长度和束腰半径
        z_R = np.imag(q)
        w0 = np.sqrt(wavelength_mm * z_R / np.pi) if z_R > 0 else w
        
        # 束腰位置（相对于当前位置）
        z = np.real(q)
        z0 = -z  # 束腰在当前位置之前为负
        
        return cls(
            wavelength_um=wavelength_um,
            waist_radius_mm=w0,
            waist_position_mm=z0,
            curvature_radius_mm=R,
            spot_size_mm=w,
            q_parameter=q,
        )


    
    def propagate(self, distance_mm: float) -> "PilotBeamParams":
        """自由空间传播
        
        使用 ABCD 矩阵法计算传播后的光束参数。
        
        自由空间传播的 ABCD 矩阵:
            | 1  d |
            | 0  1 |
        
        q 参数变换: q_out = (A*q_in + B) / (C*q_in + D) = q_in + d
        
        参数:
            distance_mm: 传播距离 (mm)，正值为正向传播
        
        返回:
            传播后的 PilotBeamParams 对象
        
        **Validates: Requirements 8.3**
        """
        # 自由空间传播: q_out = q_in + d
        q_new = self.q_parameter + distance_mm
        return PilotBeamParams.from_q_parameter(q_new, self.wavelength_um)
    
    def apply_lens(self, focal_length_mm: float) -> "PilotBeamParams":
        """薄透镜效果
        
        使用 ABCD 矩阵法计算通过薄透镜后的光束参数。
        
        薄透镜的 ABCD 矩阵:
            |  1    0  |
            | -1/f  1  |
        
        q 参数变换: q_out = q_in / (1 - q_in/f)
        
        参数:
            focal_length_mm: 焦距 (mm)，正值为会聚透镜
        
        返回:
            变换后的 PilotBeamParams 对象
        
        **Validates: Requirements 8.3**
        """
        if np.isinf(focal_length_mm):
            return self  # 无穷焦距，无效果
        
        # 薄透镜变换: 1/q_out = 1/q_in - 1/f
        # 等价于: q_out = q_in * f / (f - q_in)
        A, B, C, D = 1, 0, -1/focal_length_mm, 1
        q_new = (A * self.q_parameter + B) / (C * self.q_parameter + D)
        return PilotBeamParams.from_q_parameter(q_new, self.wavelength_um)
    
    def apply_mirror(self, radius_mm: float) -> "PilotBeamParams":
        """球面镜效果
        
        使用 ABCD 矩阵法计算反射后的光束参数。
        
        球面镜的 ABCD 矩阵（焦距 f = R/2）:
            |  1     0  |
            | -2/R   1  |
        
        参数:
            radius_mm: 曲率半径 (mm)，正值表示凹面镜
        
        返回:
            反射后的 PilotBeamParams 对象
        
        **Validates: Requirements 8.3**
        """
        if np.isinf(radius_mm):
            return self  # 平面镜，无聚焦效果
        
        # 球面镜变换: 1/q_out = 1/q_in - 2/R
        A, B, C, D = 1, 0, -2/radius_mm, 1
        q_new = (A * self.q_parameter + B) / (C * self.q_parameter + D)
        return PilotBeamParams.from_q_parameter(q_new, self.wavelength_um)
    
    def apply_refraction(
        self,
        radius_mm: float,
        n1: float,
        n2: float,
    ) -> "PilotBeamParams":
        """折射面效果
        
        使用 ABCD 矩阵法计算折射后的光束参数。
        
        折射面的 ABCD 矩阵（从 n1 到 n2，曲率半径 R）:
            |     1           0      |
            | (n1-n2)/(n2*R)  n1/n2  |
        
        参数:
            radius_mm: 曲率半径 (mm)
            n1: 入射介质折射率
            n2: 出射介质折射率
        
        返回:
            折射后的 PilotBeamParams 对象
        
        **Validates: Requirements 8.3**
        """
        if np.isinf(radius_mm):
            A, B, C, D = 1, 0, 0, n1/n2
        else:
            A = 1
            B = 0
            C = (n1 - n2) / (n2 * radius_mm)
            D = n1 / n2
        
        q_new = (A * self.q_parameter + B) / (C * self.q_parameter + D)
        return PilotBeamParams.from_q_parameter(q_new, self.wavelength_um)
    
    def compute_phase_at_radius(self, r_mm: float) -> float:
        """计算指定半径处的 Pilot Beam 相位
        
        Pilot Beam 相位公式: φ(r) = k * r² / (2 * R)
        其中 k = 2π/λ，R 是曲率半径
        
        参数:
            r_mm: 到光轴的距离 (mm)
        
        返回:
            相位值 (弧度)
        
        **Validates: Requirements 8.5, 8.6**
        """
        if np.isinf(self.curvature_radius_mm):
            return 0.0
        
        wavelength_mm = self.wavelength_um * 1e-3
        k = 2 * np.pi / wavelength_mm
        
        phase = k * r_mm**2 / (2 * self.curvature_radius_mm)
        return phase
    
    def compute_phase_grid(
        self,
        grid_size: int,
        physical_size_mm: float,
    ) -> NDArray[np.floating]:
        """在网格上计算 Pilot Beam 参考相位
        
        Pilot Beam 相位是相对于主光线的相位延迟，公式：
            φ_pilot(r) = k × r² / (2 × R)
        
        参数:
            grid_size: 网格大小 (N × N)
            physical_size_mm: 物理尺寸（直径）(mm)
        
        返回:
            参考相位网格 (弧度)，形状 (grid_size, grid_size)
            主光线处（网格中心）相位为 0
        
        **Validates: Requirements 8.5, 8.6, 8.7**
        """
        half_size = physical_size_mm / 2
        coords = np.linspace(-half_size, half_size, grid_size)
        X, Y = np.meshgrid(coords, coords)
        r_sq = X**2 + Y**2  # mm²
        
        if np.isinf(self.curvature_radius_mm):
            return np.zeros((grid_size, grid_size))
        
        wavelength_mm = self.wavelength_um * 1e-3
        k = 2 * np.pi / wavelength_mm
        
        pilot_phase = k * r_sq / (2 * self.curvature_radius_mm)
        return pilot_phase
    
    @property
    def rayleigh_length_mm(self) -> float:
        """瑞利长度 (mm)"""
        wavelength_mm = self.wavelength_um * 1e-3
        return np.pi * self.waist_radius_mm**2 / wavelength_mm
    
    @property
    def divergence_rad(self) -> float:
        """远场发散角 (弧度)"""
        wavelength_mm = self.wavelength_um * 1e-3
        return wavelength_mm / (np.pi * self.waist_radius_mm)



@dataclass
class GridSampling:
    """网格采样信息
    
    存储波前网格的采样参数，用于确保不同模块间的网格一致性。
    
    属性:
        grid_size: 网格大小 (N × N)
        physical_size_mm: 物理尺寸（直径）(mm)
        sampling_mm: 采样间隔 (mm/pixel)
        beam_ratio: PROPER beam_ratio 参数
    
    **Validates: Requirements 17.1, 17.3, 17.4, 17.5, 17.6, 17.7**
    """
    grid_size: int
    physical_size_mm: float
    sampling_mm: float
    beam_ratio: float = 0.5
    
    @classmethod
    def from_proper(cls, wfo: Any) -> "GridSampling":
        """从 PROPER 波前对象提取采样信息
        
        参数:
            wfo: PROPER 波前对象
        
        返回:
            GridSampling 对象
        
        **Validates: Requirements 17.1**
        """
        import proper
        
        grid_size = proper.prop_get_gridsize(wfo)
        sampling_m = proper.prop_get_sampling(wfo)
        sampling_mm = sampling_m * 1e3
        physical_size_mm = sampling_mm * grid_size
        beam_ratio = wfo.beam_ratio if hasattr(wfo, 'beam_ratio') else 0.5
        
        return cls(
            grid_size=grid_size,
            physical_size_mm=physical_size_mm,
            sampling_mm=sampling_mm,
            beam_ratio=beam_ratio,
        )
    
    @classmethod
    def create(
        cls,
        grid_size: int,
        physical_size_mm: float,
        beam_ratio: float = 0.5,
    ) -> "GridSampling":
        """创建网格采样信息
        
        参数:
            grid_size: 网格大小
            physical_size_mm: 物理尺寸 (mm)
            beam_ratio: PROPER beam_ratio 参数
        
        返回:
            GridSampling 对象
        """
        sampling_mm = physical_size_mm / grid_size
        return cls(
            grid_size=grid_size,
            physical_size_mm=physical_size_mm,
            sampling_mm=sampling_mm,
            beam_ratio=beam_ratio,
        )
    
    def is_compatible(
        self,
        other: "GridSampling",
        tolerance: float = 0.01,
    ) -> bool:
        """检查两个采样信息是否兼容
        
        参数:
            other: 另一个 GridSampling
            tolerance: 相对容差
        
        返回:
            True 如果兼容
        
        **Validates: Requirements 17.6**
        """
        if self.grid_size != other.grid_size:
            return False
        
        size_diff = abs(self.physical_size_mm - other.physical_size_mm)
        if self.physical_size_mm > 0 and size_diff / self.physical_size_mm > tolerance:
            return False
        
        return True
    
    def get_coordinate_arrays(self) -> Tuple[NDArray, NDArray]:
        """获取坐标数组
        
        使用与 PROPER 一致的坐标系统：
        - 中心点（索引 n//2）对应坐标 0
        - 坐标 = (索引 - n//2) * 采样间隔
        
        返回:
            (X, Y) 网格坐标数组，单位 mm
        """
        n = self.grid_size
        coords = (np.arange(n) - n // 2) * self.sampling_mm
        X, Y = np.meshgrid(coords, coords)
        return X, Y



@dataclass
class PropagationState:
    """传播状态
    
    存储传播过程中某一位置的完整状态信息。
    
    重要：仿真波前使用振幅和相位分离存储！
    - amplitude: 振幅网格（实数，非负）
    - phase: 相位网格（实数，非折叠，单位：弧度）
    
    这避免了复数形式 exp(1j*φ) 隐含的相位折叠问题。
    相位可以是任意实数，不受 [-π, π] 限制。
    
    属性:
        surface_index: 表面索引
        position: 'entrance' 或 'exit' 或 'source'
        amplitude: 振幅网格（实数，非负）
        phase: 相位网格（实数，非折叠，单位：弧度）
        pilot_beam_params: Pilot Beam 参数
        proper_wfo: PROPER 波前对象
        optical_axis_state: 光轴状态
        grid_sampling: 网格采样信息
    
    **Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5**
    """
    amplitude: NDArray[np.floating]  # 振幅（实数，非负）
    phase: NDArray[np.floating]  # 相位（实数，非折叠，弧度）
    pilot_beam_params: PilotBeamParams
    proper_wfo: Any  # PROPER wavefront object
    optical_axis_state: Optional["OpticalAxisState"]
    grid_sampling: GridSampling
    
    surface_index: int = -1
    position: str = "source"  # "source", "entrance", "exit", "intermediate"
    current_refractive_index: float = 1.0 # 当前介质折射率
    
    def get_complex_amplitude(self) -> NDArray[np.complexfloating]:
        """获取复振幅分布 (折叠相位)
        
        A * exp(i * phi)
注意：返回的复振幅会有相位折叠，仅用于显示或与 PROPER 交互。
        内部计算应使用分离的 amplitude 和 phase。
        
        返回:
            复振幅数组
        """
        return self.amplitude * np.exp(1j * self.phase)
    
    # 向后兼容：提供 simulation_amplitude 属性
    @property
    def simulation_amplitude(self) -> NDArray[np.complexfloating]:
        """向后兼容：获取复振幅形式
        
        警告：此属性已废弃，请使用 amplitude 和 phase 分离存储。
        """
        import warnings
        warnings.warn(
            "simulation_amplitude 已废弃，请使用 amplitude 和 phase 分离存储",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_complex_amplitude()
    
    def validate_consistency(self, tolerance: float = 0.1) -> bool:
        """验证数据表示的一致性
        
        检查振幅/相位与 PROPER 对象在物理上是否等价。
        
        参数:
            tolerance: 相对容差
        
        返回:
            True 如果一致，False 否则
        
        **Validates: Requirements 10.2**
        """
        import proper
        
        # 从 PROPER 提取振幅和相位
        proper_amplitude = proper.prop_get_amplitude(self.proper_wfo)
        proper_phase = proper.prop_get_phase(self.proper_wfo)
        
        # 检查振幅一致性（在有效区域内）
        valid_mask = self.amplitude > 0.01 * np.max(self.amplitude)
        if np.sum(valid_mask) == 0:
            return True  # 无有效数据
        
        # 振幅比较
        amp_diff = np.abs(proper_amplitude[valid_mask] - self.amplitude[valid_mask])
        amp_max = np.max(self.amplitude[valid_mask])
        if amp_max > 0 and np.max(amp_diff) / amp_max > tolerance:
            return False
        
        # 相位比较（考虑 2π 周期性）
        # 使用 Pilot Beam 参考相位进行解包裹后比较
        pilot_phase = self.pilot_beam_params.compute_phase_grid(
            self.grid_sampling.grid_size,
            self.grid_sampling.physical_size_mm,
        )
        
        # 解包裹 PROPER 相位
        phase_diff = proper_phase - pilot_phase
        unwrapped_proper_phase = pilot_phase + np.angle(np.exp(1j * phase_diff))
        
        # 比较相位
        phase_error = np.abs(unwrapped_proper_phase[valid_mask] - self.phase[valid_mask])
        phase_error = np.minimum(phase_error, 2*np.pi - phase_error)
        
        if np.max(phase_error) > np.pi * tolerance:
            return False
        
        return True
    
    def get_intensity(self) -> NDArray[np.floating]:
        """获取光强分布
        
        返回:
            光强数组
        """
        return self.amplitude**2
    
    def get_phase(self) -> NDArray[np.floating]:
        """获取相位分布（非折叠）
        
        返回:
            相位数组 (弧度)
        """
        return self.phase
    
    def get_total_energy(self) -> float:
        """计算总能量
        
        返回:
            总能量（任意单位）
        """
        intensity = self.get_intensity()
        pixel_area = self.grid_sampling.sampling_mm**2
        return np.sum(intensity) * pixel_area



@dataclass
class SourceDefinition:
    """入射波面定义
    
    定义入射波面，包括理想高斯光束和初始像差。
    
    属性:
        wavelength_um: 波长 (μm)
        w0_mm: 束腰半径 (mm)
        z0_mm: 束腰位置（负值表示在光源前）(mm)
        initial_phase_aberration: 初始相位像差（可选，实数，弧度）
        initial_amplitude_aberration: 初始振幅像差（可选，实数，乘法因子）
        grid_size: 网格大小
        physical_size_mm: 物理尺寸 (mm)
        beam_diam_fraction: PROPER beam_diam_fraction 参数（可选）
            
            该参数控制 PROPER 中光束直径与网格宽度的比例。
            
            实际效果：
            - beam_diam_fraction = beam_diameter / grid_width
            - 其中 beam_diameter 是传给 prop_begin 的第一个参数
            - grid_width = physical_size_mm（网格物理尺寸）
            
            PROPER 内部使用：
            - ndiam = grid_n * beam_diam_fraction（光束直径对应的像素数）
            - 采样间隔 = beam_diameter / ndiam
            
            影响：
            - 较小的值（如 0.1-0.3）：光束占网格比例小，边缘采样更充分，
              适合需要观察远场衍射的情况
            - 较大的值（如 0.5-0.8）：光束占网格比例大，中心区域采样更密集，
              适合近场传播
            
            默认值 None 表示自动计算：beam_diam_fraction = 2*w0 / physical_size_mm
            
            有效范围：0.1 ~ 0.9
    
    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6**
    """
    wavelength_um: float
    w0_mm: float
    z0_mm: float = 0.0
    initial_phase_aberration: Optional[NDArray[np.floating]] = None
    initial_amplitude_aberration: Optional[NDArray[np.floating]] = None
    grid_size: int = 512
    physical_size_mm: float = 50.0
    beam_diam_fraction: Optional[float] = None
    
    # 向后兼容：支持旧的 initial_aberration 参数
    initial_aberration: Optional[NDArray[np.complexfloating]] = field(
        default=None, repr=False
    )
    
    def create_initial_wavefront(
        self,
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating], PilotBeamParams, Any]:
        """创建初始波前
        
        返回:
            (amplitude, phase, pilot_beam_params, proper_wfo)
            - amplitude: 振幅网格（实数，非负）
            - phase: 相位网格（实数，非折叠，弧度）
            - pilot_beam_params: Pilot Beam 参数
            - proper_wfo: PROPER 波前对象
        
        **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6**
        """
        import proper
        
        # 计算高斯光束参数
        wavelength_mm = self.wavelength_um * 1e-3
        wavelength_m = self.wavelength_um * 1e-6
        z_R_mm = np.pi * self.w0_mm**2 / wavelength_mm  # 瑞利长度 (mm)
        z_R_m = z_R_mm * 1e-3  # 瑞利长度 (m)
        z_mm = -self.z0_mm  # 当前位置相对于束腰的距离 (mm)
        z_m = z_mm * 1e-3  # (m)
        
        # 创建 PROPER 波前
        # beam_diameter = 2 * w0（PROPER 固定用法）
        beam_diameter_m = 2 * self.w0_mm * 1e-3
        
        # beam_diam_fraction = 0.5（PROPER 固定用法）
        wfo = proper.prop_begin(
            beam_diameter_m,
            wavelength_m,
            self.grid_size,
            0.5,
        )
        
        # 同步 PROPER 高斯光束参数（关键！）
        wfo.w0 = self.w0_mm * 1e-3  # 束腰半径 (m)
        wfo.z_Rayleigh = z_R_m  # 瑞利长度 (m)
        wfo.z = z_m  # 当前位置 (m)
        wfo.z_w0 = 0.0  # 束腰位置 (m)，假设束腰在原点
        
        # 确定参考面类型
        rayleigh_factor = proper.rayleigh_factor
        if abs(wfo.z - wfo.z_w0) < rayleigh_factor * wfo.z_Rayleigh:
            wfo.beam_type_old = "INSIDE_"
            wfo.reference_surface = "PLANAR"
        else:
            wfo.beam_type_old = "OUTSIDE"
            wfo.reference_surface = "SPHERI"
        
        # 创建坐标网格
        n = self.grid_size
        sampling_m = proper.prop_get_sampling(wfo)
        sampling_mm = sampling_m * 1e3
        half_size = sampling_mm * n / 2
        coords = np.linspace(-half_size, half_size, n)
        X, Y = np.meshgrid(coords, coords)
        R_sq = X**2 + Y**2
        
        # 光斑大小
        w = self.w0_mm * np.sqrt(1 + (z_mm / z_R_mm)**2) if z_R_mm > 0 else self.w0_mm
        
        # 曲率半径（严格公式）
        if abs(z_mm) < 1e-15:
            R = np.inf
        else:
            R = z_mm * (1 + (z_R_mm / z_mm)**2)
        
        # 高斯振幅（实数，非负）
        amplitude = np.exp(-R_sq / w**2)
        
        # 高斯相位（实数，非折叠，弧度）
        if np.isinf(R):
            phase = np.zeros_like(R_sq)
        else:
            k = 2 * np.pi / wavelength_mm
            phase = k * R_sq / (2 * R)
        
        # 应用初始像差
        if self.initial_phase_aberration is not None:
            phase = phase + self.initial_phase_aberration
        
        if self.initial_amplitude_aberration is not None:
            amplitude = amplitude * self.initial_amplitude_aberration
        
        # 向后兼容：处理旧的 initial_aberration 参数
        if self.initial_aberration is not None:
            import warnings
            warnings.warn(
                "initial_aberration 已废弃，请使用 initial_phase_aberration 和 "
                "initial_amplitude_aberration",
                DeprecationWarning,
            )
            amplitude = amplitude * np.abs(self.initial_aberration)
            phase = phase + np.angle(self.initial_aberration)
        
        # 将波前写入 PROPER
        # 对于 SPHERI 参考面，需要减去参考球面相位
        if wfo.reference_surface == "SPHERI":
            R_ref_m = wfo.z - wfo.z_w0
            r_sq_m = (X * 1e-3)**2 + (Y * 1e-3)**2
            k_m = 2 * np.pi / wavelength_m
            ref_phase = k_m * r_sq_m / (2 * R_ref_m)  # 正号
            residual_phase = phase - ref_phase * (wavelength_m / wavelength_mm)  # 单位转换
            # 实际上 ref_phase 已经是弧度，phase 也是弧度，直接相减
            # 但需要注意坐标单位：ref_phase 用的是 m，phase 用的是 mm
            # 重新计算：
            ref_phase_rad = k_m * r_sq_m / (2 * R_ref_m)
            residual_phase = phase - ref_phase_rad
            complex_amplitude = amplitude * np.exp(1j * residual_phase)
        else:
            complex_amplitude = amplitude * np.exp(1j * phase)
        
        wfo.wfarr = proper.prop_shift_center(complex_amplitude)
        
        # 创建 Pilot Beam 参数
        pilot_beam_params = PilotBeamParams.from_gaussian_source(
            self.wavelength_um, self.w0_mm, self.z0_mm
        )
        
        return amplitude, phase, pilot_beam_params, wfo
    
    def get_grid_sampling(self) -> GridSampling:
        """获取网格采样信息
        
        返回:
            GridSampling 对象
        """
        # 网格物理尺寸固定为 4 × w0（PROPER 固定用法）
        # 当 beam_diameter = 2*w0 且 beam_diam_fraction = 0.5 时：
        # dx = beam_diameter / (grid_n * 0.5) = 4*w0 / grid_n
        # physical_size = dx * grid_n = 4 * w0
        actual_physical_size_mm = 4 * self.w0_mm
        
        return GridSampling.create(
            grid_size=self.grid_size,
            physical_size_mm=actual_physical_size_mm,
            beam_ratio=0.5,  # 固定为 0.5
        )
