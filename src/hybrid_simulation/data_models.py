"""
统一仿真入口数据模型

定义仿真配置和结果存储所需的数据类。

注意：本模块复用 hybrid_optical_propagation 中的核心数据模型：
- PilotBeamParams: 直接复用
- GridSampling: 直接复用
- PropagationState: 直接复用
- SourceDefinition: 直接复用

本模块仅定义额外的封装类，用于提供更友好的用户接口。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
from numpy.typing import NDArray


@dataclass
class SimulationConfig:
    """仿真配置信息
    
    存储仿真的全局配置参数。
    
    属性:
        wavelength_um: 波长 (μm)
        grid_size: 网格大小 (N × N)
        physical_size_mm: 物理尺寸 (mm)
        num_rays: 光线采样数量
        propagation_method: 传播方法
        beam_diam_fraction: PROPER beam_diam_fraction 参数
            
            该参数控制 PROPER 中光束直径与网格宽度的比例。
            
            实际效果：
            - beam_diam_fraction = beam_diameter / grid_width
            - 其中 beam_diameter = 2 × w0（束腰直径）
            - grid_width = physical_size_mm（网格物理尺寸）
            
            影响：
            - 较小的值（如 0.1-0.3）：光束占网格比例小，边缘采样更充分，
              适合需要观察远场衍射的情况
            - 较大的值（如 0.5-0.8）：光束占网格比例大，中心区域采样更密集，
              适合近场传播
            
            默认值 None 表示自动计算：beam_diam_fraction = 2*w0 / physical_size_mm
            
            有效范围：0.1 ~ 0.9
    """
    wavelength_um: float
    grid_size: int
    physical_size_mm: float
    num_rays: int = 200
    propagation_method: str = "local_raytracing"
    beam_diam_fraction: Optional[float] = None


@dataclass
class SourceParams:
    """光源参数
    
    存储入射高斯光束的参数。
    
    属性:
        wavelength_um: 波长 (μm)
        w0_mm: 束腰半径 (mm)
        z0_mm: 束腰位置 (mm)，负值表示束腰在光源前
        z_rayleigh_mm: 瑞利长度 (mm)
        grid_size: 网格大小
        physical_size_mm: 物理尺寸 (mm)
    """
    wavelength_um: float
    w0_mm: float
    z0_mm: float
    z_rayleigh_mm: float
    grid_size: int
    physical_size_mm: float


@dataclass
class SurfaceGeometry:
    """表面几何信息
    
    存储光学表面的几何参数。
    
    属性:
        vertex_position: 顶点位置 [x, y, z] (mm)
        surface_normal: 表面法向量 [nx, ny, nz]
        radius: 曲率半径 (mm)，平面为 inf
        conic: 圆锥常数
        semi_aperture: 半口径 (mm)
        is_mirror: 是否为反射镜
    """
    vertex_position: NDArray[np.floating]
    surface_normal: NDArray[np.floating]
    radius: float
    conic: float
    semi_aperture: float
    is_mirror: bool


@dataclass
class OpticalAxisInfo:
    """光轴状态信息
    
    存储某位置处的光轴状态。
    
    属性:
        entrance_position: 入射点位置 (mm)
        entrance_direction: 入射方向
        exit_position: 出射点位置 (mm)
        exit_direction: 出射方向
        path_length: 累积光程 (mm)
    """
    entrance_position: NDArray[np.floating]
    entrance_direction: NDArray[np.floating]
    exit_position: NDArray[np.floating]
    exit_direction: NDArray[np.floating]
    path_length: float


# ============================================================================
# 调试数据类（用于 OAP 混合光线追迹调试）
# ============================================================================

@dataclass
class RayData:
    """光线数据
    
    存储一组光线的位置、方向和 OPD 信息。
    用于调试和验证光线追迹结果。
    
    属性:
        x: X 坐标数组 (mm)
        y: Y 坐标数组 (mm)
        z: Z 坐标数组 (mm)
        L: X 方向余弦数组
        M: Y 方向余弦数组
        N: Z 方向余弦数组
        opd: 光程差数组 (mm)
        intensity: 强度数组
    
    **Validates: Requirements 12.1**
    """
    x: NDArray[np.floating]
    y: NDArray[np.floating]
    z: NDArray[np.floating]
    L: NDArray[np.floating]
    M: NDArray[np.floating]
    N: NDArray[np.floating]
    opd: NDArray[np.floating]
    intensity: NDArray[np.floating]
    
    @property
    def num_rays(self) -> int:
        """光线数量"""
        return len(self.x)
    
    def get_positions(self) -> NDArray[np.floating]:
        """获取位置数组 (N, 3)"""
        return np.column_stack([self.x, self.y, self.z])
    
    def get_directions(self) -> NDArray[np.floating]:
        """获取方向数组 (N, 3)"""
        return np.column_stack([self.L, self.M, self.N])


@dataclass
class ChiefRayData:
    """主光线数据
    
    存储主光线的几何信息，用于验证光线追迹的正确性。
    
    属性:
        entrance_position: 入射面位置 (x, y, z) (mm)
        entrance_direction: 入射方向 (L, M, N)
        intersection_point: 与表面交点 (x, y, z) (mm)
        exit_position: 出射面位置 (x, y, z) (mm)
        exit_direction: 出射方向 (L, M, N)
        surface_normal: 交点处表面法向量 (nx, ny, nz)
    
    **Validates: Requirements 12.1**
    """
    entrance_position: tuple
    entrance_direction: tuple
    intersection_point: tuple
    exit_position: tuple
    exit_direction: tuple
    surface_normal: tuple


@dataclass
class CoordinateSystemData:
    """坐标系数据
    
    存储入射面或出射面的坐标系信息。
    
    属性:
        origin: 原点位置（全局坐标）(x, y, z) (mm)
        rotation_matrix: 3x3 旋转矩阵（从局部到全局）
        z_axis: Z 轴方向（光轴方向）(L, M, N)
    
    **Validates: Requirements 12.1**
    """
    origin: tuple
    rotation_matrix: NDArray[np.floating]
    z_axis: tuple


@dataclass
class PilotBeamParamsData:
    """Pilot Beam 参数数据
    
    存储 Pilot Beam 的参数，用于调试验证。
    
    属性:
        wavelength_um: 波长 (μm)
        waist_radius_mm: 束腰半径 (mm)
        waist_position_mm: 束腰位置 (mm)
        curvature_radius_mm: 曲率半径 (mm)
        spot_size_mm: 光斑大小 (mm)
    
    **Validates: Requirements 12.1**
    """
    wavelength_um: float
    waist_radius_mm: float
    waist_position_mm: float
    curvature_radius_mm: float
    spot_size_mm: float
    
    @classmethod
    def from_pilot_beam_params(cls, params) -> "PilotBeamParamsData":
        """从 PilotBeamParams 创建
        
        参数:
            params: PilotBeamParams 对象
        
        返回:
            PilotBeamParamsData 对象
        """
        return cls(
            wavelength_um=params.wavelength_um,
            waist_radius_mm=params.waist_radius_mm,
            waist_position_mm=params.waist_position_mm,
            curvature_radius_mm=params.curvature_radius_mm,
            spot_size_mm=params.spot_size_mm,
        )
