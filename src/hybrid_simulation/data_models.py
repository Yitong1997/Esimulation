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
    """
    wavelength_um: float
    grid_size: int
    physical_size_mm: float
    num_rays: int = 200
    propagation_method: str = "local_raytracing"


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
