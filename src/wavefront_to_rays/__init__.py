"""
波前采样为几何光线模块

本模块实现将入射波前复振幅采样为几何光线的功能，
用于混合光学仿真系统中物理光学与几何光学的接口。

主要功能：
1. 将波前复振幅转换为相位面
2. 使用 optiland 进行光线追迹
3. 输出出射光束用于后续计算
4. 元件光线追迹（反射镜、折射面）
5. 光线到波前复振幅重建（雅可比矩阵方法）
"""

from .wavefront_sampler import WavefrontToRaysSampler
from .phase_surface import create_phase_surface_optic
from .element_raytracer import (
    ElementRaytracer,
    SurfaceDefinition,
    create_mirror_surface,
    create_concave_mirror_for_spherical_wave,
    compute_rotation_matrix,
    transform_rays_to_global,
    transform_rays_to_local,
)
from .global_element_raytracer import (
    GlobalElementRaytracer,
    GlobalSurfaceDefinition,
    PlaneDef,
)
from .reconstructor import RayToWavefrontReconstructor
from .exceptions import (
    ReconstructionError,
    InsufficientRaysError,
)

__all__ = [
    # 波前采样器
    "WavefrontToRaysSampler",
    "create_phase_surface_optic",
    # 元件光线追迹器
    "ElementRaytracer",
    "SurfaceDefinition",
    "create_mirror_surface",
    "create_concave_mirror_for_spherical_wave",
    # 全局坐标系光线追迹器
    "GlobalElementRaytracer",
    "GlobalSurfaceDefinition",
    "PlaneDef",
    # 坐标转换辅助函数
    "compute_rotation_matrix",
    "transform_rays_to_global",
    "transform_rays_to_local",
    # 复振幅重建器
    "RayToWavefrontReconstructor",
    # 异常类
    "ReconstructionError",
    "InsufficientRaysError",
]
