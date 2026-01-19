"""
波前采样为几何光线模块

本模块实现将入射波前复振幅采样为几何光线的功能，
用于混合光学仿真系统中物理光学与几何光学的接口。

主要功能：
1. 将波前复振幅转换为相位面
2. 使用 optiland 进行光线追迹
3. 输出出射光束用于后续计算
4. 元件光线追迹（反射镜、折射面）
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

__all__ = [
    # 波前采样器
    "WavefrontToRaysSampler",
    "create_phase_surface_optic",
    # 元件光线追迹器
    "ElementRaytracer",
    "SurfaceDefinition",
    "create_mirror_surface",
    "create_concave_mirror_for_spherical_wave",
    # 坐标转换辅助函数
    "compute_rotation_matrix",
    "transform_rays_to_global",
    "transform_rays_to_local",
]
