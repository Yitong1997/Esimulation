"""
高斯光束传输仿真模块

本模块实现基于 PROPER 物理光学传播和 optiland 几何光线追迹的
混合高斯光束传输仿真。

主要功能：
1. 高斯光束定义（束腰位置、M² 因子、波前误差）
2. 光学元件定义（反射镜、透镜）
3. 混合传输仿真（PROPER 传播 + optiland 光线追迹）
4. 结果分析与验证

作者：混合光学仿真项目
"""

from .gaussian_beam import GaussianBeam
from .optical_elements import (
    OpticalElement,
    ParabolicMirror,
    SphericalMirror,
    ThinLens,
)
from .hybrid_simulator import HybridGaussianBeamSimulator
from .abcd_calculator import ABCDCalculator

__all__ = [
    "GaussianBeam",
    "OpticalElement",
    "ParabolicMirror",
    "SphericalMirror",
    "ThinLens",
    "HybridGaussianBeamSimulator",
    "ABCDCalculator",
]
