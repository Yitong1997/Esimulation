"""
混合元件传播模块 (Hybrid Element Propagation)

本模块实现在光学元器件处的完整波前-光线-波前重建流程。
是混合光学仿真系统的核心组件，结合物理光学传播和 optiland 几何光线追迹，
实现高精度的波前计算。

传播模式：
==========

1. **直接传播模式 (Direct Mode)** - 默认推荐
   
   理论上正确的传播流程：
   - 入射面：元件顶点，垂直于入射光主光轴
   - 出射面：元件顶点，垂直于出射光主光轴
   
   数据流：
       入射面复振幅 → [波前采样] → 光线 → [元件追迹] → 出射面复振幅重建
   
   使用 DirectElementPropagator 类。

2. **切平面传播模式 (Tangent Plane Mode)** - 原有实现，暂不推荐
   
   通过切平面中转的传播流程：
   
   数据流：
       入射面 → [Tilted ASM] → 切平面 → [采样] → 光线 
       → [追迹] → [重建] → 切平面 → [Tilted ASM] → 出射面
   
   使用 HybridElementPropagator 类。

使用示例（推荐的直接传播模式）：
    >>> from hybrid_propagation import (
    ...     DirectElementPropagator,
    ...     propagate_through_element,
    ... )
    >>> from wavefront_to_rays import SurfaceDefinition
    >>> import numpy as np
    >>> 
    >>> # 创建输入复振幅
    >>> grid_size = 64
    >>> amplitude = np.ones((grid_size, grid_size), dtype=complex)
    >>> 
    >>> # 定义光学元件（平面镜）
    >>> element = SurfaceDefinition(
    ...     surface_type='mirror',
    ...     radius=np.inf,
    ...     tilt_x=0.0,
    ...     tilt_y=0.0,
    ...     semi_aperture=25.0,
    ... )
    >>> 
    >>> # 方式 1：使用便捷函数
    >>> result = propagate_through_element(
    ...     complex_amplitude=amplitude,
    ...     element=element,
    ...     wavelength=0.633,
    ...     physical_size=50.0,
    ... )
    >>> output_amplitude = result.output_amplitude
    >>> 
    >>> # 方式 2：使用类（更多控制选项）
    >>> propagator = DirectElementPropagator(
    ...     complex_amplitude=amplitude,
    ...     element=element,
    ...     wavelength=0.633,
    ...     physical_size=50.0,
    ...     num_rays=200,
    ...     use_pilot_beam=True,
    ...     debug=True,
    ... )
    >>> result = propagator.propagate()

复用的现有模块：
- WavefrontToRaysSampler: 波前采样为几何光线
- ElementRaytracer: 元件光线追迹
- SurfaceDefinition: 光学元件定义
- tilted_asm: 倾斜平面角谱传播（仅切平面模式使用）

作者：混合光学仿真项目
"""

# 导入数据类（用于验证结果）
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from numpy.typing import NDArray

# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class ValidationResult:
    """单项验证结果
    
    属性:
        passed: 验证是否通过
        message: 验证结果描述信息
        value: 实际测量值（可选）
        threshold: 阈值（可选）
    """
    passed: bool
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class PilotBeamValidationResult:
    """Pilot Beam 验证结果
    
    属性:
        is_valid: 整体验证是否通过
        phase_sampling: 相位采样验证结果
        beam_divergence: 光束发散角验证结果
        beam_size_match: 光束尺寸匹配验证结果
        max_phase_gradient: 最大相位梯度
        mean_phase_gradient: 平均相位梯度
        warnings: 警告信息列表
    """
    is_valid: bool
    phase_sampling: ValidationResult
    beam_divergence: ValidationResult
    beam_size_match: ValidationResult
    max_phase_gradient: float
    mean_phase_gradient: float
    warnings: List[str]


@dataclass
class PropagationInput:
    """传播输入数据
    
    属性:
        complex_amplitude: 入射面复振幅，形状 (N, N)
        wavelength: 波长，单位 μm
        physical_size: 物理尺寸（直径），单位 mm
        element: 元件定义 (SurfaceDefinition)
        entrance_direction: 入射方向 (L, M, N)
        entrance_position: 入射面位置 (x, y, z)，单位 mm
    """
    complex_amplitude: NDArray[np.complexfloating]
    wavelength: float
    physical_size: float
    element: Any  # SurfaceDefinition，避免循环导入
    entrance_direction: Tuple[float, float, float]
    entrance_position: Tuple[float, float, float]


@dataclass
class PropagationOutput:
    """传播输出数据
    
    属性:
        complex_amplitude: 出射面复振幅
        exit_direction: 出射方向 (L, M, N)
        exit_position: 出射面位置 (x, y, z)，单位 mm
        validation_result: Pilot Beam 验证结果
    """
    complex_amplitude: NDArray[np.complexfloating]
    exit_direction: Tuple[float, float, float]
    exit_position: Tuple[float, float, float]
    validation_result: PilotBeamValidationResult


@dataclass
class IntermediateResults:
    """中间结果
    
    属性:
        tangent_amplitude_in: 切平面输入复振幅
        rays_in: 输入光线 (RealRays)
        rays_out: 输出光线 (RealRays)
        pilot_phase: Pilot Beam 参考相位
        residual_phase: 残差相位
        tangent_amplitude_out: 切平面输出复振幅
    """
    tangent_amplitude_in: NDArray[np.complexfloating]
    rays_in: Any  # RealRays，避免循环导入
    rays_out: Any  # RealRays，避免循环导入
    pilot_phase: NDArray[np.floating]
    residual_phase: NDArray[np.floating]
    tangent_amplitude_out: NDArray[np.complexfloating]


# ============================================================================
# 模块导入（延迟导入以避免循环依赖）
# ============================================================================

# 注意：以下导入在模块实现后取消注释
# 目前使用占位符，待各子模块实现后启用

# 倾斜传播模块（已实现）
from .tilted_propagation import TiltedPropagation

# Pilot Beam 模块（已实现）
from .pilot_beam import PilotBeamCalculator, PilotBeamValidator

# 相位修正模块（已实现）
from .phase_correction import PhaseCorrector

# 复振幅重建模块（已实现）
from .amplitude_reconstruction import AmplitudeReconstructor

# 切平面传播器类（原有实现，暂不推荐）
from .propagator import HybridElementPropagator

# 直接传播器类（推荐，默认）
from .direct_propagator import (
    DirectElementPropagator,
    DirectPropagationResult,
    propagate_through_element,
)

# 默认传播器别名（指向推荐的直接传播模式）
ElementPropagator = DirectElementPropagator


# ============================================================================
# 公共 API 导出列表
# ============================================================================

__all__ = [
    # 数据类
    "ValidationResult",
    "PilotBeamValidationResult",
    "PropagationInput",
    "PropagationOutput",
    "IntermediateResults",
    "DirectPropagationResult",
    # 倾斜传播
    "TiltedPropagation",
    # Pilot Beam 模块
    "PilotBeamCalculator",
    "PilotBeamValidator",
    # 相位修正
    "PhaseCorrector",
    # 复振幅重建
    "AmplitudeReconstructor",
    # 直接传播器（推荐，默认）
    "DirectElementPropagator",
    "propagate_through_element",
    "ElementPropagator",  # 默认别名
    # 切平面传播器（原有实现，暂不推荐）
    "HybridElementPropagator",
]
