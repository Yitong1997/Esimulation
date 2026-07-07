"""
混合光学传播系统 (Hybrid Optical Propagation System)

本模块实现完整的混合光学传播系统，将 PROPER 物理光学传输与 optiland 几何光线追迹相结合，
对导入的 Zemax 序列模式光路结构进行高精度仿真。

核心功能：
=========

1. **入射波面定义**：理想高斯光束 × 初始复振幅像差
2. **光轴追踪**：预先追踪整个光学系统的主光轴
3. **自由空间传播**：使用 PROPER 执行面与面之间的衍射传播
4. **混合元件传播**：在材质变化处执行波前-光线-波前重建
5. **相位解包裹**：使用 Pilot Beam 参考相位解包裹 PROPER 折叠相位
6. **数据一致性**：在整个传播过程中保持仿真复振幅、Pilot Beam 参数和 PROPER 对象的一致性

使用示例：
=========

    >>> from hybrid_optical_propagation import (
    ...     HybridOpticalPropagator,
    ...     SourceDefinition,
    ...     PilotBeamParams,
    ... )
    >>> from sequential_system import load_zmx_file
    >>> 
    >>> # 加载光学系统
    >>> optical_system = load_zmx_file("my_system.zmx")
    >>> 
    >>> # 定义入射波面
    >>> source = SourceDefinition(
    ...     wavelength_um=0.633,
    ...     w0_mm=5.0,
    ...     z0_mm=0.0,
    ...     grid_size=512,
    ...     physical_size_mm=50.0,
    ... )
    >>> 
    >>> # 创建传播器并执行传播
    >>> propagator = HybridOpticalPropagator(
    ...     optical_system=optical_system,
    ...     source=source,
    ...     wavelength_um=0.633,
    ... )
    >>> result = propagator.propagate()

作者：混合光学仿真项目
"""

from .data_models import (
    PilotBeamParams,
    GridSampling,
    PropagationState,
    SourceDefinition,
)

from .state_converter import StateConverter

from .exceptions import (
    HybridPropagationError,
    RayTracingError,
    PhaseUnwrappingError,
    MaterialError,
    GridSamplingError,
)

from .free_space_propagator import (
    FreeSpacePropagator,
    compute_propagation_distance,
)

from .material_detection import (
    detect_material_change,
    is_paraxial_surface,
    is_coordinate_break,
    normalize_material_name,
    classify_surface_interaction,
)

from .hybrid_element_propagator import HybridElementPropagator

from .hybrid_element_propagator_global import HybridElementPropagatorGlobal

from .paraxial_propagator import (
    ParaxialPhasePropagator,
    compute_paraxial_phase_correction,
)

from .hybrid_propagator import (
    HybridOpticalPropagator,
    PropagationResult,
)

from .unit_conversion import (
    mm_to_m,
    m_to_mm,
    um_to_mm,
    mm_to_um,
    um_to_m,
    m_to_um,
    opd_waves_to_phase_rad,
    phase_rad_to_opd_waves,
    opd_mm_to_phase_rad,
    phase_rad_to_opd_mm,
    wavenumber_mm,
    wavenumber_m,
    wavelength_um_to_mm,
    wavelength_um_to_m,
    UnitConverter,
)

from .zmx_integration import (
    load_optical_system_from_zmx,
    create_propagator_from_zmx,
    get_zmx_system_info,
    ZmxOpticalSystem,
)

from .validators import (
    ValidationResult,
    PhaseContinuityValidator,
    EnergyConservationValidator,
)

__all__ = [
    # 数据模型
    "PilotBeamParams",
    "GridSampling",
    "PropagationState",
    "SourceDefinition",
    # 状态转换器
    "StateConverter",
    # 传播器
    "FreeSpacePropagator",
    "compute_propagation_distance",
    "HybridElementPropagator",
    "HybridElementPropagatorGlobal",
    "ParaxialPhasePropagator",
    "compute_paraxial_phase_correction",
    "HybridOpticalPropagator",
    "PropagationResult",
    # 材质检测
    "detect_material_change",
    "is_paraxial_surface",
    "is_coordinate_break",
    "normalize_material_name",
    "classify_surface_interaction",
    # 异常类
    "HybridPropagationError",
    "RayTracingError",
    "PhaseUnwrappingError",
    "MaterialError",
    "GridSamplingError",
    # 单位转换
    "mm_to_m",
    "m_to_mm",
    "um_to_mm",
    "mm_to_um",
    "um_to_m",
    "m_to_um",
    "opd_waves_to_phase_rad",
    "phase_rad_to_opd_waves",
    "opd_mm_to_phase_rad",
    "phase_rad_to_opd_mm",
    "wavenumber_mm",
    "wavenumber_m",
    "wavelength_um_to_mm",
    "wavelength_um_to_m",
    "UnitConverter",
    # ZMX 集成
    "load_optical_system_from_zmx",
    "create_propagator_from_zmx",
    "get_zmx_system_info",
    "ZmxOpticalSystem",
    # 验证器
    "ValidationResult",
    "PhaseContinuityValidator",
    "EnergyConservationValidator",
]
