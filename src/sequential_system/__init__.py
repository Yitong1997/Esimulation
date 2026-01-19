"""
序列光学系统模块

本模块提供类似 Zemax 序列模式的接口，用于定义和仿真光学系统。
结合 PROPER 物理光学衍射传播和 optiland 几何光线追迹，实现混合光学仿真。

主要功能：
1. 定义初始高斯光束参数
2. 按顺序定义光学面（面型、间距、倾斜等）
3. 定义采样面位置
4. 执行混合光学仿真
5. 可视化光路和输出仿真结果

使用示例：
    >>> from sequential_system import (
    ...     SequentialOpticalSystem,
    ...     GaussianBeamSource,
    ...     SphericalMirror,
    ...     SamplingPlane,
    ... )
    >>> 
    >>> # 定义光源
    >>> source = GaussianBeamSource(
    ...     wavelength=0.633,  # μm
    ...     w0=1.0,            # mm
    ...     z0=-50.0,          # mm，束腰在光源前 50mm
    ... )
    >>> 
    >>> # 创建系统并添加光学面
    >>> system = SequentialOpticalSystem(source)
    >>> system.add_surface(SphericalMirror(
    ...     radius_of_curvature=200.0,  # mm
    ...     thickness=100.0,            # mm
    ...     semi_aperture=15.0,
    ... ))
    >>> 
    >>> # 添加采样面并运行仿真
    >>> system.add_sampling_plane(distance=150.0, name="focus")
    >>> results = system.run()

作者：混合光学仿真项目
"""

# 异常类
from .exceptions import (
    SequentialSystemError,
    SourceConfigurationError,
    SurfaceConfigurationError,
    SimulationError,
    SamplingError,
)

# ZMX 解析器相关
from .zmx_parser import (
    ZmxParser,
    ZmxDataModel,
    ZmxSurfaceData,
    ZmxParseError,
    ZmxUnsupportedError,
    ZmxConversionError,
)

# ZMX 转换器相关
from .zmx_converter import (
    ElementConverter,
    ConvertedElement,
    CoordinateTransform,
    CodeGenerator,
    load_zmx_file,
    load_zmx_and_generate_code,
)

# 光源类
from .source import GaussianBeamSource

# 系统类
from .system import SequentialOpticalSystem

# 采样类
from .sampling import SamplingPlane, SamplingResult, SimulationResults

# 可视化类
from .visualization import LayoutVisualizer

# 坐标跟踪类
from .coordinate_tracking import (
    OpticalAxisTracker,
    OpticalAxisState,
    RayDirection,
    Position3D,
    LocalCoordinateSystem,
)

# 从 gaussian_beam_simulation 重新导出光学元件类
from gaussian_beam_simulation.optical_elements import (
    OpticalElement,
    ParabolicMirror,
    SphericalMirror,
    ThinLens,
    FlatMirror,
)

# 公共 API 导出列表
__all__ = [
    # 异常类
    "SequentialSystemError",
    "SourceConfigurationError",
    "SurfaceConfigurationError",
    "SimulationError",
    "SamplingError",
    # ZMX 解析器相关
    "ZmxParser",
    "ZmxDataModel",
    "ZmxSurfaceData",
    "ZmxParseError",
    "ZmxUnsupportedError",
    "ZmxConversionError",
    # ZMX 转换器相关
    "ElementConverter",
    "ConvertedElement",
    "CoordinateTransform",
    "CodeGenerator",
    "load_zmx_file",
    "load_zmx_and_generate_code",
    # 光源类
    "GaussianBeamSource",
    # 系统类
    "SequentialOpticalSystem",
    # 采样类
    "SamplingPlane",
    "SamplingResult",
    "SimulationResults",
    # 可视化类
    "LayoutVisualizer",
    # 坐标跟踪类
    "OpticalAxisTracker",
    "OpticalAxisState",
    "RayDirection",
    "Position3D",
    "LocalCoordinateSystem",
    # 光学元件类（从 gaussian_beam_simulation 重新导出）
    "OpticalElement",
    "ParabolicMirror",
    "SphericalMirror",
    "ThinLens",
    "FlatMirror",
]
