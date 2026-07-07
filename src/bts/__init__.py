"""
BTS - 混合光学仿真系统 MATLAB 风格 API

本模块提供简洁、直观的 MATLAB 风格 API，用于混合光学仿真。

核心函数：
- load_zmx(path): 从 ZMX 文件加载光学系统
- simulate(system, source): 执行混合光学仿真

核心类：
- OpticalSystem: 光学系统定义
- GaussianSource: 高斯光源定义

结果类（重导出）：
- SimulationResult: 仿真结果容器
- WavefrontData: 波前数据结构
- SurfaceRecord: 表面记录

使用示例：
=========

    >>> import bts
    >>> 
    >>> # 方式 1：从 ZMX 文件加载
    >>> system = bts.load_zmx("system.zmx")
    >>> 
    >>> # 方式 2：手动定义
    >>> system = bts.OpticalSystem("My System")
    >>> system.add_flat_mirror(z=50, tilt_x=45)
    >>> 
    >>> # 定义光源
    >>> source = bts.GaussianSource(wavelength_um=0.633, w0_mm=5.0)
    >>> 
    >>> # 执行仿真
    >>> result = bts.simulate(system, source)
    >>> 
    >>> # 查看结果
    >>> result.summary()
    >>> result.plot_all()

作者：混合光学仿真项目
"""

# ============================================================
# 重导出现有类（来自 hybrid_simulation 模块）
# ============================================================
from hybrid_simulation import (
    SimulationResult,
    WavefrontData,
    SurfaceRecord,
)

# ============================================================
# 异常类
# ============================================================
from .exceptions import (
    ParseError,
    ConfigurationError,
    SimulationError,
)

# ============================================================
# 导出 bts 模块的公共 API
# ============================================================
# 注意：以下模块将在后续任务中实现
# 当前先导入占位，确保模块结构正确

# I/O 函数
from .io import load_zmx

# 核心类
from .optical_system import OpticalSystem
from .source import GaussianSource

# 仿真函数
from .simulation import simulate

# ============================================================
# 光束测量与光阑 API
# ============================================================
from .beam_measurement import (
    measure_beam_diameter,
    measure_m2,
    apply_aperture,
    analyze_aperture_effects,
    # 数据模型
    D4sigmaResult,
    ISOD4sigmaResult,
    M2Result,
    ApertureType,
    ApertureEffectAnalysisResult,
)

# ============================================================
# 公共 API 列表
# ============================================================
__all__ = [
    # 核心函数
    "load_zmx",
    "simulate",
    # 核心类
    "OpticalSystem",
    "GaussianSource",
    # 重导出的结果类
    "SimulationResult",
    "WavefrontData",
    "SurfaceRecord",
    # 异常类
    "ParseError",
    "ConfigurationError",
    "SimulationError",
    # 光束测量与光阑 API
    "measure_beam_diameter",
    "measure_m2",
    "apply_aperture",
    "analyze_aperture_effects",
    # 光束测量数据模型
    "D4sigmaResult",
    "ISOD4sigmaResult",
    "M2Result",
    "ApertureType",
    "ApertureEffectAnalysisResult",
]
