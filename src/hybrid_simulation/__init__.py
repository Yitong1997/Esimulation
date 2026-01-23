"""
统一仿真入口模块 (Unified Simulation Entry)

本模块提供简洁的混合光学仿真 API，使主程序代码极简（< 10 行）。

核心类：
- HybridSimulator: 统一仿真入口，提供步骤化 API
- SimulationResult: 全面的结果容器
- SurfaceRecord: 单个表面的完整记录
- WavefrontData: 波前数据结构

使用示例：
=========

    >>> from hybrid_simulation import HybridSimulator
    >>> 
    >>> # 步骤 1：创建仿真器
    >>> sim = HybridSimulator()
    >>> 
    >>> # 步骤 2：加载光学系统
    >>> sim.load_zmx("system.zmx")  # 或 sim.add_flat_mirror(z=50, tilt_x=45)
    >>> 
    >>> # 步骤 3：定义光源
    >>> sim.set_source(wavelength_um=0.55, w0_mm=5.0, grid_size=256)
    >>> 
    >>> # 步骤 4：执行仿真
    >>> result = sim.run()
    >>> 
    >>> # 步骤 5：查看/保存结果
    >>> result.summary()
    >>> result.plot_all()
    >>> result.save("output/")

作者：混合光学仿真项目
"""

from .simulator import HybridSimulator
from .result import SimulationResult, SurfaceRecord, WavefrontData
from .data_models import (
    SimulationConfig,
    SourceParams,
    SurfaceGeometry,
    OpticalAxisInfo,
)
from .exceptions import ConfigurationError, SimulationError

__all__ = [
    # 主类
    "HybridSimulator",
    "SimulationResult",
    "SurfaceRecord",
    "WavefrontData",
    # 数据模型
    "SimulationConfig",
    "SourceParams",
    "SurfaceGeometry",
    "OpticalAxisInfo",
    # 异常
    "ConfigurationError",
    "SimulationError",
]
