# -*- coding: utf-8 -*-
"""
光束参数测量与光阑设置模块

本模块提供：
1. D4sigma 光束直径测量（理想方法与 ISO 11146 标准方法）
2. M² 因子测量
3. 圆形光阑支持（硬边、高斯、超高斯/软边、8 阶软边）
4. 光束参数随传输距离变化的测量与分析

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6
"""

# 数据模型导出
from .data_models import (
    D4sigmaResult,
    ISOD4sigmaResult,
    M2Result,
    PowerTransmissionResult,
    ApertureType,
    PropagationDataPoint,
    PropagationAnalysisResult,
    ApertureEffectDataPoint,
    ApertureEffectAnalysisResult,
    ComparisonResult,
)

# 异常类导出
from .exceptions import (
    BeamMeasurementError,
    InvalidInputError,
    ConvergenceError,
    InsufficientDataError,
)

# 计算器类导出
from .d4sigma_calculator import D4sigmaCalculator
from .iso_d4sigma_calculator import ISOD4sigmaCalculator
from .m2_calculator import M2Calculator

# 光阑类导出
from .circular_aperture import CircularAperture

# 分析器类导出
from .beam_propagation_analyzer import BeamPropagationAnalyzer
from .aperture_effect_analyzer import ApertureEffectAnalyzer

# 对比模块导出
from .comparison_module import ComparisonModule

# 报告生成器导出
from .report_generator import ReportGenerator

# API 函数导出
from .api import (
    measure_beam_diameter,
    measure_m2,
    apply_aperture,
    analyze_aperture_effects,
)

__all__ = [
    # API 函数
    "measure_beam_diameter",
    "measure_m2",
    "apply_aperture",
    "analyze_aperture_effects",
    # 数据模型
    "D4sigmaResult",
    "ISOD4sigmaResult",
    "M2Result",
    "PowerTransmissionResult",
    "ApertureType",
    "PropagationDataPoint",
    "PropagationAnalysisResult",
    "ApertureEffectDataPoint",
    "ApertureEffectAnalysisResult",
    "ComparisonResult",
    # 异常类
    "BeamMeasurementError",
    "InvalidInputError",
    "ConvergenceError",
    "InsufficientDataError",
    # 计算器类
    "D4sigmaCalculator",
    "ISOD4sigmaCalculator",
    "M2Calculator",
    # 光阑类
    "CircularAperture",
    # 分析器类
    "BeamPropagationAnalyzer",
    "ApertureEffectAnalyzer",
    # 对比模块
    "ComparisonModule",
    # 报告生成器
    "ReportGenerator",
]
