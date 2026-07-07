# -*- coding: utf-8 -*-
"""
Angular Spectrum Method (角谱法) Python 实现

本包提供了完整的角谱法光学衍射传播计算功能，包含 6 种不同的角谱传播方法：
- ASM: 基础角谱法
- BandLimitedASM: 带限角谱法
- ScalableASM: 可缩放角谱法
- ScaledASM: 缩放角谱法
- ShiftedASM: 偏移角谱法
- TiltedASM: 倾斜角谱法

该实现与 Julia AngularSpectrumMethod 包保持数值一致性。

示例用法：
    >>> import numpy as np
    >>> from angular_spectrum_method import asm
    >>> 
    >>> # 创建输入光场
    >>> u = np.ones((64, 64), dtype=complex)
    >>> 
    >>> # 设置参数
    >>> wavelength = 633e-9  # 波长 633nm
    >>> dx = dy = 1e-6       # 采样间隔 1μm
    >>> z = 0.01             # 传播距离 1cm
    >>> 
    >>> # 计算传播
    >>> result = asm(u, wavelength, dx, dy, z)

作者：Angular Spectrum Method Python Team
许可证：MIT
"""

__version__ = "0.1.0"
__author__ = "Angular Spectrum Method Python Team"

# 核心数学函数
from .core import rect, spatial_frequencies, transfer_function

# 工具函数
from .utils import select_region, select_region_view

# 基础角谱法
from .asm import asm, asm_

# 带限角谱法
from .band_limited_asm import band_limited_asm, band_limited_asm_

# 可缩放角谱法
from .scalable_asm import (
    scalable_asm, 
    scalable_asm_,
    distance_limit,
    minimum_distance,
    band_limit,
    PropagationWarning
)

# 缩放角谱法
from .scaled_asm import scaled_asm, scaled_asm_

# 偏移角谱法
from .shifted_asm import shifted_asm, shifted_asm_, center_frequency, bandwidth

# 倾斜角谱法
from .tilted_asm import tilted_asm, tilted_asm_, compute_sdc

# 输入验证（从 input_validation.py 模块导入）
from .input_validation import (
    ValidationError,
    validate_field,
    validate_positive,
    validate_non_negative,
    validate_rotation_matrix,
    validate_scale_factor
)

__all__ = [
    "__version__",
    # 核心数学函数
    "rect",
    "spatial_frequencies", 
    "transfer_function",
    # 工具函数
    "select_region",
    "select_region_view",
    # 基础角谱法
    "asm",
    "asm_",
    # 带限角谱法
    "band_limited_asm",
    "band_limited_asm_",
    # 可缩放角谱法
    "scalable_asm",
    "scalable_asm_",
    "distance_limit",
    "minimum_distance",
    "band_limit",
    "PropagationWarning",
    # 缩放角谱法
    "scaled_asm",
    "scaled_asm_",
    # 偏移角谱法
    "shifted_asm",
    "shifted_asm_",
    "center_frequency",
    "bandwidth",
    # 倾斜角谱法
    "tilted_asm",
    "tilted_asm_",
    "compute_sdc",
    # 输入验证
    "ValidationError",
    "validate_field",
    "validate_positive",
    "validate_non_negative",
    "validate_rotation_matrix",
    "validate_scale_factor",
]
