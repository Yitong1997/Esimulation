# -*- coding: utf-8 -*-
"""
光束测量数据模型

本模块定义光束测量相关的数据类和枚举类型。

Requirements: 1.5, 2.4, 3.3, 4.10
"""

from dataclasses import dataclass
from typing import Optional, List
from enum import Enum
import numpy as np


class ApertureType(Enum):
    """光阑类型枚举
    
    定义四种圆形光阑的振幅透过率设置方法：
    - HARD_EDGE: 硬边光阑，使用 PROPER 的 prop_circular_aperture 实现
    - GAUSSIAN: 高斯光阑，透过率按 T(r) = exp(-0.5 × (r/σ)²) 分布
    - SUPER_GAUSSIAN: 超高斯/软边光阑，透过率按 T(r) = exp(-(r/r₀)ⁿ) 分布
    - EIGHTH_ORDER: 8 阶软边光阑，使用 PROPER 的 prop_8th_order_mask 实现
    
    Requirements: 4.1, 4.2, 4.3, 4.4
    """
    HARD_EDGE = "hard_edge"           # 硬边光阑
    GAUSSIAN = "gaussian"             # 高斯光阑
    SUPER_GAUSSIAN = "super_gaussian" # 超高斯/软边光阑
    EIGHTH_ORDER = "eighth_order"     # 8 阶软边光阑


@dataclass
class D4sigmaResult:
    """D4sigma 测量结果
    
    存储使用理想二阶矩方法测量的光束直径结果。
    D4sigma 定义为光束强度分布二阶矩的 4 倍标准差。
    
    对于理想高斯光束，D4sigma = 2×w（1/e² 半径的 2 倍）。
    
    Attributes:
        dx: X 方向直径 (m)
        dy: Y 方向直径 (m)
        d_mean: 平均直径 (m)，计算为 (dx + dy) / 2
        centroid_x: 质心 X 坐标 (m)
        centroid_y: 质心 Y 坐标 (m)
        total_power: 总功率（归一化）
    
    Requirements: 1.5
    """
    dx: float           # X 方向直径 (m)
    dy: float           # Y 方向直径 (m)
    d_mean: float       # 平均直径 (m)
    centroid_x: float   # 质心 X 坐标 (m)
    centroid_y: float   # 质心 Y 坐标 (m)
    total_power: float  # 总功率（归一化）


@dataclass
class ISOD4sigmaResult(D4sigmaResult):
    """ISO 11146 标准 D4sigma 测量结果
    
    继承自 D4sigmaResult，增加 ISO 标准方法特有的信息。
    ISO 11146 标准方法包括：
    1. 背景噪声估计与去除
    2. 迭代 ROI 方法确定有效测量区域
    3. 使用 3 倍 D4sigma 作为 ROI 边界进行迭代
    
    Attributes:
        dx: X 方向直径 (m)（继承自 D4sigmaResult）
        dy: Y 方向直径 (m)（继承自 D4sigmaResult）
        d_mean: 平均直径 (m)（继承自 D4sigmaResult）
        centroid_x: 质心 X 坐标 (m)（继承自 D4sigmaResult）
        centroid_y: 质心 Y 坐标 (m)（继承自 D4sigmaResult）
        total_power: 总功率（继承自 D4sigmaResult）
        iterations: 迭代次数
        converged: 是否收敛
        roi_radius: 最终 ROI 半径 (m)
        background_level: 背景噪声水平
        warning: 警告信息（如迭代未收敛时）
    
    Requirements: 2.4, 2.5
    """
    iterations: int         # 迭代次数
    converged: bool         # 是否收敛
    roi_radius: float       # 最终 ROI 半径 (m)
    background_level: float # 背景噪声水平
    warning: Optional[str]  # 警告信息


@dataclass
class M2Result:
    """M² 测量结果
    
    存储通过多点光束直径拟合计算的 M² 因子结果。
    M² 因子表征实际光束与理想高斯光束的偏离程度。
    
    拟合公式：w(z)² = w₀² × [1 + (M² × λ × (z-z₀) / (π × w₀²))²]
    
    Attributes:
        m2_x: X 方向 M² 因子
        m2_y: Y 方向 M² 因子
        m2_mean: 平均 M² 因子，计算为 (m2_x + m2_y) / 2
        w0_x: X 方向拟合束腰 (m)
        w0_y: Y 方向拟合束腰 (m)
        z0_x: X 方向束腰位置 (m)
        z0_y: Y 方向束腰位置 (m)
        r_squared_x: X 方向拟合优度 R²
        r_squared_y: Y 方向拟合优度 R²
        wavelength: 波长 (m)
        warning: 警告信息（如测量点数不足时）
    
    Requirements: 3.3, 3.4
    """
    m2_x: float           # X 方向 M² 因子
    m2_y: float           # Y 方向 M² 因子
    m2_mean: float        # 平均 M² 因子
    w0_x: float           # X 方向拟合束腰 (m)
    w0_y: float           # Y 方向拟合束腰 (m)
    z0_x: float           # X 方向束腰位置 (m)
    z0_y: float           # Y 方向束腰位置 (m)
    r_squared_x: float    # X 方向拟合优度
    r_squared_y: float    # Y 方向拟合优度
    wavelength: float     # 波长 (m)
    warning: Optional[str] # 警告信息


@dataclass
class PowerTransmissionResult:
    """能量透过率计算结果
    
    存储高斯光束通过光阑后的能量透过率计算结果。
    
    对于硬边圆形光阑，理论透过率公式为：
    T = 1 - exp(-2 × (a/w)²)
    其中 a 为光阑半径，w 为光束半径。
    
    Attributes:
        actual_transmission: 实际透过率（通过数值积分计算）
        theoretical_transmission: 理论透过率（通过解析公式计算）
        relative_error: 相对误差，计算为 |actual - theoretical| / theoretical
        input_power: 输入功率（归一化）
        output_power: 输出功率（归一化）
    
    Requirements: 4.10, 4.11
    """
    actual_transmission: float      # 实际透过率
    theoretical_transmission: float # 理论透过率
    relative_error: float           # 相对误差
    input_power: float              # 输入功率
    output_power: float             # 输出功率


# ============================================================================
# 辅助数据模型（用于传播分析和光阑影响分析）
# ============================================================================

@dataclass
class PropagationDataPoint:
    """单个传输位置的测量数据
    
    存储在特定传输位置测量的光束参数。
    
    Attributes:
        z: 传输位置 (m)
        dx: X 方向直径 (m)
        dy: Y 方向直径 (m)
        d_mean: 平均直径 (m)
        method: 测量方法（"ideal" 或 "iso"）
    
    Requirements: 5.1, 5.3
    """
    z: float              # 传输位置 (m)
    dx: float             # X 方向直径 (m)
    dy: float             # Y 方向直径 (m)
    d_mean: float         # 平均直径 (m)
    method: str           # 测量方法


@dataclass
class PropagationAnalysisResult:
    """传播分析结果
    
    存储光束参数随传输距离变化的完整分析结果。
    
    Attributes:
        data_points: 测量数据点列表
        divergence_x: X 方向远场发散角 (rad)
        divergence_y: Y 方向远场发散角 (rad)
        divergence_mean: 平均远场发散角 (rad)
        wavelength: 波长 (m)
        w0: 初始束腰 (m)
    
    Requirements: 5.1, 5.2, 5.3, 5.4
    """
    data_points: List[PropagationDataPoint]  # 测量数据点
    divergence_x: float                       # X 方向远场发散角 (rad)
    divergence_y: float                       # Y 方向远场发散角 (rad)
    divergence_mean: float                    # 平均远场发散角 (rad)
    wavelength: float                         # 波长 (m)
    w0: float                                 # 初始束腰 (m)


@dataclass
class ApertureEffectDataPoint:
    """单个光阑配置的分析数据
    
    存储特定光阑类型和尺寸配置下的分析结果。
    
    Attributes:
        aperture_type: 光阑类型
        aperture_ratio: 光阑半径/光束半径
        power_transmission: 功率透过率
        beam_diameter_change: 光束直径变化率（相对于无光阑情况）
        divergence_change: 发散角变化率（相对于无光阑情况）
        theoretical_transmission: 理论透过率
    
    Requirements: 6.3, 6.4, 6.5
    """
    aperture_type: ApertureType
    aperture_ratio: float           # 光阑半径/光束半径
    power_transmission: float       # 功率透过率
    beam_diameter_change: float     # 光束直径变化率
    divergence_change: float        # 发散角变化率
    theoretical_transmission: float # 理论透过率


@dataclass
class ApertureEffectAnalysisResult:
    """光阑影响分析结果
    
    存储不同光阑类型和尺寸对光束传输影响的完整分析结果。
    
    Attributes:
        data_points: 各光阑配置的分析数据点列表
        aperture_types: 分析的光阑类型列表
        aperture_ratios: 分析的光阑半径/光束半径列表
        wavelength: 波长 (m)
        w0: 束腰半径 (m)
        recommendation: 光阑选型建议
    
    Requirements: 6.1, 6.2, 6.6, 6.7
    """
    data_points: List[ApertureEffectDataPoint]
    aperture_types: List[ApertureType]
    aperture_ratios: List[float]
    wavelength: float
    w0: float
    recommendation: str  # 选型建议


@dataclass
class ComparisonResult:
    """对比结果
    
    存储测量结果与理论预期的对比分析结果。
    
    Attributes:
        measured_values: 测量值数组
        theoretical_values: 理论值数组
        relative_errors: 相对误差数组
        rms_error: RMS 误差
        max_error: 最大误差
        fresnel_number: 菲涅尔数（可选，用于估算衍射效应）
    
    Requirements: 8.2, 8.3, 8.4
    """
    measured_values: np.ndarray
    theoretical_values: np.ndarray
    relative_errors: np.ndarray
    rms_error: float
    max_error: float
    fresnel_number: Optional[float]
