# -*- coding: utf-8 -*-
"""
BTS 光束测量与光阑 API 函数

本模块提供统一的光束测量和光阑功能 API 函数：
1. measure_beam_diameter() - 测量光束直径
2. measure_m2() - 测量 M² 因子
3. apply_aperture() - 应用光阑到 PROPER 波前
4. analyze_aperture_effects() - 分析光阑对光束的影响

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6
"""

from typing import Union, Optional, List, TYPE_CHECKING
import numpy as np

# 导入数据模型
from .data_models import (
    D4sigmaResult,
    ISOD4sigmaResult,
    M2Result,
    ApertureType,
    ApertureEffectAnalysisResult,
)

# 导入计算器和分析器
from .d4sigma_calculator import D4sigmaCalculator
from .iso_d4sigma_calculator import ISOD4sigmaCalculator
from .m2_calculator import M2Calculator
from .circular_aperture import CircularAperture
from .aperture_effect_analyzer import ApertureEffectAnalyzer
from .report_generator import ReportGenerator

# 导入异常类
from .exceptions import InvalidInputError

if TYPE_CHECKING:
    import proper
    from hybrid_simulation import SimulationResult


def measure_beam_diameter(
    data: Union[np.ndarray, "proper.WaveFront", "SimulationResult"],
    method: str = "ideal",
    sampling: Optional[float] = None,
    surface_index: int = -1,
    **kwargs,
) -> Union[D4sigmaResult, ISOD4sigmaResult]:
    """测量光束直径
    
    使用 D4sigma（二阶矩）方法测量光束直径。支持两种测量方法：
    - "ideal": 理想二阶矩方法，直接计算
    - "iso": ISO 11146 标准方法，包含背景噪声去除和迭代 ROI
    
    参数:
        data: 输入数据，支持以下类型：
            - numpy.ndarray: 复振幅数组（需要提供 sampling 参数）
            - proper.WaveFront: PROPER 波前对象（自动获取采样间隔）
            - SimulationResult: BTS 仿真结果（从指定表面提取波前）
        method: 测量方法
            - "ideal": 理想二阶矩方法（默认）
            - "iso": ISO 11146 标准方法
        sampling: 采样间隔 (m)
            - 如果 data 是 numpy 数组，此参数必须提供
            - 如果 data 是 PROPER 波前对象或 SimulationResult，此参数可选
        surface_index: 如果 data 是 SimulationResult，指定表面索引
            - 默认 -1 表示最后一个表面
            - 正数表示指定索引的表面
        **kwargs: 传递给 ISO 方法的额外参数
            - max_iterations: 最大迭代次数（默认 10）
            - convergence_threshold: 收敛阈值（默认 0.01）
            - roi_factor: ROI 边界因子（默认 3.0）
    
    返回:
        D4sigmaResult 或 ISOD4sigmaResult 对象
    
    Requirements: 7.1, 7.4, 7.5
    """
    # 处理 SimulationResult 输入
    amplitude_data, actual_sampling = _extract_amplitude_data(
        data, sampling, surface_index
    )
    
    # 根据方法选择计算器
    if method == "ideal":
        calculator = D4sigmaCalculator()
        return calculator.calculate(amplitude_data, actual_sampling)
    elif method == "iso":
        # 提取 ISO 方法的额外参数
        max_iterations = kwargs.get("max_iterations", 10)
        convergence_threshold = kwargs.get("convergence_threshold", 0.01)
        roi_factor = kwargs.get("roi_factor", 3.0)
        
        calculator = ISOD4sigmaCalculator(
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            roi_factor=roi_factor,
        )
        return calculator.calculate(amplitude_data, actual_sampling)
    else:
        raise InvalidInputError(
            f"不支持的测量方法: {method}。支持的方法: 'ideal', 'iso'"
        )


def _extract_amplitude_data(
    data: Union[np.ndarray, "proper.WaveFront", "SimulationResult"],
    sampling: Optional[float],
    surface_index: int,
) -> tuple:
    """从输入数据中提取复振幅数组和采样间隔"""
    # 检查是否为 SimulationResult
    if hasattr(data, 'surfaces') and hasattr(data, 'get_surface'):
        return _extract_from_simulation_result(data, sampling, surface_index)
    
    # 检查是否为 PROPER 波前对象
    if hasattr(data, 'wfarr'):
        return data, sampling
    
    # 假设是 numpy 数组
    if isinstance(data, np.ndarray):
        return data, sampling
    
    raise InvalidInputError(
        f"不支持的输入类型: {type(data).__name__}。"
        "支持的类型: numpy.ndarray, PROPER WaveFront, SimulationResult"
    )


def _extract_from_simulation_result(
    result: "SimulationResult",
    sampling: Optional[float],
    surface_index: int,
) -> tuple:
    """从 SimulationResult 中提取复振幅数据"""
    # 获取指定表面的波前数据
    if surface_index == -1:
        wavefront_data = result.get_final_wavefront()
    else:
        wavefront_data = result.get_exit_wavefront(surface_index)
    
    # 获取复振幅
    amplitude = wavefront_data.amplitude
    
    # 获取采样间隔
    if sampling is not None:
        actual_sampling = sampling
    else:
        # 从 grid 信息获取采样间隔
        physical_size_m = wavefront_data.grid.physical_size_mm * 1e-3  # mm -> m
        actual_sampling = physical_size_m / wavefront_data.grid.grid_size
    
    return amplitude, actual_sampling



def measure_m2(
    z_positions: np.ndarray,
    beam_diameters_x: np.ndarray,
    beam_diameters_y: np.ndarray,
    wavelength: float,
) -> M2Result:
    """测量 M² 因子
    
    通过多点光束直径拟合计算 M² 因子。
    
    参数:
        z_positions: 传输位置数组 (m)
        beam_diameters_x: X 方向光束直径数组 (m)
        beam_diameters_y: Y 方向光束直径数组 (m)
        wavelength: 波长 (m)
    
    返回:
        M2Result 对象
    
    Requirements: 7.2, 7.5
    """
    calculator = M2Calculator(wavelength)
    return calculator.calculate(z_positions, beam_diameters_x, beam_diameters_y)


def apply_aperture(
    wfo: "proper.WaveFront",
    aperture_type: str,
    radius: float,
    normalized: bool = False,
    **kwargs,
) -> np.ndarray:
    """应用光阑到 PROPER 波前
    
    支持四种光阑类型：
    - "hard_edge": 硬边光阑
    - "gaussian": 高斯光阑
    - "super_gaussian": 超高斯光阑
    - "eighth_order": 8 阶软边光阑
    
    参数:
        wfo: PROPER 波前对象
        aperture_type: 光阑类型字符串
        radius: 光阑半径 (m) 或归一化半径
        normalized: 是否使用归一化半径
        **kwargs: 光阑类型特定参数
            - center_x, center_y: 光阑中心位置 (m)
            - gaussian_sigma: 高斯光阑的 σ 参数 (m)
            - super_gaussian_order: 超高斯光阑的阶数 n
            - min_transmission: 8 阶光阑的最小透过率
            - max_transmission: 8 阶光阑的最大透过率
    
    返回:
        光阑透过率掩模数组
    
    Requirements: 7.3, 7.5
    """
    # 将字符串类型转换为 ApertureType 枚举
    type_mapping = {
        "hard_edge": ApertureType.HARD_EDGE,
        "gaussian": ApertureType.GAUSSIAN,
        "super_gaussian": ApertureType.SUPER_GAUSSIAN,
        "eighth_order": ApertureType.EIGHTH_ORDER,
    }
    
    if aperture_type not in type_mapping:
        raise InvalidInputError(
            f"不支持的光阑类型: {aperture_type}。"
            f"支持的类型: {list(type_mapping.keys())}"
        )
    
    apt_type = type_mapping[aperture_type]
    
    # 提取额外参数
    center_x = kwargs.get("center_x", 0.0)
    center_y = kwargs.get("center_y", 0.0)
    gaussian_sigma = kwargs.get("gaussian_sigma", None)
    super_gaussian_order = kwargs.get("super_gaussian_order", 2)
    min_transmission = kwargs.get("min_transmission", 0.0)
    max_transmission = kwargs.get("max_transmission", 1.0)
    
    # 创建光阑对象
    aperture = CircularAperture(
        aperture_type=apt_type,
        radius=radius,
        normalized=normalized,
        center_x=center_x,
        center_y=center_y,
        gaussian_sigma=gaussian_sigma,
        super_gaussian_order=super_gaussian_order,
        min_transmission=min_transmission,
        max_transmission=max_transmission,
    )
    
    # 应用光阑并返回掩模
    return aperture.apply(wfo)


def analyze_aperture_effects(
    wavelength: float,
    w0: float,
    aperture_ratios: List[float],
    aperture_types: Optional[List[str]] = None,
    grid_size: int = 256,
    propagation_distance: Optional[float] = None,
    generate_report: bool = True,
    output_dir: str = ".",
) -> ApertureEffectAnalysisResult:
    """分析光阑对光束的影响
    
    参数:
        wavelength: 波长 (m)
        w0: 束腰半径 (m)
        aperture_ratios: 光阑半径/光束半径 列表
        aperture_types: 要分析的光阑类型列表（字符串格式）
        grid_size: 网格大小
        propagation_distance: 传播距离 (m)
        generate_report: 是否生成报告
        output_dir: 报告输出目录
    
    返回:
        ApertureEffectAnalysisResult 对象
    
    Requirements: 7.6
    """
    # 将字符串类型列表转换为 ApertureType 枚举列表
    type_mapping = {
        "hard_edge": ApertureType.HARD_EDGE,
        "gaussian": ApertureType.GAUSSIAN,
        "super_gaussian": ApertureType.SUPER_GAUSSIAN,
        "eighth_order": ApertureType.EIGHTH_ORDER,
    }
    
    apt_types: Optional[List[ApertureType]] = None
    if aperture_types is not None:
        apt_types = []
        for type_str in aperture_types:
            if type_str not in type_mapping:
                raise InvalidInputError(
                    f"不支持的光阑类型: {type_str}。"
                    f"支持的类型: {list(type_mapping.keys())}"
                )
            apt_types.append(type_mapping[type_str])
    
    # 创建分析器
    analyzer = ApertureEffectAnalyzer(
        wavelength=wavelength,
        w0=w0,
        grid_size=grid_size,
        propagation_distance=propagation_distance,
    )
    
    # 执行分析
    result = analyzer.analyze(
        aperture_ratios=aperture_ratios,
        aperture_types=apt_types,
    )
    
    # 如果需要生成报告
    if generate_report:
        report_generator = ReportGenerator(output_dir=output_dir)
        report_content = report_generator.generate(
            aperture_analysis=result,
            title="光阑影响分析报告",
        )
        report_generator.save(report_content, filename="aperture_analysis_report.md")
    
    return result


# 导出的公共 API
__all__ = [
    "measure_beam_diameter",
    "measure_m2",
    "apply_aperture",
    "analyze_aperture_effects",
]
