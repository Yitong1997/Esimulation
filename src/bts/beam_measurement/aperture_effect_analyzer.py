# -*- coding: utf-8 -*-
"""
光阑影响分析器

本模块实现 ApertureEffectAnalyzer 类，用于分析不同光阑类型和尺寸对光束传输的影响。

主要功能：
1. 对比四种光阑类型（硬边、高斯、超高斯、8 阶）
2. 分析不同光阑尺寸对光束的影响
3. 测量功率透过率、光束直径变化、发散角变化
4. 生成光阑选型建议

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
"""

from typing import List, Optional, Tuple
import numpy as np

from .data_models import (
    ApertureType,
    ApertureEffectDataPoint,
    ApertureEffectAnalysisResult,
)
from .circular_aperture import CircularAperture
from .d4sigma_calculator import D4sigmaCalculator
from .exceptions import InvalidInputError


class ApertureEffectAnalyzer:
    """光阑影响分析器
    
    分析不同光阑类型和尺寸对光束传输的影响，包括：
    - 功率透过率
    - 光束直径变化
    - 发散角变化
    - 理论透过率对比
    
    支持四种光阑类型：
    - HARD_EDGE: 硬边光阑
    - GAUSSIAN: 高斯光阑
    - SUPER_GAUSSIAN: 超高斯/软边光阑
    - EIGHTH_ORDER: 8 阶软边光阑
    
    Example:
        >>> analyzer = ApertureEffectAnalyzer(
        ...     wavelength=633e-9,  # 633 nm
        ...     w0=1e-3,            # 1 mm 束腰
        ...     grid_size=256,
        ...     propagation_distance=1.0,  # 1 m
        ... )
        >>> result = analyzer.analyze(
        ...     aperture_ratios=[0.8, 1.0, 1.2, 1.5, 2.0],
        ...     aperture_types=[ApertureType.HARD_EDGE, ApertureType.GAUSSIAN],
        ... )
        >>> print(result.recommendation)
    
    Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
    """
    
    def __init__(
        self,
        wavelength: float,
        w0: float,
        grid_size: int = 256,
        propagation_distance: Optional[float] = None,
    ):
        """初始化光阑影响分析器
        
        参数:
            wavelength: 波长 (m)
            w0: 束腰半径 (m)
            grid_size: 网格大小，默认 256
            propagation_distance: 传播距离 (m)，用于测量远场效果
                - 如果为 None，默认使用 10 倍瑞利距离
        
        异常:
            InvalidInputError: 参数无效时抛出
        """
        # 验证参数
        if wavelength <= 0:
            raise InvalidInputError(
                f"波长必须为正数，收到: {wavelength}"
            )
        
        if w0 <= 0:
            raise InvalidInputError(
                f"束腰半径必须为正数，收到: {w0}"
            )
        
        if grid_size < 16:
            raise InvalidInputError(
                f"网格大小必须至少为 16，收到: {grid_size}"
            )
        
        self.wavelength = wavelength
        self.w0 = w0
        self.grid_size = grid_size
        
        # 计算瑞利距离
        self.z_rayleigh = np.pi * w0**2 / wavelength
        
        # 设置传播距离（默认 10 倍瑞利距离，确保在远场）
        if propagation_distance is None:
            self.propagation_distance = 10 * self.z_rayleigh
        else:
            if propagation_distance <= 0:
                raise InvalidInputError(
                    f"传播距离必须为正数，收到: {propagation_distance}"
                )
            self.propagation_distance = propagation_distance
        
        # 初始化 D4sigma 计算器
        self._d4sigma_calculator = D4sigmaCalculator()

    def analyze(
        self,
        aperture_ratios: List[float],
        aperture_types: Optional[List[ApertureType]] = None,
    ) -> ApertureEffectAnalysisResult:
        """分析不同光阑配置的影响
        
        对每种光阑类型和每个光阑比例执行以下分析：
        1. 创建初始高斯光束
        2. 应用光阑
        3. 计算功率透过率
        4. 传播到指定距离
        5. 测量光束直径变化和发散角变化
        
        参数:
            aperture_ratios: 光阑半径/光束半径 列表
                - 例如 [0.8, 1.0, 1.2, 1.5, 2.0]
                - 值必须为正数
            aperture_types: 要分析的光阑类型列表（可选）
                - 默认分析全部四种类型
                - 可选值：HARD_EDGE, GAUSSIAN, SUPER_GAUSSIAN, EIGHTH_ORDER
        
        返回:
            ApertureEffectAnalysisResult 对象，包含：
            - data_points: 各光阑配置的分析数据点列表
            - aperture_types: 分析的光阑类型列表
            - aperture_ratios: 分析的光阑比例列表
            - wavelength: 波长 (m)
            - w0: 束腰半径 (m)
            - recommendation: 光阑选型建议
        
        异常:
            InvalidInputError: 输入参数无效时抛出
        
        Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
        """
        # 验证 aperture_ratios
        if not aperture_ratios:
            raise InvalidInputError("光阑比例列表不能为空")
        
        for ratio in aperture_ratios:
            if ratio <= 0:
                raise InvalidInputError(
                    f"光阑比例必须为正数，收到: {ratio}"
                )
        
        # 设置默认光阑类型（全部四种）
        if aperture_types is None:
            aperture_types = [
                ApertureType.HARD_EDGE,
                ApertureType.GAUSSIAN,
                ApertureType.SUPER_GAUSSIAN,
                ApertureType.EIGHTH_ORDER,
            ]
        
        # 验证 aperture_types
        if not aperture_types:
            raise InvalidInputError("光阑类型列表不能为空")
        
        for apt in aperture_types:
            if not isinstance(apt, ApertureType):
                raise InvalidInputError(
                    f"光阑类型必须是 ApertureType 枚举，收到: {type(apt).__name__}"
                )
        
        # 首先获取无光阑情况下的参考值
        ref_diameter, ref_divergence = self._get_reference_values()
        
        # 收集所有数据点
        data_points: List[ApertureEffectDataPoint] = []
        
        for aperture_type in aperture_types:
            for ratio in aperture_ratios:
                # 分析单个光阑配置
                data_point = self._analyze_single_configuration(
                    aperture_type=aperture_type,
                    aperture_ratio=ratio,
                    ref_diameter=ref_diameter,
                    ref_divergence=ref_divergence,
                )
                data_points.append(data_point)
        
        # 生成选型建议
        recommendation = self._generate_recommendation(data_points)
        
        return ApertureEffectAnalysisResult(
            data_points=data_points,
            aperture_types=aperture_types,
            aperture_ratios=aperture_ratios,
            wavelength=self.wavelength,
            w0=self.w0,
            recommendation=recommendation,
        )

    def _get_reference_values(self) -> Tuple[float, float]:
        """获取无光阑情况下的参考值
        
        创建理想高斯光束，传播到指定距离，测量光束直径和发散角。
        
        返回:
            (ref_diameter, ref_divergence) 元组
            - ref_diameter: 参考光束直径 (m)
            - ref_divergence: 参考发散角 (rad)
        """
        import proper
        
        # 创建初始高斯光束
        wfo = self._create_initial_wavefront()
        
        # 传播到指定距离
        proper.prop_propagate(wfo, self.propagation_distance)
        
        # 测量光束直径（使用 PROPER 的 prop_get_beamradius）
        beam_radius = proper.prop_get_beamradius(wfo)
        ref_diameter = 2 * beam_radius  # D4sigma = 2 × w
        
        # 计算理论发散角
        # θ = λ / (π × w₀)
        ref_divergence = self.wavelength / (np.pi * self.w0)
        
        return ref_diameter, ref_divergence
    
    def _analyze_single_configuration(
        self,
        aperture_type: ApertureType,
        aperture_ratio: float,
        ref_diameter: float,
        ref_divergence: float,
    ) -> ApertureEffectDataPoint:
        """分析单个光阑配置
        
        参数:
            aperture_type: 光阑类型
            aperture_ratio: 光阑半径/光束半径
            ref_diameter: 参考光束直径 (m)
            ref_divergence: 参考发散角 (rad)
        
        返回:
            ApertureEffectDataPoint 对象
        
        注意:
            PROPER 使用参考球面跟踪高斯光束，prop_get_amplitude 返回的是
            相对于参考球面的偏差振幅（对于理想高斯光束是均匀的）。
            因此，功率透过率需要使用真实的高斯强度分布来计算。
        """
        import proper
        
        # 计算光阑半径
        aperture_radius = aperture_ratio * self.w0
        
        # 创建光阑
        aperture = self._create_aperture(aperture_type, aperture_radius)
        
        # 计算理论透过率（使用解析公式）
        theoretical_transmission = aperture._calculate_theoretical_transmission(
            aperture_radius, self.w0
        )
        
        # 对于功率透过率，使用数值积分计算真实高斯光束通过光阑的透过率
        power_transmission = self._calculate_power_transmission_numerical(
            aperture_type, aperture_radius
        )
        
        # 创建初始高斯光束并应用光阑
        wfo = self._create_initial_wavefront()
        aperture.apply(wfo)
        
        # 传播到指定距离
        proper.prop_propagate(wfo, self.propagation_distance)
        
        # 测量光束直径（使用 D4sigma 方法，因为光束已不再是理想高斯）
        try:
            d4sigma_result = self._d4sigma_calculator.calculate(wfo)
            measured_diameter = d4sigma_result.d_mean
        except Exception:
            # 如果 D4sigma 计算失败，使用 PROPER 的 beamradius
            beam_radius = proper.prop_get_beamradius(wfo)
            measured_diameter = 2 * beam_radius
        
        # 计算光束直径变化率
        if ref_diameter > 0:
            beam_diameter_change = (measured_diameter - ref_diameter) / ref_diameter
        else:
            beam_diameter_change = 0.0
        
        # 计算发散角变化
        # 发散角 ≈ 光束半径 / 传播距离（远场近似）
        measured_divergence = (measured_diameter / 2) / self.propagation_distance
        if ref_divergence > 0:
            divergence_change = (measured_divergence - ref_divergence) / ref_divergence
        else:
            divergence_change = 0.0
        
        return ApertureEffectDataPoint(
            aperture_type=aperture_type,
            aperture_ratio=aperture_ratio,
            power_transmission=power_transmission,
            beam_diameter_change=beam_diameter_change,
            divergence_change=divergence_change,
            theoretical_transmission=theoretical_transmission,
        )
    
    def _calculate_power_transmission_numerical(
        self,
        aperture_type: ApertureType,
        aperture_radius: float,
    ) -> float:
        """使用数值积分计算功率透过率
        
        由于 PROPER 使用参考球面跟踪高斯光束，prop_get_amplitude 返回的
        振幅是均匀的，不能直接用于计算功率透过率。
        
        本方法使用真实的高斯强度分布和光阑透过率函数进行数值积分。
        
        参数:
            aperture_type: 光阑类型
            aperture_radius: 光阑半径 (m)
        
        返回:
            功率透过率
        """
        from scipy import integrate
        
        w = self.w0  # 光束半径
        
        # 高斯光束强度分布: I(r) = exp(-2 * (r/w)²)
        def gaussian_intensity(r):
            return np.exp(-2.0 * (r / w)**2)
        
        # 根据光阑类型定义透过率函数（强度透过率 = 振幅透过率²）
        if aperture_type == ApertureType.HARD_EDGE:
            def aperture_transmission(r):
                return 1.0 if r <= aperture_radius else 0.0
        elif aperture_type == ApertureType.GAUSSIAN:
            sigma = aperture_radius
            def aperture_transmission(r):
                # 振幅透过率: T_amp = exp(-0.5 * (r/σ)²)
                # 强度透过率: T_int = T_amp² = exp(-(r/σ)²)
                return np.exp(-(r / sigma)**2)
        elif aperture_type == ApertureType.SUPER_GAUSSIAN:
            n = 4  # 超高斯阶数
            def aperture_transmission(r):
                # 振幅透过率: T_amp = exp(-(r/r₀)ⁿ)
                # 强度透过率: T_int = T_amp² = exp(-2*(r/r₀)ⁿ)
                return np.exp(-2.0 * (r / aperture_radius)**n)
        elif aperture_type == ApertureType.EIGHTH_ORDER:
            # 8 阶光阑使用近似的 sigmoid 函数
            def aperture_transmission(r):
                x = r / aperture_radius
                # 近似强度透过率
                return 1.0 / (1.0 + x**8)
        else:
            def aperture_transmission(r):
                return 1.0
        
        # 被积函数（分子）：I(r) × T(r) × r
        def integrand_numerator(r):
            return gaussian_intensity(r) * aperture_transmission(r) * r
        
        # 被积函数（分母）：I(r) × r
        def integrand_denominator(r):
            return gaussian_intensity(r) * r
        
        # 积分上限：取足够大的值（10 倍光束半径）
        r_max = 10.0 * w
        
        # 数值积分
        numerator, _ = integrate.quad(integrand_numerator, 0, r_max)
        denominator, _ = integrate.quad(integrand_denominator, 0, r_max)
        
        if denominator > 0:
            return numerator / denominator
        else:
            return 0.0

    def _create_aperture(
        self,
        aperture_type: ApertureType,
        aperture_radius: float,
    ) -> CircularAperture:
        """创建指定类型的光阑
        
        参数:
            aperture_type: 光阑类型
            aperture_radius: 光阑半径 (m)
        
        返回:
            CircularAperture 对象
        """
        if aperture_type == ApertureType.HARD_EDGE:
            return CircularAperture(
                aperture_type=ApertureType.HARD_EDGE,
                radius=aperture_radius,
                normalized=False,
            )
        elif aperture_type == ApertureType.GAUSSIAN:
            # 高斯光阑：sigma = radius
            return CircularAperture(
                aperture_type=ApertureType.GAUSSIAN,
                radius=aperture_radius,
                normalized=False,
                gaussian_sigma=aperture_radius,
            )
        elif aperture_type == ApertureType.SUPER_GAUSSIAN:
            # 超高斯光阑：使用 4 阶
            return CircularAperture(
                aperture_type=ApertureType.SUPER_GAUSSIAN,
                radius=aperture_radius,
                normalized=False,
                super_gaussian_order=4,
            )
        elif aperture_type == ApertureType.EIGHTH_ORDER:
            # 8 阶光阑：HWHM = radius
            return CircularAperture(
                aperture_type=ApertureType.EIGHTH_ORDER,
                radius=aperture_radius,
                normalized=False,
            )
        else:
            raise InvalidInputError(f"未知的光阑类型: {aperture_type}")
    
    def _create_initial_wavefront(self) -> "proper.WaveFront":
        """创建理想高斯光束的初始波前
        
        使用 PROPER 库创建一个理想高斯光束。
        
        根据 BTS 规范：
        - beam_diameter = 2 × w0
        - beam_diam_fraction = 0.5
        - 网格物理尺寸 = 4 × w0
        
        返回:
            PROPER 波前对象
        """
        import proper
        
        # 根据 BTS 规范设置参数
        beam_diameter = 2 * self.w0  # beam_diameter = 2 × 束腰半径
        beam_diam_fraction = 0.5     # 固定为 0.5
        
        # 创建波前对象
        wfo = proper.prop_begin(
            beam_diameter,
            self.wavelength,
            self.grid_size,
            beam_diam_fraction,
        )
        
        # 定义高斯光束
        proper.prop_define_entrance(wfo)
        
        return wfo

    def _generate_recommendation(
        self,
        data_points: List[ApertureEffectDataPoint],
    ) -> str:
        """生成光阑选型建议
        
        基于分析结果生成选型建议，考虑：
        - 功率透过率
        - 光束质量（直径变化、发散角变化）
        - 衍射效应
        
        参数:
            data_points: 各光阑配置的分析数据点列表
        
        返回:
            选型建议字符串
        
        Requirements: 6.7
        """
        if not data_points:
            return "无数据，无法生成建议。"
        
        # 按光阑类型分组统计
        type_stats = {}
        for dp in data_points:
            apt_type = dp.aperture_type
            if apt_type not in type_stats:
                type_stats[apt_type] = {
                    'transmissions': [],
                    'diameter_changes': [],
                    'divergence_changes': [],
                    'ratios': [],
                }
            type_stats[apt_type]['transmissions'].append(dp.power_transmission)
            type_stats[apt_type]['diameter_changes'].append(abs(dp.beam_diameter_change))
            type_stats[apt_type]['divergence_changes'].append(abs(dp.divergence_change))
            type_stats[apt_type]['ratios'].append(dp.aperture_ratio)
        
        # 生成建议文本
        lines = []
        lines.append("=" * 60)
        lines.append("光阑选型建议")
        lines.append("=" * 60)
        lines.append("")
        
        # 1. 高功率透过率需求
        lines.append("【高功率透过率需求】")
        best_transmission_type = None
        best_transmission = 0
        for apt_type, stats in type_stats.items():
            avg_trans = np.mean(stats['transmissions'])
            if avg_trans > best_transmission:
                best_transmission = avg_trans
                best_transmission_type = apt_type
        
        if best_transmission_type:
            lines.append(f"  推荐: {best_transmission_type.value} 光阑")
            lines.append(f"  平均透过率: {best_transmission * 100:.1f}%")
            # 找到透过率 > 95% 的最小光阑比例
            for dp in data_points:
                if dp.aperture_type == best_transmission_type and dp.power_transmission > 0.95:
                    lines.append(f"  建议光阑比例 ≥ {dp.aperture_ratio:.1f} (透过率 > 95%)")
                    break
        lines.append("")
        
        # 2. 高光束质量需求
        lines.append("【高光束质量需求】")
        best_quality_type = None
        best_quality_score = float('inf')
        for apt_type, stats in type_stats.items():
            # 光束质量评分 = 直径变化 + 发散角变化（越小越好）
            avg_diameter_change = np.mean(stats['diameter_changes'])
            avg_divergence_change = np.mean(stats['divergence_changes'])
            quality_score = avg_diameter_change + avg_divergence_change
            if quality_score < best_quality_score:
                best_quality_score = quality_score
                best_quality_type = apt_type
        
        if best_quality_type:
            lines.append(f"  推荐: {best_quality_type.value} 光阑")
            lines.append(f"  原因: 对光束直径和发散角影响最小")
            # 软边光阑通常具有更好的光束质量
            if best_quality_type in [ApertureType.GAUSSIAN, ApertureType.SUPER_GAUSSIAN]:
                lines.append("  软边光阑可减少衍射效应，保持光束质量")
        lines.append("")
        
        # 3. 平衡需求
        lines.append("【平衡需求（透过率与光束质量兼顾）】")
        # 计算综合评分：透过率高 + 光束质量好
        best_balanced_type = None
        best_balanced_score = -float('inf')
        for apt_type, stats in type_stats.items():
            avg_trans = np.mean(stats['transmissions'])
            avg_diameter_change = np.mean(stats['diameter_changes'])
            avg_divergence_change = np.mean(stats['divergence_changes'])
            # 综合评分 = 透过率 - (直径变化 + 发散角变化)
            balanced_score = avg_trans - 0.5 * (avg_diameter_change + avg_divergence_change)
            if balanced_score > best_balanced_score:
                best_balanced_score = balanced_score
                best_balanced_type = apt_type
        
        if best_balanced_type:
            lines.append(f"  推荐: {best_balanced_type.value} 光阑")
            if best_balanced_type == ApertureType.SUPER_GAUSSIAN:
                lines.append("  超高斯光阑在透过率和光束质量之间取得良好平衡")
            elif best_balanced_type == ApertureType.EIGHTH_ORDER:
                lines.append("  8 阶光阑具有平滑的过渡边缘，减少衍射振铃")
        lines.append("")
        
        # 4. 光阑比例建议
        lines.append("【光阑比例建议】")
        lines.append("  • 光阑比例 < 1.0: 显著截断光束，功率损失大，衍射效应明显")
        lines.append("  • 光阑比例 = 1.0: 截断 1/e² 边缘，约 86% 透过率（硬边）")
        lines.append("  • 光阑比例 = 1.5: 良好的透过率（~99%），轻微衍射")
        lines.append("  • 光阑比例 ≥ 2.0: 几乎无截断，透过率接近 100%")
        lines.append("")
        
        # 5. 各类型光阑特点总结
        lines.append("【各类型光阑特点】")
        lines.append("  • 硬边光阑: 透过率计算简单，但衍射效应最强")
        lines.append("  • 高斯光阑: 平滑过渡，衍射效应小，但透过率较低")
        lines.append("  • 超高斯光阑: 可调节阶数，平衡透过率和边缘平滑度")
        lines.append("  • 8 阶光阑: 基于 sinc 函数，具有优化的边缘过渡")
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """返回对象的字符串表示"""
        return (
            f"ApertureEffectAnalyzer("
            f"wavelength={self.wavelength:.3e} m, "
            f"w0={self.w0:.3e} m, "
            f"grid_size={self.grid_size}, "
            f"propagation_distance={self.propagation_distance:.3e} m)"
        )
