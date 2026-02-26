# -*- coding: utf-8 -*-
"""
测试报告生成器

本模块实现 ReportGenerator 类，用于生成光束参数测量与光阑分析的测试报告。

主要功能：
1. 生成 Markdown 格式的测试报告
2. 包含配置信息、对比表格、选型建议等部分
3. 支持保存到文件

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""

import os
from datetime import datetime
from typing import Optional, List

from .data_models import (
    ApertureType,
    ApertureEffectDataPoint,
    ApertureEffectAnalysisResult,
    PropagationAnalysisResult,
    ComparisonResult,
)


class ReportGenerator:
    """测试报告生成器
    
    生成光束参数测量与光阑分析的 Markdown 格式测试报告。
    
    报告内容包括：
    - 报告头部（标题、生成日期）
    - 配置信息（波长、束腰、光阑比例等）
    - 光阑影响分析表格
    - 传播分析结果（可选）
    - 测量与理论对比结果（可选）
    - 选型建议
    
    Example:
        >>> generator = ReportGenerator(output_dir="./reports")
        >>> report = generator.generate(
        ...     aperture_analysis=analysis_result,
        ...     propagation_analysis=propagation_result,
        ...     comparison_result=comparison_result,
        ...     title="光束参数测量报告",
        ... )
        >>> filepath = generator.save(report, "beam_analysis_report.md")
        >>> print(f"报告已保存到: {filepath}")
    
    Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
    """
    
    def __init__(self, output_dir: str = "."):
        """初始化报告生成器
        
        参数:
            output_dir: 输出目录，默认为当前目录
        """
        self.output_dir = output_dir
    
    def generate(
        self,
        aperture_analysis: ApertureEffectAnalysisResult,
        propagation_analysis: Optional[PropagationAnalysisResult] = None,
        comparison_result: Optional[ComparisonResult] = None,
        title: str = "光束参数测量与光阑分析报告",
    ) -> str:
        """生成完整的测试报告
        
        参数:
            aperture_analysis: 光阑影响分析结果
            propagation_analysis: 传播分析结果（可选）
            comparison_result: 对比结果（可选）
            title: 报告标题
        
        返回:
            Markdown 格式的报告内容
        
        Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
        """
        sections = []
        
        # 1. 生成报告头部
        sections.append(self._generate_header(title))
        
        # 2. 生成配置信息部分
        sections.append(self._generate_config_section(
            wavelength=aperture_analysis.wavelength,
            w0=aperture_analysis.w0,
            aperture_ratios=aperture_analysis.aperture_ratios,
        ))
        
        # 3. 生成光阑影响分析表格
        sections.append(self._generate_comparison_table(
            data_points=aperture_analysis.data_points,
        ))
        
        # 4. 如果有传播分析结果，生成传播分析部分
        if propagation_analysis is not None:
            sections.append(self._generate_propagation_section(
                propagation_analysis=propagation_analysis,
            ))
        
        # 5. 如果有对比结果，生成对比分析部分
        if comparison_result is not None:
            sections.append(self._generate_comparison_section(
                comparison_result=comparison_result,
            ))
        
        # 6. 生成选型建议部分
        sections.append(self._generate_recommendation_section(
            recommendation=aperture_analysis.recommendation,
        ))
        
        # 7. 生成报告尾部
        sections.append(self._generate_footer())
        
        return "\n\n".join(sections)

    def _generate_header(self, title: str) -> str:
        """生成报告头部
        
        参数:
            title: 报告标题
        
        返回:
            Markdown 格式的报告头部
        """
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d %H:%M:%S")
        
        lines = [
            f"# {title}",
            "",
            f"**生成日期**: {date_str}",
            "",
            "---",
        ]
        
        return "\n".join(lines)
    
    def _generate_config_section(
        self,
        wavelength: float,
        w0: float,
        aperture_ratios: List[float],
    ) -> str:
        """生成配置信息部分
        
        参数:
            wavelength: 波长 (m)
            w0: 束腰半径 (m)
            aperture_ratios: 光阑比例列表
        
        返回:
            Markdown 格式的配置信息部分
        """
        # 计算瑞利距离
        import numpy as np
        z_rayleigh = np.pi * w0**2 / wavelength
        
        # 计算理论发散角
        divergence = wavelength / (np.pi * w0)
        
        lines = [
            "## 测试配置",
            "",
            "### 光束参数",
            "",
            "| 参数 | 值 | 单位 |",
            "|------|-----|------|",
            f"| 波长 λ | {wavelength * 1e9:.2f} | nm |",
            f"| 束腰半径 w₀ | {w0 * 1e3:.4f} | mm |",
            f"| 束腰直径 2w₀ | {2 * w0 * 1e3:.4f} | mm |",
            f"| 瑞利距离 z_R | {z_rayleigh * 1e3:.2f} | mm |",
            f"| 理论发散角 θ | {divergence * 1e3:.4f} | mrad |",
            "",
            "### 光阑配置",
            "",
            "| 参数 | 值 |",
            "|------|-----|",
            f"| 光阑比例范围 | {min(aperture_ratios):.2f} ~ {max(aperture_ratios):.2f} |",
            f"| 测试点数 | {len(aperture_ratios)} |",
            f"| 光阑比例列表 | {', '.join([f'{r:.2f}' for r in aperture_ratios])} |",
        ]
        
        return "\n".join(lines)

    def _generate_comparison_table(
        self,
        data_points: List[ApertureEffectDataPoint],
    ) -> str:
        """生成对比分析表格
        
        参数:
            data_points: 光阑影响分析数据点列表
        
        返回:
            Markdown 格式的对比分析表格
        
        Requirements: 9.5
        """
        lines = [
            "## 光阑影响分析",
            "",
            "### 对比分析表格",
            "",
            "| 光阑类型 | 光阑比例 | 透过率 (%) | 理论透过率 (%) | 光束直径变化 (%) | 发散角变化 (%) |",
            "|----------|----------|------------|----------------|------------------|----------------|",
        ]
        
        # 按光阑类型和比例排序
        sorted_points = sorted(
            data_points,
            key=lambda dp: (dp.aperture_type.value, dp.aperture_ratio)
        )
        
        for dp in sorted_points:
            # 获取光阑类型的中文名称
            type_name = self._get_aperture_type_name(dp.aperture_type)
            
            # 格式化数值
            transmission_pct = dp.power_transmission * 100
            theoretical_pct = dp.theoretical_transmission * 100
            diameter_change_pct = dp.beam_diameter_change * 100
            divergence_change_pct = dp.divergence_change * 100
            
            lines.append(
                f"| {type_name} | {dp.aperture_ratio:.2f} | "
                f"{transmission_pct:.2f} | {theoretical_pct:.2f} | "
                f"{diameter_change_pct:+.2f} | {divergence_change_pct:+.2f} |"
            )
        
        # 添加表格说明
        lines.extend([
            "",
            "**说明**:",
            "- **光阑比例**: 光阑半径 / 光束半径 (a/w)",
            "- **透过率**: 实际功率透过率",
            "- **理论透过率**: 基于解析公式计算的理论透过率",
            "- **光束直径变化**: 相对于无光阑情况的变化率",
            "- **发散角变化**: 相对于无光阑情况的变化率",
        ])
        
        return "\n".join(lines)
    
    def _get_aperture_type_name(self, aperture_type: ApertureType) -> str:
        """获取光阑类型的中文名称
        
        参数:
            aperture_type: 光阑类型枚举
        
        返回:
            中文名称
        """
        type_names = {
            ApertureType.HARD_EDGE: "硬边光阑",
            ApertureType.GAUSSIAN: "高斯光阑",
            ApertureType.SUPER_GAUSSIAN: "超高斯光阑",
            ApertureType.EIGHTH_ORDER: "8阶光阑",
        }
        return type_names.get(aperture_type, aperture_type.value)

    def _generate_propagation_section(
        self,
        propagation_analysis: PropagationAnalysisResult,
    ) -> str:
        """生成传播分析部分
        
        参数:
            propagation_analysis: 传播分析结果
        
        返回:
            Markdown 格式的传播分析部分
        """
        import numpy as np
        
        lines = [
            "## 光束传播分析",
            "",
            "### 传播参数",
            "",
            "| 参数 | 值 | 单位 |",
            "|------|-----|------|",
            f"| X 方向发散角 θ_x | {propagation_analysis.divergence_x * 1e3:.4f} | mrad |",
            f"| Y 方向发散角 θ_y | {propagation_analysis.divergence_y * 1e3:.4f} | mrad |",
            f"| 平均发散角 θ_mean | {propagation_analysis.divergence_mean * 1e3:.4f} | mrad |",
            "",
            "### 光束直径随传输距离变化",
            "",
            "| 位置 z (mm) | D_x (mm) | D_y (mm) | D_mean (mm) | 测量方法 |",
            "|-------------|----------|----------|-------------|----------|",
        ]
        
        for dp in propagation_analysis.data_points:
            lines.append(
                f"| {dp.z * 1e3:.2f} | {dp.dx * 1e3:.4f} | "
                f"{dp.dy * 1e3:.4f} | {dp.d_mean * 1e3:.4f} | {dp.method} |"
            )
        
        return "\n".join(lines)
    
    def _generate_comparison_section(
        self,
        comparison_result: ComparisonResult,
    ) -> str:
        """生成测量与理论对比部分
        
        参数:
            comparison_result: 对比结果
        
        返回:
            Markdown 格式的对比分析部分
        """
        import numpy as np
        
        lines = [
            "## 测量与理论对比",
            "",
            "### 误差统计",
            "",
            "| 指标 | 值 |",
            "|------|-----|",
            f"| RMS 相对误差 | {comparison_result.rms_error * 100:.4f}% |",
            f"| 最大相对误差 | {comparison_result.max_error * 100:.4f}% |",
        ]
        
        # 如果有菲涅尔数，添加衍射效应信息
        if comparison_result.fresnel_number is not None:
            lines.extend([
                f"| 菲涅尔数 N_F | {comparison_result.fresnel_number:.4f} |",
            ])
        
        # 添加详细对比表格
        lines.extend([
            "",
            "### 详细对比",
            "",
            "| 序号 | 测量值 (mm) | 理论值 (mm) | 相对误差 (%) |",
            "|------|-------------|-------------|--------------|",
        ])
        
        for i, (measured, theoretical, error) in enumerate(zip(
            comparison_result.measured_values,
            comparison_result.theoretical_values,
            comparison_result.relative_errors,
        )):
            lines.append(
                f"| {i + 1} | {measured * 1e3:.4f} | "
                f"{theoretical * 1e3:.4f} | {error * 100:.4f} |"
            )
        
        return "\n".join(lines)

    def _generate_recommendation_section(
        self,
        recommendation: str,
    ) -> str:
        """生成选型建议部分
        
        参数:
            recommendation: 选型建议文本
        
        返回:
            Markdown 格式的选型建议部分
        
        Requirements: 9.4
        """
        lines = [
            "## 光阑选型建议",
            "",
            "```",
            recommendation,
            "```",
        ]
        
        return "\n".join(lines)
    
    def _generate_footer(self) -> str:
        """生成报告尾部
        
        返回:
            Markdown 格式的报告尾部
        """
        lines = [
            "---",
            "",
            "*本报告由 BTS 光束测量模块自动生成*",
        ]
        
        return "\n".join(lines)
    
    def save(self, content: str, filename: str = "report.md") -> str:
        """保存报告到文件
        
        参数:
            content: 报告内容
            filename: 文件名，默认为 "report.md"
        
        返回:
            保存的文件路径
        
        Requirements: 9.2
        """
        # 创建输出目录（如果不存在）
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 构建完整路径
        filepath = os.path.join(self.output_dir, filename)
        
        # 保存文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return filepath
    
    def __repr__(self) -> str:
        """返回对象的字符串表示"""
        return f"ReportGenerator(output_dir='{self.output_dir}')"
