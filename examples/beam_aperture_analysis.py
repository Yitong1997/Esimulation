# -*- coding: utf-8 -*-
"""
光束参数测量与光阑分析示例

本示例演示如何使用 BTS 光束测量模块进行：
1. 光束直径测量（D4sigma 方法）
2. 光阑应用与透过率计算
3. 光束传播分析
4. 光阑影响对比分析
5. 测试报告生成

注意：
- PROPER 使用参考球面跟踪理想高斯光束，prop_get_amplitude 返回均匀振幅
- 对于理想高斯光束，使用 prop_get_beamradius 获取光束半径
- D4sigma 方法适用于经过光阑或有像差的非理想光束

Requirements: 6.1-6.7, 9.1-9.5
"""

import sys
import os

# 添加 src 目录到路径
sys.path.insert(0, 'src')

import numpy as np
import proper

# 导入 BTS 光束测量 API
from bts.beam_measurement import (
    # API 函数
    measure_beam_diameter,
    measure_m2,
    apply_aperture,
    analyze_aperture_effects,
    # 核心类
    D4sigmaCalculator,
    BeamPropagationAnalyzer,
    ApertureEffectAnalyzer,
    ComparisonModule,
    ReportGenerator,
    # 数据模型
    ApertureType,
    CircularAperture,
)


def create_gaussian_beam(wavelength: float, w0: float, grid_size: int = 256):
    """创建理想高斯光束
    
    使用 PROPER 库创建一个理想高斯光束。
    
    参数:
        wavelength: 波长 (m)
        w0: 束腰半径 (m)
        grid_size: 网格大小
    
    返回:
        PROPER 波前对象
    """
    # 根据 BTS 规范设置参数
    # beam_diameter = 2 × w0
    # beam_diam_fraction = 0.5
    beam_diameter = 2 * w0
    beam_diam_fraction = 0.5
    
    # 创建波前对象
    wfo = proper.prop_begin(
        beam_diameter,
        wavelength,
        grid_size,
        beam_diam_fraction,
    )
    
    # 定义高斯光束入射
    proper.prop_define_entrance(wfo)
    
    return wfo



def demo_beam_diameter_measurement():
    """演示光束直径测量
    
    说明 PROPER 的参考球面跟踪机制，以及如何正确测量光束直径。
    
    注意：
    - PROPER 使用参考球面跟踪理想高斯光束
    - prop_get_amplitude 返回相对于参考球面的偏差（对于理想高斯是均匀的）
    - 应使用 prop_get_beamradius 获取 PROPER 内部跟踪的光束半径
    - D4sigma 方法适用于经过光阑后的非理想光束
    """
    print("\n" + "=" * 60)
    print("1. 光束直径测量演示")
    print("=" * 60)
    
    # 定义光束参数
    wavelength = 633e-9  # 633 nm
    w0 = 1e-3            # 1 mm 束腰
    grid_size = 256
    
    print(f"\n光束参数:")
    print(f"  波长 λ = {wavelength * 1e9:.1f} nm")
    print(f"  束腰 w₀ = {w0 * 1e3:.2f} mm")
    print(f"  理论 D4σ = 2×w₀ = {2 * w0 * 1e3:.2f} mm")
    
    # 创建高斯光束
    wfo = create_gaussian_beam(wavelength, w0, grid_size)
    
    # 方法 1：使用 PROPER 的 prop_get_beamradius（推荐用于理想高斯光束）
    print("\n方法 1: 使用 PROPER 的 prop_get_beamradius（理想高斯光束）:")
    beam_radius = proper.prop_get_beamradius(wfo)
    d4sigma_proper = 2 * beam_radius
    print(f"  光束半径 w = {beam_radius * 1e3:.4f} mm")
    print(f"  D4σ = 2×w = {d4sigma_proper * 1e3:.4f} mm")
    
    theoretical_d4sigma = 2 * w0
    error = abs(d4sigma_proper - theoretical_d4sigma) / theoretical_d4sigma * 100
    print(f"  相对误差 = {error:.4f}%")
    
    # 方法 2：对经过光阑的光束使用 D4sigma 方法
    print("\n方法 2: 对经过光阑的光束使用 D4sigma 方法:")
    
    # 创建新的波前并应用光阑
    wfo_with_aperture = create_gaussian_beam(wavelength, w0, grid_size)
    aperture = CircularAperture(
        aperture_type=ApertureType.HARD_EDGE,
        radius=1.5 * w0,  # 光阑半径 = 1.5 倍光束半径
    )
    aperture.apply(wfo_with_aperture)
    
    # 使用 D4sigma 方法测量（适用于非理想光束）
    result = measure_beam_diameter(wfo_with_aperture, method="ideal")
    print(f"  光阑半径 = 1.5 × w₀ = {1.5 * w0 * 1e3:.2f} mm")
    print(f"  D4σ (D4sigma 方法) = {result.d_mean * 1e3:.4f} mm")
    print(f"  质心位置: ({result.centroid_x * 1e3:.4f}, {result.centroid_y * 1e3:.4f}) mm")
    
    return wavelength, w0, grid_size


def demo_aperture_application():
    """演示光阑应用与透过率计算
    
    演示四种光阑类型的应用和能量透过率计算。
    """
    print("\n" + "=" * 60)
    print("2. 光阑应用与透过率计算演示")
    print("=" * 60)
    
    # 定义光束参数
    wavelength = 633e-9  # 633 nm
    w0 = 1e-3            # 1 mm 束腰
    grid_size = 256
    
    # 定义光阑比例（光阑半径/光束半径）
    aperture_ratios = [0.8, 1.0, 1.2, 1.5, 2.0]
    
    print(f"\n光束参数:")
    print(f"  波长 λ = {wavelength * 1e9:.1f} nm")
    print(f"  束腰 w₀ = {w0 * 1e3:.2f} mm")
    print(f"\n光阑比例 (a/w): {aperture_ratios}")
    
    # 测试四种光阑类型
    aperture_types = [
        ("hard_edge", "硬边光阑"),
        ("gaussian", "高斯光阑"),
        ("super_gaussian", "超高斯光阑"),
        ("eighth_order", "8阶光阑"),
    ]
    
    print("\n" + "-" * 60)
    print("各光阑类型的能量透过率:")
    print("-" * 60)
    
    for apt_type, apt_name in aperture_types:
        print(f"\n{apt_name}:")
        print(f"  {'比例':^6} | {'实际透过率':^12} | {'理论透过率':^12} | {'误差':^8}")
        print(f"  {'-'*6} | {'-'*12} | {'-'*12} | {'-'*8}")
        
        for ratio in aperture_ratios:
            # 创建新的波前
            wfo = create_gaussian_beam(wavelength, w0, grid_size)
            
            # 计算光阑半径
            aperture_radius = ratio * w0
            
            # 创建光阑对象
            apt_type_enum = ApertureType(apt_type)
            
            if apt_type == "gaussian":
                aperture = CircularAperture(
                    aperture_type=apt_type_enum,
                    radius=aperture_radius,
                    gaussian_sigma=aperture_radius,
                )
            elif apt_type == "super_gaussian":
                aperture = CircularAperture(
                    aperture_type=apt_type_enum,
                    radius=aperture_radius,
                    super_gaussian_order=4,
                )
            else:
                aperture = CircularAperture(
                    aperture_type=apt_type_enum,
                    radius=aperture_radius,
                )
            
            # 计算能量透过率
            result = aperture.calculate_power_transmission(wfo, w0)
            
            print(f"  {ratio:^6.2f} | {result.actual_transmission*100:^12.2f}% | "
                  f"{result.theoretical_transmission*100:^12.2f}% | "
                  f"{result.relative_error*100:^8.2f}%")



def demo_beam_propagation_analysis():
    """演示光束传播分析
    
    分析光束直径随传输距离的变化，并计算远场发散角。
    使用 PROPER 的 prop_get_beamradius 获取光束半径。
    """
    print("\n" + "=" * 60)
    print("3. 光束传播分析演示")
    print("=" * 60)
    
    # 定义光束参数
    wavelength = 633e-9  # 633 nm
    w0 = 1e-3            # 1 mm 束腰
    grid_size = 256
    
    # 计算瑞利距离
    z_rayleigh = np.pi * w0**2 / wavelength
    
    # 计算理论发散角
    theoretical_divergence = wavelength / (np.pi * w0)
    
    print(f"\n光束参数:")
    print(f"  波长 λ = {wavelength * 1e9:.1f} nm")
    print(f"  束腰 w₀ = {w0 * 1e3:.2f} mm")
    print(f"  瑞利距离 z_R = {z_rayleigh * 1e3:.2f} mm")
    print(f"  理论发散角 θ = {theoretical_divergence * 1e3:.4f} mrad")
    
    # 创建传播分析器
    analyzer = BeamPropagationAnalyzer(
        wavelength=wavelength,
        w0=w0,
        grid_size=grid_size,
        measurement_method="ideal",  # 使用 prop_get_beamradius
    )
    
    # 定义传输位置（从 0 到 10 倍瑞利距离）
    z_positions = [0, 0.5 * z_rayleigh, z_rayleigh, 2 * z_rayleigh, 
                   5 * z_rayleigh, 10 * z_rayleigh]
    
    print(f"\n传输位置 (相对于 z_R):")
    for z in z_positions:
        print(f"  z = {z / z_rayleigh:.1f} × z_R = {z * 1e3:.2f} mm")
    
    # 执行传播分析
    result = analyzer.analyze(z_positions)
    
    # 显示结果
    print("\n光束直径随传输距离变化:")
    print(f"  {'z (mm)':^10} | {'z/z_R':^8} | {'D_mean (mm)':^12} | {'理论值 (mm)':^12}")
    print(f"  {'-'*10} | {'-'*8} | {'-'*12} | {'-'*12}")
    
    for dp in result.data_points:
        theoretical_d = analyzer.theoretical_beam_diameter(dp.z)
        print(f"  {dp.z * 1e3:^10.2f} | {dp.z / z_rayleigh:^8.2f} | "
              f"{dp.d_mean * 1e3:^12.4f} | {theoretical_d * 1e3:^12.4f}")
    
    print(f"\n远场发散角:")
    print(f"  测量值 θ_mean = {result.divergence_mean * 1e3:.4f} mrad")
    print(f"  理论值 θ = {theoretical_divergence * 1e3:.4f} mrad")
    
    return wavelength, w0, grid_size


def demo_aperture_effect_analysis():
    """演示光阑影响分析
    
    对比不同光阑类型和尺寸对光束传输的影响。
    """
    print("\n" + "=" * 60)
    print("4. 光阑影响分析演示")
    print("=" * 60)
    
    # 定义光束参数
    wavelength = 633e-9  # 633 nm
    w0 = 1e-3            # 1 mm 束腰
    grid_size = 256
    
    # 计算瑞利距离
    z_rayleigh = np.pi * w0**2 / wavelength
    
    print(f"\n光束参数:")
    print(f"  波长 λ = {wavelength * 1e9:.1f} nm")
    print(f"  束腰 w₀ = {w0 * 1e3:.2f} mm")
    print(f"  瑞利距离 z_R = {z_rayleigh * 1e3:.2f} mm")
    
    # 定义光阑比例范围（0.8～1.5 倍光束直径）
    aperture_ratios = [0.8, 1.0, 1.2, 1.5]
    
    print(f"\n光阑比例 (a/w): {aperture_ratios}")
    
    # 创建光阑影响分析器
    analyzer = ApertureEffectAnalyzer(
        wavelength=wavelength,
        w0=w0,
        grid_size=grid_size,
        propagation_distance=10 * z_rayleigh,  # 传播到远场
    )
    
    # 执行分析（分析全部四种光阑类型）
    result = analyzer.analyze(
        aperture_ratios=aperture_ratios,
        aperture_types=[
            ApertureType.HARD_EDGE,
            ApertureType.GAUSSIAN,
            ApertureType.SUPER_GAUSSIAN,
            ApertureType.EIGHTH_ORDER,
        ],
    )
    
    # 显示分析结果
    print("\n" + "-" * 80)
    print("光阑影响分析结果:")
    print("-" * 80)
    print(f"  {'光阑类型':^12} | {'比例':^6} | {'透过率':^10} | "
          f"{'直径变化':^10} | {'发散角变化':^10}")
    print(f"  {'-'*12} | {'-'*6} | {'-'*10} | {'-'*10} | {'-'*10}")
    
    for dp in result.data_points:
        type_name = {
            ApertureType.HARD_EDGE: "硬边光阑",
            ApertureType.GAUSSIAN: "高斯光阑",
            ApertureType.SUPER_GAUSSIAN: "超高斯光阑",
            ApertureType.EIGHTH_ORDER: "8阶光阑",
        }.get(dp.aperture_type, dp.aperture_type.value)
        
        print(f"  {type_name:^12} | {dp.aperture_ratio:^6.2f} | "
              f"{dp.power_transmission*100:^10.2f}% | "
              f"{dp.beam_diameter_change*100:^+10.2f}% | "
              f"{dp.divergence_change*100:^+10.2f}%")
    
    return result



def demo_theoretical_comparison():
    """演示测量结果与理论对比
    
    使用 ComparisonModule 对比测量值与理论值。
    """
    print("\n" + "=" * 60)
    print("5. 测量结果与理论对比演示")
    print("=" * 60)
    
    # 定义光束参数
    wavelength = 633e-9  # 633 nm
    w0 = 1e-3            # 1 mm 束腰
    grid_size = 256
    
    # 计算瑞利距离
    z_rayleigh = np.pi * w0**2 / wavelength
    
    print(f"\n光束参数:")
    print(f"  波长 λ = {wavelength * 1e9:.1f} nm")
    print(f"  束腰 w₀ = {w0 * 1e3:.2f} mm")
    print(f"  瑞利距离 z_R = {z_rayleigh * 1e3:.2f} mm")
    
    # 创建对比模块
    comparison = ComparisonModule(wavelength=wavelength, w0=w0)
    
    # 创建传播分析器获取测量数据
    analyzer = BeamPropagationAnalyzer(
        wavelength=wavelength,
        w0=w0,
        grid_size=grid_size,
        measurement_method="ideal",
    )
    
    # 定义传输位置
    z_positions = [0, 0.5 * z_rayleigh, z_rayleigh, 2 * z_rayleigh, 
                   5 * z_rayleigh, 10 * z_rayleigh]
    
    # 执行传播分析
    prop_result = analyzer.analyze(z_positions)
    
    # 提取测量数据
    z_array = np.array([dp.z for dp in prop_result.data_points])
    measured_diameters = np.array([dp.d_mean for dp in prop_result.data_points])
    
    # 对比测量值与理论值
    comp_result = comparison.compare_beam_diameters(z_array, measured_diameters)
    
    # 显示对比结果
    print("\n光束直径对比:")
    print(f"  {'z (mm)':^10} | {'测量值 (mm)':^12} | {'理论值 (mm)':^12} | {'相对误差':^10}")
    print(f"  {'-'*10} | {'-'*12} | {'-'*12} | {'-'*10}")
    
    for i, (z, measured, theoretical, error) in enumerate(zip(
        z_array, comp_result.measured_values, 
        comp_result.theoretical_values, comp_result.relative_errors
    )):
        print(f"  {z * 1e3:^10.2f} | {measured * 1e3:^12.4f} | "
              f"{theoretical * 1e3:^12.4f} | {error * 100:^10.4f}%")
    
    print(f"\n误差统计:")
    print(f"  RMS 相对误差 = {comp_result.rms_error * 100:.4f}%")
    print(f"  最大相对误差 = {comp_result.max_error * 100:.4f}%")
    
    # 计算菲涅尔数示例
    print("\n菲涅尔数计算示例:")
    aperture_radius = 1.0 * w0  # 光阑半径 = 光束半径
    propagation_distance = 10 * z_rayleigh
    
    fresnel_number = comparison.calculate_fresnel_number(
        aperture_radius, propagation_distance
    )
    diffraction_effect = comparison.estimate_diffraction_effect(fresnel_number)
    
    print(f"  光阑半径 a = {aperture_radius * 1e3:.2f} mm")
    print(f"  传播距离 z = {propagation_distance * 1e3:.2f} mm")
    print(f"  菲涅尔数 N_F = {fresnel_number:.4f}")
    print(f"  衍射效应: {diffraction_effect}")
    
    return comp_result


def demo_report_generation(aperture_result):
    """演示测试报告生成
    
    使用 ReportGenerator 生成完整的 Markdown 格式测试报告。
    
    参数:
        aperture_result: 光阑影响分析结果
    """
    print("\n" + "=" * 60)
    print("6. 测试报告生成演示")
    print("=" * 60)
    
    # 创建输出目录
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建报告生成器
    report_generator = ReportGenerator(output_dir=output_dir)
    
    # 生成报告
    report_content = report_generator.generate(
        aperture_analysis=aperture_result,
        title="光束参数测量与光阑分析报告",
    )
    
    # 保存报告
    report_path = report_generator.save(
        report_content, 
        filename="beam_aperture_analysis_report.md"
    )
    
    print(f"\n报告已生成并保存到: {report_path}")
    
    # 显示报告预览（前 50 行）
    print("\n报告预览 (前 50 行):")
    print("-" * 60)
    lines = report_content.split('\n')
    for line in lines[:50]:
        print(line)
    if len(lines) > 50:
        print(f"\n... (共 {len(lines)} 行)")
    
    return report_path



def demo_api_functions():
    """演示 BTS API 函数的使用
    
    展示如何使用 analyze_aperture_effects() API 函数进行完整分析。
    """
    print("\n" + "=" * 60)
    print("7. BTS API 函数演示")
    print("=" * 60)
    
    # 定义光束参数
    wavelength = 633e-9  # 633 nm
    w0 = 1e-3            # 1 mm 束腰
    
    print(f"\n使用 analyze_aperture_effects() API 函数:")
    print(f"  波长 λ = {wavelength * 1e9:.1f} nm")
    print(f"  束腰 w₀ = {w0 * 1e3:.2f} mm")
    
    # 使用 API 函数进行分析
    result = analyze_aperture_effects(
        wavelength=wavelength,
        w0=w0,
        aperture_ratios=[0.8, 1.0, 1.2, 1.5, 2.0],
        aperture_types=["hard_edge", "gaussian"],  # 使用字符串格式
        grid_size=256,
        generate_report=True,
        output_dir="output",
    )
    
    print(f"\n分析完成！")
    print(f"  分析的光阑类型数: {len(result.aperture_types)}")
    print(f"  分析的光阑比例数: {len(result.aperture_ratios)}")
    print(f"  总数据点数: {len(result.data_points)}")
    
    # 显示选型建议
    print("\n选型建议:")
    print(result.recommendation)
    
    return result


def main():
    """主函数
    
    执行完整的光束参数测量与光阑分析工作流程。
    """
    print("=" * 60)
    print("光束参数测量与光阑分析示例")
    print("=" * 60)
    print("\n本示例演示 BTS 光束测量模块的完整功能：")
    print("  1. 光束直径测量（D4sigma 方法）")
    print("  2. 光阑应用与透过率计算")
    print("  3. 光束传播分析")
    print("  4. 光阑影响对比分析")
    print("  5. 测量结果与理论对比")
    print("  6. 测试报告生成")
    print("  7. BTS API 函数演示")
    
    # ============================================================
    # 1. 光束直径测量演示
    # ============================================================
    demo_beam_diameter_measurement()
    
    # ============================================================
    # 2. 光阑应用与透过率计算演示
    # ============================================================
    demo_aperture_application()
    
    # ============================================================
    # 3. 光束传播分析演示
    # ============================================================
    demo_beam_propagation_analysis()
    
    # ============================================================
    # 4. 光阑影响分析演示
    # ============================================================
    aperture_result = demo_aperture_effect_analysis()
    
    # ============================================================
    # 5. 测量结果与理论对比演示
    # ============================================================
    demo_theoretical_comparison()
    
    # ============================================================
    # 6. 测试报告生成演示
    # ============================================================
    demo_report_generation(aperture_result)
    
    # ============================================================
    # 7. BTS API 函数演示
    # ============================================================
    demo_api_functions()
    
    # ============================================================
    # 完成
    # ============================================================
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)
    print("\n生成的文件:")
    print("  - output/beam_aperture_analysis_report.md")
    print("  - output/aperture_analysis_report.md")


if __name__ == "__main__":
    main()
