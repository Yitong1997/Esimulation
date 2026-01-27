# Implementation Plan: 光束参数测量与光阑设置

## Overview

本实现计划将光束参数测量与光阑设置功能分解为可执行的编码任务。实现基于 PROPER 库，通过 BTS API 提供统一访问接口。

## Tasks

- [x] 1. 创建模块结构和数据模型
  - [x] 1.1 创建 `src/bts/beam_measurement/` 目录结构
    - 创建 `__init__.py`、`data_models.py`、`exceptions.py`
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_
  - [x] 1.2 实现数据模型类
    - 实现 `D4sigmaResult`、`ISOD4sigmaResult`、`M2Result`、`PowerTransmissionResult`
    - 实现 `ApertureType` 枚举
    - _Requirements: 1.5, 2.4, 3.3, 4.10_
  - [x] 1.3 实现异常类
    - 实现 `BeamMeasurementError`、`InvalidInputError`、`ConvergenceError`、`InsufficientDataError`
    - _Requirements: 2.5, 3.4_

- [x] 2. 实现 D4sigma 计算器
  - [x] 2.1 实现 `D4sigmaCalculator` 类
    - 实现 `calculate()` 方法，支持 numpy 数组和 PROPER 波前对象输入
    - 实现二阶矩计算公式 D4σ = 4 × √(∫∫ I(x,y) × (x-x̄)² dxdy / ∫∫ I(x,y) dxdy)
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
  - [ ]* 2.2 编写 D4sigma 属性测试
    - **Property 1: D4sigma 高斯光束验证**
    - **Validates: Requirements 1.2, 1.3**

- [x] 3. 实现 ISO D4sigma 计算器
  - [x] 3.1 实现 `ISOD4sigmaCalculator` 类
    - 实现背景噪声估计 `_estimate_background()`
    - 实现 ROI 掩模应用 `_apply_roi()`
    - 实现迭代 ROI 方法的 `calculate()`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_
  - [ ]* 3.2 编写 ISO D4sigma 属性测试
    - **Property 2: ISO D4sigma 去噪有效性**
    - **Validates: Requirements 2.1**

- [x] 4. Checkpoint - 确保 D4sigma 测试通过
  - 确保所有 D4sigma 相关测试通过，如有问题请询问用户


- [x] 5. 实现 M² 计算器
  - [x] 5.1 实现 `M2Calculator` 类
    - 实现光束因果曲线拟合 `_fit_beam_caustic()`
    - 实现 `calculate()` 方法，使用公式 w(z)² = w₀² × [1 + (M² × λ × (z-z₀) / (π × w₀²))²]
    - 实现测量点数不足时的警告
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
  - [ ]* 5.2 编写 M² 属性测试
    - **Property 3: M² 拟合正确性**
    - **Validates: Requirements 3.1, 3.2**

- [x] 6. 实现圆形光阑
  - [x] 6.1 实现 `CircularAperture` 类基础结构
    - 实现构造函数和参数验证
    - 实现 `apply()` 方法框架
    - _Requirements: 4.5, 4.6, 4.7_
  - [x] 6.2 实现硬边光阑
    - 实现 `_apply_hard_edge()` 方法，调用 PROPER 的 `prop_circular_aperture`
    - _Requirements: 4.1_
  - [x] 6.3 实现高斯光阑
    - 实现 `_apply_gaussian()` 方法，应用 T(r) = exp(-0.5 × (r/σ)²)
    - _Requirements: 4.2_
  - [x] 6.4 实现超高斯光阑
    - 实现 `_apply_super_gaussian()` 方法，应用 T(r) = exp(-(r/r₀)ⁿ)
    - _Requirements: 4.3, 4.8_
  - [x] 6.5 实现 8 阶软边光阑
    - 实现 `_apply_eighth_order()` 方法，调用 PROPER 的 `prop_8th_order_mask`
    - _Requirements: 4.4, 4.9_
  - [x] 6.6 实现能量透过率计算
    - 实现 `calculate_power_transmission()` 方法
    - 实现理论透过率计算 `_theoretical_transmission_hard_edge()` 等
    - _Requirements: 4.10, 4.11_
  - [ ]* 6.7 编写光阑属性测试
    - **Property 4: 光阑透过率函数验证**
    - **Property 5: 光阑半径归一化一致性**
    - **Property 6: 能量透过率计算正确性**
    - **Validates: Requirements 4.1-4.11**

- [x] 7. Checkpoint - 确保光阑测试通过
  - 确保所有光阑相关测试通过，如有问题请询问用户

- [x] 8. 实现光束传播分析器
  - [x] 8.1 实现 `BeamPropagationAnalyzer` 类
    - 实现 `analyze()` 方法，在各传输位置测量光束直径
    - 实现远场发散角计算 `_calculate_far_field_divergence()`
    - 实现可视化绘图 `plot()`
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
  - [ ]* 8.2 编写传播分析属性测试
    - **Property 7: 传播分析数据完整性**
    - **Property 8: 理论光束直径计算**
    - **Validates: Requirements 5.1, 5.4, 8.1**


- [x] 9. 实现光阑影响分析器
  - [x] 9.1 实现 `ApertureEffectAnalyzer` 类
    - 实现 `analyze()` 方法，对比四种光阑类型
    - 实现选型建议生成 `_generate_recommendation()`
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

- [x] 10. 实现对比模块
  - [x] 10.1 实现 `ComparisonModule` 类
    - 实现理论光束直径计算 `theoretical_beam_diameter()`
    - 实现菲涅尔数计算 `calculate_fresnel_number()`
    - 实现衍射效应估算 `estimate_diffraction_effect()`
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_
  - [ ]* 10.2 编写对比模块属性测试
    - **Property 9: 菲涅尔数计算正确性**
    - **Validates: Requirements 8.3**

- [x] 11. 实现报告生成器
  - [x] 11.1 实现 `ReportGenerator` 类
    - 实现 `generate()` 方法，生成 Markdown 格式报告
    - 实现配置信息、对比表格、选型建议等部分
    - 实现 `save()` 方法
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [x] 12. Checkpoint - 确保分析模块测试通过
  - 确保所有分析模块测试通过，如有问题请询问用户

- [x] 13. 实现 BTS API 集成
  - [x] 13.1 创建 `src/bts/beam_measurement.py` API 模块
    - 实现 `measure_beam_diameter()` 函数
    - 实现 `measure_m2()` 函数
    - 实现 `apply_aperture()` 函数
    - 实现 `analyze_aperture_effects()` 函数
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_
  - [x] 13.2 更新 `src/bts/__init__.py`
    - 导出新的 API 函数
    - _Requirements: 7.1, 7.2, 7.3, 7.6_
  - [ ]* 13.3 编写 BTS API 单元测试
    - 测试 API 函数存在性和参数验证
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

- [x] 14. 创建集成测试主文件
  - [x] 14.1 创建 `examples/beam_aperture_analysis.py`
    - 使用 BTS API 创建高斯光束
    - 应用不同类型和尺寸的光阑（0.8～1.5 倍光束直径）
    - 测量光束直径随传输距离变化
    - 测量远场发散角
    - 计算能量透过率并与理论对比
    - 生成完整测试报告
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 9.1, 9.2, 9.3, 9.4, 9.5_

- [x] 15. Final Checkpoint - 确保所有测试通过
  - 确保所有测试通过，如有问题请询问用户

## Notes

- 任务标记 `*` 为可选测试任务，可跳过以加快 MVP 开发
- 每个任务引用具体的需求条款以确保可追溯性
- Checkpoint 任务用于增量验证
- 属性测试验证普遍正确性属性
- 单元测试验证特定示例和边界条件
