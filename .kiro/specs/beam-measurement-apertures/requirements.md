# Requirements Document

## Introduction

本功能为 BTS 混合光学仿真系统提供光束参数测量与光阑设置能力。主要包括：
1. D4sigma 光束直径测量（理想方法与 ISO 11146 标准方法）
2. M² 因子测量
3. 圆形光阑支持（硬边、高斯、超高斯/软边、8 阶软边四种振幅透过率设置方法）
4. 光束参数随传输距离变化的测量与分析

该功能基于 PROPER 库实现，通过 BTS API 提供统一的访问接口。

## Glossary

- **D4sigma**: 光束直径的二阶矩定义，等于光束强度分布二阶矩的 4 倍标准差
- **M2_Factor**: M² 因子，表征实际光束与理想高斯光束的偏离程度
- **ISO_11146**: 国际标准化组织发布的激光光束参数测量标准
- **Circular_Aperture**: 圆形光阑，用于限制光束通过区域的光学元件
- **Hard_Edge_Aperture**: 硬边光阑，透过率为 0 或 1 的理想圆形光阑，使用 PROPER 的 prop_circular_aperture 实现
- **Gaussian_Aperture**: 高斯光阑，透过率按高斯函数 T(r) = exp(-0.5 × (r/σ)²) 分布
- **Super_Gaussian_Aperture**: 超高斯/软边光阑，透过率按 T(r) = exp(-(r/r₀)ⁿ) 分布，n 为超高斯阶数
- **Eighth_Order_Mask**: 8 阶软边光阑，使用 PROPER 的 prop_8th_order_mask 实现，基于 sinc 函数的 8 阶透过率分布
- **Beam_Waist**: 束腰，高斯光束最小光斑位置
- **Divergence_Angle**: 发散角，光束在远场的扩展角度
- **ROI**: Region of Interest，感兴趣区域
- **Second_Moment**: 二阶矩，用于计算光束直径的统计量
- **Pilot_Beam**: 导引光束，用于参考的理想高斯光束参数
- **PROPER_Wavefront**: PROPER 库中的波前对象（wfo）
- **HWHM**: Half Width at Half Maximum，半高半宽，8 阶光阑的特征尺寸参数

## Requirements

### Requirement 1: D4sigma 光束直径测量（理想方法）

**User Story:** As a 光学工程师, I want to 使用理想二阶矩方法测量光束直径, so that I can 获得精确的光束尺寸信息。

#### Acceptance Criteria

1. WHEN 提供 PROPER 波前对象或复振幅数组 THEN THE D4sigma_Calculator SHALL 计算 X 和 Y 方向的二阶矩光束直径
2. WHEN 计算二阶矩 THEN THE D4sigma_Calculator SHALL 使用公式 D4σ = 4 × √(∫∫ I(x,y) × (x-x̄)² dxdy / ∫∫ I(x,y) dxdy)
3. WHEN 光束为理想高斯光束 THEN THE D4sigma_Calculator SHALL 返回与 2×w（1/e² 半径的 2 倍）一致的结果
4. THE D4sigma_Calculator SHALL 支持输入 PROPER 波前对象或 numpy 复振幅数组
5. THE D4sigma_Calculator SHALL 返回包含 dx、dy、d_mean（平均直径）的测量结果

### Requirement 2: D4sigma 光束直径测量（ISO 11146 标准方法）

**User Story:** As a 光学工程师, I want to 使用 ISO 11146 标准方法测量光束直径, so that I can 获得与实际光束测量仪一致的结果。

#### Acceptance Criteria

1. WHEN 提供复振幅数据 THEN THE ISO_D4sigma_Calculator SHALL 执行背景噪声估计与去除
2. WHEN 执行 ISO 标准测量 THEN THE ISO_D4sigma_Calculator SHALL 使用迭代 ROI 方法确定有效测量区域
3. WHEN ROI 迭代 THEN THE ISO_D4sigma_Calculator SHALL 使用 3 倍 D4sigma 作为 ROI 边界进行迭代直到收敛
4. WHEN 迭代收敛 THEN THE ISO_D4sigma_Calculator SHALL 返回最终的 D4sigma 值
5. IF 迭代未在最大次数内收敛 THEN THE ISO_D4sigma_Calculator SHALL 返回警告信息并给出当前最佳估计
6. THE ISO_D4sigma_Calculator SHALL 支持配置最大迭代次数和收敛阈值

### Requirement 3: M² 因子测量

**User Story:** As a 光学工程师, I want to 测量光束的 M² 因子, so that I can 评估光束质量与理想高斯光束的偏离程度。

#### Acceptance Criteria

1. WHEN 提供多个传输位置的光束直径数据 THEN THE M2_Calculator SHALL 通过曲线拟合计算 M² 因子
2. WHEN 拟合光束直径变化曲线 THEN THE M2_Calculator SHALL 使用公式 w(z)² = w₀² × [1 + (M² × λ × (z-z₀) / (π × w₀²))²]
3. THE M2_Calculator SHALL 返回 M² 值、拟合束腰 w₀、束腰位置 z₀ 和拟合优度 R²
4. WHEN 测量点数少于 5 个 THEN THE M2_Calculator SHALL 返回警告信息
5. THE M2_Calculator SHALL 支持 X 和 Y 方向独立的 M² 计算

### Requirement 4: 圆形光阑（多种振幅透过率设置方法）

**User Story:** As a 光学工程师, I want to 在光束路径中添加不同类型的圆形光阑, so that I can 模拟实际光学系统中的各种孔径限制效果。

#### Acceptance Criteria

1. WHEN 指定硬边圆形光阑 THEN THE Circular_Aperture SHALL 使用 PROPER 的 prop_circular_aperture 函数应用抗锯齿硬边光阑
2. WHEN 指定高斯光阑 THEN THE Circular_Aperture SHALL 应用透过率函数 T(r) = exp(-0.5 × (r/σ)²)，其中 σ 为高斯半径参数
3. WHEN 指定超高斯/软边光阑 THEN THE Circular_Aperture SHALL 应用透过率函数 T(r) = exp(-(r/r₀)ⁿ)，其中 r₀ 为特征半径，n 为超高斯阶数
4. WHEN 指定 8 阶软边光阑 THEN THE Circular_Aperture SHALL 使用 PROPER 的 prop_8th_order_mask 函数，支持 HWHM（半高半宽）参数配置
5. THE Circular_Aperture SHALL 支持以米为单位的绝对半径或相对于光束半径的归一化半径
6. THE Circular_Aperture SHALL 支持指定光阑中心位置（xc, yc）
7. THE Circular_Aperture SHALL 支持通过 BTS API 在仿真流程中使用
8. WHEN 使用超高斯光阑 THEN THE Circular_Aperture SHALL 支持配置阶数 n（n=2 为高斯，n→∞ 趋近硬边）
9. WHEN 使用 8 阶光阑 THEN THE Circular_Aperture SHALL 支持配置最小和最大透过率参数
10. THE Circular_Aperture SHALL 提供 calculate_power_transmission() 方法计算高斯光束通过光阑后的能量透过率
11. WHEN 计算能量透过率 THEN THE Circular_Aperture SHALL 返回实际透过率与理论透过率的对比结果


### Requirement 5: 光束参数随传输距离变化测量

**User Story:** As a 光学工程师, I want to 测量光束参数随传输距离的变化, so that I can 分析光束的传输特性和发散行为。

#### Acceptance Criteria

1. WHEN 指定传输距离列表 THEN THE Beam_Propagation_Analyzer SHALL 在每个位置测量光束直径
2. THE Beam_Propagation_Analyzer SHALL 支持使用理想方法或 ISO 标准方法测量各位置的光束直径
3. THE Beam_Propagation_Analyzer SHALL 返回包含位置、光束直径、测量方法的完整数据集
4. THE Beam_Propagation_Analyzer SHALL 支持自动计算远场发散角
5. WHEN 测量完成 THEN THE Beam_Propagation_Analyzer SHALL 支持可选的可视化绘图输出

### Requirement 6: 光阑对光束的影响分析

**User Story:** As a 光学工程师, I want to 分析不同尺寸和类型光阑对光束传输的影响, so that I can 为光学系统设计选择合适的光阑配置。

#### Acceptance Criteria

1. WHEN 指定光阑尺寸范围（如 0.8～1.5 倍光束直径） THEN THE Aperture_Effect_Analyzer SHALL 对每个尺寸执行传输仿真
2. THE Aperture_Effect_Analyzer SHALL 支持对硬边、高斯、超高斯、8 阶软边四种光阑类型进行对比分析
3. THE Aperture_Effect_Analyzer SHALL 测量通过光阑后的光束直径变化
4. THE Aperture_Effect_Analyzer SHALL 测量远场发散角变化
5. THE Aperture_Effect_Analyzer SHALL 计算功率透过率
6. THE Aperture_Effect_Analyzer SHALL 生成包含理论对比的分析报告
7. WHEN 分析完成 THEN THE Aperture_Effect_Analyzer SHALL 提供光阑类型与尺寸的选型建议

### Requirement 7: BTS API 集成

**User Story:** As a 开发者, I want to 通过 BTS API 访问光束测量和光阑功能, so that I can 在仿真流程中方便地使用这些功能。

#### Acceptance Criteria

1. THE BTS_API SHALL 提供 measure_beam_diameter() 函数用于测量光束直径
2. THE BTS_API SHALL 提供 measure_m2() 函数用于测量 M² 因子
3. THE BTS_API SHALL 提供 apply_aperture() 函数用于应用光阑，支持指定光阑类型（hard_edge、gaussian、super_gaussian、eighth_order）
4. THE BTS_API SHALL 支持从 SimulationResult 中提取波前数据进行测量
5. WHEN 调用测量函数 THEN THE BTS_API SHALL 返回结构化的测量结果对象
6. THE BTS_API SHALL 提供 analyze_aperture_effects() 函数用于光阑影响分析


### Requirement 8: 测量结果与理论对比

**User Story:** As a 光学工程师, I want to 将测量结果与理论预期进行对比, so that I can 验证仿真的准确性并理解物理行为。

#### Acceptance Criteria

1. WHEN 测量高斯光束 THEN THE Comparison_Module SHALL 计算理论光束直径 w(z) = w₀ × √(1 + (z/z_R)²)
2. THE Comparison_Module SHALL 计算测量值与理论值的相对误差
3. WHEN 光束通过光阑 THEN THE Comparison_Module SHALL 基于菲涅尔数估算衍射效应
4. THE Comparison_Module SHALL 生成包含测量值、理论值、误差分析的对比报告
5. THE Comparison_Module SHALL 支持可视化绘图输出

### Requirement 9: 测试报告生成

**User Story:** As a 光学工程师, I want to 生成完整的测试报告, so that I can 记录和分享测量结果与分析结论。

#### Acceptance Criteria

1. THE Report_Generator SHALL 生成包含测试配置、测量数据、分析结果的完整报告
2. THE Report_Generator SHALL 支持 Markdown 格式输出
3. THE Report_Generator SHALL 包含可视化图表（光束直径变化曲线、M² 拟合曲线等）
4. THE Report_Generator SHALL 包含光阑选型建议和结论
5. WHEN 执行多种光阑类型与测量方法组合测试 THEN THE Report_Generator SHALL 生成对比分析表格
