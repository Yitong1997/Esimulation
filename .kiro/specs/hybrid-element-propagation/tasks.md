# 实现计划：混合元件传播 (Hybrid Element Propagation)

## 概述

本实现计划将设计文档中的混合元件传播功能分解为可执行的编码任务。实现采用 Python 语言，复用现有模块（WavefrontToRaysSampler、ElementRaytracer、tilted_asm 等）。

采用测试驱动开发（TDD）方法，每个功能模块的实现都伴随相应的测试。

## 任务

- [x] 1. 创建模块结构和基础设施
  - [x] 1.1 创建 `src/hybrid_propagation/` 目录和 `__init__.py`
    - 创建模块目录结构
    - 导出主要类和函数
    - _Requirements: 9.1_
  
  - [x] 1.2 创建 `PilotBeamWarning` 警告类
    - 在 `src/sequential_system/exceptions.py` 中添加新的警告类
    - _Requirements: 5.2, 5.4, 5.6_
  
  - [x] 1.3 创建测试目录结构
    - 创建 `tests/test_hybrid_propagation/` 目录
    - 创建 `__init__.py` 和 `conftest.py`
    - 定义通用测试 fixtures

- [x] 2. 实现 TiltedPropagation 模块
  - [x] 2.1 实现 `TiltedPropagation` 类基础结构
    - 封装 `tilted_asm` 函数
    - 实现 `_compute_rotation_matrix` 方法
    - _Requirements: 1.1, 1.3_
  
  - [x] 2.2 实现 `propagate_to_tangent_plane` 方法
    - 从入射面传播到切平面
    - 处理正入射情况（无倾斜）
    - _Requirements: 1.1, 1.2, 1.5_
  
  - [x] 2.3 实现 `propagate_from_tangent_plane` 方法
    - 从切平面传播到出射面
    - 处理反射元件的光轴方向变化
    - _Requirements: 8.1, 8.2, 8.3_
  
  - [x] 2.4 编写 TiltedPropagation 单元测试
    - 测试旋转矩阵计算正确性
    - 测试正入射情况的等价性
    - 测试能量守恒
    - _Requirements: 1.2, 1.3, 1.4_
  
  - [x] 2.5 编写 TiltedPropagation 属性测试
    - **Property 2: 旋转矩阵正交性**
    - **Property 15: 正入射等价性**
    - **Validates: Requirements 1.2, 1.3, 8.2, 8.3**

- [x] 3. 实现 PilotBeamValidator 模块
  - [x] 3.1 实现 `ValidationResult` 和 `PilotBeamValidationResult` 数据类
    - 定义验证结果数据结构
    - _Requirements: 5.8_
  
  - [x] 3.2 实现 `PilotBeamValidator` 类
    - 实现 `check_phase_sampling` 方法（检查相邻像素相位差）
    - 实现 `check_beam_divergence` 方法
    - 实现 `check_beam_size_match` 方法
    - 实现 `validate_all` 方法
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8_
  
  - [x] 3.3 编写 PilotBeamValidator 单元测试
    - 测试相位采样检测功能
    - 测试发散角检测功能
    - 测试尺寸匹配检测功能
    - _Requirements: 5.1, 5.3, 5.5_
  
  - [x] 3.4 编写 PilotBeamValidator 属性测试
    - **Property 7: 相位采样质量**
    - **Property 8: Pilot Beam 适用性检测**
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7**

- [x] 4. 实现 PilotBeamCalculator 模块
  - [x] 4.1 实现 `PilotBeamCalculator` 类基础结构
    - 初始化参数和配置
    - _Requirements: 4.1_
  
  - [x] 4.2 实现 `compute_reference_phase_proper` 方法（方案 A）
    - 使用 PROPER 计算高斯光束传播
    - 使用 tilted_asm 反向传播
    - _Requirements: 4.2_
  
  - [x] 4.3 实现 `compute_reference_phase_analytical` 方法（方案 B）
    - 解析计算高斯光束在镜面反射后的相位
    - _Requirements: 4.3_
  
  - [x] 4.4 实现 `compute_reference_phase` 统一接口
    - 根据 method 参数选择计算方法
    - 集成 PilotBeamValidator 验证
    - _Requirements: 4.4, 4.5_
  
  - [x] 4.5 编写 PilotBeamCalculator 单元测试
    - 测试两种方法的输出格式
    - 测试参考相位的合理性
    - 测试方法切换功能
    - _Requirements: 4.2, 4.3, 4.4_

- [x] 5. Checkpoint - 验证 Pilot Beam 模块
  - 运行所有 Pilot Beam 相关测试
  - 确保所有测试通过，如有问题请询问用户

- [x] 6. 实现 PhaseCorrector 模块
  - [x] 6.1 实现 `PhaseCorrector` 类
    - 实现 `interpolate_reference_phase` 方法（双线性插值）
    - 实现 `compute_residual_phase` 方法（相位包裹到 [-π, π]）
    - 实现 `correct_ray_phase` 方法
    - 实现 `check_residual_range` 方法
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_
  
  - [x] 6.2 编写 PhaseCorrector 单元测试
    - 测试插值准确性
    - 测试残差计算正确性
    - 测试相位包裹功能
    - _Requirements: 6.1, 6.2, 6.4_
  
  - [x] 6.3 编写 PhaseCorrector 属性测试
    - **Property 9: 残差相位包裹**
    - **Property 10: 相位插值一致性**
    - **Validates: Requirements 6.1, 6.4**

- [x] 7. 实现 AmplitudeReconstructor 模块
  - [x] 7.1 实现 `AmplitudeReconstructor` 类
    - 实现 `_interpolate_to_grid` 方法（使用 scipy.interpolate.griddata）
    - 实现 `_apply_reference_phase` 方法
    - 实现 `reconstruct` 主方法
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_
  
  - [x] 7.2 编写 AmplitudeReconstructor 单元测试
    - 测试网格插值功能
    - 测试参考相位加回功能
    - 测试无效区域处理
    - _Requirements: 7.2, 7.3, 7.5_
  
  - [x] 7.3 编写 AmplitudeReconstructor 属性测试
    - **Property 11: 复振幅重建完整性**
    - **Property 12: 参考相位加回**
    - **Validates: Requirements 7.2, 7.3, 7.4, 7.5**

- [x] 8. 实现 HybridElementPropagator 主类
  - [x] 8.1 实现 `HybridElementPropagator` 类初始化
    - 参数验证
    - 初始化各子模块
    - _Requirements: 9.2, 11.1, 11.2, 11.3_
  
  - [x] 8.2 实现 `propagate` 主方法
    - 集成完整的传播流程
    - 复用 WavefrontToRaysSampler 和 ElementRaytracer
    - _Requirements: 9.1, 9.3_
  
  - [x] 8.3 实现 `get_intermediate_results` 方法
    - 返回中间结果字典
    - _Requirements: 9.4_
  
  - [x] 8.4 实现调试模式
    - 可选的详细日志输出
    - _Requirements: 9.5_
  
  - [x] 8.5 编写 HybridElementPropagator 单元测试
    - 测试参数验证
    - 测试完整传播流程
    - 测试中间结果获取
    - _Requirements: 9.2, 9.3, 9.4, 11.1, 11.2, 11.3_
  
  - [x] 8.6 编写 HybridElementPropagator 属性测试
    - **Property 1: 能量守恒**
    - **Property 13: 输入验证**
    - **Property 14: 全光线无效处理**
    - **Validates: Requirements 1.4, 8.4, 11.1, 11.2, 11.3, 11.4**

- [x] 9. Checkpoint - 验证核心功能
  - 运行所有核心模块测试
  - 确保所有测试通过，如有问题请询问用户

- [x] 10. SequentialOpticalSystem 集成
  - [x] 10.1 修改 `SequentialOpticalSystem._apply_element` 方法
    - 添加混合传播模式支持
    - 保持向后兼容性
    - _Requirements: 10.1, 10.2, 10.4_
  
  - [x] 10.2 实现坐标系转换集成
    - 正确处理元件前后的坐标系转换
    - _Requirements: 10.3_
  
  - [x] 10.3 编写集成单元测试
    - 测试单元件系统
    - 测试多元件系统
    - 测试向后兼容性
    - _Requirements: 10.1, 10.2, 10.4_
  
  - [x] 10.4 编写集成属性测试
    - **Property 16: 坐标系转换往返一致性**
    - **Validates: Requirements 10.1, 10.2, 10.3, 10.4**

- [-] 11. 最终验证
  - [x] 11.1 编写端到端集成测试
    - 平面镜正入射测试
    - 凹面镜正入射测试
    - 45° 折叠镜测试
    - _Requirements: 1.1, 3.3, 8.1_
  
  - [x] 11.2 编写能量守恒验证测试
    - 不同网格大小
    - 不同光线数量
    - 不同元件类型
    - **Validates: Requirements 1.4, 8.4**
  
  - [ ] 11.3 编写性能基准测试
    - 验证 512×512 网格、100 条光线配置下完成时间 < 5 秒
    - _Requirements: 12.1_

- [ ] 12. Final Checkpoint
  - 运行完整测试套件
  - 确保所有测试通过，如有问题请询问用户

## 测试文件结构

```
tests/test_hybrid_propagation/
├── __init__.py
├── conftest.py                          # 通用 fixtures
├── test_tilted_propagation.py           # TiltedPropagation 测试
├── test_tilted_propagation_properties.py # TiltedPropagation 属性测试
├── test_pilot_beam_validator.py         # PilotBeamValidator 测试
├── test_pilot_beam_validator_properties.py # PilotBeamValidator 属性测试
├── test_pilot_beam_calculator.py        # PilotBeamCalculator 测试
├── test_phase_corrector.py              # PhaseCorrector 测试
├── test_phase_corrector_properties.py   # PhaseCorrector 属性测试
├── test_amplitude_reconstructor.py      # AmplitudeReconstructor 测试
├── test_amplitude_reconstructor_properties.py # AmplitudeReconstructor 属性测试
├── test_hybrid_propagator.py            # HybridElementPropagator 测试
├── test_hybrid_propagator_properties.py # HybridElementPropagator 属性测试
├── test_integration.py                  # 集成测试
└── test_integration_properties.py       # 集成属性测试
```

## 注意事项

- 采用 TDD 方法，每个功能模块的测试与实现同步进行
- 每个属性测试必须引用设计文档中的属性编号
- 复用现有模块时注意导入路径
- 所有代码使用中文注释
- 属性测试使用 hypothesis 库，每个属性至少运行 100 次迭代

