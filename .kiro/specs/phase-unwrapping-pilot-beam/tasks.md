# 实现计划：基于 Pilot Beam 的相位解包裹

## 概述

本实现计划将设计文档中的相位解包裹方案转化为具体的编码任务。采用增量开发方式，每个任务都建立在前一个任务的基础上，确保代码始终可运行。

## 任务列表

- [ ] 1. 创建 PhaseUnwrapper 核心类
  - [ ] 1.1 创建 `src/hybrid_propagation/phase_unwrapper.py` 文件
    - 定义 `PhaseUnwrapper` 类
    - 实现 `__init__` 方法，接受 `PilotBeamCalculator` 实例
    - 实现 `unwrap()` 方法，使用公式 `T_pilot + angle(T - T_pilot)`
    - 添加类型注解和文档字符串
    - _Requirements: 1.1, 1.2, 1.3_
  
  - [ ] 1.2 实现 `UnwrapValidationResult` 数据类
    - 在 `src/hybrid_propagation/__init__.py` 中定义
    - 包含 `is_valid`, `max_phase_diff`, `has_discontinuity`, `discontinuity_locations`, `warnings` 属性
    - _Requirements: 4.1, 4.2_
  
  - [ ] 1.3 实现 `validate_unwrapping()` 方法
    - 检查解包裹相位与参考相位差异 < π
    - 检测相位连续性（梯度 < π）
    - 返回 `UnwrapValidationResult` 对象
    - _Requirements: 1.4, 1.5, 4.1, 4.2_

- [ ] 2. 属性基测试：解包裹核心功能
  - [ ] 2.1 编写 Property 4 测试：解包裹相位差异约束
    - **Property 4: 解包裹相位差异约束**
    - **Validates: Requirements 1.3, 1.4**
    - 生成随机折叠相位和参考相位
    - 验证解包裹后差异在 [-π, π] 范围内
  
  - [ ] 2.2 编写 Property 5 测试：解包裹保持平滑性
    - **Property 5: 解包裹保持平滑性**
    - **Validates: Requirements 1.5**
    - 生成平滑的真实相位
    - 验证折叠后解包裹能恢复平滑性

- [ ] 3. 检查点 - 确保核心解包裹功能测试通过
  - 运行所有测试，确保通过
  - 如有问题，询问用户

- [ ] 4. 扩展 PilotBeamCalculator 支持入射面相位计算
  - [ ] 4.1 在 `PilotBeamCalculator` 中添加入射面支持
    - 确保 `compute_reference_phase()` 支持 `position='entrance'`
    - 验证主光线处相位为 0
    - _Requirements: 3.2, 3.5_
  
  - [ ] 4.2 编写 Property 3 测试：主光线相位为零
    - **Property 3: 主光线相位为零**
    - **Validates: Requirements 3.5**
    - 生成随机光束参数
    - 验证原点处相位为 0

- [ ] 5. 修改 WavefrontToRaysSampler 集成解包裹
  - [ ] 5.1 添加 `phase_unwrapper` 参数
    - 修改 `__init__` 方法，添加可选的 `phase_unwrapper` 参数
    - 添加 `use_unwrapped_phase` 参数，默认 True
    - _Requirements: 1.1, 1.3_
  
  - [ ] 5.2 修改 `_extract_phase()` 方法
    - 如果提供了解包裹器，对折叠相位进行解包裹
    - 保持向后兼容（无解包裹器时行为不变）
    - _Requirements: 1.3_

- [ ] 6. 属性基测试：Pilot Beam 相位计算
  - [ ] 6.1 编写 Property 2 测试：Pilot Beam 相位公式一致性
    - **Property 2: Pilot Beam 相位公式一致性**
    - **Validates: Requirements 1.2, 3.1, 3.2, 3.3**
    - 生成随机光束参数和坐标
    - 验证计算结果符合公式 φ = k*r²/(2R)
  
  - [ ] 6.2 编写 Property 6 测试：Pilot Beam 相位连续性
    - **Property 6: Pilot Beam 相位连续性**
    - **Validates: Requirements 2.2, 3.4**
    - 生成随机光束参数和网格
    - 验证相位梯度 < π

- [ ] 7. 检查点 - 确保 Pilot Beam 相关测试通过
  - 运行所有测试，确保通过
  - 如有问题，询问用户

- [ ] 8. 集成到 HybridElementPropagator
  - [ ] 8.1 修改 `propagator.py` 创建 PhaseUnwrapper
    - 在 `_init_submodules()` 中创建 `PhaseUnwrapper` 实例
    - 传递给 `WavefrontToRaysSampler`
    - _Requirements: 1.1, 1.2, 1.3_
  
  - [ ] 8.2 添加解包裹验证到传播流程
    - 在 `_sample_wavefront()` 后验证解包裹结果
    - 发出警告（如果有）
    - _Requirements: 4.1, 4.2_

- [ ] 9. 出射面 OPD 连续性验证
  - [ ] 9.1 实现出射面 OPD 连续性检查
    - 在 `_correct_phase()` 中添加 OPD 梯度检查
    - 检测 2π 跳变
    - _Requirements: 2.3, 2.4_
  
  - [ ] 9.2 编写 Property 7 测试：OPD 连续性
    - **Property 7: OPD 连续性**
    - **Validates: Requirements 2.3, 2.4**
    - 生成解包裹输入
    - 验证 OPD 梯度 < 阈值

- [ ] 10. 属性基测试：连续性检测
  - [ ] 10.1 编写 Property 8 测试：连续性检测正确性
    - **Property 8: 连续性检测正确性**
    - **Validates: Requirements 4.1, 4.2**
    - 生成带标签的相位分布
    - 验证检测结果正确

- [ ] 11. 检查点 - 确保所有属性测试通过
  - 运行所有测试，确保通过
  - 如有问题，询问用户

- [ ] 12. 集成测试
  - [ ] 12.1 编写简单球面波前测试
    - 创建已知曲率的球面波前
    - 验证解包裹后相位与理论值一致
    - _Requirements: 1.4, 1.5_
  
  - [ ] 12.2 编写大曲率波前测试
    - 创建多次 2π 折叠的波前
    - 验证解包裹后相位连续
    - _Requirements: 1.5, 2.4_
  
  - [ ] 12.3 编写端到端流程测试
    - 使用 PROPER 生成复振幅
    - 执行完整流程：解包裹 → 采样 → 追迹 → OPD 提取
    - 验证 OPD 平滑连续
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [ ] 13. 最终检查点 - 确保所有测试通过
  - 运行完整测试套件
  - 验证代码覆盖率
  - 如有问题，询问用户

## 注意事项

- 所有任务都是必需的，包括属性基测试
- 每个任务引用具体的需求条款以确保可追溯性
- 检查点任务用于验证阶段性成果
- 属性基测试使用 hypothesis 框架，每个属性至少运行 100 次迭代
