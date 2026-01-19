# 实现计划：Pilot Beam 参考相位（简化版）

## 概述

本实现计划基于简化设计：使用 ABCD 矩阵法追踪光束参数，无需 BeamEstimator 模块。

## 已完成任务

- [x] 1. 扩展 ABCDCalculator 模块
  - [x] 1.1 添加 get_beam_at_element() 方法
    - 获取指定元件入射面或出射面的光束参数
    - 支持 'entrance' 和 'exit' 位置
  
  - [x] 1.2 添加 _propagate_to_element_entrance() 内部方法
    - 传播到元件入射面（不应用元件变换）
  
  - [x] 1.3 添加 _propagate_to_element_exit() 内部方法
    - 传播到元件出射面（应用元件变换）
  
  - [x] 1.4 添加 get_all_element_beam_params() 方法
    - 获取所有元件入射面和出射面的光束参数
  
  - [x] 1.5 添加 compute_reference_phase_at_position() 方法
    - 在指定坐标处计算参考相位
  
  - [x] 1.6 添加 compute_reference_phase_grid() 方法
    - 在网格上计算参考相位
  
  - [x] 1.7 支持 FlatMirror 元件
    - 平面镜不改变光束参数，只改变方向

- [x] 2. 重构 PilotBeamCalculator
  - [x] 2.1 修改构造函数
    - 接受 GaussianBeam 和元件列表
    - 内部创建 ABCDCalculator
  
  - [x] 2.2 修改 compute_reference_phase() 方法
    - 使用 ABCDCalculator 计算
    - 支持指定元件索引和位置
  
  - [x] 2.3 添加 compute_reference_phase_at_rays() 方法
    - 在光线位置计算参考相位
  
  - [x] 2.4 添加 get_beam_params_at_element() 方法
    - 获取指定元件处的光束参数
  
  - [x] 2.5 保留向后兼容接口
    - compute_reference_phase_analytical() 标记为弃用

## 待完成任务

- [ ] 3. 编写单元测试
  - [ ] 3.1 测试 ABCDCalculator.get_beam_at_element()
    - 测试单元件系统
    - 测试多元件系统
    - 测试入射面和出射面
  
  - [ ] 3.2 测试 ABCDCalculator.compute_reference_phase_at_position()
    - 测试平面波前（R=∞）
    - 测试球面波前
  
  - [ ] 3.3 测试 PilotBeamCalculator
    - 测试与 ABCDCalculator 的集成
    - 测试参考相位计算

- [ ] 4. 编写属性测试
  - [ ] 4.1 Property 18: ABCD 变换正确性
    - 验证能量守恒
    - 验证与解析解一致
  
  - [ ] 4.2 Property 19: 参考相位连续性
    - 验证相邻像素相位差 < π

- [ ] 5. 更新 HybridElementPropagator
  - [ ] 5.1 使用新的 PilotBeamCalculator 接口
  - [ ] 5.2 传递光束和元件信息

- [ ] 6. Checkpoint - 验证改进效果
  - 运行所有测试
  - 验证 PilotBeamWarning 不再频繁出现
  - 确保能量守恒

## 已删除任务（不再需要）

以下任务因简化设计而删除：

- ~~实现 BeamEstimator 模块~~
- ~~实现 _estimate_beam_radius 方法~~
- ~~实现 _estimate_curvature_radius 方法~~
- ~~实现 _solve_waist_params 方法~~
- ~~编写 BeamEstimator 单元测试~~
- ~~编写 BeamEstimator 属性测试~~

## 测试文件结构

```
tests/test_hybrid_propagation/
├── test_abcd_reference_phase.py     # ABCD 方法测试
├── test_pilot_beam_calculator.py    # PilotBeamCalculator 测试
└── ...（现有测试文件）
```

## 注意事项

- 复用现有的 GaussianBeam 和 ABCDCalculator 模块
- 保持与现有接口的向后兼容
- 所有代码使用中文注释
- 属性测试使用 hypothesis 库
