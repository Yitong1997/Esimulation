# Implementation Plan: 混合光学传播系统

## 概述

本实现计划将混合光学传播系统分解为可执行的编码任务。系统将 PROPER 物理光学传输与 optiland 几何光线追迹相结合，实现完整的混合光学传播仿真。

## 任务列表

- [x] 1. 核心数据模型实现
  - [x] 1.1 实现 PilotBeamParams 数据类
    - 实现 `from_gaussian_source()` 和 `from_q_parameter()` 类方法
    - 实现 `propagate()`、`apply_lens()`、`apply_mirror()` 方法
    - 实现 ABCD 矩阵变换逻辑
    - _Requirements: 8.1, 8.2, 8.3_
  
  - [x] 1.2 编写 PilotBeamParams 属性测试
    - **Property 7: Pilot Beam ABCD 追踪正确性**
    - **Property 12: Pilot Beam 相位在主光线处为零**
    - **Validates: Requirements 8.1, 8.3, 8.5, 8.6**
  
  - [x] 1.3 实现 GridSampling 数据类
    - 实现 `from_proper()` 类方法
    - 实现 `is_compatible()` 方法
    - _Requirements: 17.1, 17.3_
  
  - [x] 1.4 实现 PropagationState 数据类
    - 定义所有必需属性
    - 实现 `validate_consistency()` 方法
    - _Requirements: 10.1, 10.2_
  
  - [x] 1.5 实现 SourceDefinition 数据类
    - 实现 `create_initial_wavefront()` 方法
    - 支持初始复振幅像差叠加
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

- [x] 2. 检查点 - 确保数据模型测试通过
  - 运行所有数据模型相关测试，如有问题请询问用户

- [x] 3. StateConverter 实现
  - [x] 3.1 实现 StateConverter 类基础结构
    - 实现 `_compute_pilot_phase()` 方法
    - _Requirements: 8.6, 8.7_
  
  - [x] 3.2 实现 `proper_to_simulation()` 方法
    - 从 PROPER 提取折叠相位
    - 使用 Pilot Beam 参考相位解包裹
    - 组合振幅和非折叠相位
    - _Requirements: 5.1, 5.2, 5.3, 9.3_
  
  - [x] 3.3 实现 `simulation_to_proper()` 方法
    - 使用 Pilot Beam 参数初始化 PROPER 对象
    - 计算残差相位并写入 PROPER
    - _Requirements: 9.4, 9.5_
  
  - [x] 3.4 编写 StateConverter 属性测试
    - **Property 2: 相位解包裹正确性**
    - **Property 8: 数据表示往返一致性**
    - **Validates: Requirements 5.1, 5.2, 5.4, 5.5, 9.5, 10.2**

- [x] 4. FreeSpacePropagator 实现
  - [x] 4.1 实现 FreeSpacePropagator 类
    - 实现 `_compute_propagation_distance()` 方法
    - 实现 `propagate()` 方法
    - 支持正向和逆向传播
    - _Requirements: 4.1, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8_
  
  - [x] 4.2 编写 FreeSpacePropagator 属性测试
    - **Property 3: 传播距离计算正确性**
    - **Property 4: 自由空间传播往返一致性**
    - **Validates: Requirements 4.3, 4.4, 4.5, 4.6, 4.7**

- [x] 5. 检查点 - 确保传播器基础测试通过
  - 运行所有传播器相关测试，如有问题请询问用户

- [x] 6. HybridElementPropagator 实现
  - [x] 6.1 实现材质变化检测逻辑
    - 实现 `detect_material_change()` 函数
    - 支持反射镜、透镜前后表面检测
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_
  
  - [x] 6.2 编写材质变化检测属性测试
    - **Property 5: 材质变化检测正确性**
    - **Validates: Requirements 11.1, 11.2, 11.3, 11.4, 11.5**
  
  - [x] 6.3 实现 HybridElementPropagator 类基础结构
    - 实现构造函数和方法选择逻辑
    - _Requirements: 6.1, 7.7_
  
  - [x] 6.4 实现 `_propagate_local_raytracing()` 方法
    - 从入射面采样光线（使用非折叠相位）
    - 调用 ElementRaytracer 进行光线追迹
    - 计算 OPD 和雅可比矩阵振幅
    - 使用 RayToWavefrontReconstructor 重建复振幅
    - _Requirements: 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9_
  
  - [x] 6.5 编写混合元件传播属性测试
    - **Property 9: 能量守恒（雅可比矩阵振幅）**
    - **Property 10: 波前无整体倾斜**
    - **Validates: Requirements 3.5, 6.6, 7.6, 13.3**
  
  - [ ]* 6.6 实现 `_propagate_pure_diffraction()` 方法（可选）
    - 使用 tilted_asm 从入射面传播到切平面
    - 计算表面矢高并应用相位延迟
    - 使用 tilted_asm 从切平面传播到出射面
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

- [x] 7. ParaxialPhasePropagator 实现
  - [x] 7.1 实现 ParaxialPhasePropagator 类
    - 实现 `propagate()` 方法
    - 实现 `_update_pilot_beam()` 方法
    - 应用薄透镜相位修正
    - _Requirements: 19.1, 19.2, 19.3, 19.4, 19.5, 19.6, 19.7_
  
  - [x] 7.2 编写 PARAXIAL 表面属性测试
    - **Property 11: PARAXIAL 表面相位修正正确性**
    - **Validates: Requirements 19.4, 19.5**

- [x] 8. 检查点 - 确保元件传播器测试通过
  - 运行所有元件传播器相关测试，如有问题请询问用户


- [x] 9. HybridOpticalPropagator 主类实现
  - [x] 9.1 实现 HybridOpticalPropagator 类基础结构
    - 实现构造函数
    - 初始化内部状态和子组件
    - _Requirements: 16.1_
  
  - [x] 9.2 实现光轴追踪集成
    - 集成 zemax-optical-axis-tracing 的 OpticalAxisTracker
    - 实现 `get_optical_axis_at_surface()` 方法
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7_
  
  - [x] 9.3 实现入射面/出射面定义逻辑
    - 确保入射面垂直于入射光轴
    - 确保出射面垂直于出射光轴
    - 原点为主光线与表面交点
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.6_
  
  - [x] 9.4 编写入射面/出射面属性测试
    - **Property 1: 入射面/出射面垂直于光轴**
    - **Validates: Requirements 3.1, 3.3, 13.2**
  
  - [x] 9.5 实现 `propagate()` 方法
    - 初始化入射波面
    - 循环处理每个表面
    - 根据材质变化选择传播方法
    - 更新状态并记录中间结果
    - _Requirements: 16.2_
  
  - [x] 9.6 实现 `propagate_to_surface()` 方法
    - 传播到指定表面
    - 返回该表面的传播状态
    - _Requirements: 16.3_
  
  - [x] 9.7 实现 `get_wavefront_at_surface()` 和 `get_state_at_surface()` 方法
    - 获取指定表面的波前和状态
    - _Requirements: 16.4, 10.3_
  
  - [x] 9.8 实现 `get_grid_sampling()` 方法
    - 获取指定表面的网格采样信息
    - _Requirements: 17.5, 17.6, 17.7_
  
  - [x] 9.9 编写网格采样属性测试
    - **Property 13: 网格采样信息一致性**
    - **Validates: Requirements 17.1, 17.3, 17.4**

- [x] 10. 单位转换和错误处理
  - [x] 10.1 实现单位转换工具函数
    - mm ↔ m 转换
    - OPD（波长数）↔ 相位（弧度）转换
    - 波长单位转换
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7_
  
  - [x] 10.2 编写单位转换属性测试
    - **Property 6: 单位转换正确性**
    - **Validates: Requirements 12.3, 12.4, 12.5, 12.6, 12.7**
  
  - [x] 10.3 实现异常类和错误处理
    - 定义 HybridPropagationError 及其子类
    - 实现错误检测和异常抛出逻辑
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6_

- [x] 11. 检查点 - 确保主类测试通过
  - 运行所有主类相关测试，如有问题请询问用户

- [x] 12. ZMX 文件集成
  - [x] 12.1 实现从 ZMX 文件加载光学系统
    - 集成 zmx-file-reader 的 ZmxParser
    - 支持 PARAXIAL 表面类型识别
    - _Requirements: 16.5, 19.1, 19.2, 19.3_
  
  - [x] 12.2 实现与 SequentialOpticalSystem 的兼容
    - 确保接口兼容性
    - _Requirements: 16.6_

- [x] 13. 集成测试
  - [x] 13.1 实现端到端传播测试
    - 从 ZMX 文件加载光学系统
    - 执行完整传播
    - 验证输出波前质量
    - _Requirements: 18.1, 18.4_
  
  - [x] 13.2 实现伽利略 OAP 扩束镜测试
    - 使用离轴抛物面镜配置
    - 验证光束扩展比（误差 < 5%）
    - 验证波前质量（Strehl > 0.9）
    - 验证能量守恒（误差 < 5%）
    - _Requirements: 18.2, 18.3, 18.4_
  
  - [x] 13.3 更新现有伽利略 OAP 扩束镜示例
    - 更新 examples/galilean_oap_expander.py
    - 使用新的 HybridOpticalPropagator API
    - _Requirements: 18.5_
  
  - [x] 13.4 实现与纯 PROPER 模式的对比验证
    - 对于理想元件，两种模式结果应一致
    - _Requirements: 18.6_

- [x] 14. 最终检查点 - 确保所有测试通过
  - 运行完整测试套件，如有问题请询问用户

## 注意事项

- 每个任务引用具体的需求条款以确保可追溯性
- 检查点用于确保增量验证
- 属性测试验证普遍正确性属性（使用 hypothesis 库，每个测试至少 100 次迭代）
- 单元测试验证具体示例和边界情况
- 所有测试任务均为必需，确保从一开始就全面测试
