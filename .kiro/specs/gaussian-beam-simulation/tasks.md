# 实现计划：高斯光束传输仿真模型

## 概述

本实现计划基于已批准的需求和设计文档，将高斯光束传输仿真模型分解为可执行的编码任务。实现语言为 Python，使用 PROPER 和 optiland 库。

## 任务列表

- [x] 1. 完善 GaussianBeam 类
  - [x] 1.1 增强参数验证逻辑
    - 添加波长参数验证（必须为正值）
    - 添加更详细的错误信息
    - _Requirements: 1.1, 1.3, 1.5, 9.1, 9.2, 9.3_
  
  - [x] 1.2 编写高斯光束参数计算属性测试
    - **Property 1: 高斯光束参数计算正确性**
    - **Validates: Requirements 1.8, 1.9, 1.10**
  
  - [x] 1.3 完善波前生成方法
    - 确保振幅分布符合高斯函数
    - 确保相位分布包含正确的球面波前相位
    - 支持附加波前误差
    - _Requirements: 1.11, 3.2, 3.3_
  
  - [x] 1.4 编写波前生成属性测试
    - **Property 3: 波前生成正确性**
    - **Validates: Requirements 1.11, 3.2, 3.3**

- [x] 2. 完善光学元件类
  - [x] 2.1 增强 OpticalElement 基类
    - 完善 get_entrance_position 方法
    - 完善 get_chief_ray_direction 方法
    - 添加参数验证
    - _Requirements: 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.11_
  
  - [x] 2.2 编写光学元件方向计算属性测试
    - **Property 5: 光学元件方向计算正确性**
    - **Validates: Requirements 2.11**
  
  - [x] 2.3 完善 ParabolicMirror 类
    - 确保 get_surface_definition 返回正确的表面定义
    - 添加离轴抛物面支持（如需要）
    - _Requirements: 2.1, 2.12_
  
  - [x] 2.4 完善 SphericalMirror 和 ThinLens 类
    - 确保 get_surface_definition 返回正确的表面定义
    - _Requirements: 2.2, 2.3, 2.12_

- [x] 3. 检查点 - 确保基础类测试通过
  - 运行所有单元测试和属性测试
  - 如有问题，询问用户

- [x] 4. 完善 ABCDCalculator 类
  - [x] 4.1 完善 ABCD 矩阵计算方法
    - 验证自由空间传播矩阵
    - 验证薄透镜/反射镜矩阵
    - 验证复光束参数变换
    - _Requirements: 7.2, 7.3, 7.4_
  
  - [x] 4.2 编写 ABCD 矩阵计算属性测试
    - **Property 4: ABCD 矩阵计算正确性**
    - **Validates: Requirements 7.2, 7.3, 7.4**
  
  - [x] 4.3 完善 propagate_to 方法
    - 正确处理多个元件
    - 正确处理反射镜方向反转
    - _Requirements: 7.5, 7.6_
  
  - [x] 4.4 完善 get_output_waist 和 trace_beam_profile 方法
    - _Requirements: 7.7, 7.8_

- [x] 5. 完善 HybridGaussianBeamSimulator 类
  - [x] 5.1 完善波前初始化
    - 使用 PROPER 正确初始化高斯光束波前
    - 应用高斯振幅分布
    - 应用初始相位
    - _Requirements: 3.1, 3.2, 3.3_
  
  - [x] 5.2 完善物理光学传播
    - 使用 PROPER prop_propagate 进行传播
    - 正确更新当前位置
    - _Requirements: 3.4, 3.5, 3.6_
  
  - [x] 5.3 完善混合光线追迹
    - 从 PROPER 波前提取复振幅
    - 使用 WavefrontToRaysSampler 采样（方形均匀分布）
    - 使用 ElementRaytracer 进行光线追迹
    - _Requirements: 4.1, 4.2, 4.3, 4.4_
  
  - [x] 5.4 编写方形均匀采样属性测试
    - **Property 11: 方形均匀采样覆盖**
    - **Validates: Requirements 4.3**
  
  - [x] 5.5 完善波前重建
    - 从出射光线计算 OPD
    - 使用插值重建相位分布
    - 处理无效光线
    - _Requirements: 4.5, 4.6, 4.7, 5.1, 5.3_
  
  - [x] 5.6 编写相位重建属性测试
    - **Property 10: 相位重建网格一致性**
    - **Validates: Requirements 5.4**
  
  - [x] 5.7 完善反射镜处理
    - 正确处理反射后的传播方向
    - 将重建的相位应用到 PROPER 波前
    - _Requirements: 4.8, 4.9_

- [x] 6. 检查点 - 确保仿真器核心功能测试通过
  - 运行所有单元测试和属性测试
  - 如有问题，询问用户

- [x] 7. 完善仿真流程控制
  - [x] 7.1 完善 propagate_to 方法
    - 按元件位置顺序处理
    - 记录传播历史
    - _Requirements: 6.1, 6.2, 6.3_
  
  - [x] 7.2 编写元件顺序处理属性测试
    - **Property 8: 元件顺序处理**
    - **Validates: Requirements 6.2**
  
  - [x] 7.3 完善 reset 和 get_psf 方法
    - _Requirements: 6.4, 6.5_
  
  - [x] 7.4 完善 SimulationResult 返回
    - 确保包含所有必需字段
    - 计算光束半径估计
    - 计算波前统计信息
    - _Requirements: 6.6, 6.7, 6.8_
  
  - [x] 7.5 编写仿真结果完整性属性测试
    - **Property 9: 仿真结果完整性**
    - **Validates: Requirements 6.6, 6.7, 6.8**

- [x] 8. 实现验证测试
  - [x] 8.1 实现单抛物面反射镜验证测试
    - 设置测试光学系统
    - 使用 ABCD 计算理论结果
    - 使用混合仿真计算实际结果
    - 比较束腰位置和半径
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_
  
  - [x] 8.2 编写仿真精度验证属性测试
    - **Property 6: 仿真精度验证**
    - **Validates: Requirements 8.4, 8.5**
  
  - [x] 8.3 实现平面波入射验证测试
    - 验证输出束腰位于反射镜焦点
    - _Requirements: 8.6_
  
  - [x] 8.4 编写束腰处波前曲率半径属性测试
    - **Property 7: 束腰处波前曲率半径**
    - **Validates: Requirements 8.7**
  
  - [x] 8.5 生成验证报告
    - 输出理论值、仿真值和误差
    - _Requirements: 8.8_

- [x] 9. 完善错误处理
  - [x] 9.1 添加参数验证错误处理
    - 波长非正值
    - 束腰半径非正值
    - M² 因子小于 1.0
    - 元件半口径非正值
    - _Requirements: 9.1, 9.2, 9.3, 9.4_
  
  - [x] 9.2 编写参数验证属性测试
    - **Property 2: 参数验证正确性**
    - **Validates: Requirements 1.1, 1.3, 2.5, 9.1, 9.2, 9.3, 9.4**
  
  - [x] 9.3 添加运行时错误处理
    - 光线追迹失败警告
    - PROPER 初始化失败
    - _Requirements: 9.5, 9.6_

- [x] 10. 最终检查点 - 确保所有测试通过
  - 运行完整测试套件
  - 验证测试覆盖率
  - 如有问题，询问用户

## 注意事项

- 所有任务均为必需任务，包括属性测试
- 每个任务引用了具体的需求条款以确保可追溯性
- 属性测试使用 hypothesis 库，每个测试至少运行 100 次迭代
- 检查点任务用于确保增量验证
