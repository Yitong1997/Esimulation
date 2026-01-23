# Implementation Plan: MATLAB Style API

## Overview

本实现计划将混合光学仿真系统的 API 重构为 MATLAB 风格的直观代码块结构。核心原则是复用现有模块，仅提供薄封装层。

## Tasks

- [x] 1. 创建 bts 模块基础结构
  - [x] 1.1 创建 `src/bts/__init__.py` 模块入口
    - 导出公共 API：`load_zmx`, `simulate`, `OpticalSystem`, `GaussianSource`
    - 重导出现有类：`SimulationResult`, `WavefrontData`, `SurfaceRecord`
    - _Requirements: 1.1, 1.2, 1.3, 1.4_
  
  - [x] 1.2 创建 `src/bts/exceptions.py` 异常定义
    - 定义 `ParseError` 异常类
    - 重导出现有异常：`ConfigurationError`, `SimulationError`
    - _Requirements: 2.2, 5.4_

- [x] 2. 实现 GaussianSource 类
  - [x] 2.1 创建 `src/bts/source.py`
    - 实现 `GaussianSource` 类，封装光源参数
    - 实现 `physical_size_mm` 默认值计算（8 倍束腰）
    - 实现 `z_rayleigh_mm` 属性计算
    - 实现 `print_info()` 方法
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_
  
  - [x] 2.2 编写 GaussianSource 属性测试
    - **Property 3: GaussianSource 参数存储正确性**
    - **Property 4: physical_size_mm 默认值计算**
    - **Validates: Requirements 3.1, 3.2, 3.6**

- [x] 3. 实现 OpticalSystem 类
  - [x] 3.1 创建 `src/bts/optical_system.py`
    - 实现 `OpticalSystem` 类，封装表面定义列表
    - 实现 `add_surface()` 方法（通用表面）
    - 实现 `add_flat_mirror()` 方法（复用 HybridSimulator 逻辑）
    - 实现 `add_spherical_mirror()` 方法（复用 HybridSimulator 逻辑）
    - 实现 `add_paraxial_lens()` 方法（复用 HybridSimulator 逻辑）
    - _Requirements: 2.3, 2.4, 2.5, 2.6, 2.7_
  
  - [x] 3.2 实现 OpticalSystem 信息展示方法
    - 实现 `print_info()` 方法，打印系统参数摘要
    - 实现 `plot_layout()` 方法，复用现有 ZmxOpticLoader 和 view_2d
    - 实现 `__len__()` 和 `num_surfaces` 属性
    - _Requirements: 4.1, 4.2, 4.3, 4.4_
  
  - [x] 3.3 编写 OpticalSystem 属性测试
    - **Property 5: print_info 输出包含必要信息**
    - **Validates: Requirements 4.3**

- [x] 4. 实现 I/O 函数
  - [x] 4.1 创建 `src/bts/io.py`
    - 实现 `load_zmx()` 函数，调用现有 `load_optical_system_from_zmx()`
    - 实现文件路径解析和错误处理
    - _Requirements: 1.1, 2.1, 2.2_
  
  - [x] 4.2 编写 load_zmx 属性测试
    - **Property 1: load_zmx 返回正确类型**
    - **Property 2: 不存在的文件抛出 FileNotFoundError**
    - **Validates: Requirements 1.1, 2.1, 2.2**

- [x] 5. 实现 simulate 函数
  - [x] 5.1 创建 `src/bts/simulation.py`
    - 实现 `simulate()` 函数，内部调用 `HybridSimulator.run()`
    - 实现参数验证和错误处理
    - _Requirements: 1.4, 5.1, 5.2, 5.3, 5.4, 5.5_
  
  - [x] 5.2 编写 simulate 属性测试
    - **Property 6: simulate 返回完整结果**
    - **Property 7: 仿真失败时抛出异常**
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4**

- [x] 6. Checkpoint - 核心功能验证
  - 确保所有测试通过，验证核心 API 功能正常
  - 如有问题请询问用户

- [x] 7. 增强 SimulationResult 类
  - [x] 7.1 在 `src/hybrid_simulation/result.py` 中添加便捷方法
    - 添加 `get_final_wavefront()` 方法
    - 添加 `get_entrance_wavefront(index)` 方法
    - 添加 `get_exit_wavefront(index)` 方法
    - _Requirements: 6.5_
  
  - [x] 7.2 编写结果保存/加载属性测试
    - **Property 8: 结果保存/加载往返一致性**
    - **Validates: Requirements 7.2, 7.3, 7.4**

- [x] 8. 创建示例程序
  - [x] 8.1 创建 `examples/bts_simple_example.py`
    - 基于示例 2（简单折叠镜测试）的 MATLAB 风格代码
    - 验证 API 可用性
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_
  
  - [x] 8.2 创建 `examples/bts_zmx_example.py`
    - 基于示例 3（ZMX 文件仿真）的 MATLAB 风格代码
    - 验证 ZMX 加载和仿真功能
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

- [x] 9. 创建 API 文档
  - [x] 9.1 创建 `docs/bts_api.md`
    - 编写 API 概述和快速入门
    - 编写所有公共函数和类的详细说明
    - 包含参数类型、默认值和返回值
    - 包含使用示例代码
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 10. Final Checkpoint - 完整功能验证
  - 运行所有示例程序，确保功能正常
  - 确保所有测试通过
  - 如有问题请询问用户

## Notes

- 每个任务引用具体的需求条款以确保可追溯性
- 检查点任务用于增量验证
- 属性测试验证设计文档中定义的正确性属性
- 所有测试任务均为必需，确保全面测试覆盖
