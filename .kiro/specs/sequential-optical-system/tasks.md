# Implementation Plan: Sequential Optical System

## Overview

本实现计划将序列模式混合光学仿真系统分解为可执行的编码任务。采用增量开发方式，每个任务构建在前一个任务的基础上，确保代码始终可运行和测试。

**代码复用说明：**
- `GaussianBeam` 类：直接复用 `gaussian_beam_simulation.gaussian_beam.GaussianBeam`
- `OpticalElement` 基类及子类（`ParabolicMirror`, `SphericalMirror`, `ThinLens`）：直接复用 `gaussian_beam_simulation.optical_elements`
- `ABCDCalculator` 类：直接复用 `gaussian_beam_simulation.abcd_calculator.ABCDCalculator`
- `HybridGaussianBeamSimulator` 类：直接复用 `gaussian_beam_simulation.hybrid_simulator.HybridGaussianBeamSimulator`

## Tasks

- [x] 1. 创建模块结构和异常类
  - 创建 `src/sequential_system/` 目录结构
  - 创建 `__init__.py` 导出公共 API
  - 创建 `exceptions.py` 定义异常类层次
  - _Requirements: 5.7, 10.3_

- [x] 2. 实现 GaussianBeamSource 包装类
  - [x] 2.1 创建 `source.py` 文件，实现 GaussianBeamSource 数据类
    - 定义 wavelength, w0, z0, m2 参数
    - 实现 to_gaussian_beam() 方法，返回现有 GaussianBeam 对象
    - 复用 GaussianBeam 的参数验证逻辑
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_
  
  - [x] 2.2 编写 GaussianBeamSource 属性测试
    - **Property 1: 高斯光束参数计算正确性**
    - **Property 2: 无效输入参数拒绝**
    - **Validates: Requirements 1.2, 1.4, 1.7, 1.8, 1.9, 1.10**

- [x] 3. 扩展现有光学元件类（添加 FlatMirror）
  - [x] 3.1 在 `gaussian_beam_simulation/optical_elements.py` 中添加 FlatMirror 类
    - 继承 OpticalElement 基类
    - 曲率半径为无穷大
    - 焦距返回 np.inf
    - is_reflective 返回 True
    - _Requirements: 2.3, 2.1.2_
  
  - [x] 3.2 编写 FlatMirror 单元测试
    - 测试创建和属性
    - **Validates: Requirements 2.3**

- [x] 4. Checkpoint - 确保基础类测试通过
  - 确保所有测试通过，如有问题请询问用户。

- [x] 5. 实现 SamplingPlane 和结果类
  - [x] 5.1 创建 `sampling.py` 文件，实现 SamplingPlane 数据类
    - 定义 distance, name 参数
    - 定义 result 属性（仿真后填充）
    - _Requirements: 4.1, 4.2_
  
  - [x] 5.2 实现 SamplingResult 数据类
    - 定义 wavefront, amplitude, phase, sampling, beam_radius 等属性
    - 实现 compute_m2() 方法
    - 实现 wavefront_rms, wavefront_pv 计算
    - _Requirements: 4.4, 4.5, 4.6, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_
  
  - [x] 5.3 实现 SimulationResults 容器类
    - 实现 __getitem__ 支持名称和索引访问
    - 实现 __iter__ 支持迭代
    - _Requirements: 5.6, 8.1_
  
  - [x] 5.4 编写采样结果属性测试
    - **Property 5: 采样面数据完整性**
    - **Property 7: 仿真结果完整性**
    - **Validates: Requirements 4.3, 4.4, 4.5, 4.6, 5.6**

- [x] 6. 实现 SequentialOpticalSystem 核心类
  - [x] 6.1 创建 `system.py` 文件，实现 SequentialOpticalSystem 类框架
    - 定义 __init__ 方法，接受 source, grid_size, beam_ratio 参数
    - 初始化内部状态（elements 列表, sampling_planes 列表）
    - 复用现有 OpticalElement 类作为内部表示
    - _Requirements: 3.1, 3.7, 3.8_
  
  - [x] 6.2 实现 add_surface() 方法
    - 接受现有 OpticalElement 子类实例
    - 支持链式调用（返回 self）
    - 自动计算 z_position 和 path_length
    - 处理反射面方向反转
    - _Requirements: 3.2, 3.3, 3.4, 3.5, 2.13_
  
  - [x] 6.3 实现 add_sampling_plane() 方法
    - 支持链式调用
    - 验证距离在有效范围内
    - _Requirements: 3.6, 4.1, 4.2_
  
  - [x] 6.4 实现 get_abcd_result() 方法
    - 复用现有 ABCDCalculator
    - 返回指定距离的 ABCD 计算结果
    - _Requirements: 6.1, 6.2, 6.3, 6.4_
  
  - [x] 6.5 实现 summary() 方法
    - 返回系统配置的文本摘要
    - _Requirements: 10.6_
  
  - [x] 6.6 编写系统构建属性测试
    - **Property 3: 光学面位置自动计算**
    - **Property 4: 反射面方向反转**
    - **Validates: Requirements 3.3, 3.4, 2.13, 5.5**

- [x] 7. Checkpoint - 确保系统构建测试通过
  - 确保所有测试通过，如有问题请询问用户。

- [x] 8. 实现仿真执行引擎
  - [x] 8.1 实现 run() 方法核心逻辑
    - 复用现有 HybridGaussianBeamSimulator 进行仿真
    - 在采样面位置记录波前
    - 返回 SimulationResults
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_
  
  - [x] 8.2 实现采样面波前记录
    - 在指定光程距离处提取波前数据
    - 计算光束半径和波前质量指标
    - _Requirements: 4.3, 4.4, 4.5, 4.6_
  
  - [x] 8.3 实现错误处理
    - 捕获仿真过程中的异常
    - 抛出描述性的 SimulationError
    - _Requirements: 5.7_
  
  - [x] 8.4 编写仿真执行属性测试
    - **Property 6: ABCD 与物理仿真一致性**
    - **Validates: Requirements 6.5**

- [x] 9. 实现可视化功能
  - [x] 9.1 创建 `visualization.py` 文件，实现 LayoutVisualizer 类
    - 复用现有 ABCDCalculator 计算光束包络
    - 绘制光学面位置和形状
    - 标记采样面位置
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_
  
  - [x] 9.2 实现 draw_layout() 方法
    - 支持 show 参数控制是否调用 plt.show()
    - 返回 (fig, ax) 元组
    - _Requirements: 7.6, 7.7_
  
  - [x] 9.3 编写可视化单元测试
    - 测试 show=False 不阻塞
    - 测试返回正确的 figure 和 axes
    - _Requirements: 7.6, 7.7_

- [x] 10. Checkpoint - 确保核心功能测试通过
  - 确保所有测试通过，如有问题请询问用户。

- [x] 11. 完善模块导出和文档
  - [x] 11.1 更新 `__init__.py` 导出所有公共类
    - 导出 SequentialOpticalSystem, GaussianBeamSource
    - 从 gaussian_beam_simulation 重新导出 ParabolicMirror, SphericalMirror, ThinLens, FlatMirror
    - 导出 SamplingPlane, SamplingResult, SimulationResults
    - _Requirements: 10.1, 10.2_
  
  - [x] 11.2 添加使用示例到 docstrings
    - 在 SequentialOpticalSystem 类添加完整示例
    - 确保示例代码可运行
    - _Requirements: 10.4, 10.5_

- [x] 12. 集成测试
  - [x] 12.1 编写简单反射镜系统集成测试
    - 单个凹面镜聚焦测试
    - 验证焦点位置和光束尺寸
    - _Requirements: 6.5_
  
  - [x] 12.2 编写离轴抛物面系统集成测试
    - OAP 镜准直测试
    - 验证光束方向变化
    - _Requirements: 2.1.4, 2.1.8_
  
  - [x] 12.3 编写多元件系统集成测试
    - 扩束器配置测试
    - 验证放大倍率
    - _Requirements: 3.2, 5.6_

- [x] 13. Final Checkpoint - 确保所有测试通过
  - 确保所有测试通过，如有问题请询问用户。

## Notes

- 所有任务均为必需任务，确保从一开始就进行全面测试
- 每个任务引用具体的需求条款以确保可追溯性
- Checkpoint 任务用于增量验证，确保代码质量
- 属性测试验证普遍正确性属性，每个测试运行至少 100 次迭代
- 单元测试验证具体示例和边界情况

## 代码复用总结

| 现有模块 | 复用方式 |
|---------|---------|
| `GaussianBeam` | 直接使用，GaussianBeamSource.to_gaussian_beam() 返回 |
| `OpticalElement` 基类 | 直接使用，作为光学面的内部表示 |
| `ParabolicMirror` | 直接使用，从 gaussian_beam_simulation 导入 |
| `SphericalMirror` | 直接使用，从 gaussian_beam_simulation 导入 |
| `ThinLens` | 直接使用，从 gaussian_beam_simulation 导入 |
| `ABCDCalculator` | 直接使用，用于 ABCD 计算和可视化 |
| `HybridGaussianBeamSimulator` | 直接使用，作为仿真引擎 |
| `WavefrontToRaysSampler` | 间接使用（通过 HybridGaussianBeamSimulator） |
| `ElementRaytracer` | 间接使用（通过 HybridGaussianBeamSimulator） |
