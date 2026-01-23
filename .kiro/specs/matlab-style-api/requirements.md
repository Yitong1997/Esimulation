# Requirements Document

## Introduction

本文档定义了混合光学仿真系统 API 重构的需求，目标是将当前的 Python 类封装风格改为 MATLAB 风格的直观代码块结构。重构后的 API 将提供简洁、直观的主程序编写体验，同时保持功能的完整性和强大性。

## Glossary

- **BTS**: Beam Tracing System，混合光学仿真系统的内部代号
- **MATLAB_Style_API**: 采用 MATLAB 编程风格的 API，特点是代码块分明、函数调用直观
- **Optical_System**: 光学系统对象，包含所有光学元件的定义
- **Gaussian_Source**: 高斯光源对象，定义入射光束参数
- **Simulation_Result**: 仿真结果对象，包含所有表面的波前数据
- **ZMX_File**: Zemax 序列模式光学系统定义文件
- **Code_Block**: 代码块，主程序中功能分明的代码段落

## Requirements

### Requirement 1: 模块化 API 设计

**User Story:** As a 光学工程师, I want 使用简洁的函数调用来完成仿真, so that 我可以像使用 MATLAB 一样直观地编写仿真脚本。

#### Acceptance Criteria

1. THE MATLAB_Style_API SHALL 提供 `bts.load_zmx(path)` 函数用于从 ZMX 文件加载光学系统
2. THE MATLAB_Style_API SHALL 提供 `bts.OpticalSystem()` 类用于逐行定义光学元件
3. THE MATLAB_Style_API SHALL 提供 `bts.GaussianSource(...)` 类用于定义高斯光源参数
4. THE MATLAB_Style_API SHALL 提供 `bts.simulate(system, source)` 函数用于执行仿真
5. WHEN 用户调用 API 函数时 THEN THE MATLAB_Style_API SHALL 返回类型明确的对象

### Requirement 2: 光学系统定义

**User Story:** As a 光学工程师, I want 通过两种方式定义光学系统, so that 我可以灵活选择从文件导入或手动构建。

#### Acceptance Criteria

1. WHEN 用户调用 `bts.load_zmx(path)` THEN THE MATLAB_Style_API SHALL 解析 ZMX 文件并返回 Optical_System 对象
2. WHEN ZMX 文件不存在 THEN THE MATLAB_Style_API SHALL 抛出 FileNotFoundError 并提供清晰的错误信息
3. WHEN 用户创建 `bts.OpticalSystem()` THEN THE MATLAB_Style_API SHALL 返回空的光学系统对象
4. THE Optical_System SHALL 提供 `add_surface(...)` 方法用于添加光学元件
5. THE Optical_System SHALL 提供 `add_flat_mirror(...)` 方法用于添加平面反射镜
6. THE Optical_System SHALL 提供 `add_spherical_mirror(...)` 方法用于添加球面反射镜
7. THE Optical_System SHALL 提供 `add_paraxial_lens(...)` 方法用于添加薄透镜

### Requirement 3: 光源定义

**User Story:** As a 光学工程师, I want 通过简洁的参数定义高斯光源, so that 我可以快速配置入射光束。

#### Acceptance Criteria

1. THE Gaussian_Source SHALL 接受 `wavelength_um` 参数定义波长（单位：μm）
2. THE Gaussian_Source SHALL 接受 `w0_mm` 参数定义束腰半径（单位：mm）
3. THE Gaussian_Source SHALL 接受 `grid_size` 参数定义网格大小（默认 256）
4. THE Gaussian_Source SHALL 接受可选的 `physical_size_mm` 参数定义物理尺寸
5. THE Gaussian_Source SHALL 接受可选的 `beam_diam_fraction` 参数控制 PROPER 采样
6. WHEN `physical_size_mm` 未指定 THEN THE Gaussian_Source SHALL 自动计算为 8 倍束腰半径

### Requirement 4: 仿真前信息展示

**User Story:** As a 光学工程师, I want 在仿真前查看系统参数和光路图, so that 我可以验证系统配置是否正确。

#### Acceptance Criteria

1. THE Optical_System SHALL 提供 `print_info()` 方法打印系统参数摘要
2. THE Optical_System SHALL 提供 `plot_layout(projection='YZ')` 方法绘制光路图
3. WHEN 调用 `print_info()` THEN THE MATLAB_Style_API SHALL 显示所有表面的类型、位置和参数
4. WHEN 调用 `plot_layout()` THEN THE MATLAB_Style_API SHALL 生成 2D 光路投影图

### Requirement 5: 仿真执行

**User Story:** As a 光学工程师, I want 通过单一函数调用执行仿真, so that 我可以简洁地获取仿真结果。

#### Acceptance Criteria

1. WHEN 调用 `bts.simulate(system, source)` THEN THE MATLAB_Style_API SHALL 执行完整的混合光学仿真
2. THE `bts.simulate()` 函数 SHALL 返回 Simulation_Result 对象
3. WHEN 仿真成功 THEN THE Simulation_Result SHALL 包含所有表面的波前数据
4. WHEN 仿真失败 THEN THE MATLAB_Style_API SHALL 抛出异常并提供详细错误信息
5. THE `bts.simulate()` 函数 SHALL 接受可选的 `verbose` 参数控制输出详细程度

### Requirement 6: 结果展示与访问

**User Story:** As a 光学工程师, I want 方便地查看和分析仿真结果, so that 我可以评估光学系统性能。

#### Acceptance Criteria

1. THE Simulation_Result SHALL 提供 `summary()` 方法打印结果摘要
2. THE Simulation_Result SHALL 提供 `plot_all()` 方法绘制所有表面的概览图
3. THE Simulation_Result SHALL 提供 `plot_surface(index)` 方法绘制指定表面的详细图
4. THE Simulation_Result SHALL 提供 `get_surface(index)` 方法获取指定表面的数据
5. THE Simulation_Result SHALL 提供 `get_final_wavefront()` 方法获取最终波前数据
6. WHEN 调用 `summary()` THEN THE MATLAB_Style_API SHALL 显示仿真状态、波长、网格大小和各表面残差

### Requirement 7: 结果保存

**User Story:** As a 光学工程师, I want 保存仿真结果到文件, so that 我可以后续分析或与他人分享。

#### Acceptance Criteria

1. THE Simulation_Result SHALL 提供 `save(path)` 方法保存完整结果
2. WHEN 调用 `save(path)` THEN THE MATLAB_Style_API SHALL 创建目录并保存所有数据和图表
3. THE MATLAB_Style_API SHALL 支持从保存的目录加载结果
4. THE 保存的数据 SHALL 包含配置参数、波前数据和元数据

### Requirement 8: API 文档

**User Story:** As a 开发者, I want 完整的 API 文档, so that 我可以了解所有可用功能和参数。

#### Acceptance Criteria

1. THE MATLAB_Style_API SHALL 提供 Markdown 格式的 API 文档
2. THE API 文档 SHALL 包含所有公共函数和类的说明
3. THE API 文档 SHALL 包含参数类型、默认值和返回值说明
4. THE API 文档 SHALL 包含使用示例代码
5. THE API 文档 SHALL 包含常见用例的完整示例

### Requirement 9: 主程序代码块结构

**User Story:** As a 光学工程师, I want 主程序具有清晰的代码块结构, so that 我可以快速理解和修改仿真脚本。

#### Acceptance Criteria

1. THE 主程序模板 SHALL 包含"导入与初始化"代码块
2. THE 主程序模板 SHALL 包含"定义光学系统"代码块
3. THE 主程序模板 SHALL 包含"定义光源"代码块
4. THE 主程序模板 SHALL 包含"系统信息展示"代码块
5. THE 主程序模板 SHALL 包含"执行仿真"代码块
6. THE 主程序模板 SHALL 包含"结果展示与保存"代码块
7. WHEN 用户查看主程序 THEN 每个代码块 SHALL 有清晰的注释分隔

### Requirement 10: 向后兼容性

**User Story:** As a 现有用户, I want 新 API 与现有代码兼容, so that 我不需要完全重写现有脚本。

#### Acceptance Criteria

1. THE MATLAB_Style_API SHALL 保留现有 `HybridSimulator` 类作为备选接口
2. THE MATLAB_Style_API SHALL 保留现有 `SimulationResult` 类的所有方法
3. WHEN 用户使用旧 API THEN THE MATLAB_Style_API SHALL 正常工作并输出弃用警告
