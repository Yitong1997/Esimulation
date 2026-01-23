# Requirements Document

## Introduction

本文档定义了混合光学仿真项目"统一仿真入口"功能的需求。

**核心设计原则**：
1. **主程序极简**：主程序代码不超过 20 行，步骤清晰
2. **结果存储全面**：所有中间结果完整保存，支持后续任意测试和分析
3. **高度复用**：最大程度复用现有模块，不重复造轮子

**典型使用流程**（主程序示例）：
```python
from hybrid_simulation import HybridSimulator

# 步骤 1：创建仿真器
sim = HybridSimulator()

# 步骤 2：加载光学系统（二选一）
sim.load_zmx("system.zmx")  # 或
sim.add_flat_mirror(z=50, tilt_x=45)

# 步骤 3：定义光源
sim.set_source(wavelength_um=0.55, w0_mm=5.0, grid_size=256)

# 步骤 4：执行仿真
result = sim.run()

# 步骤 5：查看/保存结果
result.summary()
result.plot_all()
result.save("output/")
```

## Glossary

- **HybridSimulator**: 统一仿真入口主类，提供步骤化 API
- **SimulationResult**: 全面的结果容器，存储所有中间和最终结果
- **SurfaceRecord**: 单个表面的完整记录，包含入射面和出射面的所有数据
- **WavefrontData**: 波前数据结构，包含振幅、相位、Pilot Beam 参数
- **Pilot Beam**: 理想高斯光束参考，用于相位解包裹
- **ZMX File**: Zemax 序列模式光学系统定义文件
- **PROPER**: 物理光学传输库
- **optiland**: 几何光线追迹库

## Requirements

### Requirement 1: 主程序简洁性

**User Story:** As a 光学工程师, I want to 用不超过 10 行代码完成完整仿真, so that I can 快速验证光学设计。

#### Acceptance Criteria

1. THE HybridSimulator SHALL 提供链式调用 API，每个步骤一行代码
2. THE HybridSimulator SHALL 提供合理的默认参数，减少必填项
3. WHEN 用户调用 run() THEN THE HybridSimulator SHALL 自动完成所有内部步骤
4. THE HybridSimulator SHALL 在每个步骤提供清晰的进度输出（可选）
5. THE HybridSimulator SHALL 支持 verbose 模式控制输出详细程度

### Requirement 2: 光学系统加载

**User Story:** As a 光学工程师, I want to 从 ZMX 文件或代码定义加载光学系统, so that I can 灵活地配置仿真场景。

#### Acceptance Criteria

1. WHEN 用户调用 load_zmx(path) THEN THE HybridSimulator SHALL 解析文件并创建光学系统
2. WHEN 用户调用 add_flat_mirror(z, tilt_x, tilt_y) THEN THE HybridSimulator SHALL 添加平面反射镜
3. WHEN 用户调用 add_spherical_mirror(z, radius, tilt_x, tilt_y) THEN THE HybridSimulator SHALL 添加球面反射镜
4. WHEN 用户调用 add_paraxial_lens(z, focal_length) THEN THE HybridSimulator SHALL 添加薄透镜
5. IF ZMX 文件不存在或格式错误 THEN THE HybridSimulator SHALL 抛出描述性错误信息
6. THE HybridSimulator SHALL 返回 self 以支持链式调用

### Requirement 3: 光源定义

**User Story:** As a 光学工程师, I want to 一行代码定义入射高斯光束, so that I can 快速配置仿真光源。

#### Acceptance Criteria

1. WHEN 用户调用 set_source(wavelength_um, w0_mm, grid_size) THEN THE HybridSimulator SHALL 创建高斯光源
2. WHEN 用户未指定 physical_size_mm THEN THE HybridSimulator SHALL 自动计算合适的物理尺寸
3. WHEN 用户提供 initial_aberration 参数 THEN THE HybridSimulator SHALL 将其与理想高斯光束相乘
4. THE HybridSimulator SHALL 自动计算并存储 Pilot Beam 参数
5. THE HybridSimulator SHALL 返回 self 以支持链式调用

### Requirement 4: 仿真执行

**User Story:** As a 光学工程师, I want to 一键执行完整仿真, so that I can 快速获得结果。

#### Acceptance Criteria

1. WHEN 用户调用 run() THEN THE HybridSimulator SHALL 执行完整的混合传播流程
2. WHEN 仿真执行时 THEN THE HybridSimulator SHALL 自动追踪主光线穿过所有表面
3. WHEN 仿真执行时 THEN THE HybridSimulator SHALL 在每个表面处记录入射面和出射面状态
4. WHEN 仿真执行时 THEN THE HybridSimulator SHALL 使用 PROPER 执行自由空间衍射传播
5. WHEN 仿真执行时 THEN THE HybridSimulator SHALL 使用 optiland 执行几何光线追迹
6. THE run() 方法 SHALL 返回 SimulationResult 对象

### Requirement 5: 结果存储（全面性）

**User Story:** As a 测试工程师, I want to 访问仿真过程中的所有中间数据, so that I can 进行任意后续分析和测试。

#### Acceptance Criteria

1. THE SimulationResult SHALL 存储光学系统配置信息
2. THE SimulationResult SHALL 存储光源参数和初始波前
3. THE SimulationResult SHALL 为每个表面存储 SurfaceRecord 对象
4. THE SurfaceRecord SHALL 包含入射面 WavefrontData（振幅、相位、Pilot Beam 参数、网格采样）
5. THE SurfaceRecord SHALL 包含出射面 WavefrontData（振幅、相位、Pilot Beam 参数、网格采样）
6. THE SurfaceRecord SHALL 包含表面几何信息（位置、法向量、曲率半径、类型）
7. THE SurfaceRecord SHALL 包含光轴状态（入射方向、出射方向、光程）
8. THE WavefrontData SHALL 提供 get_residual_phase() 方法计算相对于 Pilot Beam 的残差
9. THE WavefrontData SHALL 提供 get_intensity() 方法计算光强分布
10. THE SimulationResult SHALL 存储总光程和仿真成功/失败状态
11. THE SimulationResult SHALL 支持通过索引或名称访问任意表面数据

### Requirement 6: 结果序列化

**User Story:** As a 测试工程师, I want to 保存和加载仿真结果, so that I can 在不同时间和环境中分析数据。

#### Acceptance Criteria

1. WHEN 用户调用 result.save(path) THEN THE SimulationResult SHALL 保存所有数据到指定目录
2. THE save 方法 SHALL 保存振幅和相位数组为 .npy 文件
3. THE save 方法 SHALL 保存配置和元数据为 JSON 文件
4. THE save 方法 SHALL 保存 Pilot Beam 参数为 JSON 文件
5. WHEN 用户调用 SimulationResult.load(path) THEN THE SimulationResult SHALL 从目录加载完整结果
6. THE 加载的结果 SHALL 与原始结果具有相同的数据访问接口

### Requirement 7: 可视化

**User Story:** As a 光学工程师, I want to 一键生成标准化的可视化图表, so that I can 快速理解仿真结果。

#### Acceptance Criteria

1. WHEN 用户调用 result.plot_all() THEN THE SimulationResult SHALL 绘制所有表面的振幅和相位
2. WHEN 用户调用 result.plot_surface(index) THEN THE SimulationResult SHALL 绘制指定表面的详细图表
3. THE 图表 SHALL 包含：仿真振幅、仿真相位、Pilot Beam 参考相位、残差相位
4. THE 图表 SHALL 标注：表面序号、表面类型、Pilot Beam 参数（曲率半径、光斑大小）
5. THE 图表 SHALL 标注：相位 RMS 误差（waves）、相位 PV 误差（waves）
6. WHEN 用户指定 save_path THEN THE SimulationResult SHALL 保存图表到文件
7. THE plot 方法 SHALL 支持 show=False 参数以禁止显示窗口

### Requirement 8: 单位约定

**User Story:** As a 光学工程师, I want to 使用一致的单位系统, so that I can 避免单位转换错误。

#### Acceptance Criteria

1. THE HybridSimulator SHALL 使用毫米 (mm) 作为长度单位
2. THE HybridSimulator SHALL 使用微米 (μm) 作为波长单位
3. THE HybridSimulator SHALL 使用弧度 (rad) 作为相位单位
4. THE HybridSimulator SHALL 使用度 (deg) 作为用户输入的角度单位（内部转换为弧度）
5. WHEN 与 PROPER 交互时 THEN THE HybridSimulator SHALL 自动进行单位转换
6. THE 所有数据类 SHALL 在属性文档中明确标注单位

### Requirement 9: 错误处理

**User Story:** As a 光学工程师, I want to 获得清晰的错误信息, so that I can 快速定位问题。

#### Acceptance Criteria

1. IF 光学系统未定义 THEN THE run() 方法 SHALL 抛出 ConfigurationError
2. IF 光源未定义 THEN THE run() 方法 SHALL 抛出 ConfigurationError
3. IF 仿真失败 THEN THE SimulationResult SHALL 包含 success=False 和 error_message
4. THE 错误信息 SHALL 包含失败位置（表面索引）和建议的解决方案

### Requirement 10: 与现有模块集成

**User Story:** As a 开发者, I want to 统一入口完全复用现有模块, so that I can 避免代码重复。

#### Acceptance Criteria

1. THE HybridSimulator SHALL 内部使用 HybridOpticalPropagator 执行传播
2. THE HybridSimulator SHALL 内部使用 load_optical_system_from_zmx 解析 ZMX 文件
3. THE HybridSimulator SHALL 内部使用 SourceDefinition 创建光源
4. THE SimulationResult SHALL 封装 PropagationResult 并提供更友好的接口
5. THE WavefrontData SHALL 封装 PropagationState 的数据
6. THE HybridSimulator SHALL 不重复实现任何已有功能
