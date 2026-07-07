# 需求文档：ZOSPy POP 集成

## 简介

将 ZOSPy（Zemax OpticStudio Python 包装器）集成到现有的 POP（Physical Optics Propagation）API 中，使用户可以通过 `use_zemax_api=True` 参数切换到 Zemax OpticStudio 执行 POP 仿真。集成支持逐面定义和 ZMX 文件直接读取两种系统定义方式，且不影响原有 POP API 的任何功能。

## 术语表

- **POP_API**：现有的 Physical Optics Propagation API，位于 `pop/` 目录下
- **ZOSPy**：Zemax OpticStudio 的 Python 包装器库
- **OpticStudio**：Zemax 商业光学设计软件
- **LDE**：Lens Data Editor（透镜数据编辑器），OpticStudio 中定义光学系统的界面
- **PropagationResult**：POP_API 的仿真结果容器，包含各面状态、振幅、相位、Pilot Beam 参数等
- **PropagationOptions**：POP_API 的传播选项配置类
- **System**：POP_API 的光学系统定义类，支持 `from_zmx()` 和手动 `add_mirror()`/`add_lens()` 方式
- **GaussianSource**：POP_API 的高斯光源定义类
- **GlobalSurfaceDefinition**：POP_API 中描述单个光学表面的全局定义数据结构
- **ZMX_文件**：Zemax OpticStudio 的光学系统文件格式
- **Zemax_适配器**：负责将 POP_API 的数据结构转换为 ZOSPy 调用的中间模块

## 需求

### 需求 1：Zemax API 开关选项

**用户故事：** 作为光学工程师，我希望通过一个简单的布尔参数切换仿真后端，以便在自有传播算法和 Zemax OpticStudio 之间灵活选择。

#### 验收标准

1. THE PropagationOptions SHALL 包含一个 `use_zemax_api` 布尔字段，默认值为 `False`
2. WHEN `use_zemax_api` 为 `False` 时，THE POP_API SHALL 使用原有的 PROPER 库传播逻辑执行仿真，行为与集成前完全一致
3. WHEN `use_zemax_api` 为 `True` 时，THE POP_API SHALL 使用 ZOSPy 连接 OpticStudio 执行 POP 分析

### 需求 2：ZOSPy 连接管理

**用户故事：** 作为光学工程师，我希望系统自动管理与 OpticStudio 的连接，以便我无需手动处理连接细节。

#### 验收标准

1. WHEN `use_zemax_api` 为 `True` 时，THE Zemax_适配器 SHALL 自动建立与 OpticStudio 的 standalone 模式连接
2. WHEN 仿真完成后，THE Zemax_适配器 SHALL 正确释放 OpticStudio 连接资源
3. IF ZOSPy 库未安装，THEN THE POP_API SHALL 抛出 `ImportError` 并提供明确的安装指引信息
4. IF OpticStudio 未安装或连接失败，THEN THE Zemax_适配器 SHALL 抛出 `RuntimeError` 并提供明确的错误描述

### 需求 3：逐面定义系统的 Zemax 构建

**用户故事：** 作为光学工程师，我希望通过 `add_mirror()`/`add_lens()` 手动定义的光学系统也能使用 Zemax 后端仿真，以便与 Zemax 结果进行对比验证。

#### 验收标准

1. WHEN System 包含通过 `add_mirror()` 添加的标准反射镜面时，THE Zemax_适配器 SHALL 在 LDE 中创建对应的 Standard 类型反射面，包含正确的曲率半径、圆锥常数和半口径
2. WHEN System 包含通过 `add_mirror()` 添加的带倾斜角的反射镜面时，THE Zemax_适配器 SHALL 在 LDE 中使用 Coordinate Break 面正确表达倾斜和偏心
3. WHEN System 包含通过 `add_lens()` 添加的近轴透镜时，THE Zemax_适配器 SHALL 在 LDE 中创建对应的 Paraxial 类型面，包含正确的焦距
4. WHEN System 包含 biconic 类型表面时，THE Zemax_适配器 SHALL 在 LDE 中创建 Biconic 类型面，包含正确的 Rx、Ry、Kx、Ky 参数
5. WHEN System 包含 even_asphere 类型表面时，THE Zemax_适配器 SHALL 在 LDE 中创建 Even Asphere 类型面，包含正确的曲率半径、圆锥常数和非球面系数

### 需求 4：ZMX 文件直接加载

**用户故事：** 作为光学工程师，我希望通过 `System.from_zmx()` 加载的 ZMX 文件能直接被 Zemax 后端读取，以便获得与 Zemax 原生一致的仿真结果。

#### 验收标准

1. WHEN System 通过 `from_zmx()` 加载且 `use_zemax_api` 为 `True` 时，THE Zemax_适配器 SHALL 通过 ZOSPy 直接加载原始 ZMX 文件到 OpticStudio
2. WHEN ZMX 文件路径无效或文件损坏时，THE Zemax_适配器 SHALL 抛出 `FileNotFoundError` 或 `ValueError` 并提供明确的错误描述

### 需求 5：光源参数映射

**用户故事：** 作为光学工程师，我希望 GaussianSource 的参数能正确映射到 ZOSPy POP 分析的光束参数，以便两种后端使用相同的光源定义。

#### 验收标准

1. THE Zemax_适配器 SHALL 将 GaussianSource 的 `wavelength_um` 映射为 ZOSPy POP 分析的波长参数
2. THE Zemax_适配器 SHALL 将 GaussianSource 的 `w0_mm` 映射为 ZOSPy POP 分析的 GaussianWaist 光束类型的束腰半径参数
3. THE Zemax_适配器 SHALL 将 GaussianSource 的 `grid_size` 映射为 ZOSPy POP 分析的 x_sampling 和 y_sampling 参数
4. WHEN GaussianSource 指定了 `physical_size_mm` 时，THE Zemax_适配器 SHALL 将其映射为 ZOSPy POP 分析的 x_width 和 y_width 参数

### 需求 6：POP 分析执行与结果转换

**用户故事：** 作为光学工程师，我希望 Zemax POP 分析的结果能转换为现有的 PropagationResult 格式，以便使用现有的分析和可视化工具。

#### 验收标准

1. WHEN Zemax POP 分析完成后，THE Zemax_适配器 SHALL 将 Irradiance 数据转换为 PropagationResult 的 amplitude 字段（取平方根）
2. WHEN Zemax POP 分析完成后，THE Zemax_适配器 SHALL 将 Phase 数据转换为 PropagationResult 的 phase 字段
3. WHEN Zemax POP 分析完成后，THE Zemax_适配器 SHALL 构建包含 final_state 和 grid_sampling 的完整 PropagationResult 对象
4. THE PropagationResult SHALL 包含 `success` 标志和 `error_message` 字段以反映 Zemax 仿真的执行状态
5. IF Zemax POP 分析执行失败，THEN THE Zemax_适配器 SHALL 返回 `success=False` 的 PropagationResult 并在 `error_message` 中包含失败原因

### 需求 7：ZOSPy 可用性检测

**用户故事：** 作为光学工程师，我希望系统能在运行前检测 ZOSPy 和 OpticStudio 的可用性，以便在环境不满足时获得清晰的错误提示而非不可理解的异常。

#### 验收标准

1. THE POP_API SHALL 提供一个 `check_zemax_availability()` 函数，返回 ZOSPy 和 OpticStudio 的可用性状态
2. WHEN `use_zemax_api` 为 `True` 且 ZOSPy 不可用时，THE POP_API SHALL 在仿真开始前立即抛出 `ImportError` 并提供安装指引
3. WHEN `use_zemax_api` 为 `False` 时，THE POP_API SHALL 不检查 ZOSPy 的可用性，不导入 ZOSPy 相关模块

### 需求 8：向后兼容性

**用户故事：** 作为现有 POP API 的用户，我希望 ZOSPy 集成不影响我现有的仿真代码和结果，以便无缝升级。

#### 验收标准

1. WHEN `use_zemax_api` 未指定或为 `False` 时，THE POP_API SHALL 产生与集成前完全相同的仿真结果
2. THE POP_API SHALL 保持所有现有公共接口（函数签名、类定义、返回类型）不变
3. WHEN ZOSPy 未安装时，THE POP_API SHALL 在 `use_zemax_api=False` 模式下正常运行，无任何导入错误或警告

### 需求 9：Zemax 仿真结果的可视化与分析

**用户故事：** 作为光学工程师，我希望 Zemax 后端的仿真结果能使用现有的可视化和分析工具，以便对比两种后端的结果。

#### 验收标准

1. WHEN Zemax POP 分析返回 PropagationResult 后，THE PropagationResult SHALL 支持调用 `plot()` 方法绘制最终振幅和相位分布图
2. WHEN Zemax POP 分析返回 PropagationResult 后，THE PropagationResult SHALL 支持调用 `get_final_amplitude()`、`get_final_phase()`、`get_final_intensity()` 等现有分析方法
3. WHEN Zemax POP 分析返回 PropagationResult 后，THE PropagationResult SHALL 支持调用 `save_report()` 方法生成包含汇总表格和图片的报告
4. WHEN Zemax POP 分析返回包含多面数据的结果时，THE PropagationResult SHALL 支持调用 `plot_surface()` 方法绘制指定表面的波前分析图
5. THE Zemax_适配器 SHALL 在 PropagationResult 中填充正确的 `grid_sampling` 信息（grid_size、physical_size_mm、sampling_mm），以确保可视化工具的坐标轴标注正确
