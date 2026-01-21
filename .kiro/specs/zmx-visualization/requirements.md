# 需求文档

## 简介

本模块提供基于 zemax-optical-axis-tracing 规范的 ZMX 文件加载与可视化绘图功能。该模块复用现有的 ZMX 解析器和坐标转换功能，创建独立的可视化模块，支持 2D 和 3D 空间中的光学系统可视化，绘制光学面形的实际形状。

## 术语表

- **ZMX_Parser**: ZMX 文件解析器，将 Zemax .zmx 文件转换为结构化数据模型
- **GlobalSurfaceDefinition**: 全局坐标系中的表面定义，包含顶点位置、姿态矩阵、曲率半径等参数
- **SurfaceTraversalAlgorithm**: 表面遍历算法，追踪当前坐标系状态并生成全局坐标表面定义
- **Visualization_Module**: 可视化模块，负责将全局坐标表面定义渲染为 2D 或 3D 图形
- **Surface_Renderer**: 表面渲染器，根据表面类型和参数计算并绘制面形
- **Optical_Axis**: 光轴，光学系统中光线传播的主要方向
- **Semi_Aperture**: 半口径，表面有效区域的半径

## 需求

### 需求 1：ZMX 文件加载

**用户故事：** 作为光学工程师，我希望能够加载 ZMX 文件并获取全局坐标表面定义，以便进行可视化绘图。

#### 验收标准

1. WHEN 用户提供有效的 ZMX 文件路径 THEN ZMX_Parser SHALL 解析文件并返回 ZmxDataModel 对象
2. WHEN ZmxDataModel 被解析后 THEN SurfaceTraversalAlgorithm SHALL 遍历所有表面并生成 GlobalSurfaceDefinition 列表
3. WHEN 解析完成后 THEN 系统 SHALL 提供便捷函数直接从 ZMX 文件路径获取 GlobalSurfaceDefinition 列表
4. IF ZMX 文件不存在或格式无效 THEN 系统 SHALL 抛出描述性错误信息

### 需求 2：表面面形计算

**用户故事：** 作为光学工程师，我希望系统能够根据表面参数计算实际面形，以便准确绘制光学元件。

#### 验收标准

1. WHEN 表面类型为平面（radius = ∞）THEN Surface_Renderer SHALL 计算平面面形
2. WHEN 表面类型为球面（conic = 0）THEN Surface_Renderer SHALL 使用球面方程计算面形
3. WHEN 表面类型为非球面（conic ≠ 0）THEN Surface_Renderer SHALL 使用圆锥曲面方程计算面形
4. WHEN 表面具有非球面系数 THEN Surface_Renderer SHALL 在圆锥曲面基础上添加高阶非球面项
5. WHEN 计算面形时 THEN Surface_Renderer SHALL 使用 semi_aperture 参数限制绘制范围
6. IF semi_aperture 为零或未定义 THEN Surface_Renderer SHALL 使用默认值或根据系统入瞳直径推断

### 需求 3：2D 可视化绘图

**用户故事：** 作为光学工程师，我希望能够在 2D 平面上查看光学系统的布局，以便快速理解系统结构。

#### 验收标准

1. WHEN 用户请求 2D 可视化 THEN Visualization_Module SHALL 支持 YZ、XZ、XY 三种投影平面
2. WHEN 绘制 2D 视图时 THEN Visualization_Module SHALL 在全局坐标系中绘制所有光学表面
3. WHEN 绘制表面时 THEN Visualization_Module SHALL 使用不同颜色区分反射镜和透射元件
4. WHEN 绘制完成后 THEN Visualization_Module SHALL 显示坐标轴标签和网格
5. WHEN 用户指定 THEN Visualization_Module SHALL 支持自定义图形大小、坐标范围和标题
6. WHEN 绘制 2D 视图时 THEN Visualization_Module SHALL 可选地绘制光轴方向指示

### 需求 4：3D 可视化绘图

**用户故事：** 作为光学工程师，我希望能够在 3D 空间中查看光学系统，以便全面理解复杂折叠光路的空间布局。

#### 验收标准

1. WHEN 用户请求 3D 可视化 THEN Visualization_Module SHALL 使用 matplotlib 3D 或 VTK 渲染光学系统
2. WHEN 绘制 3D 视图时 THEN Visualization_Module SHALL 在全局坐标系中绘制所有光学表面的完整面形
3. WHEN 绘制反射镜时 THEN Visualization_Module SHALL 使用旋转对称或网格方式生成 3D 表面
4. WHEN 绘制透镜时 THEN Visualization_Module SHALL 绘制前后两个表面
5. WHEN 绘制完成后 THEN Visualization_Module SHALL 支持交互式旋转和缩放
6. WHEN 用户指定 THEN Visualization_Module SHALL 支持自定义视角、光照和材质属性

### 需求 5：光轴追踪可视化

**用户故事：** 作为光学工程师，我希望能够可视化光轴的传播路径，以便理解折叠光路中光轴方向的变化。

#### 验收标准

1. WHEN 绘制光学系统时 THEN Visualization_Module SHALL 可选地绘制光轴路径
2. WHEN 绘制光轴时 THEN Visualization_Module SHALL 使用箭头或线段表示光轴方向
3. WHEN 光轴经过反射镜时 THEN Visualization_Module SHALL 正确显示光轴方向的改变
4. WHEN 绘制光轴时 THEN Visualization_Module SHALL 使用不同颜色或线型区分入射和出射光轴

### 需求 6：示例脚本

**用户故事：** 作为开发者，我希望有示例脚本演示如何使用可视化模块，以便快速上手。

#### 验收标准

1. WHEN 用户运行示例脚本 THEN 系统 SHALL 成功加载 complicated_fold_mirrors_setup_v2.zmx 文件
2. WHEN 示例脚本执行时 THEN 系统 SHALL 生成 2D 可视化图形（YZ 投影）
3. WHEN 示例脚本执行时 THEN 系统 SHALL 生成 3D 可视化图形
4. WHEN 示例脚本执行时 THEN 系统 SHALL 打印表面信息摘要
5. WHEN 用户需要时 THEN 系统 SHALL 提供使用其他 ZMX 测试文件的示例

### 需求 7：与 optiland 可视化模块的兼容性

**用户故事：** 作为开发者，我希望可视化模块能够与 optiland 的可视化功能兼容，以便在需要时复用其功能。

#### 验收标准

1. WHEN 可视化模块设计时 THEN 系统 SHALL 评估 optiland OpticViewer 的复用可能性
2. WHEN optiland 转换成功时 THEN 系统 SHALL 支持使用 optiland 的 OpticViewer 进行可视化
3. WHEN optiland 转换不完整时 THEN 系统 SHALL 提供独立的可视化实现作为备选
4. WHEN 使用独立实现时 THEN 系统 SHALL 遵循与 optiland 类似的 API 设计风格
