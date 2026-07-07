# 需求文档

## 简介

本文档定义了 POP（Physical Optics Propagation）仿真系统 Code Review 文档的需求。目标是生成一系列详细的 steering markdown 文件，全面描述整个 POP 仿真的算法流程、数据流、以及子模块功能与计算过程。

## 术语表

- **POP**: Physical Optics Propagation，物理光学传播仿真
- **Pilot_Beam**: 导引光束，用于追踪理想高斯光束参数的参考光束
- **ABCD_Matrix**: ABCD 矩阵，用于描述光学元件对高斯光束的变换
- **OPD**: Optical Path Difference，光程差
- **Wavefront**: 波前，描述光波相位分布的等相位面
- **Grid_Sampling**: 网格采样，波前的离散化表示参数
- **Coordinate_System**: 坐标系，包括全局坐标系和局部坐标系
- **Hybrid_Propagation**: 混合传播，结合物理光学和几何光学的传播方法
- **Element_Raytracer**: 元件光线追迹器，用于追迹光线穿过光学元件
- **Reconstructor**: 重建器，将稀疏光线数据重建为网格波前

## 需求

### 需求 1：主流程文档

**用户故事：** 作为开发者，我希望有一份主流程文档，以便快速理解 POP 仿真系统的整体架构和执行流程。

#### 验收标准

1. THE Documentation SHALL 描述 POP 仿真的完整执行流程，从光源定义到最终结果输出
2. THE Documentation SHALL 包含系统架构图（使用 Mermaid 流程图）
3. THE Documentation SHALL 列出所有核心模块及其职责
4. THE Documentation SHALL 描述模块间的调用关系和数据流向
5. THE Documentation SHALL 说明两种主要传播模式：自由空间传播和元件传播


### 需求 2：坐标系处理文档

**用户故事：** 作为开发者，我希望有一份详细的坐标系处理文档，以便理解系统中各种坐标变换的实现和物理意义。

#### 验收标准

1. THE Documentation SHALL 描述全局坐标系与局部坐标系的定义和关系
2. THE Documentation SHALL 说明 Zemax 坐标断点（Coordinate Break）的处理逻辑
3. THE Documentation SHALL 描述光轴追踪（Optical Axis Tracking）的实现方法
4. THE Documentation SHALL 说明入射面和出射面坐标系的构建方法
5. THE Documentation SHALL 描述旋转矩阵的计算方法（X→Y→Z 顺序）
6. THE Documentation SHALL 说明光线在全局坐标系和局部坐标系之间的转换过程
7. THE Documentation SHALL 包含坐标变换的数学公式和物理解释

### 需求 3：Pilot Beam 追踪文档

**用户故事：** 作为开发者，我希望有一份 Pilot Beam 追踪文档，以便理解理想高斯光束参数的追踪机制。

#### 验收标准

1. THE Documentation SHALL 描述 Pilot Beam 的物理意义和作用
2. THE Documentation SHALL 说明复参数 q 的定义和计算方法
3. THE Documentation SHALL 描述 ABCD 矩阵法在各种光学元件上的应用
4. WHEN 光束经过反射镜 THEN THE Documentation SHALL 说明曲率半径的更新方法
5. WHEN 光束经过折射面 THEN THE Documentation SHALL 说明折射率变化的处理
6. THE Documentation SHALL 描述 Pilot Beam 相位网格的计算公式
7. THE Documentation SHALL 说明 Pilot Beam 参数与 PROPER 库的同步机制


### 需求 4：自由空间传播文档

**用户故事：** 作为开发者，我希望有一份自由空间传播文档，以便理解基于 PROPER 库的衍射传播实现。

#### 验收标准

1. THE Documentation SHALL 描述自由空间传播的物理原理
2. THE Documentation SHALL 说明 PROPER 库的 prop_propagate 函数的使用方法
3. THE Documentation SHALL 描述传播距离的计算方法（考虑折射率）
4. THE Documentation SHALL 说明网格采样参数在传播过程中的变化
5. THE Documentation SHALL 描述参考面类型（PLANAR/SPHERI）的选择逻辑
6. THE Documentation SHALL 说明振幅和相位从 PROPER 对象提取的方法
7. THE Documentation SHALL 描述残差相位范围检查的意义和实现

### 需求 5：元件传播文档

**用户故事：** 作为开发者，我希望有一份元件传播文档，以便理解混合光线追迹方法的实现细节。

#### 验收标准

1. THE Documentation SHALL 描述元件传播的整体流程（波前→光线→波前）
2. THE Documentation SHALL 说明从波前采样光线的方法和参数
3. THE Documentation SHALL 描述 ElementRaytracer 的光线追迹过程
4. THE Documentation SHALL 说明 OPD 计算的方法（绝对 OPD 和残差 OPD）
5. THE Documentation SHALL 描述雅可比矩阵振幅计算的物理原理
6. THE Documentation SHALL 说明 RayToWavefrontReconstructor 的重建算法
7. THE Documentation SHALL 描述相位突变检测的实现和意义
8. WHEN 处理反射镜 THEN THE Documentation SHALL 说明法向量翻转的处理
9. WHEN 处理离轴抛物面 THEN THE Documentation SHALL 说明有效焦距的计算


### 需求 6：波前采样与重建文档

**用户故事：** 作为开发者，我希望有一份波前采样与重建文档，以便理解波前与光线之间的转换机制。

#### 验收标准

1. THE Documentation SHALL 描述从波前网格采样光线的算法
2. THE Documentation SHALL 说明光线方向余弦的计算方法（相位梯度 + Pilot Beam 曲率）
3. THE Documentation SHALL 描述采样光线的有效性验证条件
4. THE Documentation SHALL 说明 RBF 插值在坐标映射中的应用
5. THE Documentation SHALL 描述雅可比行列式的数值计算方法
6. THE Documentation SHALL 说明网格重采样（griddata）的插值策略
7. THE Documentation SHALL 描述输入振幅在重建过程中的保留方法

### 需求 7：Zemax 文件解析文档

**用户故事：** 作为开发者，我希望有一份 Zemax 文件解析文档，以便理解 ZMX 文件的解析和转换过程。

#### 验收标准

1. THE Documentation SHALL 描述 ZMX 文件格式的关键字段
2. THE Documentation SHALL 说明 ZmxParser 的解析流程
3. THE Documentation SHALL 描述 SurfaceTraversalAlgorithm 的遍历算法
4. THE Documentation SHALL 说明 GlobalSurfaceDefinition 的数据结构
5. THE Documentation SHALL 描述 ZemaxToOptilandConverter 的转换逻辑
6. WHEN 遇到坐标断点 THEN THE Documentation SHALL 说明 Order 参数的处理
7. THE Documentation SHALL 说明各种表面类型（standard、biconic、even_asphere、paraxial）的处理


### 需求 8：数据模型文档

**用户故事：** 作为开发者，我希望有一份数据模型文档，以便理解系统中核心数据结构的定义和用途。

#### 验收标准

1. THE Documentation SHALL 描述 PropagationState 的结构和各字段含义
2. THE Documentation SHALL 说明 PilotBeamParams 的参数定义和计算方法
3. THE Documentation SHALL 描述 GridSampling 的采样参数和一致性检查
4. THE Documentation SHALL 说明 OpticalAxisState 的光轴状态表示
5. THE Documentation SHALL 描述 SourceDefinition 的光源定义参数
6. THE Documentation SHALL 说明振幅/相位分离存储的设计决策（避免相位折叠）
7. THE Documentation SHALL 描述与 PROPER 库数据格式的转换方法

### 需求 9：光学系统定义文档

**用户故事：** 作为开发者，我希望有一份光学系统定义文档，以便理解如何定义和配置光学系统。

#### 验收标准

1. THE Documentation SHALL 描述 System 类的使用方法
2. THE Documentation SHALL 说明从 ZMX 文件加载系统的流程
3. THE Documentation SHALL 描述手动添加光学元件的 API
4. THE Documentation SHALL 说明表面定义的关键参数（曲率半径、圆锥常数、材料等）
5. THE Documentation SHALL 描述光学系统可视化的方法（2D/3D 布局图）
6. THE Documentation SHALL 说明表面法向量和曲率中心的计算


### 需求 10：高斯光束仿真文档

**用户故事：** 作为开发者，我希望有一份高斯光束仿真文档，以便理解高斯光束的创建和传播。

#### 验收标准

1. THE Documentation SHALL 描述 GaussianSource 的参数定义
2. THE Documentation SHALL 说明初始波前的创建过程（振幅和相位）
3. THE Documentation SHALL 描述 beam_diam_fraction 参数对网格采样的影响
4. THE Documentation SHALL 说明瑞利长度和发散角的计算
5. THE Documentation SHALL 描述与 PROPER 库 prop_begin 的参数对应关系
6. THE Documentation SHALL 说明参考面类型选择对相位表示的影响

### 需求 11：结果处理文档

**用户故事：** 作为开发者，我希望有一份结果处理文档，以便理解仿真结果的提取和分析方法。

#### 验收标准

1. THE Documentation SHALL 描述 PropagationResult 的结构和访问方法
2. THE Documentation SHALL 说明最终波前的提取（振幅、相位、光强）
3. THE Documentation SHALL 描述中间状态的存储和访问
4. THE Documentation SHALL 说明总光程的计算方法
5. THE Documentation SHALL 描述结果可视化的方法
6. THE Documentation SHALL 说明调试信息的输出和分析


### 需求 12：文档格式与组织

**用户故事：** 作为文档使用者，我希望文档具有统一的格式和清晰的组织结构，以便快速查找和理解信息。

#### 验收标准

1. THE Documentation SHALL 使用 Markdown 格式编写
2. THE Documentation SHALL 包含清晰的层级标题结构
3. THE Documentation SHALL 使用 Mermaid 绘制流程图和架构图
4. THE Documentation SHALL 在描述数据流时使用表格列出参数
5. THE Documentation SHALL 不包含伪代码，只描述算法流程和物理过程
6. THE Documentation SHALL 使用中文撰写，技术术语保留英文并提供解释
7. THE Documentation SHALL 在每个模块文档中包含"输入"、"输出"、"处理流程"三个部分
8. THE Documentation SHALL 在涉及坐标变换时提供数学公式

### 需求 13：模块间接口文档

**用户故事：** 作为开发者，我希望有一份模块间接口文档，以便理解各模块之间的数据传递和调用关系。

#### 验收标准

1. THE Documentation SHALL 描述 HybridOpticalPropagator 与子组件的接口
2. THE Documentation SHALL 说明 FreeSpacePropagator 的输入输出接口
3. THE Documentation SHALL 描述 HybridElementPropagator 的输入输出接口
4. THE Documentation SHALL 说明 ElementRaytracer 与 Reconstructor 的数据传递
5. THE Documentation SHALL 描述 StateConverter 的状态转换接口
6. THE Documentation SHALL 说明 PROPER 库接口的封装方法


### 需求 14：特殊场景处理文档

**用户故事：** 作为开发者，我希望有一份特殊场景处理文档，以便理解系统对各种边界情况的处理方法。

#### 验收标准

1. WHEN 处理倾斜平面镜 THEN THE Documentation SHALL 说明光轴折叠的处理
2. WHEN 处理离轴抛物面镜 THEN THE Documentation SHALL 说明主光线交点的计算
3. WHEN 处理折射面 THEN THE Documentation SHALL 说明材料检测和折射率获取
4. WHEN 处理坐标断点 THEN THE Documentation SHALL 说明连续坐标断点的处理
5. WHEN 处理近轴面形 THEN THE Documentation SHALL 说明理想薄透镜的处理
6. IF 光线追迹失败 THEN THE Documentation SHALL 说明错误处理机制
7. IF 相位突变超过阈值 THEN THE Documentation SHALL 说明警告和调试方法

### 需求 15：文档文件组织

**用户故事：** 作为文档维护者，我希望文档文件有清晰的组织结构，以便维护和更新。

#### 验收标准

1. THE Documentation SHALL 组织为以下文件结构：
   - `00-overview.md`: 系统概述和架构
   - `01-coordinate-system.md`: 坐标系处理
   - `02-pilot-beam.md`: Pilot Beam 追踪
   - `03-free-space-propagation.md`: 自由空间传播
   - `04-element-propagation.md`: 元件传播
   - `05-wavefront-sampling-reconstruction.md`: 波前采样与重建
   - `06-zmx-parsing.md`: Zemax 文件解析
   - `07-data-models.md`: 数据模型
   - `08-interfaces.md`: 模块接口
   - `09-special-cases.md`: 特殊场景处理
   - `10-code-issues.md`: 代码问题记录
2. THE Documentation SHALL 存放在 `.kiro/steering/pop-code-review/` 目录下
3. THE Documentation SHALL 在每个文件开头包含文档目的和适用范围说明


### 需求 16：代码问题检查与记录

**用户故事：** 作为开发者，我希望在 Code Review 过程中发现的代码问题被系统记录，以便后续修复和改进。

#### 验收标准

1. THE Documentation SHALL 创建 `10-code-issues.md` 文件记录所有发现的代码问题
2. THE Documentation SHALL 按模块分类记录问题（如坐标系处理、传播算法、数据模型等）
3. FOR EACH 问题 THE Documentation SHALL 记录以下信息：
   - 问题所在文件路径和行号（如可确定）
   - 问题类型（逻辑错误、潜在 Bug、代码风格、性能问题、文档缺失等）
   - 问题描述（清晰说明问题是什么）
   - 严重程度（高/中/低）
   - 建议修复方案（如有）
4. THE Documentation SHALL 使用表格格式组织问题列表，便于快速浏览
5. THE Documentation SHALL 在文档开头提供问题统计摘要（按类型和严重程度）
6. IF 发现物理公式实现错误 THEN THE Documentation SHALL 标注为高严重程度
7. IF 发现坐标变换逻辑错误 THEN THE Documentation SHALL 标注为高严重程度
8. THE Documentation SHALL 区分"确认的错误"和"潜在风险点"

