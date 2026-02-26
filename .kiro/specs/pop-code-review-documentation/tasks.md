# 实施计划：POP Code Review 文档

## 概述

本计划将创建 11 个 steering markdown 文件，全面描述 POP 仿真系统的算法流程、数据流和模块功能，并记录代码审查过程中发现的问题。

## 任务

- [x] 1. 创建文档目录结构
  - 创建 `.kiro/steering/pop-code-review/` 目录
  - _Requirements: 15.2_

- [x] 2. 创建系统概述文档
  - [x] 2.1 创建 `00-overview.md` 文件
    - 编写 POP 仿真系统整体架构描述
    - 添加系统架构 Mermaid 流程图
    - 列出核心模块及其职责
    - 描述两种传播模式（自由空间传播、元件传播）
    - 说明与 PROPER 和 optiland 库的集成关系
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 3. 创建坐标系处理文档
  - [x] 3.1 创建 `01-coordinate-system.md` 文件
    - 描述全局坐标系与局部坐标系的定义
    - 说明 CurrentCoordinateSystem 类的状态追踪
    - 描述坐标断点处理（Order=0 和 Order=1）
    - 添加旋转矩阵计算公式
    - 说明光轴追踪算法
    - 描述入射面/出射面坐标系构建
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7_


- [x] 4. 创建 Pilot Beam 追踪文档
  - [x] 4.1 创建 `02-pilot-beam.md` 文件
    - 描述 Pilot Beam 的物理意义
    - 说明复参数 q 的定义和计算
    - 描述 ABCD 矩阵法的应用
    - 添加各类光学元件的 ABCD 矩阵公式
    - 说明 Pilot Beam 相位网格计算
    - 描述与 PROPER 库的参数同步
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

- [x] 5. 创建自由空间传播文档
  - [x] 5.1 创建 `03-free-space-propagation.md` 文件
    - 描述衍射传播的物理原理
    - 说明 PROPER 库 prop_propagate 的使用
    - 描述传播距离计算方法
    - 说明网格采样参数的更新
    - 描述参考面类型选择逻辑
    - 说明振幅/相位提取方法
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_

- [x] 6. 创建元件传播文档
  - [x] 6.1 创建 `04-element-propagation.md` 文件
    - 描述混合传播的整体流程
    - 添加波前→光线→波前流程图
    - 说明光线采样过程
    - 描述 ElementRaytracer 的光线追迹
    - 说明 OPD 计算方法
    - 描述 Pilot Beam 参数更新
    - 说明残差 OPD 计算
    - 描述波前重建过程
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9_


- [x] 7. 创建波前采样与重建文档
  - [x] 7.1 创建 `05-wavefront-sampling-reconstruction.md` 文件
    - 描述波前采样算法
    - 说明方向余弦计算方法
    - 描述采样光线有效性验证
    - 说明 RBF 插值在坐标映射中的应用
    - 描述雅可比行列式数值计算
    - 说明网格重采样策略
    - 描述输入振幅保留方法
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

- [x] 8. 创建 Zemax 文件解析文档
  - [x] 8.1 创建 `06-zmx-parsing.md` 文件
    - 描述 ZMX 文件格式关键字段
    - 说明 ZmxParser 解析流程
    - 描述 SurfaceTraversalAlgorithm 遍历算法
    - 说明 GlobalSurfaceDefinition 数据结构
    - 描述 ZemaxToOptilandConverter 转换逻辑
    - 说明各种表面类型的处理
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7_

- [x] 9. 创建数据模型文档
  - [x] 9.1 创建 `07-data-models.md` 文件
    - 描述 PropagationState 结构
    - 说明 PilotBeamParams 参数定义
    - 描述 GridSampling 采样参数
    - 说明 OpticalAxisState 光轴状态
    - 描述 SourceDefinition 光源定义
    - 说明振幅/相位分离存储设计
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_


- [x] 10. 创建模块接口文档
  - [x] 10.1 创建 `08-interfaces.md` 文件
    - 描述 HybridOpticalPropagator 主接口
    - 说明 FreeSpacePropagator 接口
    - 描述 HybridElementPropagator 接口
    - 说明 ElementRaytracer 接口
    - 描述 RayToWavefrontReconstructor 接口
    - 说明 StateConverter 接口
    - 添加接口参数表格
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6_

- [x] 11. 创建特殊场景处理文档
  - [x] 11.1 创建 `09-special-cases.md` 文件
    - 描述倾斜平面镜处理
    - 说明离轴抛物面镜处理
    - 描述折射面处理
    - 说明连续坐标断点处理
    - 描述近轴面形处理
    - 说明错误处理机制
    - 描述相位突变调试方法
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7_

- [x] 12. 创建代码问题记录文档
  - [x] 12.1 创建 `10-code-issues.md` 文件
    - 在阅读各模块代码时记录发现的问题
    - 按模块分类组织问题（坐标系、传播算法、数据模型等）
    - 使用表格格式记录：文件路径、问题类型、严重程度、问题描述、建议修复方案
    - 标注问题严重程度（高/中/低）
    - 区分"确认的错误"和"潜在风险点"
    - 在文档开头提供问题统计摘要
    - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.5, 16.6, 16.7, 16.8_

- [x] 13. 检查点 - 文档完整性验证
  - 确认所有 11 个文档文件已创建
  - 验证每个文档包含必需章节
  - 检查 Mermaid 图表语法正确性
  - 确认文档间交叉引用有效
  - 验证代码问题记录完整

## 注意事项

- 所有文档使用中文撰写，技术术语保留英文并提供解释
- 使用 Mermaid 绘制流程图和架构图
- 不包含伪代码，只描述算法流程和物理过程
- 每个文档包含"概述"、"核心流程"、"数据流"等标准章节

