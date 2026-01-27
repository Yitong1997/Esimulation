# Requirements Document

## Introduction

本文档定义了混合光学传播仿真系统的代码审查需求。审查目标是识别当前系统中的架构问题，包括主光线重复追迹、坐标系定义分散、表面定义重复、ElementRaytracer 职责过重等问题，并生成代码审查报告，提出通过删减和简化现有模块来改进代码质量的建议。

**注意：本规范的目标是生成代码审查报告，而不是实际修改源代码。重构建议应聚焦于删减和简化现有模块，而不是添加新模块。**

## Glossary

- **Global_Coordinate_System**: 全局坐标系，右手系，Z 轴为初始光轴方向，Y 轴垂直向上
- **Current_Coordinate_System**: Zemax 序列模式中随光路演化的当前坐标系
- **Entrance_Plane_Coordinate_System**: 入射面局部坐标系，原点在入射面中心，Z 轴为入射方向
- **Exit_Plane_Coordinate_System**: 出射面局部坐标系，原点在出射面中心，Z 轴为出射方向
- **Chief_Ray**: 主光线，从物点中心通过光阑中心的光线
- **Optical_Axis_State**: 光轴状态，包含位置、方向和累积光程
- **GlobalSurfaceDefinition**: 全局坐标系中的表面定义数据结构
- **ElementRaytracer**: 元件光线追迹器，负责通过光学表面追迹光线
- **HybridOpticalPropagator**: 混合光学传播器，协调 PROPER 和 optiland
- **HybridElementPropagator**: 混合元件传播器，处理单个光学元件的传播
- **Code_Review_Report**: 代码审查报告，包含问题识别和改进建议

## Requirements

### Requirement 1: 识别主光线重复追迹问题

**User Story:** 作为代码审查者，我希望识别系统中主光线被重复追迹的所有位置，这样可以提出消除重复的建议。

#### Acceptance Criteria

1. THE Code_Review_Report SHALL 列出所有调用主光线追迹的代码位置
2. THE Code_Review_Report SHALL 分析每次主光线追迹的目的和必要性
3. THE Code_Review_Report SHALL 识别哪些主光线追迹是冗余的
4. THE Code_Review_Report SHALL 建议如何通过删减代码来消除重复追迹
5. THE Code_Review_Report SHALL 评估消除重复追迹后的性能提升预期

### Requirement 2: 识别坐标系定义分散问题

**User Story:** 作为代码审查者，我希望识别坐标系转换逻辑分散在多个文件中的问题，这样可以提出整合建议。

#### Acceptance Criteria

1. THE Code_Review_Report SHALL 列出所有包含坐标转换逻辑的文件和函数
2. THE Code_Review_Report SHALL 分析各处坐标转换逻辑的一致性
3. THE Code_Review_Report SHALL 识别重复的坐标转换代码
4. THE Code_Review_Report SHALL 建议如何通过删减重复代码来整合坐标转换逻辑
5. THE Code_Review_Report SHALL 评估整合后代码可维护性的提升

### Requirement 3: 识别表面定义重复问题

**User Story:** 作为代码审查者，我希望识别系统中表面定义格式不统一的问题，这样可以提出统一建议。

#### Acceptance Criteria

1. THE Code_Review_Report SHALL 列出所有创建表面定义的方法
2. THE Code_Review_Report SHALL 分析 `_create_surface_definition_for_tracing()` 和 `_create_surface_definition()` 的功能重叠
3. THE Code_Review_Report SHALL 识别哪些表面定义创建方法可以删除
4. THE Code_Review_Report SHALL 建议如何统一使用 GlobalSurfaceDefinition
5. THE Code_Review_Report SHALL 评估统一后代码简洁性的提升

### Requirement 4: 识别 ElementRaytracer 职责过重问题

**User Story:** 作为代码审查者，我希望识别 ElementRaytracer 类承担过多职责的问题，这样可以提出简化建议。

#### Acceptance Criteria

1. THE Code_Review_Report SHALL 分析 ElementRaytracer 类的所有职责
2. THE Code_Review_Report SHALL 识别哪些职责不属于光线追迹器的核心功能
3. THE Code_Review_Report SHALL 统计 ElementRaytracer 的代码行数和复杂度
4. THE Code_Review_Report SHALL 建议如何通过删减非核心功能来简化该类
5. THE Code_Review_Report SHALL 评估简化后代码可测试性的提升

### Requirement 5: 识别 HybridElementPropagator 复杂度问题

**User Story:** 作为代码审查者，我希望识别 HybridElementPropagator 类过于复杂的问题，这样可以提出简化建议。

#### Acceptance Criteria

1. THE Code_Review_Report SHALL 分析 HybridElementPropagator 类的所有方法
2. THE Code_Review_Report SHALL 识别哪些方法可以删除或合并
3. THE Code_Review_Report SHALL 统计 HybridElementPropagator 的代码行数和复杂度
4. THE Code_Review_Report SHALL 建议如何通过删减代码来简化该类
5. THE Code_Review_Report SHALL 评估简化后代码可读性的提升

### Requirement 6: 生成代码审查报告

**User Story:** 作为代码审查者，我希望生成一份完整的代码审查报告，这样可以清晰地传达发现的问题和改进建议。

#### Acceptance Criteria

1. THE Code_Review_Report SHALL 包含执行摘要，概述主要发现
2. THE Code_Review_Report SHALL 包含详细的问题列表，每个问题包含位置、描述和严重程度
3. THE Code_Review_Report SHALL 包含具体的改进建议，聚焦于删减和简化
4. THE Code_Review_Report SHALL 包含代码行数统计和复杂度分析
5. THE Code_Review_Report SHALL 包含优先级排序的改进建议
6. THE Code_Review_Report SHALL 使用 Markdown 格式输出到 `.kiro/specs/coordinate-system-refactoring/code-review-report.md`

### Requirement 7: 保持向后兼容性分析

**User Story:** 作为代码审查者，我希望分析任何建议的改动对现有 API 的影响，这样可以确保建议是可行的。

#### Acceptance Criteria

1. THE Code_Review_Report SHALL 分析每个改进建议对公共 API 的影响
2. THE Code_Review_Report SHALL 识别哪些改动可以在不破坏 API 的情况下进行
3. THE Code_Review_Report SHALL 标记任何可能破坏向后兼容性的建议
4. THE Code_Review_Report SHALL 建议如何在保持 API 兼容的前提下进行改进
