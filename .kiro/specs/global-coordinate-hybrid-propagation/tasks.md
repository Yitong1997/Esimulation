# 实现任务

## 任务概述

本任务列表实现全局坐标系混合传播功能，包括 `GlobalElementRaytracer` 和 `HybridElementPropagatorGlobal` 两个核心类，并集成到 BTS API 数据流中。

## ⚠️ 核心难点说明

### 主要难点：入射面光线的正确定义

在绝对坐标系下，需要将主光线的绝对坐标扩展为入射面和入射光线的绝对坐标。具体挑战包括：

1. **入射面定义**：入射面必须垂直于入射光轴，原点为主光线与元件表面的交点
2. **光线初始化**：从波前采样的光线需要正确转换到全局坐标系
3. **出射面定义**：出射面必须垂直于出射光轴，原点为主光线与元件表面的交点

### optiland 原生能力

optiland 已提供基于绝对坐标的光路追迹功能（参见 `optiland-master/docs/gallery/reflective/laser_system.ipynb`）：
- 使用 `x=, y=, z=` 定义表面顶点的绝对位置
- 使用 `rx=, ry=, rz=` 定义表面朝向
- 支持反射镜、折射面等多种表面类型

### 本任务的核心工作

在 optiland 原生能力基础上，实现：
1. 入射面和出射面的正确定义（垂直于入出射光轴）
2. 光线从波前采样到全局坐标系的转换
3. 光线从全局坐标系到出射面局部坐标系的转换
4. 与 BTS API 数据流的集成

## 任务列表

- [x] 1. 创建数据模型和基础设施
  - [x] 1.1 在 `src/wavefront_to_rays/global_element_raytracer.py` 中创建 `PlaneDef` 数据类，定义平面的位置和法向量 (Requirements 1.1, 1.2, 1.3)
  - [x] 1.2 创建 `GlobalElementRaytracer` 类的基本结构，包括构造函数和参数验证 (Requirements 1.1-1.5, 10.1-10.4)
  - [x] 1.3 实现法向量归一化验证逻辑，未归一化时抛出 ValueError (Requirements 1.2, 1.3)

- [x] 2. 实现 GlobalElementRaytracer 核心功能
  - [x] 2.1 实现 `_create_optic_global()` 方法，使用 optiland 原生的绝对坐标 API 创建光学系统 (Requirements 3.2, 4.1-4.4)
  - [x] 2.2 实现 `trace_chief_ray()` 方法，追迹主光线并计算出射方向和交点，确定入射面和出射面的原点 (Requirements 1.4, 1.5, 2.1, 2.2)
  - [x] 2.3 实现 `trace()` 方法，执行全局坐标系光线追迹 (Requirements 3.1, 3.3-3.6)
  - [x] 2.4 复用 `ElementRaytracer` 的带符号 OPD 计算逻辑 (Requirements 3.6, 7.1)
  - [x] 2.5 复用 `ElementRaytracer` 的抛物面反射修正逻辑 (Requirements 4.5)
  - [x] 2.6 实现出射面投影和 OPD 增量计算 (Requirements 2.3, 2.4, 2.5)

- [x] 3. 实现 HybridElementPropagatorGlobal 类（核心难点）
  - [x] 3.1 在 `src/hybrid_optical_propagation/hybrid_element_propagator_global.py` 中创建类的基本结构 (Requirements 8.1, 8.4)
  - [x] 3.2 **【关键】** 实现入射面定义：根据入射光轴方向和主光线交点，定义垂直于光轴的入射面 (Requirements 1.4, 1.5, 5.1)
  - [x] 3.3 **【关键】** 实现从入射面局部坐标系到全局坐标系的光线转换，将波前采样的光线扩展为全局坐标 (Requirements 5.1-5.4)
  - [x] 3.4 **【关键】** 实现出射面定义：根据出射光轴方向和主光线交点，定义垂直于光轴的出射面 (Requirements 2.1, 2.2)
  - [x] 3.5 实现从全局坐标系到出射面局部坐标系的光线转换 (Requirements 6.1-6.4)
  - [x] 3.6 实现 `propagate()` 方法，执行完整的混合传播流程 (Requirements 8.2, 8.3)
  - [x] 3.7 实现残差 OPD 计算：残差 OPD = 绝对 OPD + Pilot Beam OPD (Requirements 7.2, 7.3)
  - [x] 3.8 实现 Pilot Beam 参数更新逻辑 (Requirements 9.1-9.4)

- [x] 4. BTS API 集成
  - [x] 4.1 更新 `HybridOpticalPropagator` 以支持使用 `HybridElementPropagatorGlobal` (Requirements 8.4)
  - [x] 4.2 确保 `bts.simulate()` 数据流正确调用全局坐标系传播器
  - [x] 4.3 验证与现有 `GlobalSurfaceDefinition` 数据模型的兼容性

- [x] 5. 属性测试实现
  - [x] 5.1 (PBT) Property 1: 法向量归一化验证 - 验证系统正确接受归一化向量并拒绝非归一化向量 (Requirements 1.2, 1.3, 10.1)
  - [x] 5.2 (PBT) Property 2: 方向余弦归一化保持 - 验证坐标转换后方向余弦仍满足 L² + M² + N² = 1 (Requirements 5.2, 10.4)
  - [x] 5.3 (PBT) Property 3: 旋转矩阵正交性 - 验证 R × R^T = I 且 det(R) = 1 (Requirements 5.3, 6.3)
  - [x] 5.4 (PBT) Property 4: 坐标转换可逆性 - 验证局部→全局→局部转换后得到原始值 (Requirements 5.1, 6.1)
  - [x] 5.5 (PBT) Property 5: OPD 坐标转换不变性 - 验证 OPD 在坐标转换过程中保持不变 (Requirements 6.2)
  - [x] 5.6 (PBT) Property 6: 主光线 OPD 为零 - 验证主光线的相对 OPD 为 0 (Requirements 7.1)

- [x] 6. 集成测试和验证
  - [x] 6.1 (PBT) Property 9: 平面镜传输精度 - 验证不同倾斜角度（0°-60°）的平面镜 RMS < 1 milli-wave (Requirements 11.5)
  - [x] 6.2 创建验证脚本 `tests/integration/test_global_coordinate_hybrid_propagation.py`，测试复杂折叠镜系统 (Requirements 11.1, 11.2, 11.4)
  - [x] 6.3 实现与标准 `HybridElementPropagator` 的结果对比测试 (Requirements 11.3)
  - [x] 6.4 通过 BTS API 执行端到端测试，验证完整数据流

- [x] 7. 模块导出和文档
  - [x] 7.1 更新 `src/wavefront_to_rays/__init__.py`，导出 `GlobalElementRaytracer` 和 `PlaneDef`
  - [x] 7.2 更新 `src/hybrid_optical_propagation/__init__.py`，导出 `HybridElementPropagatorGlobal`
