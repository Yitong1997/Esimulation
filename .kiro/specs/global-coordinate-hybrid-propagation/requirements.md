# 需求文档

## 简介

本文档定义了"全局坐标系混合传播"（Global Coordinate Hybrid Propagation）功能的需求。该功能实现一个新的混合元件传播器（HybridElementPropagatorGlobal）和光线追迹器（GlobalElementRaytracer），直接在全局坐标系中操作，避免局部坐标变换以简化 optiland 集成并提高精度。

### 背景

当前的 `ElementRaytracer` 使用局部坐标变换方法：
1. 将光线从入射面局部坐标系转换到全局坐标系
2. 在 optiland 中进行光线追迹
3. 将结果从全局坐标系转换回出射面局部坐标系

这种方法在复杂折叠镜系统中可能引入坐标变换误差。新的全局坐标系方法旨在：
- 直接在全局坐标系中定义入射面和出射面
- 使用 optiland 的全局坐标能力进行光线追迹
- 简化计算流程并提高精度

### 核心目标

1. 创建 `GlobalElementRaytracer` 类，在全局坐标系中进行光线追迹
2. 创建 `HybridElementPropagatorGlobal` 类，使用全局坐标系方法进行混合传播
3. 验证新方法在复杂折叠镜系统中的精度

## 术语表

- **Global_Coordinate_System（全局坐标系）**: 整个光学系统使用的统一坐标系，Z 轴为初始光轴方向，Y 轴垂直向上
- **Local_Coordinate_System（局部坐标系）**: 相对于特定平面（入射面或出射面）的坐标系，Z 轴垂直于该平面
- **Entrance_Plane（入射面）**: 垂直于入射主光线的平面，由全局空间中的点和法向量定义
- **Exit_Plane（出射面）**: 垂直于出射主光线的平面，由全局空间中的点和法向量定义
- **Chief_Ray（主光线）**: 从物面中心出发、通过光阑中心的光线，用于定义光轴方向
- **Vertex_Position（顶点位置）**: 光学表面顶点在全局坐标系中的绝对位置 (x, y, z)
- **Surface_Normal（表面法向量）**: 光学表面在主光线交点处的法向量
- **OPD（光程差）**: Optical Path Difference，光线相对于主光线的光程差异
- **Pilot_Beam（导引光束）**: 用于计算参考相位的理想高斯光束
- **Residual_OPD（残差光程差）**: 实际出射面光线光程 与 Pilot Beam 理论光程的差值

## 需求

### 需求 1：全局坐标系入射面定义

**用户故事：** 作为光学仿真工程师，我希望在全局坐标系中定义入射面，以便直接使用全局坐标进行光线采样。

#### 验收标准

1. WHEN 创建 GlobalElementRaytracer THEN System SHALL 接受入射面的全局位置 (x, y, z) 和法向量 (nx, ny, nz) 作为参数
2. WHEN 入射面定义完成 THEN System SHALL 验证法向量已归一化（L² + M² + N² = 1）
3. WHEN 入射面法向量未归一化 THEN System SHALL 抛出 ValueError 并提供清晰的错误信息
4. THE GlobalElementRaytracer SHALL 使用入射主光线方向作为入射面法向量
5. THE GlobalElementRaytracer SHALL 使用主光线与表面的交点作为入射面原点

### 需求 2：全局坐标系出射面定义

**用户故事：** 作为光学仿真工程师，我希望在全局坐标系中定义出射面，以便正确计算出射光线的位置和 OPD。

#### 验收标准

1. WHEN 光线追迹完成 THEN System SHALL 根据出射主光线方向定义出射面法向量
2. WHEN 出射面定义完成 THEN System SHALL 使用主光线与表面的交点作为出射面原点
3. THE GlobalElementRaytracer SHALL 计算每条光线与出射面的交点位置
4. THE GlobalElementRaytracer SHALL 计算光线从表面到出射面的 OPD 增量
5. WHEN 光线方向与出射面法向量平行 THEN System SHALL 正确处理此边界情况

### 需求 3：全局坐标系光线追迹

**用户故事：** 作为光学仿真工程师，我希望在全局坐标系中进行光线追迹，以避免局部坐标变换带来的误差。

#### 验收标准

1. WHEN 输入光线到达 GlobalElementRaytracer THEN System SHALL 直接使用光线的全局坐标进行追迹
2. WHEN 设置 optiland 光学系统 THEN System SHALL 使用表面的绝对位置 (x, y, z) 和旋转角度 (rx, ry, rz) 定义表面
3. WHEN 追迹完成 THEN System SHALL 输出光线在全局坐标系中的位置和方向
4. THE GlobalElementRaytracer SHALL 支持反射镜（包括平面镜、球面镜、抛物面镜）
5. THE GlobalElementRaytracer SHALL 支持折射面
6. THE GlobalElementRaytracer SHALL 正确计算每条光线的累积 OPD

### 需求 4：表面定义与全局坐标

**用户故事：** 作为光学仿真工程师，我希望使用绝对坐标定义光学表面，以便与 ZMX 文件加载后的处理方式一致。

#### 验收标准

1. WHEN 定义光学表面 THEN System SHALL 接受顶点位置 (x, y, z) 作为绝对坐标
2. WHEN 定义光学表面 THEN System SHALL 接受旋转角度 (rx, ry, rz) 定义表面朝向
3. THE GlobalElementRaytracer SHALL 使用 optiland 的全局坐标能力设置表面
4. THE GlobalElementRaytracer SHALL 正确处理离轴系统（如 OAP）的表面定义
5. WHEN 表面为离轴抛物面 THEN System SHALL 通过顶点位置自然实现离轴效果，无需额外的离轴参数

### 需求 5：波前采样与全局坐标转换

**用户故事：** 作为光学仿真工程师，我希望将波前采样的光线正确转换到全局坐标系，以便进行全局光线追迹。

#### 验收标准

1. WHEN 从波前采样光线 THEN HybridElementPropagatorGlobal SHALL 使用入射光轴信息将光线转换到全局坐标系
2. WHEN 转换完成 THEN System SHALL 保持光线方向余弦的归一化（L² + M² + N² = 1）
3. THE HybridElementPropagatorGlobal SHALL 正确计算从入射面局部坐标系到全局坐标系的旋转矩阵
4. THE HybridElementPropagatorGlobal SHALL 正确应用入射面位置的平移变换

### 需求 6：出射光线到局部坐标系转换

**用户故事：** 作为光学仿真工程师，我希望将全局坐标系中的出射光线转换回出射面局部坐标系，以便进行波前重建。

#### 验收标准

1. WHEN 光线追迹完成 THEN HybridElementPropagatorGlobal SHALL 将出射光线从全局坐标系转换到出射面局部坐标系
2. WHEN 转换完成 THEN System SHALL 保持光线 OPD 不变（OPD 是标量，不受坐标变换影响）
3. THE HybridElementPropagatorGlobal SHALL 正确计算从全局坐标系到出射面局部坐标系的旋转矩阵
4. THE HybridElementPropagatorGlobal SHALL 使用出射面原点作为平移参考点

### 需求 7：OPD 计算与参考球面处理

**用户故事：** 作为光学仿真工程师，我希望在全局坐标系中正确计算 OPD，以便获得准确的波前信息。

#### 验收标准

1. WHEN 计算 OPD THEN System SHALL 使用主光线作为参考（主光线 OPD = 0）
2. WHEN 计算残差 OPD THEN System SHALL 使用公式：残差 OPD = 绝对 OPD + Pilot Beam OPD
3. THE HybridElementPropagatorGlobal SHALL 在出射面上计算 Pilot Beam 理论 OPD
4. THE HybridElementPropagatorGlobal SHALL 去除残差 OPD 中的低阶项（倾斜、二次、三次）
5. WHEN 残差 OPD 计算完成 THEN System SHALL 验证残差 OPD 平滑连续，无 2π 跳变

### 需求 8：混合传播流程

**用户故事：** 作为光学仿真工程师，我希望有一个统一的 API 执行全局坐标系混合传播，以便简化系统集成。

#### 验收标准

1. THE HybridElementPropagatorGlobal SHALL 提供 `propagate()` 方法执行完整的混合传播流程
2. WHEN 执行传播 THEN System SHALL 按以下顺序执行：
   - 从波前采样光线并转换到全局坐标系
   - 使用 GlobalElementRaytracer 进行追迹
   - 在全局坐标系中计算 OPD
   - 将光线转换回出射面局部坐标系
   - 重建波前
3. THE HybridElementPropagatorGlobal SHALL 返回出射面的传播状态（PropagationState）
4. THE HybridElementPropagatorGlobal SHALL 支持与现有 HybridElementPropagator 相同的接口

### 需求 9：Pilot Beam 参数更新

**用户故事：** 作为光学仿真工程师，我希望正确更新 Pilot Beam 参数，以便在出射面上进行正确的相位重建。

#### 验收标准

1. WHEN 光线通过反射镜 THEN System SHALL 使用 ABCD 变换更新 Pilot Beam 曲率半径
2. WHEN 光线通过离轴抛物面 THEN System SHALL 将出射波前视为近似平面波（R ≈ ∞）
3. WHEN 光线通过折射面 THEN System SHALL 使用折射面 ABCD 变换更新参数
4. THE HybridElementPropagatorGlobal SHALL 使用更新后的 Pilot Beam 参数计算出射面相位

### 需求 10：错误处理

**用户故事：** 作为光学仿真工程师，我希望系统能够正确处理各种错误情况，以便快速定位和解决问题。

#### 验收标准

1. IF 入射面法向量未归一化 THEN System SHALL 抛出 ValueError
2. IF 表面定义参数无效 THEN System SHALL 抛出 ValueError
3. IF 所有光线都无效 THEN System SHALL 抛出 SimulationError 并建议可能的解决方案
4. IF 光线方向余弦未归一化 THEN System SHALL 抛出 ValueError
5. WHEN 发生错误 THEN System SHALL 提供清晰的错误信息和诊断建议

### 需求 11：验证与测试

**用户故事：** 作为光学仿真工程师，我希望验证全局坐标系方法的精度，以确保其在复杂系统中的可靠性。

#### 验收标准

1. THE System SHALL 提供验证脚本用于测试复杂折叠镜系统
2. WHEN 运行验证测试 THEN System SHALL 计算 Pilot Beam 误差（光线追迹 OPD 与 Pilot Beam OPD 的差异）
3. THE System SHALL 支持与标准 HybridElementPropagator 的结果对比
4. WHEN 验证完成 THEN System SHALL 输出详细的误差分析报告
5. THE System SHALL 在不同倾斜角度的平面镜测试中达到 RMS < 1 milli-wave 的精度

