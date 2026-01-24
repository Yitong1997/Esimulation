# 需求文档

## 简介

本文档定义了修复 `ElementRaytracer._compute_exit_chief_direction` 方法的需求。该方法当前在计算离轴抛物面（OAP）的出射主光线方向时存在 bug，导致返回错误的出射方向。

## 问题根源分析

### optiland 的 `dy` 参数问题

经过代码分析，发现 optiland 的 `dy`（偏心）参数**不适合用于定义离轴抛物面**：

| 方面 | Zemax/正确方式 | optiland `dy` 方式 |
|------|---------------|-------------------|
| 抛物面顶点位置 | 全局原点 | 偏移到 (0, dy, 0) |
| 光束入射位置 | (0, d, z) | (0, 0, z) |
| 交点计算 | 在离轴位置 d²/(2R) | 在顶点附近 ≈ 0 |
| 出射方向 | 正确的离轴反射 | 近似轴上反射 |

optiland 的 `dy` 参数将整个表面平移，而不是让光束在抛物面的离轴位置入射。这导致 `trace_generic(Hx=0, Hy=0, Px=0, Py=0)` 追迹的主光线实际上在抛物面顶点附近入射，而不是在离轴位置。

### 正确的解决方案

应该使用**解析几何方法**直接计算主光线与抛物面的交互：
1. 使用抛物面方程 `z = (x² + y²) / (2R)` 计算交点
2. 使用梯度计算表面法向量
3. 使用反射公式计算出射方向

这与 `scripts/oap_debug/step1_chief_ray.py` 中的 `ChiefRayTracer` 类实现一致。

## 术语表

- **ElementRaytracer**: 元件光线追迹器类，负责将输入光线通过光学表面进行追迹
- **OAP (Off-Axis Parabola)**: 离轴抛物面镜，一种常用的反射光学元件
- **Chief_Ray**: 主光线，从物点中心通过光瞳中心的光线
- **Exit_Direction**: 出射方向，光线经过光学元件后的传播方向
- **optiland**: 几何光线追迹库，用于计算光线与光学表面的交互
- **ChiefRayTracer**: 验证脚本中实现的正确主光线追迹器类
- **Analytic_Method**: 解析方法，使用数学公式直接计算几何关系

## 需求

### 需求 1：正确计算离轴抛物面的出射方向

**用户故事：** 作为光学仿真系统的用户，我希望 ElementRaytracer 能正确计算离轴抛物面的出射主光线方向，以便进行准确的混合光学传播仿真。

#### 验收标准

1. WHEN 追迹主光线通过离轴抛物面（f=2000mm, d=200mm）THEN ElementRaytracer SHALL 返回出射方向约为 [0, -0.0998, -0.995]（出射角约 5.7248°，向 -Z 方向反射）
2. WHEN 追迹主光线通过轴上抛物面（d=0mm）THEN ElementRaytracer SHALL 返回出射方向 [0, 0, -1]（沿 -Z 轴反射）
3. FOR ALL 有效的 OAP 参数组合，ElementRaytracer 计算的出射角度 SHALL 与理论值 2×arctan(d/R) 的误差小于 0.01°
4. THE ElementRaytracer SHALL 使用解析几何方法计算离轴抛物面的出射方向，而不依赖 optiland 的偏心参数

### 需求 2：实现解析几何计算方法

**用户故事：** 作为开发者，我希望使用解析几何方法计算主光线与抛物面的交互，以确保计算的正确性和可靠性。

#### 验收标准

1. THE ElementRaytracer SHALL 使用抛物面方程 z = (x² + y²) / (2R) 计算主光线与表面的交点
2. THE ElementRaytracer SHALL 使用抛物面梯度 ∇z = (x/R, y/R) 计算交点处的表面法向量
3. THE ElementRaytracer SHALL 使用反射公式 r = i - 2(i·n)n 计算反射方向
4. WHEN 表面具有 conic=-1（抛物面）且 off_axis_distance > 0 THEN ElementRaytracer SHALL 使用解析方法而非 optiland 追迹

### 需求 3：保持现有功能的兼容性

**用户故事：** 作为光学仿真系统的用户，我希望修复不会破坏现有的平面镜和球面镜功能，以便现有的仿真代码继续正常工作。

#### 验收标准

1. WHEN 追迹主光线通过平面镜 THEN ElementRaytracer SHALL 返回正确的反射方向
2. WHEN 追迹主光线通过球面镜 THEN ElementRaytracer SHALL 返回正确的反射方向
3. WHEN 追迹主光线通过倾斜平面镜 THEN ElementRaytracer SHALL 返回正确的反射方向
4. FOR ALL 现有的回归测试用例，修复后的代码 SHALL 继续通过

### 需求 4：通过所有验证测试

**用户故事：** 作为开发者，我希望修复后的代码能通过所有 4 组测试参数的验证，以确保修复的正确性和完整性。

#### 验收标准

1. WHEN 运行验证脚本 step2_coordinate_system.py THEN 所有 4 组参数（长焦距_轴上、长焦距_离轴、超长焦距_轴上、超长焦距_离轴）SHALL 通过出射方向一致性验证
2. WHEN 比较 ElementRaytracer 的出射方向与 ChiefRayTracer 的理论计算 THEN 方向误差 SHALL 小于 0.01
3. WHEN 运行倾斜平面镜回归测试 THEN 所有角度（0°-60°）的 RMS 误差 SHALL 小于 1 milli-wave

### 需求 5：内部实现改进

**用户故事：** 作为开发者，我希望修复是内部实现的改进，不改变外部 API，以便调用者无需修改代码。

#### 验收标准

1. THE ElementRaytracer SHALL NOT 添加新的公共 API 参数
2. THE SurfaceDefinition SHALL NOT 修改其接口
3. WHEN 调用 trace_chief_ray() 方法 THEN 返回值格式 SHALL 保持不变
4. THE 修复 SHALL 对外部调用者透明
