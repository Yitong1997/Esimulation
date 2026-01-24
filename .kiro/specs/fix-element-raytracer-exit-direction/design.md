# 设计文档

## 简介

本文档描述修复 `ElementRaytracer._compute_exit_chief_direction` 方法的设计方案。该方法当前使用 optiland 的 `dy` 参数来处理离轴抛物面，但这种方式无法正确计算出射方向。

## 问题分析

### 当前实现的问题

当前 `_compute_exit_chief_direction` 方法（第 1269-1370 行）的流程：

1. 创建临时 optiland 光学系统
2. 使用 `dy=surface_def.off_axis_distance` 设置离轴距离
3. 追迹主光线 `trace_generic(Hx=0, Hy=0, Px=0, Py=0)`
4. 从追迹结果提取出射方向

**问题根源**：optiland 的 `dy` 参数将整个抛物面平移，而不是让光束在抛物面的离轴位置入射。

| 参数 | 期望行为 | optiland `dy` 实际行为 |
|------|----------|----------------------|
| 抛物面顶点 | 位于坐标原点 | 平移到 (0, dy, 0) |
| 主光线入射位置 | (0, d, z) | (0, 0, z) |
| 交点 z 坐标 | d²/(2R) | ≈ 0 |
| 出射方向 | 正确的离轴反射 | 近似轴上反射 |

### 验证数据

对于 "长焦距_离轴" 参数（f=2000mm, d=200mm, R=4000mm）：

| 项目 | 期望值 | 当前输出 |
|------|--------|----------|
| 出射角度 | 5.7248° | 0° |
| 出射方向 | [0, -0.0998, -0.995] | [0, 0, 1] |

## 设计方案

### 方案概述

对于抛物面（conic=-1）且有离轴距离的情况，使用解析几何方法直接计算主光线与抛物面的交互，而不依赖 optiland 的追迹。

### 解析几何计算方法

#### 1. 抛物面方程

抛物面在当前坐标系中的方程：
```
z = (x² + y²) / (2R)
```
其中 R 是曲率半径。

#### 2. 主光线与抛物面的交点

对于沿 +Z 方向入射、在 (0, d, 0) 位置的主光线：
- 交点 x 坐标：0
- 交点 y 坐标：d（离轴距离）
- 交点 z 坐标：d²/(2R)

#### 3. 表面法向量

抛物面的隐式方程：F(x, y, z) = x² + y² - 2Rz = 0

梯度：∇F = (2x, 2y, -2R)

归一化法向量（指向入射侧）：
```
n = (-x/R, -y/R, 1) / |n|
```

在交点 (0, d, d²/(2R)) 处：
```
n = (0, -d/R, 1) / sqrt(1 + d²/R²)
```

#### 4. 反射方向

使用反射公式：
```
r = i - 2(i·n)n
```

其中 i = (0, 0, 1) 是入射方向。

### 实现策略

在 `_compute_exit_chief_direction` 方法中添加条件分支：

```python
def _compute_exit_chief_direction(self) -> Tuple[float, float, float]:
    surface = self.surfaces[0]
    
    # 对于离轴抛物面，使用解析方法
    if (surface.conic == -1.0 and 
        surface.off_axis_distance != 0.0 and
        surface.surface_type == 'mirror'):
        return self._compute_oap_exit_direction_analytic(surface)
    
    # 其他情况使用 optiland 追迹
    return self._compute_exit_direction_via_optiland()
```

### 新增方法

```python
def _compute_oap_exit_direction_analytic(
    self,
    surface: SurfaceDefinition,
) -> Tuple[float, float, float]:
    """使用解析方法计算离轴抛物面的出射主光线方向
    
    参数:
        surface: 离轴抛物面的表面定义
    
    返回:
        出射主光线方向 (L, M, N)，全局坐标系，归一化
    """
```

### 坐标系处理

1. **入射面局部坐标系**：主光线沿 +Z 方向入射
2. **解析计算**：在入射面局部坐标系中进行
3. **结果转换**：使用 `self.rotation_matrix` 转换到全局坐标系

### 倾斜处理

如果表面有倾斜（tilt_x 或 tilt_y），需要：
1. 将入射方向转换到表面局部坐标系
2. 在表面局部坐标系中计算反射
3. 将反射方向转换回入射面局部坐标系

对于当前的 OAP 用例，倾斜角度由离轴几何自动确定，不需要额外的 tilt_x/tilt_y 参数。

## 数据流

```
输入：SurfaceDefinition（包含 radius, conic, off_axis_distance）
  ↓
判断：是否为离轴抛物面？
  ↓
┌─ 是：解析计算
│   1. 计算交点 (0, d, d²/(2R))
│   2. 计算法向量 (0, -d/R, 1)/|n|
│   3. 计算反射方向 r = i - 2(i·n)n
│   4. 转换到全局坐标系
│
└─ 否：optiland 追迹（现有逻辑）
  ↓
输出：出射方向 (L, M, N)
```

## 验证方法

### 单元测试

1. 轴上抛物面（d=0）：出射方向应为 [0, 0, -1]
2. 离轴抛物面：出射角度应等于 2×arctan(d/R)
3. 多组参数验证：与 `ChiefRayTracer` 的理论计算对比

### 集成测试

1. 运行 `step2_coordinate_system.py` 验证脚本
2. 所有 4 组参数应通过出射方向一致性验证
3. 方向误差应小于 0.01

### 回归测试

1. 倾斜平面镜测试：所有角度 RMS < 1 milli-wave
2. 离轴抛物面镜测试：相位 RMS < 10 milli-waves

## 正确性属性

### 属性 1：出射角度等于理论值

**描述**：对于离轴抛物面，出射角度应等于 2×arctan(d/R)

**验证方法**：
```python
# 理论出射角度
theoretical_angle = 2 * np.arctan(d / R)

# 实际出射角度（从出射方向计算）
actual_angle = np.arctan2(-exit_dir[1], -exit_dir[2])

# 误差应小于 0.01°
assert abs(actual_angle - theoretical_angle) < np.radians(0.01)
```

### 属性 2：轴上情况的正确性

**描述**：当 d=0 时，出射方向应为 [0, 0, -1]

**验证方法**：
```python
if off_axis_distance == 0:
    assert np.allclose(exit_direction, [0, 0, -1], atol=1e-10)
```

### 属性 3：与 ChiefRayTracer 一致

**描述**：ElementRaytracer 的出射方向应与 ChiefRayTracer 的理论计算一致

**验证方法**：
```python
# ChiefRayTracer 计算
tracer = ChiefRayTracer()
result = tracer.trace_chief_ray(oap_params)
expected_dir = result['reflection_direction']

# ElementRaytracer 计算
raytracer = ElementRaytracer(surfaces=[surface], wavelength=0.55)
actual_dir = raytracer.trace_chief_ray()

# 方向误差应小于 0.01
error = np.linalg.norm(np.array(actual_dir) - np.array(expected_dir))
assert error < 0.01
```

## 影响范围

### 修改的文件

- `src/wavefront_to_rays/element_raytracer.py`
  - 修改 `_compute_exit_chief_direction` 方法
  - 新增 `_compute_oap_exit_direction_analytic` 方法

### 不修改的文件

- `SurfaceDefinition` 类：接口保持不变
- 其他公共 API：保持不变
- 测试文件：只需运行现有验证脚本

## 风险评估

### 低风险

- 修改是内部实现改进，不改变外部 API
- 只影响离轴抛物面的出射方向计算
- 其他表面类型（平面镜、球面镜）继续使用 optiland 追迹

### 验证覆盖

- 4 组测试参数覆盖轴上和离轴情况
- 回归测试确保不破坏现有功能
