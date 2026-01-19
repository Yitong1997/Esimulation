# 混合光线追迹重构设计文档

## 1. 架构概述

### 1.1 核心设计原则

**完全复用 optiland 的光线追迹能力**：不手动实现任何光线传播逻辑，而是通过正确配置 optiland 光学系统（包括倾斜的出射面），让 optiland 完成整个追迹过程。

### 1.2 关键洞察

optiland 的 `add_surface()` 方法支持 `rx`, `ry` 参数，可以添加任意倾斜的表面。我们可以：
1. 预先计算出射主光线方向
2. 添加一个**倾斜的透明平面**（material='air'）作为出射面
3. 让 optiland 直接追迹到这个倾斜平面

### 1.3 组件关系

```
┌─────────────────────────────────────────────────────────────────┐
│                      ElementRaytracer                            │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  trace() 方法流程（简化版）                               │   │
│  │                                                           │   │
│  │  1. 坐标变换：入射面局部 → 全局                           │   │
│  │  2. optiland 追迹（元件表面 + 倾斜出射面）                │   │
│  │  3. 坐标变换：全局 → 出射面局部                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  _create_optic() 方法                                     │   │
│  │                                                           │   │
│  │  1. 预先计算出射主光线方向（反射定律）                    │   │
│  │  2. 从方向计算旋转角度 (rx, ry)                           │   │
│  │  3. 添加元件表面                                          │   │
│  │  4. 添加倾斜的透明平面作为出射面                          │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 详细设计

### 2.1 光学系统结构

```
optiland 光学系统：
- index=0: 物面（无穷远）
- index=1: 光学表面（反射镜，设为光阑）
- index=2: 倾斜的透明平面（出射面）
```

### 2.2 计算出射主光线方向

```python
def _compute_exit_chief_direction(self) -> Tuple[float, float, float]:
    """
    计算出射主光线方向
    
    对于反射镜：使用反射公式 r = d - 2(d·n)n
    
    返回:
        出射主光线方向 (L, M, N)，全局坐标系，归一化
    """
    surface = self.surfaces[0]
    
    if surface.surface_type == 'mirror':
        # 入射方向（全局坐标系）
        d = np.array(self.chief_ray_direction)
        
        # 表面法向量（考虑倾斜）
        # 初始法向量沿 -Z（指向入射侧）
        n = np.array([0.0, 0.0, -1.0])
        
        # 应用倾斜（旋转顺序：X → Y）
        if surface.tilt_x != 0:
            c, s = np.cos(surface.tilt_x), np.sin(surface.tilt_x)
            Rx = np.array([
                [1, 0, 0],
                [0, c, -s],
                [0, s, c]
            ])
            n = Rx @ n
        
        if surface.tilt_y != 0:
            c, s = np.cos(surface.tilt_y), np.sin(surface.tilt_y)
            Ry = np.array([
                [c, 0, s],
                [0, 1, 0],
                [-s, 0, c]
            ])
            n = Ry @ n
        
        # 反射公式：r = d - 2(d·n)n
        r = d - 2 * np.dot(d, n) * n
        
        # 归一化
        r = r / np.linalg.norm(r)
        
        return tuple(r)
    else:
        # 折射面：暂时返回入射方向
        return self.chief_ray_direction
```

### 2.3 方向到旋转角度的转换

```python
def _direction_to_rotation_angles(
    self,
    direction: Tuple[float, float, float],
) -> Tuple[float, float]:
    """
    将方向向量转换为旋转角度 (rx, ry)
    
    参数:
        direction: 方向向量 (L, M, N)，必须归一化
    
    返回:
        (rx, ry): 旋转角度（弧度）
    
    说明:
        旋转顺序为 X → Y
        初始方向为 (0, 0, 1)
        旋转后方向为 direction
        
    推导:
        设方向为 (L, M, N)
        初始法向量为 n0 = (0, 0, 1)
        
        绕 X 轴旋转 rx 后：
        n1 = (0, -sin(rx), cos(rx))
        
        绕 Y 轴旋转 ry 后：
        n2 = (sin(ry)*cos(rx), -sin(rx), cos(ry)*cos(rx))
        
        要使 n2 = (L, M, N)：
        sin(ry)*cos(rx) = L
        -sin(rx) = M
        cos(ry)*cos(rx) = N
        
        解得：
        rx = -arcsin(M)
        ry = arctan2(L, N)
    """
    L, M, N = direction
    
    # rx = -arcsin(M)
    M_clamped = np.clip(M, -1.0, 1.0)
    rx = -np.arcsin(M_clamped)
    
    # ry = arctan2(L, N)
    ry = np.arctan2(L, N)
    
    return (rx, ry)
```

### 2.4 创建光学系统（包含倾斜出射面）

```python
def _create_optic(self) -> None:
    """
    创建 optiland 光学系统（包含倾斜的出射面）
    
    系统结构：
    - index=0: 物面（无穷远）
    - index=1: 光学表面（设为光阑）
    - index=2: 倾斜的透明平面（出射面）
    """
    from optiland.optic import Optic
    
    optic = Optic()
    
    # 设置系统参数
    optic.set_aperture(aperture_type='EPD', value=aperture_diameter)
    optic.set_field_type(field_type='angle')
    optic.add_field(y=0, x=0)
    optic.add_wavelength(value=self.wavelength, is_primary=True)
    
    # 添加物面
    optic.add_surface(index=0, radius=np.inf, thickness=np.inf)
    
    # 添加光学表面（包含倾斜）
    surface_def = self.surfaces[0]
    optic.add_surface(
        index=1,
        radius=surface_def.radius,
        thickness=0.0,  # 出射面在同一位置
        material='mirror',
        is_stop=True,
        conic=surface_def.conic,
        rx=surface_def.tilt_x,
        ry=surface_def.tilt_y,
    )
    
    # =========================================================
    # 关键：添加倾斜的透明平面作为出射面
    # =========================================================
    
    # 计算出射主光线方向
    exit_direction = self._compute_exit_chief_direction()
    
    # 从方向计算旋转角度
    exit_rx, exit_ry = self._direction_to_rotation_angles(exit_direction)
    
    # 添加倾斜的透明平面
    optic.add_surface(
        index=2,
        radius=np.inf,      # 平面
        thickness=0.0,
        material='air',     # 透明，光线直接穿过
        rx=exit_rx,
        ry=exit_ry,
    )
    
    # 保存出射面信息
    self.exit_chief_direction = exit_direction
    self.exit_rotation_matrix = compute_rotation_matrix(exit_direction)
    
    self.optic = optic
```

### 2.5 简化的 trace() 方法

```python
def trace(self, input_rays: RealRays) -> RealRays:
    """
    执行光线追迹（简化版）
    
    流程：
    1. 坐标变换：入射面局部 → 全局
    2. optiland 追迹（自动处理元件表面和倾斜出射面）
    3. 坐标变换：全局 → 出射面局部
    """
    # 1. 坐标变换：入射面局部 → 全局
    rays_global = transform_rays_to_global(
        input_rays,
        self.rotation_matrix,
        self.entrance_position,
    )
    
    # 2. 使用 optiland 追迹
    # optiland 会自动：
    # - 追迹到元件表面，计算反射
    # - 追迹到倾斜的出射面，更新位置和 OPD
    surface_group = self.optic.surface_group
    surface_group.trace(rays_global, skip=1)
    
    # 3. 坐标变换：全局 → 出射面局部
    output_rays = transform_rays_to_local(
        rays_global,
        self.exit_rotation_matrix,
        self.entrance_position,  # 出射面位置与入射面相同
    )
    
    self.output_rays = output_rays
    return output_rays
```

## 3. 数据流图

```
入射光线（入射面局部坐标系）
         │
         ▼
┌─────────────────────────────┐
│ transform_rays_to_global()  │
│ 入射面局部 → 全局           │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ surface_group.trace()       │
│                             │
│ optiland 自动完成：         │
│ 1. 追迹到元件表面           │
│ 2. 计算反射/折射            │
│ 3. 追迹到倾斜出射面         │
│ 4. 更新位置和 OPD           │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ transform_rays_to_local()   │
│ 全局 → 出射面局部           │
└─────────────────────────────┘
         │
         ▼
出射光线（出射面局部坐标系）
```

## 4. 与原方案的对比

### 4.1 原方案（分步追迹 + 手动传播）

```
1. optiland 追迹到元件表面（不添加像面）
2. 手动计算出射主光线方向
3. 创建虚拟出射面几何
4. 用 optiland 计算距离
5. 手动传播光线（更新位置和 OPD）  ← 需要手动实现
6. 坐标变换
```

### 4.2 新方案（直接追迹）

```
1. 预先计算出射主光线方向
2. 在 optiland 中添加倾斜的透明平面
3. optiland 直接完成整个追迹  ← 全部由 optiland 处理
4. 坐标变换
```

### 4.3 新方案的优势

1. **更简洁**：不需要 `_propagate_to_exit_plane()` 方法
2. **更可靠**：完全复用 optiland 已验证的传播逻辑
3. **代码更少**：减少约 50 行代码
4. **OPD 计算更准确**：optiland 会正确处理折射率和光程

## 5. 关键接口定义

### 5.1 新增方法

```python
class ElementRaytracer:
    def _compute_exit_chief_direction(self) -> Tuple[float, float, float]: ...
    def _direction_to_rotation_angles(
        self, direction: Tuple[float, float, float]
    ) -> Tuple[float, float]: ...
```

### 5.2 修改的方法

```python
class ElementRaytracer:
    def _create_optic(self) -> None:
        """添加倾斜的透明平面作为出射面"""
        ...
    
    def trace(self, input_rays: RealRays) -> RealRays:
        """简化版，直接使用 optiland 追迹"""
        ...
```

### 5.3 新增属性

```python
class ElementRaytracer:
    exit_chief_direction: Tuple[float, float, float]  # 出射主光线方向
    exit_rotation_matrix: NDArray  # 出射面旋转矩阵
```

## 6. 测试策略

### 6.1 单元测试

1. **方向到旋转角度转换测试**
   - (0, 0, 1) → (0, 0)
   - (0, -1, 0) → (π/2, 0)
   - (1, 0, 0) → (0, π/2)

2. **出射主光线方向计算测试**
   - 正入射平面镜：出射方向 (0, 0, -1)
   - 45° 折叠镜（tilt_x=π/4）：出射方向 (0, -1, 0)

### 6.2 集成测试

1. **45° 折叠镜**
   - 平面波入射
   - 验证输出仍为平面波
   - WFE RMS < 0.01 波长

2. **伽利略式 OAP 扩束镜**
   - 完整系统测试
   - 验证扩束比和 WFE

## 7. 正确性属性

### 7.1 方向余弦归一化
- 出射光线的方向余弦满足 L² + M² + N² = 1

### 7.2 出射面位置
- 出射光线在出射面局部坐标系中 z ≈ 0（数值精度范围内）

### 7.3 OPD 连续性
- 相邻光线的 OPD 差异应该是连续的，不应有突变
