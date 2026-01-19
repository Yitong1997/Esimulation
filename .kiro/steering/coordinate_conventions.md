<!------------------------------------------------------------------------------------
# 坐标系与方向约定

本文件定义了序列光学系统中的坐标系统、光轴方向跟踪和倾斜定义规则。
inclusion: always
------------------------------------------------------------------------------------>

## 核心设计原则

### 1. 光轴是动态的

在序列光学系统中，光轴（主光线方向）会随着光学元件而改变：
- 反射元件会改变光轴方向
- 折射元件通常不改变光轴方向（对于轴上光线）
- 所有面的倾斜都相对于**当前光轴**定义

### 2. 采样面垂直于光轴

采样面默认垂直于当前光轴（主光线方向）：
- 采样面的法向量与当前光轴方向一致
- 这确保了波前采样的物理意义正确

### 3. 倾斜定义相对于入射光轴

元件的倾斜角度（`tilt_x`, `tilt_y`）相对于入射光轴定义：
- `tilt_x`: 绕局部 X 轴旋转（俯仰）
- `tilt_y`: 绕局部 Y 轴旋转（偏航）
- 正方向遵循右手定则

## 坐标系定义

### 全局坐标系

初始全局坐标系（光源处）：
```
        Y (向上)
        |
        |
        |________ Z (初始光轴方向)
       /
      /
     X (指向屏幕内)
```

### 局部坐标系

每个元件和采样面都有自己的局部坐标系：
- **原点**：元件顶点或采样面中心
- **Z 轴**：沿当前光轴方向（入射方向）
- **X 轴**：水平方向（与全局 Y 轴和局部 Z 轴都垂直）
- **Y 轴**：由右手定则确定

## 光轴方向跟踪

### 反射定律

对于反射元件，出射光轴方向由反射定律计算：
```
r = d - 2(d·n)n
```
其中：
- `d`: 入射方向（单位向量）
- `n`: 表面法向量（指向入射侧）
- `r`: 反射方向（单位向量）

### 表面法向量

表面法向量由元件倾斜决定：
1. 初始法向量沿入射光轴的反方向（指向入射侧）
2. 应用 `tilt_x` 旋转（绕局部 X 轴）
3. 应用 `tilt_y` 旋转（绕局部 Y 轴）

### 45° 折叠镜示例

```python
# 入射光沿 +Z 方向
incident = (0, 0, 1)

# 45° 折叠镜：tilt_x = π/4
# 表面法向量初始为 (0, 0, -1)
# 绕 X 轴旋转 45° 后：(0, -sin(45°), -cos(45°)) = (0, -0.707, -0.707)

# 反射后光线方向：
# r = d - 2(d·n)n
# d·n = (0, 0, 1)·(0, -0.707, -0.707) = -0.707
# r = (0, 0, 1) - 2*(-0.707)*(0, -0.707, -0.707)
# r = (0, 0, 1) + (0, -1, -1)
# r = (0, -1, 0)  # 沿 -Y 方向
```

## 倾斜类型

### 折叠倾斜 (is_fold=True)

用于折叠光路，改变光束传播方向：
- 不引入波前倾斜（OPD 倾斜）
- PROPER 在"展开"的光路上传播
- 仅影响光轴方向计算

### 失调倾斜 (is_fold=False)

表示元件失调或故意引入的倾斜：
- 引入波前倾斜
- OPD 变化：`ΔW = x·sin(tilt_y) + y·sin(tilt_x)`
- 对于反射镜，OPD 加倍

## 实现要点

### 1. 光轴状态跟踪

使用 `OpticalAxisTracker` 类跟踪光轴演变：
```python
tracker = OpticalAxisTracker()
for element in elements:
    state_before, state_after = tracker.add_element(element)
    # state_before: 元件前的光轴状态
    # state_after: 元件后的光轴状态（反射后）
```

### 2. 采样面定位

采样面应使用当前光轴状态定位：
```python
state = tracker.get_state_at_distance(sampling_distance)
# state.position: 采样面中心位置
# state.direction: 采样面法向量（光轴方向）
```

### 3. 几何光线追迹

在 `ElementRaytracer` 中传递倾斜参数：
```python
# 需要将 tilt_x, tilt_y 传递给 optiland 表面
optic.add_surface(
    index=surface_index,
    radius=radius,
    thickness=thickness,
    material=material,
    rx=tilt_x,  # 绕 X 轴旋转
    ry=tilt_y,  # 绕 Y 轴旋转
)
```

## 2D 光路布置图

绘制 2D 光路图时：
1. 使用 `OpticalAxisTracker` 计算光束路径
2. 在 YZ 平面投影（默认）
3. 标注元件位置和方向
4. 显示采样面位置

```python
# 获取光束路径
z_coords, y_coords = tracker.calculate_beam_path_2d(projection="yz")

# 获取元件位置
for element, position, dir_before, dir_after in tracker.get_element_global_positions():
    # 绘制元件
    pass
```
