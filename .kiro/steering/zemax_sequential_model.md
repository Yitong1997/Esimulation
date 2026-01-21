<!------------------------------------------------------------------------------------
# Zemax 序列模式光路定义规范

本文件定义了 Zemax 序列模式的光路结构，以及如何转换到 optiland 全局坐标系。

inclusion: always
------------------------------------------------------------------------------------>

## 概述

本项目采用 **Zemax 序列模式** 的光路定义方式，结合：
- **optiland**：几何光线追迹、OPD 计算（全局坐标系）
- **PROPER**：物理光学波前传播

---

## 坐标系统

### 全局坐标系（optiland 使用）

采用右手坐标系：Z 轴为初始光轴方向，Y 轴垂直向上，X 轴由右手定则确定。

optiland 中所有表面的位置和姿态都定义在这个固定的全局坐标系中。

### 当前坐标系（Zemax 元件定义坐标系）

**Zemax 使用一个随光路演化的"当前坐标系"来定义元件。**

- **初始状态**：与全局坐标系重合
- **作用**：元件的位置、姿态、曲率方向等参数都相对于当前坐标系定义
- **演化**：由表面厚度和坐标断点控制

### 光轴坐标系（波前采样用）

用于 PROPER 波前采样，始终以主光线方向为 Z 轴。入射面和出射面始终垂直于光轴。

---

## 当前坐标系的演化规则

### 厚度（Thickness）

每个表面的厚度参数使当前坐标系原点沿其 Z 轴平移：

```
新原点 = 旧原点 + thickness × 当前Z轴方向
```

厚度可正可负，负值表示沿当前 Z 轴负方向移动。

### 坐标断点（Coordinate Break）

**坐标断点是一个虚拟表面，用于位移和旋转当前坐标系，从而改变后续元件定义时的坐标原点和坐标轴方向。**

坐标断点本身不产生光学作用，仅改变当前坐标系。

#### 参数定义

| PARM | 参数 | 作用 | 单位 |
|------|------|------|------|
| 1 | Decenter X | 沿当前 X 轴平移坐标原点 | mm |
| 2 | Decenter Y | 沿当前 Y 轴平移坐标原点 | mm |
| 3 | Tilt About X | 绕当前 X 轴旋转坐标系 | 度 |
| 4 | Tilt About Y | 绕当前 Y 轴旋转坐标系 | 度 |
| 5 | Tilt About Z | 绕当前 Z 轴旋转坐标系 | 度 |
| 6 | Order | 变换顺序 | 0 或 1 |

#### 变换顺序

- **Order = 0**（默认）：先平移（Decenter），后旋转（Tilt）
- **Order = 1**：先旋转（Tilt），后平移（Decenter）

#### 旋转顺序

当存在多轴旋转时，按 X → Y → Z 的顺序依次旋转。

#### 坐标断点的厚度

坐标断点也有厚度参数（DISZ），在完成偏心和旋转变换后，再沿新的 Z 轴方向平移。

### ⚠️ 反射镜不会自动改变当前坐标系

**Zemax 中反射镜本身不会改变当前坐标系的方向。** 反射镜只是带有 MIRROR 材料的普通表面。要使当前坐标系跟随反射后的光路方向，必须手动使用坐标断点进行旋转。


---

## Zemax 到 optiland 的坐标转换

### 核心思路

**追踪当前坐标系相对于全局坐标系的累积变换，将每个表面的位置和姿态转换到全局坐标系。**

### 转换算法

```python
# 当前坐标系状态
current_origin = np.array([0, 0, 0])  # 原点在全局坐标系中的位置
current_axes = np.eye(3)              # 轴向量矩阵 [X, Y, Z] 作为列向量

for surface in zemax_surfaces:
    if surface.type == 'COORDBRK':
        # 坐标断点：变换当前坐标系
        dx, dy = surface.decenter_x, surface.decenter_y
        tx, ty, tz = np.radians([surface.tilt_x, surface.tilt_y, surface.tilt_z])
        
        if surface.order == 0:  # 先平移后旋转
            current_origin += current_axes @ [dx, dy, 0]
            R = rotation_matrix_xyz(tx, ty, tz)
            current_axes = current_axes @ R
        else:  # 先旋转后平移
            R = rotation_matrix_xyz(tx, ty, tz)
            current_axes = current_axes @ R
            current_origin += current_axes @ [dx, dy, 0]
        
        # 坐标断点的厚度
        current_origin += surface.thickness * current_axes[:, 2]
        
    else:
        # 光学表面：记录全局坐标后，沿当前 Z 轴移动
        global_pos = current_origin.copy()
        global_rot = current_axes.copy()
        # ... 转换为 optiland 参数
        current_origin += surface.thickness * current_axes[:, 2]
```

---

## 复振幅采样平面

### 入射面（Entrance Plane）

垂直于入射光轴的平面，用于 PROPER 波前采样输入。原点为主光线与元件表面的交点。

### 出射面（Exit Plane）

垂直于出射光轴的平面，用于 PROPER 波前采样输出。原点为主光线与元件表面的交点。

### 元件切平面（Tangent Plane）

与元件表面相切的平面，用于计算反射/折射方向。

---

## 表面参数定义

### ⚠️ 关键概念：面形参数在当前坐标系中定义

**Zemax 中所有表面形状参数（曲率半径、圆锥常数、非球面系数等）都是相对于当前坐标系定义的，而不是全局坐标系。**

这意味着：
1. **曲率半径 R 的符号**：R > 0 表示曲率中心在**当前坐标系**的 +Z 方向
2. **坐标系旋转后**：后续表面的 R 符号仍然相对于**新的当前坐标系**
3. **转换到全局坐标系**：曲率中心位置 = 顶点位置 + R × 当前Z轴方向

**示例：45° 折叠镜后的凹面镜**

```
初始状态：当前坐标系 = 全局坐标系
    Z轴 = (0, 0, 1)

坐标断点：绕 X 轴旋转 45°
    新 Z轴 = (0, -0.707, 0.707)

凹面镜：R = +100 mm（在当前坐标系中，曲率中心在 +Z 方向）
    全局曲率中心 = 顶点 + 100 × (0, -0.707, 0.707)
                 = 顶点 + (0, -70.7, 70.7)
```

**注意**：曲率半径的数值不需要修改，因为 `current_z_axis` 已经包含了坐标系旋转的信息。

### 曲率半径符号约定

| 曲率半径 | 曲率中心位置 | 表面类型 |
|----------|--------------|----------|
| R > 0 | 在表面顶点的 +Z 方向（当前坐标系） | 凹面 |
| R < 0 | 在表面顶点的 -Z 方向（当前坐标系） | 凸面 |
| R = ∞ | 无穷远 | 平面 |

### 圆锥常数

| k 值 | 表面类型 |
|------|----------|
| k = 0 | 球面 |
| k = -1 | 抛物面 |
| k < -1 | 双曲面 |
| -1 < k < 0 | 扁椭球面 |
| k > 0 | 长椭球面 |

---

## 单位约定

| 项目 | Zemax 文件 | 内部计算 | optiland | PROPER |
|------|-----------|----------|----------|--------|
| 长度 | mm | mm | mm | m |
| 角度 | 度 | 弧度 | 弧度 | - |

---

## ZMX 文件格式要点

### 表面定义

```
SURF <index>
  TYPE <type>        # STANDARD, COORDBRK, EVENASPH 等
  CURV <curvature>   # 曲率 = 1/R
  DISZ <thickness>   # 厚度
  GLAS <material>    # 材料（MIRROR 表示反射镜）
  PARM 1-6           # 坐标断点参数（仅 COORDBRK）
```

### 材料

- `GLAS MIRROR`：反射镜
- `GLAS <name>`：玻璃材料
- 无 GLAS：空气
