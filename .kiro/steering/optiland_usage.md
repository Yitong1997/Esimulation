<!------------------------------------------------------------------------------------
# optiland 库使用规范

inclusion: fileMatch
fileMatchPattern: '**/raytracing/**,**/optiland/**,**/*raytrac*,**/*opd*'
------------------------------------------------------------------------------------>

## 调用方式

- `from optiland.optic import Optic` — 注意：从子模块导入
- `from optiland.wavefront import Wavefront`

**不要**：`from optiland import Optic`（会报错）

## 核心概念

- `Optic` 类表示完整的光学系统
- `Wavefront` 模块计算 OPD（单位：波长数）

---

## 🚫🚫🚫 绝对禁止：偏心参数

**永远删除、永远不使用、永远不提及以下参数：**
- `dy` — 表面 Y 方向偏心
- `dx` — 表面 X 方向偏心
- `off_axis_distance` — 离轴距离参数

**这些参数已被永久废弃，不存在于本项目中。**

---

## ✅ 正确做法：绝对坐标定位

optiland 表面必须使用**绝对坐标定位模式**：

```python
optic.add_surface(
    index=1,
    radius=200.0,
    conic=-1.0,
    material='mirror',
    # 绝对坐标定位（必须使用）
    x=0.0,      # 顶点 X 坐标
    y=100.0,    # 顶点 Y 坐标（离轴量直接体现在这里）
    z=0.0,      # 顶点 Z 坐标
    rx=0.0,     # 绕 X 轴旋转（弧度）
    ry=0.0,     # 绕 Y 轴旋转（弧度）
    rz=0.0,     # 绕 Z 轴旋转（弧度）
)
```

**离轴效果**：通过 `x=, y=` 坐标自然实现，无需任何额外参数。

---

## ⚠️ 已知问题

- `GridPhaseProfile` 覆盖整个正方形网格，需要圆形光瞳时必须显式应用掩模
- `GridPhaseProfile` 计算 OPD 时存在 1000 倍放大问题
