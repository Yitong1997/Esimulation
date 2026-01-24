<!------------------------------------------------------------------------------------
# 复振幅表示与换算规范

本文件定义了混合光学仿真系统中复振幅表示的关系和数据流。

inclusion: always
------------------------------------------------------------------------------------>

## 概述

| 表示方式 | 存储内容 | 适用场景 |
|----------|----------|----------|
| **仿真复振幅** | 绝对相位（非折叠） | 系统内部统一表示 |
| **PROPER 复振幅** | 复振幅经参考波面处理后的结果 | 自由空间衍射传播 |

---

## 仿真复振幅

- 相位连续，无 2π 跳变
- 主光线处相位定义为 0
- 可直接用于几何光线追迹的相位采样

---

## PROPER 复振幅

**⚠️ 核心原则**：
- `wfarr` 存储**复振幅经参考波面处理后的结果**
- 高斯光束参数（w0, z_w0, z_Rayleigh）**只用于选择传播算法**

**参考面类型**：

| reference_surface | 条件 | 处理方式 |
|-------------------|------|----------|
| PLANAR | 近场（`|z - z_w0| < z_R`） | 无参考波面处理 |
| SPHERI | 远场（`|z - z_w0| ≥ z_R`） | 减去参考球面相位 |

**SPHERI 参考面**：
- 曲率半径：`R_ref = z - z_w0`（PROPER 远场近似）
- 相位：`φ_ref = k × r² / (2 × R_ref)`（正号）
- wfarr = 仿真复振幅 × exp(-i × φ_ref)

---

## Pilot Beam 参考相位

用于几何光线追迹时的相位解包裹。

- 相位公式：`φ_pilot(r) = k × r² / (2 × R)`
- 曲率半径使用**严格公式**：`R = z × (1 + (z_R/z)²)`
- Gouy 相位在计算相对相位时被抵消（同一平面上为常数）

---

## ⚠️ 关键区别：PROPER 参考面 vs Pilot Beam

| 项目 | PROPER 参考面 | Pilot Beam |
|------|---------------|------------|
| **曲率半径公式** | `R_ref = z - z_w0`（远场近似） | `R = z × (1 + (z_R/z)²)`（严格精确） |
| **适用范围** | 仅远场（z >> z_R） | 任意距离 |
| **用途** | PROPER 内部参考面计算 | 几何光线追迹相位解包裹 |

**⚠️ 禁止混用两种曲率半径公式。**

---

## 数据流

### 写入 PROPER

```
仿真复振幅
    ↓
判断参考面类型（根据 z 与 z_R 的关系）
    ↓
┌─ PLANAR：直接写入
└─ SPHERI：减去参考球面相位后写入
    ↓
使用 prop_shift_center 移到 FFT 坐标系
    ↓
写入 wfo.wfarr
```

### 从 PROPER 读取

```
wfo.wfarr
    ↓
prop_get_amplitude / prop_get_phase（自动移回中心）
    ↓
┌─ PLANAR：直接使用
└─ SPHERI：加回参考球面相位
    ↓
得到完整相位（包裹的）
    ↓
使用 Pilot Beam 解包裹
    ↓
仿真复振幅
```

### 几何光线追迹

```
入射面：仿真复振幅 → Pilot Beam 解包裹 → 光线采样

出射面：
出射光线相位 - Pilot Beam 相位 = 残差相位
（注意：相位用减法，但 OPD 用加法，因为符号约定不同）
    ↓
残差 OPD 重采样到网格
    ↓
加回 Pilot Beam 相位 → 仿真复振幅
```

---

## 单位约定

| 量 | PROPER 内部 | 仿真系统 |
|----|-------------|----------|
| 长度 | m | mm |
| 波长 | m | μm |
| 相位 | rad | rad |

---

## ⚠️⚠️⚠️ 强制规定：prop_begin 参数

### beam_diameter 必须等于 2×w0

**`prop_begin` 的 `beam_diameter` 参数必须永远等于 2 倍束腰（2×w0）！**

### beam_diam_fraction 必须等于 0.5

**`prop_begin` 的 `beam_diam_fraction` 参数必须永远等于 0.5！**

### 网格物理尺寸固定为 4×w0

当使用上述参数时，PROPER 的网格物理尺寸自动计算为：
```
dx = beam_diameter / (grid_n × beam_diam_fraction)
   = (2 × w0) / (grid_n × 0.5)
   = 4 × w0 / grid_n

physical_size = dx × grid_n = 4 × w0
```

**因此 `physical_size_mm` 参数必须等于 `4 × w0`，不可自定义！**

这是 PROPER 库的固定用法，不可更改：

```python
# ✅ 正确用法
beam_diameter = 2 * w0  # beam_diameter = 2 × 束腰半径
physical_size = 4 * w0  # 网格物理尺寸 = 4 × 束腰半径
wfo = proper.prop_begin(beam_diameter, wavelength, grid_size, 0.5)

# 🚫 错误用法
# beam_diameter = w0  # 错误！
# beam_diameter = 4 * w0  # 错误！
# beam_diameter = 2 * w_init  # 错误！（w_init 是某位置的光斑，不是束腰）
# beam_diameter = physical_size  # 错误！（网格物理尺寸不是光束直径）
# beam_diam_fraction = 0.25  # 错误！
# beam_diam_fraction = beam_diameter / grid_width  # 错误！
# physical_size_mm = 8 * w0  # 错误！（必须是 4 × w0）
```

**原因**：
- PROPER 内部使用 `beam_diameter` 来定义高斯光束的 1/e² 强度直径，即 2×w0
- `beam_diam_fraction = 0.5` 确保网格采样与光束参数正确匹配
- 网格物理尺寸由 PROPER 内部计算，不可自定义

---

## ⚠️ 关键注意事项

1. wfarr 存储经参考波面处理后的复振幅
2. 高斯光束参数只用于选择传播算法
3. prop_shift_center：写入 wfarr 时必须使用
4. 参考球面相位符号：正号
5. SPHERI 参考面：写入时减去，读取时加回
6. 相位解包裹：读取的相位是包裹相位，需用 Pilot Beam 解包裹
7. 远场传播后采样和振幅会变化
8. **beam_diameter = 2 × w0**（PROPER 固定用法）
9. **beam_diam_fraction = 0.5**（PROPER 固定用法）

---

## 验证结论

1. **PROPER 自由空间传播本身不是误差来源**（精度 < 0.001 milli-waves）
2. **StateConverter 的写入/读取是完美可逆的**
3. **误差发生在表面处理（几何光线追迹）过程中**
