<!------------------------------------------------------------------------------------
# PROPER 库使用规范

inclusion: fileMatch
fileMatchPattern: '**/propagation/**,**/proper/**,**/*propagat*'
------------------------------------------------------------------------------------>

## 概述

**核心原则**：
- PROPER 的 `wfarr` 存储复振幅经参考波面处理后的结果
- 高斯光束参数（w0, z_w0, z_Rayleigh）只用于选择传播算法
- 所有 wfo 属性必须根据 pilot beam 参数正确设置

---

## ⚠️ 已验证：PROPER 物理光学传播非常准确

**经过验证，PROPER 的自由空间衍射传播精度极高：**

- 这证明 PROPER 传播本身不是误差来源
- **调试时不要怀疑 PROPER 传播的准确性**

**误差来源在几何光线追迹流程中**（入射面 → 出射面），不在 PROPER 传播中。

验证脚本：`scripts/verify_proper_pilot_beam_consistency.py`

---

## 单位约定

| 量 | 仿真系统 | PROPER |
|----|----------|--------|
| 长度 | mm | m |
| 波长 | μm | m |
| 采样 | mm | m |
| 相位 | rad | rad |

---

## Pilot Beam 参数

| 参数 | 符号 | 说明 |
|------|------|------|
| 束腰半径 | w0 | 1/e² 强度半径 |
| 束腰位置 | z_w0 | 束腰在光轴上的位置 |
| 当前位置 | z | 当前采样面的位置 |
| 瑞利距离 | z_R | z_R = π×w0²/λ |
| 曲率半径 | R | R = z×(1 + (z_R/z)²)（严格公式） |
| 光斑半径 | w | w = w0×√(1 + (z/z_R)²) |

---

## 参考面类型

| reference_surface | 条件 | 处理方式 |
|-------------------|------|----------|
| PLANAR | `|z - z_w0| < z_R` | 无参考波面处理 |
| SPHERI | `|z - z_w0| ≥ z_R` | 减去/加回参考球面相位 |

**SPHERI 参考球面**：
- 曲率半径：`R_ref = z - z_w0`（PROPER 远场近似）
- 相位：`φ_ref = k × r² / (2 × R_ref)`（正号！）

---

## 写入 PROPER 的数据流

1. **初始化 wfo**
   - beam_diameter = 2 × w0
   - 设置所有属性：z, z_w0, w0, z_Rayleigh, dx
   - 确定 reference_surface 类型

2. **处理参考波面**
   - PLANAR：直接使用仿真复振幅
   - SPHERI：仿真复振幅 × exp(-i × φ_ref)

3. **写入 wfarr**
   - 必须使用 prop_shift_center 移到 FFT 坐标系


---

## 从 PROPER 读取的数据流

1. **读取振幅和相位**
   - prop_get_amplitude / prop_get_phase 自动移回中心
   - 读取的相位是包裹相位 [-π, π]

2. **处理参考波面**
   - PLANAR：直接使用
   - SPHERI：完整相位 = PROPER相位 + φ_ref

3. **相位解包裹**
   - 使用 Pilot Beam 参考相位解包裹
   - 解包裹公式：unwrapped = pilot_phase + angle(exp(i × (wrapped - pilot_phase)))

4. **组合仿真复振幅**
   - 振幅 × exp(i × 解包裹后的相位)

---

## 网格和采样

**采样变化**：
- 远场传播时采样会变化
- 新采样 = λ × |dz| / (grid_size × 旧采样)

**网格尺寸**：
- beam_diam_fraction = beam_diameter / grid_width
- 应在合理范围内（0.1 ~ 0.9）

---

## 能量/振幅处理

**远场振幅缩放**：
- 远场传播后振幅有缩放因子（FFT 归一化导致）
- 如需保持能量守恒，需要归一化

**能量守恒验证**：
- 传播前后总能量（∑|A|² × dx²）应保持不变

---

## ⚠️ 关键注意事项

1. **wfarr 存储经参考波面处理后的复振幅**
2. **高斯光束参数只用于选择传播算法**
3. **prop_shift_center**：写入 wfarr 时必须使用
4. **参考球面相位符号**：正号 `+k×r²/(2×R_ref)`
5. **SPHERI 参考面**：写入时减去，读取时加回
6. **相位解包裹**：读取的相位是包裹相位，需要用 Pilot Beam 解包裹
7. **采样变化**：远场传播后采样会变化
8. **振幅缩放**：远场传播后振幅有缩放因子

---

## 常用函数

| 函数 | 用途 |
|------|------|
| prop_begin | 初始化 wfo |
| prop_propagate | 传播 |
| prop_get_amplitude | 读取振幅（已移到中心） |
| prop_get_phase | 读取相位（已移到中心） |
| prop_get_sampling | 读取采样 |
| prop_shift_center | 移到 FFT 坐标系（写入时必须使用） |
| prop_multiply | 乘法 |
| prop_add_phase | 添加相位（单位：米） |
