<!------------------------------------------------------------------------------------
# PROPER 库使用规范

inclusion: fileMatch
fileMatchPattern: '**/propagation/**,**/proper/**,**/*propagat*'
------------------------------------------------------------------------------------>

## 核心原则

- `wfarr` 存储复振幅经参考波面处理后的结果
- 高斯光束参数（w0, z_w0, z_Rayleigh）只用于选择传播算法
- 所有 wfo 属性必须根据 pilot beam 参数正确设置

---

## ⚠️ 已验证：PROPER 物理光学传播非常准确

| 测试项目 | 误差 | 结论 |
|----------|------|------|
| StateConverter 写入/读取 | 0.000000 waves | 完美可逆 |
| PROPER 传播后入射面相位 | 0.0001-0.0002 milli-waves | 极其精确 |

**关键结论**：
1. PROPER 自由空间传播本身不是误差来源
2. StateConverter 的写入/读取是完美可逆的
3. 误差发生在表面处理（几何光线追迹）过程中
4. 调试时不要怀疑 PROPER 传播或 StateConverter 的准确性

---

## 单位约定

| 量 | 仿真系统 | PROPER |
|----|----------|--------|
| 长度 | mm | m |
| 波长 | μm | m |
| 相位 | rad | rad |

---

## Pilot Beam 参数

| 参数 | 说明 |
|------|------|
| w0 | 束腰半径（1/e² 强度） |
| z_w0 | 束腰位置 |
| z_R | 瑞利距离 = π×w0²/λ |
| R | 曲率半径 = z×(1 + (z_R/z)²)（严格公式） |
| w | 光斑半径 = w0×√(1 + (z/z_R)²) |

---

## 参考面类型

| reference_surface | 条件 | 处理方式 |
|-------------------|------|----------|
| PLANAR | `|z - z_w0| < z_R` | 无参考波面处理 |
| SPHERI | `|z - z_w0| ≥ z_R` | 减去/加回参考球面相位 |

**SPHERI 参考球面**：
- 曲率半径：`R_ref = z - z_w0`（PROPER 远场近似）
- 相位：`φ_ref = k × r² / (2 × R_ref)`（正号）

---

## 数据流

### 写入 PROPER

```
仿真复振幅
    ↓
判断参考面类型
    ↓
┌─ PLANAR：直接写入
└─ SPHERI：减去参考球面相位后写入
    ↓
prop_shift_center 移到 FFT 坐标系
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
使用 Pilot Beam 解包裹
    ↓
仿真复振幅
```

---

## ⚠️ 关键注意事项

1. wfarr 存储经参考波面处理后的复振幅
2. 高斯光束参数只用于选择传播算法
3. prop_shift_center：写入 wfarr 时必须使用
4. 参考球面相位符号：正号
5. SPHERI 参考面：写入时减去，读取时加回
6. 相位解包裹：读取的相位是包裹相位，需用 Pilot Beam 解包裹
7. 远场传播后采样和振幅会变化

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
