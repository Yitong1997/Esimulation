<!------------------------------------------------------------------------------------
# 复振幅表示与换算规范

本文件定义了混合光学仿真系统中三种复振幅表示的关系、换算方法和适用场景。

inclusion: always
------------------------------------------------------------------------------------>

## 概述

混合光学仿真系统使用三种复振幅表示：

| 表示方式 | 存储内容 | 适用场景 | 参考波定义 |
|----------|----------|----------|------------|
| **仿真复振幅** | 绝对相位（非折叠） | 系统内部统一表示 | 无（绝对相位） |
| **PROPER 复振幅** | 相对于 PROPER 参考面的残差 | 自由空间衍射传播 | PROPER 内部参考面 |
| **Pilot Beam 参考** | 理想高斯光束相位 | 几何光线追迹时的相位解包裹 | 理想高斯光束 |

---

## 三种表示的定义

### 1. 仿真复振幅（Simulation Amplitude）

**定义**：系统内部使用的统一复振幅表示，存储绝对相位（非折叠）。

```
仿真复振幅 = |A| × exp(i × φ_absolute)

其中：
- |A| 是振幅
- φ_absolute 是绝对相位（弧度），非折叠，范围可超出 [-π, π]
- 主光线处相位定义为 0
```

**特点**：
- 相位是连续的，无 2π 跳变
- 可直接用于几何光线追迹的相位采样
- 是系统内部传递的标准格式

### 2. PROPER 复振幅（PROPER wfarr）

**定义**：PROPER 库内部存储的复振幅，是相对于 PROPER 参考面的残差。

```
PROPER wfarr = |A| × exp(i × φ_residual_proper)

其中：
- φ_residual_proper = φ_absolute - φ_proper_ref
- φ_proper_ref 是 PROPER 参考面相位
```

**PROPER 参考面类型**：

| reference_surface | 条件 | 参考相位公式 |
|-------------------|------|--------------|
| "PLANAR" | 在瑞利距离内 | φ_proper_ref = 0 |
| "SPHERI" | 在瑞利距离外 | φ_proper_ref = -k × r² / (2 × R_ref) |

其中 `R_ref = z - z_w0` 是 PROPER 跟踪的参考球面曲率半径。

**⚠️ 重要：PROPER 使用远场近似曲率**

PROPER 的参考面曲率半径 `R_ref = z - z_w0` 是**远场近似**，不是严格的高斯光束曲率公式：
- 严格公式：`R(z) = z × (1 + (z_R/z)²)`
- PROPER 近似：`R_ref = z - z_w0 ≈ z`（当 z >> z_R 时成立）

这意味着在近场（z ≈ z_R 或 z < z_R）时，PROPER 参考面与真实高斯光束波前存在偏差。

**特点**：
- 对于理想高斯光束，残差相位接近零（仅在远场）
- 近场时残差相位可能较大
- 残差相位通常很小，不易发生 2π 折叠
- 适合 FFT 传播（避免大相位梯度导致的采样问题）

### 3. Pilot Beam 参考相位

**定义**：基于 ABCD 法则独立追踪的理想高斯光束相位，用于几何光线追迹时的相位解包裹。

**Pilot Beam 相位公式**（相对于主光线的相位延迟）：

```
φ_pilot(r) = k × r² / (2 × R)

其中：
- k = 2π/λ 是波数
- r² = x² + y² 是到光轴的距离平方
- R 是曲率半径（从 q 参数计算）：R = z × (1 + (z_R/z)²)
```

**⚠️ 重要：Pilot Beam 使用严格精确的高斯光束曲率公式**

Pilot Beam 的曲率半径使用**严格精确**的高斯光束公式：
```
R(z) = z × (1 + (z_R/z)²)
```

这与 PROPER 的远场近似 `R_ref = z - z_w0` **完全不同**：
- Pilot Beam：在任意传播距离（近场、远场）都精确
- PROPER：仅在远场（z >> z_R）时近似精确

**为什么 Pilot Beam 必须使用严格公式**：
- 几何光线追迹需要精确的参考相位进行解包裹
- 残差相位必须最小化，以确保网格重采样时不出现 2π 误差
- 近场传播时，远场近似会导致显著的残差相位误差

**为什么不包含 Gouy 相位**：
- Gouy 相位 ψ(z) = arctan(z/z_R) 是空间常数（在同一 z 平面上所有点相同）
- Pilot Beam 相位定义为相对于主光线的相位延迟
- 主光线处 (r=0) 和边缘处的 Gouy 相位相同，所以在计算相对相位时会被抵消
- 因此 Pilot Beam 相位只包含球面波前相位

**特点**：
- 以非折叠方式解析计算
- 主光线处相位为 0
- 与 PROPER 参考面相位符号相反
- 高斯光束参数（w0, z_w0, z_R）与 PROPER 理论上同步

---

## ⚠️ 关键区别：PROPER 参考面 vs Pilot Beam

### 曲率半径公式的本质区别

| 项目 | PROPER 参考面 | Pilot Beam |
|------|---------------|------------|
| **曲率半径公式** | `R_ref = z - z_w0`（远场近似） | `R = z × (1 + (z_R/z)²)`（严格精确） |
| **适用范围** | 仅远场（z >> z_R） | 任意距离（近场、远场均精确） |
| **近场误差** | 显著偏差 | 无误差 |
| **用途** | FFT 传播时减小相位梯度 | 几何光线追迹时的相位解包裹 |

**⚠️ 禁止混用**：PROPER 参考面相位计算和 Pilot Beam 相位计算使用不同的曲率半径公式，**绝对不能混用**。



**高斯光束参数同步**：
- w0 和 z_R 参数应保持同步
- 但曲率半径计算公式必须严格区分

---

## 适用场景

### 场景 1：自由空间衍射传播（使用 PROPER）

```
┌─────────────────────────────────────────────────────────────────┐
│ 自由空间传播流程                                                 │
│                                                                  │
│   仿真复振幅 ──→ PROPER 复振幅 ──→ prop_propagate ──→ PROPER 复振幅 ──→ 仿真复振幅
│              写入              传播                提取
│                                                                  │
│   PROPER 的优势：                                                │
│   - 参考面跟踪避免了大相位梯度                                   │
│   - FFT 传播时采样充足                                           │
│   - 残差相位小，不易折叠                                         │
└─────────────────────────────────────────────────────────────────┘
```

### 场景 2：几何光线追迹（使用 Pilot Beam）

```
┌─────────────────────────────────────────────────────────────────┐
│ 几何光线追迹流程                                                 │
│                                                                  │
│   入射面：                                                       │
│   仿真复振幅 ──→ 使用 Pilot Beam 解包裹 ──→ 光线采样             │
│                                                                  │
│   光线追迹：                                                     │
│   光线携带非折叠相位 ──→ 元件追迹 ──→ 出射光线（非折叠相位）     │
│                                                                  │
│   出射面：                                                       │
│   出射光线相位 - Pilot Beam 相位 = 残差 OPD                      │
│   残差 OPD 重采样 ──→ 加回 Pilot Beam 相位 ──→ 仿真复振幅        │
│                                                                  │
│   Pilot Beam 的优势：                                            │
│   - 残差 OPD 小，重采样时不易出现 2π 误差                        │
│   - 解析计算，无折叠                                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 换算公式

### 1. 仿真复振幅 → PROPER 复振幅（写入 PROPER）

```python
def simulation_to_proper(simulation_amplitude, wfo, grid_sampling):
    """将仿真复振幅写入 PROPER 对象"""
    import proper
    
    # 1. 计算 PROPER 参考面相位
    proper_ref_phase = compute_proper_reference_phase(wfo, grid_sampling)
    
    # 2. 计算残差相位
    simulation_phase = np.angle(simulation_amplitude)  # 可能需要先解包裹
    residual_phase = simulation_phase - proper_ref_phase
    
    # 3. 写入 PROPER（移到 FFT 坐标系）
    amplitude = np.abs(simulation_amplitude)
    wfo.wfarr = proper.prop_shift_center(amplitude * np.exp(1j * residual_phase))
```

### 2. PROPER 复振幅 → 仿真复振幅（从 PROPER 提取）

```python
def proper_to_simulation(wfo, grid_sampling):
    """从 PROPER 提取仿真复振幅"""
    import proper
    
    # 1. 提取 PROPER 存储的残差（已移到中心）
    amplitude = proper.prop_get_amplitude(wfo)
    residual_phase = proper.prop_get_phase(wfo)  # 相对于 PROPER 参考面
    
    # 2. 计算 PROPER 参考面相位
    proper_ref_phase = compute_proper_reference_phase(wfo, grid_sampling)
    
    # 3. 重建绝对相位
    absolute_phase = proper_ref_phase + residual_phase
    
    # 4. 组合为仿真复振幅
    simulation_amplitude = amplitude * np.exp(1j * absolute_phase)
    
    return simulation_amplitude
```

### 3. PROPER 参考面相位计算

**⚠️ 注意：此函数使用 PROPER 的远场近似曲率，仅用于与 PROPER 库交互，不可用于 Pilot Beam 计算！**

```python
def compute_proper_reference_phase(wfo, grid_sampling):
    """计算 PROPER 参考面相位
    
    警告：使用远场近似曲率 R_ref = z - z_w0，
    仅用于与 PROPER 库的复振幅转换，不可用于 Pilot Beam！
    """
    if wfo.reference_surface == "PLANAR":
        # 平面参考，相位为零
        return np.zeros((grid_sampling.grid_size, grid_sampling.grid_size))
    
    # 球面参考（远场近似！）
    R_ref_m = wfo.z - wfo.z_w0  # PROPER 远场近似曲率半径（m）
    
    if abs(R_ref_m) < 1e-10:
        return np.zeros((grid_sampling.grid_size, grid_sampling.grid_size))
    
    # 创建坐标网格
    X_mm, Y_mm = grid_sampling.get_coordinate_arrays()
    r_sq_m = (X_mm * 1e-3)**2 + (Y_mm * 1e-3)**2  # m²
    
    k = 2 * np.pi / wfo.lamda  # 波数（1/m）
    
    # PROPER 参考球面相位（注意负号！）
    # 与 prop_qphase 中的公式一致：exp(i*π/(λ*c) * r²) 其中 c = R_ref
    proper_ref_phase = -k * r_sq_m / (2 * R_ref_m)
    
    return proper_ref_phase
```

### 4. Pilot Beam 相位计算

**⚠️ 注意：此函数使用严格精确的高斯光束曲率公式，用于几何光线追迹的相位解包裹！**

```python
def compute_pilot_beam_phase(pilot_params, grid_sampling):
    """计算 Pilot Beam 参考相位（相对于主光线的相位延迟）
    
    使用严格精确的高斯光束曲率公式 R = z × (1 + (z_R/z)²)，
    确保在任意传播距离（近场、远场）都精确。
    
    警告：不可使用 PROPER 的远场近似曲率！
    """
    # pilot_params.curvature_radius_mm 已经使用严格公式计算
    R_pilot_mm = pilot_params.curvature_radius_mm
    
    if np.isinf(R_pilot_mm):
        return np.zeros((grid_sampling.grid_size, grid_sampling.grid_size))
    
    X_mm, Y_mm = grid_sampling.get_coordinate_arrays()
    r_sq_mm = X_mm**2 + Y_mm**2  # mm²
    
    wavelength_mm = pilot_params.wavelength_um * 1e-3
    k = 2 * np.pi / wavelength_mm  # 波数（1/mm）
    
    # Pilot Beam 相位（正号！）
    # 这是相对于主光线的相位延迟，主光线处为 0
    pilot_phase = k * r_sq_mm / (2 * R_pilot_mm)
    
    return pilot_phase
```

**注意**：
- `pilot_params.curvature_radius_mm` 由 `PilotBeamParams` 类使用严格公式 `R = z × (1 + (z_R/z)²)` 计算
- 不包含 Gouy 相位，因为 Gouy 相位是空间常数，在计算相对于主光线的相位时会被抵消
- **绝对不能**使用 PROPER 的 `wfo.z - wfo.z_w0` 作为曲率半径！

### 5. 使用 Pilot Beam 解包裹仿真复振幅

```python
def unwrap_with_pilot_beam(wrapped_phase, pilot_params, grid_sampling):
    """使用 Pilot Beam 参考相位解包裹"""
    # 计算 Pilot Beam 参考相位（非折叠）
    pilot_phase = compute_pilot_beam_phase(pilot_params, grid_sampling)
    
    # 解包裹公式
    phase_diff = wrapped_phase - pilot_phase
    unwrapped_phase = pilot_phase + np.angle(np.exp(1j * phase_diff))
    
    return unwrapped_phase
```

---

## 完整流程示例

### 自由空间传播

```
1. 起始状态：仿真复振幅 A_sim

2. 写入 PROPER：
   - 计算 PROPER 参考相位 φ_proper_ref
   - 残差 = angle(A_sim) - φ_proper_ref
   - wfo.wfarr = shift_center(|A_sim| × exp(i × 残差))

3. PROPER 传播：
   - proper.prop_propagate(wfo, distance)
   - PROPER 内部自动更新参考面参数

4. 从 PROPER 提取：
   - 残差 = prop_get_phase(wfo)
   - φ_proper_ref_new = 计算新的 PROPER 参考相位
   - 绝对相位 = φ_proper_ref_new + 残差
   - A_sim_new = |A| × exp(i × 绝对相位)

5. 同步更新 Pilot Beam 参数（ABCD 法则）
```

### 几何光线追迹

```
1. 入射面：仿真复振幅 A_sim

2. 相位解包裹（如果需要）：
   - 计算 Pilot Beam 相位 φ_pilot
   - 解包裹：φ_unwrapped = φ_pilot + angle(exp(i × (angle(A_sim) - φ_pilot)))

3. 光线采样：
   - 从解包裹后的相位采样光线
   - 光线携带非折叠相位

4. 元件追迹：
   - 使用 ElementRaytracer 追迹
   - 计算出射光线位置和 OPD

5. 出射面重建：
   - 计算出射面 Pilot Beam 相位 φ_pilot_out
   - 残差 OPD = 光线 OPD - 对应位置的 Pilot Beam OPD
   - 残差 OPD 重采样到网格
   - 绝对相位 = φ_pilot_out + 残差相位
   - A_sim_out = |A_out| × exp(i × 绝对相位)

6. 更新 Pilot Beam 参数（ABCD 法则，根据元件类型）
```

---

## prop_shift_center 使用规范

**PROPER 的 wfarr 存储在 FFT 坐标系中（零频在角落）**

```python
# 读取时（prop_get_* 函数内部已处理）：
amplitude = proper.prop_get_amplitude(wfo)  # 已移到中心
phase = proper.prop_get_phase(wfo)          # 已移到中心

# 写入时（必须手动处理）：
wfo.wfarr = proper.prop_shift_center(field)  # 移到 FFT 坐标系
```

---

## 单位约定

| 量 | PROPER 内部 | 仿真系统 | Pilot Beam |
|----|-------------|----------|------------|
| 长度 | m | mm | mm |
| 波长 | m (wfo.lamda) | μm | μm |
| 相位 | rad | rad | rad |
| 曲率半径 | m (z - z_w0)（远场近似） | mm | mm（严格公式） |

**⚠️ 曲率半径单位相同，但计算公式不同！**

---


---

## ⚠️ 总结：曲率半径公式使用规范

### 绝对禁止混用

| 场景 | 正确做法 | 错误做法 |
|------|----------|----------|
| **与 PROPER 库交互** | 使用 `R = z - z_w0`（远场近似） | 使用严格公式 |
| **几何光线追迹相位解包裹** | 使用 `R = z × (1 + (z_R/z)²)`（严格公式） | 使用远场近似 |
| **Pilot Beam 相位计算** | 使用 `PilotBeamParams.curvature_radius_mm` | 使用 `wfo.z - wfo.z_w0` |

### 为什么 Pilot Beam 必须使用严格公式

1. **残差相位最小化**：严格公式确保参考相位与真实高斯光束波前精确匹配
2. **近场精度**：在 z ≈ z_R 时，远场近似误差可达 100%
3. **网格重采样安全**：残差相位小，不会在重采样时产生 2π 跳变
4. **普适性**：无论近场还是远场，都能正确工作

### 代码审查检查点

在代码审查时，检查以下模式：

```python
# ❌ 错误：在 Pilot Beam 计算中使用 PROPER 的远场近似
R = wfo.z - wfo.z_w0  # 用于 Pilot Beam 相位计算

# ✅ 正确：使用 PilotBeamParams 的严格曲率
R = pilot_params.curvature_radius_mm  # 已使用严格公式计算

# ❌ 错误：在 PROPER 交互中使用严格公式
R_ref = z * (1 + (z_R/z)**2)  # 用于 PROPER 参考面

# ✅ 正确：PROPER 交互使用其内部约定
R_ref = wfo.z - wfo.z_w0  # PROPER 远场近似
```

