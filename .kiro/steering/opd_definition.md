<!------------------------------------------------------------------------------------
# OPD（光程差）定义与统一规范

本文件明确定义了混合光学仿真系统中 OPD 的概念、各模块的使用约定，以及正确的转换方法。

inclusion: always
------------------------------------------------------------------------------------>

## ⚠️ 核心原则：数据流向

**绝对禁止直接使用 Pilot Beam 对仿真复振幅或光线相位进行赋值！**

正确的数据流向：
```
仿真复振幅 → 光线参数（通过物理仿真）
光线参数 → 仿真复振幅（通过物理仿真）
```

错误的做法：
```
❌ rays.opd = pilot_beam_opd  # 禁止！
❌ simulation_amplitude = pilot_beam_phase  # 禁止！
```

**Pilot Beam 的唯一用途**：
1. 作为参考相位进行相位解包裹（unwrap）
2. 计算残差 OPD 用于网格重采样
3. 验证仿真结果的正确性

**光线的所有参数必须从仿真复振幅中直接或间接仿真得到，反之亦然。**

---

## 概述

OPD（Optical Path Difference，光程差）在本系统中有多种含义，必须明确区分：

| OPD 类型 | 定义 | 单位 | 使用场景 |
|----------|------|------|----------|
| **绝对光程** | 从物面到当前位置的总光程 | mm | optiland 内部 |
| **相对光程差** | 相对于主光线的光程差 | mm 或 waves | 光线追迹输出 |
| **Pilot Beam OPD** | Pilot Beam 参考相位对应的光程 | mm | 相位解包裹（仅作参考） |
| **残差 OPD** | 实际光程与 Pilot Beam 光程的差值 | mm 或 waves | 网格重采样 |

---

## 核心定义

### 1. 绝对光程（optiland 内部）

**定义**：从物面（object plane）到当前位置的累积光程。

```
绝对光程 = Σ (n_i × d_i)

其中：
- n_i 是第 i 段介质的折射率
- d_i 是第 i 段的几何距离
```

**特点**：
- optiland 的 `rays.opd` 属性存储的就是绝对光程
- 数值通常很大（几十到几百 mm）
- 不能直接用于波前重建

**⚠️ 重要**：`WavefrontToRaysSampler` 输出的 `rays.opd` 是绝对光程，不是相对于 Pilot Beam 的 OPD！

### 2. 相对光程差（相对于主光线）

**定义**：某条光线的光程与主光线（chief ray）光程的差值。

```
相对 OPD = 光线光程 - 主光线光程
```

**特点**：
- 主光线定义为通过光瞳中心（Px=0, Py=0）的光线
- 相对 OPD 反映了波前的形状
- 可以转换为波长数：`OPD_waves = OPD_mm / wavelength_mm`

### 3. Pilot Beam OPD

**定义**：理想高斯光束（Pilot Beam）在某位置的相位对应的光程。

```
Pilot Beam 相位: φ_pilot(r) = k × r² / (2 × R)
Pilot Beam OPD: OPD_pilot = φ_pilot / k = r² / (2 × R)

其中：
- k = 2π/λ 是波数
- r² = x² + y² 是到光轴的距离平方
- R 是 Pilot Beam 的曲率半径（使用严格高斯光束公式）
```

**特点**：
- 主光线处（r=0）OPD 为 0
- 用于相位解包裹和残差计算
- 曲率半径使用严格公式：`R = z × (1 + (z_R/z)²)`

### 4. 残差 OPD

**定义**：实际光程与 Pilot Beam 光程的差值。

```
残差 OPD = 实际 OPD - Pilot Beam OPD
```

**特点**：
- 残差 OPD 应该很小（通常 < 1 波长）
- 小的残差确保网格重采样时不会出现 2π 跳变
- 用于波前重建时的相位插值

---

## 各模块的 OPD 约定

### WavefrontToRaysSampler

**输入**：仿真复振幅（simulation_amplitude）

**输出**：`output_rays.opd` = **绝对光程**（从物面到像面）

**⚠️ 问题**：输出的 OPD 不是相对于 Pilot Beam 的，不能直接用于混合传播！

```python
# WavefrontToRaysSampler 的 OPD 输出
sampler = WavefrontToRaysSampler(...)
output_rays = sampler.get_output_rays()

# output_rays.opd 是绝对光程，数值很大（~80mm）
# 不是相对于 Pilot Beam 的 OPD！
```

### ElementRaytracer

**输入**：`input_rays.opd` = 入射光线的初始 OPD

**处理**：累加传播过程中的光程增量

```python
# 带符号的 OPD 增量
opd_increment = n * t  # n=折射率, t=传播距离（带符号）
rays.opd = rays.opd + opd_increment
```

**输出**：`output_rays.opd` = 入射 OPD + 传播光程增量

**关键**：输出 OPD 的含义取决于输入 OPD 的设置！

### HybridElementPropagator

**正确做法**：在调用 ElementRaytracer 之前，将光线 OPD 设置为 Pilot Beam OPD

```python
# 正确流程
# 1. 从仿真复振幅采样光线
sampler = WavefrontToRaysSampler(...)
output_rays = sampler.get_output_rays()

# 2. 计算 Pilot Beam OPD（不是使用 sampler 的 OPD！）
ray_x, ray_y = sampler.get_ray_positions()
r_sq = ray_x**2 + ray_y**2
R = pilot_beam_params.curvature_radius_mm
pilot_opd_mm = r_sq / (2 * R)  # Pilot Beam OPD

# 3. 设置光线 OPD 为 Pilot Beam OPD
output_rays.opd = pilot_opd_mm

# 4. 执行光线追迹
raytracer = ElementRaytracer(...)
traced_rays = raytracer.trace(output_rays)

# 5. 出射光线 OPD = Pilot Beam OPD + 传播光程增量
#    这就是相对于入射面 Pilot Beam 的总 OPD
```

---

## OPD 转换公式

### 相位 ↔ OPD

```python
# 相位（弧度）转 OPD（mm）
wavelength_mm = wavelength_um * 1e-3
k = 2 * np.pi / wavelength_mm
opd_mm = phase_rad / k

# OPD（mm）转相位（弧度）
phase_rad = k * opd_mm

# OPD（mm）转波长数
opd_waves = opd_mm / wavelength_mm

# 波长数转 OPD（mm）
opd_mm = opd_waves * wavelength_mm
```

### Pilot Beam 相位 ↔ OPD

```python
def compute_pilot_beam_opd_mm(ray_x, ray_y, curvature_radius_mm):
    """计算 Pilot Beam OPD（mm）
    
    参数:
        ray_x, ray_y: 光线位置（mm）
        curvature_radius_mm: Pilot Beam 曲率半径（mm），使用严格公式计算
    
    返回:
        OPD（mm），主光线处为 0
    """
    if np.isinf(curvature_radius_mm):
        return np.zeros_like(ray_x)
    
    r_sq = ray_x**2 + ray_y**2
    return r_sq / (2 * curvature_radius_mm)
```

---

## 正确的混合传播流程

### 入射面处理

```
┌─────────────────────────────────────────────────────────────────┐
│ 入射面                                                           │
│                                                                  │
│ 1. 从仿真复振幅采样光线                                          │
│    sampler = WavefrontToRaysSampler(simulation_amplitude, ...)   │
│    rays = sampler.get_output_rays()                              │
│    ray_x, ray_y = sampler.get_ray_positions()                    │
│                                                                  │
│ 2. 计算 Pilot Beam OPD（不使用 sampler 的 OPD！）                │
│    pilot_opd_mm = compute_pilot_beam_opd_mm(ray_x, ray_y, R)     │
│                                                                  │
│ 3. 设置光线 OPD 为 Pilot Beam OPD                                │
│    rays.opd = pilot_opd_mm                                       │
│                                                                  │
│ 4. 执行光线追迹                                                  │
│    traced_rays = raytracer.trace(rays)                           │
└─────────────────────────────────────────────────────────────────┘
```

### 出射面处理

```
┌─────────────────────────────────────────────────────────────────┐
│ 出射面                                                           │
│                                                                  │
│ 1. 获取出射光线 OPD                                              │
│    output_opd_mm = traced_rays.opd                               │
│    # 这是相对于入射面 Pilot Beam 的总 OPD                        │
│                                                                  │
│ 2. 计算出射面 Pilot Beam OPD                                     │
│    exit_pilot_opd_mm = compute_pilot_beam_opd_mm(                │
│        exit_ray_x, exit_ray_y, exit_R                            │
│    )                                                             │
│                                                                  │
│ 3. 计算残差 OPD                                                  │
│    residual_opd_mm = output_opd_mm - exit_pilot_opd_mm           │
│    # 残差应该很小，可以安全地进行网格重采样                      │
│                                                                  │
│ 4. 重采样残差 OPD 到网格                                         │
│    residual_grid = interpolate(residual_opd_mm, ...)             │
│                                                                  │
│ 5. 重建仿真复振幅                                                │
│    total_opd_grid = exit_pilot_opd_grid + residual_grid          │
│    phase_grid = k * total_opd_grid                               │
│    simulation_amplitude = amplitude * exp(1j * phase_grid)       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 常见错误

### 错误 1：直接使用 WavefrontToRaysSampler 的 OPD

```python
# ❌ 错误
sampler = WavefrontToRaysSampler(...)
rays = sampler.get_output_rays()
# rays.opd 是绝对光程（~80mm），不是 Pilot Beam OPD！
raytracer.trace(rays)  # 出射 OPD 会有 ~80mm 的偏移

# ✅ 正确
sampler = WavefrontToRaysSampler(...)
rays = sampler.get_output_rays()
ray_x, ray_y = sampler.get_ray_positions()
pilot_opd_mm = compute_pilot_beam_opd_mm(ray_x, ray_y, R)
rays.opd = pilot_opd_mm  # 覆盖为 Pilot Beam OPD
raytracer.trace(rays)
```

### 错误 2：混淆绝对光程和相对光程差

```python
# ❌ 错误：将绝对光程当作相对 OPD
opd_waves = rays.opd / wavelength_mm  # 如果 rays.opd 是绝对光程，结果无意义

# ✅ 正确：先计算相对于主光线的 OPD
chief_opd = rays.opd[chief_ray_index]
relative_opd = rays.opd - chief_opd
opd_waves = relative_opd / wavelength_mm
```

### 错误 3：使用 PROPER 远场近似曲率计算 Pilot Beam OPD

```python
# ❌ 错误：使用 PROPER 的远场近似
R = wfo.z - wfo.z_w0  # 远场近似，近场不准确

# ✅ 正确：使用严格高斯光束公式
R = pilot_beam_params.curvature_radius_mm  # 已使用严格公式计算
```

---

## 验证检查点

在调试混合传播时，检查以下关键点：

1. **入射面光线 OPD**：
   - 应该等于 Pilot Beam OPD
   - 范围应该很小（通常 < 1mm）
   - 主光线处应该为 0

2. **出射面光线 OPD**：
   - 相对于出射面 Pilot Beam 的残差应该很小
   - 残差 RMS 应该 < 0.01 waves（对于理想系统）

3. **网格重采样**：
   - 残差 OPD 应该平滑连续
   - 不应有 2π 跳变

```python
# 验证代码示例
def verify_opd_at_entrance(rays, pilot_beam_params, wavelength_um):
    """验证入射面光线 OPD"""
    ray_x, ray_y = np.asarray(rays.x), np.asarray(rays.y)
    ray_opd = np.asarray(rays.opd)
    
    # 计算 Pilot Beam OPD
    R = pilot_beam_params.curvature_radius_mm
    pilot_opd = compute_pilot_beam_opd_mm(ray_x, ray_y, R)
    
    # 计算差异
    diff_mm = ray_opd - pilot_opd
    wavelength_mm = wavelength_um * 1e-3
    diff_waves = diff_mm / wavelength_mm
    
    print(f"入射面 OPD 验证:")
    print(f"  光线 OPD 范围: [{np.min(ray_opd):.6f}, {np.max(ray_opd):.6f}] mm")
    print(f"  Pilot Beam OPD 范围: [{np.min(pilot_opd):.6f}, {np.max(pilot_opd):.6f}] mm")
    print(f"  差异 RMS: {np.std(diff_waves):.6f} waves")
    
    if np.std(diff_waves) > 0.01:
        print("  [WARNING] 入射面 OPD 与 Pilot Beam 不一致！")
    else:
        print("  [OK] 入射面 OPD 正确")
```

---

## 单位约定总结

| 量 | 单位 | 说明 |
|----|------|------|
| 光线位置 (x, y, z) | mm | optiland 标准 |
| 光线 OPD | mm | optiland 标准 |
| 波长 | μm | 系统标准 |
| Pilot Beam 曲率半径 | mm | 系统标准 |
| OPD（波长数） | waves | 相对于波长的无量纲数 |
| 相位 | rad | 弧度 |

---

## ⚠️ 已知问题：optiland 相位面 OPD 计算

### 问题描述

optiland 的 `PhaseInteractionModel` 中 OPD 计算存在**单位不一致问题**：

```python
# optiland/interactions/phase_interaction_model.py 第 122-124 行
k0 = 2 * be.pi / rays.w  # rays.w 是波长，单位 μm
opd_shift = -phase_val / k0  # 计算结果单位是 μm
rays.opd = rays.opd + opd_shift  # 但 rays.opd 单位是 mm！
```

其中：
- `k0 = 2π / wavelength_um`（单位：rad/μm）
- `phase_val` 单位是 rad
- 所以 `opd_shift = -phase_val / k0` 单位是 **μm**
- 但 `rays.opd` 单位是 **mm**！

### 问题影响

1. **OPD 被放大 1000 倍**：optiland 把 μm 当作 mm 加到 OPD 上
2. **符号相反**：optiland 使用负号，但期望的 OPD 应该是正的（正相位 → 正 OPD）

### WavefrontToRaysSampler 的处理方式

`WavefrontToRaysSampler` 采用以下策略处理这个问题：

1. **光线方向**：将相位缩小 1000 倍传给 optiland，使相位梯度正确
   ```python
   corrected_phase = extended_phase / 1000.0
   ```

2. **光线 OPD**：不依赖 optiland 的 OPD 计算，而是直接从输入相位插值计算
   ```python
   # 从输入相位网格插值得到每条光线位置的相位
   phase_at_rays = interpolator(points)
   # 将相位转换为 OPD：OPD_mm = phase_rad * wavelength_mm / (2π)
   opd_mm = phase_at_rays * wavelength_mm / (2 * np.pi)
   rays.opd = opd_mm
   ```

### OPD 符号约定

与 `ElementRaytracer` 保持一致：
- **正相位 → 正 OPD**（相位延迟）
- `OPD_mm = phase_rad * wavelength_mm / (2π)`

### 其他使用 optiland 相位面的代码

以下文件也使用了 optiland 的 `GridPhaseProfile`，需要注意同样的问题：

1. `src/wavefront_to_rays/phase_surface.py` - `create_phase_surface_optic()` 函数
   - 直接接受相位网格作为输入
   - **未处理 1000 倍单位问题**
   - 如果用于光线追迹，需要类似的修正

### 验证数据

```
测试条件：R_curvature = 10000 mm, x = 5 mm
期望 OPD = r² / (2R) = 25 / 20000 = 0.00125 mm
optiland 计算的 opd_shift = -1.25 mm（放大 1000 倍，符号相反）
WavefrontToRaysSampler 修正后的 OPD = 0.00125 mm（正确）
```

---

## ⚠️ 已知问题：相位折叠（Wrapping）

### 问题描述

当从复振幅中提取相位时，`np.angle()` 返回的相位范围是 `[-π, π]`，这会导致相位折叠。

```python
# WavefrontToRaysSampler._extract_phase()
phase = np.angle(self.wavefront_amplitude)  # 范围 [-π, π]
```

### 问题影响

当输入波前的相位超过 `[-π, π]` 范围时：
- 提取的相位会被折叠
- 光线方向计算会出错（相位梯度在折叠边界处不连续）
- OPD 计算会出错（插值得到的相位是折叠后的值）

### 当前状态

**等待用户指导**。用户指示："入射面相位元件的构建不应该有任何提取2pi或减参考量的处理。"

可能的解决方案：
1. 要求调用者确保输入相位在 `[-π, π]` 范围内
2. 在采样器内部进行相位解包裹
3. 直接接受相位数组作为输入（而不是复振幅）
4. 其他方案

---
