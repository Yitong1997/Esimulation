# 需求文档

## 简介

本文档定义了混合光学追迹验证与修复功能的需求。该功能旨在检查和修复混合光学仿真系统中的关键问题，包括光线追迹方法、相位折叠处理、光线方向传递、振幅计算和能量守恒验证。

混合光学仿真系统结合 PROPER（物理光学传输）和 optiland（几何光线追迹）两个库，在元件处执行波前-光线-波前重建流程。本功能将确保整个流程的正确性和数值稳定性。

**特别重要**：本文档强调 PROPER 参考面与 Pilot Beam 在曲率半径计算公式上的本质区别，这是确保相位解包裹正确性的关键。

## 术语表

- **Hybrid_Raytracer**: 混合光线追迹器，执行波前到光线再到波前的转换流程
- **Phase_Unwrapper**: 相位解包裹器，将折叠相位转换为连续相位
- **Pilot_Beam**: 导引光束，基于 ABCD 法则计算的理想参考波前，使用**严格精确**的高斯光束曲率公式
- **PROPER_Reference_Surface**: PROPER 内部参考面，使用**远场近似**的曲率公式
- **Jacobian_Calculator**: 雅可比矩阵计算器，用于基于能量守恒计算振幅变化
- **Ray_Sampler**: 光线采样器，从波前复振幅采样几何光线
- **Amplitude_Reconstructor**: 振幅重建器，从光线数据重建波前振幅
- **OPD**: 光程差（Optical Path Difference）
- **Thin_Phase_Element**: 薄相位元件，用于将波前相位转换为光线方向
- **Rayleigh_Length**: 瑞利长度 z_R = π × w0² / λ，高斯光束的特征长度

## 需求

### 需求 1：光线采样方法修正

**用户故事：** 作为光学仿真工程师，我希望使用正确的光线采样方法，以便从波前复振幅准确地生成几何光线。

#### 验收标准

1. WHEN 从入射面波前采样光线 THEN Ray_Sampler SHALL 创建平行光作为入射光束
2. WHEN 创建薄相位元件 THEN Ray_Sampler SHALL 使用入射波前的相位分布定义该元件
3. WHEN 平行光通过薄相位元件 THEN Ray_Sampler SHALL 利用相位梯度生成具有正确方向的出射光线
4. WHEN 光线追迹完成 THEN Ray_Sampler SHALL 输出光线位置、方向和相对于主光线的 OPD
5. WHEN 采样光线数量不足 THEN Ray_Sampler SHALL 抛出 InsufficientRaysError 异常

### 需求 2：相位解包裹修复

**用户故事：** 作为光学仿真工程师，我希望正确处理相位折叠问题，以便在光线采样和重建过程中保持相位连续性。

#### 验收标准

1. WHEN 从 PROPER 提取相位 THEN Phase_Unwrapper SHALL 识别折叠相位（范围 [-π, π]）
2. WHEN 计算 Pilot_Beam 参考相位 THEN Phase_Unwrapper SHALL 使用 ABCD 法则生成非折叠的连续相位
3. WHEN 执行相位解包裹 THEN Phase_Unwrapper SHALL 应用公式 T_unwrapped = T_pilot + angle(exp(j*(T - T_pilot)))
4. WHEN 解包裹后的相位与 Pilot_Beam 相位差异超过 π THEN Phase_Unwrapper SHALL 发出警告
5. WHEN 在出射面重采样 THEN Phase_Unwrapper SHALL 确保 OPD 平滑连续，无 2π 跳变

### 需求 3：光线方向传递修复

**用户故事：** 作为光学仿真工程师，我希望在两次追迹过程中正确传递光线方向，以便准确计算 OPD 和振幅。

#### 验收标准

1. WHEN 从入射面采样光线 THEN Hybrid_Raytracer SHALL 保存每条光线的初始方向余弦 (L, M, N)
2. WHEN 追迹到元件表面 THEN Hybrid_Raytracer SHALL 记录光线与表面交点处的方向
3. WHEN 追迹到出射面 THEN Hybrid_Raytracer SHALL 记录最终光线方向
4. WHEN 计算 OPD THEN Hybrid_Raytracer SHALL 使用正确的光线方向计算几何光程
5. WHEN 光线方向发生突变 THEN Hybrid_Raytracer SHALL 检测并报告异常

### 需求 4：雅可比矩阵振幅计算验证

**用户故事：** 作为光学仿真工程师，我希望验证振幅计算的正确性，以便确保能量守恒原理被正确应用。

#### 验收标准

1. WHEN 计算雅可比矩阵 THEN Jacobian_Calculator SHALL 使用输入/输出光线位置的坐标映射
2. WHEN 计算振幅变化 THEN Jacobian_Calculator SHALL 应用公式 A_out/A_in = 1/sqrt(|J|)
3. WHEN 雅可比行列式接近零 THEN Jacobian_Calculator SHALL 使用最小值限制避免除零
4. WHEN 振幅计算完成 THEN Jacobian_Calculator SHALL 归一化振幅以保持相对变化
5. WHEN 无效光线区域 THEN Jacobian_Calculator SHALL 将振幅设为零

### 需求 5：能量守恒验证

**用户故事：** 作为光学仿真工程师，我希望验证能量守恒，以便确保仿真结果的物理正确性。

#### 验收标准

1. WHEN 传播前后 THEN Energy_Validator SHALL 计算总能量（光强积分）
2. WHEN 无损耗元件 THEN Energy_Validator SHALL 验证传播前后总能量相等（相对误差 < 1%）
3. WHEN 有损耗元件 THEN Energy_Validator SHALL 验证能量损失符合预期
4. WHEN 能量不守恒 THEN Energy_Validator SHALL 报告能量变化量和可能原因
5. WHEN 能量变化超过阈值 THEN Energy_Validator SHALL 发出警告

### 需求 6：Pilot Beam 计算验证

**用户故事：** 作为光学仿真工程师，我希望验证 Pilot Beam 在整个仿真流程中的正确性，以便确保相位解包裹的准确性。

#### 验收标准

1. WHEN 初始化 Pilot_Beam THEN Pilot_Beam SHALL 从高斯光源参数正确计算复参数 q
2. WHEN 自由空间传播 THEN Pilot_Beam SHALL 应用 ABCD 矩阵 q_out = q_in + d
3. WHEN 通过薄透镜 THEN Pilot_Beam SHALL 应用 ABCD 矩阵 1/q_out = 1/q_in - 1/f
4. WHEN 反射于球面镜 THEN Pilot_Beam SHALL 应用 ABCD 矩阵 1/q_out = 1/q_in - 2/R
5. WHEN 计算参考相位 THEN Pilot_Beam SHALL 使用公式 φ(r) = k*r²/(2*R)
6. WHEN 主光线处 THEN Pilot_Beam SHALL 确保参考相位为零

### 需求 9：曲率半径公式正确性验证

**用户故事：** 作为光学仿真工程师，我希望确保 PROPER 参考面和 Pilot Beam 使用正确的曲率半径公式，以避免近场传播时的相位误差。

#### 背景说明

PROPER 参考面和 Pilot Beam 使用**本质不同**的曲率半径计算公式：

| 项目 | PROPER 参考面 | Pilot Beam |
|------|---------------|------------|
| **曲率半径公式** | `R_ref = z - z_w0`（远场近似） | `R = z × (1 + (z_R/z)²)`（严格精确） |
| **适用范围** | 仅远场（z >> z_R） | 任意距离（近场、远场均精确） |
| **近场行为** | **使用平面参考面**（reference_surface = "PLANAR"） | 使用严格曲率公式 |
| **近场误差** | 显著偏差（可达 100%） | 无误差 |
| **用途** | FFT 传播时减小相位梯度 | 几何光线追迹时的相位解包裹 |

**重要说明**：PROPER 在瑞利距离内（|z - z_w0| < rayleigh_factor × z_R）自动切换为平面参考面（"PLANAR"），此时参考相位为零。这是 PROPER 的内部优化策略，但 Pilot Beam 始终使用严格公式计算曲率半径。

**绝对禁止混用这两种公式！**

#### 验收标准

1. WHEN 计算 Pilot_Beam 曲率半径 THEN Curvature_Calculator SHALL 使用严格公式 `R = z × (1 + (z_R/z)²)`
2. WHEN 计算 PROPER 参考面曲率半径 THEN Curvature_Calculator SHALL 使用远场近似 `R_ref = z - z_w0`
3. WHEN 在近场（z ≈ z_R）计算 Pilot_Beam 相位 THEN Curvature_Calculator SHALL NOT 使用 PROPER 的远场近似公式
4. WHEN 与 PROPER 库交互 THEN State_Converter SHALL 使用 `wfo.z - wfo.z_w0` 作为参考面曲率半径
5. WHEN 进行几何光线追迹相位解包裹 THEN Phase_Unwrapper SHALL 使用 `PilotBeamParams.curvature_radius_mm`（严格公式）
6. WHEN 代码审查发现混用公式 THEN Code_Reviewer SHALL 标记为严重错误

### 需求 10：代码审查与公式验证

**用户故事：** 作为光学仿真工程师，我希望通过代码审查确保曲率半径公式的正确使用，以防止难以发现的数值错误。

#### 代码审查检查点

以下模式应在代码审查时重点检查：

**错误模式（必须修复）：**
```python
# ❌ 错误：在 Pilot Beam 计算中使用 PROPER 的远场近似
R = wfo.z - wfo.z_w0  # 用于 Pilot Beam 相位计算

# ❌ 错误：在 PROPER 交互中使用严格公式
R_ref = z * (1 + (z_R/z)**2)  # 用于 PROPER 参考面
```

**正确模式：**
```python
# ✅ 正确：使用 PilotBeamParams 的严格曲率
R = pilot_params.curvature_radius_mm  # 已使用严格公式计算

# ✅ 正确：PROPER 交互使用其内部约定
R_ref = wfo.z - wfo.z_w0  # PROPER 远场近似
```

#### 验收标准

1. WHEN 审查 PilotBeamParams 类 THEN Code_Reviewer SHALL 验证 `curvature_radius_mm` 使用严格公式 `R = z × (1 + (z_R/z)²)`
2. WHEN 审查 `from_q_parameter` 方法 THEN Code_Reviewer SHALL 验证曲率半径从 `1/q` 的实部正确提取
3. WHEN 审查 `compute_phase_grid` 方法 THEN Code_Reviewer SHALL 验证使用 `self.curvature_radius_mm` 而非 PROPER 参数
4. WHEN 审查 PROPER 参考面相位计算 THEN Code_Reviewer SHALL 验证使用 `wfo.z - wfo.z_w0`
5. WHEN 审查相位解包裹代码 THEN Code_Reviewer SHALL 验证 Pilot Beam 相位使用严格公式
6. WHEN 发现公式混用 THEN Code_Reviewer SHALL 创建高优先级修复任务

### 需求 7：中间结果测试

**用户故事：** 作为光学仿真工程师，我希望测试每一步的中间结果，以便快速定位和修复问题。

#### 验收标准

1. WHEN 测试光线采样 THEN Test_Suite SHALL 验证采样光线的位置和方向正确性
2. WHEN 测试相位解包裹 THEN Test_Suite SHALL 验证解包裹后相位的连续性
3. WHEN 测试光线追迹 THEN Test_Suite SHALL 验证 OPD 计算的正确性
4. WHEN 测试振幅重建 THEN Test_Suite SHALL 验证雅可比矩阵计算的正确性
5. WHEN 测试能量守恒 THEN Test_Suite SHALL 验证传播前后能量变化在容许范围内
6. WHEN 测试 Pilot_Beam THEN Test_Suite SHALL 验证 ABCD 变换的正确性
7. WHEN 测试最终复振幅 THEN Test_Suite SHALL 验证振幅和相位的整体正确性
8. WHEN 测试曲率半径计算 THEN Test_Suite SHALL 分别验证 PROPER 参考面和 Pilot Beam 使用正确公式

### 需求 8：错误处理与诊断

**用户故事：** 作为光学仿真工程师，我希望获得清晰的错误信息和诊断数据，以便快速定位问题。

#### 验收标准

1. WHEN 发生相位突变 THEN Diagnostic_Reporter SHALL 报告突变位置和幅度
2. WHEN 能量不守恒 THEN Diagnostic_Reporter SHALL 报告能量变化的详细信息
3. WHEN 光线追迹失败 THEN Diagnostic_Reporter SHALL 报告失败光线的位置和原因
4. WHEN 振幅计算异常 THEN Diagnostic_Reporter SHALL 报告雅可比矩阵的异常值
5. WHEN 需要调试 THEN Diagnostic_Reporter SHALL 提供可视化中间结果的方法
6. WHEN 检测到曲率半径公式混用 THEN Diagnostic_Reporter SHALL 发出严重警告并指明位置

### 需求 11：近场传播验证

**用户故事：** 作为光学仿真工程师，我希望验证系统在近场传播（z ≈ z_R）时的正确性，以确保 Pilot Beam 严格公式的必要性。

#### 背景说明

在近场传播时（传播距离 z 接近瑞利长度 z_R），PROPER 的远场近似公式会产生显著误差：

- 当 z = z_R 时，严格公式 `R = z × (1 + (z_R/z)²) = 2z`
- 而远场近似 `R_ref = z - z_w0 ≈ z`（假设 z_w0 ≈ 0）
- 误差达到 100%！

#### 验收标准

1. WHEN 传播距离 z < 2 × z_R THEN Near_Field_Validator SHALL 标记为近场传播
2. WHEN 近场传播时使用 Pilot Beam THEN Near_Field_Validator SHALL 验证使用严格曲率公式
3. WHEN 近场传播时计算残差相位 THEN Near_Field_Validator SHALL 验证残差相位小于 π/4
4. WHEN 近场传播时进行网格重采样 THEN Near_Field_Validator SHALL 验证无 2π 跳变
5. WHEN 比较近场与远场结果 THEN Near_Field_Validator SHALL 验证两种公式在远场趋于一致

### 需求 12：属性基测试（PBT）验证

**用户故事：** 作为光学仿真工程师，我希望通过属性基测试验证曲率半径公式的正确性，以确保在各种参数组合下都能正确工作。

#### 验收标准

1. WHEN 生成任意高斯光束参数 THEN PBT_Validator SHALL 验证 Pilot Beam 曲率半径公式正确
2. WHEN 生成任意传播距离 THEN PBT_Validator SHALL 验证 ABCD 变换后曲率半径正确
3. WHEN 生成近场和远场混合场景 THEN PBT_Validator SHALL 验证两种公式在各自适用范围内正确
4. WHEN 生成边界条件（z → 0, z → ∞）THEN PBT_Validator SHALL 验证公式数值稳定
5. WHEN 生成随机相位分布 THEN PBT_Validator SHALL 验证解包裹后相位连续


---

## 附录 A：代码审查清单

### A.1 PilotBeamParams 类审查

**文件**: `src/hybrid_optical_propagation/data_models.py`

| 检查项 | 预期行为 | 审查状态 |
|--------|----------|----------|
| `from_gaussian_source` 方法 | 曲率半径使用 `R = z * (1 + (z_R / z)**2)` | ☐ |
| `from_q_parameter` 方法 | 曲率半径从 `1/Re(1/q)` 提取 | ☐ |
| `propagate` 方法 | q 参数变换后重新计算曲率半径 | ☐ |
| `apply_lens` 方法 | ABCD 变换后重新计算曲率半径 | ☐ |
| `apply_mirror` 方法 | ABCD 变换后重新计算曲率半径 | ☐ |
| `compute_phase_grid` 方法 | 使用 `self.curvature_radius_mm` | ☐ |

### A.2 PROPER 参考面相位计算审查

**相关文件**: 所有与 PROPER 交互的模块

| 检查项 | 预期行为 | 审查状态 |
|--------|----------|----------|
| 参考面类型判断 | 检查 `wfo.reference_surface` 是 "PLANAR" 还是 "SPHERI" | ☐ |
| 平面参考面处理 | 当 `reference_surface == "PLANAR"` 时，参考相位为零 | ☐ |
| 球面参考面曲率半径 | 使用 `wfo.z - wfo.z_w0` | ☐ |
| 参考面相位公式 | `φ_ref = -k × r² / (2 × R_ref)` | ☐ |
| 不使用严格公式 | 不使用 `z * (1 + (z_R/z)**2)` | ☐ |

### A.3 相位解包裹代码审查

**相关文件**: 相位解包裹相关模块

| 检查项 | 预期行为 | 审查状态 |
|--------|----------|----------|
| Pilot Beam 相位计算 | 使用 `PilotBeamParams.compute_phase_grid` | ☐ |
| 解包裹公式 | `T_unwrapped = T_pilot + angle(exp(j*(T - T_pilot)))` | ☐ |
| 不使用 PROPER 参数 | 不直接使用 `wfo.z`, `wfo.z_w0` 计算 Pilot Beam 相位 | ☐ |

### A.4 禁止模式检查

在代码审查时，搜索以下模式并标记为错误：

```python
# 搜索模式 1：在 Pilot Beam 上下文中使用 wfo.z - wfo.z_w0
# 正则表达式: pilot.*wfo\.z\s*-\s*wfo\.z_w0

# 搜索模式 2：在 PROPER 参考面上下文中使用严格公式
# 正则表达式: proper.*\(1\s*\+\s*\(z_R.*z\)

# 搜索模式 3：直接使用 wfo 参数计算 Pilot Beam 曲率
# 正则表达式: curvature.*=.*wfo\.
```

---

## 附录 B：公式参考

### B.1 高斯光束曲率半径公式

**严格精确公式（Pilot Beam 使用）：**
```
R(z) = z × (1 + (z_R/z)²)

其中：
- z: 相对于束腰的传播距离
- z_R = π × w0² / λ: 瑞利长度
```

**远场近似公式（PROPER 参考面使用）：**
```
当 |z - z_w0| >= rayleigh_factor × z_R 时（远场）：
    R_ref = z - z_w0
    reference_surface = "SPHERI"

当 |z - z_w0| < rayleigh_factor × z_R 时（近场/瑞利距离内）：
    R_ref = ∞（平面参考）
    reference_surface = "PLANAR"

其中：
- z: 当前位置
- z_w0: 束腰位置
- z_R: 瑞利长度
- rayleigh_factor: PROPER 内部参数（默认约为 1.0）
```

### B.2 两种公式的数值比较

| z/z_R | 严格公式 R/z | 远场近似 R_ref/z | 相对误差 |
|-------|--------------|------------------|----------|
| 0.5 | 2.5 | 1.0 | 150% |
| 1.0 | 2.0 | 1.0 | 100% |
| 2.0 | 1.25 | 1.0 | 25% |
| 5.0 | 1.04 | 1.0 | 4% |
| 10.0 | 1.01 | 1.0 | 1% |

**结论**：只有当 z > 10 × z_R 时，远场近似误差才小于 1%。

### B.3 Pilot Beam 相位公式

```
φ_pilot(r) = k × r² / (2 × R)

其中：
- k = 2π/λ: 波数
- r² = x² + y²: 到光轴的距离平方
- R: 使用严格公式计算的曲率半径
```

**注意**：不包含 Gouy 相位，因为 Gouy 相位是空间常数，在计算相对于主光线的相位时会被抵消。
