# 需求文档：混合传播复振幅重建优化

## 简介

本文档定义了"混合传播复振幅重建优化"功能的需求。该功能旨在改进 `SequentialOpticalSystem` 中 `_apply_element_hybrid()` 方法的复振幅重建流程。

### 当前问题

当前实现存在以下问题：

1. **只处理相位，忽略振幅**：当前实现只计算和应用相位变化（OPD），完全忽略了光强/振幅变化
2. **未复用 optiland 的重采样技术**：optiland 的 PSF 模块有成熟的光线→复振幅重建代码，当前未复用
3. **重采样只处理相位**：应当同时重采样振幅和相位（或直接重采样复振幅）

### 核心改进目标

1. **同时重建振幅和相位**：光学元件会改变光强分布，需要正确处理
2. **使用雅可比矩阵方法计算振幅**：基于能量守恒原理，通过网格变形计算振幅
3. **正确处理理论曲率相位**：在像差 OPD 基础上加回元件的理论聚焦相位

### 核心数据流

```
PROPER 波前 → 采样光线 → ElementRaytracer 光线追迹 → 获取输入/输出位置 + OPD 
→ 雅可比矩阵计算振幅 → RayToWavefrontReconstructor 重建复振幅 → 加回理论曲率相位 → 更新 PROPER 波前
```

## 术语表

- **Complex_Amplitude（复振幅）**: 包含振幅和相位信息的复数场，形式为 A·exp(iφ)
- **OPD（光程差）**: Optical Path Difference，光线相对于参考光线的光程差异，单位为波长数
- **Jacobian（雅可比矩阵）**: 描述输入面到输出面坐标映射的局部线性变换矩阵
- **Aberration_OPD（像差 OPD）**: 实际 OPD 与理想聚焦 OPD 的差值，表示元件引入的波前畸变
- **RayToWavefrontReconstructor**: 新增的复振幅重建器类，封装光线→复振幅重建逻辑

## 需求

### 需求 1：ElementRaytracer 输出完整光线数据

**用户故事：** 作为光学仿真工程师，我希望光线追迹后能获取输入/输出光线位置和 OPD，以便进行雅可比矩阵计算和复振幅重建。

#### 验收标准

1. WHEN ElementRaytracer 完成追迹 THEN 系统 SHALL 输出每条光线的 OPD（波长数）
2. THE ElementRaytracer SHALL 保存输入光线位置，并提供 `get_input_positions()` 方法
3. THE ElementRaytracer SHALL 提供 `get_output_positions()` 方法获取出射光线位置
4. THE ElementRaytracer SHALL 保持现有 `get_relative_opd_waves()` 方法的功能
5. THE ElementRaytracer SHALL 提供 `get_valid_ray_mask()` 方法获取有效光线掩模

### 需求 2：复振幅重建器（雅可比矩阵方法）

**用户故事：** 作为光学仿真工程师，我希望有一个封装好的复振幅重建器，使用雅可比矩阵方法计算振幅。

#### 验收标准

1. THE 系统 SHALL 提供 `RayToWavefrontReconstructor` 类封装重建逻辑
2. WHEN 执行重建 THEN 系统 SHALL 使用复振幅公式：`A × exp(-j × 2π × OPD)`
3. WHEN 计算振幅 THEN 系统 SHALL 使用雅可比矩阵方法，基于能量守恒原理
4. THE 振幅计算公式 SHALL 为：`A_out / A_in = 1 / sqrt(|J|)`，其中 |J| 是雅可比行列式
5. THE RayToWavefrontReconstructor SHALL 支持将稀疏光线数据插值到 PROPER 网格
6. THE RayToWavefrontReconstructor SHALL 正确处理无效光线区域（振幅为 0）
7. THE 雅可比矩阵计算 SHALL 复用已有的光线追迹结果，不需要额外追迹

### 需求 3：理论曲率相位处理

**用户故事：** 作为光学仿真工程师，我希望在像差 OPD 基础上正确加回元件的理论聚焦相位。

#### 验收标准

1. WHEN 计算理论曲率相位 THEN 系统 SHALL 使用公式 `φ_theory = -k·r²/(2f)`
2. WHEN 元件为平面镜（f = ∞） THEN 系统 SHALL 设置理论曲率相位为 0
3. WHEN 元件有折叠倾斜（is_fold=True） THEN 理论曲率相位 SHALL NOT 包含倾斜分量
4. THE 系统 SHALL 在像差复振幅重建后加回理论曲率相位

### 需求 4：PROPER 波前更新

**用户故事：** 作为光学仿真工程师，我希望正确更新 PROPER 波前，使其包含元件引入的完整效应。

#### 验收标准

1. WHEN 复振幅重建完成 THEN 系统 SHALL 将重建的复振幅转换为 PROPER 格式
2. WHEN 更新 PROPER 波前 THEN 系统 SHALL 正确处理 FFT 坐标系（使用 prop_shift_center）
3. THE 系统 SHALL 更新 PROPER 的高斯光束跟踪参数（通过 `_update_gaussian_params_only()`）
4. THE 系统 SHALL 保持与现有接口的兼容性

### 需求 5：坐标系统一致性

**用户故事：** 作为光学仿真工程师，我希望光线坐标和 PROPER 网格坐标正确对应。

#### 验收标准

1. THE 光线采样坐标 SHALL 与 PROPER 网格坐标使用相同的原点和方向
2. WHEN 创建采样光线 THEN 系统 SHALL 使用 PROPER 网格的物理尺寸确定采样范围
3. WHEN 执行插值 THEN 系统 SHALL 正确处理光线坐标到网格索引的映射
4. THE 系统 SHALL 在采样范围外的区域设置振幅为 0

### 需求 6：错误处理、诊断与相位突变检测

**用户故事：** 作为光学仿真工程师，我希望系统能够正确处理边界情况、检测相位突变并提供诊断信息。

#### 验收标准

1. IF 有效光线数量 < 4 THEN 系统 SHALL 抛出异常并提供清晰的错误信息
2. IF 重采样后网格上相邻像素相位差 > π THEN 系统 SHALL 发出 UserWarning 警告
3. THE 相位突变检测 SHALL 在重采样完成后、加回理想相位之前执行
4. IF 雅可比行列式接近零 THEN 系统 SHALL 使用最小值限制避免除零
5. THE 系统 SHALL 提供可选的调试输出（中间结果）

### 需求 7：向后兼容性

**用户故事：** 作为光学仿真工程师，我希望新实现不破坏现有功能。

#### 验收标准

1. THE 系统 SHALL 保持 `_apply_element_hybrid()` 方法的现有接口
2. THE 系统 SHALL 通过现有的单元测试和集成测试
3. WHEN 使用新实现 THEN 系统 SHALL 产生与理论预期一致的结果

## 设计概要

### 关键组件

1. **RayToWavefrontReconstructor**（新增）
   - 封装光线→复振幅重建逻辑
   - 使用雅可比矩阵方法计算振幅
   - 输入：光线数据（输入/输出位置、OPD）
   - 输出：复振幅网格

2. **ElementRaytracer**（修改）
   - 新增 `get_input_positions()` 方法（保存输入光线位置）
   - 新增 `get_output_positions()` 方法（获取输出光线位置）
   - 用于雅可比矩阵计算

3. **_apply_element_hybrid()**（重构）
   - 使用 RayToWavefrontReconstructor 进行复振幅重建
   - 正确处理理论曲率相位
   - 更新 PROPER 波前（振幅 + 相位）

### 振幅计算方法

使用雅可比矩阵方法（网格变形），基于能量守恒原理：

| 组件 | 说明 |
|------|------|
| 振幅公式 | `A_out / A_in = 1 / sqrt(\|J\|)`，其中 \|J\| 是雅可比行列式 |
| 复振幅公式 | `A × exp(-j × 2π × OPD)` |
| 数据来源 | 复用 ElementRaytracer 的输入/输出光线位置 |

## 测试策略

### 单元测试

1. **RayToWavefrontReconstructor 测试**
   - 测试均匀光线输入的重建结果
   - 测试带像差的光线输入
   - 测试边界情况（无效光线、边界光线）

2. **雅可比矩阵振幅测试**
   - 验证雅可比行列式计算的正确性
   - 验证能量守恒（误差 < 5%）
   - 验证数值稳定性

### 集成测试

1. **伽利略 OAP 扩束器测试**
   - 验证光束扩展比
   - 验证波前质量

2. **与纯 PROPER 模式对比**
   - 对于理想元件，两种模式结果应一致
