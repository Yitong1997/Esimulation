# 需求文档：混合传播模式光线追迹 OPD 重构

## 概述

重构混合光学传播模式，使其完全使用真实的几何光线追迹计算 OPD，而不依赖 PROPER 的 `prop_lens` 函数。

## 背景

当前的混合传播模式（`_apply_element_hybrid` 方法）存在以下问题：

1. **近轴近似与精确追迹的不一致**：使用 `prop_lens` 处理理想聚焦效果，然后用光线追迹计算"像差"。但 `prop_lens` 使用近轴近似，而光线追迹是精确的，两者之间存在约 1% 的差异，这个差异被错误地当作"像差"。

2. **参考面机制不匹配**：光线追迹计算的 OPD 是绝对的几何光程差，而 PROPER 使用的是相对于参考球面的相位偏差。当前实现没有正确处理这两者之间的转换。

3. **高斯光束参数更新问题**：当不使用 `prop_lens` 时，需要手动更新 PROPER 的高斯光束跟踪参数（z_w0, w0, z_Rayleigh 等）。

## 术语表

- **OPD (Optical Path Difference)**：光程差，光线实际光程与参考光程的差值
- **PROPER**：物理光学传播库，使用 FFT 进行波前传播
- **prop_lens**：PROPER 中的理想透镜函数，使用近轴近似
- **参考球面**：PROPER 内部跟踪的参考波前，用于减少相位存储需求
- **高斯光束参数**：描述高斯光束的参数，包括束腰位置 z_w0、束腰半径 w0、瑞利距离 z_Rayleigh
- **ElementRaytracer**：元件光线追迹器，使用 optiland 进行几何光线追迹

## 用户故事

### US-1: 完全光线追迹 OPD 计算

**作为**光学仿真用户，**我希望**混合传播模式完全使用几何光线追迹计算元件引入的 OPD，**以便**获得精确的波前变换结果，不受近轴近似的限制。

#### 验收标准

1. WHEN 混合传播模式处理光学元件 THEN 系统 SHALL 使用 ElementRaytracer 计算完整的 OPD，而不是仅计算"像差"
2. WHEN 计算元件 OPD THEN 系统 SHALL NOT 使用 PROPER 的 prop_lens 函数
3. WHEN 光线追迹返回 OPD THEN 系统 SHALL 正确处理 OPD 符号（ElementRaytracer 的 OPD 符号与 PROPER 相反）
4. WHEN 处理非理想元件（如带像差的球面镜）THEN 系统 SHALL 正确计算包含像差的 OPD

### US-2: 参考面变换

**作为**光学仿真用户，**我希望**系统能够正确处理光线追迹 OPD 与 PROPER 参考面之间的转换，**以便**波前相位正确反映实际光学效果。

#### 验收标准

1. WHEN 光线追迹计算绝对 OPD THEN 系统 SHALL 将其转换为相对于 PROPER 参考面的相位偏差
2. WHEN PROPER 参考面为球面（SPHERI）THEN 系统 SHALL 减去参考球面的相位贡献
3. WHEN PROPER 参考面为平面（PLANAR）THEN 系统 SHALL 直接使用光线追迹的 OPD
4. WHEN 计算参考球面相位 THEN 系统 SHALL 使用 PROPER 的 z_w0 参数确定参考球面曲率

### US-3: 高斯光束参数更新

**作为**光学仿真用户，**我希望**系统能够正确更新 PROPER 的高斯光束跟踪参数，**以便**后续传播和采样正确工作。

#### 验收标准

1. WHEN 光学元件改变光束聚焦 THEN 系统 SHALL 更新 PROPER 的虚拟束腰位置 z_w0
2. WHEN 光学元件改变光束聚焦 THEN 系统 SHALL 更新 PROPER 的束腰半径 w0
3. WHEN 光学元件改变光束聚焦 THEN 系统 SHALL 更新 PROPER 的瑞利距离 z_Rayleigh
4. WHEN 更新高斯光束参数 THEN 系统 SHALL 正确设置 beam_type_old 和 reference_surface 属性
5. WHEN 更新高斯光束参数 THEN 系统 MAY 使用 prop_lens 的参数更新逻辑（但不使用其相位计算）

### US-4: 独立验证 ElementRaytracer OPD 计算

**作为**光学仿真用户，**我希望**能够独立验证 ElementRaytracer 计算 OPD 的正确性，**以便**确保光线追迹模块本身的准确性。

#### 验收标准

1. WHEN 平行光入射凹面镜 THEN ElementRaytracer 计算的 OPD SHALL 与解析公式一致（相对误差 < 0.1%）
2. WHEN 平行光入射抛物面镜 THEN ElementRaytracer 计算的 OPD SHALL 为常数（RMS < 0.01 波）
3. WHEN 平行光入射球面镜 THEN ElementRaytracer 计算的 OPD SHALL 包含正确的球差
4. WHEN 平行光入射平面镜 THEN ElementRaytracer 计算的 OPD SHALL 为常数（RMS < 0.001 波）
5. WHEN 光线入射 45° 折叠镜 THEN ElementRaytracer 计算的 OPD SHALL 正确处理坐标变换

### US-5: 集成验证 - 简单光路

**作为**光学仿真用户，**我希望**混合传播模式在简单光路中能够正确工作，**以便**验证与 PROPER 集成的正确性。

#### 验收标准

1. WHEN 高斯光束通过单个凹面镜 THEN 混合模式与 ABCD 理论的光束半径误差 SHALL 小于 1%
2. WHEN 高斯光束通过单个抛物面镜 THEN 波前误差（WFE）RMS SHALL 小于 0.1 波
3. WHEN 高斯光束通过单个平面镜 THEN 波前误差（WFE）RMS SHALL 小于 0.01 波
4. WHEN 高斯光束通过 45° 折叠镜 THEN 输出光束方向 SHALL 正确改变，WFE RMS < 0.01 波

### US-6: 集成验证 - 复杂光路

**作为**光学仿真用户，**我希望**混合传播模式在复杂光路中能够正确工作，**以便**验证多元件系统的正确性。

#### 验收标准

1. WHEN 使用 galilean_oap_expander.py 示例验证 THEN 混合模式与 ABCD 理论的光束半径误差 SHALL 小于 1%
2. WHEN 使用伽利略式扩束镜 THEN 放大倍率 SHALL 与设计值一致（误差 < 1%）
3. WHEN 使用多元件折叠光路 THEN 各采样面的 WFE RMS SHALL 小于 0.1 波
4. WHEN 使用包含多个 OAP 的系统 THEN 输出光束参数 SHALL 与 ABCD 理论一致

### US-7: 相位采样检查

**作为**光学仿真用户，**我希望**系统能够检测并警告相位采样不足的情况，**以便**避免相位包裹导致的错误。

#### 验收标准

1. WHEN 相邻像素间相位差超过 π THEN 系统 SHALL 发出警告
2. WHEN 检测到相位采样不足 THEN 系统 SHALL 建议增加网格大小或减小光束尺寸
3. WHEN 相位梯度过大 THEN 系统 SHALL 记录警告但继续执行

## 非功能性需求

### NFR-1: 性能

- 混合传播模式的单元件处理时间应 < 100ms（512×512 网格，100 条光线）
- 光线追迹和插值的总时间应 < 50ms

### NFR-2: 兼容性

- 应保持与现有 SequentialOpticalSystem 接口的向后兼容
- 应保持与现有 ElementRaytracer 接口的兼容
- 应复用现有的 _update_proper_gaussian_params 方法

### NFR-3: 可测试性

- 每个核心方法应有独立的单元测试
- 应有属性测试验证数学正确性
- 应有集成测试验证完整系统行为

### NFR-4: 代码质量

- 应遵循项目现有的代码风格和文档规范
- 应使用中文注释和文档字符串
- 应使用类型注解

## 约束条件

1. **ElementRaytracer OPD 符号**：ElementRaytracer 的 OPD 符号与 PROPER 相反，需要取反
2. **大 OPD 值**：对于 f=-50mm 的抛物面镜，r=10mm 处的 OPD 约为 1580 波，需要正确处理
3. **PROPER 参考面机制**：需要正确理解和处理 PROPER 的 PLANAR/SPHERI 参考面
4. **高斯光束跟踪**：需要正确更新 PROPER 的高斯光束跟踪参数以确保后续传播正确

## 不在范围内

1. 折射元件的支持（当前仅支持反射镜）
2. 非高斯光束的支持
3. 多波长支持
4. 偏振效应
