# 需求文档

## 简介

修复离轴抛物面镜（OAP）在混合传播模式下的像差计算 bug。当前系统在 `_apply_element_hybrid` 方法中，对于 `is_fold=False` 的情况使用了错误的差分方法计算像差，导致伽利略 OAP 扩束镜系统测试中残差 RMS 约为 1.5 waves。

**核心物理原理**：抛物面镜的定义特性是将轴上无穷远点源（平行光）聚焦到焦点，无像差。这与球面镜不同——球面镜倾斜会引入像散、彗差等。对于 OAP，即使使用 45° 离轴角度，对轴上平行光入射仍然是无像差的。

**当前问题**：`is_fold=False` 分支使用差分方法（比较带倾斜和不带倾斜的表面），这种方法对于抛物面镜是错误的，因为抛物面本身就是无像差的。

## 术语表

- **OAP (Off-Axis_Parabolic_Mirror)**：离轴抛物面镜，从母抛物面上切取的离轴部分
- **Hybrid_Propagation_Mode**：混合传播模式，结合 PROPER 物理光学传播和 optiland 几何光线追迹
- **ElementRaytracer**：元件光线追迹器，使用 optiland 进行几何光线追迹计算 OPD
- **is_fold**：倾斜类型标志，True 表示折叠光路（不引入波前倾斜），False 表示失调倾斜（引入真实像差）
- **WFE (Wavefront_Error)**：波前误差，相对于理想波前的偏差
- **Conic_Constant**：圆锥常数，-1 表示抛物面，0 表示球面
- **Differential_Method**：差分方法，通过比较两种配置的差异来计算像差（当前错误的方法）
- **Raytracing_Based_Method**：基于光线追迹的方法，使用完整的几何光线追迹计算 OPD

## 需求

### 需求 1：`is_fold=False` 必须使用基于光线追迹的混合传播模型

**用户故事：** 作为光学系统设计师，我希望 `is_fold=False` 时使用正确的基于光线追迹的方法计算像差，以便获得物理正确的仿真结果。

#### 验收标准

1. WHEN 元件使用 is_fold=False 设置 THEN THE System SHALL 使用 ElementRaytracer 进行完整的光线追迹
2. WHEN 计算 is_fold=False 元件的像差 THEN THE System SHALL 计算相对于主光线的 OPD
3. WHEN 计算 is_fold=False 元件的像差 THEN THE System SHALL 减去理想聚焦 OPD（使用精确公式）
4. THE System SHALL NOT 对 is_fold=False 情况使用差分方法或简化公式
5. THE System SHALL NOT 对 is_fold=False 情况比较"带倾斜"和"不带倾斜"的表面

### 需求 2：抛物面镜无像差特性（物理正确性）

**用户故事：** 作为光学系统设计师，我希望抛物面镜对轴上平行光入射是无像差的，以便正确模拟 OAP 扩束镜系统。

#### 验收标准

1. WHEN 抛物面镜（conic = -1）接收轴上平行光入射 THEN THE System SHALL 产生接近零的波前误差（RMS < 0.1 waves）
2. FOR ALL 有效的抛物面镜配置（任意焦距、任意倾斜角度），轴上平行光入射的 WFE RMS SHALL 小于 0.1 waves
3. WHEN 抛物面镜使用 is_fold=False 设置 THEN THE System SHALL 同样不引入像差（因为抛物面对轴上光是无像差的）
4. THE System SHALL 正确反映抛物面镜的定义特性：将轴上平行光聚焦到焦点，无像差


### 需求 3：`is_fold=True` 代码保持不变

**用户故事：** 作为开发者，我希望 `is_fold=True` 的代码保持不变，因为它已经正确工作。

#### 验收标准

1. THE System SHALL NOT 修改 is_fold=True 分支的代码逻辑
2. WHEN 元件使用 is_fold=True 设置 THEN THE System SHALL 继续使用现有的正确实现
3. WHEN 元件使用 is_fold=True 设置 THEN THE System SHALL 不引入波前倾斜（仅改变光束传播方向）

### 需求 4：计算误差在容差范围内

**用户故事：** 作为光学系统设计师，我希望计算误差在可接受的容差范围内，以便信任仿真结果。

#### 验收标准

1. FOR ALL 无像差光学元件（如抛物面镜对轴上光），计算的 WFE RMS SHALL 小于 0.1 waves
2. WHEN 比较混合传播模式与解析解 THEN THE System SHALL 产生一致的结果（误差 < 1%）
3. THE System SHALL 在数值精度范围内正确计算 OPD

### 需求 5：球面镜像差行为保持不变

**用户故事：** 作为光学系统设计师，我希望球面镜的像差行为保持不变，以便正确模拟球面镜系统。

#### 验收标准

1. WHEN 球面镜（conic = 0）使用 is_fold=True 设置 THEN THE System SHALL 不引入波前倾斜
2. WHEN 球面镜使用 is_fold=False 设置且有倾斜 THEN THE System SHALL 引入真实的像差（像散、彗差等）
3. WHEN 球面镜使用 is_fold=False 设置且倾斜角度大于 5° THEN THE System SHALL 发出警告

### 需求 6：平面镜行为保持不变

**用户故事：** 作为光学系统设计师，我希望平面镜的行为保持不变，以便正确模拟折叠光路。

#### 验收标准

1. WHEN 平面镜（radius = infinity）使用任何 is_fold 设置 THEN THE System SHALL 不引入像差
2. WHEN 平面镜使用 is_fold=True 设置 THEN THE System SHALL 仅改变光束传播方向

### 需求 7：伽利略 OAP 扩束镜系统验证

**用户故事：** 作为光学系统设计师，我希望伽利略 OAP 扩束镜系统能够正确工作，以便验证修复的正确性。

#### 验收标准

1. WHEN 运行伽利略 OAP 扩束镜系统仿真 THEN THE System SHALL 产生正确的放大倍率（误差 < 1%）
2. WHEN 运行伽利略 OAP 扩束镜系统仿真 THEN THE System SHALL 产生接近零的波前误差（RMS < 0.1 waves）
3. WHEN 比较混合传播模式与 ABCD 矩阵方法 THEN THE System SHALL 产生一致的光束半径（误差 < 1%）

### 需求 8：代码兼容性

**用户故事：** 作为开发者，我希望修复不破坏现有 API，以便现有代码能够继续工作。

#### 验收标准

1. THE System SHALL 保持与现有 API 的兼容性
2. THE System SHALL 不改变 ParabolicMirror、SphericalMirror、FlatMirror 类的公共接口
3. THE System SHALL 不改变 SequentialOpticalSystem 类的公共接口
