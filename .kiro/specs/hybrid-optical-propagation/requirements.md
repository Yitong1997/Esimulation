# 需求文档：混合光学传播系统

## 简介

本功能实现完整的混合光学传播系统，将 PROPER 物理光学传输与 optiland 几何光线追迹相结合，对导入的 Zemax 序列模式光路结构进行高精度仿真。系统支持从入射波面到最终输出的完整传播链路，包括自由空间衍射传播和元件处的混合传播仿真。

### 核心设计原则

- **物理正确性**：严格遵循物理光学和几何光学原理
- **模块化设计**：各传播阶段独立封装，便于测试和维护
- **数据一致性**：在整个传播过程中保持复振幅、Pilot Beam 参数和 PROPER 对象的一致性
- **与现有系统集成**：复用 zemax-optical-axis-tracing、hybrid-amplitude-reconstruction、phase-unwrapping-pilot-beam 等已有功能

### 依赖的现有 Spec

1. **zemax-optical-axis-tracing**：Zemax 序列模式光路结构定义和坐标转换
2. **hybrid-amplitude-reconstruction**：混合传播复振幅重建优化
3. **phase-unwrapping-pilot-beam**：基于 Pilot Beam 的相位解包裹

## 术语表

- **Simulation_Amplitude（仿真复振幅）**: 包含振幅和非折叠相位的复数场，形式为 A·exp(iφ)，相位相对于主光线定义
- **PROPER_Amplitude（PROPER 复振幅）**: PROPER 库内部存储的复振幅，相位相对于参考球面波前存储
- **Pilot_Beam（导频光束）**: 基于 ABCD 法则计算的理想高斯光束，用于相位解包裹和参考
- **Entrance_Plane（入射面）**: 垂直于入射光轴的平面，原点为主光线与元件表面的交点
- **Exit_Plane（出射面）**: 垂直于出射光轴的平面，原点为主光线与元件表面的交点
- **Tangent_Plane（元件切平面）**: 与元件表面相切的平面，用于纯衍射方法
- **Wrapped_Phase（折叠相位）**: PROPER 返回的相位，范围在 [-π, π]
- **Unwrapped_Phase（非折叠相位）**: 解包裹后的连续相位
- **Optical_Axis_State（光轴状态）**: 包含主光线位置和方向的状态信息
- **Free_Space_Propagation（自由空间传播）**: 面与面之间的衍射传播
- **Hybrid_Element_Propagation（混合元件传播）**: 元件处的波前-光线-波前重建流程
- **Local_Raytracing_Method（局部光线追迹方法）**: 使用几何光线追迹计算元件 OPD 的方法
- **Pure_Diffraction_Method（纯衍射方法）**: 使用 tilted_asm 和表面矢高计算的方法

## 需求

### 需求 1：入射波面定义

**用户故事：** 作为光学仿真工程师，我希望能够定义入射波面，包括理想高斯光束和初始复振幅像差，以便进行完整的光学系统仿真。

#### 验收标准

1.1 THE System SHALL 支持定义理想高斯光束参数（波长、束腰半径、束腰位置）
1.2 THE System SHALL 支持叠加初始复振幅像差（振幅修正和相位修正）
1.3 WHEN 初始化入射波面 THEN System SHALL 创建对应的 PROPER 波前对象
1.4 WHEN 初始化入射波面 THEN System SHALL 初始化 Pilot_Beam 参数
1.5 THE 入射波面 SHALL 定义在垂直于初始光轴的平面上
1.6 THE 初始复振幅像差 SHALL 以相对于理想高斯光束的形式定义

### 需求 2：光轴追踪（前置步骤）

**用户故事：** 作为光学仿真工程师，我希望系统能够预先追踪整个光学系统的光轴，以便获取所有面处主光线的位置和方向。

#### 验收标准

2.1 WHEN 开始仿真 THEN System SHALL 首先使用 optiland 追踪光学系统的主光轴
2.2 THE System SHALL 记录所有表面处主光线入射时的位置（全局坐标，mm）
2.3 THE System SHALL 记录所有表面处主光线入射时的方向（方向余弦）
2.4 THE System SHALL 记录所有表面处主光线出射时的位置（全局坐标，mm）
2.5 THE System SHALL 记录所有表面处主光线出射时的方向（方向余弦）
2.6 THE 光轴追踪结果 SHALL 复用 zemax-optical-axis-tracing spec 的实现
2.7 THE System SHALL 提供 `get_optical_axis_at_surface(index)` 方法获取指定面的光轴状态

### 需求 3：入射面和出射面定义

**用户故事：** 作为光学仿真工程师，我希望系统能够正确定义每个元件的入射面和出射面，以便进行波前采样和重建。

#### 验收标准

3.1 THE Entrance_Plane SHALL 垂直于入射光轴
3.2 THE Entrance_Plane 原点 SHALL 为主光线与元件表面的交点
3.3 THE Exit_Plane SHALL 垂直于出射光轴
3.4 THE Exit_Plane 原点 SHALL 为主光线与元件表面的交点
3.5 THE Entrance_Plane 和 Exit_Plane SHALL 不包含整体倾斜相位
3.6 WHEN 元件为反射镜 THEN 入射面和出射面的原点 SHALL 相同（均为交点）

### 需求 4：自由空间传播

**用户故事：** 作为光学仿真工程师，我希望系统能够正确执行面与面之间的自由空间衍射传播，包括处理负厚度的情况。

#### 验收标准

4.1 THE System SHALL 使用 PROPER 的衍射传播功能执行自由空间传播
4.2 THE System SHALL 自动判断当前属于近场还是远场范畴
4.3 WHEN 传播距离为正（沿主光线方向） THEN System SHALL 执行正向传播
4.4 WHEN 传播距离为负（逆主光线方向） THEN System SHALL 执行逆向传播
4.5 THE 传播距离 SHALL 以主光线交点连线方向矢量的长度绝对值计算
4.6 WHEN 方向矢量与主光线方向相同 THEN 传播距离 SHALL 为正
4.7 WHEN 方向矢量与主光线方向相反 THEN 传播距离 SHALL 为负
4.8 THE 传播后 System SHALL 在下一个面的入射面得到 PROPER 形式的仿真复振幅

### 需求 5：相位解包裹

**用户故事：** 作为光学仿真工程师，我希望系统能够正确解包裹 PROPER 提取的折叠相位，以便进行光线采样和 OPD 计算。

#### 验收标准

5.1 THE System SHALL 使用 Pilot_Beam 参考相位进行解包裹
5.2 THE 解包裹公式 SHALL 为：`T_unwrapped = T_pilot + angle(exp(1j * (T - T_pilot)))`
5.3 WHEN 从 PROPER 提取相位 THEN System SHALL 识别其为折叠相位（范围 [-π, π]）
5.4 THE 解包裹后相位 SHALL 与 Pilot_Beam 相位差异小于 π
5.5 THE 解包裹后相位 SHALL 平滑连续，无 2π 跳变
5.6 THE System SHALL 复用 phase-unwrapping-pilot-beam spec 的实现


### 需求 6：混合元件传播 - 局部光线追迹方法（默认）

**用户故事：** 作为光学仿真工程师，我希望系统能够使用局部光线追迹方法计算元件处的波前变化，以便获得高精度的 OPD 计算结果。

#### 验收标准

6.1 THE Local_Raytracing_Method SHALL 为默认的元件传播方法
6.2 WHEN 入射至与上一个面不同材质的面时 THEN System SHALL 触发混合元件传播
6.3 THE System SHALL 从入射面采样光线，使用非折叠的仿真复振幅
6.4 THE System SHALL 使用 ElementRaytracer 进行光线追迹
6.5 THE System SHALL 计算每条光线的 OPD（相对于主光线）
6.6 THE System SHALL 使用雅可比矩阵方法计算出射振幅
6.7 THE System SHALL 在出射面重建仿真复振幅
6.8 THE System SHALL 将出射面仿真复振幅转换为 PROPER 形式
6.9 THE System SHALL 复用 hybrid-amplitude-reconstruction spec 的实现

### 需求 7：混合元件传播 - 纯衍射方法（可选）

**用户故事：** 作为光学仿真工程师，我希望系统能够提供纯衍射方法作为替代选项，以便在某些场景下使用。

#### 验收标准

7.1 THE Pure_Diffraction_Method SHALL 作为可选的元件传播方法
7.2 THE System SHALL 使用 tilted_asm 从入射面传播到元件切平面
7.3 THE System SHALL 在切平面计算表面矢高（由切平面处的几何面形计算）
7.4 THE System SHALL 将表面矢高转换为相位延迟量并乘以复振幅
7.5 THE System SHALL 使用 tilted_asm 从切平面传播到出射面
7.6 THE 计算结果 SHALL 不包含倾斜相位（因为垂直于主光轴）
7.7 THE System SHALL 提供方法选择参数 `propagation_method='local_raytracing'|'pure_diffraction'`

### 需求 8：Pilot Beam 参数传递

**用户故事：** 作为光学仿真工程师，我希望系统能够在整个传播过程中正确追踪和传递 Pilot Beam 参数，以便进行相位解包裹和参考。

#### 验收标准

8.1 THE System SHALL 使用 ABCD 法则追踪 Pilot_Beam 参数
8.2 THE Pilot_Beam 参数 SHALL 包含：束腰位置、束腰半径、当前曲率半径、当前光斑大小
8.3 WHEN 经过光学元件 THEN System SHALL 更新 Pilot_Beam 参数
8.4 THE Pilot_Beam 相位 SHALL 以非折叠方式解析计算
8.5 THE Pilot_Beam 相位 SHALL 定义为相对于主光线的相位延迟（主光线处为 0）
8.6 THE Pilot_Beam 相位公式 SHALL 为：`φ_pilot(r) = k × r² / (2 × R)`，其中 k = 2π/λ
8.7 THE System SHALL 在每个入射面和出射面计算 Pilot_Beam 参考相位

### 需求 9：仿真复振幅与 PROPER 复振幅转换

**用户故事：** 作为光学仿真工程师，我希望系统能够正确在仿真复振幅和 PROPER 复振幅之间转换，以便在不同模块间传递数据。

#### 验收标准

9.1 THE Simulation_Amplitude SHALL 使用非折叠相位，相对于主光线定义
9.2 THE PROPER_Amplitude SHALL 使用 PROPER 内部的参考球面波前定义
9.3 WHEN 从 PROPER 提取仿真复振幅 THEN System SHALL：
    a) 提取折叠相位
    b) 使用 Pilot_Beam 参考相位解包裹
    c) 组合振幅和非折叠相位得到仿真复振幅
9.4 WHEN 将仿真复振幅写入 PROPER THEN System SHALL：
    a) 使用 Pilot_Beam 参数初始化 PROPER 对象
    b) 计算仿真复振幅与参考波面的残差
    c) 将残差赋值至 PROPER 对象的复振幅属性
9.5 THE 转换过程 SHALL 保证（复振幅与参考波的合计）等于仿真复振幅

### 需求 10：数据传递一致性

**用户故事：** 作为光学仿真工程师，我希望系统在整个传播过程中保持数据的一致性，以便追踪和调试。

#### 验收标准

10.1 THE System SHALL 在每个传播位置维护以下数据：
    a) 仿真复振幅（非折叠相位）
    b) Pilot_Beam 参数
    c) PROPER 形式的复振幅
10.2 THE 三种数据表示 SHALL 在物理上等价
10.3 THE System SHALL 提供 `get_state_at_surface(index)` 方法获取指定面的完整状态
10.4 THE System SHALL 支持保存中间结果用于调试
10.5 THE System SHALL 提供验证方法检查数据一致性

### 需求 11：材质变化检测

**用户故事：** 作为光学仿真工程师，我希望系统能够自动检测材质变化，以便正确触发混合元件传播。

#### 验收标准

11.1 THE System SHALL 检测相邻面之间的材质变化
11.2 WHEN 入射至反射镜面 THEN System SHALL 触发混合元件传播
11.3 WHEN 入射至透镜前表面（空气→玻璃） THEN System SHALL 触发混合元件传播
11.4 WHEN 从透镜后表面出射（玻璃→空气） THEN System SHALL 触发混合元件传播
11.5 WHEN 相邻面材质相同（如空气→空气） THEN System SHALL 仅执行自由空间传播
11.6 THE System SHALL 支持自定义材质折射率

### 需求 12：单位约定

**用户故事：** 作为光学仿真工程师，我希望系统使用一致的单位约定，以避免单位转换错误。

#### 验收标准

12.1 THE optiland 模块 SHALL 使用毫米（mm）作为长度单位
12.2 THE PROPER 模块 SHALL 使用米（m）作为长度单位
12.3 THE System SHALL 在 optiland 和 PROPER 之间自动进行单位转换
12.4 THE 波长 SHALL 以微米（μm）为单位输入，内部转换为所需单位
12.5 THE OPD SHALL 以波长数为单位
12.6 THE 相位 SHALL 以弧度为单位
12.7 THE 转换公式 SHALL 为：`phase = 2π × opd_waves`

### 需求 13：坐标系统

**用户故事：** 作为光学仿真工程师，我希望系统使用一致的坐标系统，以便正确定义光学元件和追踪光路。

#### 验收标准

13.1 THE 全局坐标系 SHALL 为右手系，Z 轴为初始光轴方向，Y 轴垂直向上
13.2 THE 入射面和出射面 SHALL 始终垂直于当前光轴
13.3 THE 波前 SHALL 不具有整体倾斜（Tilt）
13.4 THE System SHALL 遵循 Zemax 序列模式的坐标系演化规则
13.5 THE System SHALL 复用 zemax-optical-axis-tracing spec 的坐标转换实现

### 需求 14：错误处理

**用户故事：** 作为光学仿真工程师，我希望系统能够正确处理错误情况并提供清晰的错误信息。

#### 验收标准

14.1 IF 光线追迹失败（无交点） THEN System SHALL 抛出异常并提供详细信息
14.2 IF 有效光线数量不足 THEN System SHALL 抛出异常并建议增加采样密度
14.3 IF 相位解包裹失败（残差过大） THEN System SHALL 发出警告
14.4 IF 传播距离为零 THEN System SHALL 跳过自由空间传播步骤
14.5 IF 材质折射率无效 THEN System SHALL 抛出异常
14.6 THE 所有错误信息 SHALL 包含足够的上下文用于调试

### 需求 15：性能要求

**用户故事：** 作为光学仿真工程师，我希望系统具有合理的性能，以便进行实际工程应用。

#### 验收标准

15.1 THE 单次自由空间传播 SHALL 在 100ms 内完成（512×512 网格）
15.2 THE 单次混合元件传播 SHALL 在 500ms 内完成（512×512 网格，200×200 光线）
15.3 THE 相位解包裹 SHALL 在 10ms 内完成（512×512 网格）
15.4 THE System SHALL 支持批量处理多个波长
15.5 THE System SHALL 支持并行计算（可选）

### 需求 16：接口设计

**用户故事：** 作为光学仿真工程师，我希望系统提供清晰的 API 接口，以便集成到现有工作流程中。

#### 验收标准

16.1 THE System SHALL 提供 `HybridOpticalPropagator` 主类
16.2 THE System SHALL 提供 `propagate()` 方法执行完整传播
16.3 THE System SHALL 提供 `propagate_to_surface(index)` 方法传播到指定面
16.4 THE System SHALL 提供 `get_wavefront_at_surface(index)` 方法获取指定面的波前
16.5 THE System SHALL 支持从 ZMX 文件加载光学系统定义
16.6 THE System SHALL 与现有 SequentialOpticalSystem 类兼容


### 需求 17：网格物理尺寸匹配

**用户故事：** 作为光学仿真工程师，我希望系统能够正确处理不同传播位置的网格物理尺寸，以确保计算精度。

#### 验收标准

17.1 THE System SHALL 在每个传播位置追踪网格的物理尺寸
17.2 WHEN 执行自由空间传播 THEN System SHALL 根据 PROPER 的采样规则更新网格物理尺寸
17.3 WHEN 执行混合元件传播 THEN System SHALL 确保入射面和出射面的网格物理尺寸正确匹配
17.4 THE System SHALL 在光线采样时使用正确的网格物理尺寸确定采样范围
17.5 THE System SHALL 在复振幅重建时使用正确的网格物理尺寸进行插值
17.6 IF 网格物理尺寸不匹配 THEN System SHALL 执行重采样或发出警告
17.7 THE System SHALL 提供 `get_grid_sampling(surface_index)` 方法获取指定面的网格采样信息

### 需求 18：端到端测试

**用户故事：** 作为光学仿真工程师，我希望系统能够通过端到端测试验证完整传播链路的正确性。

#### 验收标准

18.1 THE System SHALL 支持从 ZMX 文件加载激光扩束镜系统进行端到端测试
18.2 THE System SHALL 支持伽利略 OAP 激光扩束镜结构的端到端测试
18.3 FOR 伽利略 OAP 扩束镜测试：
    a) System SHALL 使用离轴抛物面镜（OAP）配置
    b) System SHALL NOT 设置镜面倾斜量（tilt_x, tilt_y = 0）
    c) System SHALL 使用离轴距离（off-axis distance）定义 OAP
18.4 THE 端到端测试 SHALL 验证：
    a) 光束扩展比与理论值一致（误差 < 5%）
    b) 波前质量（Strehl ratio > 0.9 对于理想系统）
    c) 能量守恒（误差 < 5%）
18.5 THE System SHALL 支持更新现有伽利略 OAP 扩束镜示例的光路定义方式
18.6 THE 端到端测试 SHALL 包含与纯 PROPER 模式的对比验证


### 需求 19：PARAXIAL 表面类型处理

**用户故事：** 作为光学仿真工程师，我希望系统能够正确处理 Zemax 中的 PARAXIAL 表面类型（理想薄透镜），使用 optiland 的薄相位元件进行处理。

#### 验收标准

19.1 THE System SHALL 识别 Zemax ZMX 文件中的 PARAXIAL 表面类型
19.2 WHEN 表面类型为 PARAXIAL THEN System SHALL 使用 optiland 的薄相位元件进行处理
19.3 THE PARAXIAL 表面 SHALL 从 ZMX 文件中读取焦距参数（PARM 1）
19.4 THE 薄相位元件处理 SHALL 直接在 PROPER 波前上应用相位修正
19.5 THE 薄相位元件相位公式 SHALL 为：`φ = -k × r² / (2f)`，其中 f 为焦距
19.6 WHEN 处理 PARAXIAL 表面 THEN System SHALL 跳过完整的光线追迹流程
19.7 THE System SHALL 更新 Pilot_Beam 参数以反映 PARAXIAL 表面的聚焦效果
19.8 THE System SHALL NOT 对非 PARAXIAL 表面类型擅自使用傍轴近似（保持混合衍射传播仿真的意义）

