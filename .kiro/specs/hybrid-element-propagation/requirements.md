# 需求文档

## 简介

本文档定义了"混合元件传播"（Hybrid Element Propagation）功能的需求，该功能实现在带有倾斜的光学元器件处的完整波前-光线-波前重建流程。此功能是混合光学仿真系统的核心组件，结合 PROPER 物理光学传播和 optiland 几何光线追迹，实现高精度的波前计算。

核心目标是在元器件处实现：
1. 入射面复振幅 → 切平面复振幅（使用 tilted_asm）
2. 切平面复振幅 → 几何光线采样
3. 光线通过元件追迹（计算 OPD）
4. 光线 → 切平面复振幅重建
5. 切平面复振幅 → 出射面复振幅（使用 tilted_asm 反向传播）

同时引入 Pilot Beam 参考相位机制，解决相位混叠问题，确保复振幅重建的正确性。

## 术语表

- **Wavefront（波前）**: 等相位面，在本系统中以复振幅数组表示
- **Complex_Amplitude（复振幅）**: 包含振幅和相位信息的复数场，形式为 A·exp(iφ)
- **Entrance_Plane（入射面）**: 垂直于入射主光轴的平面，是模块的输入接口
- **Exit_Plane（出射面）**: 垂直于出射主光轴的平面，是模块的输出接口
- **Tangent_Plane（切平面）**: 光学元件顶点处与表面相切的平面，用于中间计算的局部坐标系
- **OPD（光程差）**: Optical Path Difference，光线相对于参考光线的光程差异
- **Pilot_Beam（导引光束）**: 用于计算参考相位的理想高斯光束，类似 Zemax 的 Pilot Beam 机制
- **Tilted_ASM（倾斜角谱法）**: 计算倾斜平面上衍射场的算法，用于入射面/出射面与切平面之间的传播
- **Element_Raytracer（元件光线追迹器）**: 基于 optiland 的光线追迹模块
- **WavefrontToRaysSampler（波前采样器）**: 将波前复振幅采样为几何光线的模块
- **Phase_Unwrapping（相位解包裹）**: 将包裹相位（-π 到 π）转换为连续相位的过程
- **Reference_Phase（参考相位）**: Pilot Beam 在采样面上的相位分布
- **Residual_Phase（残差相位）**: 实际相位与参考相位的差值
- **Local_Coordinate_System（局部坐标系）**: 切平面使用的坐标系，Z 轴垂直于切平面

## 需求

### 需求 1：入射面到切平面传播

**用户故事：** 作为光学仿真工程师，我希望将入射面的复振幅传播到元件切平面，以便在切平面局部坐标系中进行光线相关的计算。

#### 验收标准

1. WHEN 输入复振幅位于入射面且元件存在倾斜 THEN Hybrid_Element_Propagator SHALL 使用 Tilted_ASM 计算从入射面到切平面的复振幅传播
2. WHEN 元件无倾斜（正入射） THEN Hybrid_Element_Propagator SHALL 直接使用输入复振幅作为切平面复振幅（无需倾斜传播）
3. WHEN 执行倾斜传播 THEN Hybrid_Element_Propagator SHALL 正确计算入射面到切平面的旋转矩阵
4. THE Hybrid_Element_Propagator SHALL 保持传播前后的能量守恒（误差 < 1%）
5. THE Hybrid_Element_Propagator SHALL 将传播结果转换到切平面的局部坐标系中

### 需求 2：切平面波前采样

**用户故事：** 作为光学仿真工程师，我希望在切平面将复振幅采样为几何光线，以便进行后续的光线追迹计算。

#### 验收标准

1. WHEN 切平面复振幅准备就绪 THEN Wavefront_Sampler SHALL 提取复振幅的相位分布并创建等效相位面
2. WHEN 执行光线采样 THEN Wavefront_Sampler SHALL 生成平面波入射到相位面的光线
3. WHEN 采样完成 THEN Wavefront_Sampler SHALL 将输入复振幅的振幅分布赋值给对应光线的强度
4. THE Wavefront_Sampler SHALL 支持可配置的采样光线数量（默认 100 条）
5. THE Wavefront_Sampler SHALL 支持多种光线分布类型（hexapolar、rectangular 等）
6. THE Wavefront_Sampler SHALL 在切平面的局部坐标系中输出光线数据

### 需求 3：元件光线追迹

**用户故事：** 作为光学仿真工程师，我希望追迹光线通过光学元件，以便获取每条光线的 OPD 和出射方向。

#### 验收标准

1. WHEN 输入光线到达元件 THEN Element_Raytracer SHALL 追迹光线从切平面到镜面再返回切平面
2. WHEN 追迹完成 THEN Element_Raytracer SHALL 输出每条光线在切平面局部坐标系中的位置、方向和累积 OPD
3. WHEN 元件为反射镜 THEN Element_Raytracer SHALL 正确计算反射后的光线方向
4. WHEN 元件为折射面 THEN Element_Raytracer SHALL 正确计算折射后的光线方向
5. THE Element_Raytracer SHALL 支持带倾斜的光学元件（tilt_x, tilt_y 参数）
6. THE Element_Raytracer SHALL 标记无效光线（未能到达元件或被遮挡的光线）

### 需求 4：Pilot Beam 参考相位计算

**用户故事：** 作为光学仿真工程师，我希望计算 Pilot Beam 的参考相位，以便消除相位混叠问题。

#### 验收标准

1. WHEN 需要计算参考相位 THEN Pilot_Beam_Calculator SHALL 创建与输入光束匹配的理想高斯光束
2. WHEN 使用方案 A THEN Pilot_Beam_Calculator SHALL 使用 PROPER 计算高斯光束在薄相位元件近似下传播到切平面的相位，然后用 Tilted_ASM 反向传播回镜面切面
3. WHEN 使用方案 B THEN Pilot_Beam_Calculator SHALL 设置理想高斯光束 Pilot Beam，计算其在镜面反射后（取镜面顶点处的理论曲率）切平面上各位置的波前相位
4. THE Pilot_Beam_Calculator SHALL 输出切平面上每个采样点的参考相位值
5. THE Pilot_Beam_Calculator SHALL 支持在两种方案之间切换以进行精度对比

### 需求 5：Pilot Beam 适用性检测

**用户故事：** 作为光学仿真工程师，我希望系统能够检测 Pilot Beam 方法的适用性，以便在不满足条件时获得警告并采取相应措施。

#### 验收标准

1. WHEN Pilot Beam 相位计算完成 THEN Pilot_Beam_Validator SHALL 检查相邻像素间的相位差是否超过 π/2
2. IF 相邻像素相位差超过 π/2 THEN Pilot_Beam_Validator SHALL 发出采样不足警告并建议增加网格分辨率
3. WHEN 输入光束参数已知 THEN Pilot_Beam_Validator SHALL 检查光束发散角是否过大（超过设定阈值）
4. IF 光束发散角过大 THEN Pilot_Beam_Validator SHALL 发出警告并建议使用更小的传播步长
5. WHEN Pilot Beam 与实际光束参数已知 THEN Pilot_Beam_Validator SHALL 检查两者的光束尺寸差异是否超过 50%
6. IF 光束尺寸差异过大 THEN Pilot_Beam_Validator SHALL 发出警告并建议调整 Pilot Beam 参数
7. THE Pilot_Beam_Validator SHALL 计算并报告相位梯度的最大值和平均值
8. THE Pilot_Beam_Validator SHALL 提供 `validate()` 方法返回验证结果和详细诊断信息

### 需求 6：相位差值计算与修正

**用户故事：** 作为光学仿真工程师，我希望计算并修正光线相位与参考相位的差值，以避免相位混叠问题。

#### 验收标准

1. WHEN 光线追迹完成 THEN Phase_Corrector SHALL 在 Pilot Beam 相位图上插值获取每条光线位置的参考相位
2. WHEN 插值完成 THEN Phase_Corrector SHALL 计算实际光线相位与参考相位的差值（残差相位）
3. WHEN 差值计算完成 THEN Phase_Corrector SHALL 从光线相位中减去差值得到修正后的相位
4. THE Phase_Corrector SHALL 确保残差相位在 [-π, π] 范围内以避免相位包裹
5. IF 残差相位超出 [-π, π] 范围 THEN Phase_Corrector SHALL 发出警告并记录日志

### 需求 7：切平面复振幅重建

**用户故事：** 作为光学仿真工程师，我希望从修正后的光线数据重建切平面的复振幅，以便进行后续的出射面传播。

#### 验收标准

1. WHEN 光线相位修正完成 THEN Amplitude_Reconstructor SHALL 使用 optiland 的复振幅重建功能生成切平面复振幅
2. WHEN 重建完成 THEN Amplitude_Reconstructor SHALL 将 Pilot Beam 参考相位加回到重建的复振幅中
3. WHEN 最终切平面复振幅生成 THEN Amplitude_Reconstructor SHALL 输出完整的复振幅数组（振幅 + 相位）
4. THE Amplitude_Reconstructor SHALL 支持可配置的输出网格大小
5. THE Amplitude_Reconstructor SHALL 处理无效光线区域（设为零振幅或 NaN）

### 需求 8：切平面到出射面传播

**用户故事：** 作为光学仿真工程师，我希望将切平面的复振幅传播到出射面，以便输出垂直于出射主光轴的复振幅。

#### 验收标准

1. WHEN 切平面复振幅重建完成 THEN Hybrid_Element_Propagator SHALL 使用 Tilted_ASM 计算从切平面到出射面的复振幅传播
2. WHEN 元件为反射镜 THEN Hybrid_Element_Propagator SHALL 正确计算出射面相对于切平面的旋转矩阵（考虑反射后的光轴方向变化）
3. WHEN 元件无倾斜（正入射） THEN Hybrid_Element_Propagator SHALL 直接使用切平面复振幅作为出射面复振幅
4. THE Hybrid_Element_Propagator SHALL 保持传播前后的能量守恒（误差 < 1%）
5. THE Hybrid_Element_Propagator SHALL 输出在出射面全局坐标系中的复振幅

### 需求 9：API 整合

**用户故事：** 作为光学仿真工程师，我希望有一个统一的 API 来执行完整的混合元件传播，以便简化系统集成。

#### 验收标准

1. THE HybridElementPropagator SHALL 提供单一入口方法 `propagate()` 执行完整的入射面→切平面→光线追迹→切平面→出射面流程
2. THE HybridElementPropagator SHALL 接受入射面复振幅、元件定义、波长等参数
3. THE HybridElementPropagator SHALL 返回出射面的复振幅数组
4. WHEN 传播完成 THEN HybridElementPropagator SHALL 提供中间结果的访问方法（光线数据、OPD、切平面复振幅等）
5. THE HybridElementPropagator SHALL 支持可选的调试模式输出详细的中间步骤信息

### 需求 10：SequentialOpticalSystem 集成

**用户故事：** 作为光学仿真工程师，我希望 SequentialOpticalSystem 默认使用混合元件传播，以便获得更准确的仿真结果。

#### 验收标准

1. WHEN SequentialOpticalSystem 遇到光学元件 THEN System SHALL 调用 HybridElementPropagator 处理元件处的波前传播
2. THE SequentialOpticalSystem SHALL 支持通过参数切换使用纯 PROPER 传播或混合传播
3. WHEN 使用混合传播 THEN SequentialOpticalSystem SHALL 正确处理元件前后的坐标系转换
4. THE SequentialOpticalSystem SHALL 保持现有 API 的向后兼容性

### 需求 11：错误处理

**用户故事：** 作为光学仿真工程师，我希望系统能够正确处理各种错误情况，以便快速定位和解决问题。

#### 验收标准

1. IF 输入复振幅数组形状无效 THEN HybridElementPropagator SHALL 抛出 ValueError 并提供清晰的错误信息
2. IF 波长参数无效（非正数） THEN HybridElementPropagator SHALL 抛出 ValueError
3. IF 元件定义参数无效 THEN HybridElementPropagator SHALL 抛出 ValueError
4. IF 所有光线都无效 THEN HybridElementPropagator SHALL 抛出 SimulationError 并建议可能的解决方案
5. IF 相位残差过大（超过 π） THEN HybridElementPropagator SHALL 发出警告但继续执行

### 需求 12：性能要求

**用户故事：** 作为光学仿真工程师，我希望混合传播的性能在可接受范围内，以便进行实际的光学系统仿真。

#### 验收标准

1. THE HybridElementPropagator SHALL 在 512×512 网格、100 条光线配置下完成单次传播时间 < 5 秒
2. THE HybridElementPropagator SHALL 支持批量处理多个采样面以提高效率
3. THE HybridElementPropagator SHALL 使用 NumPy 向量化操作优化计算性能
