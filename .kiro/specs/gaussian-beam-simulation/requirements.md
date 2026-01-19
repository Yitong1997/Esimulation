# 需求文档

## 简介

本文档定义了高斯光束传输仿真模型的需求规范。该模型基于现有的 `wavefront_sampler` 与 `element_raytracer` 模块，结合 PROPER 物理光学传播和 optiland 几何光线追迹，实现高斯光束通过光学系统的混合仿真。

仿真模型的核心功能包括：
1. 高斯光束的定义与初始化（支持 M² 因子、束腰位置、附加波前误差）
2. 光学元件的定义（反射镜、透射镜，支持旋转与偏心）
3. 物理光学传播（使用 PROPER 库）
4. 混合光线追迹（波前采样 + 元件光线追迹）
5. 基于 ABCD 矩阵法的理论验证

## 术语表

- **Gaussian_Beam**: 高斯光束，一种横向强度分布为高斯函数的光束
- **Waist_Radius (w0)**: 束腰半径，高斯光束最窄处的半径（1/e² 强度半径）
- **Waist_Position (z0)**: 束腰位置，高斯光束最窄处在光轴上的位置
- **M2_Factor**: M² 因子，光束质量因子，表征实际光束与理想高斯光束的偏离程度
- **Rayleigh_Distance (zR)**: 瑞利距离，光束半径增大到 √2 倍束腰半径的距离
- **Wavefront_Error**: 波前误差，实际波前与理想波前的偏差
- **PROPER**: 物理光学传播库，用于衍射传播计算
- **Optiland**: 几何光学库，用于光线追迹和 OPD 计算
- **ABCD_Matrix**: ABCD 矩阵，用于描述高斯光束通过光学系统的变换
- **Hybrid_Simulator**: 混合仿真器，结合物理光学和几何光学的仿真方法
- **Wavefront_Sampler**: 波前采样器，将波前复振幅采样为几何光线
- **Element_Raytracer**: 元件光线追迹器，对光线进行光学元件追迹
- **OPD**: 光程差（Optical Path Difference）
- **PSF**: 点扩散函数（Point Spread Function）

## 需求

### 需求 1：高斯光束定义

**用户故事：** 作为光学工程师，我希望能够定义具有完整参数的高斯光束，以便进行精确的光束传输仿真。

#### 验收标准

1. THE Gaussian_Beam SHALL 接受束腰半径 w0 参数（单位：mm），且 w0 必须为正值
2. THE Gaussian_Beam SHALL 接受束腰位置 z0 参数（单位：mm），表示束腰相对于全局坐标系原点的位置
3. THE Gaussian_Beam SHALL 接受 M² 因子参数，默认值为 1.0，且 M² >= 1.0
4. THE Gaussian_Beam SHALL 接受初始面位置 z_init 参数（单位：mm），表示波前生成的位置
5. THE Gaussian_Beam SHALL 接受波长参数（单位：μm），本仿真固定为 0.5 μm
6. THE Gaussian_Beam SHALL 接受可选的波前误差函数，用于定义附加的波前误差形状
7. WHEN 波前误差函数被指定时，THE Gaussian_Beam SHALL 支持 Zernike 多项式形式的波前误差
8. THE Gaussian_Beam SHALL 根据参数自动计算瑞利距离 zR = π * w0² / (M² * λ)
9. THE Gaussian_Beam SHALL 提供方法计算任意位置 z 处的光束半径 w(z)
10. THE Gaussian_Beam SHALL 提供方法计算任意位置 z 处的波前曲率半径 R(z)
11. THE Gaussian_Beam SHALL 提供方法生成指定位置的波前复振幅（包含振幅和相位）

### 需求 2：光学元件定义

**用户故事：** 作为光学工程师，我希望能够定义各种光学元件，以便构建完整的光学系统进行仿真。

#### 验收标准

1. THE Optical_Element SHALL 支持定义抛物面反射镜，包含母抛物面焦距参数
2. THE Optical_Element SHALL 支持定义球面反射镜，包含曲率半径参数
3. THE Optical_Element SHALL 支持定义薄透镜，包含焦距参数
4. THE Optical_Element SHALL 接受元件位置参数 z_position（单位：mm）
5. THE Optical_Element SHALL 接受半口径参数 semi_aperture（单位：mm）
6. THE Optical_Element SHALL 接受绕 X 轴旋转角度 tilt_x（单位：rad）
7. THE Optical_Element SHALL 接受绕 Y 轴旋转角度 tilt_y（单位：rad）
8. THE Optical_Element SHALL 接受 X 方向偏心 decenter_x（单位：mm）
9. THE Optical_Element SHALL 接受 Y 方向偏心 decenter_y（单位：mm）
10. THE Optical_Element SHALL 提供方法获取入射面中心位置（全局坐标系）
11. THE Optical_Element SHALL 提供方法获取主光线方向（考虑倾斜）
12. THE Optical_Element SHALL 提供方法获取 optiland 表面定义

### 需求 3：物理光学传播

**用户故事：** 作为光学工程师，我希望使用物理光学方法进行波前传播，以便准确模拟衍射效应。

#### 验收标准

1. THE Hybrid_Simulator SHALL 使用 PROPER 库初始化高斯光束波前
2. THE Hybrid_Simulator SHALL 应用高斯振幅分布到初始波前
3. THE Hybrid_Simulator SHALL 应用初始相位（球面波前相位 + 波前误差）到初始波前
4. WHEN 传播距离被指定时，THE Hybrid_Simulator SHALL 使用 PROPER 的 prop_propagate 函数进行物理光学传播
5. THE Hybrid_Simulator SHALL 支持传播距离远小于瑞利距离的近场传播
6. THE Hybrid_Simulator SHALL 在传播过程中保持波前的采样精度
7. THE Hybrid_Simulator SHALL 提供方法获取当前波前的复振幅分布
8. THE Hybrid_Simulator SHALL 提供方法获取当前波前的采样间隔

### 需求 4：混合光线追迹

**用户故事：** 作为光学工程师，我希望在元件处使用混合方法进行光线追迹，以便准确计算元件引入的相位变化。

#### 验收标准

1. WHEN 波前到达元件位置时，THE Hybrid_Simulator SHALL 从 PROPER 波前提取复振幅
2. THE Hybrid_Simulator SHALL 使用 Wavefront_Sampler 将波前采样为几何光线
3. THE Hybrid_Simulator SHALL 使用方形均匀采样分布，确保光线充满整个方形区域
4. THE Hybrid_Simulator SHALL 使用 Element_Raytracer 对光线进行追迹
5. THE Hybrid_Simulator SHALL 从出射光线计算相对于主光线的 OPD
6. THE Hybrid_Simulator SHALL 从出射光线重建出射面的相位分布
7. THE Hybrid_Simulator SHALL 从出射光线重建出射面的振幅分布
8. THE Hybrid_Simulator SHALL 将重建的相位应用到 PROPER 波前
9. IF 元件为反射镜，THEN THE Hybrid_Simulator SHALL 正确处理反射后的传播方向

### 需求 5：波前重建

**用户故事：** 作为光学工程师，我希望从出射光线准确重建波前，以便继续进行物理光学传播。

#### 验收标准

1. THE Hybrid_Simulator SHALL 从离散光线位置和 OPD 数据重建连续相位分布
2. THE Hybrid_Simulator SHALL 使用插值方法将离散数据转换为网格数据
3. THE Hybrid_Simulator SHALL 正确处理无效光线（被遮挡或渐晕的光线）
4. THE Hybrid_Simulator SHALL 确保重建的相位分布与原始网格大小一致
5. IF 有效光线数量不足，THEN THE Hybrid_Simulator SHALL 返回零相位分布并记录警告

### 需求 6：仿真流程控制

**用户故事：** 作为光学工程师，我希望能够控制仿真流程，以便灵活地进行各种仿真场景。

#### 验收标准

1. THE Hybrid_Simulator SHALL 提供 propagate_to(z) 方法，传播到指定位置
2. THE Hybrid_Simulator SHALL 按元件位置顺序依次处理光学元件
3. THE Hybrid_Simulator SHALL 记录每一步传播的历史信息
4. THE Hybrid_Simulator SHALL 提供 reset() 方法，重置仿真器到初始状态
5. THE Hybrid_Simulator SHALL 提供 get_psf() 方法，获取点扩散函数
6. THE Hybrid_Simulator SHALL 返回 SimulationResult 对象，包含振幅、相位、采样间隔等信息
7. THE SimulationResult SHALL 包含估计的光束半径
8. THE SimulationResult SHALL 包含波前 RMS 和 PV 统计信息

### 需求 7：ABCD 矩阵验证

**用户故事：** 作为光学工程师，我希望使用 ABCD 矩阵法计算理论结果，以便验证混合仿真的准确性。

#### 验收标准

1. THE ABCD_Calculator SHALL 接受高斯光束和光学元件列表作为输入
2. THE ABCD_Calculator SHALL 计算初始位置的复光束参数 q
3. THE ABCD_Calculator SHALL 提供自由空间传播矩阵
4. THE ABCD_Calculator SHALL 提供薄透镜/反射镜变换矩阵
5. THE ABCD_Calculator SHALL 提供 propagate_to(z) 方法，计算指定位置的光束参数
6. THE ABCD_Calculator SHALL 返回 ABCDResult 对象，包含光束半径、曲率半径、束腰位置等
7. THE ABCD_Calculator SHALL 提供 get_output_waist() 方法，获取输出束腰位置和半径
8. THE ABCD_Calculator SHALL 提供 trace_beam_profile() 方法，追迹光束轮廓

### 需求 8：验证测试

**用户故事：** 作为光学工程师，我希望通过验证测试确认仿真结果的准确性，以便信任仿真结果。

#### 验收标准

1. THE Validation_Test SHALL 设置单个抛物面反射镜作为测试光学系统
2. THE Validation_Test SHALL 使用 ABCD 矩阵法计算理论输出束腰位置
3. THE Validation_Test SHALL 使用混合仿真计算实际输出束腰位置
4. THE Validation_Test SHALL 比较理论和实际的束腰位置，误差应小于 5%
5. THE Validation_Test SHALL 比较理论和实际的束腰半径，误差应小于 5%
6. WHEN 输入为平面波前时，THE Validation_Test SHALL 验证输出束腰位于反射镜焦点
7. THE Validation_Test SHALL 验证在束腰位置波前曲率半径趋近于无穷大（平场波前）
8. THE Validation_Test SHALL 输出详细的比较报告，包含理论值、仿真值和误差

### 需求 9：错误处理

**用户故事：** 作为光学工程师，我希望系统能够正确处理错误情况，以便快速定位和解决问题。

#### 验收标准

1. IF 波长参数为非正值，THEN THE System SHALL 抛出 ValueError 并提供明确的错误信息
2. IF 束腰半径为非正值，THEN THE System SHALL 抛出 ValueError 并提供明确的错误信息
3. IF M² 因子小于 1.0，THEN THE System SHALL 抛出 ValueError 并提供明确的错误信息
4. IF 元件半口径为非正值，THEN THE System SHALL 抛出 ValueError 并提供明确的错误信息
5. IF 光线追迹失败（无有效光线），THEN THE System SHALL 记录警告并返回默认结果
6. IF PROPER 波前初始化失败，THEN THE System SHALL 抛出 RuntimeError 并提供诊断信息
