# Requirements Document

## Introduction

本文档定义了离轴抛物面（OAP）混合光学追迹误差系统化调试的需求。目标是通过排除法逐步定位误差来源，建立一个渐进式的验证流程。每个步骤验证通过后记录为"已验证"，后续步骤不再重复验证前面的内容。

## ⚠️⚠️⚠️ 强制禁止事项（极其重要）

### 🚫🚫🚫 绝对禁止的参数和方法

以下参数和方法已被**永久废弃**，在本项目中**不存在**，**永远不要使用、不要提及、不要尝试添加**：

| 禁止项 | 说明 |
|--------|------|
| `off_axis_distance` | 离轴距离参数 |
| `dy` | optiland 表面 Y 方向偏心参数 |
| `dx` | optiland 表面 X 方向偏心参数 |
| `add_oap` | 离轴抛物面添加方法 |
| `semi_aperture` | 半口径参数 |
| `aperture` | 口径参数 |

### ✅ 正确做法：绝对坐标定位

离轴效果必须通过修改元件的位置坐标来实现：
- Y 方向离轴 100mm → 设置 `y=100`
- X 方向离轴 50mm → 设置 `x=50`

### 🚫 禁止直接赋值 Pilot Beam 相位

**绝对禁止直接使用 Pilot Beam 对仿真复振幅或光线相位进行赋值！**

Pilot Beam 的唯一用途：
1. 作为参考相位进行相位解包裹
2. 计算残差 OPD 用于网格重采样
3. 验证仿真结果的正确性

## Glossary

- **OAP**: Off-Axis Parabola，离轴抛物面镜
- **Pilot_Beam**: 参考高斯光束，用于相位解包裹和 OPD 计算
- **Chief_Ray**: 主光线，光束中心的光线
- **OPD**: Optical Path Difference，光程差
- **Entrance_Surface**: 入射面，垂直于入射光轴的平面
- **Exit_Surface**: 出射面，垂直于出射光轴的平面
- **Residual_OPD**: 残差光程差，实际 OPD + Pilot_Beam 理论 OPD（注意是加法，因为符号约定）
- **ElementRaytracer**: 元件光线追迹器，使用 optiland 进行光线追迹
- **HybridElementPropagator**: 混合元件传播器，执行波前-光线-波前重建流程
- **Effective_Focal_Length**: 等效焦距，主光线交点到焦点的距离

## Requirements

### Requirement 1: 主光线追迹验证

**User Story:** As a developer, I want to verify that the chief ray tracing is correct for OAP, so that I can ensure the intersection point and exit direction are accurate.

#### Acceptance Criteria

1. WHEN tracing the chief ray through an OAP THEN the System SHALL compute the intersection point using the parabola equation z = r²/(2R)
2. WHEN the chief ray intersects an OAP at off-axis position (0, d) THEN the System SHALL compute z_intersection = d²/(2R)
3. WHEN computing the surface normal at the intersection point THEN the System SHALL use the gradient of the parabola: ∇z = (x/R, y/R)
4. WHEN computing the reflected direction THEN the System SHALL use the reflection formula: r = i - 2(i·n)n
5. WHEN the off-axis distance is d and radius is R THEN the exit angle SHALL equal 2×arctan(d/R)

### Requirement 2: 入射面坐标系验证

**User Story:** As a developer, I want to verify that the entrance surface coordinate system is correct, so that I can ensure rays are properly sampled.

#### Acceptance Criteria

1. THE Entrance_Surface SHALL be perpendicular to the incident chief ray direction
2. THE Entrance_Surface origin SHALL be located at the chief ray intersection with the optical surface
3. WHEN transforming rays from entrance surface local coordinates to global coordinates THEN the System SHALL use the rotation matrix computed from chief ray direction
4. WHEN the chief ray direction is (0, 0, 1) THEN the entrance surface local coordinates SHALL coincide with global coordinates

### Requirement 3: 出射面坐标系验证

**User Story:** As a developer, I want to verify that the exit surface coordinate system is correct, so that I can ensure output rays are properly transformed.

#### Acceptance Criteria

1. THE Exit_Surface SHALL be perpendicular to the exit chief ray direction
2. THE Exit_Surface origin SHALL be located at the chief ray intersection with the optical surface
3. WHEN transforming rays from global coordinates to exit surface local coordinates THEN the System SHALL use the transpose of the exit rotation matrix
4. WHEN the exit direction is computed from reflection THEN the exit surface Z-axis SHALL align with the exit chief ray direction

### Requirement 4: Pilot Beam 参数验证

**User Story:** As a developer, I want to verify that the Pilot Beam parameters are correctly computed at the exit surface, so that I can ensure accurate OPD calculation.

#### Acceptance Criteria

1. WHEN computing the effective focal length for an OAP THEN the System SHALL use f_eff = sqrt(d² + (f - z_intersection)²)
2. WHEN computing the effective curvature radius THEN the System SHALL use R_eff = 2 × f_eff
3. WHEN applying ABCD transformation for an OAP THEN the System SHALL use the effective curvature radius instead of the nominal radius
4. THE Pilot_Beam curvature radius at exit surface SHALL match the expected value based on ABCD transformation

### Requirement 5: optiland 离轴追迹验证

**User Story:** As a developer, I want to verify that optiland correctly traces rays through an off-axis parabola, so that I can ensure the raytracing core is accurate.

#### Acceptance Criteria

1. WHEN tracing rays through an OAP in optiland THEN the System SHALL correctly handle the off-axis geometry
2. WHEN using absolute coordinates in optiland THEN the System SHALL NOT require explicit off-axis distance parameter
3. WHEN the parabola vertex is at origin and rays enter at off-axis position THEN optiland SHALL compute correct intersection points
4. THE traced ray OPD from optiland SHALL be consistent with theoretical calculations for ideal OAP

### Requirement 6: 残差 OPD 计算验证

**User Story:** As a developer, I want to verify that the residual OPD calculation is correct, so that I can ensure accurate wavefront reconstruction.

#### Acceptance Criteria

1. WHEN computing residual OPD THEN the System SHALL use: residual = absolute_opd + pilot_opd (note: addition due to sign convention)
2. FOR an ideal OAP THEN the residual OPD RMS SHALL be less than 1 milli-wave
3. WHEN the Pilot Beam parameters are correct THEN the residual OPD SHALL be smooth and continuous without 2π jumps
4. THE residual OPD at the chief ray position SHALL be zero

### Requirement 7: 网格重采样验证

**User Story:** As a developer, I want to verify that the grid resampling process is correct, so that I can ensure accurate wavefront reconstruction.

#### Acceptance Criteria

1. WHEN resampling residual OPD from ray positions to grid THEN the System SHALL use appropriate interpolation method
2. THE resampled residual OPD SHALL preserve the smoothness of the original data
3. WHEN adding back Pilot Beam phase THEN the System SHALL use the same grid coordinates as the residual OPD
4. THE final reconstructed phase SHALL match the expected theoretical phase within tolerance

### Requirement 8: 测试参数组合

**User Story:** As a developer, I want to test with different parameter combinations, so that I can identify parameter-dependent errors.

#### Acceptance Criteria

1. THE System SHALL support testing with focal length = 2000 mm (R = 4000 mm)
2. THE System SHALL support testing with focal length = 100000 mm (R = 200000 mm, near-flat)
3. THE System SHALL support testing with off-axis distance = 0 mm (on-axis case)
4. THE System SHALL support testing with off-axis distance = 200 mm (off-axis case)
5. WHEN testing on-axis case (d = 0) THEN the results SHALL match the spherical mirror case

### Requirement 9: 验证状态追踪

**User Story:** As a developer, I want to track verification status for each step, so that I can avoid re-verifying already confirmed components.

#### Acceptance Criteria

1. THE System SHALL maintain a verification status for each debug step
2. WHEN a step is verified THEN the System SHALL mark it as "已验证" (verified)
3. THE System SHALL NOT re-verify steps that are already marked as verified
4. IF a later step fails THEN the System SHALL NOT automatically invalidate earlier verified steps
5. THE verification status SHALL be persisted across debug sessions

### Requirement 10: 约束条件

**User Story:** As a developer, I want to ensure the debug process follows project constraints, so that I can maintain code integrity.

#### Acceptance Criteria

1. THE System SHALL NOT assign Pilot_Beam phase directly to simulation amplitude or ray phase
2. THE System SHALL NOT add new parameters like "off_axis_distance" to the API
3. THE actual off-axis amount SHALL be determined by surface absolute coordinates (x, y, z) and angles
4. THE System SHALL NOT modify internal API interfaces unless a clear error is found
5. WHEN a component is verified THEN subsequent steps SHALL NOT question its correctness

### Requirement 11: 禁止使用的参数和方法

**User Story:** As a developer, I want to ensure that deprecated parameters and methods are never used, so that I can maintain API consistency.

#### Acceptance Criteria

1. THE System SHALL NOT use `off_axis_distance` parameter in any code
2. THE System SHALL NOT use `dy` or `dx` parameters in optiland surface definitions
3. THE System SHALL NOT use `add_oap` method
4. THE System SHALL NOT use `semi_aperture` or `aperture` parameters
5. WHEN defining off-axis surfaces THEN the System SHALL use absolute coordinates (x, y, z) only
6. THE System SHALL NOT set aperture or semi-aperture for any surface (Gaussian beam range is determined by w0)

### Requirement 12: 测试规范

**User Story:** As a developer, I want all tests to follow project testing standards, so that I can ensure consistent and reliable testing.

#### Acceptance Criteria

1. ALL tests SHALL be performed through the BTS main function API (`bts.simulate()`)
2. THE System SHALL NOT directly use low-level modules like `ElementRaytracer` or `WavefrontSampler` for testing
3. WHEN defining optical systems for testing THEN the System SHALL use `bts.OpticalSystem`
4. WHEN defining light sources for testing THEN the System SHALL use `bts.GaussianSource`
5. THE grid physical size SHALL always be 4×w0 (fixed by PROPER library)
6. THE `beam_diameter` parameter in `prop_begin` SHALL always equal 2×w0
7. THE `beam_diam_fraction` parameter in `prop_begin` SHALL always equal 0.5
