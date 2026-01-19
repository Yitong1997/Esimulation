# Requirements Document

## Introduction

本文档定义了序列模式混合光学仿真系统的需求规范。该系统旨在提供类似 Zemax 序列模式的清晰易用接口，实现高斯光束在光学系统中的混合仿真（物理光学衍射传播 + 几何光线追迹）。

系统核心功能包括：
1. 定义初始高斯光束参数
2. 按顺序定义光学面（面型、间距、倾斜等）
3. 定义采样面位置
4. 执行混合光学仿真
5. 可视化光路和输出仿真结果

## Glossary

- **Sequential_System**: 序列光学系统，光学元件按顺序排列，光束依次通过每个元件
- **Gaussian_Beam**: 高斯光束，具有高斯强度分布的激光光束
- **Surface**: 光学面，光学系统中的单个表面（反射面或折射面）
- **Sampling_Plane**: 采样面，用于记录波前复振幅的观察平面
- **ABCD_Matrix**: ABCD 矩阵，用于描述高斯光束通过光学系统的变换
- **Wavefront**: 波前，等相位面的复振幅分布
- **OPD**: 光程差（Optical Path Difference），光线相对于参考光线的光程差异
- **M2_Factor**: M² 因子，光束质量因子，描述实际光束与理想高斯光束的偏离程度
- **Beam_Waist**: 束腰，高斯光束半径最小的位置
- **Rayleigh_Range**: 瑞利距离，光束半径增大到束腰半径 √2 倍的距离

## Requirements

### Requirement 1: 高斯光束定义

**User Story:** As a 光学工程师, I want to 定义初始高斯光束的完整参数, so that I can 准确描述入射光束的特性。

#### Acceptance Criteria

1. THE Gaussian_Beam SHALL accept wavelength parameter in micrometers (μm)
2. WHEN wavelength is provided, THE Gaussian_Beam SHALL validate it is a positive finite value
3. THE Gaussian_Beam SHALL accept beam waist radius (w0) parameter in millimeters (mm)
4. WHEN w0 is provided, THE Gaussian_Beam SHALL validate it is a positive finite value
5. THE Gaussian_Beam SHALL accept beam waist position (z0) parameter in millimeters (mm)
6. THE Gaussian_Beam SHALL accept M² factor parameter with default value of 1.0
7. WHEN M² factor is provided, THE Gaussian_Beam SHALL validate it is >= 1.0
8. THE Gaussian_Beam SHALL compute Rayleigh range as zR = π * w0² / (M² * λ)
9. THE Gaussian_Beam SHALL compute beam radius at any position z as w(z) = w0 * sqrt(1 + ((z-z0)/zR)²)
10. THE Gaussian_Beam SHALL compute wavefront curvature radius at any position z

### Requirement 2: 光学面定义

**User Story:** As a 光学工程师, I want to 按顺序定义各种类型的光学面, so that I can 构建完整的光学系统。

#### Acceptance Criteria

1. THE Surface SHALL support standard surface type (spherical)
2. THE Surface SHALL support conic surface type with conic constant parameter
3. THE Surface SHALL support flat surface type (radius = infinity)
4. THE Surface SHALL support parabolic surface type (conic constant = -1)
5. THE Surface SHALL accept radius of curvature parameter in millimeters (mm)
6. WHEN radius is positive, THE Surface SHALL represent a concave surface (center of curvature in +Z direction)
7. WHEN radius is negative, THE Surface SHALL represent a convex surface
8. THE Surface SHALL accept thickness parameter (distance to next surface) in millimeters (mm)
9. THE Surface SHALL accept material parameter ('mirror' for reflective, material name for refractive)
10. THE Surface SHALL accept semi-aperture parameter in millimeters (mm)
11. THE Surface SHALL accept tilt_x and tilt_y parameters in radians for surface tilting
12. THE Surface SHALL accept decenter_x and decenter_y parameters in millimeters for surface decentering
13. WHEN surface is reflective, THE Sequential_System SHALL reverse the propagation direction
14. THE Surface SHALL support off-axis parabolic mirror with off_axis_distance parameter in millimeters
15. WHEN off_axis_distance is specified for parabolic mirror, THE Surface SHALL compute correct local curvature at off-axis position

### Requirement 2.1: 反射镜类型

**User Story:** As a 光学工程师, I want to 定义各种类型的反射镜, so that I can 构建反射式光学系统。

#### Acceptance Criteria

1. THE Sequential_System SHALL support spherical mirror definition with radius of curvature
2. THE Sequential_System SHALL support flat mirror definition (radius = infinity)
3. THE Sequential_System SHALL support parabolic mirror definition with parent focal length
4. THE Sequential_System SHALL support off-axis parabolic mirror with parent focal length and off-axis distance
5. WHEN defining off-axis parabolic mirror, THE system SHALL accept parent_focal_length parameter in millimeters
6. WHEN defining off-axis parabolic mirror, THE system SHALL accept off_axis_distance parameter in millimeters
7. THE off-axis parabolic mirror SHALL compute effective focal length based on off-axis geometry
8. THE off-axis parabolic mirror SHALL handle beam direction change correctly after reflection
9. WHEN mirror has tilt, THE Sequential_System SHALL correctly compute the reflected beam direction

### Requirement 3: 序列系统构建

**User Story:** As a 光学工程师, I want to 使用简洁的接口构建序列光学系统, so that I can 快速定义和修改光学配置。

#### Acceptance Criteria

1. THE Sequential_System SHALL accept a Gaussian_Beam object as input source
2. THE Sequential_System SHALL accept a list of Surface objects in sequential order
3. THE Sequential_System SHALL automatically compute z-position of each surface based on thickness values
4. THE Sequential_System SHALL track cumulative optical path length along the beam path
5. WHEN surfaces are added, THE Sequential_System SHALL validate the configuration is physically realizable
6. THE Sequential_System SHALL support adding Sampling_Plane at any position along the optical path
7. THE Sequential_System SHALL accept grid_size parameter for wavefront sampling (default 512)
8. THE Sequential_System SHALL accept beam_ratio parameter for PROPER initialization (default 0.5)

### Requirement 4: 采样面定义

**User Story:** As a 光学工程师, I want to 在光路中任意位置定义采样面, so that I can 获取该位置的波前信息。

#### Acceptance Criteria

1. THE Sampling_Plane SHALL accept position parameter as optical path distance from source
2. THE Sampling_Plane SHALL accept optional name parameter for identification
3. WHEN simulation runs, THE Sequential_System SHALL record wavefront at each Sampling_Plane
4. THE Sampling_Plane SHALL store complex amplitude distribution
5. THE Sampling_Plane SHALL store sampling interval (mm/pixel)
6. THE Sampling_Plane SHALL compute and store beam radius from amplitude distribution

### Requirement 5: 混合仿真执行

**User Story:** As a 光学工程师, I want to 执行混合光学仿真, so that I can 获得准确的波前传播结果。

#### Acceptance Criteria

1. WHEN simulation starts, THE Sequential_System SHALL initialize PROPER wavefront with Gaussian beam parameters
2. THE Sequential_System SHALL use PROPER prop_propagate for free-space propagation between surfaces
3. WHEN beam encounters a thin lens, THE Sequential_System SHALL use PROPER prop_lens to apply phase
4. WHEN beam encounters a mirror or complex surface, THE Sequential_System SHALL use hybrid method (wavefront sampling + ray tracing + phase reconstruction)
5. THE Sequential_System SHALL track propagation direction changes at reflective surfaces
6. WHEN simulation completes, THE Sequential_System SHALL return results for all Sampling_Planes
7. IF an error occurs during simulation, THEN THE Sequential_System SHALL raise descriptive exception with context

### Requirement 6: ABCD 矩阵验证

**User Story:** As a 光学工程师, I want to 使用 ABCD 矩阵法验证仿真结果, so that I can 确保仿真的正确性。

#### Acceptance Criteria

1. THE Sequential_System SHALL provide ABCD matrix calculation for the optical system
2. THE ABCD_Calculator SHALL compute beam radius at any optical path distance
3. THE ABCD_Calculator SHALL compute wavefront curvature at any optical path distance
4. THE ABCD_Calculator SHALL compute output beam waist position and radius
5. WHEN comparing with physical simulation, THE beam radius error SHALL be within acceptable tolerance for paraxial beams

### Requirement 7: 光路可视化

**User Story:** As a 光学工程师, I want to 可视化整个光路的 2D 图, so that I can 直观理解光束传播过程。

#### Acceptance Criteria

1. THE Sequential_System SHALL provide 2D layout visualization method
2. THE visualization SHALL show optical surfaces with correct positions and orientations
3. THE visualization SHALL show beam envelope based on ABCD calculation
4. THE visualization SHALL mark Sampling_Plane positions
5. THE visualization SHALL support optional display of beam waist positions
6. WHEN show=False is specified, THE visualization SHALL not call plt.show() (for testing)
7. THE visualization SHALL return the matplotlib figure and axes objects

### Requirement 8: 仿真结果输出

**User Story:** As a 光学工程师, I want to 获取完整的仿真结果, so that I can 进行后续分析。

#### Acceptance Criteria

1. THE SimulationResult SHALL contain complex amplitude distribution at each Sampling_Plane
2. THE SimulationResult SHALL contain beam radius at each Sampling_Plane
3. THE SimulationResult SHALL contain sampling interval at each Sampling_Plane
4. THE SimulationResult SHALL provide method to compute M² factor from amplitude distribution
5. THE SimulationResult SHALL provide method to compute wavefront RMS and PV
6. THE SimulationResult SHALL provide method to extract phase distribution
7. THE SimulationResult SHALL provide method to extract amplitude distribution

### Requirement 9: 代码复用与兼容性

**User Story:** As a 开发者, I want to 复用现有模块并保持兼容性, so that I can 避免重复开发并确保稳定性。

#### Acceptance Criteria

1. THE Sequential_System SHALL reuse existing GaussianBeam class from gaussian_beam_simulation module
2. THE Sequential_System SHALL reuse existing ABCDCalculator class for ABCD calculations
3. THE Sequential_System SHALL reuse existing WavefrontToRaysSampler for hybrid method
4. THE Sequential_System SHALL reuse existing ElementRaytracer for ray tracing
5. THE Sequential_System SHALL maintain backward compatibility with existing optical element classes
6. WHEN new surface types are needed, THE system SHALL extend existing OpticalElement base class

### Requirement 10: 简洁的用户接口

**User Story:** As a 用户, I want to 使用简洁直观的 API 定义和运行仿真, so that I can 快速上手并高效工作。

#### Acceptance Criteria

1. THE user interface SHALL allow defining complete system in less than 20 lines of code for simple cases
2. THE Sequential_System SHALL provide fluent interface for adding surfaces
3. THE Sequential_System SHALL provide clear error messages in Chinese for invalid configurations
4. THE Sequential_System SHALL provide example code in docstrings
5. WHEN running simulation, THE user SHALL only need to call a single run() method
6. THE Sequential_System SHALL provide summary() method to print system configuration
