# Requirements Document

## Introduction

本功能实现基于 Zemax 序列模式的光路结构定义，将 Zemax 表面序列转换为 optiland 全局坐标系中的光学系统定义。该功能是混合光学仿真系统的核心基础，为后续的 OPD 计算和 PROPER 波前传播提供准确的光学系统定义。

**核心设计原则**：
- **统一处理**：所有表面类型（包括连续坐标断点、空气面）使用统一算法处理，不做特殊化
- **简洁代码**：垂直接入 optiland 库，最小化中间层代码
- **Zemax 兼容**：严格遵循 Zemax 序列模式的坐标系演化规则

## Glossary

- **Current_Coordinate_System**: 当前坐标系，Zemax 序列模式中随光路演化的局部坐标系，由原点位置和轴向量矩阵组成
- **Global_Coordinate_System**: 全局坐标系，optiland 使用的固定参考坐标系，右手系，Z 轴为初始光轴方向
- **Axes_Matrix**: 轴向量矩阵，3×3 矩阵，列向量分别为当前坐标系的 X、Y、Z 轴在全局坐标系中的方向
- **Coordinate_Break**: 坐标断点，Zemax 中用于改变当前坐标系的虚拟表面，支持偏心和旋转
- **Transformation_Order**: 变换顺序，坐标断点的 Order 参数，0 表示先平移后旋转，1 表示先旋转后平移
- **Virtual_Surface**: 虚拟表面，不产生光学作用的表面（如坐标断点），仅用于改变当前坐标系
- **Optical_Surface**: 光学表面，产生光学作用的表面（如反射镜、透镜面）
- **Surface_Shape_Parameters**: 表面形状参数（曲率半径、圆锥常数、非球面系数），始终相对于当前坐标系定义
- **GlobalSurfaceDefinition**: 全局坐标表面定义，存储转换后的表面在全局坐标系中的完整参数

## Requirements

### Requirement 1: 当前坐标系数据结构

**User Story:** As a developer, I want to track the current coordinate system state, so that I can correctly interpret Zemax surface parameters defined in local coordinates.

#### Acceptance Criteria

1.1 THE CurrentCoordinateSystem SHALL store origin position as a 3D vector (x, y, z) in millimeters
1.2 THE CurrentCoordinateSystem SHALL store axes orientation as a 3×3 matrix where columns represent X, Y, Z axis directions in global coordinates
1.3 THE CurrentCoordinateSystem SHALL be initialized with origin at (0, 0, 0) and axes as identity matrix
1.4 THE CurrentCoordinateSystem SHALL provide x_axis, y_axis, z_axis properties returning the respective column vectors
1.5 THE CurrentCoordinateSystem SHALL be immutable - all transformation methods return new instances

### Requirement 2: 坐标断点变换

**User Story:** As a developer, I want to correctly process coordinate breaks, so that I can model tilted and decentered optical elements following Zemax conventions.

#### Acceptance Criteria

2.1 WHEN Order=0 (先平移后旋转), THE CoordinateBreakProcessor SHALL:
    a) First translate origin by (decenter_x × current_X_axis + decenter_y × current_Y_axis)
    b) Then rotate axes by R_xyz = R_z(tilt_z) × R_y(tilt_y) × R_x(tilt_x)
    c) Finally advance origin by thickness × new_Z_axis

2.2 WHEN Order=1 (先旋转后平移), THE CoordinateBreakProcessor SHALL:
    a) First rotate axes by R_xyz = R_z(tilt_z) × R_y(tilt_y) × R_x(tilt_x)
    b) Then translate origin by (decenter_x × new_X_axis + decenter_y × new_Y_axis)
    c) Finally advance origin by thickness × new_Z_axis

2.3 THE rotation sequence SHALL be X → Y → Z (先绕X轴，再绕Y轴，最后绕Z轴)
2.4 THE decenter translation SHALL use the current/rotated axes directions, not global axes
2.5 WHEN tilt angles are provided in degrees, THE processor SHALL convert to radians internally


### Requirement 3: 厚度处理

**User Story:** As a developer, I want thickness values to correctly advance the coordinate system origin, so that surfaces are positioned correctly in global coordinates.

#### Acceptance Criteria

3.1 WHEN a surface has thickness t, THE origin SHALL advance by t × current_Z_axis direction
3.2 THE thickness advancement SHALL occur AFTER processing the surface (not before)
3.3 WHEN thickness is negative, THE origin SHALL move in the negative Z axis direction (backward propagation)
3.4 FOR coordinate breaks, THE thickness advancement SHALL occur AFTER decenter and rotation transformations
3.5 THE thickness value SHALL be preserved unchanged (no sign flipping based on surface type)

### Requirement 4: 旋转矩阵计算

**User Story:** As a developer, I want correct rotation matrices, so that coordinate transformations match Zemax behavior exactly.

#### Acceptance Criteria

4.1 THE rotation matrix R_x(θ) around X axis SHALL be:
    ```
    | 1    0       0    |
    | 0  cos(θ) -sin(θ) |
    | 0  sin(θ)  cos(θ) |
    ```

4.2 THE rotation matrix R_y(θ) around Y axis SHALL be:
    ```
    |  cos(θ)  0  sin(θ) |
    |    0     1    0    |
    | -sin(θ)  0  cos(θ) |
    ```

4.3 THE rotation matrix R_z(θ) around Z axis SHALL be:
    ```
    | cos(θ) -sin(θ)  0 |
    | sin(θ)  cos(θ)  0 |
    |   0       0     1 |
    ```

4.4 THE combined rotation R_xyz SHALL equal R_z × R_y × R_x (matrix multiplication order)
4.5 THE new axes matrix SHALL equal old_axes @ R_xyz (right multiplication)

### Requirement 5: 统一表面遍历算法

**User Story:** As a developer, I want a unified traversal algorithm that handles all surface types consistently, so that the code is simple and handles complex configurations automatically.

#### Acceptance Criteria

5.1 THE SurfaceTraversalAlgorithm SHALL process surfaces in sequential index order (0, 1, 2, ...)
5.2 THE algorithm SHALL NOT special-case consecutive coordinate breaks - each is processed independently
5.3 THE algorithm SHALL NOT special-case air surfaces - they update coordinate system like any other surface
5.4 FOR each surface, THE algorithm SHALL:
    a) If coordinate_break: apply transformation to CurrentCoordinateSystem, do NOT create optiland surface
    b) If optical_surface: record global position/orientation, create GlobalSurfaceDefinition
    c) Advance origin by thickness × current_Z_axis

5.5 THE algorithm SHALL distinguish surface types by surface_type field, NOT by material or other properties
5.6 WHEN multiple coordinate breaks are consecutive, THE algorithm SHALL apply each transformation cumulatively


### Requirement 6: Zemax 曲率半径符号转换

**User Story:** As a developer, I want to correctly interpret Zemax radius sign conventions, so that surfaces are oriented correctly in optiland.

#### Acceptance Criteria

6.1 WHEN Zemax radius R > 0, THE curvature center SHALL be at vertex + R × current_Z_axis (凹面朝向入射光)
6.2 WHEN Zemax radius R < 0, THE curvature center SHALL be at vertex + R × current_Z_axis (凸面朝向入射光)
6.3 WHEN Zemax radius R = ∞, THE surface SHALL be treated as a flat surface
6.4 THE radius value SHALL be passed to optiland unchanged - the orientation matrix handles the coordinate transformation
6.5 THE curvature center position in global coordinates SHALL be: vertex_global + R × current_Z_axis_global

### Requirement 7: 圆锥常数和非球面系数转换

**User Story:** As a developer, I want conic constants and asphere coefficients to be correctly transferred, so that surface shapes are preserved.

#### Acceptance Criteria

7.1 THE conic constant k SHALL be passed to optiland unchanged:
    - k = 0: sphere
    - k = -1: parabola
    - k < -1: hyperbola
    - -1 < k < 0: oblate ellipsoid
    - k > 0: prolate ellipsoid

7.2 THE even asphere coefficients (A4, A6, A8, ...) SHALL be passed unchanged
7.3 ALL surface shape parameters SHALL be interpreted in the current coordinate system at the surface
7.4 THE surface shape equation z = f(r) SHALL use the local coordinate system where z is along current_Z_axis

### Requirement 8: 全局坐标表面定义

**User Story:** As a developer, I want a clear data structure for converted surfaces, so that I can easily create optiland objects.

#### Acceptance Criteria

8.1 THE GlobalSurfaceDefinition SHALL store:
    - index: original Zemax surface index
    - surface_type: 'standard', 'even_asphere', 'flat'
    - vertex_position: 3D position in global coordinates (mm)
    - orientation: 3×3 rotation matrix (columns = local X, Y, Z axes in global)
    - radius: curvature radius (mm), same sign as Zemax
    - conic: conic constant, same value as Zemax
    - is_mirror: boolean indicating reflective surface
    - semi_aperture: half-diameter (mm)
    - material: material name
    - asphere_coeffs: list of even asphere coefficients

8.2 THE GlobalSurfaceDefinition SHALL provide a curvature_center property:
    - Returns vertex_position + radius × orientation[:, 2] for finite radius
    - Returns None for flat surfaces (infinite radius)

8.3 THE GlobalSurfaceDefinition SHALL provide a surface_normal property:
    - Returns -orientation[:, 2] (pointing toward incident light)


### Requirement 9: 反射镜与当前坐标系

**User Story:** As a developer, I want to understand that mirrors do NOT automatically change the current coordinate system, so that I model Zemax behavior correctly.

#### Acceptance Criteria

9.1 WHEN processing a mirror surface, THE CurrentCoordinateSystem SHALL NOT be automatically rotated
9.2 THE mirror surface SHALL only advance the origin by thickness × current_Z_axis
9.3 TO make the coordinate system follow the reflected beam, THE user MUST add explicit coordinate breaks
9.4 THE GlobalSurfaceDefinition for a mirror SHALL record the orientation at the time of definition
9.5 THE optical axis direction change due to reflection SHALL be tracked separately from CurrentCoordinateSystem

### Requirement 10: optiland 垂直集成

**User Story:** As a developer, I want minimal code to convert GlobalSurfaceDefinition to optiland objects, so that the integration is simple and maintainable.

#### Acceptance Criteria

10.1 THE OptilandConverter SHALL create optiland Surface objects directly from GlobalSurfaceDefinition
10.2 THE converter SHALL use optiland's coordinate_system parameter to set surface orientation
10.3 FOR reflective surfaces, THE converter SHALL set material='mirror' or equivalent optiland property
10.4 THE converter SHALL support:
    - Standard surfaces (spherical)
    - Even asphere surfaces
    - Flat surfaces (infinite radius)

10.5 THE converter SHALL NOT require intermediate data structures beyond GlobalSurfaceDefinition
10.6 THE converter SHALL provide a single method to build complete optiland Optic from surface list

### Requirement 11: 单位约定

**User Story:** As a developer, I want consistent units throughout the conversion, so that there are no scaling errors.

#### Acceptance Criteria

11.1 ALL length values (position, thickness, radius, semi_aperture) SHALL be in millimeters
11.2 ALL internal angle calculations SHALL use radians
11.3 WHEN reading from ZMX files, angles in degrees SHALL be converted to radians
11.4 THE coordinate system SHALL be right-handed with Z as initial optical axis
11.5 WHEN interfacing with optiland, units SHALL match optiland's conventions (mm for lengths)


### Requirement 12: 验证测试

**User Story:** As a developer, I want validation tests against known configurations, so that I can verify the implementation is correct.

#### Acceptance Criteria

12.1 FOR a single coordinate break with tilt_x=45°, THE Z axis SHALL rotate 45° in the YZ plane
12.2 FOR two consecutive coordinate breaks (45° each), THE transformations SHALL accumulate correctly
12.3 FOR a coordinate break followed by a mirror, THE mirror position SHALL be at the transformed origin
12.4 FOR Order=0 vs Order=1 with same parameters, THE results SHALL differ as specified
12.5 THE converted system ray tracing results SHALL match Zemax within numerical tolerance (< 1e-6 mm position, < 1e-6 direction cosines)

### Requirement 13: 错误处理

**User Story:** As a developer, I want clear error messages for invalid configurations, so that I can diagnose problems quickly.

#### Acceptance Criteria

13.1 IF surface index is negative or out of range, THE system SHALL raise ValueError with index value
13.2 IF radius is exactly zero (invalid), THE system SHALL raise ValueError
13.3 IF Order parameter is not 0 or 1, THE system SHALL raise ValueError
13.4 IF rotation matrix becomes singular (numerical error), THE system SHALL re-orthogonalize and log warning
13.5 ALL errors SHALL include context (surface index, parameter name, value)

### Requirement 14: 代码架构

**User Story:** As a developer, I want a clean code architecture, so that the implementation is maintainable and testable.

#### Acceptance Criteria

14.1 THE implementation SHALL be in a single module: `src/sequential_system/coordinate_system.py`
14.2 THE module SHALL contain:
    - CurrentCoordinateSystem dataclass
    - CoordinateBreakProcessor class with static methods
    - GlobalSurfaceDefinition dataclass
    - SurfaceTraversalAlgorithm class

14.3 THE existing `coordinate_tracking.py` SHALL be preserved for optical axis tracking (different purpose)
14.4 THE new module SHALL NOT depend on existing SequentialOpticalSystem internals
14.5 THE module SHALL be independently testable with unit tests and property-based tests

