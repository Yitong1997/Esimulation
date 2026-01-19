# Requirements Document

## Introduction

本文档定义了 ZMX 文件读取功能的需求规格。该功能允许用户从 Zemax 序列模式的 .zmx 文件中读取光学系统定义，并将其转换为本项目的序列光学系统格式（SequentialOpticalSystem）。

本功能参考 optiland 库的 ZMX 文件读取实现，但专注于提取光学面形、间距、位置顺序、偏移和倾斜参数，暂不处理入射光、镀膜等高级功能。

## Glossary

- **ZMX_Parser**: ZMX 文件解析器，负责读取和解析 .zmx 文件格式
- **Surface_Data**: 光学表面数据结构，包含面形、材料、间距等信息
- **Coordinate_Break**: Zemax 中的坐标断点面，用于定义偏移和倾斜
- **Sequential_System**: 序列光学系统，本项目的光学系统定义格式
- **Element_Converter**: 元件转换器，将 ZMX 表面数据转换为项目光学元件

## Requirements

### Requirement 1: ZMX 文件解析

**User Story:** As a 光学工程师, I want to 读取 Zemax .zmx 文件, so that I can 在本仿真系统中使用已有的光学设计。

#### Acceptance Criteria

1. WHEN a valid .zmx file path is provided, THE ZMX_Parser SHALL read and parse the file content
2. WHEN the .zmx file uses UTF-16, UTF-8, or ISO-8859-1 encoding, THE ZMX_Parser SHALL correctly decode the file
3. WHEN the .zmx file contains MODE SEQ, THE ZMX_Parser SHALL accept the file as valid sequential mode
4. IF the .zmx file contains MODE other than SEQ, THEN THE ZMX_Parser SHALL raise an error indicating only sequential mode is supported
5. IF the .zmx file path does not exist, THEN THE ZMX_Parser SHALL raise a FileNotFoundError with descriptive message

### Requirement 2: 光学表面数据提取

**User Story:** As a 光学工程师, I want to 提取光学表面的面形参数, so that I can 准确重建光学系统。

#### Acceptance Criteria

1. WHEN parsing a STANDARD surface type, THE ZMX_Parser SHALL extract radius (from CURV), thickness (from DISZ), and conic constant (from CONI)
2. WHEN parsing an EVENASPH surface type, THE ZMX_Parser SHALL extract radius, thickness, conic, and even asphere coefficients (from PARM)
3. WHEN parsing a surface with GLAS MIRROR, THE ZMX_Parser SHALL mark the surface as reflective
4. WHEN parsing a surface with glass material name, THE ZMX_Parser SHALL store the material name
5. WHEN parsing a surface with STOP flag, THE ZMX_Parser SHALL mark the surface as aperture stop
6. WHEN parsing a surface with DIAM, THE ZMX_Parser SHALL extract the semi-diameter value

### Requirement 3: 坐标断点处理

**User Story:** As a 光学工程师, I want to 正确处理坐标断点（Coordinate Break）, so that I can 支持折叠光路和倾斜元件。

#### Acceptance Criteria

1. WHEN parsing a COORDBRK surface type, THE ZMX_Parser SHALL extract decenter_x (PARM 1), decenter_y (PARM 2), tilt_x (PARM 3), tilt_y (PARM 4), tilt_z (PARM 5)
2. WHEN a COORDBRK surface has non-zero tilt values, THE ZMX_Parser SHALL convert degrees to radians
3. WHEN a COORDBRK surface has thickness (DISZ), THE ZMX_Parser SHALL include it in coordinate transformation
4. WHEN multiple COORDBRK surfaces are chained, THE ZMX_Parser SHALL accumulate the coordinate transformations correctly
5. WHEN a COORDBRK precedes a mirror surface with 45-degree tilt, THE ZMX_Parser SHALL identify this as a fold mirror configuration
6. WHEN a COORDBRK follows a mirror surface with matching tilt, THE ZMX_Parser SHALL recognize this as the post-reflection coordinate restoration

### Requirement 4: 数据模型构建

**User Story:** As a 开发者, I want to 将解析的数据组织成结构化模型, so that I can 方便地进行后续转换。

#### Acceptance Criteria

1. THE ZMX_Parser SHALL create a ZmxDataModel containing surfaces dictionary, wavelengths list, and aperture information
2. FOR ALL parsed surfaces, THE ZMX_Parser SHALL assign sequential index starting from 0
3. WHEN parsing wavelengths (WAVM), THE ZMX_Parser SHALL extract wavelength values in micrometers
4. WHEN parsing aperture (ENPD), THE ZMX_Parser SHALL extract entrance pupil diameter in millimeters

### Requirement 5: 元件转换

**User Story:** As a 开发者, I want to 将 ZMX 表面数据转换为项目光学元件, so that I can 在 SequentialOpticalSystem 中使用。

#### Acceptance Criteria

1. WHEN converting a mirror surface (GLAS MIRROR) with finite radius, THE Element_Converter SHALL create a SphericalMirror element
2. WHEN converting a mirror surface with conic = -1, THE Element_Converter SHALL create a ParabolicMirror element
3. WHEN converting a mirror surface with infinite radius, THE Element_Converter SHALL create a FlatMirror element
4. WHEN converting a refractive surface pair, THE Element_Converter SHALL create appropriate lens elements
5. WHEN a surface has preceding COORDBRK, THE Element_Converter SHALL apply tilt_x and tilt_y to the element
6. WHEN a surface has preceding COORDBRK with decenter, THE Element_Converter SHALL apply decenter_x and decenter_y to the element
7. WHEN a mirror has 45-degree tilt (±π/4 radians), THE Element_Converter SHALL identify it as a fold mirror and set is_fold=True
8. WHEN a mirror has small tilt (less than 5 degrees), THE Element_Converter SHALL treat it as misalignment and set is_fold=False

### Requirement 6: 折叠光路处理

**User Story:** As a 光学工程师, I want to 正确处理折叠光路中的反射镜序列, so that I can 准确模拟复杂的折叠光学系统。

#### Acceptance Criteria

1. WHEN a fold mirror sequence is detected (COORDBRK + MIRROR + COORDBRK), THE Element_Converter SHALL correctly calculate the propagation distance after reflection
2. WHEN multiple fold mirrors exist in sequence, THE Element_Converter SHALL track the cumulative coordinate transformation
3. WHEN a fold mirror has negative thickness in post-reflection COORDBRK, THE Element_Converter SHALL interpret this as propagation in the reflected direction
4. WHEN converting fold mirrors, THE Element_Converter SHALL preserve the correct optical path length between elements
5. WHEN the fold angle is exactly 45 degrees, THE Element_Converter SHALL set tilt_x or tilt_y to π/4 radians accordingly

### Requirement 7: 系统构建

**User Story:** As a 光学工程师, I want to 从 ZMX 文件直接创建 SequentialOpticalSystem, so that I can 快速进行仿真。

#### Acceptance Criteria

1. THE Element_Converter SHALL produce a list of OpticalElement objects in correct sequence order
2. WHEN building the system, THE Element_Converter SHALL set thickness for each element based on DISZ values and coordinate transformations
3. WHEN the system contains fold mirrors (45-degree tilted mirrors), THE Element_Converter SHALL set is_fold=True for those elements
4. THE Element_Converter SHALL preserve semi_aperture from DIAM values for each element
5. WHEN the system contains multiple fold mirrors, THE Element_Converter SHALL correctly chain the elements with proper thickness values

### Requirement 8: 错误处理

**User Story:** As a 用户, I want to 获得清晰的错误信息, so that I can 诊断和修复问题。

#### Acceptance Criteria

1. IF an unsupported surface type is encountered, THEN THE ZMX_Parser SHALL raise ValueError with the surface type name
2. IF required surface parameters are missing, THEN THE ZMX_Parser SHALL use sensible defaults (radius=inf, thickness=0, conic=0)
3. IF the file cannot be decoded with any supported encoding, THEN THE ZMX_Parser SHALL raise ValueError with encoding error details
4. WHEN an error occurs during parsing, THE ZMX_Parser SHALL include the line number and content in the error message

### Requirement 9: 代码生成

**User Story:** As a 光学工程师, I want to 生成可复制的 Python 代码, so that I can 方便地编辑和自定义光学系统定义。

#### Acceptance Criteria

1. THE Element_Converter SHALL provide a generate_code() method that returns Python source code as a string
2. WHEN generating code, THE Element_Converter SHALL produce valid Python code that creates the equivalent SequentialOpticalSystem
3. WHEN generating code, THE Element_Converter SHALL include all element parameters (thickness, semi_aperture, tilt_x, tilt_y, decenter_x, decenter_y, is_fold)
4. WHEN generating code, THE Element_Converter SHALL include comments indicating the original ZMX surface index for each element
5. WHEN generating code for fold mirrors, THE Element_Converter SHALL clearly indicate the fold angle in comments
6. THE generated code SHALL be properly formatted with correct indentation and line breaks
7. THE generated code SHALL include necessary import statements at the top

### Requirement 10: 测试验证

**User Story:** As a 开发者, I want to 验证 ZMX 读取功能的正确性, so that I can 确保与 Zemax 的兼容性。

#### Acceptance Criteria

1. WHEN loading complicated_fold_mirrors_setup_v2.zmx, THE ZMX_Parser SHALL correctly identify all mirror surfaces and their types
2. WHEN loading complicated_fold_mirrors_setup_v2.zmx, THE ZMX_Parser SHALL correctly extract all coordinate break transformations including tilts and decenters
3. WHEN loading complicated_fold_mirrors_setup_v2.zmx, THE Element_Converter SHALL produce correct fold mirror sequence with proper is_fold flags
4. WHEN loading one_mirror_up_45deg.zmx, THE ZMX_Parser SHALL produce a system with one 45-degree fold mirror
5. FOR ALL test ZMX files, THE Element_Converter SHALL produce valid OpticalElement objects that can be added to SequentialOpticalSystem
6. WHEN loading a ZMX file with multiple fold mirrors, THE Element_Converter SHALL correctly calculate the optical path length through the folded system
