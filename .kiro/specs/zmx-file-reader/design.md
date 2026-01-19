# Design Document: ZMX File Reader

## Overview

本设计文档描述 ZMX 文件读取功能的技术实现方案。该功能允许用户从 Zemax 序列模式的 .zmx 文件中读取光学系统定义，并将其转换为本项目的序列光学系统格式。

设计参考了 optiland 库的 ZMX 文件读取实现（`optiland/fileio/zemax_handler.py`），但针对本项目的需求进行了简化和定制：
- 专注于光学面形、间距、位置顺序、偏移和倾斜参数
- 输出为项目的 `OpticalElement` 类型（而非 optiland 的 `Optic` 对象）
- 提供代码生成功能，方便用户编辑和自定义

## Architecture

### 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        ZMX File Reader                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────────────┐    │
│  │  ZmxParser  │───▶│ ZmxDataModel│───▶│ ElementConverter │    │
│  └─────────────┘    └─────────────┘    └──────────────────┘    │
│        │                   │                    │               │
│        ▼                   ▼                    ▼               │
│   读取 .zmx 文件      结构化数据模型      OpticalElement 列表    │
│   解析操作符                                                    │
│   提取表面数据                              ┌──────────────┐    │
│                                            │ CodeGenerator │    │
│                                            └──────────────┘    │
│                                                   │             │
│                                                   ▼             │
│                                            Python 源代码        │
└─────────────────────────────────────────────────────────────────┘
```

### 数据流

```
.zmx 文件 ──▶ ZmxParser.parse() ──▶ ZmxDataModel
                                         │
                                         ▼
                              ElementConverter.convert()
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
            List[OpticalElement]   CodeGenerator      SequentialOpticalSystem
                                   .generate_code()        (可选)
                                         │
                                         ▼
                                   Python 源代码字符串
```

### 模块位置

新模块将放置在 `src/sequential_system/` 目录下：

```
src/sequential_system/
├── __init__.py              # 更新导出
├── zmx_parser.py            # ZMX 文件解析器（新增）
├── zmx_converter.py         # 元件转换器和代码生成器（新增）
├── system.py                # 现有系统类
└── ...
```

## Components and Interfaces

### 1. ZmxParser 类

负责读取和解析 .zmx 文件。

```python
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np


@dataclass
class ZmxSurfaceData:
    """ZMX 表面数据结构"""
    index: int                          # 表面索引
    surface_type: str                   # 表面类型：standard, coordinate_break, even_asphere
    radius: float = np.inf              # 曲率半径 (mm)
    thickness: float = 0.0              # 厚度/间距 (mm)
    conic: float = 0.0                  # 圆锥常数
    material: str = "air"               # 材料
    is_mirror: bool = False             # 是否为反射镜
    is_stop: bool = False               # 是否为光阑
    semi_diameter: float = 0.0          # 半口径 (mm)
    # 坐标断点参数
    decenter_x: float = 0.0             # X 偏心 (mm)
    decenter_y: float = 0.0             # Y 偏心 (mm)
    tilt_x_deg: float = 0.0             # X 轴旋转 (度)
    tilt_y_deg: float = 0.0             # Y 轴旋转 (度)
    tilt_z_deg: float = 0.0             # Z 轴旋转 (度)
    # 非球面系数
    asphere_coeffs: List[float] = field(default_factory=list)
    # 原始注释
    comment: str = ""


@dataclass
class ZmxDataModel:
    """ZMX 数据模型"""
    surfaces: Dict[int, ZmxSurfaceData] = field(default_factory=dict)
    wavelengths: List[float] = field(default_factory=list)  # 波长列表 (μm)
    primary_wavelength_index: int = 0
    entrance_pupil_diameter: float = 0.0  # 入瞳直径 (mm)
    
    def get_surface(self, index: int) -> Optional[ZmxSurfaceData]:
        """获取指定索引的表面数据"""
        return self.surfaces.get(index)
    
    def get_mirror_surfaces(self) -> List[ZmxSurfaceData]:
        """获取所有反射镜表面"""
        return [s for s in self.surfaces.values() if s.is_mirror]


class ZmxParser:
    """ZMX 文件解析器
    
    解析 Zemax .zmx 文件并提取光学系统数据。
    
    参数:
        filepath: .zmx 文件路径
    
    示例:
        >>> parser = ZmxParser("system.zmx")
        >>> data_model = parser.parse()
        >>> print(f"共 {len(data_model.surfaces)} 个表面")
    """
    
    SUPPORTED_ENCODINGS = ["utf-16", "utf-8", "iso-8859-1"]
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._data_model = ZmxDataModel()
        self._current_surface_index = -1
        self._current_surface: Optional[ZmxSurfaceData] = None
    
    def parse(self) -> ZmxDataModel:
        """解析 ZMX 文件
        
        返回:
            ZmxDataModel: 解析后的数据模型
        
        异常:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式错误或编码不支持
        """
        pass
    
    def _try_read_file(self) -> List[str]:
        """尝试使用不同编码读取文件"""
        pass
    
    def _parse_line(self, line: str) -> None:
        """解析单行数据"""
        pass
    
    def _parse_surface(self, data: List[str]) -> None:
        """解析 SURF 操作符"""
        pass
    
    def _parse_type(self, data: List[str]) -> None:
        """解析 TYPE 操作符"""
        pass
    
    def _parse_curv(self, data: List[str]) -> None:
        """解析 CURV 操作符（曲率）"""
        pass
    
    def _parse_disz(self, data: List[str]) -> None:
        """解析 DISZ 操作符（厚度）"""
        pass
    
    def _parse_coni(self, data: List[str]) -> None:
        """解析 CONI 操作符（圆锥常数）"""
        pass
    
    def _parse_glas(self, data: List[str]) -> None:
        """解析 GLAS 操作符（材料）"""
        pass
    
    def _parse_parm(self, data: List[str]) -> None:
        """解析 PARM 操作符（参数）"""
        pass
    
    def _parse_diam(self, data: List[str]) -> None:
        """解析 DIAM 操作符（直径）"""
        pass
    
    def _parse_stop(self, data: List[str]) -> None:
        """解析 STOP 操作符"""
        pass
    
    def _parse_comm(self, data: List[str]) -> None:
        """解析 COMM 操作符（注释）"""
        pass
    
    def _parse_enpd(self, data: List[str]) -> None:
        """解析 ENPD 操作符（入瞳直径）"""
        pass
    
    def _parse_wavm(self, data: List[str]) -> None:
        """解析 WAVM 操作符（波长）"""
        pass
    
    def _parse_mode(self, data: List[str]) -> None:
        """解析 MODE 操作符"""
        pass
    
    def _finalize_current_surface(self) -> None:
        """完成当前表面的解析"""
        pass
```

### 2. ElementConverter 类

将 ZmxDataModel 转换为项目的 OpticalElement 列表。

```python
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class ConvertedElement:
    """转换后的元件数据"""
    element: 'OpticalElement'           # 光学元件对象
    zmx_surface_index: int              # 原始 ZMX 表面索引
    zmx_comment: str                    # 原始注释
    is_fold_mirror: bool                # 是否为折叠镜
    fold_angle_deg: float               # 折叠角度（度）


class ElementConverter:
    """ZMX 数据到 OpticalElement 的转换器
    
    将 ZmxDataModel 转换为项目的 OpticalElement 列表。
    
    参数:
        data_model: ZmxDataModel 对象
    
    示例:
        >>> converter = ElementConverter(data_model)
        >>> elements = converter.convert()
        >>> code = converter.generate_code()
        >>> print(code)
    """
    
    # 折叠镜角度阈值（度）
    FOLD_ANGLE_THRESHOLD = 5.0
    
    def __init__(self, data_model: ZmxDataModel):
        self.data_model = data_model
        self._converted_elements: List[ConvertedElement] = []
        self._accumulated_transform = CoordinateTransform()
    
    def convert(self) -> List['OpticalElement']:
        """执行转换
        
        返回:
            List[OpticalElement]: 转换后的光学元件列表
        """
        pass
    
    def get_converted_elements(self) -> List[ConvertedElement]:
        """获取带元数据的转换结果"""
        return self._converted_elements
    
    def generate_code(self, include_imports: bool = True) -> str:
        """生成 Python 源代码
        
        参数:
            include_imports: 是否包含 import 语句
        
        返回:
            str: Python 源代码字符串
        """
        pass
    
    def _process_surfaces(self) -> None:
        """处理所有表面"""
        pass
    
    def _process_coordinate_break(self, surface: ZmxSurfaceData) -> None:
        """处理坐标断点"""
        pass
    
    def _process_mirror_surface(self, surface: ZmxSurfaceData) -> None:
        """处理反射镜表面"""
        pass
    
    def _process_refractive_surface(self, surface: ZmxSurfaceData) -> None:
        """处理折射表面"""
        pass
    
    def _create_mirror_element(
        self,
        surface: ZmxSurfaceData,
        thickness: float,
        tilt_x: float,
        tilt_y: float,
        decenter_x: float,
        decenter_y: float,
    ) -> 'OpticalElement':
        """创建反射镜元件"""
        pass
    
    def _is_fold_mirror(self, tilt_x_deg: float, tilt_y_deg: float) -> bool:
        """判断是否为折叠镜"""
        pass
    
    def _calculate_thickness_after_reflection(
        self,
        current_index: int,
    ) -> float:
        """计算反射后的传播距离"""
        pass


@dataclass
class CoordinateTransform:
    """坐标变换累积器"""
    decenter_x: float = 0.0
    decenter_y: float = 0.0
    decenter_z: float = 0.0
    tilt_x_rad: float = 0.0
    tilt_y_rad: float = 0.0
    tilt_z_rad: float = 0.0
    
    def apply_coordinate_break(
        self,
        dx: float,
        dy: float,
        dz: float,
        rx_deg: float,
        ry_deg: float,
        rz_deg: float,
    ) -> None:
        """应用坐标断点变换"""
        pass
    
    def reset(self) -> None:
        """重置变换"""
        pass
```

### 3. CodeGenerator 类

生成可复制的 Python 代码。

```python
class CodeGenerator:
    """Python 代码生成器
    
    从转换后的元件列表生成 Python 源代码。
    """
    
    INDENT = "    "  # 4 空格缩进
    
    def __init__(self, converted_elements: List[ConvertedElement]):
        self.elements = converted_elements
    
    def generate(self, include_imports: bool = True) -> str:
        """生成完整的 Python 代码
        
        参数:
            include_imports: 是否包含 import 语句
        
        返回:
            str: Python 源代码
        """
        pass
    
    def _generate_imports(self) -> str:
        """生成 import 语句"""
        pass
    
    def _generate_element_code(self, elem: ConvertedElement) -> str:
        """生成单个元件的代码"""
        pass
    
    def _format_float(self, value: float, precision: int = 6) -> str:
        """格式化浮点数"""
        pass
```

### 4. 便捷函数

提供简单的 API 入口。

```python
def load_zmx_file(filepath: str) -> List['OpticalElement']:
    """从 ZMX 文件加载光学元件
    
    参数:
        filepath: .zmx 文件路径
    
    返回:
        List[OpticalElement]: 光学元件列表
    
    示例:
        >>> elements = load_zmx_file("system.zmx")
        >>> for elem in elements:
        ...     print(elem)
    """
    parser = ZmxParser(filepath)
    data_model = parser.parse()
    converter = ElementConverter(data_model)
    return converter.convert()


def load_zmx_and_generate_code(filepath: str) -> Tuple[List['OpticalElement'], str]:
    """从 ZMX 文件加载光学元件并生成代码
    
    参数:
        filepath: .zmx 文件路径
    
    返回:
        Tuple[List[OpticalElement], str]: (元件列表, Python 源代码)
    
    示例:
        >>> elements, code = load_zmx_and_generate_code("system.zmx")
        >>> print(code)
    """
    parser = ZmxParser(filepath)
    data_model = parser.parse()
    converter = ElementConverter(data_model)
    elements = converter.convert()
    code = converter.generate_code()
    return elements, code
```

## Data Models

### ZMX 文件格式

ZMX 文件是 Zemax 的原生格式，使用操作符-数据的行格式：

```
OPERAND [data1] [data2] ...
```

关键操作符：

| 操作符 | 说明 | 示例 |
|--------|------|------|
| MODE | 模式（SEQ=序列） | `MODE SEQ` |
| ENPD | 入瞳直径 | `ENPD 20` |
| WAVM | 波长 | `WAVM 1 0.55 1` |
| SURF | 表面开始 | `SURF 1` |
| TYPE | 表面类型 | `TYPE STANDARD` |
| CURV | 曲率 | `CURV 0.01 0 0 0 0` |
| DISZ | 厚度 | `DISZ 10` |
| CONI | 圆锥常数 | `CONI -1` |
| GLAS | 材料 | `GLAS MIRROR 0 0 1.5 40` |
| DIAM | 直径 | `DIAM 10 0 0 0 1` |
| STOP | 光阑标记 | `STOP` |
| PARM | 参数 | `PARM 1 0` |
| COMM | 注释 | `COMM M1` |

### 坐标断点（COORDBRK）参数映射

```
PARM 1 → decenter_x (mm)
PARM 2 → decenter_y (mm)
PARM 3 → tilt_x (度，绕 X 轴旋转)
PARM 4 → tilt_y (度，绕 Y 轴旋转)
PARM 5 → tilt_z (度，绕 Z 轴旋转)
PARM 6 → order (旋转顺序标志)
DISZ   → thickness (mm，沿当前 Z 轴的位移)
```

### 折叠镜序列模式

典型的折叠镜在 ZMX 中的定义模式：

```
SURF N      # 坐标断点（反射前）
  TYPE COORDBRK
  PARM 3 45         # 45 度倾斜
  DISZ 0

SURF N+1    # 反射镜表面
  TYPE STANDARD
  GLAS MIRROR
  DISZ 0

SURF N+2    # 坐标断点（反射后）
  TYPE COORDBRK
  PARM 3 45         # 匹配的 45 度倾斜
  DISZ -50          # 负厚度表示反射方向传播
```

### 元件类型映射

| ZMX 条件 | 项目元件类型 |
|----------|--------------|
| GLAS MIRROR + radius=∞ | FlatMirror |
| GLAS MIRROR + conic=-1 | ParabolicMirror |
| GLAS MIRROR + finite radius | SphericalMirror |
| 折射材料 | ThinLens（简化处理） |



## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Surface Data Extraction Completeness

*For any* valid ZMX file containing STANDARD surfaces, parsing then accessing the surface data SHALL return all expected parameters (radius, thickness, conic, material, is_mirror, is_stop, semi_diameter) with correct values.

**Validates: Requirements 2.1, 2.3, 2.4, 2.5, 2.6**

### Property 2: Coordinate Break Parameter Extraction

*For any* COORDBRK surface in a ZMX file, parsing SHALL extract all coordinate break parameters (decenter_x, decenter_y, tilt_x, tilt_y, tilt_z, thickness) and convert tilt values from degrees to radians correctly.

**Validates: Requirements 3.1, 3.2, 3.3**

### Property 3: Mirror Type Classification

*For any* mirror surface (GLAS MIRROR), the Element_Converter SHALL create the correct element type:
- FlatMirror when radius is infinite
- ParabolicMirror when conic = -1
- SphericalMirror otherwise

**Validates: Requirements 5.1, 5.2, 5.3**

### Property 4: Fold Mirror Detection and Configuration

*For any* mirror with tilt angle ≥ 5 degrees, the Element_Converter SHALL set is_fold=True and correctly apply the tilt angle. For mirrors with tilt < 5 degrees, is_fold SHALL be False.

**Validates: Requirements 5.7, 5.8, 7.3**

### Property 5: Fold Mirror Sequence Thickness Calculation

*For any* fold mirror sequence (COORDBRK + MIRROR + COORDBRK), the Element_Converter SHALL correctly calculate the propagation distance by interpreting negative thickness in post-reflection COORDBRK as propagation in the reflected direction.

**Validates: Requirements 6.1, 6.3, 6.4**

### Property 6: Element Sequence Ordering

*For any* ZMX file, the converted OpticalElement list SHALL maintain the same optical sequence order as the original ZMX surfaces, with correct thickness values between elements.

**Validates: Requirements 7.1, 7.2, 7.4, 7.5**

### Property 7: Code Generation Round-Trip

*For any* set of converted OpticalElements, generating code then executing that code SHALL produce an equivalent set of OpticalElements with identical parameters.

**Validates: Requirements 9.1, 9.2, 9.3**

### Property 8: Code Generation Completeness

*For any* generated code, it SHALL include:
- All necessary import statements
- Comments with original ZMX surface indices
- Fold angle comments for fold mirrors
- Proper Python formatting (indentation, line breaks)

**Validates: Requirements 9.4, 9.5, 9.6, 9.7**

### Property 9: Encoding Robustness

*For any* valid ZMX file content, the parser SHALL successfully decode the file regardless of whether it uses UTF-16, UTF-8, or ISO-8859-1 encoding.

**Validates: Requirements 1.2**

### Property 10: Default Value Application

*For any* surface with missing optional parameters, the parser SHALL apply sensible defaults (radius=inf, thickness=0, conic=0) without raising errors.

**Validates: Requirements 8.2**

## Error Handling

### 异常类型

```python
class ZmxParseError(Exception):
    """ZMX 文件解析错误"""
    def __init__(self, message: str, line_number: int = None, line_content: str = None):
        self.line_number = line_number
        self.line_content = line_content
        super().__init__(self._format_message(message))
    
    def _format_message(self, message: str) -> str:
        if self.line_number is not None:
            return f"第 {self.line_number} 行解析错误: {message}\n内容: {self.line_content}"
        return f"ZMX 解析错误: {message}"


class ZmxUnsupportedError(ZmxParseError):
    """不支持的 ZMX 特性"""
    pass


class ZmxConversionError(Exception):
    """ZMX 到 OpticalElement 转换错误"""
    pass
```

### 错误处理策略

| 错误情况 | 处理方式 |
|----------|----------|
| 文件不存在 | 抛出 FileNotFoundError |
| 编码不支持 | 尝试所有支持的编码，全部失败后抛出 ValueError |
| 非序列模式 | 抛出 ZmxUnsupportedError |
| 不支持的表面类型 | 抛出 ZmxUnsupportedError，包含类型名称 |
| 缺少必需参数 | 使用默认值，记录警告 |
| 无效的数值 | 抛出 ZmxParseError，包含行号和内容 |

## Testing Strategy

### 单元测试

使用 pytest 框架进行单元测试：

```python
# tests/test_zmx_parser.py

import pytest
from sequential_system.zmx_parser import ZmxParser, ZmxDataModel

class TestZmxParser:
    """ZMX 解析器单元测试"""
    
    def test_parse_standard_surface(self):
        """测试标准表面解析"""
        pass
    
    def test_parse_coordinate_break(self):
        """测试坐标断点解析"""
        pass
    
    def test_parse_mirror_surface(self):
        """测试反射镜表面解析"""
        pass
    
    def test_encoding_utf16(self):
        """测试 UTF-16 编码"""
        pass
    
    def test_encoding_utf8(self):
        """测试 UTF-8 编码"""
        pass
    
    def test_file_not_found(self):
        """测试文件不存在错误"""
        pass
    
    def test_unsupported_mode(self):
        """测试不支持的模式错误"""
        pass
```

### 属性基测试

使用 hypothesis 进行属性基测试：

```python
# tests/property/test_zmx_properties.py

from hypothesis import given, strategies as st
import pytest

class TestZmxProperties:
    """ZMX 解析器属性基测试"""
    
    @given(
        radius=st.floats(min_value=10.0, max_value=1000.0),
        thickness=st.floats(min_value=0.0, max_value=100.0),
        conic=st.floats(min_value=-2.0, max_value=0.0),
    )
    def test_surface_data_extraction(self, radius, thickness, conic):
        """
        **Feature: zmx-file-reader, Property 1: Surface Data Extraction Completeness**
        **Validates: Requirements 2.1, 2.3, 2.4, 2.5, 2.6**
        
        测试表面数据提取的完整性
        """
        pass
    
    @given(
        tilt_deg=st.floats(min_value=-90.0, max_value=90.0),
    )
    def test_tilt_conversion(self, tilt_deg):
        """
        **Feature: zmx-file-reader, Property 2: Coordinate Break Parameter Extraction**
        **Validates: Requirements 3.1, 3.2, 3.3**
        
        测试倾斜角度从度到弧度的转换
        """
        import numpy as np
        expected_rad = np.deg2rad(tilt_deg)
        # 验证转换正确性
        pass
    
    @given(
        radius=st.one_of(
            st.just(float('inf')),
            st.floats(min_value=10.0, max_value=1000.0),
        ),
        conic=st.one_of(
            st.just(-1.0),
            st.floats(min_value=-0.5, max_value=0.0),
        ),
    )
    def test_mirror_type_classification(self, radius, conic):
        """
        **Feature: zmx-file-reader, Property 3: Mirror Type Classification**
        **Validates: Requirements 5.1, 5.2, 5.3**
        
        测试反射镜类型分类
        """
        pass
    
    @given(
        tilt_deg=st.floats(min_value=0.0, max_value=90.0),
    )
    def test_fold_mirror_detection(self, tilt_deg):
        """
        **Feature: zmx-file-reader, Property 4: Fold Mirror Detection and Configuration**
        **Validates: Requirements 5.7, 5.8, 7.3**
        
        测试折叠镜检测
        """
        is_fold = tilt_deg >= 5.0
        # 验证 is_fold 标志正确性
        pass
```

### 集成测试

使用实际的 ZMX 测试文件：

```python
# tests/integration/test_zmx_integration.py

import pytest
from pathlib import Path

class TestZmxIntegration:
    """ZMX 文件集成测试"""
    
    @pytest.fixture
    def zmx_test_dir(self):
        return Path("optiland-master/tests/zemax_files")
    
    def test_complicated_fold_mirrors(self, zmx_test_dir):
        """
        测试 complicated_fold_mirrors_setup_v2.zmx
        **Validates: Requirements 10.1, 10.2, 10.3**
        """
        pass
    
    def test_one_mirror_up_45deg(self, zmx_test_dir):
        """
        测试 one_mirror_up_45deg.zmx
        **Validates: Requirements 10.4**
        """
        pass
    
    def test_simple_fold_mirror_up(self, zmx_test_dir):
        """
        测试 simple_fold_mirror_up.zmx
        """
        pass
    
    def test_all_zmx_files_produce_valid_elements(self, zmx_test_dir):
        """
        测试所有 ZMX 文件都能生成有效元件
        **Validates: Requirements 10.5**
        """
        pass
```

### 测试配置

```python
# tests/conftest.py

import pytest
import tempfile
from pathlib import Path

@pytest.fixture
def temp_zmx_file():
    """创建临时 ZMX 文件的 fixture"""
    def _create_zmx(content: str, encoding: str = "utf-16"):
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.zmx',
            encoding=encoding,
            delete=False,
        ) as f:
            f.write(content)
            return Path(f.name)
    return _create_zmx

@pytest.fixture
def minimal_zmx_content():
    """最小有效 ZMX 文件内容"""
    return """MODE SEQ
ENPD 20
WAVM 1 0.55 1
SURF 0
  TYPE STANDARD
  DISZ INFINITY
SURF 1
  TYPE STANDARD
  DISZ 0
"""
```
