"""
ZMX 文件解析器

本模块提供 Zemax .zmx 文件的解析功能，将 ZMX 文件中的光学系统定义
转换为结构化的数据模型，以便后续转换为项目的 OpticalElement 类型。

主要类：
- ZmxSurfaceData: 单个光学表面的数据结构
- ZmxDataModel: 完整 ZMX 文件的数据模型
- ZmxParser: ZMX 文件解析器

支持的 ZMX 特性：
- 序列模式（MODE SEQ）
- 标准表面（STANDARD）
- 坐标断点（COORDBRK）
- 偶次非球面（EVENASPH）
- 反射镜（GLAS MIRROR）
- 波长和入瞳直径

使用示例：
    >>> from sequential_system.zmx_parser import ZmxParser
    >>> parser = ZmxParser("system.zmx")
    >>> data_model = parser.parse()
    >>> print(f"共 {len(data_model.surfaces)} 个表面")
    >>> for idx, surface in data_model.surfaces.items():
    ...     print(f"  表面 {idx}: {surface.surface_type}")

作者：混合光学仿真项目
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


# =============================================================================
# 异常类定义
# =============================================================================

class ZmxParseError(Exception):
    """ZMX 文件解析错误
    
    当解析 ZMX 文件时遇到格式错误、无效数据或其他解析问题时抛出此异常。
    
    属性:
        line_number: 发生错误的行号（从 1 开始），如果无法确定则为 None
        line_content: 发生错误的行内容，如果无法确定则为 None
        message: 错误描述信息
    
    示例:
        >>> raise ZmxParseError("无效的曲率值", line_number=42, line_content="CURV abc")
        ZmxParseError: 第 42 行解析错误: 无效的曲率值
        内容: CURV abc
    """
    
    def __init__(
        self, 
        message: str, 
        line_number: Optional[int] = None, 
        line_content: Optional[str] = None
    ):
        """初始化 ZmxParseError
        
        参数:
            message: 错误描述信息
            line_number: 发生错误的行号（从 1 开始）
            line_content: 发生错误的行内容
        """
        self.line_number = line_number
        self.line_content = line_content
        self.message = message
        super().__init__(self._format_message(message))
    
    def _format_message(self, message: str) -> str:
        """格式化错误信息
        
        根据是否有行号和行内容信息，生成详细的错误消息。
        
        参数:
            message: 原始错误描述
        
        返回:
            格式化后的完整错误消息
        """
        if self.line_number is not None:
            formatted = f"第 {self.line_number} 行解析错误: {message}"
            if self.line_content is not None:
                formatted += f"\n内容: {self.line_content}"
            return formatted
        return f"ZMX 解析错误: {message}"


class ZmxUnsupportedError(ZmxParseError):
    """不支持的 ZMX 特性错误
    
    当 ZMX 文件中包含本解析器不支持的特性时抛出此异常。
    例如：非序列模式、不支持的表面类型等。
    
    继承自 ZmxParseError，具有相同的属性和行为。
    
    示例:
        >>> raise ZmxUnsupportedError(
        ...     "不支持的表面类型: TOROIDAL",
        ...     line_number=15,
        ...     line_content="TYPE TOROIDAL"
        ... )
        ZmxUnsupportedError: 第 15 行解析错误: 不支持的表面类型: TOROIDAL
        内容: TYPE TOROIDAL
        
        >>> raise ZmxUnsupportedError("不支持非序列模式")
        ZmxUnsupportedError: ZMX 解析错误: 不支持非序列模式
    """
    pass


class ZmxConversionError(Exception):
    """ZMX 到 OpticalElement 转换错误
    
    当将 ZMX 数据模型转换为项目的 OpticalElement 类型时发生错误时抛出此异常。
    例如：无法确定元件类型、参数不兼容等。
    
    属性:
        message: 错误描述信息
        surface_index: 发生错误的 ZMX 表面索引（可选）
        surface_type: 发生错误的表面类型（可选）
    
    示例:
        >>> raise ZmxConversionError(
        ...     "无法确定反射镜类型",
        ...     surface_index=3,
        ...     surface_type="standard"
        ... )
        ZmxConversionError: 转换错误 (表面 3, 类型: standard): 无法确定反射镜类型
    """
    
    def __init__(
        self, 
        message: str, 
        surface_index: Optional[int] = None, 
        surface_type: Optional[str] = None
    ):
        """初始化 ZmxConversionError
        
        参数:
            message: 错误描述信息
            surface_index: 发生错误的 ZMX 表面索引
            surface_type: 发生错误的表面类型
        """
        self.message = message
        self.surface_index = surface_index
        self.surface_type = surface_type
        super().__init__(self._format_message(message))
    
    def _format_message(self, message: str) -> str:
        """格式化错误信息
        
        根据是否有表面索引和类型信息，生成详细的错误消息。
        
        参数:
            message: 原始错误描述
        
        返回:
            格式化后的完整错误消息
        """
        if self.surface_index is not None:
            context_parts = [f"表面 {self.surface_index}"]
            if self.surface_type is not None:
                context_parts.append(f"类型: {self.surface_type}")
            context = ", ".join(context_parts)
            return f"转换错误 ({context}): {message}"
        return f"ZMX 转换错误: {message}"


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass
class ZmxSurfaceData:
    """ZMX 表面数据结构
    
    存储从 ZMX 文件中解析出的单个光学表面的所有参数。
    
    属性:
        index: 表面索引（从 0 开始）
        surface_type: 表面类型，如 'standard', 'coordinate_break', 'even_asphere'
        radius: 曲率半径 (mm)，平面为 np.inf
        thickness: 厚度/间距 (mm)，到下一表面的距离
        conic: 圆锥常数，球面为 0，抛物面为 -1
        material: 材料名称，如 'air', 'mirror', 'N-BK7'
        is_mirror: 是否为反射镜
        is_stop: 是否为光阑（孔径光阑）
        semi_diameter: 半口径 (mm)
        decenter_x: X 方向偏心 (mm)，用于坐标断点
        decenter_y: Y 方向偏心 (mm)，用于坐标断点
        tilt_x_deg: 绕 X 轴旋转角度 (度)，用于坐标断点
        tilt_y_deg: 绕 Y 轴旋转角度 (度)，用于坐标断点
        tilt_z_deg: 绕 Z 轴旋转角度 (度)，用于坐标断点
        order: 旋转顺序标志 (0-5)，用于坐标断点
        asphere_coeffs: 非球面系数列表，用于偶次非球面
        comment: 原始注释，通常包含元件名称
    
    示例:
        >>> surface = ZmxSurfaceData(
        ...     index=1,
        ...     surface_type='standard',
        ...     radius=100.0,
        ...     thickness=50.0,
        ...     material='mirror',
        ...     is_mirror=True,
        ... )
        >>> print(f"表面 {surface.index}: R={surface.radius} mm")
    """
    index: int                          # 表面索引
    surface_type: str                   # 表面类型：standard, coordinate_break, even_asphere, biconic
    radius: float = np.inf              # 曲率半径 (mm)，对于双锥面为 Y 方向曲率半径
    thickness: float = 0.0              # 厚度/间距 (mm)
    conic: float = 0.0                  # 圆锥常数，对于双锥面为 Y 方向圆锥常数
    material: str = "air"               # 材料
    is_mirror: bool = False             # 是否为反射镜
    is_stop: bool = False               # 是否为光阑
    is_ignored: bool = False            # 是否被忽略（HIDE 第六位为 1）
    semi_diameter: float = 0.0          # 半口径 (mm)
    # 坐标断点参数
    decenter_x: float = 0.0             # X 偏心 (mm)
    decenter_y: float = 0.0             # Y 偏心 (mm)
    tilt_x_deg: float = 0.0             # X 轴旋转 (度)
    tilt_y_deg: float = 0.0             # Y 轴旋转 (度)
    tilt_z_deg: float = 0.0             # Z 轴旋转 (度)
    order: int = 0                      # 旋转顺序标志 (0-5)
    # 非球面系数
    asphere_coeffs: List[float] = field(default_factory=list)
    # 双锥面参数（BICONIC）
    # Zemax 中 CURV/CONI 是 Y 方向，RARM1/RARM2 是 X 方向
    radius_x: float = np.inf            # X 方向曲率半径 (mm)
    conic_x: float = 0.0                # X 方向圆锥常数
    # 近轴面形参数（PARAXIAL）
    focal_length: float = np.inf        # 焦距 (mm)，用于理想薄透镜
    # 原始注释
    comment: str = ""
    
    def __repr__(self) -> str:
        """返回表面的字符串表示"""
        parts = [f"ZmxSurfaceData(index={self.index}, type='{self.surface_type}'"]
        
        if self.is_ignored:
            parts.append(", is_ignored=True")
        if self.radius != np.inf:
            parts.append(f", radius={self.radius:.4f}")
        if self.thickness != 0.0:
            parts.append(f", thickness={self.thickness:.4f}")
        if self.conic != 0.0:
            parts.append(f", conic={self.conic:.4f}")
        if self.material != "air":
            parts.append(f", material='{self.material}'")
        if self.is_mirror:
            parts.append(", is_mirror=True")
        if self.is_stop:
            parts.append(", is_stop=True")
        if self.semi_diameter > 0:
            parts.append(f", semi_diameter={self.semi_diameter:.4f}")
        if self.comment:
            parts.append(f", comment='{self.comment}'")
        
        # 坐标断点参数
        if self.surface_type == 'coordinate_break':
            if self.decenter_x != 0.0:
                parts.append(f", decenter_x={self.decenter_x:.4f}")
            if self.decenter_y != 0.0:
                parts.append(f", decenter_y={self.decenter_y:.4f}")
            if self.tilt_x_deg != 0.0:
                parts.append(f", tilt_x_deg={self.tilt_x_deg:.4f}")
            if self.tilt_y_deg != 0.0:
                parts.append(f", tilt_y_deg={self.tilt_y_deg:.4f}")
            if self.tilt_z_deg != 0.0:
                parts.append(f", tilt_z_deg={self.tilt_z_deg:.4f}")
        
        # 双锥面参数
        if self.surface_type == 'biconic':
            if self.radius_x != np.inf:
                parts.append(f", radius_x={self.radius_x:.4f}")
            if self.conic_x != 0.0:
                parts.append(f", conic_x={self.conic_x:.4f}")
        
        parts.append(")")
        return "".join(parts)


@dataclass
class ZmxDataModel:
    """ZMX 数据模型
    
    存储从 ZMX 文件中解析出的完整光学系统数据。
    
    属性:
        surfaces: 表面数据字典，键为表面索引
        wavelengths: 波长列表 (μm)
        primary_wavelength_index: 主波长索引（从 0 开始）
        entrance_pupil_diameter: 入瞳直径 (mm)
    
    示例:
        >>> data_model = ZmxDataModel()
        >>> data_model.surfaces[0] = ZmxSurfaceData(index=0, surface_type='standard')
        >>> data_model.wavelengths.append(0.55)
        >>> data_model.entrance_pupil_diameter = 20.0
        >>> 
        >>> # 获取特定表面
        >>> surface = data_model.get_surface(0)
        >>> 
        >>> # 获取所有反射镜
        >>> mirrors = data_model.get_mirror_surfaces()
    """
    surfaces: Dict[int, ZmxSurfaceData] = field(default_factory=dict)
    wavelengths: List[float] = field(default_factory=list)  # 波长列表 (μm)
    primary_wavelength_index: int = 0
    entrance_pupil_diameter: float = 0.0  # 入瞳直径 (mm)
    
    def get_surface(self, index: int) -> Optional[ZmxSurfaceData]:
        """获取指定索引的表面数据
        
        参数:
            index: 表面索引
        
        返回:
            ZmxSurfaceData 对象，如果索引不存在则返回 None
        
        示例:
            >>> surface = data_model.get_surface(1)
            >>> if surface is not None:
            ...     print(f"表面 1 类型: {surface.surface_type}")
        """
        return self.surfaces.get(index)
    
    def get_mirror_surfaces(self) -> List[ZmxSurfaceData]:
        """获取所有反射镜表面
        
        返回:
            所有 is_mirror=True 的表面列表，按索引排序
        
        示例:
            >>> mirrors = data_model.get_mirror_surfaces()
            >>> print(f"共 {len(mirrors)} 个反射镜")
            >>> for mirror in mirrors:
            ...     print(f"  表面 {mirror.index}: R={mirror.radius} mm")
        """
        return sorted(
            [s for s in self.surfaces.values() if s.is_mirror],
            key=lambda s: s.index
        )
    
    def get_coordinate_break_surfaces(self) -> List[ZmxSurfaceData]:
        """获取所有坐标断点表面
        
        返回:
            所有 surface_type='coordinate_break' 的表面列表，按索引排序
        
        示例:
            >>> coord_breaks = data_model.get_coordinate_break_surfaces()
            >>> for cb in coord_breaks:
            ...     print(f"  表面 {cb.index}: tilt_x={cb.tilt_x_deg}°")
        """
        return sorted(
            [s for s in self.surfaces.values() if s.surface_type == 'coordinate_break'],
            key=lambda s: s.index
        )
    
    def get_surface_count(self) -> int:
        """获取表面总数
        
        返回:
            表面总数
        """
        return len(self.surfaces)
    
    def get_max_surface_index(self) -> int:
        """获取最大表面索引
        
        返回:
            最大表面索引，如果没有表面则返回 -1
        """
        if not self.surfaces:
            return -1
        return max(self.surfaces.keys())
    
    def __repr__(self) -> str:
        """返回数据模型的字符串表示"""
        mirror_count = len(self.get_mirror_surfaces())
        coord_break_count = len(self.get_coordinate_break_surfaces())
        
        return (
            f"ZmxDataModel("
            f"surfaces={len(self.surfaces)}, "
            f"mirrors={mirror_count}, "
            f"coord_breaks={coord_break_count}, "
            f"wavelengths={self.wavelengths}, "
            f"entrance_pupil_diameter={self.entrance_pupil_diameter} mm)"
        )


# =============================================================================
# ZMX 文件解析器
# =============================================================================

class ZmxParser:
    """ZMX 文件解析器
    
    解析 Zemax .zmx 文件并提取光学系统数据。
    
    支持的编码格式：
    - UTF-16（Zemax 默认格式）
    - UTF-8
    - ISO-8859-1（Latin-1）
    
    支持的表面类型：
    - STANDARD：标准球面/非球面
    - COORDBRK：坐标断点
    - EVENASPH：偶次非球面
    
    参数:
        filepath: .zmx 文件路径
    
    属性:
        filepath: 文件路径
        SUPPORTED_ENCODINGS: 支持的编码列表
    
    示例:
        >>> parser = ZmxParser("system.zmx")
        >>> data_model = parser.parse()
        >>> print(f"共 {len(data_model.surfaces)} 个表面")
        >>> 
        >>> # 获取所有反射镜
        >>> mirrors = data_model.get_mirror_surfaces()
        >>> for mirror in mirrors:
        ...     print(f"  表面 {mirror.index}: R={mirror.radius} mm")
    
    异常:
        FileNotFoundError: 文件不存在
        ZmxParseError: 文件编码不支持或解析错误
        ZmxUnsupportedError: 不支持的 ZMX 特性（如非序列模式）
    """
    
    # 支持的文件编码列表，按优先级排序
    # UTF-16 是 Zemax 的默认编码，放在首位
    SUPPORTED_ENCODINGS = ["utf-16", "utf-8", "iso-8859-1"]
    
    # 操作符到解析方法的映射字典
    # 键为 ZMX 文件中的操作符名称，值为对应的解析方法名称
    _OPERATOR_HANDLERS = {
        "MODE": "_parse_mode",
        "SURF": "_parse_surface",
        "TYPE": "_parse_type",
        "CURV": "_parse_curv",
        "DISZ": "_parse_disz",
        "CONI": "_parse_coni",
        "GLAS": "_parse_glas",
        "PARM": "_parse_parm",
        "DIAM": "_parse_diam",
        "STOP": "_parse_stop",
        "COMM": "_parse_comm",
        "ENPD": "_parse_enpd",
        "WAVM": "_parse_wavm",
        "RARM": "_parse_rarm",  # 双锥面 X 方向参数
        "HIDE": "_parse_hide",  # 忽略表面标记
    }
    
    def __init__(self, filepath: str):
        """初始化 ZMX 解析器
        
        参数:
            filepath: .zmx 文件路径
        
        示例:
            >>> parser = ZmxParser("path/to/system.zmx")
        """
        self.filepath = filepath
        self._data_model = ZmxDataModel()
        self._current_surface_index = -1
        self._current_surface: Optional[ZmxSurfaceData] = None
        self._current_line_number = 0
        self._current_line_content = ""
    
    def parse(self) -> ZmxDataModel:
        """解析 ZMX 文件
        
        读取并解析 ZMX 文件，返回结构化的数据模型。
        
        返回:
            ZmxDataModel: 解析后的数据模型，包含所有表面、波长和系统参数
        
        异常:
            FileNotFoundError: 文件不存在
            ZmxParseError: 文件编码不支持或解析错误
            ZmxUnsupportedError: 不支持的 ZMX 特性
        
        示例:
            >>> parser = ZmxParser("system.zmx")
            >>> data_model = parser.parse()
            >>> print(f"入瞳直径: {data_model.entrance_pupil_diameter} mm")
            >>> print(f"波长: {data_model.wavelengths} μm")
        """
        # 读取文件内容
        lines = self._try_read_file()
        
        # 重置数据模型
        self._data_model = ZmxDataModel()
        self._current_surface_index = -1
        self._current_surface = None
        
        # 逐行解析
        for line_number, line in enumerate(lines, start=1):
            self._current_line_number = line_number
            self._current_line_content = line
            self._parse_line(line)
        
        # 完成最后一个表面的解析
        self._finalize_current_surface()
        
        return self._data_model
    
    def _try_read_file(self) -> List[str]:
        """尝试使用不同编码读取文件
        
        按照 SUPPORTED_ENCODINGS 列表中的顺序尝试不同的编码，
        直到成功读取文件或所有编码都失败。
        
        返回:
            List[str]: 文件内容的行列表
        
        异常:
            FileNotFoundError: 文件不存在，包含描述性错误信息
            ZmxParseError: 所有支持的编码都无法解码文件
        
        实现说明:
            1. 首先检查文件是否存在
            2. 按顺序尝试 UTF-16、UTF-8、ISO-8859-1 编码
            3. 如果某个编码成功，立即返回结果
            4. 如果所有编码都失败，抛出 ZmxParseError
        """
        import os
        
        # 检查文件是否存在
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(
                f"ZMX 文件不存在: {self.filepath}"
            )
        
        # 检查是否为文件（而非目录）
        if not os.path.isfile(self.filepath):
            raise FileNotFoundError(
                f"路径不是文件: {self.filepath}"
            )
        
        # 记录尝试过的编码和对应的错误
        encoding_errors = []
        
        # 按顺序尝试不同的编码
        for encoding in self.SUPPORTED_ENCODINGS:
            try:
                with open(self.filepath, 'r', encoding=encoding) as f:
                    content = f.read()
                    # 按行分割，处理不同的换行符
                    lines = content.splitlines()
                    return lines
            except UnicodeDecodeError as e:
                # 记录解码错误，继续尝试下一个编码
                encoding_errors.append(f"{encoding}: {str(e)}")
            except UnicodeError as e:
                # 处理其他 Unicode 相关错误
                encoding_errors.append(f"{encoding}: {str(e)}")
        
        # 所有编码都失败，抛出详细的错误信息
        error_details = "\n  ".join(encoding_errors)
        raise ZmxParseError(
            f"无法使用任何支持的编码读取文件。\n"
            f"尝试的编码: {', '.join(self.SUPPORTED_ENCODINGS)}\n"
            f"错误详情:\n  {error_details}"
        )
    
    def _parse_line(self, line: str) -> None:
        """解析单行数据
        
        解析 ZMX 文件中的单行，根据操作符调用相应的解析方法。
        
        参数:
            line: 要解析的行内容
        
        说明:
            ZMX 文件格式为 "OPERAND [data1] [data2] ..."
            此方法提取操作符并分发到对应的解析方法。
            
            - 空行和纯空白行会被跳过
            - 不识别的操作符会被静默忽略（允许扩展性）
        """
        # 跳过空行和纯空白行
        stripped = line.strip()
        if not stripped:
            return
        
        # 分割行内容，提取操作符和数据
        # ZMX 格式：OPERAND [data1] [data2] ...
        parts = stripped.split()
        if not parts:
            return
        
        # 提取操作符（第一个词）
        operator = parts[0].upper()
        
        # 提取数据部分（操作符之后的所有内容）
        data = parts[1:] if len(parts) > 1 else []
        
        # 查找对应的处理方法
        handler_name = self._OPERATOR_HANDLERS.get(operator)
        if handler_name is not None:
            # 获取处理方法并调用
            handler = getattr(self, handler_name, None)
            if handler is not None:
                handler(data)
        # 不识别的操作符静默忽略，保持扩展性
    
    def _finalize_current_surface(self) -> None:
        """完成当前表面的解析
        
        将当前正在解析的表面数据保存到数据模型中。
        
        说明:
            当遇到新的 SURF 操作符或文件结束时调用此方法，
            确保前一个表面的数据被正确保存。
            
            此方法为占位符，将在任务 2.3 中实现完整功能。
        """
        if self._current_surface is not None:
            self._data_model.surfaces[self._current_surface.index] = self._current_surface
            self._current_surface = None
    
    def _parse_mode(self, data: List[str]) -> None:
        """解析 MODE 操作符
        
        验证 ZMX 文件是否为序列模式（SEQ）。
        本解析器仅支持序列模式，非序列模式将抛出异常。
        
        参数:
            data: MODE 操作符后的数据列表，预期为 ["SEQ"] 或 ["NSC"]
        
        异常:
            ZmxUnsupportedError: 如果模式不是 SEQ
        
        示例:
            ZMX 文件中的行：
            - "MODE SEQ" -> 有效，序列模式
            - "MODE NSC" -> 抛出 ZmxUnsupportedError
        """
        if not data:
            # 没有模式数据，静默忽略
            return
        
        mode = data[0].upper()
        
        if mode != "SEQ":
            raise ZmxUnsupportedError(
                f"不支持的模式: {mode}。本解析器仅支持序列模式（SEQ）",
                line_number=self._current_line_number,
                line_content=self._current_line_content
            )
    
    # =========================================================================
    # 表面相关操作符解析方法
    # =========================================================================
    
    def _parse_surface(self, data: List[str]) -> None:
        """解析 SURF 操作符（表面开始）
        
        当遇到 SURF 操作符时，完成当前表面的解析并创建新的表面数据对象。
        
        参数:
            data: SURF 操作符后的数据列表，预期为 [surface_index]
        
        说明:
            ZMX 文件中的 SURF 操作符标记一个新表面的开始。
            格式：SURF <index>
            
            此方法会：
            1. 调用 _finalize_current_surface() 保存当前表面
            2. 创建新的 ZmxSurfaceData 对象
            3. 设置表面索引和默认类型（standard）
        
        示例:
            ZMX 文件中的行：
            - "SURF 0" -> 创建索引为 0 的表面
            - "SURF 1" -> 创建索引为 1 的表面
        """
        # 完成当前表面的解析
        self._finalize_current_surface()
        
        # 解析表面索引
        if data:
            try:
                surface_index = int(data[0])
            except ValueError:
                raise ZmxParseError(
                    f"无效的表面索引: {data[0]}",
                    line_number=self._current_line_number,
                    line_content=self._current_line_content
                )
        else:
            # 如果没有提供索引，使用递增的索引
            surface_index = self._current_surface_index + 1
        
        # 更新当前表面索引
        self._current_surface_index = surface_index
        
        # 创建新的表面数据对象，默认类型为 standard
        self._current_surface = ZmxSurfaceData(
            index=surface_index,
            surface_type='standard'
        )
    
    def _parse_type(self, data: List[str]) -> None:
        """解析 TYPE 操作符（表面类型）
        
        设置当前表面的类型。
        
        参数:
            data: TYPE 操作符后的数据列表，预期为 [type_name]
        
        支持的表面类型:
            - STANDARD: 标准球面/非球面 -> 'standard'
            - COORDBRK: 坐标断点 -> 'coordinate_break'
            - EVENASPH: 偶次非球面 -> 'even_asphere'
        
        异常:
            ZmxUnsupportedError: 如果表面类型不支持
        
        示例:
            ZMX 文件中的行：
            - "TYPE STANDARD" -> surface_type = 'standard'
            - "TYPE COORDBRK" -> surface_type = 'coordinate_break'
            - "TYPE EVENASPH" -> surface_type = 'even_asphere'
            - "TYPE TOROIDAL" -> 抛出 ZmxUnsupportedError
        """
        if not data or self._current_surface is None:
            return
        
        type_name = data[0].upper()
        
        # 表面类型映射
        type_mapping = {
            'STANDARD': 'standard',
            'COORDBRK': 'coordinate_break',
            'EVENASPH': 'even_asphere',
            'BICONICX': 'biconic',
            'PARAXIAL': 'paraxial',
        }
        
        if type_name in type_mapping:
            self._current_surface.surface_type = type_mapping[type_name]
        else:
            raise ZmxUnsupportedError(
                f"不支持的表面类型: {type_name}",
                line_number=self._current_line_number,
                line_content=self._current_line_content
            )
    
    def _parse_curv(self, data: List[str]) -> None:
        """解析 CURV 操作符（曲率）
        
        将曲率值转换为曲率半径并设置到当前表面。
        
        参数:
            data: CURV 操作符后的数据列表，曲率值在 data[0]
        
        说明:
            ZMX 文件中存储的是曲率（curvature = 1/radius），
            需要转换为曲率半径。
            
            转换规则：
            - 曲率为 0 时，半径为无穷大（平面）
            - 曲率非零时，半径 = 1 / 曲率
        
        示例:
            ZMX 文件中的行：
            - "CURV 0.01 0 0 0 0" -> radius = 100.0 mm
            - "CURV 0 0 0 0 0" -> radius = inf（平面）
            - "CURV -0.005 0 0 0 0" -> radius = -200.0 mm
        """
        if not data or self._current_surface is None:
            return
        
        try:
            curvature = float(data[0])
        except ValueError:
            raise ZmxParseError(
                f"无效的曲率值: {data[0]}",
                line_number=self._current_line_number,
                line_content=self._current_line_content
            )
        
        # 曲率转半径：radius = 1/curvature
        # 曲率为 0 时，半径为无穷大
        if curvature == 0.0:
            self._current_surface.radius = np.inf
        else:
            self._current_surface.radius = 1.0 / curvature
    
    def _parse_disz(self, data: List[str]) -> None:
        """解析 DISZ 操作符（厚度）
        
        设置当前表面到下一表面的厚度/间距。
        
        参数:
            data: DISZ 操作符后的数据列表，厚度值在 data[0]
        
        说明:
            厚度表示从当前表面顶点到下一表面顶点的距离（mm）。
            特殊值 "INFINITY" 表示无穷大（通常用于物面）。
        
        示例:
            ZMX 文件中的行：
            - "DISZ 50" -> thickness = 50.0 mm
            - "DISZ -30" -> thickness = -30.0 mm（反射后传播）
            - "DISZ INFINITY" -> thickness = inf
            - "DISZ 0" -> thickness = 0.0 mm
        """
        if not data or self._current_surface is None:
            return
        
        thickness_str = data[0].upper()
        
        # 处理 INFINITY 特殊值
        if thickness_str == "INFINITY":
            self._current_surface.thickness = np.inf
        else:
            try:
                self._current_surface.thickness = float(data[0])
            except ValueError:
                raise ZmxParseError(
                    f"无效的厚度值: {data[0]}",
                    line_number=self._current_line_number,
                    line_content=self._current_line_content
                )
    
    def _parse_coni(self, data: List[str]) -> None:
        """解析 CONI 操作符（圆锥常数）
        
        设置当前表面的圆锥常数。
        
        参数:
            data: CONI 操作符后的数据列表，圆锥常数在 data[0]
        
        说明:
            圆锥常数定义了表面的形状：
            - conic = 0: 球面
            - conic = -1: 抛物面
            - conic < -1: 双曲面
            - -1 < conic < 0: 椭球面（扁）
            - conic > 0: 椭球面（长）
        
        示例:
            ZMX 文件中的行：
            - "CONI 0" -> conic = 0.0（球面）
            - "CONI -1" -> conic = -1.0（抛物面）
            - "CONI -0.5" -> conic = -0.5（椭球面）
        """
        if not data or self._current_surface is None:
            return
        
        try:
            self._current_surface.conic = float(data[0])
        except ValueError:
            raise ZmxParseError(
                f"无效的圆锥常数: {data[0]}",
                line_number=self._current_line_number,
                line_content=self._current_line_content
            )
    
    def _parse_diam(self, data: List[str]) -> None:
        """解析 DIAM 操作符（半口径）
        
        设置当前表面的半口径。
        
        参数:
            data: DIAM 操作符后的数据列表，半口径值在 data[0]
        
        说明:
            半口径（semi-diameter）是表面有效区域的半径（mm）。
            DIAM 操作符格式：DIAM <value> [flags...]
            只需要第一个值（半口径）。
        
        示例:
            ZMX 文件中的行：
            - "DIAM 25 0 0 0 1" -> semi_diameter = 25.0 mm
            - "DIAM 10.5 1 0 0 0" -> semi_diameter = 10.5 mm
        """
        if not data or self._current_surface is None:
            return
        
        try:
            self._current_surface.semi_diameter = float(data[0])
        except ValueError:
            raise ZmxParseError(
                f"无效的半口径值: {data[0]}",
                line_number=self._current_line_number,
                line_content=self._current_line_content
            )
    
    def _parse_stop(self, data: List[str]) -> None:
        """解析 STOP 操作符（光阑标记）
        
        将当前表面标记为光阑（孔径光阑）。
        
        参数:
            data: STOP 操作符后的数据列表（通常为空）
        
        说明:
            STOP 操作符标记当前表面为系统的孔径光阑。
            一个光学系统通常只有一个光阑。
        
        示例:
            ZMX 文件中的行：
            - "STOP" -> is_stop = True
        """
        if self._current_surface is None:
            return
        
        self._current_surface.is_stop = True
    
    def _parse_comm(self, data: List[str]) -> None:
        """解析 COMM 操作符（注释）
        
        设置当前表面的注释内容。
        
        参数:
            data: COMM 操作符后的数据列表，所有内容连接为注释
        
        说明:
            注释通常包含元件名称或描述信息。
            所有数据部分用空格连接形成完整注释。
        
        示例:
            ZMX 文件中的行：
            - "COMM M1" -> comment = "M1"
            - "COMM Primary Mirror" -> comment = "Primary Mirror"
            - "COMM Fold Mirror 45deg" -> comment = "Fold Mirror 45deg"
        """
        if self._current_surface is None:
            return
        
        # 将所有数据部分用空格连接
        self._current_surface.comment = " ".join(data) if data else ""
    
    def _parse_glas(self, data: List[str]) -> None:
        """解析 GLAS 操作符（材料）
        
        设置当前表面的材料属性。如果材料是 MIRROR，则标记为反射镜。
        
        参数:
            data: GLAS 操作符后的数据列表，材料名称在 data[0]
        
        说明:
            GLAS 操作符格式：GLAS <material_name> [other_params...]
            
            - 如果材料名称是 "MIRROR"（不区分大小写），设置 is_mirror = True
            - 如果材料名称是 "__BLANK"，设置为空气（避免读取玻璃库）
            - 否则设置 material = 材料名称
        
        示例:
            ZMX 文件中的行：
            - "GLAS MIRROR 0 0 1.5 40" -> is_mirror = True, material = "mirror"
            - "GLAS N-BK7 0 0 1.5 40" -> is_mirror = False, material = "N-BK7"
            - "GLAS __BLANK 0 0 1.5 40" -> is_mirror = False, material = "air"
            - "GLAS BK7" -> is_mirror = False, material = "BK7"
        """
        if not data or self._current_surface is None:
            return
        
        material_name = data[0]
        
        # 检查是否为反射镜
        if material_name.upper() == "MIRROR":
            self._current_surface.is_mirror = True
            self._current_surface.material = "mirror"
        # 检查是否为 __BLANK（空白材料，视为空气）
        elif material_name == "__BLANK":
            self._current_surface.is_mirror = False
            self._current_surface.material = "air"
        else:
            self._current_surface.is_mirror = False
            self._current_surface.material = material_name
    
    def _parse_parm(self, data: List[str]) -> None:
        """解析 PARM 操作符（参数）
        
        解析表面参数，根据表面类型提取不同的参数：
        - 坐标断点（COORDBRK）：偏心和倾斜参数
        - 近轴面形（PARAXIAL）：焦距参数
        
        参数:
            data: PARM 操作符后的数据列表
                - data[0]: 参数索引
                - data[1]: 参数值
        
        说明:
            PARM 操作符格式：PARM <param_index> <value>
            
            对于 COORDBRK 表面类型，参数映射如下（Zemax 约定）：
            - PARM 1 → decenter_x (mm)
            - PARM 2 → decenter_y (mm)
            - PARM 3 → tilt_x (度，绕 X 轴旋转)
            - PARM 4 → tilt_y (度，绕 Y 轴旋转)
            - PARM 5 → tilt_z (度，绕 Z 轴旋转)
            - PARM 6 → order (旋转顺序标志)
            
            对于 PARAXIAL 表面类型，参数映射如下：
            - PARM 1 → focal_length (mm)，理想薄透镜焦距
            
            注意：角度值以度为单位存储在 tilt_x_deg, tilt_y_deg, tilt_z_deg 字段中。
            转换为弧度的工作在 ElementConverter 中进行。
        
        示例:
            ZMX 文件中的行（COORDBRK）：
            - "PARM 1 0" -> decenter_x = 0.0 mm
            - "PARM 3 45" -> tilt_x_deg = 45.0 度
            
            ZMX 文件中的行（PARAXIAL）：
            - "PARM 1 100" -> focal_length = 100.0 mm
        
        异常:
            ZmxParseError: 如果参数索引或参数值无效
        """
        # 需要至少两个数据项：参数索引和参数值
        if len(data) < 2 or self._current_surface is None:
            return
        
        # 只处理坐标断点和近轴面形类型的表面
        if self._current_surface.surface_type not in ('coordinate_break', 'paraxial'):
            return
        
        # 解析参数索引
        try:
            param_index = int(data[0])
        except ValueError:
            raise ZmxParseError(
                f"无效的参数索引: {data[0]}",
                line_number=self._current_line_number,
                line_content=self._current_line_content
            )
        
        # 解析参数值
        try:
            param_value = float(data[1])
        except ValueError:
            raise ZmxParseError(
                f"无效的参数值: {data[1]}",
                line_number=self._current_line_number,
                line_content=self._current_line_content
            )
        
        # 根据表面类型和参数索引设置对应的属性
        if self._current_surface.surface_type == 'coordinate_break':
            # COORDBRK 参数映射（Zemax 约定）
            if param_index == 1:
                # PARM 1 → decenter_x (mm)
                self._current_surface.decenter_x = param_value
            elif param_index == 2:
                # PARM 2 → decenter_y (mm)
                self._current_surface.decenter_y = param_value
            elif param_index == 3:
                # PARM 3 → tilt_x (度，绕 X 轴旋转)
                self._current_surface.tilt_x_deg = param_value
            elif param_index == 4:
                # PARM 4 → tilt_y (度，绕 Y 轴旋转)
                self._current_surface.tilt_y_deg = param_value
            elif param_index == 5:
                # PARM 5 → tilt_z (度，绕 Z 轴旋转)
                self._current_surface.tilt_z_deg = param_value
            elif param_index == 6:
                # PARM 6 → order (旋转顺序标志)
                self._current_surface.order = int(param_value)
        
        elif self._current_surface.surface_type == 'paraxial':
            # PARAXIAL 参数映射（Zemax 约定）
            if param_index == 1:
                # PARM 1 → focal_length (mm)
                self._current_surface.focal_length = param_value
    
    def _parse_enpd(self, data: List[str]) -> None:
        """解析 ENPD 操作符（入瞳直径）
        
        设置光学系统的入瞳直径。
        
        参数:
            data: ENPD 操作符后的数据列表，入瞳直径值在 data[0]
        
        说明:
            ENPD 操作符格式：ENPD <diameter>
            入瞳直径单位为毫米（mm）。
        
        示例:
            ZMX 文件中的行：
            - "ENPD 20" -> entrance_pupil_diameter = 20.0 mm
            - "ENPD 25.5" -> entrance_pupil_diameter = 25.5 mm
        
        异常:
            ZmxParseError: 如果入瞳直径值无效
        """
        if not data:
            return
        
        try:
            self._data_model.entrance_pupil_diameter = float(data[0])
        except ValueError:
            raise ZmxParseError(
                f"无效的入瞳直径值: {data[0]}",
                line_number=self._current_line_number,
                line_content=self._current_line_content
            )
    
    def _parse_wavm(self, data: List[str]) -> None:
        """解析 WAVM 操作符（波长）
        
        解析波长定义并添加到数据模型的波长列表中。
        
        参数:
            data: WAVM 操作符后的数据列表
                - data[0]: 波长索引（从 1 开始）
                - data[1]: 波长值（单位：μm）
                - data[2]: 权重（可选，1 表示主波长）
        
        说明:
            WAVM 操作符格式：WAVM <index> <wavelength> [weight]
            
            - 波长索引从 1 开始（Zemax 约定）
            - 波长值单位为微米（μm）
            - 如果权重为 1，则设置为主波长
            - 波长按索引顺序添加到列表中
        
        示例:
            ZMX 文件中的行：
            - "WAVM 1 0.55 1" -> 添加 0.55 μm 作为主波长
            - "WAVM 2 0.486 0.5" -> 添加 0.486 μm，权重 0.5
            - "WAVM 3 0.656 0.5" -> 添加 0.656 μm，权重 0.5
        
        异常:
            ZmxParseError: 如果波长索引或波长值无效
        """
        if len(data) < 2:
            return
        
        try:
            wavelength_index = int(data[0])  # 从 1 开始的索引
        except ValueError:
            raise ZmxParseError(
                f"无效的波长索引: {data[0]}",
                line_number=self._current_line_number,
                line_content=self._current_line_content
            )
        
        try:
            wavelength_value = float(data[1])  # 波长值（μm）
        except ValueError:
            raise ZmxParseError(
                f"无效的波长值: {data[1]}",
                line_number=self._current_line_number,
                line_content=self._current_line_content
            )
        
        # 解析权重（可选）
        weight = 1.0
        if len(data) >= 3:
            try:
                weight = float(data[2])
            except ValueError:
                # 权重解析失败时使用默认值，不抛出异常
                weight = 1.0
        
        # 确保波长列表有足够的空间
        # 波长索引从 1 开始，转换为从 0 开始的列表索引
        list_index = wavelength_index - 1
        
        # 扩展列表以容纳新波长
        while len(self._data_model.wavelengths) <= list_index:
            self._data_model.wavelengths.append(0.0)
        
        # 设置波长值
        self._data_model.wavelengths[list_index] = wavelength_value
        
        # 如果权重为 1，设置为主波长
        if weight == 1.0:
            self._data_model.primary_wavelength_index = list_index

    def _parse_rarm(self, data: List[str]) -> None:
        """解析 RARM 操作符（双锥面 X 方向参数）
        
        解析双锥面（BICONIC）表面的 X 方向曲率半径和圆锥常数。
        
        参数:
            data: RARM 操作符后的数据列表
                - data[0]: 参数索引（1 = X 方向曲率半径，2 = X 方向圆锥常数）
                - data[1]: 参数值
        
        说明:
            RARM 操作符格式：RARM <param_index> <value>
            
            Zemax 双锥面参数约定：
            - CURV/CONI: Y 方向曲率和圆锥常数（使用标准解析）
            - RARM 1: X 方向曲率半径（注意：是半径，不是曲率）
            - RARM 2: X 方向圆锥常数
        
        示例:
            ZMX 文件中的行：
            - "RARM 1 50.0" -> radius_x = 50.0 mm
            - "RARM 2 -1.0" -> conic_x = -1.0（抛物面）
        
        异常:
            ZmxParseError: 如果参数索引或参数值无效
        """
        if len(data) < 2 or self._current_surface is None:
            return
        
        # 解析参数索引
        try:
            param_index = int(data[0])
        except ValueError:
            raise ZmxParseError(
                f"无效的 RARM 参数索引: {data[0]}",
                line_number=self._current_line_number,
                line_content=self._current_line_content
            )
        
        # 解析参数值
        try:
            param_value = float(data[1])
        except ValueError:
            raise ZmxParseError(
                f"无效的 RARM 参数值: {data[1]}",
                line_number=self._current_line_number,
                line_content=self._current_line_content
            )
        
        # 根据参数索引设置对应的属性
        if param_index == 1:
            # RARM 1 → X 方向曲率半径 (mm)
            self._current_surface.radius_x = param_value
        elif param_index == 2:
            # RARM 2 → X 方向圆锥常数
            self._current_surface.conic_x = param_value

    def _parse_hide(self, data: List[str]) -> None:
        """解析 HIDE 操作符（表面隐藏/忽略标记）
        
        解析表面的隐藏标记，当第六位为 1 时，该表面应被忽略。
        
        参数:
            data: HIDE 操作符后的数据列表，包含 12 个标志位
                - data[5]: 第六位，1 表示忽略该表面
        
        说明:
            HIDE 操作符格式：HIDE <flag1> <flag2> ... <flag12>
            
            当第六位（索引 5）为 1 时，该表面应被完全忽略：
            - 不进行光线追迹
            - 不进行绘制
            - 不考虑坐标变换
        
        示例:
            ZMX 文件中的行：
            - "HIDE 0 0 0 0 0 1 0 0 0 0 0 0" -> is_ignored = True
            - "HIDE 0 0 0 0 0 0 0 0 0 0 0 0" -> is_ignored = False
        """
        if self._current_surface is None:
            return
        
        # 检查是否有足够的数据（至少需要 6 个值）
        if len(data) >= 6:
            try:
                # 第六位（索引 5）为 1 表示忽略该表面
                ignore_flag = int(data[5])
                self._current_surface.is_ignored = (ignore_flag == 1)
            except (ValueError, IndexError):
                # 解析失败时保持默认值（不忽略）
                pass
