"""
ZMX 元件转换器

本模块提供将 ZMX 数据模型转换为项目 OpticalElement 类型的功能。

主要类：
- CoordinateTransform: 坐标变换累积器，用于跟踪坐标断点的累积效果
- ConvertedElement: 转换后的元件数据，包含元件对象和元数据
- ElementConverter: ZMX 数据到 OpticalElement 的转换器（待实现）
- CodeGenerator: Python 代码生成器（待实现）

使用示例：
    >>> from sequential_system.zmx_parser import ZmxParser
    >>> from sequential_system.zmx_converter import ElementConverter
    >>> 
    >>> parser = ZmxParser("system.zmx")
    >>> data_model = parser.parse()
    >>> 
    >>> converter = ElementConverter(data_model)
    >>> elements = converter.convert()
    >>> code = converter.generate_code()
    >>> print(code)

作者：混合光学仿真项目
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, TYPE_CHECKING
import numpy as np

# 导入反射镜元件类
from gaussian_beam_simulation.optical_elements import (
    FlatMirror,
    ParabolicMirror,
    SphericalMirror,
)

if TYPE_CHECKING:
    from gaussian_beam_simulation.optical_elements import OpticalElement
    from sequential_system.zmx_parser import ZmxDataModel, ZmxSurfaceData


# =============================================================================
# 坐标变换数据类
# =============================================================================

@dataclass
class CoordinateTransform:
    """坐标变换累积器
    
    用于跟踪 ZMX 文件中坐标断点（COORDBRK）的累积效果。
    在处理折叠光路时，需要累积多个坐标断点的变换。
    
    属性:
        decenter_x: X 方向累积偏心 (mm)
        decenter_y: Y 方向累积偏心 (mm)
        decenter_z: Z 方向累积偏心 (mm)，通常来自 DISZ
        tilt_x_rad: 绕 X 轴累积旋转角度 (弧度)
        tilt_y_rad: 绕 Y 轴累积旋转角度 (弧度)
        tilt_z_rad: 绕 Z 轴累积旋转角度 (弧度)
    
    方法:
        apply_coordinate_break: 应用一个坐标断点的变换
        reset: 重置所有变换为零
        copy: 创建当前变换的副本
    
    示例:
        >>> transform = CoordinateTransform()
        >>> # 应用第一个坐标断点：45度倾斜
        >>> transform.apply_coordinate_break(
        ...     dx=0, dy=0, dz=0,
        ...     rx_deg=45, ry_deg=0, rz_deg=0
        ... )
        >>> print(f"累积倾斜: {np.rad2deg(transform.tilt_x_rad):.1f}°")
        累积倾斜: 45.0°
        >>> 
        >>> # 应用第二个坐标断点：再倾斜45度
        >>> transform.apply_coordinate_break(
        ...     dx=0, dy=0, dz=-50,
        ...     rx_deg=45, ry_deg=0, rz_deg=0
        ... )
        >>> print(f"累积倾斜: {np.rad2deg(transform.tilt_x_rad):.1f}°")
        累积倾斜: 90.0°
        >>> 
        >>> # 重置变换
        >>> transform.reset()
        >>> print(f"重置后倾斜: {np.rad2deg(transform.tilt_x_rad):.1f}°")
        重置后倾斜: 0.0°
    
    验证需求:
        - Requirements 3.4: 坐标断点累积
        - Requirements 6.2: 折叠光路坐标变换跟踪
    """
    
    decenter_x: float = 0.0  # mm
    decenter_y: float = 0.0  # mm
    decenter_z: float = 0.0  # mm
    tilt_x_rad: float = 0.0  # 弧度
    tilt_y_rad: float = 0.0  # 弧度
    tilt_z_rad: float = 0.0  # 弧度
    
    def apply_coordinate_break(
        self,
        dx: float,
        dy: float,
        dz: float,
        rx_deg: float,
        ry_deg: float,
        rz_deg: float,
    ) -> None:
        """应用坐标断点变换
        
        将一个坐标断点的变换累积到当前变换中。
        偏心值直接累加，旋转角度从度转换为弧度后累加。
        
        参数:
            dx: X 方向偏心 (mm)
            dy: Y 方向偏心 (mm)
            dz: Z 方向偏心/厚度 (mm)，通常来自 DISZ
            rx_deg: 绕 X 轴旋转角度 (度)
            ry_deg: 绕 Y 轴旋转角度 (度)
            rz_deg: 绕 Z 轴旋转角度 (度)
        
        说明:
            - 偏心值直接累加到对应的 decenter_x/y/z 属性
            - 旋转角度从度转换为弧度后累加到 tilt_x/y/z_rad 属性
            - 这是简化的累积方式，假设旋转顺序为 XYZ
            - 对于复杂的三维旋转，可能需要使用旋转矩阵进行精确计算
        
        示例:
            >>> transform = CoordinateTransform()
            >>> transform.apply_coordinate_break(
            ...     dx=5.0, dy=0, dz=10.0,
            ...     rx_deg=45, ry_deg=0, rz_deg=0
            ... )
            >>> print(f"偏心: ({transform.decenter_x}, {transform.decenter_y}, {transform.decenter_z}) mm")
            偏心: (5.0, 0.0, 10.0) mm
            >>> print(f"倾斜: {np.rad2deg(transform.tilt_x_rad):.1f}°")
            倾斜: 45.0°
        
        **Validates: Requirements 3.4, 6.2**
        """
        # 累积偏心
        self.decenter_x += dx
        self.decenter_y += dy
        self.decenter_z += dz
        
        # 累积旋转（度转弧度）
        self.tilt_x_rad += np.deg2rad(rx_deg)
        self.tilt_y_rad += np.deg2rad(ry_deg)
        self.tilt_z_rad += np.deg2rad(rz_deg)
    
    def reset(self) -> None:
        """重置所有变换为零
        
        将所有偏心和旋转值重置为初始状态（零）。
        
        示例:
            >>> transform = CoordinateTransform(
            ...     decenter_x=5.0,
            ...     tilt_x_rad=np.pi/4
            ... )
            >>> transform.reset()
            >>> print(f"偏心: {transform.decenter_x} mm")
            偏心: 0.0 mm
            >>> print(f"倾斜: {transform.tilt_x_rad} rad")
            倾斜: 0.0 rad
        
        **Validates: Requirements 3.4**
        """
        self.decenter_x = 0.0
        self.decenter_y = 0.0
        self.decenter_z = 0.0
        self.tilt_x_rad = 0.0
        self.tilt_y_rad = 0.0
        self.tilt_z_rad = 0.0
    
    def copy(self) -> 'CoordinateTransform':
        """创建当前变换的副本
        
        返回:
            CoordinateTransform: 当前变换的深拷贝
        
        示例:
            >>> transform = CoordinateTransform(decenter_x=5.0, tilt_x_rad=0.5)
            >>> transform_copy = transform.copy()
            >>> transform_copy.decenter_x = 10.0
            >>> print(f"原始: {transform.decenter_x} mm")
            原始: 5.0 mm
            >>> print(f"副本: {transform_copy.decenter_x} mm")
            副本: 10.0 mm
        """
        return CoordinateTransform(
            decenter_x=self.decenter_x,
            decenter_y=self.decenter_y,
            decenter_z=self.decenter_z,
            tilt_x_rad=self.tilt_x_rad,
            tilt_y_rad=self.tilt_y_rad,
            tilt_z_rad=self.tilt_z_rad,
        )
    
    @property
    def has_decenter(self) -> bool:
        """是否有偏心
        
        返回:
            bool: 如果任何偏心值非零则返回 True
        """
        return (
            self.decenter_x != 0.0 or
            self.decenter_y != 0.0 or
            self.decenter_z != 0.0
        )
    
    @property
    def has_tilt(self) -> bool:
        """是否有倾斜
        
        返回:
            bool: 如果任何旋转值非零则返回 True
        """
        return (
            self.tilt_x_rad != 0.0 or
            self.tilt_y_rad != 0.0 or
            self.tilt_z_rad != 0.0
        )
    
    @property
    def tilt_x_deg(self) -> float:
        """绕 X 轴旋转角度（度）"""
        return np.rad2deg(self.tilt_x_rad)
    
    @property
    def tilt_y_deg(self) -> float:
        """绕 Y 轴旋转角度（度）"""
        return np.rad2deg(self.tilt_y_rad)
    
    @property
    def tilt_z_deg(self) -> float:
        """绕 Z 轴旋转角度（度）"""
        return np.rad2deg(self.tilt_z_rad)
    
    def __repr__(self) -> str:
        """返回变换的字符串表示"""
        parts = ["CoordinateTransform("]
        
        if self.has_decenter:
            parts.append(
                f"decenter=({self.decenter_x:.3f}, {self.decenter_y:.3f}, {self.decenter_z:.3f}) mm"
            )
        
        if self.has_tilt:
            if self.has_decenter:
                parts.append(", ")
            parts.append(
                f"tilt=({self.tilt_x_deg:.2f}°, {self.tilt_y_deg:.2f}°, {self.tilt_z_deg:.2f}°)"
            )
        
        if not self.has_decenter and not self.has_tilt:
            parts.append("identity")
        
        parts.append(")")
        return "".join(parts)


# =============================================================================
# 转换后元件数据类
# =============================================================================

@dataclass
class ConvertedElement:
    """转换后的元件数据
    
    存储从 ZMX 表面数据转换得到的光学元件及其元数据。
    
    属性:
        element: 光学元件对象（OpticalElement 子类实例）
        zmx_surface_index: 原始 ZMX 表面索引
        zmx_comment: 原始 ZMX 注释（通常包含元件名称）
        is_fold_mirror: 是否为折叠镜
        fold_angle_deg: 折叠角度（度），仅当 is_fold_mirror=True 时有意义
    
    说明:
        - element 属性存储实际的光学元件对象（如 FlatMirror, ParabolicMirror 等）
        - zmx_surface_index 用于追溯原始 ZMX 文件中的表面
        - zmx_comment 保留原始注释，便于识别元件
        - is_fold_mirror 和 fold_angle_deg 用于标识折叠镜配置
    
    示例:
        >>> from gaussian_beam_simulation.optical_elements import FlatMirror
        >>> 
        >>> # 创建一个 45 度折叠镜的转换结果
        >>> mirror = FlatMirror(
        ...     thickness=100.0,
        ...     semi_aperture=25.0,
        ...     tilt_x=np.pi/4,
        ...     is_fold=True,
        ... )
        >>> converted = ConvertedElement(
        ...     element=mirror,
        ...     zmx_surface_index=5,
        ...     zmx_comment="M1 - Fold Mirror",
        ...     is_fold_mirror=True,
        ...     fold_angle_deg=45.0,
        ... )
        >>> print(f"元件类型: {type(converted.element).__name__}")
        元件类型: FlatMirror
        >>> print(f"ZMX 索引: {converted.zmx_surface_index}")
        ZMX 索引: 5
        >>> print(f"是否折叠镜: {converted.is_fold_mirror}")
        是否折叠镜: True
        >>> print(f"折叠角度: {converted.fold_angle_deg}°")
        折叠角度: 45.0°
    
    验证需求:
        - Requirements 5.7: 折叠镜识别
        - Requirements 9.4: 代码生成时包含原始 ZMX 索引注释
        - Requirements 9.5: 代码生成时包含折叠角度注释
    """
    
    element: Any  # OpticalElement 类型，使用 Any 避免循环导入
    zmx_surface_index: int
    zmx_comment: str = ""
    is_fold_mirror: bool = False
    fold_angle_deg: float = 0.0
    
    @property
    def element_type(self) -> str:
        """获取元件类型名称
        
        返回:
            str: 元件类的名称（如 'FlatMirror', 'ParabolicMirror' 等）
        """
        if self.element is not None:
            return type(self.element).__name__
        return "Unknown"
    
    @property
    def has_comment(self) -> bool:
        """是否有注释
        
        返回:
            bool: 如果 zmx_comment 非空则返回 True
        """
        return bool(self.zmx_comment and self.zmx_comment.strip())
    
    def get_code_comment(self) -> str:
        """生成代码注释
        
        生成用于代码生成的注释字符串，包含 ZMX 索引和原始注释。
        
        返回:
            str: 格式化的注释字符串
        
        示例:
            >>> converted = ConvertedElement(
            ...     element=None,
            ...     zmx_surface_index=5,
            ...     zmx_comment="M1",
            ...     is_fold_mirror=True,
            ...     fold_angle_deg=45.0,
            ... )
            >>> print(converted.get_code_comment())
            # ZMX Surface 5: M1 (Fold Mirror, 45.0°)
        """
        parts = [f"# ZMX Surface {self.zmx_surface_index}"]
        
        if self.has_comment:
            parts.append(f": {self.zmx_comment}")
        
        if self.is_fold_mirror:
            parts.append(f" (Fold Mirror, {self.fold_angle_deg:.1f}°)")
        
        return "".join(parts)
    
    def __repr__(self) -> str:
        """返回转换元件的字符串表示"""
        parts = [
            f"ConvertedElement(",
            f"type={self.element_type}, ",
            f"zmx_index={self.zmx_surface_index}",
        ]
        
        if self.has_comment:
            parts.append(f", comment='{self.zmx_comment}'")
        
        if self.is_fold_mirror:
            parts.append(f", fold_angle={self.fold_angle_deg:.1f}°")
        
        parts.append(")")
        return "".join(parts)


# =============================================================================
# ElementConverter 类
# =============================================================================

class ElementConverter:
    """ZMX 数据到 OpticalElement 的转换器
    
    将 ZmxDataModel 转换为项目的 OpticalElement 列表。
    
    参数:
        data_model: ZmxDataModel 对象，包含从 ZMX 文件解析的数据
    
    属性:
        FOLD_ANGLE_THRESHOLD: 折叠镜角度阈值（度），默认 5.0
            - 倾斜角度 >= 此阈值的反射镜被识别为折叠镜
            - 倾斜角度 < 此阈值的反射镜被视为失调
    
    示例:
        >>> from sequential_system.zmx_parser import ZmxParser
        >>> from sequential_system.zmx_converter import ElementConverter
        >>> 
        >>> # 解析 ZMX 文件
        >>> parser = ZmxParser("system.zmx")
        >>> data_model = parser.parse()
        >>> 
        >>> # 转换为 OpticalElement 列表
        >>> converter = ElementConverter(data_model)
        >>> elements = converter.convert()
        >>> 
        >>> # 获取带元数据的转换结果
        >>> converted_elements = converter.get_converted_elements()
        >>> for ce in converted_elements:
        ...     print(f"{ce.element_type}: ZMX Surface {ce.zmx_surface_index}")
        >>> 
        >>> # 生成 Python 代码
        >>> code = converter.generate_code()
        >>> print(code)
    
    验证需求:
        - Requirements 5.7: 折叠镜识别（tilt >= 5°）
        - Requirements 5.8: 失调识别（tilt < 5°）
        - Requirements 7.1: 生成正确顺序的 OpticalElement 列表
    
    注意:
        is_fold 参数必须始终为 False，不再根据角度阈值判断。
        所有反射镜都使用完整光线追迹计算 OPD。
    """
    
    # 折叠镜角度阈值（度）- 仅用于 is_fold_mirror 标记，不影响 is_fold 参数
    # is_fold 参数始终为 False
    FOLD_ANGLE_THRESHOLD: float = 5.0
    
    def __init__(self, data_model: 'ZmxDataModel'):
        """初始化 ElementConverter
        
        参数:
            data_model: ZmxDataModel 对象，包含从 ZMX 文件解析的数据
        
        说明:
            初始化时会创建内部状态变量：
            - _data_model: 存储输入的数据模型
            - _converted_elements: 存储转换后的元件列表
            - _accumulated_transform: 存储累积的坐标变换
            - _skip_next_coordbrk_count: 需要跳过的坐标断点数量
        
        示例:
            >>> from sequential_system.zmx_parser import ZmxParser
            >>> parser = ZmxParser("system.zmx")
            >>> data_model = parser.parse()
            >>> converter = ElementConverter(data_model)
        """
        self._data_model = data_model
        self._converted_elements: List[ConvertedElement] = []
        self._accumulated_transform = CoordinateTransform()
        self._skip_next_coordbrk_count = 0  # 需要跳过的坐标断点数量
    
    def convert(self) -> List[Any]:
        """执行转换
        
        将 ZmxDataModel 中的表面数据转换为 OpticalElement 列表。
        
        返回:
            List[OpticalElement]: 转换后的光学元件列表，按光学顺序排列
        
        说明:
            转换流程：
            1. 重置内部状态（清空已转换元件列表，重置坐标变换）
            2. 调用 _process_surfaces() 处理所有表面
            3. 返回转换后的元件列表
            
            转换规则：
            - COORDBRK 表面：累积坐标变换，不生成元件
            - MIRROR 表面：根据曲率和圆锥常数创建对应类型的反射镜
            - 折射表面：创建透镜元件（简化处理）
        
        示例:
            >>> converter = ElementConverter(data_model)
            >>> elements = converter.convert()
            >>> print(f"共转换 {len(elements)} 个元件")
            >>> for elem in elements:
            ...     print(f"  {type(elem).__name__}: thickness={elem.thickness} mm")
        
        **Validates: Requirements 7.1**
        """
        # 重置内部状态
        self._converted_elements = []
        self._accumulated_transform.reset()
        self._skip_next_coordbrk_count = 0
        
        # 处理所有表面
        self._process_surfaces()
        
        # 返回元件列表
        return [ce.element for ce in self._converted_elements]
    
    def get_converted_elements(self) -> List[ConvertedElement]:
        """获取带元数据的转换结果
        
        返回包含元件对象和元数据的 ConvertedElement 列表。
        
        返回:
            List[ConvertedElement]: 转换后的元件数据列表
        
        说明:
            每个 ConvertedElement 包含：
            - element: 光学元件对象
            - zmx_surface_index: 原始 ZMX 表面索引
            - zmx_comment: 原始注释
            - is_fold_mirror: 是否为折叠镜
            - fold_angle_deg: 折叠角度（度）
            
            如果尚未调用 convert()，返回空列表。
        
        示例:
            >>> converter = ElementConverter(data_model)
            >>> converter.convert()
            >>> converted = converter.get_converted_elements()
            >>> for ce in converted:
            ...     print(f"{ce.element_type}: {ce.zmx_comment}")
            ...     if ce.is_fold_mirror:
            ...         print(f"  折叠角度: {ce.fold_angle_deg}°")
        """
        return self._converted_elements
    
    def generate_code(self, include_imports: bool = True) -> str:
        """生成 Python 源代码
        
        从转换后的元件列表生成可执行的 Python 代码。
        
        参数:
            include_imports: 是否包含 import 语句，默认 True
        
        返回:
            str: Python 源代码字符串
        
        说明:
            生成的代码包含：
            - import 语句（如果 include_imports=True）
            - 元件创建代码，每个元件一行
            - 注释，包含原始 ZMX 表面索引和折叠角度
            
            如果尚未调用 convert()，返回空字符串或仅包含 import 语句。
        
        示例:
            >>> converter = ElementConverter(data_model)
            >>> converter.convert()
            >>> code = converter.generate_code()
            >>> print(code)
            # 输出类似：
            # from gaussian_beam_simulation.optical_elements import (
            #     FlatMirror, ParabolicMirror, SphericalMirror, ThinLens
            # )
            # 
            # # ZMX Surface 3: M1 (Fold Mirror, 45.0°)
            # m1 = FlatMirror(
            #     thickness=100.0,
            #     semi_aperture=25.0,
            #     tilt_x=0.7853981633974483,
            #     is_fold=True,
            # )
        
        **Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7**
        """
        # 使用 CodeGenerator 生成代码
        generator = CodeGenerator(self._converted_elements)
        return generator.generate(include_imports=include_imports)
    
    # =========================================================================
    # 内部处理方法
    # =========================================================================
    
    def _process_surfaces(self) -> None:
        """处理所有表面
        
        遍历 ZmxDataModel 中的所有表面，根据表面类型调用相应的处理方法。
        
        说明:
            处理顺序按表面索引从小到大排序。
            
            跳过规则：
            - 物面（index=0）：通常是无穷远物体，应跳过
            - 像面（最后一个表面）：是探测器位置，应跳过
            - 空气间隔（无材料的标准表面且非反射镜）：应跳过
            
            表面类型处理：
            - coordinate_break: 调用 _process_coordinate_break()
            - 反射镜表面: 调用 _process_mirror_surface()
            - 折射表面: 调用 _process_refractive_surface()
        
        **Validates: Requirements 7.1, 7.2, 7.4, 7.5**
        """
        # 按索引排序处理所有表面
        sorted_indices = sorted(self._data_model.surfaces.keys())
        
        if len(sorted_indices) == 0:
            return
        
        # 获取最大索引（像面索引）
        max_index = sorted_indices[-1]
        
        for index in sorted_indices:
            surface = self._data_model.surfaces[index]
            
            # 跳过物面（index=0）
            if index == 0:
                continue
            
            # 跳过像面（最后一个表面）
            if index == max_index:
                continue
            
            # 根据表面类型分发处理
            if surface.surface_type == 'coordinate_break':
                self._process_coordinate_break(surface)
            elif surface.is_mirror:
                self._process_mirror_surface(surface)
            else:
                # 折射表面或其他类型
                # 跳过空气间隔（无材料的标准表面且非反射镜且非光阑）
                if self._is_air_gap(surface):
                    continue
                self._process_refractive_surface(surface)
    
    def _is_air_gap(self, surface: 'ZmxSurfaceData') -> bool:
        """判断表面是否为空气间隔
        
        空气间隔是指没有实际光学作用的表面，通常用于定义元件之间的距离。
        
        参数:
            surface: ZmxSurfaceData 对象
        
        返回:
            bool: 如果是空气间隔则返回 True
        
        说明:
            空气间隔的判断条件：
            - 表面类型为 standard
            - 材料为 air 或空
            - 不是反射镜
            - 不是光阑（光阑可能需要保留）
            - 曲率半径为无穷大（平面）
        
        **Validates: Requirements 7.2**
        """
        # 坐标断点不是空气间隔
        if surface.surface_type == 'coordinate_break':
            return False
        
        # 反射镜不是空气间隔
        if surface.is_mirror:
            return False
        
        # 光阑不是空气间隔（可能需要保留）
        if surface.is_stop:
            return False
        
        # 有材料的表面不是空气间隔
        material = surface.material.lower() if surface.material else ""
        if material and material != "air":
            return False
        
        # 有曲率的表面不是空气间隔
        if not np.isinf(surface.radius):
            return False
        
        # 满足所有条件，是空气间隔
        return True
    
    def _process_coordinate_break(self, surface: 'ZmxSurfaceData') -> None:
        """处理坐标断点
        
        累积坐标断点的变换参数，不生成元件。
        
        参数:
            surface: ZmxSurfaceData 对象，表面类型为 coordinate_break
        
        说明:
            坐标断点用于定义偏心和倾斜，其参数会累积到 _accumulated_transform 中，
            在处理后续的光学表面时应用。
            
            坐标断点参数映射（Zemax 约定）：
            - decenter_x: X 方向偏心 (mm)
            - decenter_y: Y 方向偏心 (mm)
            - tilt_x_deg: 绕 X 轴旋转 (度)
            - tilt_y_deg: 绕 Y 轴旋转 (度)
            - tilt_z_deg: 绕 Z 轴旋转 (度)
            - thickness: Z 方向位移 (mm)
            
            在折叠镜序列中，反射后的"恢复"坐标断点会被跳过，
            以避免累积错误的变换。
        
        示例:
            >>> # 处理 45 度折叠镜前的坐标断点
            >>> surface = ZmxSurfaceData(
            ...     index=3,
            ...     surface_type='coordinate_break',
            ...     tilt_x_deg=45.0,
            ...     thickness=0.0,
            ... )
            >>> converter._process_coordinate_break(surface)
            >>> print(f"累积倾斜: {converter._accumulated_transform.tilt_x_deg:.1f}°")
            累积倾斜: 45.0°
        
        **Validates: Requirements 3.4, 3.5, 3.6, 6.2**
        """
        # 检查是否需要跳过此坐标断点
        if self._skip_next_coordbrk_count > 0:
            self._skip_next_coordbrk_count -= 1
            return
        
        # 从 surface 中提取坐标断点参数
        dx = surface.decenter_x
        dy = surface.decenter_y
        dz = surface.thickness  # DISZ 作为 Z 方向位移
        rx_deg = surface.tilt_x_deg
        ry_deg = surface.tilt_y_deg
        rz_deg = surface.tilt_z_deg
        
        # 累积坐标变换
        self._accumulated_transform.apply_coordinate_break(
            dx=dx,
            dy=dy,
            dz=dz,
            rx_deg=rx_deg,
            ry_deg=ry_deg,
            rz_deg=rz_deg,
        )
    
    def _process_mirror_surface(self, surface: 'ZmxSurfaceData') -> None:
        """处理反射镜表面
        
        根据反射镜的曲率和圆锥常数创建对应类型的反射镜元件。
        
        参数:
            surface: ZmxSurfaceData 对象，is_mirror=True
        
        说明:
            反射镜类型判断规则：
            - radius = inf: FlatMirror
            - conic = -1: ParabolicMirror
            - 其他: SphericalMirror
            
            处理流程：
            1. 获取当前累积的坐标变换（来自前面的 COORDBRK）
            2. 计算反射后的传播距离
            3. 创建反射镜元件
            4. 创建 ConvertedElement 并添加到列表
            5. 重置累积变换（反射后坐标系改变）
            6. 标记需要跳过的后续坐标断点数量
        
        示例:
            >>> # 处理 45 度折叠镜
            >>> surface = ZmxSurfaceData(
            ...     index=4,
            ...     surface_type='standard',
            ...     radius=np.inf,
            ...     is_mirror=True,
            ...     comment="M1",
            ... )
            >>> # 假设之前已处理了 45 度坐标断点
            >>> converter._process_mirror_surface(surface)
        
        **Validates: Requirements 5.1, 5.2, 5.3, 6.1, 6.3, 6.4, 6.5**
        """
        # 获取当前累积的坐标变换
        tilt_x_rad = self._accumulated_transform.tilt_x_rad
        tilt_y_rad = self._accumulated_transform.tilt_y_rad
        decenter_x = self._accumulated_transform.decenter_x
        decenter_y = self._accumulated_transform.decenter_y
        
        # 计算反射后的传播距离
        thickness = self._calculate_thickness_after_reflection(surface.index)
        
        # 创建反射镜元件
        mirror = self._create_mirror_element(
            surface=surface,
            thickness=thickness,
            tilt_x=tilt_x_rad,
            tilt_y=tilt_y_rad,
            decenter_x=decenter_x,
            decenter_y=decenter_y,
        )
        
        # 判断是否为折叠镜，计算折叠角度
        tilt_x_deg = np.rad2deg(tilt_x_rad)
        tilt_y_deg = np.rad2deg(tilt_y_rad)
        is_fold = self._is_fold_mirror(tilt_x_deg, tilt_y_deg)
        fold_angle_deg = max(abs(tilt_x_deg), abs(tilt_y_deg)) if is_fold else 0.0
        
        # 创建 ConvertedElement 并添加到列表
        converted = ConvertedElement(
            element=mirror,
            zmx_surface_index=surface.index,
            zmx_comment=surface.comment,
            is_fold_mirror=is_fold,
            fold_angle_deg=fold_angle_deg,
        )
        self._converted_elements.append(converted)
        
        # 重置累积变换（反射后坐标系改变）
        self._accumulated_transform.reset()
        
        # 标记需要跳过的后续坐标断点
        # 在折叠镜序列中，反射后通常有一个"恢复"坐标断点
        # 我们需要跳过它，以避免累积错误的变换
        self._skip_next_coordbrk_count = self._count_post_reflection_coordbrks(surface.index)
    
    def _process_refractive_surface(self, surface: 'ZmxSurfaceData') -> None:
        """处理折射表面
        
        处理折射表面，创建透镜元件（简化处理）。
        
        参数:
            surface: ZmxSurfaceData 对象，非反射镜表面
        
        说明:
            当前实现为简化处理，将折射表面转换为薄透镜。
            完整实现需要考虑表面配对和材料属性。
        
        **Validates: Requirements 5.4**
        """
        # TODO: 在后续任务中实现完整的折射表面处理逻辑
        pass
    
    def _is_fold_mirror(self, tilt_x_deg: float, tilt_y_deg: float) -> bool:
        """判断是否为折叠镜
        
        根据倾斜角度判断反射镜是否为折叠镜。
        
        参数:
            tilt_x_deg: 绕 X 轴旋转角度（度）
            tilt_y_deg: 绕 Y 轴旋转角度（度）
        
        返回:
            bool: 如果倾斜角度 >= FOLD_ANGLE_THRESHOLD 则返回 True
        
        说明:
            判断规则：
            - 计算总倾斜角度 = max(|tilt_x|, |tilt_y|)
            - 如果总倾斜角度 >= FOLD_ANGLE_THRESHOLD (5°)，则为折叠镜
            - 否则为失调
        
        示例:
            >>> converter = ElementConverter(data_model)
            >>> converter._is_fold_mirror(45.0, 0.0)  # 45° 倾斜
            True
            >>> converter._is_fold_mirror(2.0, 1.0)   # 小角度倾斜
            False
        
        **Validates: Requirements 5.7, 5.8, 7.3**
        """
        # 计算最大倾斜角度
        max_tilt = max(abs(tilt_x_deg), abs(tilt_y_deg))
        return max_tilt >= self.FOLD_ANGLE_THRESHOLD
    
    def _create_mirror_element(
        self,
        surface: 'ZmxSurfaceData',
        thickness: float,
        tilt_x: float,  # 弧度
        tilt_y: float,  # 弧度
        decenter_x: float,  # mm
        decenter_y: float,  # mm
    ) -> Any:
        """创建反射镜元件
        
        根据反射镜的曲率半径和圆锥常数创建对应类型的反射镜元件。
        
        参数:
            surface: ZmxSurfaceData 对象，包含反射镜的几何参数
            thickness: 传播距离 (mm)，到下一元件的间距
            tilt_x: 绕 X 轴旋转角度 (弧度)
            tilt_y: 绕 Y 轴旋转角度 (弧度)
            decenter_x: X 方向偏心 (mm)
            decenter_y: Y 方向偏心 (mm)
        
        返回:
            OpticalElement: 创建的反射镜元件，类型为以下之一：
                - FlatMirror: 当 radius = inf 时
                - ParabolicMirror: 当 conic = -1 时
                - SphericalMirror: 其他情况
        
        说明:
            反射镜类型判断规则（按优先级）：
            1. radius = inf（无穷大）: 创建 FlatMirror（平面镜）
            2. conic = -1: 创建 ParabolicMirror（抛物面镜）
            3. 其他: 创建 SphericalMirror（球面镜）
            
            折叠镜判断：
            - 根据倾斜角度判断是否为折叠镜（用于 is_fold_mirror 标记）
            - is_fold 参数始终为 False（使用完整光线追迹）
        
        示例:
            >>> # 创建平面折叠镜（45度倾斜）
            >>> surface = ZmxSurfaceData(index=3, radius=np.inf, is_mirror=True)
            >>> mirror = converter._create_mirror_element(
            ...     surface=surface,
            ...     thickness=100.0,
            ...     tilt_x=np.pi/4,  # 45度
            ...     tilt_y=0.0,
            ...     decenter_x=0.0,
            ...     decenter_y=0.0,
            ... )
            >>> print(type(mirror).__name__)
            FlatMirror
            >>> print(mirror.is_fold)
            False
            
            >>> # 创建抛物面镜
            >>> surface = ZmxSurfaceData(index=5, radius=200.0, conic=-1.0, is_mirror=True)
            >>> mirror = converter._create_mirror_element(
            ...     surface=surface,
            ...     thickness=150.0,
            ...     tilt_x=0.0,
            ...     tilt_y=0.0,
            ...     decenter_x=0.0,
            ...     decenter_y=0.0,
            ... )
            >>> print(type(mirror).__name__)
            ParabolicMirror
        
        **Validates: Requirements 5.1, 5.2, 5.3, 5.7, 5.8**
        """
        # 获取表面参数
        radius = surface.radius
        conic = surface.conic
        semi_aperture = surface.semi_diameter if surface.semi_diameter > 0 else 10.0  # 默认值
        
        # 判断是否为折叠镜
        tilt_x_deg = np.rad2deg(tilt_x)
        tilt_y_deg = np.rad2deg(tilt_y)
        is_fold = self._is_fold_mirror(tilt_x_deg, tilt_y_deg)
        
        # 根据曲率半径和圆锥常数判断反射镜类型
        # 规则 1: radius = inf -> FlatMirror
        if np.isinf(radius):
            return FlatMirror(
                thickness=thickness,
                semi_aperture=semi_aperture,
                tilt_x=tilt_x,
                tilt_y=tilt_y,
                decenter_x=decenter_x,
                decenter_y=decenter_y,
                is_fold=is_fold,
                name=surface.comment if surface.comment else None,
            )
        
        # 规则 2: conic = -1 -> ParabolicMirror
        # 使用容差比较浮点数
        if np.isclose(conic, -1.0, rtol=1e-6, atol=1e-9):
            # 抛物面镜的焦距 = R/2
            parent_focal_length = radius / 2.0
            return ParabolicMirror(
                thickness=thickness,
                semi_aperture=semi_aperture,
                parent_focal_length=parent_focal_length,
                tilt_x=tilt_x,
                tilt_y=tilt_y,
                decenter_x=decenter_x,
                decenter_y=decenter_y,
                is_fold=is_fold,
                name=surface.comment if surface.comment else None,
            )
        
        # 规则 3: 其他 -> SphericalMirror
        return SphericalMirror(
            thickness=thickness,
            semi_aperture=semi_aperture,
            radius_of_curvature=radius,
            tilt_x=tilt_x,
            tilt_y=tilt_y,
            decenter_x=decenter_x,
            decenter_y=decenter_y,
            is_fold=is_fold,
            name=surface.comment if surface.comment else None,
        )
    
    def _calculate_thickness_after_reflection(
        self,
        current_index: int,
    ) -> float:
        """计算反射后的传播距离
        
        根据反射镜后的坐标断点计算传播距离。
        
        参数:
            current_index: 当前反射镜表面的索引
        
        返回:
            float: 传播距离（mm），始终为正值
        
        说明:
            在折叠光路中，反射后的传播距离通常由后续坐标断点的负厚度表示。
            
            典型的折叠镜序列：
            - SURF N: COORDBRK (前，定义倾斜)
            - SURF N+1: MIRROR (反射镜，thickness=0)
            - SURF N+2: COORDBRK (后，thickness 为负值表示反射方向传播)
            
            计算规则：
            1. 查找反射镜后的下一个表面
            2. 如果是坐标断点且厚度为负，取绝对值作为传播距离
            3. 如果不是坐标断点或厚度为正，使用反射镜表面的 thickness
            4. 如果没有后续表面，返回 0
        
        示例:
            >>> # 假设 surface 5 是 COORDBRK，thickness = -20
            >>> # 则反射镜（surface 4）后的传播距离为 20 mm
            >>> thickness = converter._calculate_thickness_after_reflection(4)
            >>> print(f"传播距离: {thickness} mm")
            传播距离: 20.0 mm
        
        **Validates: Requirements 6.1, 6.3, 6.4**
        """
        # 获取当前反射镜表面
        current_surface = self._data_model.get_surface(current_index)
        if current_surface is None:
            return 0.0
        
        # 查找反射镜后的下一个表面
        # 按索引排序获取所有表面索引
        sorted_indices = sorted(self._data_model.surfaces.keys())
        
        # 找到当前索引在排序列表中的位置
        try:
            current_pos = sorted_indices.index(current_index)
        except ValueError:
            return 0.0
        
        # 检查是否有下一个表面
        if current_pos + 1 >= len(sorted_indices):
            # 没有后续表面，使用当前表面的 thickness
            return abs(current_surface.thickness)
        
        # 获取下一个表面
        next_index = sorted_indices[current_pos + 1]
        next_surface = self._data_model.get_surface(next_index)
        
        if next_surface is None:
            return abs(current_surface.thickness)
        
        # 检查下一个表面是否为坐标断点
        if next_surface.surface_type == 'coordinate_break':
            # 坐标断点的厚度
            cb_thickness = next_surface.thickness
            
            # 负厚度表示反射方向传播，取绝对值
            if cb_thickness < 0:
                return abs(cb_thickness)
            elif cb_thickness > 0:
                # 正厚度也是有效的传播距离
                return cb_thickness
            else:
                # 厚度为 0，继续查找下一个表面
                # 可能有多个连续的坐标断点
                return self._find_thickness_in_subsequent_surfaces(next_index)
        else:
            # 下一个表面不是坐标断点
            # 使用当前反射镜表面的 thickness（如果非零）
            if current_surface.thickness != 0:
                return abs(current_surface.thickness)
            # 否则使用下一个表面的 thickness
            return abs(next_surface.thickness)
    
    def _find_thickness_in_subsequent_surfaces(
        self,
        start_index: int,
    ) -> float:
        """在后续表面中查找传播距离
        
        当坐标断点的厚度为 0 时，继续查找后续表面的厚度。
        
        参数:
            start_index: 起始表面索引
        
        返回:
            float: 找到的传播距离（mm）
        
        说明:
            遍历后续表面，查找第一个非零厚度值。
            对于负厚度，取绝对值。
        """
        sorted_indices = sorted(self._data_model.surfaces.keys())
        
        try:
            start_pos = sorted_indices.index(start_index)
        except ValueError:
            return 0.0
        
        # 从下一个表面开始查找
        for i in range(start_pos + 1, len(sorted_indices)):
            surface_index = sorted_indices[i]
            surface = self._data_model.get_surface(surface_index)
            
            if surface is None:
                continue
            
            # 如果是坐标断点，检查其厚度
            if surface.surface_type == 'coordinate_break':
                if surface.thickness != 0:
                    return abs(surface.thickness)
            else:
                # 非坐标断点表面，使用其厚度
                if surface.thickness != 0:
                    return abs(surface.thickness)
                # 如果厚度为 0，继续查找
        
        return 0.0
    
    def _count_post_reflection_coordbrks(
        self,
        mirror_index: int,
    ) -> int:
        """计算反射镜后需要跳过的坐标断点数量
        
        在折叠镜序列中，反射后通常有一个"恢复"坐标断点，
        用于将坐标系恢复到与反射后光轴平行的状态。
        
        参数:
            mirror_index: 反射镜表面的索引
        
        返回:
            int: 需要跳过的坐标断点数量
        
        说明:
            典型的折叠镜序列：
            - SURF N: COORDBRK (前，定义倾斜) - 不跳过
            - SURF N+1: MIRROR (反射镜)
            - SURF N+2: COORDBRK (后，恢复坐标系) - 需要跳过
            
            判断规则：
            1. 查找反射镜后的下一个表面
            2. 如果是坐标断点，则需要跳过 1 个
            3. 如果不是坐标断点，则不需要跳过
        
        **Validates: Requirements 3.5, 3.6, 6.2**
        """
        sorted_indices = sorted(self._data_model.surfaces.keys())
        
        try:
            mirror_pos = sorted_indices.index(mirror_index)
        except ValueError:
            return 0
        
        # 检查是否有下一个表面
        if mirror_pos + 1 >= len(sorted_indices):
            return 0
        
        # 获取下一个表面
        next_index = sorted_indices[mirror_pos + 1]
        next_surface = self._data_model.get_surface(next_index)
        
        if next_surface is None:
            return 0
        
        # 如果下一个表面是坐标断点，需要跳过它
        if next_surface.surface_type == 'coordinate_break':
            return 1
        
        return 0


# =============================================================================
# CodeGenerator 类
# =============================================================================

class CodeGenerator:
    """Python 代码生成器
    
    从转换后的元件列表生成 Python 源代码。
    
    参数:
        converted_elements: ConvertedElement 列表，包含转换后的元件及其元数据
    
    属性:
        INDENT: 缩进字符串，默认为 4 个空格
    
    示例:
        >>> from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        >>> from gaussian_beam_simulation.optical_elements import FlatMirror
        >>> 
        >>> # 创建转换后的元件
        >>> mirror = FlatMirror(
        ...     thickness=100.0,
        ...     semi_aperture=25.0,
        ...     tilt_x=0.7853981633974483,
        ...     is_fold=True,
        ... )
        >>> converted = ConvertedElement(
        ...     element=mirror,
        ...     zmx_surface_index=3,
        ...     zmx_comment="M1",
        ...     is_fold_mirror=True,
        ...     fold_angle_deg=45.0,
        ... )
        >>> 
        >>> # 生成代码
        >>> generator = CodeGenerator([converted])
        >>> code = generator.generate()
        >>> print(code)
        from gaussian_beam_simulation.optical_elements import (
            FlatMirror,
        )
        
        # ZMX Surface 3: M1 (Fold Mirror, 45.0°)
        m1 = FlatMirror(
            thickness=100.0,
            semi_aperture=25.0,
            tilt_x=0.7853981633974483,
            is_fold=True,
        )
    
    验证需求:
        - Requirements 9.1: 提供 generate_code() 方法返回 Python 源代码
        - Requirements 9.7: 生成的代码包含必要的 import 语句
    """
    
    INDENT = "    "  # 4 空格缩进
    
    def __init__(self, converted_elements: List[ConvertedElement]):
        """初始化 CodeGenerator
        
        参数:
            converted_elements: ConvertedElement 列表，包含转换后的元件及其元数据
        
        示例:
            >>> generator = CodeGenerator(converted_elements)
        """
        self.elements = converted_elements
    
    def generate(self, include_imports: bool = True) -> str:
        """生成完整的 Python 代码
        
        参数:
            include_imports: 是否包含 import 语句，默认 True
        
        返回:
            str: Python 源代码字符串
        
        说明:
            生成的代码包含：
            1. import 语句（如果 include_imports=True）
            2. 每个元件的创建代码，包含：
               - 注释：原始 ZMX 表面索引和名称
               - 折叠镜注释：折叠角度
               - 元件创建语句：包含所有参数
        
        示例:
            >>> generator = CodeGenerator(converted_elements)
            >>> code = generator.generate()
            >>> print(code)
            >>> 
            >>> # 不包含 import 语句
            >>> code_no_imports = generator.generate(include_imports=False)
        
        **Validates: Requirements 9.1, 9.7**
        """
        lines = []
        
        # 生成 import 语句
        if include_imports:
            imports = self._generate_imports()
            if imports:
                lines.append(imports)
                lines.append("")  # 空行分隔
        
        # 如果没有元件，返回注释
        if not self.elements:
            lines.append("# 没有转换的元件")
            return "\n".join(lines)
        
        # 生成每个元件的代码
        for i, elem in enumerate(self.elements):
            if i > 0:
                lines.append("")  # 元件之间空行分隔
            element_code = self._generate_element_code(elem)
            lines.append(element_code)
        
        return "\n".join(lines)
    
    def _generate_imports(self) -> str:
        """生成 import 语句
        
        根据转换后的元件类型，生成必要的 import 语句。
        
        返回:
            str: import 语句字符串
        
        说明:
            - 只导入实际使用的元件类型
            - 按字母顺序排列导入的类
            - 使用多行格式以提高可读性
        
        示例:
            >>> generator = CodeGenerator(converted_elements)
            >>> imports = generator._generate_imports()
            >>> print(imports)
            from gaussian_beam_simulation.optical_elements import (
                FlatMirror,
                ParabolicMirror,
            )
        
        **Validates: Requirements 9.7**
        """
        if not self.elements:
            return ""
        
        # 收集所有使用的元件类型
        element_types = set()
        for elem in self.elements:
            if elem.element is not None:
                element_types.add(type(elem.element).__name__)
        
        if not element_types:
            return ""
        
        # 按字母顺序排列
        sorted_types = sorted(element_types)
        
        # 生成 import 语句
        lines = ["from gaussian_beam_simulation.optical_elements import ("]
        for element_type in sorted_types:
            lines.append(f"{self.INDENT}{element_type},")
        lines.append(")")
        
        return "\n".join(lines)
    
    def _generate_element_code(self, elem: ConvertedElement) -> str:
        """生成单个元件的代码
        
        参数:
            elem: ConvertedElement 对象
        
        返回:
            str: 元件创建代码字符串
        
        说明:
            生成的代码包含：
            1. 注释行：ZMX 表面索引和原始注释
            2. 折叠镜注释：如果是折叠镜，显示折叠角度
            3. 元件创建语句：变量名 = 类名(参数...)
            
            参数包含：
            - thickness: 传播距离
            - semi_aperture: 半口径
            - tilt_x, tilt_y: 倾斜角度（如果非零）
            - decenter_x, decenter_y: 偏心（如果非零）
            - is_fold: 是否为折叠镜
            - 其他类型特定参数（如 radius_of_curvature, parent_focal_length）
        
        **Validates: Requirements 9.2, 9.3, 9.4, 9.5, 9.6**
        """
        lines = []
        
        # 生成注释
        comment = elem.get_code_comment()
        lines.append(comment)
        
        # 获取元件对象
        element = elem.element
        if element is None:
            lines.append("# 元件为空")
            return "\n".join(lines)
        
        # 获取元件类型和变量名
        element_type = type(element).__name__
        var_name = self._generate_variable_name(elem)
        
        # 开始元件创建语句
        lines.append(f"{var_name} = {element_type}(")
        
        # 生成参数
        params = self._get_element_params(element, elem)
        for param_name, param_value in params:
            formatted_value = self._format_value(param_value)
            lines.append(f"{self.INDENT}{param_name}={formatted_value},")
        
        # 结束元件创建语句
        lines.append(")")
        
        return "\n".join(lines)
    
    def _generate_variable_name(self, elem: ConvertedElement) -> str:
        """生成变量名
        
        参数:
            elem: ConvertedElement 对象
        
        返回:
            str: 变量名
        
        说明:
            变量名生成规则：
            1. 如果有注释，使用注释的小写形式（去除空格和特殊字符）
            2. 如果没有注释，使用 "element_{index}" 格式
        """
        if elem.has_comment:
            # 使用注释生成变量名
            name = elem.zmx_comment.strip()
            # 转换为小写，替换空格和特殊字符为下划线
            name = name.lower()
            name = ''.join(c if c.isalnum() else '_' for c in name)
            # 去除连续的下划线
            while '__' in name:
                name = name.replace('__', '_')
            # 去除首尾下划线
            name = name.strip('_')
            if name:
                return name
        
        # 使用默认变量名
        return f"element_{elem.zmx_surface_index}"
    
    def _get_element_params(
        self,
        element: Any,
        elem: ConvertedElement,
    ) -> List[tuple]:
        """获取元件参数列表
        
        参数:
            element: 光学元件对象
            elem: ConvertedElement 对象
        
        返回:
            List[tuple]: 参数名和值的列表 [(name, value), ...]
        
        说明:
            根据元件类型返回相应的参数列表。
            参数生成规则：
            - thickness 和 semi_aperture 始终生成
            - tilt_x, tilt_y 只在非零时生成
            - decenter_x, decenter_y 只在非零时生成
            - is_fold 只在为 True 时生成
        
        **Validates: Requirements 9.3**
        """
        params = []
        element_type = type(element).__name__
        
        # 通用参数（始终生成）
        params.append(("thickness", element.thickness))
        params.append(("semi_aperture", element.semi_aperture))
        
        # 类型特定参数
        if element_type == "ParabolicMirror":
            params.append(("parent_focal_length", element.parent_focal_length))
        elif element_type == "SphericalMirror":
            params.append(("radius_of_curvature", element.radius_of_curvature))
        
        # 倾斜参数（只在非零时生成）
        if hasattr(element, 'tilt_x') and element.tilt_x != 0.0:
            params.append(("tilt_x", element.tilt_x))
        if hasattr(element, 'tilt_y') and element.tilt_y != 0.0:
            params.append(("tilt_y", element.tilt_y))
        
        # 偏心参数（只在非零时生成）
        if hasattr(element, 'decenter_x') and element.decenter_x != 0.0:
            params.append(("decenter_x", element.decenter_x))
        if hasattr(element, 'decenter_y') and element.decenter_y != 0.0:
            params.append(("decenter_y", element.decenter_y))
        
        # is_fold 参数（只在为 True 时生成）
        if hasattr(element, 'is_fold') and element.is_fold:
            params.append(("is_fold", True))
        
        return params
    
    def _format_value(self, value: Any) -> str:
        """格式化参数值
        
        参数:
            value: 参数值
        
        返回:
            str: 格式化后的字符串
        
        说明:
            - 布尔值：True/False
            - 浮点数：保留足够精度，去除不必要的尾随零
            - 字符串：添加引号
            - 其他：使用 repr()
        """
        if isinstance(value, bool):
            return str(value)
        elif isinstance(value, float):
            return self._format_float(value)
        elif isinstance(value, str):
            return repr(value)
        else:
            return repr(value)
    
    def _format_float(self, value: float, precision: int = 6) -> str:
        """格式化浮点数
        
        参数:
            value: 浮点数值
            precision: 精度（小数位数），默认 6
        
        返回:
            str: 格式化后的字符串
        
        说明:
            - 对于整数值，不显示小数点（如 100.0 -> 100.0）
            - 对于小数值，保留足够精度
            - 去除不必要的尾随零
            - 对于非常小或非常大的数，使用科学计数法
        """
        if np.isinf(value):
            return "float('inf')" if value > 0 else "float('-inf')"
        if np.isnan(value):
            return "float('nan')"
        
        # 检查是否为整数值
        if value == int(value) and abs(value) < 1e10:
            return f"{int(value)}.0"
        
        # 使用足够的精度
        formatted = f"{value:.{precision}f}"
        
        # 去除尾随零，但保留至少一位小数
        if '.' in formatted:
            formatted = formatted.rstrip('0')
            if formatted.endswith('.'):
                formatted += '0'
        
        return formatted


# =============================================================================
# 便捷函数
# =============================================================================

def load_zmx_file(filepath: str) -> List[Any]:
    """从 ZMX 文件加载光学元件
    
    这是一个便捷函数，用于快速从 Zemax .zmx 文件加载光学元件。
    内部会自动创建 ZmxParser 和 ElementConverter 实例。
    
    参数:
        filepath: .zmx 文件路径
    
    返回:
        List[OpticalElement]: 光学元件列表，按光学顺序排列
    
    异常:
        FileNotFoundError: 文件不存在
        ValueError: 文件格式错误或编码不支持
        ZmxParseError: ZMX 文件解析错误
        ZmxUnsupportedError: 不支持的 ZMX 特性
    
    示例:
        >>> elements = load_zmx_file("system.zmx")
        >>> for elem in elements:
        ...     print(f"{type(elem).__name__}: thickness={elem.thickness} mm")
        FlatMirror: thickness=100.0 mm
        ParabolicMirror: thickness=150.0 mm
        
        >>> # 将元件添加到 SequentialOpticalSystem
        >>> from sequential_system import SequentialOpticalSystem
        >>> system = SequentialOpticalSystem()
        >>> for elem in elements:
        ...     system.add_element(elem)
    
    **Validates: Requirements 7.1**
    """
    from sequential_system.zmx_parser import ZmxParser
    
    parser = ZmxParser(filepath)
    data_model = parser.parse()
    converter = ElementConverter(data_model)
    return converter.convert()


def load_zmx_and_generate_code(filepath: str) -> tuple:
    """从 ZMX 文件加载光学元件并生成代码
    
    这是一个便捷函数，用于从 Zemax .zmx 文件加载光学元件，
    同时生成可复制的 Python 源代码。
    
    参数:
        filepath: .zmx 文件路径
    
    返回:
        Tuple[List[OpticalElement], str]: 元组包含：
            - 光学元件列表，按光学顺序排列
            - Python 源代码字符串，可直接执行
    
    异常:
        FileNotFoundError: 文件不存在
        ValueError: 文件格式错误或编码不支持
        ZmxParseError: ZMX 文件解析错误
        ZmxUnsupportedError: 不支持的 ZMX 特性
    
    示例:
        >>> elements, code = load_zmx_and_generate_code("system.zmx")
        >>> 
        >>> # 查看元件
        >>> print(f"共 {len(elements)} 个元件")
        共 3 个元件
        >>> 
        >>> # 查看生成的代码
        >>> print(code)
        from gaussian_beam_simulation.optical_elements import (
            FlatMirror,
            ParabolicMirror,
        )
        
        # ZMX Surface 3: M1 (Fold Mirror, 45.0°)
        m1 = FlatMirror(
            thickness=100.0,
            semi_aperture=25.0,
            tilt_x=0.7853981633974483,
            is_fold=True,
        )
        
        # ZMX Surface 5: M2
        m2 = ParabolicMirror(
            thickness=150.0,
            semi_aperture=30.0,
            parent_focal_length=200.0,
        )
        >>> 
        >>> # 将代码保存到文件
        >>> with open("generated_system.py", "w") as f:
        ...     f.write(code)
    
    **Validates: Requirements 7.1, 9.1**
    """
    from sequential_system.zmx_parser import ZmxParser
    
    parser = ZmxParser(filepath)
    data_model = parser.parse()
    converter = ElementConverter(data_model)
    elements = converter.convert()
    code = converter.generate_code()
    return elements, code


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "CoordinateTransform",
    "ConvertedElement",
    "ElementConverter",
    "CodeGenerator",
    "load_zmx_file",
    "load_zmx_and_generate_code",
]
