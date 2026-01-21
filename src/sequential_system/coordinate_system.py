"""
Zemax 光轴追踪与坐标转换模块

本模块实现基于 Zemax 序列模式的光路结构定义，将 Zemax 表面序列转换为
optiland 全局坐标系中的光学系统定义。

核心设计原则：
- 统一处理：所有表面类型（包括连续坐标断点、空气面）使用统一算法处理
- 简洁代码：垂直接入 optiland 库，最小化中间层代码
- Zemax 兼容：严格遵循 Zemax 序列模式的坐标系演化规则

主要类：
- CurrentCoordinateSystem: 当前坐标系状态（不可变）
- CoordinateBreakProcessor: 坐标断点处理器
- GlobalSurfaceDefinition: 全局坐标表面定义
- SurfaceTraversalAlgorithm: 表面遍历算法
- ZemaxToOptilandConverter: Zemax-optiland 转换器

使用示例：
    >>> from sequential_system.coordinate_system import (
    ...     CurrentCoordinateSystem,
    ...     CoordinateBreakProcessor,
    ...     SurfaceTraversalAlgorithm,
    ...     ZemaxToOptilandConverter,
    ... )
    >>> 
    >>> # 创建初始坐标系
    >>> cs = CurrentCoordinateSystem.identity()
    >>> print(f"原点: {cs.origin}")
    >>> print(f"Z轴: {cs.z_axis}")
    >>> 
    >>> # 应用坐标断点变换
    >>> cs_new = CoordinateBreakProcessor.process(
    ...     cs, decenter_x=0, decenter_y=0,
    ...     tilt_x_rad=np.pi/4, tilt_y_rad=0, tilt_z_rad=0,
    ...     order=0, thickness=100.0
    ... )

作者：混合光学仿真项目
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from sequential_system.zmx_parser import ZmxDataModel, ZmxSurfaceData


# =============================================================================
# CurrentCoordinateSystem 类
# =============================================================================

@dataclass(frozen=True)
class CurrentCoordinateSystem:
    """当前坐标系状态（不可变）
    
    追踪 Zemax 序列模式中"当前坐标系"相对于全局坐标系的位置和姿态。
    该类是不可变的，所有变换方法都返回新实例。
    
    属性:
        origin: 当前坐标系原点在全局坐标系中的位置 (mm)，shape (3,)
        axes: 轴向量矩阵，3×3 矩阵，列向量为 [X, Y, Z] 轴在全局坐标系中的方向
    
    示例:
        >>> cs = CurrentCoordinateSystem.identity()
        >>> print(f"原点: {cs.origin}")
        原点: [0. 0. 0.]
        >>> print(f"Z轴: {cs.z_axis}")
        Z轴: [0. 0. 1.]
        >>> 
        >>> # 沿 Z 轴前进 100mm
        >>> cs_new = cs.advance_along_z(100.0)
        >>> print(f"新原点: {cs_new.origin}")
        新原点: [  0.   0. 100.]
    
    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**
    """
    origin: np.ndarray  # shape (3,)，单位 mm
    axes: np.ndarray    # shape (3, 3)，列向量为 X, Y, Z 轴
    
    def __post_init__(self):
        """验证并转换输入数据"""
        # 由于 frozen=True，需要使用 object.__setattr__
        object.__setattr__(self, 'origin', np.asarray(self.origin, dtype=np.float64))
        object.__setattr__(self, 'axes', np.asarray(self.axes, dtype=np.float64))
        
        # 验证形状
        if self.origin.shape != (3,):
            raise ValueError(f"origin 形状必须为 (3,)，实际为 {self.origin.shape}")
        if self.axes.shape != (3, 3):
            raise ValueError(f"axes 形状必须为 (3, 3)，实际为 {self.axes.shape}")
    
    @classmethod
    def identity(cls) -> "CurrentCoordinateSystem":
        """创建与全局坐标系重合的初始状态
        
        返回:
            原点在 (0, 0, 0)，轴为单位矩阵的坐标系状态
        
        示例:
            >>> cs = CurrentCoordinateSystem.identity()
            >>> np.allclose(cs.origin, [0, 0, 0])
            True
            >>> np.allclose(cs.axes, np.eye(3))
            True
        
        **Validates: Requirements 1.3**
        """
        return cls(
            origin=np.array([0.0, 0.0, 0.0]),
            axes=np.eye(3)
        )
    
    @property
    def x_axis(self) -> np.ndarray:
        """当前 X 轴在全局坐标系中的方向
        
        返回:
            shape (3,) 的单位向量
        
        **Validates: Requirements 1.4**
        """
        return self.axes[:, 0].copy()
    
    @property
    def y_axis(self) -> np.ndarray:
        """当前 Y 轴在全局坐标系中的方向
        
        返回:
            shape (3,) 的单位向量
        
        **Validates: Requirements 1.4**
        """
        return self.axes[:, 1].copy()
    
    @property
    def z_axis(self) -> np.ndarray:
        """当前 Z 轴在全局坐标系中的方向
        
        返回:
            shape (3,) 的单位向量
        
        **Validates: Requirements 1.4**
        """
        return self.axes[:, 2].copy()

    def advance_along_z(self, thickness: float) -> "CurrentCoordinateSystem":
        """沿当前 Z 轴方向前进指定距离
        
        参数:
            thickness: 前进距离 (mm)，可正可负
        
        返回:
            新的坐标系状态（不修改原对象）
        
        示例:
            >>> cs = CurrentCoordinateSystem.identity()
            >>> cs_new = cs.advance_along_z(100.0)
            >>> np.allclose(cs_new.origin, [0, 0, 100])
            True
            >>> 
            >>> # 负厚度向后移动
            >>> cs_back = cs.advance_along_z(-50.0)
            >>> np.allclose(cs_back.origin, [0, 0, -50])
            True
        
        **Validates: Requirements 3.1, 3.3**
        """
        new_origin = self.origin + thickness * self.z_axis
        return CurrentCoordinateSystem(origin=new_origin, axes=self.axes.copy())
    
    def apply_decenter(self, dx: float, dy: float) -> "CurrentCoordinateSystem":
        """应用偏心（沿当前 X、Y 轴平移）
        
        参数:
            dx: X 方向偏心 (mm)
            dy: Y 方向偏心 (mm)
        
        返回:
            新的坐标系状态
        
        示例:
            >>> cs = CurrentCoordinateSystem.identity()
            >>> cs_new = cs.apply_decenter(5.0, 10.0)
            >>> np.allclose(cs_new.origin, [5, 10, 0])
            True
        
        **Validates: Requirements 2.4**
        """
        new_origin = self.origin + dx * self.x_axis + dy * self.y_axis
        return CurrentCoordinateSystem(origin=new_origin, axes=self.axes.copy())
    
    def apply_rotation(
        self, 
        tilt_x: float, 
        tilt_y: float, 
        tilt_z: float
    ) -> "CurrentCoordinateSystem":
        """应用旋转（绕当前 X、Y、Z 轴依次旋转）
        
        旋转顺序为 X → Y → Z，即先绕 X 轴旋转，再绕 Y 轴，最后绕 Z 轴。
        组合旋转矩阵为 R_xyz = R_z × R_y × R_x。
        
        参数:
            tilt_x: 绕 X 轴旋转角度 (弧度)
            tilt_y: 绕 Y 轴旋转角度 (弧度)
            tilt_z: 绕 Z 轴旋转角度 (弧度)
        
        返回:
            新的坐标系状态
        
        示例:
            >>> cs = CurrentCoordinateSystem.identity()
            >>> # 绕 X 轴旋转 45 度
            >>> cs_new = cs.apply_rotation(np.pi/4, 0, 0)
            >>> # Z 轴应该旋转到 YZ 平面内
            >>> np.allclose(cs_new.z_axis, [0, -np.sin(np.pi/4), np.cos(np.pi/4)])
            True
        
        **Validates: Requirements 2.3, 4.4, 4.5**
        """
        R_xyz = CoordinateBreakProcessor.rotation_matrix_xyz(tilt_x, tilt_y, tilt_z)
        new_axes = self.axes @ R_xyz
        return CurrentCoordinateSystem(origin=self.origin.copy(), axes=new_axes)
    
    def __repr__(self) -> str:
        """返回坐标系状态的字符串表示"""
        return (
            f"CurrentCoordinateSystem(\n"
            f"  origin={self.origin},\n"
            f"  z_axis={self.z_axis}\n"
            f")"
        )


# =============================================================================
# CoordinateBreakProcessor 类
# =============================================================================

class CoordinateBreakProcessor:
    """坐标断点处理器
    
    根据 Zemax 坐标断点参数更新当前坐标系状态。
    支持 Order=0（先平移后旋转）和 Order=1（先旋转后平移）两种变换顺序。
    
    所有方法都是静态方法，不需要实例化。
    
    示例:
        >>> cs = CurrentCoordinateSystem.identity()
        >>> # Order=0: 先平移后旋转
        >>> cs_new = CoordinateBreakProcessor.process(
        ...     cs, decenter_x=5.0, decenter_y=0,
        ...     tilt_x_rad=np.pi/4, tilt_y_rad=0, tilt_z_rad=0,
        ...     order=0, thickness=100.0
        ... )
    
    **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**
    """
    
    @staticmethod
    def rotation_matrix_x(angle: float) -> np.ndarray:
        """绕 X 轴旋转矩阵
        
        参数:
            angle: 旋转角度 (弧度)
        
        返回:
            3×3 旋转矩阵
        
        旋转矩阵形式:
            | 1    0       0    |
            | 0  cos(θ) -sin(θ) |
            | 0  sin(θ)  cos(θ) |
        
        **Validates: Requirements 4.1**
        """
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c]
        ], dtype=np.float64)
    
    @staticmethod
    def rotation_matrix_y(angle: float) -> np.ndarray:
        """绕 Y 轴旋转矩阵
        
        参数:
            angle: 旋转角度 (弧度)
        
        返回:
            3×3 旋转矩阵
        
        旋转矩阵形式:
            |  cos(θ)  0  sin(θ) |
            |    0     1    0    |
            | -sin(θ)  0  cos(θ) |
        
        **Validates: Requirements 4.2**
        """
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c]
        ], dtype=np.float64)
    
    @staticmethod
    def rotation_matrix_z(angle: float) -> np.ndarray:
        """绕 Z 轴旋转矩阵
        
        参数:
            angle: 旋转角度 (弧度)
        
        返回:
            3×3 旋转矩阵
        
        旋转矩阵形式:
            | cos(θ) -sin(θ)  0 |
            | sin(θ)  cos(θ)  0 |
            |   0       0     1 |
        
        **Validates: Requirements 4.3**
        """
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    @staticmethod
    def rotation_matrix_xyz(
        tilt_x: float, 
        tilt_y: float, 
        tilt_z: float
    ) -> np.ndarray:
        """组合旋转矩阵（X → Y → Z 顺序）
        
        计算 R_xyz = R_z × R_y × R_x，即先绕 X 轴旋转，再绕 Y 轴，最后绕 Z 轴。
        
        参数:
            tilt_x: 绕 X 轴旋转角度 (弧度)
            tilt_y: 绕 Y 轴旋转角度 (弧度)
            tilt_z: 绕 Z 轴旋转角度 (弧度)
        
        返回:
            3×3 组合旋转矩阵
        
        **Validates: Requirements 2.3, 4.4**
        """
        R_x = CoordinateBreakProcessor.rotation_matrix_x(tilt_x)
        R_y = CoordinateBreakProcessor.rotation_matrix_y(tilt_y)
        R_z = CoordinateBreakProcessor.rotation_matrix_z(tilt_z)
        # R_xyz = R_z × R_y × R_x
        return R_z @ R_y @ R_x
    
    @staticmethod
    def process(
        current_cs: CurrentCoordinateSystem,
        decenter_x: float,
        decenter_y: float,
        tilt_x_rad: float,
        tilt_y_rad: float,
        tilt_z_rad: float,
        order: int,
        thickness: float
    ) -> CurrentCoordinateSystem:
        """处理坐标断点，返回更新后的坐标系状态
        
        参数:
            current_cs: 当前坐标系状态
            decenter_x: X 方向偏心 (mm)
            decenter_y: Y 方向偏心 (mm)
            tilt_x_rad: 绕 X 轴旋转角度 (弧度)
            tilt_y_rad: 绕 Y 轴旋转角度 (弧度)
            tilt_z_rad: 绕 Z 轴旋转角度 (弧度)
            order: 变换顺序，0=先平移后旋转，1=先旋转后平移
            thickness: 坐标断点厚度 (mm)
        
        返回:
            更新后的坐标系状态
        
        异常:
            ValueError: 如果 order 不是 0 或 1
        
        变换顺序说明:
            Order=0（先平移后旋转）:
                1. 平移: origin += axes @ [dx, dy, 0]
                2. 旋转: axes = axes @ R_xyz
                3. 厚度: origin += thickness × new_z_axis
            
            Order=1（先旋转后平移）:
                1. 旋转: axes = axes @ R_xyz
                2. 平移: origin += new_axes @ [dx, dy, 0]
                3. 厚度: origin += thickness × new_z_axis
        
        示例:
            >>> cs = CurrentCoordinateSystem.identity()
            >>> # Order=0: 先偏心 5mm，再旋转 45 度
            >>> cs_new = CoordinateBreakProcessor.process(
            ...     cs, decenter_x=5.0, decenter_y=0,
            ...     tilt_x_rad=np.pi/4, tilt_y_rad=0, tilt_z_rad=0,
            ...     order=0, thickness=100.0
            ... )
        
        **Validates: Requirements 2.1, 2.2, 3.4**
        """
        if order not in (0, 1):
            raise ValueError(f"Order 参数必须为 0 或 1，实际为 {order}")
        
        if order == 0:
            # Order=0: 先平移后旋转
            # 1. 平移（使用当前轴）
            cs_after_decenter = current_cs.apply_decenter(decenter_x, decenter_y)
            # 2. 旋转
            cs_after_rotation = cs_after_decenter.apply_rotation(
                tilt_x_rad, tilt_y_rad, tilt_z_rad
            )
            # 3. 厚度（沿新 Z 轴）
            cs_final = cs_after_rotation.advance_along_z(thickness)
        else:
            # Order=1: 先旋转后平移
            # 1. 旋转
            cs_after_rotation = current_cs.apply_rotation(
                tilt_x_rad, tilt_y_rad, tilt_z_rad
            )
            # 2. 平移（使用旋转后的轴）
            cs_after_decenter = cs_after_rotation.apply_decenter(decenter_x, decenter_y)
            # 3. 厚度（沿新 Z 轴）
            cs_final = cs_after_decenter.advance_along_z(thickness)
        
        return cs_final


# =============================================================================
# GlobalSurfaceDefinition 类
# =============================================================================

@dataclass
class GlobalSurfaceDefinition:
    """全局坐标系中的表面定义
    
    存储从 Zemax 转换后的表面参数，所有位置和方向都在全局坐标系中表示。
    
    属性:
        index: 原始 Zemax 表面索引
        surface_type: 表面类型 ('standard', 'even_asphere', 'flat', 'biconic')
        vertex_position: 表面顶点在全局坐标系中的位置 (mm)
        orientation: 表面姿态矩阵，列向量为表面局部坐标系的 X, Y, Z 轴
        radius: 曲率半径 (mm)，正值表示曲率中心在表面 +Z 方向
                对于双锥面，这是 Y 方向曲率半径
        conic: 圆锥常数，对于双锥面，这是 Y 方向圆锥常数
        is_mirror: 是否为反射镜
        semi_aperture: 半口径 (mm)
        material: 材料名称
        asphere_coeffs: 非球面系数列表（用于偶次非球面）
        comment: 注释/名称
        thickness: 到下一表面的厚度 (mm)
        radius_x: X 方向曲率半径 (mm)，仅用于双锥面
        conic_x: X 方向圆锥常数，仅用于双锥面
    
    示例:
        >>> surface = GlobalSurfaceDefinition(
        ...     index=1,
        ...     surface_type='standard',
        ...     vertex_position=np.array([0, 0, 100]),
        ...     orientation=np.eye(3),
        ...     radius=200.0,
        ...     is_mirror=True,
        ...     semi_aperture=25.0,
        ... )
        >>> print(f"曲率中心: {surface.curvature_center}")
    
    **Validates: Requirements 8.1, 8.2, 8.3**
    """
    index: int
    surface_type: str
    vertex_position: np.ndarray  # shape (3,)
    orientation: np.ndarray      # shape (3, 3)
    radius: float = np.inf
    conic: float = 0.0
    is_mirror: bool = False
    semi_aperture: float = 0.0
    material: str = "air"
    asphere_coeffs: List[float] = field(default_factory=list)
    comment: str = ""
    thickness: float = 0.0
    # 双锥面参数
    radius_x: float = np.inf     # X 方向曲率半径 (mm)
    conic_x: float = 0.0         # X 方向圆锥常数
    
    def __post_init__(self):
        """验证并转换输入数据"""
        self.vertex_position = np.asarray(self.vertex_position, dtype=np.float64)
        self.orientation = np.asarray(self.orientation, dtype=np.float64)
        
        if self.vertex_position.shape != (3,):
            raise ValueError(
                f"vertex_position 形状必须为 (3,)，实际为 {self.vertex_position.shape}"
            )
        if self.orientation.shape != (3, 3):
            raise ValueError(
                f"orientation 形状必须为 (3, 3)，实际为 {self.orientation.shape}"
            )

    @property
    def surface_normal(self) -> np.ndarray:
        """表面法向量（指向入射侧，即 -Z 方向）
        
        返回:
            shape (3,) 的单位向量，指向入射光来的方向
        
        **Validates: Requirements 8.3**
        """
        return -self.orientation[:, 2]
    
    @property
    def curvature_center(self) -> Optional[np.ndarray]:
        """曲率中心在全局坐标系中的位置
        
        对于有限曲率半径的表面，曲率中心位于:
            vertex_position + radius × orientation[:, 2]
        
        返回:
            曲率中心位置，如果是平面（无穷大半径）则返回 None
        
        示例:
            >>> surface = GlobalSurfaceDefinition(
            ...     index=1, surface_type='standard',
            ...     vertex_position=np.array([0, 0, 0]),
            ...     orientation=np.eye(3),
            ...     radius=100.0,
            ... )
            >>> np.allclose(surface.curvature_center, [0, 0, 100])
            True
        
        **Validates: Requirements 8.2, 6.5**
        """
        if np.isinf(self.radius):
            return None
        # 曲率中心 = 顶点 + R × Z轴方向
        return self.vertex_position + self.radius * self.orientation[:, 2]
    
    @property
    def local_z_axis(self) -> np.ndarray:
        """表面局部 Z 轴在全局坐标系中的方向"""
        return self.orientation[:, 2].copy()
    
    def to_optiland_params(self) -> Dict:
        """转换为 optiland 表面参数字典
        
        返回:
            包含 optiland 所需参数的字典
        """
        params = {
            'radius': self.radius,
            'conic': self.conic,
            'is_mirror': self.is_mirror,
            'semi_diameter': self.semi_aperture,
            'material': self.material,
        }
        if self.asphere_coeffs:
            params['asphere_coeffs'] = self.asphere_coeffs
        return params
    
    def __repr__(self) -> str:
        """返回表面定义的字符串表示"""
        parts = [f"GlobalSurfaceDefinition(index={self.index}"]
        parts.append(f", type='{self.surface_type}'")
        
        if self.comment:
            parts.append(f", comment='{self.comment}'")
        
        parts.append(f", vertex={self.vertex_position}")
        
        if not np.isinf(self.radius):
            parts.append(f", radius={self.radius:.4f}")
        
        if self.is_mirror:
            parts.append(", is_mirror=True")
        
        parts.append(")")
        return "".join(parts)


# =============================================================================
# SurfaceTraversalAlgorithm 类
# =============================================================================

class SurfaceTraversalAlgorithm:
    """表面遍历算法
    
    遍历 Zemax 表面序列，追踪当前坐标系状态，生成全局坐标表面定义。
    
    设计原则：
    - 统一处理所有表面类型，不对连续坐标断点或空气面做特殊处理
    - 虚拟表面（坐标断点）只更新坐标系，不生成 optiland 表面
    - 光学表面生成全局坐标定义后，再更新坐标系
    
    示例:
        >>> from sequential_system.zmx_parser import ZmxParser
        >>> parser = ZmxParser("system.zmx")
        >>> zmx_data = parser.parse()
        >>> 
        >>> traversal = SurfaceTraversalAlgorithm(zmx_data)
        >>> global_surfaces = traversal.traverse()
        >>> 
        >>> for surface in global_surfaces:
        ...     print(f"表面 {surface.index}: {surface.surface_type}")
    
    **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6**
    """
    
    def __init__(self, zmx_data: 'ZmxDataModel'):
        """初始化遍历算法
        
        参数:
            zmx_data: ZMX 解析后的数据模型
        """
        self._zmx_data = zmx_data
        self._current_cs = CurrentCoordinateSystem.identity()
        self._global_surfaces: List[GlobalSurfaceDefinition] = []
    
    def traverse(self) -> List[GlobalSurfaceDefinition]:
        """遍历所有表面，返回全局坐标表面定义列表
        
        返回:
            GlobalSurfaceDefinition 列表，只包含光学表面（不含坐标断点）
        
        **Validates: Requirements 5.1**
        """
        # 重置状态
        self._current_cs = CurrentCoordinateSystem.identity()
        self._global_surfaces = []
        
        # 按索引顺序处理所有表面
        sorted_indices = sorted(self._zmx_data.surfaces.keys())
        
        for index in sorted_indices:
            surface = self._zmx_data.surfaces[index]
            self._process_surface(surface)
        
        return self._global_surfaces
    
    def _process_surface(self, surface: 'ZmxSurfaceData') -> None:
        """处理单个表面
        
        根据表面类型：
        - is_ignored=True: 完全跳过，不更新坐标系，不生成表面定义
        - coordinate_break: 更新当前坐标系
        - standard/even_asphere: 生成全局坐标定义，然后更新坐标系
        
        参数:
            surface: ZmxSurfaceData 对象
        
        **Validates: Requirements 5.4**
        """
        # 检查是否为被忽略的表面（HIDE 第六位为 1）
        if surface.is_ignored:
            # 被忽略的表面：不进行光线追迹、不绘制、不考虑坐标变换
            return
        
        if self._is_virtual_surface(surface):
            # 坐标断点：只更新坐标系，不生成表面定义
            self._process_coordinate_break(surface)
        else:
            # 光学表面：生成全局坐标定义，然后更新坐标系
            self._process_optical_surface(surface)
    
    def _is_virtual_surface(self, surface: 'ZmxSurfaceData') -> bool:
        """判断是否为虚拟表面（不产生光学作用）
        
        参数:
            surface: ZmxSurfaceData 对象
        
        返回:
            如果是坐标断点则返回 True
        
        **Validates: Requirements 5.5**
        """
        return surface.surface_type == 'coordinate_break'

    def _process_coordinate_break(self, surface: 'ZmxSurfaceData') -> None:
        """处理坐标断点
        
        应用坐标断点的变换到当前坐标系。
        
        参数:
            surface: ZmxSurfaceData 对象，表面类型为 coordinate_break
        
        **Validates: Requirements 5.6**
        """
        # 将角度从度转换为弧度
        tilt_x_rad = np.deg2rad(surface.tilt_x_deg)
        tilt_y_rad = np.deg2rad(surface.tilt_y_deg)
        tilt_z_rad = np.deg2rad(surface.tilt_z_deg)
        
        # 获取 order 参数（默认为 0）
        # ZmxSurfaceData 中没有 order 字段，默认使用 Order=0
        order = getattr(surface, 'order', 0)
        
        # 特殊处理：无穷大厚度视为 0
        thickness = surface.thickness if np.isfinite(surface.thickness) else 0.0
        
        # 应用坐标断点变换
        self._current_cs = CoordinateBreakProcessor.process(
            current_cs=self._current_cs,
            decenter_x=surface.decenter_x,
            decenter_y=surface.decenter_y,
            tilt_x_rad=tilt_x_rad,
            tilt_y_rad=tilt_y_rad,
            tilt_z_rad=tilt_z_rad,
            order=order,
            thickness=thickness
        )
    
    def _process_optical_surface(self, surface: 'ZmxSurfaceData') -> None:
        """处理光学表面
        
        生成全局坐标定义，然后更新坐标系（沿 Z 轴前进厚度）。
        
        参数:
            surface: ZmxSurfaceData 对象
        
        **Validates: Requirements 5.4, 9.1, 9.2**
        """
        # 创建全局坐标表面定义
        global_surface = self._create_global_surface(surface)
        self._global_surfaces.append(global_surface)
        
        # 更新坐标系：沿当前 Z 轴前进厚度
        # 注意：反射镜不会自动改变坐标系方向
        # 特殊处理：无穷大厚度（物面）不更新坐标系位置
        if np.isfinite(surface.thickness):
            self._current_cs = self._current_cs.advance_along_z(surface.thickness)
    
    def _create_global_surface(
        self, 
        surface: 'ZmxSurfaceData'
    ) -> GlobalSurfaceDefinition:
        """从 Zemax 表面创建全局坐标定义
        
        参数:
            surface: ZmxSurfaceData 对象
        
        返回:
            GlobalSurfaceDefinition 对象
        
        **Validates: Requirements 8.1**
        """
        # 确定表面类型
        if surface.surface_type == 'biconic':
            surface_type = 'biconic'
        elif np.isinf(surface.radius):
            surface_type = 'flat'
        elif surface.surface_type == 'even_asphere':
            surface_type = 'even_asphere'
        else:
            surface_type = 'standard'
        
        return GlobalSurfaceDefinition(
            index=surface.index,
            surface_type=surface_type,
            vertex_position=self._current_cs.origin.copy(),
            orientation=self._current_cs.axes.copy(),
            radius=surface.radius,
            conic=surface.conic,
            is_mirror=surface.is_mirror,
            semi_aperture=surface.semi_diameter,
            material=surface.material,
            asphere_coeffs=surface.asphere_coeffs.copy() if surface.asphere_coeffs else [],
            comment=surface.comment,
            thickness=surface.thickness,
            # 双锥面参数
            radius_x=surface.radius_x,
            conic_x=surface.conic_x,
        )
    
    @property
    def current_coordinate_system(self) -> CurrentCoordinateSystem:
        """获取当前坐标系状态"""
        return self._current_cs


# =============================================================================
# ZemaxToOptilandConverter 类
# =============================================================================

class ZemaxToOptilandConverter:
    """Zemax 到 optiland 转换器
    
    将 GlobalSurfaceDefinition 列表转换为 optiland Optic 对象。
    
    示例:
        >>> from sequential_system.zmx_parser import ZmxParser
        >>> from sequential_system.coordinate_system import (
        ...     SurfaceTraversalAlgorithm,
        ...     ZemaxToOptilandConverter,
        ... )
        >>> 
        >>> parser = ZmxParser("system.zmx")
        >>> zmx_data = parser.parse()
        >>> 
        >>> traversal = SurfaceTraversalAlgorithm(zmx_data)
        >>> global_surfaces = traversal.traverse()
        >>> 
        >>> converter = ZemaxToOptilandConverter(global_surfaces)
        >>> optic = converter.convert()
    
    **Validates: Requirements 10.1, 10.5, 10.6**
    """
    
    def __init__(
        self, 
        global_surfaces: List[GlobalSurfaceDefinition],
        wavelength: float = 0.55,
        entrance_pupil_diameter: float = 10.0
    ):
        """初始化转换器
        
        参数:
            global_surfaces: 全局坐标表面定义列表
            wavelength: 波长 (μm)，默认 0.55
            entrance_pupil_diameter: 入瞳直径 (mm)，默认 10.0
        """
        self._surfaces = global_surfaces
        self._wavelength = wavelength
        self._entrance_pupil_diameter = entrance_pupil_diameter
    
    def convert(self) -> "Optic":
        """转换为 optiland Optic 对象
        
        返回:
            配置好的 optiland Optic 对象
        
        **Validates: Requirements 10.1, 10.6**
        """
        from optiland.optic import Optic
        
        # 创建 optiland Optic 对象
        optic = Optic()
        
        # 设置波长
        optic.add_wavelength(self._wavelength, is_primary=True)
        
        # 设置入瞳
        optic.set_aperture(aperture_type='EPD', value=self._entrance_pupil_diameter)
        
        # 设置物面（无穷远物体）
        optic.set_field_type(field_type='angle')
        optic.add_field(y=0)
        
        # 添加物面
        optic.add_surface(index=0, radius=np.inf, thickness=np.inf)
        
        # 添加所有光学表面
        for i, surface in enumerate(self._surfaces):
            self._add_surface_to_optic(optic, surface, i + 1)
        
        # 添加像面
        self._add_image_surface(optic)
        
        return optic
    
    @staticmethod
    def _extract_euler_angles(orientation: np.ndarray) -> tuple:
        """从姿态矩阵提取欧拉角（X → Y → Z 顺序）
        
        optiland 使用与 Zemax 相同的旋转顺序：先绕 X 轴，再绕 Y 轴，最后绕 Z 轴。
        组合旋转矩阵为 R = R_z @ R_y @ R_x。
        
        参数:
            orientation: 3×3 姿态矩阵，列向量为 [X, Y, Z] 轴方向
        
        返回:
            (rx, ry, rz) 欧拉角元组（弧度）
        
        注意:
            使用 scipy.spatial.transform.Rotation 进行转换，
            确保与 optiland 的旋转约定一致。
        """
        from scipy.spatial.transform import Rotation as R
        
        # scipy 的 from_matrix 期望旋转矩阵，我们的 orientation 就是旋转矩阵
        # 使用 'xyz' 顺序（内旋），对应 R_z @ R_y @ R_x
        rotation = R.from_matrix(orientation)
        euler_angles = rotation.as_euler('xyz')
        
        return euler_angles[0], euler_angles[1], euler_angles[2]

    def _add_surface_to_optic(
        self, 
        optic: "Optic", 
        surface: GlobalSurfaceDefinition,
        index: int
    ) -> None:
        """向 optiland Optic 添加单个表面
        
        参数:
            optic: optiland Optic 对象
            surface: GlobalSurfaceDefinition 对象
            index: 表面在 optiland 中的索引
        
        **Validates: Requirements 10.2, 10.3, 10.4**
        """
        # 确定材料
        # optiland 使用 'air' 字符串表示空气，不接受 None
        if surface.is_mirror:
            material = 'mirror'
        elif surface.material.lower() == 'air' or surface.material == '':
            material = 'air'
        else:
            material = surface.material
        
        # 从姿态矩阵提取欧拉角
        rx, ry, rz = self._extract_euler_angles(surface.orientation)
        
        # 提取顶点位置
        x, y, z = surface.vertex_position
        
        # 构建基本参数字典
        # 注意：optiland 不允许同时定义 thickness 和 z
        # 使用绝对定位模式（x, y, z）时，不传递 thickness
        params = {
            'index': index,
            'radius': surface.radius,
            'conic': surface.conic,
            'material': material,
            'is_stop': (index == 1),  # 简化处理：第一个表面为光阑
            # 坐标系统参数（使用绝对定位模式）
            'x': float(x),
            'y': float(y),
            'z': float(z),
            'rx': float(rx),
            'ry': float(ry),
            'rz': float(rz),
        }
        
        # 处理双锥面类型
        if surface.surface_type == 'biconic':
            # optiland 使用 surface_type='biconic' 和特定参数
            # radius/conic 是 Y 方向，radius_x/conic_x 是 X 方向
            params['surface_type'] = 'biconic'
            params['radius_y'] = surface.radius  # Y 方向曲率半径
            params['conic_y'] = surface.conic    # Y 方向圆锥常数
            params['radius_x'] = surface.radius_x  # X 方向曲率半径
            params['conic_x'] = surface.conic_x    # X 方向圆锥常数
            # 移除标准参数，因为 biconic 使用 radius_x/radius_y
            del params['radius']
            del params['conic']
        
        # 添加非球面系数（如果有）
        if surface.surface_type == 'even_asphere' and surface.asphere_coeffs:
            params['coefficients'] = surface.asphere_coeffs
        
        # 添加表面
        optic.add_surface(**params)
    
    def _add_image_surface(self, optic: "Optic") -> None:
        """添加像面
        
        像面位置基于最后一个光学表面的位置和厚度计算。
        
        参数:
            optic: optiland Optic 对象
        """
        if not self._surfaces:
            # 没有光学表面，像面在原点
            optic.add_surface(
                index=1,
                radius=np.inf,
                z=0.0
            )
            return
        
        # 获取最后一个表面
        last_surface = self._surfaces[-1]
        
        # 计算像面位置：最后一个表面顶点 + thickness × Z轴方向
        image_position = (
            last_surface.vertex_position + 
            last_surface.thickness * last_surface.orientation[:, 2]
        )
        
        # 像面姿态与最后一个表面相同
        rx, ry, rz = self._extract_euler_angles(last_surface.orientation)
        
        optic.add_surface(
            index=len(self._surfaces) + 1,
            radius=np.inf,
            x=float(image_position[0]),
            y=float(image_position[1]),
            z=float(image_position[2]),
            rx=float(rx),
            ry=float(ry),
            rz=float(rz),
        )
    
    def get_global_surfaces(self) -> List[GlobalSurfaceDefinition]:
        """获取全局坐标表面定义列表"""
        return self._surfaces


# =============================================================================
# 便捷函数
# =============================================================================

def convert_zmx_to_optiland(
    zmx_file_path: str,
    wavelength: Optional[float] = None,
    entrance_pupil_diameter: Optional[float] = None
) -> "Optic":
    """将 ZMX 文件转换为 optiland Optic 对象
    
    这是一个便捷函数，整合了 ZmxParser → SurfaceTraversalAlgorithm → 
    ZemaxToOptilandConverter 的完整流程。
    
    参数:
        zmx_file_path: ZMX 文件路径
        wavelength: 波长 (μm)，如果为 None 则使用 ZMX 文件中的主波长
        entrance_pupil_diameter: 入瞳直径 (mm)，如果为 None 则使用 ZMX 文件中的值
    
    返回:
        配置好的 optiland Optic 对象
    
    异常:
        FileNotFoundError: ZMX 文件不存在
        ZmxParseError: ZMX 文件解析错误
        ZmxUnsupportedError: 不支持的 ZMX 特性
    
    示例:
        >>> optic = convert_zmx_to_optiland("system.zmx")
        >>> print(f"共 {optic.surface_count} 个表面")
    
    **Validates: Requirements 13.1, 13.2, 13.3, 13.4, 13.5**
    """
    from sequential_system.zmx_parser import ZmxParser
    
    # 解析 ZMX 文件
    parser = ZmxParser(zmx_file_path)
    zmx_data = parser.parse()
    
    # 确定波长和入瞳直径
    if wavelength is None:
        if zmx_data.wavelengths:
            wavelength = zmx_data.wavelengths[zmx_data.primary_wavelength_index]
        else:
            wavelength = 0.55  # 默认值
    
    if entrance_pupil_diameter is None:
        entrance_pupil_diameter = zmx_data.entrance_pupil_diameter
        if entrance_pupil_diameter <= 0:
            entrance_pupil_diameter = 10.0  # 默认值
    
    # 遍历表面
    traversal = SurfaceTraversalAlgorithm(zmx_data)
    global_surfaces = traversal.traverse()
    
    # 转换为 optiland
    converter = ZemaxToOptilandConverter(
        global_surfaces,
        wavelength=wavelength,
        entrance_pupil_diameter=entrance_pupil_diameter
    )
    
    return converter.convert()


def convert_zmx_to_global_surfaces(
    zmx_file_path: str
) -> List[GlobalSurfaceDefinition]:
    """将 ZMX 文件转换为全局坐标表面定义列表
    
    这是一个便捷函数，用于获取中间结果（全局坐标表面定义）。
    
    参数:
        zmx_file_path: ZMX 文件路径
    
    返回:
        GlobalSurfaceDefinition 列表
    
    示例:
        >>> surfaces = convert_zmx_to_global_surfaces("system.zmx")
        >>> for s in surfaces:
        ...     print(f"表面 {s.index}: {s.vertex_position}")
    """
    from sequential_system.zmx_parser import ZmxParser
    
    # 解析 ZMX 文件
    parser = ZmxParser(zmx_file_path)
    zmx_data = parser.parse()
    
    # 遍历表面
    traversal = SurfaceTraversalAlgorithm(zmx_data)
    return traversal.traverse()
