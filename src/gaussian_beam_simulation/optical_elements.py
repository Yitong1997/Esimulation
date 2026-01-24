"""
光学元件定义模块

本模块定义光学元件的参数和属性。

设计理念：Zemax 序列模式
- 元件按顺序排列
- 使用 thickness（到下一元件的间距）定义位置关系
- 光束沿光路自动传播

支持的元件类型：
- 抛物面反射镜（ParabolicMirror）
- 球面反射镜（SphericalMirror）
- 薄透镜（ThinLens）

每个元件包含：
- 几何参数（焦距、曲率半径等）
- 间距参数（thickness：到下一元件的距离）
- 孔径参数
- 可选的倾斜和偏心

作者：混合光学仿真项目
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray


@dataclass
class OpticalElement(ABC):
    """光学元件基类
    
    定义光学元件的通用属性和接口。
    
    Zemax 序列模式设计：
    - thickness: 到下一元件的间距（沿光路方向）
    - z_position: 由系统自动计算（不需要手动指定）
    
    参数:
        thickness: 到下一元件的间距（mm），沿光路方向
        semi_aperture: 半口径（mm），必须为正值
        tilt_x: 绕 X 轴旋转角度（rad），默认 0
        tilt_y: 绕 Y 轴旋转角度（rad），默认 0
        decenter_x: X 方向偏心（mm），默认 0
        decenter_y: Y 方向偏心（mm），默认 0
        is_fold: 是否为折叠倾斜（True=折叠光路，不引入波前倾斜；False=失调，引入波前倾斜），默认 False
        name: 元件名称，可选
    
    异常:
        ValueError: 当参数不满足约束条件时抛出
    
    注意:
        - is_fold=False（默认）：使用完整光线追迹计算 OPD
          入射面和出射面垂直于各自的光轴，OPD 不包含整体倾斜
        - is_fold=True：跳过光线追迹中的倾斜处理，仅更新光轴方向
        - 对于大角度折叠镜，推荐使用默认的 is_fold=False
    """
    
    thickness: float  # mm，到下一元件的间距
    semi_aperture: float  # mm
    tilt_x: float = 0.0  # rad
    tilt_y: float = 0.0  # rad
    decenter_x: float = 0.0  # mm
    decenter_y: float = 0.0  # mm
    is_fold: bool = False  # 默认为失调倾斜（引入波前倾斜）
    name: Optional[str] = None
    
    # 由系统计算的位置（不需要手动指定）
    _z_position: float = field(default=0.0, init=False, repr=False)
    _path_length: float = field(default=0.0, init=False, repr=False)  # 从起点的光程距离
    
    def __post_init__(self) -> None:
        """初始化后验证参数"""
        self._validate_base_parameters()
    
    @property
    def z_position(self) -> float:
        """元件的 z 坐标位置（mm）
        
        由系统根据元件顺序和间距自动计算。
        """
        return self._z_position
    
    @z_position.setter
    def z_position(self, value: float) -> None:
        """设置 z 坐标位置（由系统内部调用）"""
        self._z_position = value
    
    @property
    def path_length(self) -> float:
        """从起点到此元件的光程距离（mm）"""
        return self._path_length
    
    @path_length.setter
    def path_length(self, value: float) -> None:
        """设置光程距离（由系统内部调用）"""
        self._path_length = value
    
    def _validate_base_parameters(self) -> None:
        """验证基类参数"""
        # 验证 thickness
        if not isinstance(self.thickness, (int, float)):
            raise ValueError(
                f"间距 (thickness) 必须为数值类型，"
                f"实际类型为 {type(self.thickness).__name__}"
            )
        if not np.isfinite(self.thickness):
            raise ValueError(
                f"间距 (thickness) 必须为有限实数，"
                f"实际值为 {self.thickness}"
            )
        if self.thickness < 0:
            raise ValueError(
                f"间距 (thickness) 必须为非负值，"
                f"实际值为 {self.thickness} mm"
            )
        
        # 验证 semi_aperture 必须为正值
        if not isinstance(self.semi_aperture, (int, float)):
            raise ValueError(
                f"半口径 (semi_aperture) 必须为数值类型，"
                f"实际类型为 {type(self.semi_aperture).__name__}"
            )
        if not np.isfinite(self.semi_aperture):
            raise ValueError(
                f"半口径 (semi_aperture) 必须为有限实数，"
                f"实际值为 {self.semi_aperture}"
            )
        if self.semi_aperture <= 0:
            raise ValueError(
                f"半口径 (semi_aperture) 必须为正值，"
                f"实际值为 {self.semi_aperture} mm"
            )
        
        # 验证 tilt_x 必须为有限实数
        if not isinstance(self.tilt_x, (int, float)):
            raise ValueError(
                f"X 轴旋转角度 (tilt_x) 必须为数值类型，"
                f"实际类型为 {type(self.tilt_x).__name__}"
            )
        if not np.isfinite(self.tilt_x):
            raise ValueError(
                f"X 轴旋转角度 (tilt_x) 必须为有限实数，"
                f"实际值为 {self.tilt_x}"
            )
        
        # 验证 tilt_y 必须为有限实数
        if not isinstance(self.tilt_y, (int, float)):
            raise ValueError(
                f"Y 轴旋转角度 (tilt_y) 必须为数值类型，"
                f"实际类型为 {type(self.tilt_y).__name__}"
            )
        if not np.isfinite(self.tilt_y):
            raise ValueError(
                f"Y 轴旋转角度 (tilt_y) 必须为有限实数，"
                f"实际值为 {self.tilt_y}"
            )
        
        # 验证 decenter_x 必须为有限实数
        if not isinstance(self.decenter_x, (int, float)):
            raise ValueError(
                f"X 方向偏心 (decenter_x) 必须为数值类型，"
                f"实际类型为 {type(self.decenter_x).__name__}"
            )
        if not np.isfinite(self.decenter_x):
            raise ValueError(
                f"X 方向偏心 (decenter_x) 必须为有限实数，"
                f"实际值为 {self.decenter_x}"
            )
        
        # 验证 decenter_y 必须为有限实数
        if not isinstance(self.decenter_y, (int, float)):
            raise ValueError(
                f"Y 方向偏心 (decenter_y) 必须为数值类型，"
                f"实际类型为 {type(self.decenter_y).__name__}"
            )
        if not np.isfinite(self.decenter_y):
            raise ValueError(
                f"Y 方向偏心 (decenter_y) 必须为有限实数，"
                f"实际值为 {self.decenter_y}"
            )
    
    @property
    @abstractmethod
    def focal_length(self) -> float:
        """焦距（mm）"""
        pass
    
    @property
    @abstractmethod
    def element_type(self) -> str:
        """元件类型"""
        pass
    
    @property
    def is_reflective(self) -> bool:
        """是否为反射元件"""
        return False
    
    @property
    def aperture_diameter(self) -> float:
        """孔径直径（mm）"""
        return 2 * self.semi_aperture
    
    def get_entrance_position(self) -> Tuple[float, float, float]:
        """获取入射面中心位置（全局坐标系）"""
        return (self.decenter_x, self.decenter_y, self._z_position)
    
    def get_chief_ray_direction(self) -> Tuple[float, float, float]:
        """获取主光线方向（考虑倾斜）"""
        L, M, N = 0.0, 0.0, 1.0
        
        if self.tilt_x != 0:
            cos_x = np.cos(self.tilt_x)
            sin_x = np.sin(self.tilt_x)
            M_new = M * cos_x - N * sin_x
            N_new = M * sin_x + N * cos_x
            M, N = M_new, N_new
        
        if self.tilt_y != 0:
            cos_y = np.cos(self.tilt_y)
            sin_y = np.sin(self.tilt_y)
            L_new = L * cos_y + N * sin_y
            N_new = -L * sin_y + N * cos_y
            L, N = L_new, N_new
        
        norm = np.sqrt(L**2 + M**2 + N**2)
        if norm < 1e-15:
            raise ValueError("方向余弦归一化失败")
        
        return (L / norm, M / norm, N / norm)
    
    @abstractmethod
    def get_surface_definition(self):
        """获取 optiland 表面定义"""
        pass
    
    def __repr__(self) -> str:
        name_str = f"'{self.name}'" if self.name else "unnamed"
        return (
            f"{self.__class__.__name__}({name_str}, "
            f"thickness={self.thickness}mm, f={self.focal_length}mm, "
            f"aperture={self.aperture_diameter}mm)"
        )


@dataclass
class ParabolicMirror(OpticalElement):
    """抛物面反射镜
    
    Zemax 序列模式：使用 thickness 指定到下一元件的间距。
    
    参数:
        thickness: 到下一元件的间距（mm），沿光路方向
        semi_aperture: 半口径（mm）
        parent_focal_length: 母抛物面焦距（mm）
            - 正值表示凹面镜（聚焦）
            - 负值表示凸面镜（发散）
        tilt_x, tilt_y: 旋转角度（rad）
        decenter_x, decenter_y: 偏心（mm）
        name: 元件名称
    
    示例:
        >>> # 创建焦距 100mm 的凹面抛物面镜，到下一元件间距 150mm
        >>> mirror = ParabolicMirror(
        ...     thickness=150.0,
        ...     semi_aperture=10.0,
        ...     parent_focal_length=100.0,
        ... )
    """
    
    parent_focal_length: float = 100.0  # mm
    
    def __post_init__(self) -> None:
        """初始化后验证参数"""
        self._validate_base_parameters()
        
        if not isinstance(self.parent_focal_length, (int, float)):
            raise ValueError(
                f"母抛物面焦距 (parent_focal_length) 必须为数值类型，"
                f"实际类型为 {type(self.parent_focal_length).__name__}"
            )
        if not np.isfinite(self.parent_focal_length):
            raise ValueError(
                f"母抛物面焦距 (parent_focal_length) 必须为有限实数，"
                f"实际值为 {self.parent_focal_length}"
            )
        if self.parent_focal_length == 0:
            raise ValueError("母抛物面焦距 (parent_focal_length) 不能为零")
    
    @property
    def focal_length(self) -> float:
        return self.parent_focal_length
    
    @property
    def element_type(self) -> str:
        return "parabolic_mirror"
    
    @property
    def is_reflective(self) -> bool:
        return True
    
    @property
    def vertex_radius(self) -> float:
        """顶点曲率半径（mm）：R = 2f"""
        return 2 * self.parent_focal_length
    
    @property
    def conic_constant(self) -> float:
        """圆锥常数：抛物面 k = -1"""
        return -1.0
    
    def get_surface_definition(self):
        """获取 optiland 表面定义"""
        from wavefront_to_rays.element_raytracer import SurfaceDefinition
        
        return SurfaceDefinition(
            surface_type='mirror',
            radius=self.vertex_radius,
            thickness=0.0,
            material='mirror',
            semi_aperture=self.semi_aperture,
            conic=self.conic_constant,
            tilt_x=self.tilt_x,
            tilt_y=self.tilt_y,
        )


@dataclass
class SphericalMirror(OpticalElement):
    """球面反射镜
    
    Zemax 序列模式：使用 thickness 指定到下一元件的间距。
    
    参数:
        thickness: 到下一元件的间距（mm），沿光路方向
        semi_aperture: 半口径（mm）
        radius_of_curvature: 曲率半径（mm）
            - 正值表示凹面镜
            - 负值表示凸面镜
            - np.inf 表示平面镜
        tilt_x, tilt_y: 旋转角度（rad）
        decenter_x, decenter_y: 偏心（mm）
        name: 元件名称
    
    示例:
        >>> # 创建曲率半径 200mm 的凹面镜（焦距 100mm）
        >>> mirror = SphericalMirror(
        ...     thickness=150.0,
        ...     semi_aperture=10.0,
        ...     radius_of_curvature=200.0,
        ... )
    """
    
    radius_of_curvature: float = 200.0  # mm
    
    def __post_init__(self) -> None:
        """初始化后验证参数"""
        self._validate_base_parameters()
        
        if not isinstance(self.radius_of_curvature, (int, float)):
            raise ValueError(
                f"曲率半径 (radius_of_curvature) 必须为数值类型，"
                f"实际类型为 {type(self.radius_of_curvature).__name__}"
            )
        if np.isnan(self.radius_of_curvature):
            raise ValueError("曲率半径 (radius_of_curvature) 不能为 NaN")
        if self.radius_of_curvature == 0:
            raise ValueError("曲率半径 (radius_of_curvature) 不能为零")
    
    @property
    def focal_length(self) -> float:
        if np.isinf(self.radius_of_curvature):
            return np.inf
        return self.radius_of_curvature / 2
    
    @property
    def element_type(self) -> str:
        return "spherical_mirror"
    
    @property
    def is_reflective(self) -> bool:
        return True
    
    def get_surface_definition(self):
        """获取 optiland 表面定义"""
        from wavefront_to_rays.element_raytracer import SurfaceDefinition
        
        return SurfaceDefinition(
            surface_type='mirror',
            radius=self.radius_of_curvature,
            thickness=0.0,
            material='mirror',
            semi_aperture=self.semi_aperture,
            conic=0.0,
            tilt_x=self.tilt_x,
            tilt_y=self.tilt_y,
        )


@dataclass
class ThinLens(OpticalElement):
    """薄透镜
    
    Zemax 序列模式：使用 thickness 指定到下一元件的间距。
    
    参数:
        thickness: 到下一元件的间距（mm），沿光路方向
        semi_aperture: 半口径（mm）
        focal_length_value: 焦距（mm）
            - 正值表示会聚透镜
            - 负值表示发散透镜
        tilt_x, tilt_y: 旋转角度（rad）
        decenter_x, decenter_y: 偏心（mm）
        name: 元件名称
    
    示例:
        >>> # 创建焦距 50mm 的会聚透镜
        >>> lens = ThinLens(
        ...     thickness=100.0,
        ...     semi_aperture=12.5,
        ...     focal_length_value=50.0,
        ... )
    """
    
    focal_length_value: float = 50.0  # mm
    
    def __post_init__(self) -> None:
        """初始化后验证参数"""
        self._validate_base_parameters()
        
        if not isinstance(self.focal_length_value, (int, float)):
            raise ValueError(
                f"焦距 (focal_length_value) 必须为数值类型，"
                f"实际类型为 {type(self.focal_length_value).__name__}"
            )
        if not np.isfinite(self.focal_length_value):
            raise ValueError(
                f"焦距 (focal_length_value) 必须为有限实数，"
                f"实际值为 {self.focal_length_value}"
            )
        if self.focal_length_value == 0:
            raise ValueError("焦距 (focal_length_value) 不能为零")
    
    @property
    def focal_length(self) -> float:
        return self.focal_length_value
    
    @property
    def element_type(self) -> str:
        return "thin_lens"
    
    @property
    def is_reflective(self) -> bool:
        return False
    
    @property
    def optical_power(self) -> float:
        """光焦度（1/mm）"""
        return 1.0 / self.focal_length_value
    
    def get_surface_definition(self):
        """获取 optiland 表面定义
        
        薄透镜使用 PROPER 的 prop_lens 处理，返回 None
        """
        return None


@dataclass
class FlatMirror(OpticalElement):
    """平面反射镜
    
    Zemax 序列模式：使用 thickness 指定到下一元件的间距。
    
    平面镜的曲率半径为无穷大，焦距也为无穷大。
    常用于折叠光路或改变光束方向。
    
    参数:
        thickness: 到下一元件的间距（mm），沿光路方向
        semi_aperture: 半口径（mm）
        tilt_x, tilt_y: 旋转角度（rad）
        decenter_x, decenter_y: 偏心（mm）
        name: 元件名称
    
    示例:
        >>> # 创建 45 度折叠镜
        >>> mirror = FlatMirror(
        ...     thickness=100.0,
        ...     semi_aperture=15.0,
        ...     tilt_x=np.pi/4,  # 45 度倾斜
        ... )
    
    验证需求:
        - Requirements 2.3: 支持平面反射镜（radius = infinity）
        - Requirements 2.1.2: 支持平面镜定义
    """
    
    def __post_init__(self) -> None:
        """初始化后验证参数"""
        self._validate_base_parameters()
    
    @property
    def focal_length(self) -> float:
        """焦距（mm）：平面镜焦距为无穷大"""
        return np.inf
    
    @property
    def element_type(self) -> str:
        return "flat_mirror"
    
    @property
    def is_reflective(self) -> bool:
        return True
    
    @property
    def radius_of_curvature(self) -> float:
        """曲率半径（mm）：平面镜曲率半径为无穷大"""
        return np.inf
    
    def get_surface_definition(self):
        """获取 optiland 表面定义"""
        from wavefront_to_rays.element_raytracer import SurfaceDefinition
        
        return SurfaceDefinition(
            surface_type='mirror',
            radius=np.inf,
            thickness=0.0,
            material='mirror',
            semi_aperture=self.semi_aperture,
            conic=0.0,
            tilt_x=self.tilt_x,
            tilt_y=self.tilt_y,
        )
