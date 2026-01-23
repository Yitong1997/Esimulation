"""
OpticalSystem 类模块

提供光学系统定义功能，封装表面定义列表。
复用 HybridSimulator 中的表面创建逻辑。
"""

from typing import Optional, Tuple, List, Any, TYPE_CHECKING
from dataclasses import dataclass, field
import numpy as np

if TYPE_CHECKING:
    from sequential_system.coordinate_system import GlobalSurfaceDefinition


@dataclass
class SurfaceDefinition:
    """表面定义（内部使用）
    
    属性:
        index: 表面索引
        surface_type: 表面类型 ('standard', 'paraxial', 'coordbrk')
        z: Z 位置 (mm)
        radius: 曲率半径 (mm)，inf 表示平面
        conic: 圆锥常数
        semi_aperture: 半口径 (mm)
        is_mirror: 是否为反射镜
        tilt_x: 绕 X 轴旋转角度（度）
        tilt_y: 绕 Y 轴旋转角度（度）
        material: 材料名称
        focal_length: 焦距（仅用于 paraxial）
    """
    index: int
    surface_type: str
    z: float
    radius: float = float('inf')
    conic: float = 0.0
    semi_aperture: float = 25.0
    is_mirror: bool = False
    tilt_x: float = 0.0
    tilt_y: float = 0.0
    material: str = ""
    focal_length: Optional[float] = None


class OpticalSystem:
    """光学系统定义
    
    支持两种构建方式：
    1. 从 ZMX 文件加载：bts.load_zmx("system.zmx")
    2. 逐行定义元件：system.add_surface(...)
    
    属性:
        name: 系统名称
        _surfaces: 表面定义列表（内部使用，SurfaceDefinition 类型）
        _global_surfaces: 全局坐标表面定义列表（内部使用，GlobalSurfaceDefinition 类型）
        _source_path: ZMX 文件路径（如果从文件加载）
    
    示例:
        >>> import bts
        >>> 
        >>> # 方式 1：逐行定义
        >>> system = bts.OpticalSystem("My System")
        >>> system.add_flat_mirror(z=50, tilt_x=45)
        >>> system.add_spherical_mirror(z=150, radius=200)
        >>> 
        >>> # 方式 2：链式调用
        >>> system = (bts.OpticalSystem("My System")
        ...     .add_flat_mirror(z=50, tilt_x=45)
        ...     .add_spherical_mirror(z=150, radius=200))
    """
    
    def __init__(self, name: str = "Unnamed System") -> None:
        """创建空的光学系统
        
        参数:
            name: 系统名称
        """
        self.name = name
        self._surfaces: List[SurfaceDefinition] = []
        self._global_surfaces: List["GlobalSurfaceDefinition"] = []
        self._source_path: Optional[str] = None
        # 内部使用：存储从 ZMX 加载的原始数据
        self._zmx_surfaces: Optional[List[Any]] = None
    
    @property
    def num_surfaces(self) -> int:
        """表面数量"""
        return len(self._surfaces)
    
    def __len__(self) -> int:
        """返回表面数量"""
        return len(self._surfaces)
    
    def _create_rotation_matrix(
        self,
        tilt_x_deg: float,
        tilt_y_deg: float,
    ) -> np.ndarray:
        """创建旋转矩阵
        
        参数:
            tilt_x_deg: 绕 X 轴旋转角度（度）
            tilt_y_deg: 绕 Y 轴旋转角度（度）
        
        返回:
            3x3 旋转矩阵
        """
        tilt_x_rad = np.radians(tilt_x_deg)
        tilt_y_rad = np.radians(tilt_y_deg)
        
        # 绕 X 轴旋转
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(tilt_x_rad), -np.sin(tilt_x_rad)],
            [0, np.sin(tilt_x_rad), np.cos(tilt_x_rad)],
        ])
        # 绕 Y 轴旋转
        Ry = np.array([
            [np.cos(tilt_y_rad), 0, np.sin(tilt_y_rad)],
            [0, 1, 0],
            [-np.sin(tilt_y_rad), 0, np.cos(tilt_y_rad)],
        ])
        
        # 组合旋转矩阵：先绕 X 轴，再绕 Y 轴
        return Ry @ Rx
    
    def _create_global_surface(
        self,
        z: float,
        radius: float,
        conic: float,
        semi_aperture: float,
        is_mirror: bool,
        tilt_x: float,
        tilt_y: float,
        material: str,
        surface_type: str = 'standard',
        focal_length: Optional[float] = None,
    ) -> "GlobalSurfaceDefinition":
        """创建全局坐标表面定义
        
        复用 HybridSimulator 中的逻辑。
        
        参数:
            z: Z 位置 (mm)
            radius: 曲率半径 (mm)
            conic: 圆锥常数
            semi_aperture: 半口径 (mm)
            is_mirror: 是否为反射镜
            tilt_x: 绕 X 轴旋转角度（度）
            tilt_y: 绕 Y 轴旋转角度（度）
            material: 材料名称
            surface_type: 表面类型
            focal_length: 焦距（仅用于 paraxial）
        
        返回:
            GlobalSurfaceDefinition 对象
        """
        from sequential_system.coordinate_system import GlobalSurfaceDefinition
        
        # 计算姿态矩阵
        orientation = self._create_rotation_matrix(tilt_x, tilt_y)
        
        # 创建全局表面定义
        return GlobalSurfaceDefinition(
            index=len(self._global_surfaces),
            surface_type=surface_type,
            vertex_position=np.array([0.0, 0.0, z]),
            orientation=orientation,
            radius=radius,
            conic=conic,
            semi_aperture=semi_aperture,
            is_mirror=is_mirror,
            material=material,
            focal_length=focal_length if focal_length is not None else np.inf,
        )
    
    def add_surface(
        self,
        z: float,
        radius: float = float('inf'),
        conic: float = 0.0,
        semi_aperture: float = 25.0,
        is_mirror: bool = False,
        tilt_x: float = 0.0,
        tilt_y: float = 0.0,
        material: str = "",
    ) -> "OpticalSystem":
        """添加通用光学表面（支持链式调用）
        
        参数:
            z: Z 位置 (mm)
            radius: 曲率半径 (mm)，默认 inf（平面）
            conic: 圆锥常数，默认 0（球面）
            semi_aperture: 半口径 (mm)，默认 25.0
            is_mirror: 是否为反射镜，默认 False
            tilt_x: 绕 X 轴旋转角度（度），默认 0
            tilt_y: 绕 Y 轴旋转角度（度），默认 0
            material: 材料名称，默认空字符串（空气）
        
        返回:
            self（支持链式调用）
        
        示例:
            >>> system = bts.OpticalSystem()
            >>> system.add_surface(z=100, radius=200, is_mirror=True, tilt_x=45)
        
        **Validates: Requirements 2.4**
        """
        # 创建 SurfaceDefinition
        surface_def = SurfaceDefinition(
            index=len(self._surfaces),
            surface_type='standard',
            z=z,
            radius=radius,
            conic=conic,
            semi_aperture=semi_aperture,
            is_mirror=is_mirror,
            tilt_x=tilt_x,
            tilt_y=tilt_y,
            material=material if material else ('MIRROR' if is_mirror else ''),
        )
        self._surfaces.append(surface_def)
        
        # 创建 GlobalSurfaceDefinition
        global_surface = self._create_global_surface(
            z=z,
            radius=radius,
            conic=conic,
            semi_aperture=semi_aperture,
            is_mirror=is_mirror,
            tilt_x=tilt_x,
            tilt_y=tilt_y,
            material=material if material else ('MIRROR' if is_mirror else ''),
        )
        self._global_surfaces.append(global_surface)
        
        return self
    
    def add_flat_mirror(
        self,
        z: float,
        tilt_x: float = 0.0,
        tilt_y: float = 0.0,
        semi_aperture: float = 25.0,
    ) -> "OpticalSystem":
        """添加平面反射镜（支持链式调用）
        
        参数:
            z: Z 位置 (mm)
            tilt_x: 绕 X 轴旋转角度（度），默认 0
            tilt_y: 绕 Y 轴旋转角度（度），默认 0
            semi_aperture: 半口径 (mm)，默认 25.0
        
        返回:
            self（支持链式调用）
        
        示例:
            >>> system = bts.OpticalSystem()
            >>> system.add_flat_mirror(z=50, tilt_x=45)  # 45° 折叠镜
        
        **Validates: Requirements 2.5**
        """
        return self.add_surface(
            z=z,
            radius=float('inf'),
            conic=0.0,
            semi_aperture=semi_aperture,
            is_mirror=True,
            tilt_x=tilt_x,
            tilt_y=tilt_y,
            material='MIRROR',
        )
    
    def add_spherical_mirror(
        self,
        z: float,
        radius: float,
        tilt_x: float = 0.0,
        tilt_y: float = 0.0,
        semi_aperture: float = 25.0,
    ) -> "OpticalSystem":
        """添加球面反射镜（支持链式调用）
        
        参数:
            z: Z 位置 (mm)
            radius: 曲率半径 (mm)，正值为凹面镜
            tilt_x: 绕 X 轴旋转角度（度），默认 0
            tilt_y: 绕 Y 轴旋转角度（度），默认 0
            semi_aperture: 半口径 (mm)，默认 25.0
        
        返回:
            self（支持链式调用）
        
        示例:
            >>> system = bts.OpticalSystem()
            >>> system.add_spherical_mirror(z=100, radius=200)  # 凹面镜，f=100mm
        
        **Validates: Requirements 2.6**
        """
        return self.add_surface(
            z=z,
            radius=radius,
            conic=0.0,
            semi_aperture=semi_aperture,
            is_mirror=True,
            tilt_x=tilt_x,
            tilt_y=tilt_y,
            material='MIRROR',
        )
    
    def add_paraxial_lens(
        self,
        z: float,
        focal_length: float,
        semi_aperture: float = 25.0,
    ) -> "OpticalSystem":
        """添加薄透镜（支持链式调用）
        
        参数:
            z: Z 位置 (mm)
            focal_length: 焦距 (mm)
            semi_aperture: 半口径 (mm)，默认 25.0
        
        返回:
            self（支持链式调用）
        
        示例:
            >>> system = bts.OpticalSystem()
            >>> system.add_paraxial_lens(z=50, focal_length=100)  # f=100mm 薄透镜
        
        **Validates: Requirements 2.7**
        """
        # 创建 SurfaceDefinition
        surface_def = SurfaceDefinition(
            index=len(self._surfaces),
            surface_type='paraxial',
            z=z,
            radius=float('inf'),
            conic=0.0,
            semi_aperture=semi_aperture,
            is_mirror=False,
            tilt_x=0.0,
            tilt_y=0.0,
            material='',
            focal_length=focal_length,
        )
        self._surfaces.append(surface_def)
        
        # 创建 GlobalSurfaceDefinition
        global_surface = self._create_global_surface(
            z=z,
            radius=float('inf'),
            conic=0.0,
            semi_aperture=semi_aperture,
            is_mirror=False,
            tilt_x=0.0,
            tilt_y=0.0,
            material='',
            surface_type='paraxial',
            focal_length=focal_length,
        )
        self._global_surfaces.append(global_surface)
        
        return self
    
    def get_global_surfaces(self) -> List["GlobalSurfaceDefinition"]:
        """获取全局坐标表面定义列表
        
        返回:
            GlobalSurfaceDefinition 列表，用于 HybridSimulator
        """
        return self._global_surfaces
    
    def print_info(self) -> None:
        """打印系统参数摘要
        
        显示系统名称、表面数量，以及每个表面的详细参数。
        
        示例:
            >>> system = bts.OpticalSystem("My System")
            >>> system.add_flat_mirror(z=50, tilt_x=45)
            >>> system.print_info()
            ============================================================
            光学系统: My System
            表面数量: 1
            ============================================================
            
            表面 0: standard
              位置: z = 50.000 mm
              曲率半径: 无穷大 (平面)
              反射镜: 是
              倾斜: tilt_x = 45.00°, tilt_y = 0.00°
        
        **Validates: Requirements 4.1, 4.3**
        """
        print(f"\n{'='*60}")
        print(f"光学系统: {self.name}")
        print(f"表面数量: {len(self._surfaces)}")
        if self._source_path:
            print(f"源文件: {self._source_path}")
        print(f"{'='*60}")
        
        if len(self._surfaces) == 0:
            print("\n(系统为空，尚未添加任何表面)")
            return
        
        for surface in self._surfaces:
            print(f"\n表面 {surface.index}: {surface.surface_type}")
            print(f"  位置: z = {surface.z:.3f} mm")
            
            # 曲率半径
            if np.isinf(surface.radius):
                print(f"  曲率半径: 无穷大 (平面)")
            else:
                print(f"  曲率半径: {surface.radius:.3f} mm")
            
            # 圆锥常数（仅非零时显示）
            if surface.conic != 0:
                conic_type = self._get_conic_type(surface.conic)
                print(f"  圆锥常数: {surface.conic:.6f} ({conic_type})")
            
            # 半口径
            print(f"  半口径: {surface.semi_aperture:.3f} mm")
            
            # 反射镜标识
            if surface.is_mirror:
                print(f"  反射镜: 是")
            
            # 材料（非空气时显示）
            if surface.material and surface.material.upper() not in ('', 'AIR', 'MIRROR'):
                print(f"  材料: {surface.material}")
            
            # 倾斜角度（非零时显示）
            if surface.tilt_x != 0 or surface.tilt_y != 0:
                print(f"  倾斜: tilt_x = {surface.tilt_x:.2f}°, tilt_y = {surface.tilt_y:.2f}°")
            
            # 焦距（仅 paraxial 类型）
            if surface.surface_type == 'paraxial' and surface.focal_length is not None:
                print(f"  焦距: {surface.focal_length:.3f} mm")
    
    def _get_conic_type(self, conic: float) -> str:
        """根据圆锥常数返回表面类型描述
        
        参数:
            conic: 圆锥常数
        
        返回:
            表面类型描述字符串
        """
        if conic == 0:
            return "球面"
        elif conic == -1:
            return "抛物面"
        elif conic < -1:
            return "双曲面"
        elif -1 < conic < 0:
            return "扁椭球面"
        else:  # conic > 0
            return "长椭球面"
    
    def plot_layout(
        self,
        projection: str = "YZ",
        num_rays: int = 5,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> Tuple[Any, Any]:
        """绘制光路图
        
        复用现有的 ZmxOpticLoader 和 view_2d 功能进行可视化。
        
        参数:
            projection: 投影平面 ('YZ', 'XZ', 'XY')，默认 'YZ'
            num_rays: 光线数量，默认 5
            save_path: 保存路径（可选），如果指定则保存图像
            show: 是否显示图形，默认 True
        
        返回:
            (fig, ax) 元组，matplotlib Figure 和 Axes 对象
        
        示例:
            >>> system = bts.load_zmx("system.zmx")
            >>> fig, ax = system.plot_layout(projection='YZ', num_rays=5)
            >>> 
            >>> # 保存到文件
            >>> system.plot_layout(save_path="layout.png", show=False)
        
        **Validates: Requirements 4.2, 4.4**
        """
        import matplotlib.pyplot as plt
        
        # 检查系统是否为空
        if len(self._surfaces) == 0:
            print("警告: 系统为空，无法绘制光路图")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "系统为空", ha='center', va='center', fontsize=14)
            ax.set_title(f"光学系统: {self.name}")
            return fig, ax
        
        # 尝试创建 optiland Optic 对象进行可视化
        try:
            optic = self._create_optiland_optic()
            
            # 使用 view_2d 进行可视化
            from sequential_system.zmx_visualization import view_2d
            
            fig, ax, _ = view_2d(
                optic,
                projection=projection,
                num_rays=num_rays,
            )
            
            # 设置标题
            ax.set_title(f"光学系统: {self.name} ({projection} 投影)")
            
        except Exception as e:
            # 如果创建 optiland Optic 失败，使用简化的可视化
            print(f"警告: 无法使用 optiland 可视化 ({e})，使用简化视图")
            fig, ax = self._plot_simple_layout(projection)
        
        # 保存图像
        if save_path:
            from pathlib import Path
            save_dir = Path(save_path).parent
            if save_dir and not save_dir.exists():
                save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"光路图已保存到: {save_path}")
        
        # 显示图形
        if show:
            plt.show()
        
        return fig, ax
    
    def _create_optiland_optic(self) -> Any:
        """创建 optiland Optic 对象用于可视化
        
        返回:
            optiland Optic 对象
        """
        from sequential_system.coordinate_system import (
            GlobalSurfaceDefinition,
            ZemaxToOptilandConverter,
        )
        
        # 如果已有全局表面定义，直接使用
        if self._global_surfaces:
            converter = ZemaxToOptilandConverter(
                self._global_surfaces,
                wavelength=0.633,  # 默认 He-Ne 波长
                entrance_pupil_diameter=10.0,
            )
            return converter.convert()
        
        # 否则，从 SurfaceDefinition 创建 GlobalSurfaceDefinition
        global_surfaces = []
        for surface in self._surfaces:
            # 计算姿态矩阵
            orientation = self._create_rotation_matrix(surface.tilt_x, surface.tilt_y)
            
            global_surface = GlobalSurfaceDefinition(
                index=surface.index,
                surface_type=surface.surface_type,
                vertex_position=np.array([0.0, 0.0, surface.z]),
                orientation=orientation,
                radius=surface.radius,
                conic=surface.conic,
                semi_aperture=surface.semi_aperture,
                is_mirror=surface.is_mirror,
                material=surface.material if surface.material else ('MIRROR' if surface.is_mirror else ''),
                focal_length=surface.focal_length if surface.focal_length is not None else np.inf,
            )
            global_surfaces.append(global_surface)
        
        # 使用转换器创建 optiland Optic
        converter = ZemaxToOptilandConverter(
            global_surfaces,
            wavelength=0.633,  # 默认 He-Ne 波长
            entrance_pupil_diameter=10.0,
        )
        return converter.convert()
    
    def _plot_simple_layout(self, projection: str = "YZ") -> Tuple[Any, Any]:
        """简化的光路图绘制（当 optiland 不可用时使用）
        
        参数:
            projection: 投影平面
        
        返回:
            (fig, ax) 元组
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 绘制每个表面
        for surface in self._surfaces:
            z = surface.z
            semi_ap = surface.semi_aperture
            
            # 根据投影选择坐标
            if projection.upper() == 'YZ':
                x_coord = z
                y_min, y_max = -semi_ap, semi_ap
            elif projection.upper() == 'XZ':
                x_coord = z
                y_min, y_max = -semi_ap, semi_ap
            else:  # XY
                x_coord = 0
                y_min, y_max = -semi_ap, semi_ap
            
            # 绘制表面（简化为垂直线）
            color = 'blue' if surface.is_mirror else 'gray'
            linestyle = '-' if surface.is_mirror else '--'
            ax.plot([x_coord, x_coord], [y_min, y_max], 
                   color=color, linestyle=linestyle, linewidth=2,
                   label=f"表面 {surface.index}" if surface.index == 0 else "")
            
            # 标注表面索引
            ax.annotate(f"{surface.index}", (x_coord, y_max + 2), 
                       ha='center', fontsize=8)
        
        # 设置坐标轴
        ax.set_xlabel(f"Z (mm)" if projection.upper() in ('YZ', 'XZ') else "X (mm)")
        ax.set_ylabel(f"{'Y' if projection.upper() in ('YZ', 'XY') else 'X'} (mm)")
        ax.set_title(f"光学系统: {self.name} ({projection} 投影) - 简化视图")
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        return fig, ax
