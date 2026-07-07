"""
OpticalSystem 类模块

提供光学系统定义功能，封装表面定义列表。
复用 HybridSimulator 中的表面创建逻辑。
"""

from typing import Optional, Tuple, List, Any, Union, TYPE_CHECKING
from dataclasses import dataclass, field
import numpy as np

if TYPE_CHECKING:
    from sequential_system.coordinate_system import GlobalSurfaceDefinition, CoordinateBreakProcessor


@dataclass
class SurfaceDefinition:
    """表面定义（内部使用）
    
    属性:
        index: 表面索引
        surface_type: 表面类型 ('standard', 'paraxial', 'coordbrk')
        position: 顶点位置 (x, y, z) (mm)，使用绝对坐标
        radius: 曲率半径 (mm)，inf 表示平面
        conic: 圆锥常数
        is_mirror: 是否为反射镜
        tilt_x: 绕 X 轴旋转角度（度）
        tilt_y: 绕 Y 轴旋转角度（度）
        material: 材料名称
        focal_length: 焦距（仅用于 paraxial）
    
    注意：
        离轴量由绝对坐标的 (x, y) 值决定，通过位置坐标自然实现。
        例如，离轴抛物面镜的离轴量 = sqrt(x² + y²)。
        
        🚫 禁止设置口径/半口径参数！光束范围由 w0 自然决定。
    """
    index: int
    surface_type: str
    position: Tuple[float, float, float]  # (x, y, z) 绝对坐标
    radius: float = float('inf')
    conic: float = 0.0
    is_mirror: bool = False
    tilt_x: float = 0.0
    tilt_y: float = 0.0
    tilt_z: float = 0.0
    material: str = ""
    focal_length: Optional[float] = None
    
    @property
    def z(self) -> float:
        """Z 位置"""
        return self.position[2]
    
    @property
    def x(self) -> float:
        """X 位置"""
        return self.position[0]
    
    @property
    def y(self) -> float:
        """Y 位置"""
        return self.position[1]


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
    

    
    def _create_global_surface(
        self,
        position: Tuple[float, float, float],
        radius: float,
        conic: float,
        is_mirror: bool,
        tilt_x: float,
        tilt_y: float,
        tilt_z: float,
        material: str,
        surface_type: str = 'standard',
        focal_length: Optional[float] = None,
    ) -> "GlobalSurfaceDefinition":
        """创建全局坐标表面定义
        
        复用 HybridSimulator 中的逻辑。
        
        参数:
            position: 顶点位置 (x, y, z) (mm)，使用绝对坐标
            radius: 曲率半径 (mm)
            conic: 圆锥常数
            is_mirror: 是否为反射镜
            tilt_x: 绕 X 轴旋转角度（度）
            tilt_y: 绕 Y 轴旋转角度（度）
            tilt_z: 绕 Z 轴旋转角度（度）
            material: 材料名称
            surface_type: 表面类型
            focal_length: 焦距（仅用于 paraxial）
        
        返回:
            GlobalSurfaceDefinition 对象
        
        注意:
            🚫 禁止设置口径/半口径参数！使用默认值 1000mm（足够大）。
        """
        from sequential_system.coordinate_system import GlobalSurfaceDefinition, CoordinateBreakProcessor
        
        # 计算姿态矩阵
        # 将角度转换为弧度
        tilt_x_rad = np.radians(tilt_x)
        tilt_y_rad = np.radians(tilt_y)
        tilt_z_rad = np.radians(tilt_z)
        
        # 使用 CoordinateBreakProcessor 计算旋转矩阵 (Rz @ Ry @ Rx)
        orientation = CoordinateBreakProcessor.rotation_matrix_xyz(
            tilt_x_rad, tilt_y_rad, tilt_z_rad
        )
        
        # 创建全局表面定义
        # 🚫 semi_aperture 使用默认大值，不允许用户设置
        return GlobalSurfaceDefinition(
            index=len(self._global_surfaces),
            surface_type=surface_type,
            vertex_position=np.array([position[0], position[1], position[2]]),
            orientation=orientation,
            radius=radius,
            conic=conic,
            semi_aperture=1000.0,  # 固定大值，不允许用户设置
            is_mirror=is_mirror,
            material=material,
            focal_length=focal_length if focal_length is not None else np.inf,
        )
    
    def add_surface(
        self,
        position: Optional[Tuple[float, float, float]] = None,
        z: Optional[float] = None,
        x: float = 0.0,
        y: float = 0.0,
        radius: float = float('inf'),
        conic: float = 0.0,
        is_mirror: bool = False,
        tilt_x: float = 0.0,
        tilt_y: float = 0.0,
        tilt_z: float = 0.0,
        material: str = "",
    ) -> "OpticalSystem":
        """添加通用光学表面（支持链式调用）
        
        使用绝对坐标定义表面位置，与 ZMX 文件加载后的处理方式一致。
        
        参数:
            position: 顶点位置 (x, y, z) (mm)，使用绝对坐标。
                      如果指定此参数，则忽略 x, y, z 参数。
            z: Z 位置 (mm)，与 x, y 配合使用
            x: X 位置 (mm)，默认 0.0
            y: Y 位置 (mm)，默认 0.0
            radius: 曲率半径 (mm)，默认 inf（平面）。
                    正值表示凸面，负值表示凹面。
            conic: 圆锥常数，默认 0（球面），-1 为抛物面
            is_mirror: 是否为反射镜，默认 False
            tilt_x: 绕 X 轴旋转角度（度），默认 0
            tilt_y: 绕 Y 轴旋转角度（度），默认 0
            tilt_z: 绕 Z 轴旋转角度（度），默认 0
            material: 材料名称，默认空字符串（空气）
        
        返回:
            self（支持链式调用）
        
        注意:
            🚫 禁止设置口径/半口径参数！光束范围由 w0 自然决定。
        
        示例:
            >>> system = bts.OpticalSystem()
            >>> # 方式 1：使用 position 元组
            >>> system.add_surface(position=(0, 0, 100), radius=200, is_mirror=True)
            >>> 
            >>> # 方式 2：使用 x, y, z 参数
            >>> system.add_surface(z=100, radius=200, is_mirror=True, tilt_x=45)
        
        **Validates: Requirements 2.4**
        """
        # 确定位置
        if position is not None:
            pos = position
        elif z is not None:
            pos = (x, y, z)
        else:
            raise ValueError("必须指定 position 或 z 参数")
        
        # 创建 SurfaceDefinition
        surface_def = SurfaceDefinition(
            index=len(self._surfaces),
            surface_type='standard',
            position=pos,
            radius=radius,
            conic=conic,
            is_mirror=is_mirror,
            tilt_x=tilt_x,
            tilt_y=tilt_y,
            tilt_z=tilt_z,
            material=material if material else ('MIRROR' if is_mirror else ''),
        )
        self._surfaces.append(surface_def)
        
        # 创建 GlobalSurfaceDefinition
        global_surface = self._create_global_surface(
            position=pos,
            radius=radius,
            conic=conic,
            is_mirror=is_mirror,
            tilt_x=tilt_x,
            tilt_y=tilt_y,
            tilt_z=tilt_z,
            material=material if material else ('MIRROR' if is_mirror else ''),
        )
        self._global_surfaces.append(global_surface)
        
        return self
    
    def add_flat_mirror(
        self,
        position: Optional[Tuple[float, float, float]] = None,
        z: Optional[float] = None,
        x: float = 0.0,
        y: float = 0.0,
        tilt_x: float = 0.0,
        tilt_y: float = 0.0,
        tilt_z: float = 0.0,
    ) -> "OpticalSystem":
        """添加平面反射镜（支持链式调用）
        
        参数:
            position: 顶点位置 (x, y, z) (mm)，使用绝对坐标
            z: Z 位置 (mm)，与 x, y 配合使用
            x: X 位置 (mm)，默认 0.0
            y: Y 位置 (mm)，默认 0.0
            tilt_x: 绕 X 轴旋转角度（度），默认 0
            tilt_y: 绕 Y 轴旋转角度（度），默认 0
            tilt_z: 绕 Z 轴旋转角度（度），默认 0
        
        返回:
            self（支持链式调用）
        
        注意:
            🚫 禁止设置口径/半口径参数！
        
        示例:
            >>> system = bts.OpticalSystem()
            >>> system.add_flat_mirror(z=50, tilt_x=45)  # 45° 折叠镜
        
        **Validates: Requirements 2.5**
        """
        return self.add_surface(
            position=position,
            z=z,
            x=x,
            y=y,
            radius=float('inf'),
            conic=0.0,
            is_mirror=True,
            tilt_x=tilt_x,
            tilt_y=tilt_y,
            tilt_z=tilt_z,
            material='MIRROR',
        )
    
    def add_spherical_mirror(
        self,
        radius: float,
        position: Optional[Tuple[float, float, float]] = None,
        z: Optional[float] = None,
        x: float = 0.0,
        y: float = 0.0,
        tilt_x: float = 0.0,
        tilt_y: float = 0.0,
        tilt_z: float = 0.0,
    ) -> "OpticalSystem":
        """添加球面反射镜（支持链式调用）
        
        参数:
            radius: 曲率半径 (mm)，正值为凸面镜（发散），负值为凹面镜（聚焦）
            position: 顶点位置 (x, y, z) (mm)，使用绝对坐标
            z: Z 位置 (mm)，与 x, y 配合使用
            x: X 位置 (mm)，默认 0.0
            y: Y 位置 (mm)，默认 0.0
            tilt_x: 绕 X 轴旋转角度（度），默认 0
            tilt_y: 绕 Y 轴旋转角度（度），默认 0
            tilt_z: 绕 Z 轴旋转角度（度），默认 0
        
        返回:
            self（支持链式调用）
        
        注意:
            🚫 禁止设置口径/半口径参数！
        
        示例:
            >>> system = bts.OpticalSystem()
            >>> system.add_spherical_mirror(z=100, radius=-200)  # 凹面镜，f=100mm
        
        **Validates: Requirements 2.6**
        """
        return self.add_surface(
            position=position,
            z=z,
            x=x,
            y=y,
            radius=radius,
            conic=0.0,
            is_mirror=True,
            tilt_x=tilt_x,
            tilt_y=tilt_y,
            tilt_z=tilt_z,
            material='MIRROR',
        )
    
    def add_parabolic_mirror(
        self,
        radius: float,
        position: Optional[Tuple[float, float, float]] = None,
        z: Optional[float] = None,
        x: float = 0.0,
        y: float = 0.0,
        tilt_x: float = 0.0,
        tilt_y: float = 0.0,
        tilt_z: float = 0.0,
    ) -> "OpticalSystem":
        """添加抛物面反射镜（支持链式调用）
        
        离轴抛物面镜（OAP）通过 (x, y) 坐标指定离轴量。
        
        参数:
            radius: 曲率半径 (mm)，R = 2f。正值为凸面，负值为凹面。
            position: 顶点位置 (x, y, z) (mm)，使用绝对坐标
            z: Z 位置 (mm)，与 x, y 配合使用
            x: X 位置 (mm)，默认 0.0
            y: Y 位置 (mm)，默认 0.0，离轴量由此坐标决定
            tilt_x: 绕 X 轴旋转角度（度），默认 0
            tilt_y: 绕 Y 轴旋转角度（度），默认 0
            tilt_z: 绕 Z 轴旋转角度（度），默认 0
        
        返回:
            self（支持链式调用）
        
        注意:
            🚫 禁止设置口径/半口径参数！
            离轴量由 (x, y) 坐标自然决定，无需额外参数。
        
        示例:
            >>> system = bts.OpticalSystem()
            >>> # 同轴抛物面镜
            >>> system.add_parabolic_mirror(z=100, radius=200)
            >>> 
            >>> # 离轴抛物面镜（OAP），Y 方向离轴 100mm
            >>> system.add_parabolic_mirror(z=0, y=100, radius=200)
        
        **Validates: Requirements 2.6**
        """
        return self.add_surface(
            position=position,
            z=z,
            x=x,
            y=y,
            radius=radius,
            conic=-1.0,  # 抛物面
            is_mirror=True,
            tilt_x=tilt_x,
            tilt_y=tilt_y,
            tilt_z=tilt_z,
            material='MIRROR',
        )
    
    def add_paraxial_lens(
        self,
        focal_length: float,
        position: Optional[Tuple[float, float, float]] = None,
        z: Optional[float] = None,
        x: float = 0.0,
        y: float = 0.0,
    ) -> "OpticalSystem":
        """添加薄透镜（支持链式调用）
        
        参数:
            focal_length: 焦距 (mm)
            position: 顶点位置 (x, y, z) (mm)，使用绝对坐标
            z: Z 位置 (mm)，与 x, y 配合使用
            x: X 位置 (mm)，默认 0.0
            y: Y 位置 (mm)，默认 0.0
        
        返回:
            self（支持链式调用）
        
        注意:
            🚫 禁止设置口径/半口径参数！
        
        示例:
            >>> system = bts.OpticalSystem()
            >>> system.add_paraxial_lens(z=50, focal_length=100)  # f=100mm 薄透镜
        
        **Validates: Requirements 2.7**
        """
        # 确定位置
        if position is not None:
            pos = position
        elif z is not None:
            pos = (x, y, z)
        else:
            raise ValueError("必须指定 position 或 z 参数")
        
        # 创建 SurfaceDefinition
        surface_def = SurfaceDefinition(
            index=len(self._surfaces),
            surface_type='paraxial',
            position=pos,
            radius=float('inf'),
            conic=0.0,
            is_mirror=False,
            tilt_x=0.0,
            tilt_y=0.0,
            material='',
            focal_length=focal_length,
        )
        self._surfaces.append(surface_def)
        
        # 创建 GlobalSurfaceDefinition
        global_surface = self._create_global_surface(
            position=pos,
            radius=float('inf'),
            conic=0.0,
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
              位置: (0.000, 0.000, 50.000) mm
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
            # 显示完整的 (x, y, z) 位置
            print(f"  位置: ({surface.x:.3f}, {surface.y:.3f}, {surface.z:.3f}) mm")
            
            # 曲率半径
            if np.isinf(surface.radius):
                print(f"  曲率半径: 无穷大 (平面)")
            else:
                print(f"  曲率半径: {surface.radius:.3f} mm")
            
            # 圆锥常数（仅非零时显示）
            if surface.conic != 0:
                conic_type = self._get_conic_type(surface.conic)
                print(f"  圆锥常数: {surface.conic:.6f} ({conic_type})")
            
            # 反射镜标识
            if surface.is_mirror:
                print(f"  反射镜: 是")
            
            # 材料（非空气时显示）
            if surface.material and surface.material.upper() not in ('', 'AIR', 'MIRROR'):
                print(f"  材料: {surface.material}")
            
            # 倾斜角度（非零时显示）
            if surface.tilt_x != 0 or surface.tilt_y != 0 or surface.tilt_z != 0:
                print(f"  倾斜: tilt_x = {surface.tilt_x:.2f}°, tilt_y = {surface.tilt_y:.2f}°, tilt_z = {surface.tilt_z:.2f}°")
            
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
        mode: str = '2d',
        projection: str = "YZ",
        num_rays: int = 5,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> Tuple[Any, Any]:
        """绘制光路图
        
        复用现有的 ZmxOpticLoader 和 view_2d/view_3d 功能进行可视化。
        
        参数:
            mode: 可视化模式，'2d' 或 '3d'，默认 '2d'
            projection: 投影平面 ('YZ', 'XZ', 'XY')，默认 'YZ'（仅 2D）
            num_rays: 光线数量，默认 5
            save_path: 保存路径（可选），如果指定则保存图像（仅 2D）
            show: 是否显示图形，默认 True
        
        返回:
            mode='2d': (fig, ax) 元组
            mode='3d': None
        
        示例:
            >>> system = bts.load_zmx("system.zmx")
            >>> fig, ax = system.plot_layout(mode='2d', projection='YZ')
            >>> 
            >>> # 3D 可视化
            >>> system.plot_layout(mode='3d')
        
        **Validates: Requirements 4.2, 4.4**
        """
        import matplotlib.pyplot as plt
        
        # 检查系统是否为空
        if len(self._surfaces) == 0:
            print("警告: 系统为空，无法绘制光路图")
            if mode == '2d':
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, "系统为空", ha='center', va='center', fontsize=14)
                ax.set_title(f"光学系统: {self.name}")
                return fig, ax
            return None
        
        # 3D 可视化处理
        if mode == '3d':
            try:
                optic = self._create_optiland_optic()
                from sequential_system.zmx_visualization import view_3d
                if show:
                    print(f"正在打开 3D 视图: {self.name}...")
                    view_3d(optic)
                return None
            except Exception as e:
                print(f"错误: 无法使用 3D 可视化 ({e})")
                return None
            
        # 2D 可视化处理 (mode='2d')
        # 尝试创建 optiland Optic 对象进行可视化
        
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
            tilt_x_rad = np.radians(surface.tilt_x)
            tilt_y_rad = np.radians(surface.tilt_y)
            tilt_z_rad = np.radians(surface.tilt_z)
            
            # 使用 CoordinateBreakProcessor 计算旋转矩阵
            orientation = CoordinateBreakProcessor.rotation_matrix_xyz(
                tilt_x_rad, tilt_y_rad, tilt_z_rad
            )
            
            # 使用完整的 (x, y, z) 位置
            global_surface = GlobalSurfaceDefinition(
                index=surface.index,
                surface_type=surface.surface_type,
                vertex_position=np.array([surface.x, surface.y, surface.z]),
                orientation=orientation,
                radius=surface.radius,
                conic=surface.conic,
                semi_aperture=1000.0,  # 固定大值，不允许用户设置
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
