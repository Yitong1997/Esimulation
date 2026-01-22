"""
ZMX 文件可视化模块

本模块提供基于 zemax-optical-axis-tracing 规范的 ZMX 文件可视化功能。
核心设计原则是最大程度复用 optiland 的可视化模块（OpticViewer、OpticViewer3D）。

主要组件：
- ZmxOpticLoader: ZMX 文件到 optiland Optic 的加载器
- visualize_zmx(): 一站式 ZMX 文件可视化便捷函数
- view_2d(): 2D 可视化辅助函数
- view_3d(): 3D 可视化辅助函数

使用示例：
    >>> from sequential_system.zmx_visualization import visualize_zmx
    >>> import matplotlib.pyplot as plt
    >>> 
    >>> # 2D 可视化
    >>> fig, ax, _ = visualize_zmx('system.zmx', mode='2d')
    >>> plt.show()
    >>> 
    >>> # 3D 可视化
    >>> visualize_zmx('system.zmx', mode='3d')

作者：混合光学仿真项目
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from optiland.optic import Optic
    from sequential_system.zmx_parser import ZmxDataModel
    from sequential_system.coordinate_system import GlobalSurfaceDefinition


class ZmxOpticLoader:
    """ZMX 文件到 optiland Optic 的加载器
    
    封装 ZMX → ZmxParser → SurfaceTraversalAlgorithm → ZemaxToOptilandConverter
    的完整转换流程。
    
    属性：
        zmx_file_path: ZMX 文件路径
        zmx_data: 解析后的 ZMX 数据模型
        global_surfaces: 全局坐标表面定义列表
        optic: optiland Optic 对象
    
    示例：
        >>> loader = ZmxOpticLoader('system.zmx')
        >>> optic = loader.load()
        >>> 
        >>> # 访问中间数据
        >>> for surface in loader.global_surfaces:
        ...     print(f"表面 {surface.index}: {surface.comment}")
    """
    
    def __init__(self, zmx_file_path: Union[str, Path]):
        """初始化加载器
        
        参数：
            zmx_file_path: ZMX 文件路径
        """
        self.zmx_file_path = Path(zmx_file_path)
        self._zmx_data: Optional['ZmxDataModel'] = None
        self._global_surfaces: Optional[List['GlobalSurfaceDefinition']] = None
        self._optic: Optional['Optic'] = None
    
    def load(self) -> 'Optic':
        """加载 ZMX 文件并转换为 optiland Optic 对象
        
        返回：
            optiland Optic 对象
        
        异常：
            FileNotFoundError: ZMX 文件不存在
        """
        if not self.zmx_file_path.exists():
            raise FileNotFoundError(f"ZMX 文件不存在: {self.zmx_file_path}")
        
        from sequential_system.zmx_parser import ZmxParser
        from sequential_system.coordinate_system import (
            SurfaceTraversalAlgorithm,
            ZemaxToOptilandConverter,
        )
        
        # 1. 解析 ZMX 文件
        parser = ZmxParser(str(self.zmx_file_path))
        self._zmx_data = parser.parse()
        
        # 2. 遍历表面，生成全局坐标定义
        traversal = SurfaceTraversalAlgorithm(self._zmx_data)
        self._global_surfaces = traversal.traverse()
        
        # 3. 确定波长和入瞳直径
        if self._zmx_data.wavelengths:
            wavelength = self._zmx_data.wavelengths[
                self._zmx_data.primary_wavelength_index
            ]
        else:
            wavelength = 0.55  # 默认可见光波长 (μm)
        
        epd = self._zmx_data.entrance_pupil_diameter
        if epd <= 0:
            epd = 10.0  # 默认入瞳直径 (mm)
        
        # 4. 转换为 optiland Optic
        converter = ZemaxToOptilandConverter(
            self._global_surfaces,
            wavelength=wavelength,
            entrance_pupil_diameter=epd
        )
        self._optic = converter.convert()
        
        return self._optic
    
    @property
    def zmx_data(self) -> Optional['ZmxDataModel']:
        """获取解析后的 ZMX 数据模型"""
        return self._zmx_data
    
    @property
    def global_surfaces(self) -> Optional[List['GlobalSurfaceDefinition']]:
        """获取全局坐标表面定义列表"""
        return self._global_surfaces
    
    @property
    def optic(self) -> Optional['Optic']:
        """获取 optiland Optic 对象"""
        return self._optic
    
    def print_surface_info(self) -> None:
        """打印表面信息摘要"""
        if self._global_surfaces is None:
            print("尚未加载 ZMX 文件，请先调用 load() 方法")
            return
        
        print(f"\n{'='*60}")
        print(f"ZMX 文件: {self.zmx_file_path.name}")
        print(f"共 {len(self._global_surfaces)} 个光学表面")
        print(f"{'='*60}")
        
        for surface in self._global_surfaces:
            print(f"\n表面 {surface.index}: {surface.comment or '(无名称)'}")
            print(f"  类型: {surface.surface_type}")
            print(f"  位置: ({surface.vertex_position[0]:.3f}, "
                  f"{surface.vertex_position[1]:.3f}, "
                  f"{surface.vertex_position[2]:.3f}) mm")
            
            if surface.is_mirror:
                print(f"  反射镜: 是")
            
            if surface.surface_type == 'paraxial':
                # 近轴面形显示焦距
                if not np.isinf(surface.focal_length):
                    print(f"  焦距: {surface.focal_length:.3f} mm")
            elif surface.surface_type == 'biconic':
                # 双锥面显示两个方向的曲率半径和圆锥常数
                print(f"  双锥面参数:")
                if not np.isinf(surface.radius):
                    print(f"    Y 方向曲率半径: {surface.radius:.3f} mm")
                else:
                    print(f"    Y 方向曲率半径: 无穷大 (平面)")
                if surface.conic != 0:
                    print(f"    Y 方向圆锥常数: {surface.conic:.6f}")
                if not np.isinf(surface.radius_x):
                    print(f"    X 方向曲率半径: {surface.radius_x:.3f} mm")
                else:
                    print(f"    X 方向曲率半径: 无穷大 (平面)")
                if surface.conic_x != 0:
                    print(f"    X 方向圆锥常数: {surface.conic_x:.6f}")
            else:
                # 标准表面显示曲率半径
                if not np.isinf(surface.radius):
                    print(f"  曲率半径: {surface.radius:.3f} mm")
                if surface.conic != 0:
                    print(f"  圆锥常数: {surface.conic:.6f}")


import numpy as np


def view_2d(
    optic: 'Optic',
    projection: str = 'YZ',
    num_rays: int = 3,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs
) -> Tuple:
    """使用 optiland OpticViewer 进行 2D 可视化
    
    参数：
        optic: optiland Optic 对象
        projection: 投影平面 ('YZ', 'XZ', 'XY')
        num_rays: 光线数量
        figsize: 图形大小
        **kwargs: 传递给 OpticViewer.view() 的其他参数
    
    返回：
        (fig, ax, interaction_manager) 元组
    """
    from optiland.visualization.system.optic_viewer import OpticViewer
    
    viewer = OpticViewer(optic)
    
    # 构建 view 参数
    view_kwargs = {
        'projection': projection,
        'num_rays': num_rays,
    }
    if figsize is not None:
        view_kwargs['figsize'] = figsize
    
    # 合并用户提供的其他参数
    view_kwargs.update(kwargs)
    
    return viewer.view(**view_kwargs)


def view_3d(optic: 'Optic', **kwargs) -> None:
    """使用 optiland OpticViewer3D 进行 3D 可视化
    
    参数：
        optic: optiland Optic 对象
        **kwargs: 传递给 OpticViewer3D.view() 的参数
    
    注意：
        需要安装 VTK 库才能使用 3D 可视化功能。
    """
    try:
        from optiland.visualization.system.optic_viewer_3d import OpticViewer3D
    except ImportError:
        print("错误: 3D 可视化需要安装 VTK 库")
        print("请运行: pip install vtk")
        return
    
    viewer = OpticViewer3D(optic)
    viewer.view(**kwargs)


def visualize_zmx(
    zmx_file_path: Union[str, Path],
    mode: str = '2d',
    projection: str = 'YZ',
    num_rays: int = 3,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    show_info: bool = False,
    **kwargs
) -> Optional[Tuple]:
    """可视化 ZMX 文件定义的光学系统
    
    一站式便捷函数，直接复用 optiland 的 OpticViewer/OpticViewer3D 进行渲染。
    
    参数：
        zmx_file_path: ZMX 文件路径
        mode: '2d' 或 '3d'
        projection: 2D 投影平面 ('YZ', 'XZ', 'XY')，仅 mode='2d' 时有效
        num_rays: 光线数量
        figsize: 图形大小
        title: 图形标题
        show_info: 是否打印表面信息摘要
        **kwargs: 传递给 OpticViewer.view() 的其他参数
    
    返回：
        mode='2d': (fig, ax, interaction_manager) 元组
        mode='3d': None (VTK 窗口)
    
    示例：
        >>> # 2D 可视化
        >>> fig, ax, _ = visualize_zmx('system.zmx', mode='2d', projection='YZ')
        >>> plt.show()
        >>> 
        >>> # 3D 可视化
        >>> visualize_zmx('system.zmx', mode='3d')
    """
    # 加载 ZMX 文件
    loader = ZmxOpticLoader(zmx_file_path)
    optic = loader.load()
    
    # 打印表面信息（如果需要）
    if show_info:
        loader.print_surface_info()
    
    # 根据模式选择可视化方法
    if mode.lower() == '2d':
        result = view_2d(
            optic,
            projection=projection,
            num_rays=num_rays,
            figsize=figsize,
            **kwargs
        )
        
        # 设置标题
        if title is not None and result is not None:
            fig, ax, _ = result
            ax.set_title(title)
        
        return result
    
    elif mode.lower() == '3d':
        view_3d(optic, **kwargs)
        return None
    
    else:
        raise ValueError(f"不支持的可视化模式: {mode}，请使用 '2d' 或 '3d'")


def load_zmx_optic(zmx_file_path: Union[str, Path]) -> 'Optic':
    """加载 ZMX 文件并返回 optiland Optic 对象
    
    这是一个简化的便捷函数，仅返回 Optic 对象。
    如果需要访问中间数据（如全局坐标表面定义），请使用 ZmxOpticLoader 类。
    
    参数：
        zmx_file_path: ZMX 文件路径
    
    返回：
        optiland Optic 对象
    
    示例：
        >>> optic = load_zmx_optic('system.zmx')
        >>> print(f"共 {optic.surface_count} 个表面")
    """
    loader = ZmxOpticLoader(zmx_file_path)
    return loader.load()


# 导出的公共接口
__all__ = [
    'ZmxOpticLoader',
    'visualize_zmx',
    'view_2d',
    'view_3d',
    'load_zmx_optic',
]
