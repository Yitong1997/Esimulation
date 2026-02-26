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
from typing import Callable, List, Optional, Tuple, Union, TYPE_CHECKING, Sequence

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


def _coerce_vtk_color(color, fallback=(0.0, 0.0, 0.0)) -> Tuple[float, float, float]:
    if color is None:
        return fallback
    if isinstance(color, (list, tuple)) and len(color) == 3:
        vals = [float(v) for v in color]
        if any(v > 1.0 for v in vals):
            return (vals[0] / 255.0, vals[1] / 255.0, vals[2] / 255.0)
        return (vals[0], vals[1], vals[2])
    if isinstance(color, str):
        name = color.strip().lower()
        named = {
            "black": (0.0, 0.0, 0.0),
            "white": (1.0, 1.0, 1.0),
            "gray": (0.5, 0.5, 0.5),
            "grey": (0.5, 0.5, 0.5),
            "tab:blue": (0.122, 0.467, 0.706),
            "tab:orange": (1.0, 0.498, 0.055),
            "tab:green": (0.173, 0.627, 0.173),
            "tab:red": (0.839, 0.153, 0.157),
            "tab:purple": (0.580, 0.404, 0.741),
        }
        if name in named:
            return named[name]
        if name.startswith("#") and len(name) == 7:
            try:
                r = int(name[1:3], 16) / 255.0
                g = int(name[3:5], 16) / 255.0
                b = int(name[5:7], 16) / 255.0
                return (r, g, b)
            except ValueError:
                return fallback
    return fallback


def _vtk_points_actor(points, color, point_size):
    import vtk

    if points is None or len(points) == 0:
        return None
    vtk_points = vtk.vtkPoints()
    verts = vtk.vtkCellArray()
    for p in points:
        pid = vtk_points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
        verts.InsertNextCell(1)
        verts.InsertCellPoint(pid)
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)
    poly.SetVerts(verts)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetPointSize(float(point_size))
    return actor


def _vtk_lines_actor(segments, color, line_width):
    import vtk

    if not segments:
        return None
    vtk_points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    for start, end in segments:
        pid0 = vtk_points.InsertNextPoint(float(start[0]), float(start[1]), float(start[2]))
        pid1 = vtk_points.InsertNextPoint(float(end[0]), float(end[1]), float(end[2]))
        lines.InsertNextCell(2)
        lines.InsertCellPoint(pid0)
        lines.InsertCellPoint(pid1)
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)
    poly.SetLines(lines)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetLineWidth(float(line_width))
    return actor


def _vtk_polyline_actor(points, color, line_width):
    import vtk

    if points is None or len(points) < 2:
        return None
    vtk_points = vtk.vtkPoints()
    for p in points:
        vtk_points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
    polyline = vtk.vtkPolyLine()
    polyline.GetPointIds().SetNumberOfIds(len(points))
    for i in range(len(points)):
        polyline.GetPointIds().SetId(i, i)
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(polyline)
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)
    poly.SetLines(cells)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetLineWidth(float(line_width))
    return actor


def _vtk_add_text(renderer, text, position, color, font_size=12, scale=0.6):
    import vtk

    if hasattr(vtk, "vtkBillboardTextActor3D"):
        actor = vtk.vtkBillboardTextActor3D()
        actor.SetInput(text)
        actor.SetPosition(float(position[0]), float(position[1]), float(position[2]))
        actor.GetTextProperty().SetColor(*color)
        actor.GetTextProperty().SetFontSize(int(font_size))
        renderer.AddActor(actor)
        return actor
    text_source = vtk.vtkVectorText()
    text_source.SetText(text)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(text_source.GetOutputPort())
    actor = vtk.vtkFollower()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.SetScale(float(scale))
    actor.SetPosition(float(position[0]), float(position[1]), float(position[2]))
    actor.SetCamera(renderer.GetActiveCamera())
    renderer.AddActor(actor)
    return actor


def _add_overlay_3d(
    renderer,
    surfaces: Sequence["GlobalSurfaceDefinition"],
    *,
    show_normals: bool,
    normal_scale: Optional[float],
    normal_color,
    show_axes: bool,
    axis_scale: Optional[float],
    axis_labels: bool,
    show_optical_axis: bool,
    optical_axis_color,
    annotate_surface_ids: bool,
    annotate_intersections: bool,
    point_color,
    point_size: float,
    line_width: float,
    text_color,
    coord_color,
    text_size: int,
    label_offset: Optional[float],
) -> None:
    import numpy as np

    if not surfaces:
        return
    vertices = np.array([s.vertex_position for s in surfaces], dtype=float)
    if vertices.size == 0:
        return

    point_color = _coerce_vtk_color(point_color, fallback=(0.122, 0.467, 0.706))
    normal_color = _coerce_vtk_color(normal_color, fallback=(0.580, 0.404, 0.741))
    optical_axis_color = _coerce_vtk_color(optical_axis_color, fallback=(0.839, 0.153, 0.157))
    text_color = _coerce_vtk_color(text_color, fallback=(0.0, 0.0, 0.0))
    coord_color = _coerce_vtk_color(coord_color, fallback=(0.5, 0.5, 0.5))

    points_actor = _vtk_points_actor(vertices, point_color, point_size)
    if points_actor is not None:
        renderer.AddActor(points_actor)

    if show_optical_axis and len(vertices) > 1:
        axis_actor = _vtk_polyline_actor(vertices, optical_axis_color, line_width)
        if axis_actor is not None:
            renderer.AddActor(axis_actor)

    axis_x_segments = []
    axis_y_segments = []
    axis_z_segments = []
    normal_segments = []

    for surface in surfaces:
        origin = np.asarray(surface.vertex_position, dtype=float)
        semi_ap = float(getattr(surface, "semi_aperture", 0.0) or 0.0)
        axis_len = axis_scale if axis_scale is not None else max(4.0, semi_ap * 0.4)
        normal_len = normal_scale if normal_scale is not None else max(5.0, semi_ap * 0.6)
        if show_axes:
            x_axis = surface.orientation[:, 0] * axis_len
            y_axis = surface.orientation[:, 1] * axis_len
            z_axis = surface.orientation[:, 2] * axis_len
            axis_x_segments.append((origin, origin + x_axis))
            axis_y_segments.append((origin, origin + y_axis))
            axis_z_segments.append((origin, origin + z_axis))
            if axis_labels:
                _vtk_add_text(
                    renderer,
                    "X",
                    origin + x_axis,
                    _coerce_vtk_color("tab:red", fallback=(0.839, 0.153, 0.157)),
                    font_size=text_size,
                )
                _vtk_add_text(
                    renderer,
                    "Y",
                    origin + y_axis,
                    _coerce_vtk_color("tab:green", fallback=(0.173, 0.627, 0.173)),
                    font_size=text_size,
                )
                _vtk_add_text(
                    renderer,
                    "Z",
                    origin + z_axis,
                    _coerce_vtk_color("tab:blue", fallback=(0.122, 0.467, 0.706)),
                    font_size=text_size,
                )
        if show_normals:
            normal = surface.orientation[:, 2] * normal_len
            normal_segments.append((origin, origin + normal))

        label_offset_vec = np.zeros(3, dtype=float)
        if label_offset is not None:
            label_offset_vec = surface.orientation[:, 2] * float(label_offset)

        if annotate_surface_ids:
            _vtk_add_text(
                renderer,
                f"S{surface.index}",
                origin + label_offset_vec,
                text_color,
                font_size=text_size,
            )
        if annotate_intersections:
            _vtk_add_text(
                renderer,
                f"({origin[0]:.2f},{origin[1]:.2f},{origin[2]:.2f})",
                origin + label_offset_vec,
                coord_color,
                font_size=max(8, int(text_size * 0.8)),
                scale=0.5,
            )

    if show_axes:
        x_actor = _vtk_lines_actor(axis_x_segments, _coerce_vtk_color("tab:red"), line_width)
        y_actor = _vtk_lines_actor(axis_y_segments, _coerce_vtk_color("tab:green"), line_width)
        z_actor = _vtk_lines_actor(axis_z_segments, _coerce_vtk_color("tab:blue"), line_width)
        for actor in (x_actor, y_actor, z_actor):
            if actor is not None:
                renderer.AddActor(actor)

    if show_normals:
        n_actor = _vtk_lines_actor(normal_segments, normal_color, line_width)
        if n_actor is not None:
            renderer.AddActor(n_actor)


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
    
    overlay_surfaces = kwargs.pop("overlay_surfaces", None)
    overlay_show_normals = bool(kwargs.pop("overlay_show_normals", False))
    overlay_normal_scale = kwargs.pop("overlay_normal_scale", None)
    overlay_normal_color = kwargs.pop("overlay_normal_color", "tab:purple")
    overlay_show_axes = bool(kwargs.pop("overlay_show_axes", False))
    overlay_axis_scale = kwargs.pop("overlay_axis_scale", None)
    overlay_axis_labels = bool(kwargs.pop("overlay_axis_labels", False))
    overlay_show_optical_axis = bool(kwargs.pop("overlay_show_optical_axis", False))
    overlay_optical_axis_color = kwargs.pop("overlay_optical_axis_color", "tab:red")
    overlay_annotate_surface_ids = bool(kwargs.pop("overlay_annotate_surface_ids", False))
    overlay_annotate_intersections = bool(kwargs.pop("overlay_annotate_intersections", False))
    overlay_point_color = kwargs.pop("overlay_point_color", "tab:blue")
    overlay_point_size = float(kwargs.pop("overlay_point_size", 4.0))
    overlay_line_width = float(kwargs.pop("overlay_line_width", 1.2))
    overlay_text_color = kwargs.pop("overlay_text_color", "black")
    overlay_coord_color = kwargs.pop("overlay_coord_color", "gray")
    overlay_text_size = int(kwargs.pop("overlay_text_size", 12))
    overlay_label_offset = kwargs.pop("overlay_label_offset", None)

    overlay_enabled = (
        overlay_surfaces is not None
        and (
            overlay_show_normals
            or overlay_show_axes
            or overlay_show_optical_axis
            or overlay_annotate_surface_ids
            or overlay_annotate_intersections
        )
    )

    renderer_hook = None
    if overlay_enabled:
        def renderer_hook(renderer):
            _add_overlay_3d(
                renderer,
                overlay_surfaces,
                show_normals=overlay_show_normals,
                normal_scale=overlay_normal_scale,
                normal_color=overlay_normal_color,
                show_axes=overlay_show_axes,
                axis_scale=overlay_axis_scale,
                axis_labels=overlay_axis_labels,
                show_optical_axis=overlay_show_optical_axis,
                optical_axis_color=overlay_optical_axis_color,
                annotate_surface_ids=overlay_annotate_surface_ids,
                annotate_intersections=overlay_annotate_intersections,
                point_color=overlay_point_color,
                point_size=overlay_point_size,
                line_width=overlay_line_width,
                text_color=overlay_text_color,
                coord_color=overlay_coord_color,
                text_size=overlay_text_size,
                label_offset=overlay_label_offset,
            )

    viewer = OpticViewer3D(optic)
    viewer.view(renderer_hook=renderer_hook, **kwargs)


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
