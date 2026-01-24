"""
I/O 函数模块

提供 ZMX 文件加载等 I/O 功能。

本模块复用现有的 ZMX 解析器和坐标转换逻辑，
将 ZMX 文件中的光学系统定义转换为 OpticalSystem 对象。

**Validates: Requirements 1.1, 2.1, 2.2**
"""

from pathlib import Path
from typing import TYPE_CHECKING
import os

from .exceptions import ParseError

if TYPE_CHECKING:
    from .optical_system import OpticalSystem


def load_zmx(path: str) -> "OpticalSystem":
    """从 ZMX 文件加载光学系统
    
    解析 ZMX 文件并创建 OpticalSystem 对象。
    内部调用现有的 ZMX 解析器和坐标转换逻辑。
    
    参数:
        path: ZMX 文件路径（绝对路径或相对路径）
    
    返回:
        OpticalSystem 对象，包含从 ZMX 文件加载的所有表面定义
    
    异常:
        FileNotFoundError: 文件不存在
        ParseError: 解析错误（ZMX 文件格式无效或包含不支持的特性）
    
    示例:
        >>> import bts
        >>> 
        >>> # 加载 ZMX 文件
        >>> system = bts.load_zmx("path/to/system.zmx")
        >>> 
        >>> # 查看系统信息
        >>> system.print_info()
        >>> 
        >>> # 绘制光路图
        >>> system.plot_layout()
    
    **Validates: Requirements 1.1, 2.1, 2.2**
    """
    # 导入所需模块（延迟导入避免循环依赖）
    from .optical_system import OpticalSystem, SurfaceDefinition
    from hybrid_optical_propagation import load_optical_system_from_zmx
    from sequential_system.zmx_parser import ZmxParser, ZmxParseError, ZmxUnsupportedError
    import numpy as np
    
    # 1. 检查文件是否存在
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"ZMX 文件不存在: {path}")
    
    if not filepath.is_file():
        raise FileNotFoundError(f"路径不是文件: {path}")
    
    # 2. 解析 ZMX 文件
    try:
        # 使用现有的 load_optical_system_from_zmx 函数加载全局表面定义
        global_surfaces = load_optical_system_from_zmx(str(filepath))
        
        # 同时解析 ZMX 文件获取原始数据（用于提取更多信息）
        parser = ZmxParser(str(filepath))
        zmx_data = parser.parse()
        
    except ZmxUnsupportedError as e:
        # 不支持的 ZMX 特性
        raise ParseError(
            message=str(e),
            file_path=str(filepath),
        ) from e
    except ZmxParseError as e:
        # ZMX 解析错误
        raise ParseError(
            message=str(e),
            file_path=str(filepath),
            line_number=getattr(e, 'line_number', None),
        ) from e
    except Exception as e:
        # 其他解析错误
        raise ParseError(
            message=f"解析 ZMX 文件失败: {e}",
            file_path=str(filepath),
        ) from e
    
    # 3. 创建 OpticalSystem 对象
    system_name = filepath.stem  # 使用文件名（不含扩展名）作为系统名称
    system = OpticalSystem(name=system_name)
    
    # 4. 设置源文件路径
    system._source_path = str(filepath)
    
    # 5. 设置全局表面定义（直接使用解析结果）
    system._global_surfaces = global_surfaces
    
    # 6. 从全局表面定义创建 SurfaceDefinition 列表
    for gs in global_surfaces:
        # 从全局表面定义提取参数
        # 计算倾斜角度（从姿态矩阵反推）
        tilt_x, tilt_y = _extract_tilt_angles(gs.orientation)
        
        # 创建 SurfaceDefinition，使用完整的 (x, y, z) 位置
        surface_def = SurfaceDefinition(
            index=gs.index,
            surface_type=gs.surface_type,
            position=(
                float(gs.vertex_position[0]),
                float(gs.vertex_position[1]),
                float(gs.vertex_position[2]),
            ),
            radius=gs.radius,
            conic=gs.conic,
            semi_aperture=gs.semi_aperture,
            is_mirror=gs.is_mirror,
            tilt_x=tilt_x,
            tilt_y=tilt_y,
            material=gs.material,
            focal_length=gs.focal_length if not np.isinf(gs.focal_length) else None,
        )
        system._surfaces.append(surface_def)
    
    return system


def _extract_tilt_angles(orientation: "np.ndarray") -> tuple:
    """从姿态矩阵提取倾斜角度
    
    假设旋转顺序为：先绕 X 轴，再绕 Y 轴。
    
    参数:
        orientation: 3x3 旋转矩阵
    
    返回:
        (tilt_x_deg, tilt_y_deg) 元组，单位为度
    """
    import numpy as np
    
    # 从旋转矩阵提取角度
    # 假设 R = Ry @ Rx
    # R[2,0] = sin(tilt_y)
    # R[2,1] = -sin(tilt_x) * cos(tilt_y)
    # R[2,2] = cos(tilt_x) * cos(tilt_y)
    
    # 提取 tilt_y
    sin_tilt_y = orientation[2, 0]
    sin_tilt_y = np.clip(sin_tilt_y, -1.0, 1.0)  # 防止数值误差
    tilt_y_rad = np.arcsin(sin_tilt_y)
    
    # 提取 tilt_x
    cos_tilt_y = np.cos(tilt_y_rad)
    if abs(cos_tilt_y) > 1e-10:
        sin_tilt_x = -orientation[2, 1] / cos_tilt_y
        cos_tilt_x = orientation[2, 2] / cos_tilt_y
        sin_tilt_x = np.clip(sin_tilt_x, -1.0, 1.0)
        tilt_x_rad = np.arctan2(sin_tilt_x, cos_tilt_x)
    else:
        # 万向节锁定情况，tilt_y = ±90°
        tilt_x_rad = 0.0
    
    # 转换为度
    tilt_x_deg = np.degrees(tilt_x_rad)
    tilt_y_deg = np.degrees(tilt_y_rad)
    
    return tilt_x_deg, tilt_y_deg
