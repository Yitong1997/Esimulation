"""
ZMX 文件集成模块

本模块提供从 ZMX 文件加载光学系统并创建 HybridOpticalPropagator 的功能。

主要功能：
- 从 ZMX 文件加载光学系统
- 转换为 GlobalSurfaceDefinition 列表
- 创建 HybridOpticalPropagator 实例

**Validates: Requirements 16.5, 16.6, 19.1, 19.2, 19.3**
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING
import numpy as np

from .data_models import SourceDefinition
from .hybrid_propagator import HybridOpticalPropagator, PropagationResult

if TYPE_CHECKING:
    from sequential_system.coordinate_system import GlobalSurfaceDefinition
    from sequential_system.zmx_parser import ZmxDataModel


def load_optical_system_from_zmx(
    filepath: str,
) -> List["GlobalSurfaceDefinition"]:
    """从 ZMX 文件加载光学系统
    
    解析 ZMX 文件并转换为 GlobalSurfaceDefinition 列表。
    
    参数:
        filepath: ZMX 文件路径
    
    返回:
        GlobalSurfaceDefinition 列表
    
    异常:
        FileNotFoundError: 文件不存在
        ZmxParseError: 解析错误
    
    **Validates: Requirements 16.5**
    """
    from sequential_system.zmx_parser import ZmxParser
    from sequential_system.coordinate_system import SurfaceTraversalAlgorithm
    
    # 解析 ZMX 文件
    parser = ZmxParser(filepath)
    zmx_data = parser.parse()
    
    # 遍历表面并转换为全局坐标
    traversal = SurfaceTraversalAlgorithm(zmx_data)
    global_surfaces = traversal.traverse()
    
    return global_surfaces


def create_propagator_from_zmx(
    filepath: str,
    source: SourceDefinition,
    wavelength_um: Optional[float] = None,
    grid_size: int = 512,
    num_rays: int = 200,
    propagation_method: str = "local_raytracing",
) -> HybridOpticalPropagator:
    """从 ZMX 文件创建 HybridOpticalPropagator
    
    参数:
        filepath: ZMX 文件路径
        source: 入射波面定义
        wavelength_um: 波长 (μm)，如果为 None 则使用 ZMX 文件中的主波长
        grid_size: 网格大小
        num_rays: 光线采样数量
        propagation_method: 传播方法
    
    返回:
        HybridOpticalPropagator 实例
    
    **Validates: Requirements 16.5, 16.6**
    """
    from sequential_system.zmx_parser import ZmxParser
    
    # 加载光学系统
    optical_system = load_optical_system_from_zmx(filepath)
    
    # 如果未指定波长，从 ZMX 文件获取
    if wavelength_um is None:
        parser = ZmxParser(filepath)
        zmx_data = parser.parse()
        if zmx_data.wavelengths:
            wavelength_um = zmx_data.wavelengths[0]
        else:
            wavelength_um = source.wavelength_um
    
    # 创建传播器
    propagator = HybridOpticalPropagator(
        optical_system=optical_system,
        source=source,
        wavelength_um=wavelength_um,
        grid_size=grid_size,
        num_rays=num_rays,
        propagation_method=propagation_method,
    )
    
    return propagator


def get_zmx_system_info(filepath: str) -> dict:
    """获取 ZMX 文件的系统信息
    
    参数:
        filepath: ZMX 文件路径
    
    返回:
        包含系统信息的字典
    """
    from sequential_system.zmx_parser import ZmxParser
    from sequential_system.coordinate_system import SurfaceTraversalAlgorithm
    
    # 解析 ZMX 文件
    parser = ZmxParser(filepath)
    zmx_data = parser.parse()
    
    # 遍历表面
    traversal = SurfaceTraversalAlgorithm(zmx_data)
    global_surfaces = traversal.traverse()
    
    # 统计信息
    num_surfaces = len(global_surfaces)
    num_mirrors = sum(1 for s in global_surfaces if s.is_mirror)
    num_paraxial = sum(1 for s in global_surfaces if s.surface_type == 'paraxial')
    
    # 表面类型统计
    surface_types = {}
    for s in global_surfaces:
        t = s.surface_type
        surface_types[t] = surface_types.get(t, 0) + 1
    
    return {
        'filepath': filepath,
        'num_surfaces': num_surfaces,
        'num_mirrors': num_mirrors,
        'num_paraxial': num_paraxial,
        'surface_types': surface_types,
        'wavelengths': zmx_data.wavelengths,
        'entrance_pupil_diameter': zmx_data.entrance_pupil_diameter,
    }


class ZmxOpticalSystem:
    """ZMX 光学系统封装类
    
    封装从 ZMX 文件加载的光学系统，提供便捷的访问接口。
    
    属性:
        filepath: ZMX 文件路径
        surfaces: GlobalSurfaceDefinition 列表
        wavelengths: 波长列表 (μm)
        entrance_pupil_diameter: 入瞳直径 (mm)
    
    **Validates: Requirements 16.6**
    """
    
    def __init__(self, filepath: str):
        """从 ZMX 文件初始化
        
        参数:
            filepath: ZMX 文件路径
        """
        from sequential_system.zmx_parser import ZmxParser
        from sequential_system.coordinate_system import SurfaceTraversalAlgorithm
        
        self._filepath = filepath
        
        # 解析 ZMX 文件
        parser = ZmxParser(filepath)
        self._zmx_data = parser.parse()
        
        # 遍历表面
        traversal = SurfaceTraversalAlgorithm(self._zmx_data)
        self._surfaces = traversal.traverse()
    
    @property
    def filepath(self) -> str:
        """ZMX 文件路径"""
        return self._filepath
    
    @property
    def surfaces(self) -> List["GlobalSurfaceDefinition"]:
        """GlobalSurfaceDefinition 列表"""
        return self._surfaces
    
    @property
    def wavelengths(self) -> List[float]:
        """波长列表 (μm)"""
        return self._zmx_data.wavelengths
    
    @property
    def primary_wavelength(self) -> float:
        """主波长 (μm)"""
        if self._zmx_data.wavelengths:
            return self._zmx_data.wavelengths[0]
        return 0.55  # 默认值
    
    @property
    def entrance_pupil_diameter(self) -> float:
        """入瞳直径 (mm)"""
        return self._zmx_data.entrance_pupil_diameter
    
    @property
    def num_surfaces(self) -> int:
        """表面数量"""
        return len(self._surfaces)
    
    def get_mirrors(self) -> List["GlobalSurfaceDefinition"]:
        """获取所有反射镜"""
        return [s for s in self._surfaces if s.is_mirror]
    
    def get_paraxial_surfaces(self) -> List["GlobalSurfaceDefinition"]:
        """获取所有 PARAXIAL 表面"""
        return [s for s in self._surfaces if s.surface_type == 'paraxial']
    
    def create_propagator(
        self,
        source: SourceDefinition,
        wavelength_um: Optional[float] = None,
        grid_size: int = 512,
        num_rays: int = 200,
        propagation_method: str = "local_raytracing",
    ) -> HybridOpticalPropagator:
        """创建 HybridOpticalPropagator
        
        参数:
            source: 入射波面定义
            wavelength_um: 波长 (μm)，如果为 None 则使用主波长
            grid_size: 网格大小
            num_rays: 光线采样数量
            propagation_method: 传播方法
        
        返回:
            HybridOpticalPropagator 实例
        """
        if wavelength_um is None:
            wavelength_um = self.primary_wavelength
        
        return HybridOpticalPropagator(
            optical_system=self._surfaces,
            source=source,
            wavelength_um=wavelength_um,
            grid_size=grid_size,
            num_rays=num_rays,
            propagation_method=propagation_method,
        )
    
    def __repr__(self) -> str:
        return (
            f"ZmxOpticalSystem('{self._filepath}', "
            f"surfaces={self.num_surfaces}, "
            f"wavelength={self.primary_wavelength}μm)"
        )

