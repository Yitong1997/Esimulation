"""
材质变化检测模块

本模块实现光学系统中材质变化的检测逻辑，用于确定何时需要执行混合元件传播。

核心功能：
1. 检测反射镜面
2. 检测折射界面（空气→玻璃、玻璃→空气）
3. 识别 PARAXIAL 表面（需要单独处理）

**Validates: Requirements 11.1-11.5**
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sequential_system.coordinate_system import GlobalSurfaceDefinition


def detect_material_change(
    current_surface: "GlobalSurfaceDefinition",
    previous_surface: Optional["GlobalSurfaceDefinition"] = None,
) -> bool:
    """检测是否需要混合元件传播
    
    触发条件:
    1. 反射镜面（is_mirror = True）
    2. 材质从空气变为玻璃（入射至透镜前表面）
    3. 材质从玻璃变为空气（从透镜后表面出射）
    
    不触发条件:
    1. 相邻面材质相同（如空气→空气）
    2. PARAXIAL 表面（单独处理）
    3. 坐标断点（虚拟表面）
    
    参数:
        current_surface: 当前表面定义
        previous_surface: 前一表面定义（可选）
    
    返回:
        True 如果需要混合元件传播，False 否则
    
    **Validates: Requirements 11.1, 11.2, 11.3, 11.4, 11.5**
    """
    # PARAXIAL 表面单独处理
    if is_paraxial_surface(current_surface):
        return False
    
    # 坐标断点不触发
    if is_coordinate_break(current_surface):
        return False
    
    # 反射镜始终触发
    if current_surface.is_mirror:
        return True
    
    # 检测材质变化
    curr_material = normalize_material_name(current_surface.material)
    
    if previous_surface is None:
        # 第一个表面：如果不是空气，则触发
        return curr_material != 'air'
    
    prev_material = normalize_material_name(previous_surface.material)
    
    # 材质变化检测
    if prev_material != curr_material:
        # 空气→玻璃 或 玻璃→空气
        if 'air' in [prev_material, curr_material]:
            return True
    
    return False


def is_paraxial_surface(surface: "GlobalSurfaceDefinition") -> bool:
    """检测是否为 PARAXIAL 表面（理想薄透镜）
    
    参数:
        surface: 表面定义
    
    返回:
        True 如果是 PARAXIAL 表面
    
    **Validates: Requirements 11.4**
    """
    return surface.surface_type.lower() == 'paraxial'


def is_coordinate_break(surface: "GlobalSurfaceDefinition") -> bool:
    """检测是否为坐标断点（虚拟表面）
    
    参数:
        surface: 表面定义
    
    返回:
        True 如果是坐标断点
    
    **Validates: Requirements 11.5**
    """
    return surface.surface_type.lower() in ['coordbrk', 'coordinate_break']


def normalize_material_name(material: str) -> str:
    """标准化材料名称
    
    将材料名称转换为小写，并处理特殊情况。
    
    参数:
        material: 原始材料名称
    
    返回:
        标准化后的材料名称
    """
    if material is None:
        return 'air'
    
    material_lower = material.lower().strip()
    
    # 空字符串视为空气
    if not material_lower:
        return 'air'
    
    # 常见的空气表示
    if material_lower in ['air', 'vacuum', '']:
        return 'air'
    
    # 反射镜材料
    if material_lower == 'mirror':
        return 'mirror'
    
    return material_lower


def get_material_refractive_index(material: str, wavelength_um: float) -> float:
    """获取材料的折射率
    
    参数:
        material: 材料名称
        wavelength_um: 波长 (μm)
    
    返回:
        折射率
    
    注意:
        这是一个简化实现，实际应用中应使用材料数据库。
    """
    material_lower = normalize_material_name(material)
    
    if material_lower == 'air':
        return 1.0
    elif material_lower == 'mirror':
        return 1.0  # 反射镜不涉及折射
    elif material_lower in ['bk7', 'n-bk7']:
        # BK7 玻璃的近似折射率（Sellmeier 公式简化）
        return 1.5168
    elif material_lower in ['sf11', 'n-sf11']:
        return 1.7847
    elif material_lower in ['fused_silica', 'silica']:
        return 1.4585
    else:
        # 默认玻璃折射率
        return 1.5


def classify_surface_interaction(
    current_surface: "GlobalSurfaceDefinition",
    previous_surface: Optional["GlobalSurfaceDefinition"] = None,
) -> str:
    """分类表面交互类型
    
    参数:
        current_surface: 当前表面定义
        previous_surface: 前一表面定义（可选）
    
    返回:
        交互类型字符串:
        - 'reflection': 反射
        - 'refraction_enter': 进入介质（空气→玻璃）
        - 'refraction_exit': 离开介质（玻璃→空气）
        - 'paraxial': PARAXIAL 表面
        - 'coordinate_break': 坐标断点
        - 'free_space': 自由空间传播（无材质变化）
    
    **Validates: Requirements 11.1, 11.2, 11.3, 11.4, 11.5**
    """
    # PARAXIAL 表面
    if is_paraxial_surface(current_surface):
        return 'paraxial'
    
    # 坐标断点
    if is_coordinate_break(current_surface):
        return 'coordinate_break'
    
    # 反射镜
    if current_surface.is_mirror:
        return 'reflection'
    
    # 材质变化检测
    curr_material = normalize_material_name(current_surface.material)
    
    if previous_surface is None:
        prev_material = 'air'
    else:
        prev_material = normalize_material_name(previous_surface.material)
    
    # 分类折射类型
    if prev_material == 'air' and curr_material != 'air':
        return 'refraction_enter'
    elif prev_material != 'air' and curr_material == 'air':
        return 'refraction_exit'
    elif prev_material != curr_material:
        # 玻璃→玻璃（不同材料）
        return 'refraction_enter'  # 简化处理
    
    return 'free_space'
