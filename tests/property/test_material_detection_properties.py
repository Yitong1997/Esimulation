"""
材质变化检测属性基测试

使用 hypothesis 库验证材质变化检测的正确性属性。

**Feature: hybrid-optical-propagation**
**Validates: Requirements 11.1, 11.2, 11.3, 11.4, 11.5**
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings
from dataclasses import dataclass, field
from typing import List

import sys
sys.path.insert(0, 'src')

from hybrid_optical_propagation.material_detection import (
    detect_material_change,
    is_paraxial_surface,
    is_coordinate_break,
    normalize_material_name,
    classify_surface_interaction,
)


# ============================================================================
# 模拟 GlobalSurfaceDefinition 用于测试
# ============================================================================

@dataclass
class MockSurfaceDefinition:
    """模拟表面定义，用于测试"""
    index: int = 0
    surface_type: str = 'standard'
    vertex_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    orientation: np.ndarray = field(default_factory=lambda: np.eye(3))
    radius: float = np.inf
    conic: float = 0.0
    is_mirror: bool = False
    material: str = "air"
    asphere_coeffs: List[float] = field(default_factory=list)
    comment: str = ""
    thickness: float = 0.0
    focal_length: float = np.inf


# ============================================================================
# 测试策略定义
# ============================================================================

# 材料名称策略
glass_materials = st.sampled_from(['bk7', 'n-bk7', 'sf11', 'fused_silica', 'glass'])
air_materials = st.sampled_from(['air', 'vacuum', '', 'AIR', 'Air'])
all_materials = st.one_of(glass_materials, air_materials)

# 表面类型策略
surface_types = st.sampled_from(['standard', 'even_asphere', 'flat', 'biconic'])
paraxial_types = st.sampled_from(['paraxial', 'PARAXIAL', 'Paraxial'])
coordbrk_types = st.sampled_from(['coordbrk', 'COORDBRK', 'coordinate_break'])


# ============================================================================
# Property 5: 材质变化检测正确性
# ============================================================================

@settings(max_examples=100)
@given(
    material=all_materials,
    surface_type=surface_types,
)
def test_property_5_mirror_always_triggers(
    material: str,
    surface_type: str,
):
    """
    **Feature: hybrid-optical-propagation, Property 5: 材质变化检测正确性**
    **Validates: Requirements 11.1**
    
    反射镜面始终触发混合元件传播。
    """
    # 创建反射镜表面
    mirror_surface = MockSurfaceDefinition(
        surface_type=surface_type,
        is_mirror=True,
        material=material,
    )
    
    # 创建前一表面（任意材料）
    prev_surface = MockSurfaceDefinition(
        material='air',
    )
    
    # 反射镜应该始终触发
    assert detect_material_change(mirror_surface, prev_surface), (
        f"反射镜应触发混合元件传播，材料={material}"
    )
    
    # 即使没有前一表面也应触发
    assert detect_material_change(mirror_surface, None), (
        "反射镜应触发混合元件传播（无前一表面）"
    )


@settings(max_examples=100)
@given(
    glass_material=glass_materials,
)
def test_property_5_air_to_glass_triggers(
    glass_material: str,
):
    """
    **Feature: hybrid-optical-propagation, Property 5: 材质变化检测正确性**
    **Validates: Requirements 11.2**
    
    空气→玻璃触发混合元件传播。
    """
    # 前一表面：空气
    prev_surface = MockSurfaceDefinition(
        material='air',
    )
    
    # 当前表面：玻璃
    curr_surface = MockSurfaceDefinition(
        material=glass_material,
        is_mirror=False,
    )
    
    assert detect_material_change(curr_surface, prev_surface), (
        f"空气→{glass_material} 应触发混合元件传播"
    )


@settings(max_examples=100)
@given(
    glass_material=glass_materials,
)
def test_property_5_glass_to_air_triggers(
    glass_material: str,
):
    """
    **Feature: hybrid-optical-propagation, Property 5: 材质变化检测正确性**
    **Validates: Requirements 11.3**
    
    玻璃→空气触发混合元件传播。
    """
    # 前一表面：玻璃
    prev_surface = MockSurfaceDefinition(
        material=glass_material,
    )
    
    # 当前表面：空气
    curr_surface = MockSurfaceDefinition(
        material='air',
        is_mirror=False,
    )
    
    assert detect_material_change(curr_surface, prev_surface), (
        f"{glass_material}→空气 应触发混合元件传播"
    )


@settings(max_examples=100)
@given(
    air_material1=air_materials,
    air_material2=air_materials,
)
def test_property_5_same_material_no_trigger(
    air_material1: str,
    air_material2: str,
):
    """
    **Feature: hybrid-optical-propagation, Property 5: 材质变化检测正确性**
    **Validates: Requirements 11.4**
    
    同材质（如空气→空气）不触发混合元件传播。
    """
    # 前一表面：空气
    prev_surface = MockSurfaceDefinition(
        material=air_material1,
    )
    
    # 当前表面：空气（不同表示方式）
    curr_surface = MockSurfaceDefinition(
        material=air_material2,
        is_mirror=False,
    )
    
    assert not detect_material_change(curr_surface, prev_surface), (
        f"空气→空气 不应触发混合元件传播 ({air_material1}→{air_material2})"
    )


@settings(max_examples=50)
@given(
    paraxial_type=paraxial_types,
    material=all_materials,
)
def test_property_5_paraxial_no_trigger(
    paraxial_type: str,
    material: str,
):
    """
    **Feature: hybrid-optical-propagation, Property 5: 材质变化检测正确性**
    **Validates: Requirements 11.4**
    
    PARAXIAL 表面不触发混合元件传播（单独处理）。
    """
    # PARAXIAL 表面
    paraxial_surface = MockSurfaceDefinition(
        surface_type=paraxial_type,
        material=material,
        is_mirror=False,
    )
    
    # 前一表面
    prev_surface = MockSurfaceDefinition(
        material='air',
    )
    
    assert not detect_material_change(paraxial_surface, prev_surface), (
        f"PARAXIAL 表面不应触发混合元件传播 (type={paraxial_type})"
    )


@settings(max_examples=50)
@given(
    coordbrk_type=coordbrk_types,
)
def test_property_5_coordbrk_no_trigger(
    coordbrk_type: str,
):
    """
    **Feature: hybrid-optical-propagation, Property 5: 材质变化检测正确性**
    **Validates: Requirements 11.5**
    
    坐标断点不触发混合元件传播。
    """
    # 坐标断点表面
    coordbrk_surface = MockSurfaceDefinition(
        surface_type=coordbrk_type,
        is_mirror=False,
    )
    
    # 前一表面
    prev_surface = MockSurfaceDefinition(
        material='air',
    )
    
    assert not detect_material_change(coordbrk_surface, prev_surface), (
        f"坐标断点不应触发混合元件传播 (type={coordbrk_type})"
    )


# ============================================================================
# 辅助函数测试
# ============================================================================

@settings(max_examples=100)
@given(
    paraxial_type=paraxial_types,
)
def test_is_paraxial_surface(paraxial_type: str):
    """测试 PARAXIAL 表面识别"""
    surface = MockSurfaceDefinition(surface_type=paraxial_type)
    assert is_paraxial_surface(surface), f"应识别为 PARAXIAL 表面: {paraxial_type}"


@settings(max_examples=100)
@given(
    surface_type=surface_types,
)
def test_is_not_paraxial_surface(surface_type: str):
    """测试非 PARAXIAL 表面识别"""
    surface = MockSurfaceDefinition(surface_type=surface_type)
    assert not is_paraxial_surface(surface), f"不应识别为 PARAXIAL 表面: {surface_type}"


@settings(max_examples=100)
@given(
    coordbrk_type=coordbrk_types,
)
def test_is_coordinate_break(coordbrk_type: str):
    """测试坐标断点识别"""
    surface = MockSurfaceDefinition(surface_type=coordbrk_type)
    assert is_coordinate_break(surface), f"应识别为坐标断点: {coordbrk_type}"


@settings(max_examples=100)
@given(
    air_material=air_materials,
)
def test_normalize_air_materials(air_material: str):
    """测试空气材料标准化"""
    normalized = normalize_material_name(air_material)
    assert normalized == 'air', f"'{air_material}' 应标准化为 'air'，实际为 '{normalized}'"


@settings(max_examples=100)
@given(
    glass_material=glass_materials,
)
def test_normalize_glass_materials(glass_material: str):
    """测试玻璃材料标准化"""
    normalized = normalize_material_name(glass_material)
    assert normalized != 'air', f"'{glass_material}' 不应标准化为 'air'"


# ============================================================================
# 表面交互分类测试
# ============================================================================

def test_classify_reflection():
    """测试反射分类"""
    mirror = MockSurfaceDefinition(is_mirror=True)
    prev = MockSurfaceDefinition(material='air')
    
    assert classify_surface_interaction(mirror, prev) == 'reflection'


def test_classify_refraction_enter():
    """测试进入介质分类"""
    glass_surface = MockSurfaceDefinition(material='bk7', is_mirror=False)
    air_surface = MockSurfaceDefinition(material='air')
    
    assert classify_surface_interaction(glass_surface, air_surface) == 'refraction_enter'


def test_classify_refraction_exit():
    """测试离开介质分类"""
    air_surface = MockSurfaceDefinition(material='air', is_mirror=False)
    glass_surface = MockSurfaceDefinition(material='bk7')
    
    assert classify_surface_interaction(air_surface, glass_surface) == 'refraction_exit'


def test_classify_paraxial():
    """测试 PARAXIAL 分类"""
    paraxial = MockSurfaceDefinition(surface_type='paraxial')
    prev = MockSurfaceDefinition(material='air')
    
    assert classify_surface_interaction(paraxial, prev) == 'paraxial'


def test_classify_coordinate_break():
    """测试坐标断点分类"""
    coordbrk = MockSurfaceDefinition(surface_type='coordbrk')
    prev = MockSurfaceDefinition(material='air')
    
    assert classify_surface_interaction(coordbrk, prev) == 'coordinate_break'


def test_classify_free_space():
    """测试自由空间分类"""
    air1 = MockSurfaceDefinition(material='air', is_mirror=False)
    air2 = MockSurfaceDefinition(material='air')
    
    assert classify_surface_interaction(air1, air2) == 'free_space'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
