"""
SequentialOpticalSystem 属性测试

验证：
- Property 3: 光学面位置自动计算
- Property 4: 反射面方向反转

**Validates: Requirements 3.3, 3.4, 2.13, 5.5**
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings

from sequential_system.system import SequentialOpticalSystem, ABCDResult
from sequential_system.source import GaussianBeamSource
from gaussian_beam_simulation.optical_elements import (
    SphericalMirror, ThinLens, ParabolicMirror, FlatMirror
)


class TestSystemCreation:
    """测试系统创建"""
    
    def test_create_basic_system(self):
        """测试创建基本系统"""
        source = GaussianBeamSource(wavelength=0.633, w0=1.0, z0=-50.0)
        system = SequentialOpticalSystem(source)
        
        assert system.source == source
        assert system.grid_size == 512
        assert system.beam_ratio == 0.5
        assert len(system.elements) == 0
        assert len(system.sampling_planes) == 0
    
    def test_create_system_with_custom_params(self):
        """测试使用自定义参数创建系统"""
        source = GaussianBeamSource(wavelength=0.633, w0=1.0)
        system = SequentialOpticalSystem(source, grid_size=256, beam_ratio=0.3)
        
        assert system.grid_size == 256
        assert system.beam_ratio == 0.3


class TestAddSurface:
    """测试添加光学面"""
    
    def test_add_single_surface(self):
        """测试添加单个光学面"""
        source = GaussianBeamSource(wavelength=0.633, w0=1.0)
        system = SequentialOpticalSystem(source)
        
        mirror = SphericalMirror(
            thickness=100.0,
            semi_aperture=15.0,
            radius_of_curvature=200.0,
        )
        
        result = system.add_surface(mirror)
        
        assert result is system  # 链式调用
        assert len(system.elements) == 1
        assert system.elements[0] == mirror
    
    def test_chain_add_surfaces(self):
        """测试链式添加光学面"""
        source = GaussianBeamSource(wavelength=0.633, w0=1.0)
        system = SequentialOpticalSystem(source)
        
        system.add_surface(SphericalMirror(
            thickness=100.0, semi_aperture=15.0, radius_of_curvature=200.0
        )).add_surface(ThinLens(
            thickness=50.0, semi_aperture=10.0, focal_length_value=50.0
        ))
        
        assert len(system.elements) == 2


class TestAddSamplingPlane:
    """测试添加采样面"""
    
    def test_add_sampling_plane(self):
        """测试添加采样面"""
        source = GaussianBeamSource(wavelength=0.633, w0=1.0)
        system = SequentialOpticalSystem(source)
        
        result = system.add_sampling_plane(distance=100.0, name="focus")
        
        assert result is system
        assert len(system.sampling_planes) == 1
        assert system.sampling_planes[0].distance == 100.0
        assert system.sampling_planes[0].name == "focus"


# ============================================================================
# Property 3: 光学面位置自动计算
# ============================================================================

@given(
    thickness1=st.floats(min_value=10.0, max_value=200.0),
    thickness2=st.floats(min_value=10.0, max_value=200.0),
)
@settings(max_examples=50)
def test_surface_position_calculation(thickness1, thickness2):
    """
    **Feature: sequential-optical-system, Property 3: 光学面位置自动计算**
    **Validates: Requirements 3.3, 3.4**
    
    验证光学面位置根据 thickness 自动计算
    """
    source = GaussianBeamSource(wavelength=0.633, w0=1.0)
    system = SequentialOpticalSystem(source)
    
    # 添加两个透镜（非反射）
    lens1 = ThinLens(thickness=thickness1, semi_aperture=10.0, focal_length_value=50.0)
    lens2 = ThinLens(thickness=thickness2, semi_aperture=10.0, focal_length_value=50.0)
    
    system.add_surface(lens1)
    system.add_surface(lens2)
    
    # 验证位置计算
    assert lens1.z_position == 0.0
    assert lens1.path_length == 0.0
    
    np.testing.assert_allclose(lens2.z_position, thickness1, rtol=1e-10)
    np.testing.assert_allclose(lens2.path_length, thickness1, rtol=1e-10)
    
    # 验证总光程
    np.testing.assert_allclose(
        system.total_path_length, thickness1 + thickness2, rtol=1e-10
    )


# ============================================================================
# Property 4: 反射面方向反转
# ============================================================================

@given(
    thickness1=st.floats(min_value=10.0, max_value=200.0),
    thickness2=st.floats(min_value=10.0, max_value=200.0),
)
@settings(max_examples=50)
def test_reflective_surface_direction_reversal(thickness1, thickness2):
    """
    **Feature: sequential-optical-system, Property 4: 反射面方向反转**
    **Validates: Requirements 2.13, 5.5**
    
    验证反射面后传播方向反转
    """
    source = GaussianBeamSource(wavelength=0.633, w0=1.0)
    system = SequentialOpticalSystem(source)
    
    # 添加反射镜
    mirror = SphericalMirror(
        thickness=thickness1, semi_aperture=15.0, radius_of_curvature=200.0
    )
    
    # 添加第二个元件
    lens = ThinLens(
        thickness=thickness2, semi_aperture=10.0, focal_length_value=50.0
    )
    
    system.add_surface(mirror)
    system.add_surface(lens)
    
    # 反射镜在 z=0
    assert mirror.z_position == 0.0
    
    # 反射后方向反转，第二个元件在 -z 方向
    # z_position = 0 + (-1) * thickness1 = -thickness1
    np.testing.assert_allclose(lens.z_position, -thickness1, rtol=1e-10)
    
    # 但光程距离仍然是累加的
    np.testing.assert_allclose(lens.path_length, thickness1, rtol=1e-10)


def test_multiple_reflections():
    """测试多次反射"""
    source = GaussianBeamSource(wavelength=0.633, w0=1.0)
    system = SequentialOpticalSystem(source)
    
    # 添加两个反射镜
    mirror1 = SphericalMirror(
        thickness=100.0, semi_aperture=15.0, radius_of_curvature=200.0
    )
    mirror2 = FlatMirror(thickness=50.0, semi_aperture=15.0)
    lens = ThinLens(thickness=30.0, semi_aperture=10.0, focal_length_value=50.0)
    
    system.add_surface(mirror1)
    system.add_surface(mirror2)
    system.add_surface(lens)
    
    # mirror1 在 z=0
    assert mirror1.z_position == 0.0
    
    # mirror2 在 z=-100（反射后向 -z 传播）
    np.testing.assert_allclose(mirror2.z_position, -100.0, rtol=1e-10)
    
    # lens 在 z=-100+50=-50（第二次反射后向 +z 传播）
    np.testing.assert_allclose(lens.z_position, -50.0, rtol=1e-10)


class TestSummary:
    """测试 summary 方法"""
    
    def test_summary_output(self):
        """测试摘要输出"""
        source = GaussianBeamSource(wavelength=0.633, w0=1.0, z0=-50.0)
        system = SequentialOpticalSystem(source)
        
        system.add_surface(SphericalMirror(
            thickness=100.0, semi_aperture=15.0, radius_of_curvature=200.0
        ))
        system.add_sampling_plane(distance=150.0, name="focus")
        
        summary = system.summary()
        
        assert "序列光学系统配置摘要" in summary
        assert "0.633" in summary
        assert "1.0" in summary
        assert "focus" in summary


class TestABCDResult:
    """测试 ABCD 计算"""
    
    def test_get_abcd_result(self):
        """测试获取 ABCD 结果"""
        source = GaussianBeamSource(wavelength=0.633, w0=1.0, z0=-50.0)
        system = SequentialOpticalSystem(source)
        
        system.add_surface(SphericalMirror(
            thickness=100.0, semi_aperture=15.0, radius_of_curvature=200.0
        ))
        
        result = system.get_abcd_result(distance=50.0)
        
        assert isinstance(result, ABCDResult)
        assert result.distance == 50.0
        assert result.w > 0
