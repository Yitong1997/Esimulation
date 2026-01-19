"""
FlatMirror 单元测试

测试平面反射镜的创建和属性。

**Validates: Requirements 2.3**
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pytest

from gaussian_beam_simulation.optical_elements import FlatMirror


class TestFlatMirrorCreation:
    """测试 FlatMirror 创建"""
    
    def test_create_basic_flat_mirror(self):
        """测试创建基本平面镜"""
        mirror = FlatMirror(
            thickness=100.0,
            semi_aperture=15.0,
        )
        
        assert mirror.thickness == 100.0
        assert mirror.semi_aperture == 15.0
        assert mirror.tilt_x == 0.0
        assert mirror.tilt_y == 0.0
    
    def test_create_tilted_flat_mirror(self):
        """测试创建倾斜平面镜"""
        mirror = FlatMirror(
            thickness=100.0,
            semi_aperture=15.0,
            tilt_x=np.pi/4,
            name="fold_mirror",
        )
        
        assert mirror.tilt_x == np.pi/4
        assert mirror.name == "fold_mirror"
    
    def test_create_decentered_flat_mirror(self):
        """测试创建偏心平面镜"""
        mirror = FlatMirror(
            thickness=50.0,
            semi_aperture=10.0,
            decenter_x=2.0,
            decenter_y=1.5,
        )
        
        assert mirror.decenter_x == 2.0
        assert mirror.decenter_y == 1.5


class TestFlatMirrorProperties:
    """测试 FlatMirror 属性"""
    
    def test_focal_length_is_infinity(self):
        """
        **Validates: Requirements 2.3**
        
        验证平面镜焦距为无穷大
        """
        mirror = FlatMirror(thickness=100.0, semi_aperture=15.0)
        
        assert np.isinf(mirror.focal_length)
        assert mirror.focal_length > 0  # 正无穷
    
    def test_radius_of_curvature_is_infinity(self):
        """
        **Validates: Requirements 2.3**
        
        验证平面镜曲率半径为无穷大
        """
        mirror = FlatMirror(thickness=100.0, semi_aperture=15.0)
        
        assert np.isinf(mirror.radius_of_curvature)
    
    def test_is_reflective(self):
        """
        **Validates: Requirements 2.3**
        
        验证平面镜是反射元件
        """
        mirror = FlatMirror(thickness=100.0, semi_aperture=15.0)
        
        assert mirror.is_reflective is True
    
    def test_element_type(self):
        """验证元件类型"""
        mirror = FlatMirror(thickness=100.0, semi_aperture=15.0)
        
        assert mirror.element_type == "flat_mirror"
    
    def test_aperture_diameter(self):
        """验证孔径直径"""
        mirror = FlatMirror(thickness=100.0, semi_aperture=15.0)
        
        assert mirror.aperture_diameter == 30.0


class TestFlatMirrorValidation:
    """测试 FlatMirror 参数验证"""
    
    def test_reject_negative_thickness(self):
        """验证拒绝负厚度"""
        with pytest.raises(ValueError) as exc_info:
            FlatMirror(thickness=-10.0, semi_aperture=15.0)
        
        assert 'thickness' in str(exc_info.value).lower()
    
    def test_reject_zero_semi_aperture(self):
        """验证拒绝零半口径"""
        with pytest.raises(ValueError) as exc_info:
            FlatMirror(thickness=100.0, semi_aperture=0.0)
        
        assert 'semi_aperture' in str(exc_info.value).lower()
    
    def test_reject_negative_semi_aperture(self):
        """验证拒绝负半口径"""
        with pytest.raises(ValueError) as exc_info:
            FlatMirror(thickness=100.0, semi_aperture=-5.0)
        
        assert 'semi_aperture' in str(exc_info.value).lower()


class TestFlatMirrorSurfaceDefinition:
    """测试 FlatMirror 表面定义"""
    
    def test_get_surface_definition(self):
        """验证获取表面定义"""
        mirror = FlatMirror(thickness=100.0, semi_aperture=15.0)
        
        surface_def = mirror.get_surface_definition()
        
        assert surface_def is not None
        assert surface_def.surface_type == 'mirror'
        assert np.isinf(surface_def.radius)
        assert surface_def.semi_aperture == 15.0
        assert surface_def.conic == 0.0
