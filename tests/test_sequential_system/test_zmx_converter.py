"""
ZMX 元件转换器单元测试

测试 ElementConverter 类的元件转换功能，包括：
- 反射镜类型分类
- 折叠镜检测
- 坐标变换累积
- 厚度计算

**Validates: Requirements 5.1-5.8, 6.1-6.5, 7.1-7.5**
"""

import pytest
import numpy as np
import os
import sys

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from sequential_system.zmx_converter import (
    CoordinateTransform,
    ConvertedElement,
    ElementConverter,
)
from sequential_system.zmx_parser import (
    ZmxParser,
    ZmxDataModel,
    ZmxSurfaceData,
)
from gaussian_beam_simulation.optical_elements import (
    FlatMirror,
    ParabolicMirror,
    SphericalMirror,
)


# =============================================================================
# CoordinateTransform 测试
# =============================================================================

class TestCoordinateTransform:
    """测试 CoordinateTransform 坐标变换累积器
    
    **Validates: Requirements 3.4, 6.2**
    """
    
    def test_default_values(self):
        """测试默认值为零"""
        transform = CoordinateTransform()
        
        assert transform.decenter_x == 0.0
        assert transform.decenter_y == 0.0
        assert transform.decenter_z == 0.0
        assert transform.tilt_x_rad == 0.0
        assert transform.tilt_y_rad == 0.0
        assert transform.tilt_z_rad == 0.0

    def test_apply_coordinate_break_decenter(self):
        """测试应用坐标断点的偏心累积
        
        **Validates: Requirements 3.4**
        """
        transform = CoordinateTransform()
        
        # 应用第一个坐标断点
        transform.apply_coordinate_break(
            dx=5.0, dy=3.0, dz=10.0,
            rx_deg=0, ry_deg=0, rz_deg=0
        )
        
        assert transform.decenter_x == 5.0
        assert transform.decenter_y == 3.0
        assert transform.decenter_z == 10.0
        
        # 应用第二个坐标断点，累积偏心
        transform.apply_coordinate_break(
            dx=2.0, dy=1.0, dz=5.0,
            rx_deg=0, ry_deg=0, rz_deg=0
        )
        
        assert transform.decenter_x == 7.0
        assert transform.decenter_y == 4.0
        assert transform.decenter_z == 15.0
    
    def test_apply_coordinate_break_tilt(self):
        """测试应用坐标断点的倾斜累积（度转弧度）
        
        **Validates: Requirements 3.4, 6.2**
        """
        transform = CoordinateTransform()
        
        # 应用 45 度倾斜
        transform.apply_coordinate_break(
            dx=0, dy=0, dz=0,
            rx_deg=45.0, ry_deg=0, rz_deg=0
        )
        
        assert transform.tilt_x_rad == pytest.approx(np.pi / 4, rel=1e-6)
        assert transform.tilt_y_rad == 0.0
        assert transform.tilt_z_rad == 0.0
        
        # 再应用 45 度倾斜，累积为 90 度
        transform.apply_coordinate_break(
            dx=0, dy=0, dz=0,
            rx_deg=45.0, ry_deg=0, rz_deg=0
        )
        
        assert transform.tilt_x_rad == pytest.approx(np.pi / 2, rel=1e-6)
    
    def test_apply_coordinate_break_multiple_axes(self):
        """测试多轴同时倾斜
        
        **Validates: Requirements 3.4**
        """
        transform = CoordinateTransform()
        
        transform.apply_coordinate_break(
            dx=1.0, dy=2.0, dz=3.0,
            rx_deg=30.0, ry_deg=45.0, rz_deg=60.0
        )
        
        assert transform.decenter_x == 1.0
        assert transform.decenter_y == 2.0
        assert transform.decenter_z == 3.0
        assert transform.tilt_x_rad == pytest.approx(np.deg2rad(30.0), rel=1e-6)
        assert transform.tilt_y_rad == pytest.approx(np.deg2rad(45.0), rel=1e-6)
        assert transform.tilt_z_rad == pytest.approx(np.deg2rad(60.0), rel=1e-6)
    
    def test_reset(self):
        """测试重置方法
        
        **Validates: Requirements 3.4**
        """
        transform = CoordinateTransform(
            decenter_x=5.0,
            decenter_y=3.0,
            decenter_z=10.0,
            tilt_x_rad=np.pi / 4,
            tilt_y_rad=np.pi / 6,
            tilt_z_rad=np.pi / 3,
        )
        
        # 重置
        transform.reset()
        
        assert transform.decenter_x == 0.0
        assert transform.decenter_y == 0.0
        assert transform.decenter_z == 0.0
        assert transform.tilt_x_rad == 0.0
        assert transform.tilt_y_rad == 0.0
        assert transform.tilt_z_rad == 0.0

    def test_copy(self):
        """测试复制方法"""
        original = CoordinateTransform(
            decenter_x=5.0,
            tilt_x_rad=np.pi / 4,
        )
        
        copy = original.copy()
        
        # 验证值相同
        assert copy.decenter_x == original.decenter_x
        assert copy.tilt_x_rad == original.tilt_x_rad
        
        # 验证是独立对象
        copy.decenter_x = 10.0
        assert original.decenter_x == 5.0
    
    def test_has_decenter_property(self):
        """测试 has_decenter 属性"""
        transform = CoordinateTransform()
        assert transform.has_decenter == False
        
        transform.decenter_x = 1.0
        assert transform.has_decenter == True
        
        transform.decenter_x = 0.0
        transform.decenter_y = 2.0
        assert transform.has_decenter == True
    
    def test_has_tilt_property(self):
        """测试 has_tilt 属性"""
        transform = CoordinateTransform()
        assert transform.has_tilt == False
        
        transform.tilt_x_rad = 0.1
        assert transform.has_tilt == True
        
        transform.tilt_x_rad = 0.0
        transform.tilt_y_rad = 0.2
        assert transform.has_tilt == True
    
    def test_tilt_deg_properties(self):
        """测试倾斜角度（度）属性"""
        transform = CoordinateTransform(
            tilt_x_rad=np.pi / 4,
            tilt_y_rad=np.pi / 6,
            tilt_z_rad=np.pi / 3,
        )
        
        assert transform.tilt_x_deg == pytest.approx(45.0, rel=1e-6)
        assert transform.tilt_y_deg == pytest.approx(30.0, rel=1e-6)
        assert transform.tilt_z_deg == pytest.approx(60.0, rel=1e-6)
    
    def test_repr(self):
        """测试字符串表示"""
        # 空变换
        transform = CoordinateTransform()
        assert "identity" in repr(transform)
        
        # 有偏心
        transform = CoordinateTransform(decenter_x=5.0)
        assert "decenter" in repr(transform)
        
        # 有倾斜
        transform = CoordinateTransform(tilt_x_rad=np.pi / 4)
        assert "tilt" in repr(transform)


# =============================================================================
# ConvertedElement 测试
# =============================================================================

class TestConvertedElement:
    """测试 ConvertedElement 转换后元件数据类
    
    **Validates: Requirements 5.7, 9.4, 9.5**
    """
    
    def test_basic_creation(self):
        """测试基本创建"""
        mirror = FlatMirror(thickness=100.0, semi_aperture=25.0)
        
        converted = ConvertedElement(
            element=mirror,
            zmx_surface_index=5,
            zmx_comment="M1",
            is_fold_mirror=True,
            fold_angle_deg=45.0,
        )
        
        assert converted.element is mirror
        assert converted.zmx_surface_index == 5
        assert converted.zmx_comment == "M1"
        assert converted.is_fold_mirror == True
        assert converted.fold_angle_deg == 45.0

    def test_element_type_property(self):
        """测试 element_type 属性"""
        flat_mirror = FlatMirror(thickness=100.0, semi_aperture=25.0)
        converted = ConvertedElement(
            element=flat_mirror,
            zmx_surface_index=1,
        )
        assert converted.element_type == "FlatMirror"
        
        parabolic_mirror = ParabolicMirror(
            thickness=100.0,
            semi_aperture=25.0,
            parent_focal_length=200.0,
        )
        converted = ConvertedElement(
            element=parabolic_mirror,
            zmx_surface_index=2,
        )
        assert converted.element_type == "ParabolicMirror"
        
        spherical_mirror = SphericalMirror(
            thickness=100.0,
            semi_aperture=25.0,
            radius_of_curvature=500.0,
        )
        converted = ConvertedElement(
            element=spherical_mirror,
            zmx_surface_index=3,
        )
        assert converted.element_type == "SphericalMirror"
    
    def test_element_type_with_none(self):
        """测试 element 为 None 时的 element_type"""
        converted = ConvertedElement(
            element=None,
            zmx_surface_index=1,
        )
        assert converted.element_type == "Unknown"
    
    def test_has_comment_property(self):
        """测试 has_comment 属性"""
        # 无注释
        converted = ConvertedElement(
            element=None,
            zmx_surface_index=1,
            zmx_comment="",
        )
        assert converted.has_comment == False
        
        # 空白注释
        converted = ConvertedElement(
            element=None,
            zmx_surface_index=1,
            zmx_comment="   ",
        )
        assert converted.has_comment == False
        
        # 有注释
        converted = ConvertedElement(
            element=None,
            zmx_surface_index=1,
            zmx_comment="M1",
        )
        assert converted.has_comment == True
    
    def test_get_code_comment(self):
        """测试代码注释生成
        
        **Validates: Requirements 9.4, 9.5**
        """
        # 基本注释
        converted = ConvertedElement(
            element=None,
            zmx_surface_index=5,
        )
        comment = converted.get_code_comment()
        assert "ZMX Surface 5" in comment
        
        # 带原始注释
        converted = ConvertedElement(
            element=None,
            zmx_surface_index=5,
            zmx_comment="M1",
        )
        comment = converted.get_code_comment()
        assert "ZMX Surface 5" in comment
        assert "M1" in comment
        
        # 折叠镜注释
        converted = ConvertedElement(
            element=None,
            zmx_surface_index=5,
            zmx_comment="M1",
            is_fold_mirror=True,
            fold_angle_deg=45.0,
        )
        comment = converted.get_code_comment()
        assert "ZMX Surface 5" in comment
        assert "M1" in comment
        assert "Fold Mirror" in comment
        assert "45.0" in comment
    
    def test_repr(self):
        """测试字符串表示"""
        mirror = FlatMirror(thickness=100.0, semi_aperture=25.0)
        converted = ConvertedElement(
            element=mirror,
            zmx_surface_index=5,
            zmx_comment="M1",
            is_fold_mirror=True,
            fold_angle_deg=45.0,
        )
        
        repr_str = repr(converted)
        assert "FlatMirror" in repr_str
        assert "5" in repr_str
        assert "M1" in repr_str
        assert "45.0" in repr_str


# =============================================================================
# ElementConverter 反射镜类型分类测试
# =============================================================================

class TestElementConverterMirrorTypeClassification:
    """测试 ElementConverter 的反射镜类型分类
    
    **Validates: Requirements 5.1, 5.2, 5.3**
    """
    
    @pytest.fixture
    def basic_data_model(self):
        """创建基本的数据模型"""
        model = ZmxDataModel()
        # 物面
        model.surfaces[0] = ZmxSurfaceData(
            index=0,
            surface_type='standard',
            thickness=np.inf,
        )
        return model
    
    def test_flat_mirror_creation_infinite_radius(self, basic_data_model):
        """测试 radius=inf 创建 FlatMirror
        
        **Validates: Requirements 5.3**
        """
        model = basic_data_model
        
        # 添加平面反射镜（radius=inf）
        model.surfaces[1] = ZmxSurfaceData(
            index=1,
            surface_type='standard',
            radius=np.inf,
            thickness=0.0,
            is_mirror=True,
            semi_diameter=25.0,
            comment="Flat Mirror",
        )
        # 像面
        model.surfaces[2] = ZmxSurfaceData(
            index=2,
            surface_type='standard',
            thickness=0.0,
        )
        
        converter = ElementConverter(model)
        elements = converter.convert()
        
        assert len(elements) == 1
        assert isinstance(elements[0], FlatMirror)
    
    def test_parabolic_mirror_creation_conic_minus_one(self, basic_data_model):
        """测试 conic=-1 创建 ParabolicMirror
        
        **Validates: Requirements 5.2**
        """
        model = basic_data_model
        
        # 添加抛物面反射镜（conic=-1）
        model.surfaces[1] = ZmxSurfaceData(
            index=1,
            surface_type='standard',
            radius=200.0,  # 有限曲率半径
            conic=-1.0,    # 抛物面
            thickness=0.0,
            is_mirror=True,
            semi_diameter=25.0,
            comment="Parabolic Mirror",
        )
        # 像面
        model.surfaces[2] = ZmxSurfaceData(
            index=2,
            surface_type='standard',
            thickness=0.0,
        )
        
        converter = ElementConverter(model)
        elements = converter.convert()
        
        assert len(elements) == 1
        assert isinstance(elements[0], ParabolicMirror)
        # 验证焦距 = R/2
        assert elements[0].parent_focal_length == pytest.approx(100.0, rel=1e-6)
    
    def test_spherical_mirror_creation_finite_radius(self, basic_data_model):
        """测试有限 radius 且 conic!=−1 创建 SphericalMirror
        
        **Validates: Requirements 5.1**
        """
        model = basic_data_model
        
        # 添加球面反射镜
        model.surfaces[1] = ZmxSurfaceData(
            index=1,
            surface_type='standard',
            radius=300.0,  # 有限曲率半径
            conic=0.0,     # 球面（conic=0）
            thickness=0.0,
            is_mirror=True,
            semi_diameter=25.0,
            comment="Spherical Mirror",
        )
        # 像面
        model.surfaces[2] = ZmxSurfaceData(
            index=2,
            surface_type='standard',
            thickness=0.0,
        )
        
        converter = ElementConverter(model)
        elements = converter.convert()
        
        assert len(elements) == 1
        assert isinstance(elements[0], SphericalMirror)
        assert elements[0].radius_of_curvature == 300.0

    def test_spherical_mirror_with_nonzero_conic(self, basic_data_model):
        """测试 conic 不为 -1 时创建 SphericalMirror
        
        **Validates: Requirements 5.1**
        """
        model = basic_data_model
        
        # 添加椭球面反射镜（conic=-0.5）
        model.surfaces[1] = ZmxSurfaceData(
            index=1,
            surface_type='standard',
            radius=300.0,
            conic=-0.5,  # 椭球面，不是抛物面
            thickness=0.0,
            is_mirror=True,
            semi_diameter=25.0,
        )
        # 像面
        model.surfaces[2] = ZmxSurfaceData(
            index=2,
            surface_type='standard',
            thickness=0.0,
        )
        
        converter = ElementConverter(model)
        elements = converter.convert()
        
        assert len(elements) == 1
        # 非 -1 的 conic 应该创建 SphericalMirror（简化处理）
        assert isinstance(elements[0], SphericalMirror)
    
    def test_parabolic_mirror_conic_tolerance(self, basic_data_model):
        """测试 conic 接近 -1 时的容差判断
        
        **Validates: Requirements 5.2**
        """
        model = basic_data_model
        
        # 添加 conic 非常接近 -1 的反射镜
        model.surfaces[1] = ZmxSurfaceData(
            index=1,
            surface_type='standard',
            radius=200.0,
            conic=-1.0000001,  # 非常接近 -1
            thickness=0.0,
            is_mirror=True,
            semi_diameter=25.0,
        )
        # 像面
        model.surfaces[2] = ZmxSurfaceData(
            index=2,
            surface_type='standard',
            thickness=0.0,
        )
        
        converter = ElementConverter(model)
        elements = converter.convert()
        
        assert len(elements) == 1
        # 应该识别为抛物面镜
        assert isinstance(elements[0], ParabolicMirror)


# =============================================================================
# ElementConverter 折叠镜检测测试
# =============================================================================

class TestElementConverterFoldMirrorDetection:
    """测试 ElementConverter 的折叠镜检测
    
    **Validates: Requirements 5.7, 5.8, 7.3**
    """
    
    def test_fold_angle_threshold_constant(self):
        """测试折叠镜角度阈值常量"""
        assert ElementConverter.FOLD_ANGLE_THRESHOLD == 5.0
    
    def test_is_fold_mirror_above_threshold(self):
        """测试倾斜角度 >= 5° 识别为折叠镜
        
        **Validates: Requirements 5.7**
        """
        model = ZmxDataModel()
        converter = ElementConverter(model)
        
        # 45 度倾斜 - 应该是折叠镜
        assert converter._is_fold_mirror(45.0, 0.0) == True
        assert converter._is_fold_mirror(0.0, 45.0) == True
        
        # 10 度倾斜 - 应该是折叠镜
        assert converter._is_fold_mirror(10.0, 0.0) == True
        
        # 5 度倾斜 - 边界值，应该是折叠镜
        assert converter._is_fold_mirror(5.0, 0.0) == True
        assert converter._is_fold_mirror(0.0, 5.0) == True

    def test_is_fold_mirror_below_threshold(self):
        """测试倾斜角度 < 5° 识别为失调
        
        **Validates: Requirements 5.8**
        """
        model = ZmxDataModel()
        converter = ElementConverter(model)
        
        # 4.9 度倾斜 - 应该是失调
        assert converter._is_fold_mirror(4.9, 0.0) == False
        assert converter._is_fold_mirror(0.0, 4.9) == False
        
        # 2 度倾斜 - 应该是失调
        assert converter._is_fold_mirror(2.0, 0.0) == False
        
        # 0 度倾斜 - 应该是失调
        assert converter._is_fold_mirror(0.0, 0.0) == False
    
    def test_is_fold_mirror_boundary_values(self):
        """测试边界值 (4.9°, 5.0°, 5.1°)
        
        **Validates: Requirements 5.7, 5.8**
        """
        model = ZmxDataModel()
        converter = ElementConverter(model)
        
        # 4.9° - 低于阈值，失调
        assert converter._is_fold_mirror(4.9, 0.0) == False
        
        # 5.0° - 等于阈值，折叠镜
        assert converter._is_fold_mirror(5.0, 0.0) == True
        
        # 5.1° - 高于阈值，折叠镜
        assert converter._is_fold_mirror(5.1, 0.0) == True
    
    def test_is_fold_mirror_negative_angles(self):
        """测试负角度的折叠镜检测
        
        **Validates: Requirements 5.7, 5.8**
        """
        model = ZmxDataModel()
        converter = ElementConverter(model)
        
        # 负 45 度倾斜 - 应该是折叠镜
        assert converter._is_fold_mirror(-45.0, 0.0) == True
        assert converter._is_fold_mirror(0.0, -45.0) == True
        
        # 负 4.9 度倾斜 - 应该是失调
        assert converter._is_fold_mirror(-4.9, 0.0) == False
    
    def test_fold_mirror_with_coordbrk(self):
        """测试带坐标断点的折叠镜检测
        
        **Validates: Requirements 5.5, 5.7, 7.3**
        """
        model = ZmxDataModel()
        
        # 物面
        model.surfaces[0] = ZmxSurfaceData(
            index=0,
            surface_type='standard',
            thickness=np.inf,
        )
        # 坐标断点（45度倾斜）
        model.surfaces[1] = ZmxSurfaceData(
            index=1,
            surface_type='coordinate_break',
            tilt_x_deg=45.0,
            thickness=0.0,
        )
        # 平面反射镜
        model.surfaces[2] = ZmxSurfaceData(
            index=2,
            surface_type='standard',
            radius=np.inf,
            thickness=0.0,
            is_mirror=True,
            semi_diameter=25.0,
            comment="Fold Mirror",
        )
        # 坐标断点（恢复）
        model.surfaces[3] = ZmxSurfaceData(
            index=3,
            surface_type='coordinate_break',
            tilt_x_deg=45.0,
            thickness=-100.0,  # 负厚度表示反射方向传播
        )
        # 像面
        model.surfaces[4] = ZmxSurfaceData(
            index=4,
            surface_type='standard',
            thickness=0.0,
        )
        
        converter = ElementConverter(model)
        elements = converter.convert()
        
        assert len(elements) == 1
        assert isinstance(elements[0], FlatMirror)
        assert elements[0].is_fold == True
        
        # 验证转换元件的元数据
        converted = converter.get_converted_elements()
        assert len(converted) == 1
        assert converted[0].is_fold_mirror == True
        assert converted[0].fold_angle_deg == pytest.approx(45.0, rel=1e-6)

    def test_misaligned_mirror_small_tilt(self):
        """测试小角度倾斜的失调反射镜
        
        **Validates: Requirements 5.8**
        """
        model = ZmxDataModel()
        
        # 物面
        model.surfaces[0] = ZmxSurfaceData(
            index=0,
            surface_type='standard',
            thickness=np.inf,
        )
        # 坐标断点（2度倾斜 - 失调）
        model.surfaces[1] = ZmxSurfaceData(
            index=1,
            surface_type='coordinate_break',
            tilt_x_deg=2.0,
            thickness=0.0,
        )
        # 平面反射镜
        model.surfaces[2] = ZmxSurfaceData(
            index=2,
            surface_type='standard',
            radius=np.inf,
            thickness=100.0,
            is_mirror=True,
            semi_diameter=25.0,
        )
        # 像面
        model.surfaces[3] = ZmxSurfaceData(
            index=3,
            surface_type='standard',
            thickness=0.0,
        )
        
        converter = ElementConverter(model)
        elements = converter.convert()
        
        assert len(elements) == 1
        assert isinstance(elements[0], FlatMirror)
        assert elements[0].is_fold == False  # 失调，不是折叠镜
        
        # 验证转换元件的元数据
        converted = converter.get_converted_elements()
        assert len(converted) == 1
        assert converted[0].is_fold_mirror == False


# =============================================================================
# ElementConverter 厚度计算测试
# =============================================================================

class TestElementConverterThicknessCalculation:
    """测试 ElementConverter 的厚度计算
    
    **Validates: Requirements 6.1, 6.3, 6.4, 7.2**
    """
    
    def test_thickness_from_negative_coordbrk(self):
        """测试从后续坐标断点获取负厚度
        
        **Validates: Requirements 6.3**
        """
        model = ZmxDataModel()
        
        # 物面
        model.surfaces[0] = ZmxSurfaceData(
            index=0,
            surface_type='standard',
            thickness=np.inf,
        )
        # 坐标断点（45度倾斜）
        model.surfaces[1] = ZmxSurfaceData(
            index=1,
            surface_type='coordinate_break',
            tilt_x_deg=45.0,
            thickness=0.0,
        )
        # 平面反射镜（thickness=0）
        model.surfaces[2] = ZmxSurfaceData(
            index=2,
            surface_type='standard',
            radius=np.inf,
            thickness=0.0,
            is_mirror=True,
            semi_diameter=25.0,
        )
        # 坐标断点（负厚度 -50 表示反射方向传播 50mm）
        model.surfaces[3] = ZmxSurfaceData(
            index=3,
            surface_type='coordinate_break',
            tilt_x_deg=45.0,
            thickness=-50.0,
        )
        # 像面
        model.surfaces[4] = ZmxSurfaceData(
            index=4,
            surface_type='standard',
            thickness=0.0,
        )
        
        converter = ElementConverter(model)
        elements = converter.convert()
        
        assert len(elements) == 1
        # 厚度应该是 50（负厚度的绝对值）
        assert elements[0].thickness == pytest.approx(50.0, rel=1e-6)

    def test_thickness_from_mirror_surface(self):
        """测试从反射镜表面获取厚度
        
        **Validates: Requirements 7.2**
        """
        model = ZmxDataModel()
        
        # 物面
        model.surfaces[0] = ZmxSurfaceData(
            index=0,
            surface_type='standard',
            thickness=np.inf,
        )
        # 平面反射镜（thickness=100）
        model.surfaces[1] = ZmxSurfaceData(
            index=1,
            surface_type='standard',
            radius=np.inf,
            thickness=100.0,
            is_mirror=True,
            semi_diameter=25.0,
        )
        # 像面
        model.surfaces[2] = ZmxSurfaceData(
            index=2,
            surface_type='standard',
            thickness=0.0,
        )
        
        converter = ElementConverter(model)
        elements = converter.convert()
        
        assert len(elements) == 1
        assert elements[0].thickness == pytest.approx(100.0, rel=1e-6)
    
    def test_thickness_zero_search_subsequent(self):
        """测试厚度为 0 时查找后续表面
        
        **Validates: Requirements 6.4**
        """
        model = ZmxDataModel()
        
        # 物面
        model.surfaces[0] = ZmxSurfaceData(
            index=0,
            surface_type='standard',
            thickness=np.inf,
        )
        # 坐标断点（45度倾斜）
        model.surfaces[1] = ZmxSurfaceData(
            index=1,
            surface_type='coordinate_break',
            tilt_x_deg=45.0,
            thickness=0.0,
        )
        # 平面反射镜（thickness=0）
        model.surfaces[2] = ZmxSurfaceData(
            index=2,
            surface_type='standard',
            radius=np.inf,
            thickness=0.0,
            is_mirror=True,
            semi_diameter=25.0,
        )
        # 坐标断点（thickness=0）
        model.surfaces[3] = ZmxSurfaceData(
            index=3,
            surface_type='coordinate_break',
            tilt_x_deg=45.0,
            thickness=0.0,
        )
        # 另一个坐标断点（thickness=-75）
        model.surfaces[4] = ZmxSurfaceData(
            index=4,
            surface_type='coordinate_break',
            thickness=-75.0,
        )
        # 像面
        model.surfaces[5] = ZmxSurfaceData(
            index=5,
            surface_type='standard',
            thickness=0.0,
        )
        
        converter = ElementConverter(model)
        elements = converter.convert()
        
        assert len(elements) == 1
        # 应该找到后续坐标断点的厚度 75
        assert elements[0].thickness == pytest.approx(75.0, rel=1e-6)
    
    def test_calculate_thickness_after_reflection_method(self):
        """测试 _calculate_thickness_after_reflection 方法
        
        **Validates: Requirements 6.1, 6.3, 6.4**
        """
        model = ZmxDataModel()
        
        # 物面
        model.surfaces[0] = ZmxSurfaceData(
            index=0,
            surface_type='standard',
            thickness=np.inf,
        )
        # 反射镜（thickness=0）
        model.surfaces[1] = ZmxSurfaceData(
            index=1,
            surface_type='standard',
            radius=np.inf,
            thickness=0.0,
            is_mirror=True,
        )
        # 坐标断点（thickness=-30）
        model.surfaces[2] = ZmxSurfaceData(
            index=2,
            surface_type='coordinate_break',
            thickness=-30.0,
        )
        # 像面
        model.surfaces[3] = ZmxSurfaceData(
            index=3,
            surface_type='standard',
            thickness=0.0,
        )
        
        converter = ElementConverter(model)
        
        # 直接测试方法
        thickness = converter._calculate_thickness_after_reflection(1)
        assert thickness == pytest.approx(30.0, rel=1e-6)


# =============================================================================
# ElementConverter 集成测试
# =============================================================================

class TestElementConverterIntegration:
    """ElementConverter 集成测试
    
    使用真实的 ZMX 测试文件进行测试
    
    **Validates: Requirements 10.1-10.6**
    """
    
    @pytest.fixture
    def zmx_test_dir(self):
        """ZMX 测试文件目录"""
        return "optiland-master/tests/zemax_files"
    
    def test_one_mirror_up_45deg(self, zmx_test_dir):
        """测试 one_mirror_up_45deg.zmx - 单个 45 度折叠镜
        
        **Validates: Requirements 10.4**
        """
        zmx_path = os.path.join(zmx_test_dir, "one_mirror_up_45deg.zmx")
        
        if not os.path.exists(zmx_path):
            pytest.skip(f"测试文件不存在: {zmx_path}")
        
        # 解析 ZMX 文件
        parser = ZmxParser(zmx_path)
        data_model = parser.parse()
        
        # 转换为元件
        converter = ElementConverter(data_model)
        elements = converter.convert()
        
        # 验证生成了元件
        assert len(elements) >= 1
        
        # 查找折叠镜
        converted = converter.get_converted_elements()
        fold_mirrors = [ce for ce in converted if ce.is_fold_mirror]
        
        # 应该有至少一个折叠镜
        assert len(fold_mirrors) >= 1
        
        # 验证折叠角度接近 45 度
        for fm in fold_mirrors:
            assert fm.fold_angle_deg == pytest.approx(45.0, abs=1.0)
    
    def test_complicated_fold_mirrors_setup_v2(self, zmx_test_dir):
        """测试 complicated_fold_mirrors_setup_v2.zmx - 复杂折叠光路
        
        **Validates: Requirements 10.1, 10.2, 10.3, 10.6**
        """
        zmx_path = os.path.join(zmx_test_dir, "complicated_fold_mirrors_setup_v2.zmx")
        
        if not os.path.exists(zmx_path):
            pytest.skip(f"测试文件不存在: {zmx_path}")
        
        # 解析 ZMX 文件
        parser = ZmxParser(zmx_path)
        data_model = parser.parse()
        
        # 验证解析了反射镜表面
        mirrors = data_model.get_mirror_surfaces()
        assert len(mirrors) > 0, "应该识别到反射镜表面"
        
        # 验证解析了坐标断点
        coord_breaks = data_model.get_coordinate_break_surfaces()
        assert len(coord_breaks) > 0, "应该识别到坐标断点"
        
        # 转换为元件
        converter = ElementConverter(data_model)
        elements = converter.convert()
        
        # 验证生成了元件
        assert len(elements) > 0, "应该生成光学元件"
        
        # 获取转换元件的元数据
        converted = converter.get_converted_elements()
        
        # 验证折叠镜的 is_fold 标志
        fold_mirrors = [ce for ce in converted if ce.is_fold_mirror]
        assert len(fold_mirrors) > 0, "应该识别到折叠镜"
        
        # 验证所有折叠镜的 is_fold 属性正确设置
        for ce in fold_mirrors:
            assert ce.element.is_fold == True, f"折叠镜 {ce.zmx_surface_index} 的 is_fold 应为 True"

    def test_elements_have_valid_thickness(self, zmx_test_dir):
        """测试所有元件都有有效的厚度值
        
        **Validates: Requirements 7.2, 7.5**
        """
        zmx_path = os.path.join(zmx_test_dir, "one_mirror_up_45deg.zmx")
        
        if not os.path.exists(zmx_path):
            pytest.skip(f"测试文件不存在: {zmx_path}")
        
        parser = ZmxParser(zmx_path)
        data_model = parser.parse()
        
        converter = ElementConverter(data_model)
        elements = converter.convert()
        
        # 验证所有元件的厚度都是有效的数值
        for i, elem in enumerate(elements):
            assert hasattr(elem, 'thickness'), f"元件 {i} 应该有 thickness 属性"
            assert isinstance(elem.thickness, (int, float)), f"元件 {i} 的 thickness 应该是数值"
            assert not np.isnan(elem.thickness), f"元件 {i} 的 thickness 不应该是 NaN"
            # 厚度应该是非负的（绝对值）
            assert elem.thickness >= 0, f"元件 {i} 的 thickness 应该 >= 0"
    
    def test_elements_have_valid_semi_aperture(self, zmx_test_dir):
        """测试所有元件都有有效的半口径值
        
        **Validates: Requirements 7.4**
        """
        zmx_path = os.path.join(zmx_test_dir, "one_mirror_up_45deg.zmx")
        
        if not os.path.exists(zmx_path):
            pytest.skip(f"测试文件不存在: {zmx_path}")
        
        parser = ZmxParser(zmx_path)
        data_model = parser.parse()
        
        converter = ElementConverter(data_model)
        elements = converter.convert()
        
        # 验证所有元件的半口径都是有效的数值
        for i, elem in enumerate(elements):
            assert hasattr(elem, 'semi_aperture'), f"元件 {i} 应该有 semi_aperture 属性"
            assert isinstance(elem.semi_aperture, (int, float)), f"元件 {i} 的 semi_aperture 应该是数值"
            assert elem.semi_aperture > 0, f"元件 {i} 的 semi_aperture 应该 > 0"


# =============================================================================
# ElementConverter 其他功能测试
# =============================================================================

class TestElementConverterMiscellaneous:
    """ElementConverter 其他功能测试"""
    
    def test_convert_returns_list(self):
        """测试 convert() 返回列表"""
        model = ZmxDataModel()
        converter = ElementConverter(model)
        
        result = converter.convert()
        
        assert isinstance(result, list)
    
    def test_convert_empty_model(self):
        """测试空数据模型的转换"""
        model = ZmxDataModel()
        converter = ElementConverter(model)
        
        elements = converter.convert()
        
        assert elements == []
    
    def test_get_converted_elements_before_convert(self):
        """测试在 convert() 之前调用 get_converted_elements()"""
        model = ZmxDataModel()
        converter = ElementConverter(model)
        
        # 在 convert() 之前调用应该返回空列表
        converted = converter.get_converted_elements()
        
        assert converted == []
    
    def test_convert_resets_state(self):
        """测试多次调用 convert() 重置状态"""
        model = ZmxDataModel()
        
        # 物面
        model.surfaces[0] = ZmxSurfaceData(
            index=0,
            surface_type='standard',
            thickness=np.inf,
        )
        # 反射镜
        model.surfaces[1] = ZmxSurfaceData(
            index=1,
            surface_type='standard',
            radius=np.inf,
            thickness=100.0,
            is_mirror=True,
            semi_diameter=25.0,
        )
        # 像面
        model.surfaces[2] = ZmxSurfaceData(
            index=2,
            surface_type='standard',
            thickness=0.0,
        )
        
        converter = ElementConverter(model)
        
        # 第一次转换
        elements1 = converter.convert()
        assert len(elements1) == 1
        
        # 第二次转换应该返回相同结果
        elements2 = converter.convert()
        assert len(elements2) == 1
        
        # 两次结果应该是不同的对象
        assert elements1[0] is not elements2[0]

    def test_skip_object_and_image_surfaces(self):
        """测试跳过物面和像面
        
        **Validates: Requirements 7.1**
        """
        model = ZmxDataModel()
        
        # 物面（index=0）- 应该跳过
        model.surfaces[0] = ZmxSurfaceData(
            index=0,
            surface_type='standard',
            thickness=np.inf,
        )
        # 反射镜
        model.surfaces[1] = ZmxSurfaceData(
            index=1,
            surface_type='standard',
            radius=np.inf,
            thickness=100.0,
            is_mirror=True,
            semi_diameter=25.0,
        )
        # 像面（最后一个）- 应该跳过
        model.surfaces[2] = ZmxSurfaceData(
            index=2,
            surface_type='standard',
            thickness=0.0,
        )
        
        converter = ElementConverter(model)
        elements = converter.convert()
        
        # 只应该有一个元件（反射镜）
        assert len(elements) == 1
    
    def test_skip_air_gaps(self):
        """测试跳过空气间隔
        
        **Validates: Requirements 7.2**
        """
        model = ZmxDataModel()
        
        # 物面
        model.surfaces[0] = ZmxSurfaceData(
            index=0,
            surface_type='standard',
            thickness=np.inf,
        )
        # 空气间隔（平面，无材料）- 应该跳过
        model.surfaces[1] = ZmxSurfaceData(
            index=1,
            surface_type='standard',
            radius=np.inf,
            thickness=50.0,
            material='air',
        )
        # 反射镜
        model.surfaces[2] = ZmxSurfaceData(
            index=2,
            surface_type='standard',
            radius=np.inf,
            thickness=100.0,
            is_mirror=True,
            semi_diameter=25.0,
        )
        # 像面
        model.surfaces[3] = ZmxSurfaceData(
            index=3,
            surface_type='standard',
            thickness=0.0,
        )
        
        converter = ElementConverter(model)
        elements = converter.convert()
        
        # 只应该有一个元件（反射镜）
        assert len(elements) == 1
    
    def test_element_sequence_order(self):
        """测试元件序列顺序正确
        
        **Validates: Requirements 7.1**
        """
        model = ZmxDataModel()
        
        # 物面
        model.surfaces[0] = ZmxSurfaceData(
            index=0,
            surface_type='standard',
            thickness=np.inf,
        )
        # 第一个反射镜
        model.surfaces[1] = ZmxSurfaceData(
            index=1,
            surface_type='standard',
            radius=np.inf,
            thickness=100.0,
            is_mirror=True,
            semi_diameter=25.0,
            comment="M1",
        )
        # 第二个反射镜
        model.surfaces[2] = ZmxSurfaceData(
            index=2,
            surface_type='standard',
            radius=200.0,
            conic=-1.0,
            thickness=150.0,
            is_mirror=True,
            semi_diameter=30.0,
            comment="M2",
        )
        # 像面
        model.surfaces[3] = ZmxSurfaceData(
            index=3,
            surface_type='standard',
            thickness=0.0,
        )
        
        converter = ElementConverter(model)
        elements = converter.convert()
        
        assert len(elements) == 2
        
        # 验证顺序正确
        converted = converter.get_converted_elements()
        assert converted[0].zmx_surface_index == 1
        assert converted[1].zmx_surface_index == 2
        
        # 验证类型正确
        assert isinstance(elements[0], FlatMirror)
        assert isinstance(elements[1], ParabolicMirror)


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# =============================================================================
# CodeGenerator 测试
# =============================================================================

class TestCodeGenerator:
    """测试 CodeGenerator 代码生成器
    
    测试代码生成功能，包括：
    - 代码可执行性
    - 参数完整性
    - 注释正确性
    - 格式正确性
    
    **Validates: Requirements 9.1-9.7**
    """
    
    # =========================================================================
    # 基础功能测试
    # =========================================================================
    
    def test_generate_empty_elements(self):
        """测试空元件列表的代码生成"""
        from sequential_system.zmx_converter import CodeGenerator
        
        generator = CodeGenerator([])
        code = generator.generate()
        
        # 应该包含注释说明没有元件
        assert "没有转换的元件" in code
    
    def test_generate_with_imports(self):
        """测试包含 import 语句的代码生成
        
        **Validates: Requirements 9.7**
        """
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        mirror = FlatMirror(
            thickness=100.0,
            semi_aperture=25.0,
        )
        converted = ConvertedElement(
            element=mirror,
            zmx_surface_index=1,
        )
        
        generator = CodeGenerator([converted])
        code = generator.generate(include_imports=True)
        
        # 验证包含 import 语句
        assert "from gaussian_beam_simulation.optical_elements import" in code
        assert "FlatMirror" in code
    
    def test_generate_without_imports(self):
        """测试不包含 import 语句的代码生成
        
        **Validates: Requirements 9.7**
        """
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        mirror = FlatMirror(
            thickness=100.0,
            semi_aperture=25.0,
        )
        converted = ConvertedElement(
            element=mirror,
            zmx_surface_index=1,
        )
        
        generator = CodeGenerator([converted])
        code = generator.generate(include_imports=False)
        
        # 验证不包含 import 语句
        assert "from gaussian_beam_simulation.optical_elements import" not in code
        # 但应该包含元件创建代码
        assert "FlatMirror(" in code
    
    # =========================================================================
    # 代码可执行性测试
    # =========================================================================
    
    def test_generated_code_executable_flat_mirror(self):
        """测试生成的 FlatMirror 代码可执行
        
        **Validates: Requirements 9.1, 9.2**
        """
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        # 创建一个 FlatMirror 元件
        mirror = FlatMirror(
            thickness=100.0,
            semi_aperture=25.0,
            tilt_x=np.pi / 4,
            is_fold=True,
        )
        converted = ConvertedElement(
            element=mirror,
            zmx_surface_index=3,
            zmx_comment="M1",
            is_fold_mirror=True,
            fold_angle_deg=45.0,
        )
        
        # 生成代码
        generator = CodeGenerator([converted])
        code = generator.generate(include_imports=True)
        
        # 执行生成的代码
        exec_globals = {}
        exec(code, exec_globals)
        
        # 验证创建了正确的元件对象
        assert "m1" in exec_globals
        created_mirror = exec_globals["m1"]
        assert isinstance(created_mirror, FlatMirror)
        assert created_mirror.thickness == pytest.approx(100.0, rel=1e-6)
        assert created_mirror.semi_aperture == pytest.approx(25.0, rel=1e-6)
        assert created_mirror.tilt_x == pytest.approx(np.pi / 4, rel=1e-6)
        assert created_mirror.is_fold == True
    
    def test_generated_code_executable_parabolic_mirror(self):
        """测试生成的 ParabolicMirror 代码可执行
        
        **Validates: Requirements 9.1, 9.2**
        """
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        # 创建一个 ParabolicMirror 元件
        mirror = ParabolicMirror(
            thickness=150.0,
            semi_aperture=30.0,
            parent_focal_length=200.0,
        )
        converted = ConvertedElement(
            element=mirror,
            zmx_surface_index=5,
            zmx_comment="OAP1",
        )
        
        # 生成代码
        generator = CodeGenerator([converted])
        code = generator.generate(include_imports=True)
        
        # 执行生成的代码
        exec_globals = {}
        exec(code, exec_globals)
        
        # 验证创建了正确的元件对象
        assert "oap1" in exec_globals
        created_mirror = exec_globals["oap1"]
        assert isinstance(created_mirror, ParabolicMirror)
        assert created_mirror.thickness == pytest.approx(150.0, rel=1e-6)
        assert created_mirror.semi_aperture == pytest.approx(30.0, rel=1e-6)
        assert created_mirror.parent_focal_length == pytest.approx(200.0, rel=1e-6)
    
    def test_generated_code_executable_spherical_mirror(self):
        """测试生成的 SphericalMirror 代码可执行
        
        **Validates: Requirements 9.1, 9.2**
        """
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        # 创建一个 SphericalMirror 元件
        mirror = SphericalMirror(
            thickness=120.0,
            semi_aperture=20.0,
            radius_of_curvature=500.0,
        )
        converted = ConvertedElement(
            element=mirror,
            zmx_surface_index=7,
            zmx_comment="SM1",
        )
        
        # 生成代码
        generator = CodeGenerator([converted])
        code = generator.generate(include_imports=True)
        
        # 执行生成的代码
        exec_globals = {}
        exec(code, exec_globals)
        
        # 验证创建了正确的元件对象
        assert "sm1" in exec_globals
        created_mirror = exec_globals["sm1"]
        assert isinstance(created_mirror, SphericalMirror)
        assert created_mirror.thickness == pytest.approx(120.0, rel=1e-6)
        assert created_mirror.semi_aperture == pytest.approx(20.0, rel=1e-6)
        assert created_mirror.radius_of_curvature == pytest.approx(500.0, rel=1e-6)
    
    def test_generated_code_executable_multiple_elements(self):
        """测试生成的多元件代码可执行
        
        **Validates: Requirements 9.1, 9.2**
        """
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        # 创建多个元件
        mirror1 = FlatMirror(
            thickness=100.0,
            semi_aperture=25.0,
            tilt_x=np.pi / 4,
            is_fold=True,
        )
        converted1 = ConvertedElement(
            element=mirror1,
            zmx_surface_index=3,
            zmx_comment="M1",
            is_fold_mirror=True,
            fold_angle_deg=45.0,
        )
        
        mirror2 = ParabolicMirror(
            thickness=150.0,
            semi_aperture=30.0,
            parent_focal_length=200.0,
        )
        converted2 = ConvertedElement(
            element=mirror2,
            zmx_surface_index=5,
            zmx_comment="OAP1",
        )
        
        # 生成代码
        generator = CodeGenerator([converted1, converted2])
        code = generator.generate(include_imports=True)
        
        # 执行生成的代码
        exec_globals = {}
        exec(code, exec_globals)
        
        # 验证创建了正确的元件对象
        assert "m1" in exec_globals
        assert "oap1" in exec_globals
        assert isinstance(exec_globals["m1"], FlatMirror)
        assert isinstance(exec_globals["oap1"], ParabolicMirror)

    
    # =========================================================================
    # 参数完整性测试
    # =========================================================================
    
    def test_params_thickness_always_present(self):
        """测试 thickness 参数始终存在
        
        **Validates: Requirements 9.3**
        """
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        mirror = FlatMirror(
            thickness=100.0,
            semi_aperture=25.0,
        )
        converted = ConvertedElement(
            element=mirror,
            zmx_surface_index=1,
        )
        
        generator = CodeGenerator([converted])
        code = generator.generate(include_imports=False)
        
        # 验证 thickness 参数存在
        assert "thickness=" in code
        assert "thickness=100.0" in code
    
    def test_params_semi_aperture_always_present(self):
        """测试 semi_aperture 参数始终存在
        
        **Validates: Requirements 9.3**
        """
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        mirror = FlatMirror(
            thickness=100.0,
            semi_aperture=25.0,
        )
        converted = ConvertedElement(
            element=mirror,
            zmx_surface_index=1,
        )
        
        generator = CodeGenerator([converted])
        code = generator.generate(include_imports=False)
        
        # 验证 semi_aperture 参数存在
        assert "semi_aperture=" in code
        assert "semi_aperture=25.0" in code
    
    def test_params_tilt_x_present_when_nonzero(self):
        """测试 tilt_x 在非零时存在
        
        **Validates: Requirements 9.3**
        """
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        # 有 tilt_x 的元件
        mirror_with_tilt = FlatMirror(
            thickness=100.0,
            semi_aperture=25.0,
            tilt_x=np.pi / 4,
        )
        converted_with_tilt = ConvertedElement(
            element=mirror_with_tilt,
            zmx_surface_index=1,
        )
        
        generator = CodeGenerator([converted_with_tilt])
        code = generator.generate(include_imports=False)
        
        # 验证 tilt_x 参数存在
        assert "tilt_x=" in code
    
    def test_params_tilt_x_absent_when_zero(self):
        """测试 tilt_x 在为零时不存在
        
        **Validates: Requirements 9.3**
        """
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        # 没有 tilt_x 的元件
        mirror_no_tilt = FlatMirror(
            thickness=100.0,
            semi_aperture=25.0,
            tilt_x=0.0,
        )
        converted_no_tilt = ConvertedElement(
            element=mirror_no_tilt,
            zmx_surface_index=2,
        )
        
        generator = CodeGenerator([converted_no_tilt])
        code = generator.generate(include_imports=False)
        
        # 验证 tilt_x 参数不存在
        assert "tilt_x=" not in code
    
    def test_params_tilt_y_present_when_nonzero(self):
        """测试 tilt_y 在非零时存在
        
        **Validates: Requirements 9.3**
        """
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        mirror = FlatMirror(
            thickness=100.0,
            semi_aperture=25.0,
            tilt_y=np.pi / 6,
        )
        converted = ConvertedElement(
            element=mirror,
            zmx_surface_index=1,
        )
        
        generator = CodeGenerator([converted])
        code = generator.generate(include_imports=False)
        
        # 验证 tilt_y 参数存在
        assert "tilt_y=" in code
    
    def test_params_tilt_y_absent_when_zero(self):
        """测试 tilt_y 在为零时不存在
        
        **Validates: Requirements 9.3**
        """
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        mirror = FlatMirror(
            thickness=100.0,
            semi_aperture=25.0,
            tilt_y=0.0,
        )
        converted = ConvertedElement(
            element=mirror,
            zmx_surface_index=1,
        )
        
        generator = CodeGenerator([converted])
        code = generator.generate(include_imports=False)
        
        # 验证 tilt_y 参数不存在
        assert "tilt_y=" not in code
    
    def test_params_decenter_x_present_when_nonzero(self):
        """测试 decenter_x 在非零时存在
        
        **Validates: Requirements 9.3**
        """
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        mirror = FlatMirror(
            thickness=100.0,
            semi_aperture=25.0,
            decenter_x=5.0,
        )
        converted = ConvertedElement(
            element=mirror,
            zmx_surface_index=1,
        )
        
        generator = CodeGenerator([converted])
        code = generator.generate(include_imports=False)
        
        # 验证 decenter_x 参数存在
        assert "decenter_x=" in code
    
    def test_params_decenter_x_absent_when_zero(self):
        """测试 decenter_x 在为零时不存在
        
        **Validates: Requirements 9.3**
        """
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        mirror = FlatMirror(
            thickness=100.0,
            semi_aperture=25.0,
            decenter_x=0.0,
        )
        converted = ConvertedElement(
            element=mirror,
            zmx_surface_index=1,
        )
        
        generator = CodeGenerator([converted])
        code = generator.generate(include_imports=False)
        
        # 验证 decenter_x 参数不存在
        assert "decenter_x=" not in code
    
    def test_params_decenter_y_present_when_nonzero(self):
        """测试 decenter_y 在非零时存在
        
        **Validates: Requirements 9.3**
        """
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        mirror = FlatMirror(
            thickness=100.0,
            semi_aperture=25.0,
            decenter_y=3.0,
        )
        converted = ConvertedElement(
            element=mirror,
            zmx_surface_index=1,
        )
        
        generator = CodeGenerator([converted])
        code = generator.generate(include_imports=False)
        
        # 验证 decenter_y 参数存在
        assert "decenter_y=" in code
    
    def test_params_decenter_y_absent_when_zero(self):
        """测试 decenter_y 在为零时不存在
        
        **Validates: Requirements 9.3**
        """
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        mirror = FlatMirror(
            thickness=100.0,
            semi_aperture=25.0,
            decenter_y=0.0,
        )
        converted = ConvertedElement(
            element=mirror,
            zmx_surface_index=1,
        )
        
        generator = CodeGenerator([converted])
        code = generator.generate(include_imports=False)
        
        # 验证 decenter_y 参数不存在
        assert "decenter_y=" not in code
    
    def test_params_is_fold_present_when_true(self):
        """测试 is_fold 在为 True 时存在
        
        **Validates: Requirements 9.3**
        """
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        mirror = FlatMirror(
            thickness=100.0,
            semi_aperture=25.0,
            is_fold=True,
        )
        converted = ConvertedElement(
            element=mirror,
            zmx_surface_index=1,
            is_fold_mirror=True,
            fold_angle_deg=45.0,
        )
        
        generator = CodeGenerator([converted])
        code = generator.generate(include_imports=False)
        
        # 验证 is_fold 参数存在
        assert "is_fold=True" in code
    
    def test_params_is_fold_absent_when_false(self):
        """测试 is_fold 在为 False 时不存在
        
        **Validates: Requirements 9.3**
        """
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        mirror = FlatMirror(
            thickness=100.0,
            semi_aperture=25.0,
            is_fold=False,
        )
        converted = ConvertedElement(
            element=mirror,
            zmx_surface_index=1,
        )
        
        generator = CodeGenerator([converted])
        code = generator.generate(include_imports=False)
        
        # 验证 is_fold 参数不存在
        assert "is_fold=" not in code

    
    # =========================================================================
    # 注释正确性测试
    # =========================================================================
    
    def test_comment_contains_zmx_surface_index(self):
        """测试注释包含 ZMX 表面索引
        
        **Validates: Requirements 9.4**
        """
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        mirror = FlatMirror(
            thickness=100.0,
            semi_aperture=25.0,
        )
        converted = ConvertedElement(
            element=mirror,
            zmx_surface_index=7,
        )
        
        generator = CodeGenerator([converted])
        code = generator.generate(include_imports=False)
        
        # 验证注释包含 ZMX 表面索引
        assert "# ZMX Surface 7" in code
    
    def test_comment_contains_original_comment(self):
        """测试注释包含原始 ZMX 注释
        
        **Validates: Requirements 9.4**
        """
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        mirror = FlatMirror(
            thickness=100.0,
            semi_aperture=25.0,
        )
        converted = ConvertedElement(
            element=mirror,
            zmx_surface_index=5,
            zmx_comment="Primary Mirror",
        )
        
        generator = CodeGenerator([converted])
        code = generator.generate(include_imports=False)
        
        # 验证注释包含原始注释
        assert "Primary Mirror" in code
        assert "# ZMX Surface 5" in code
    
    def test_comment_contains_fold_angle_for_fold_mirror(self):
        """测试折叠镜注释包含折叠角度
        
        **Validates: Requirements 9.5**
        """
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        mirror = FlatMirror(
            thickness=100.0,
            semi_aperture=25.0,
            tilt_x=np.pi / 4,
            is_fold=True,
        )
        converted = ConvertedElement(
            element=mirror,
            zmx_surface_index=3,
            zmx_comment="M1",
            is_fold_mirror=True,
            fold_angle_deg=45.0,
        )
        
        generator = CodeGenerator([converted])
        code = generator.generate(include_imports=False)
        
        # 验证注释包含折叠镜标识和角度
        assert "Fold Mirror" in code
        assert "45.0" in code
    
    def test_comment_no_fold_angle_for_non_fold_mirror(self):
        """测试非折叠镜注释不包含折叠角度
        
        **Validates: Requirements 9.5**
        """
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        mirror = FlatMirror(
            thickness=100.0,
            semi_aperture=25.0,
        )
        converted = ConvertedElement(
            element=mirror,
            zmx_surface_index=3,
            zmx_comment="M1",
            is_fold_mirror=False,
            fold_angle_deg=0.0,
        )
        
        generator = CodeGenerator([converted])
        code = generator.generate(include_imports=False)
        
        # 验证注释不包含折叠镜标识
        assert "Fold Mirror" not in code
    
    def test_comment_different_fold_angles(self):
        """测试不同折叠角度的注释
        
        **Validates: Requirements 9.5**
        """
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        # 30 度折叠镜
        mirror1 = FlatMirror(
            thickness=100.0,
            semi_aperture=25.0,
            tilt_x=np.pi / 6,
            is_fold=True,
        )
        converted1 = ConvertedElement(
            element=mirror1,
            zmx_surface_index=3,
            zmx_comment="M1",
            is_fold_mirror=True,
            fold_angle_deg=30.0,
        )
        
        # 60 度折叠镜
        mirror2 = FlatMirror(
            thickness=100.0,
            semi_aperture=25.0,
            tilt_y=np.pi / 3,
            is_fold=True,
        )
        converted2 = ConvertedElement(
            element=mirror2,
            zmx_surface_index=5,
            zmx_comment="M2",
            is_fold_mirror=True,
            fold_angle_deg=60.0,
        )
        
        generator = CodeGenerator([converted1, converted2])
        code = generator.generate(include_imports=False)
        
        # 验证两个折叠角度都存在
        assert "30.0" in code
        assert "60.0" in code
    
    # =========================================================================
    # 格式正确性测试
    # =========================================================================
    
    def test_format_import_statement(self):
        """测试 import 语句格式
        
        **Validates: Requirements 9.6, 9.7**
        """
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        mirror = FlatMirror(
            thickness=100.0,
            semi_aperture=25.0,
        )
        converted = ConvertedElement(
            element=mirror,
            zmx_surface_index=1,
        )
        
        generator = CodeGenerator([converted])
        code = generator.generate(include_imports=True)
        
        # 验证 import 语句格式
        assert "from gaussian_beam_simulation.optical_elements import (" in code
        assert "    FlatMirror," in code
        assert ")" in code
    
    def test_format_import_multiple_types(self):
        """测试多类型 import 语句格式
        
        **Validates: Requirements 9.6, 9.7**
        """
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        # 创建不同类型的元件
        flat_mirror = FlatMirror(thickness=100.0, semi_aperture=25.0)
        parabolic_mirror = ParabolicMirror(
            thickness=150.0, semi_aperture=30.0, parent_focal_length=200.0
        )
        spherical_mirror = SphericalMirror(
            thickness=120.0, semi_aperture=20.0, radius_of_curvature=500.0
        )
        
        converted_elements = [
            ConvertedElement(element=flat_mirror, zmx_surface_index=1),
            ConvertedElement(element=parabolic_mirror, zmx_surface_index=2),
            ConvertedElement(element=spherical_mirror, zmx_surface_index=3),
        ]
        
        generator = CodeGenerator(converted_elements)
        code = generator.generate(include_imports=True)
        
        # 验证所有类型都被导入
        assert "FlatMirror," in code
        assert "ParabolicMirror," in code
        assert "SphericalMirror," in code
    
    def test_format_indentation_4_spaces(self):
        """测试缩进为 4 空格
        
        **Validates: Requirements 9.6**
        """
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        mirror = FlatMirror(
            thickness=100.0,
            semi_aperture=25.0,
        )
        converted = ConvertedElement(
            element=mirror,
            zmx_surface_index=1,
        )
        
        generator = CodeGenerator([converted])
        code = generator.generate(include_imports=False)
        
        # 验证缩进为 4 空格
        lines = code.split('\n')
        for line in lines:
            if line.startswith(' '):
                # 检查缩进是 4 的倍数
                leading_spaces = len(line) - len(line.lstrip(' '))
                assert leading_spaces % 4 == 0, f"缩进不是 4 的倍数: '{line}'"
    
    def test_format_float_precision(self):
        """测试浮点数格式化
        
        **Validates: Requirements 9.6**
        """
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        # 使用精确的浮点数
        mirror = FlatMirror(
            thickness=100.123456,
            semi_aperture=25.5,
        )
        converted = ConvertedElement(
            element=mirror,
            zmx_surface_index=1,
        )
        
        generator = CodeGenerator([converted])
        code = generator.generate(include_imports=False)
        
        # 验证浮点数被正确格式化
        assert "thickness=100.123456" in code
        assert "semi_aperture=25.5" in code
    
    def test_format_float_integer_value(self):
        """测试整数值的浮点数格式化
        
        **Validates: Requirements 9.6**
        """
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        mirror = FlatMirror(
            thickness=100.0,
            semi_aperture=25.0,
        )
        converted = ConvertedElement(
            element=mirror,
            zmx_surface_index=1,
        )
        
        generator = CodeGenerator([converted])
        code = generator.generate(include_imports=False)
        
        # 验证整数值保留小数点
        assert "thickness=100.0" in code
        assert "semi_aperture=25.0" in code
    
    def test_format_element_creation_multiline(self):
        """测试元件创建语句为多行格式
        
        **Validates: Requirements 9.6**
        """
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        mirror = FlatMirror(
            thickness=100.0,
            semi_aperture=25.0,
            tilt_x=np.pi / 4,
            is_fold=True,
        )
        converted = ConvertedElement(
            element=mirror,
            zmx_surface_index=1,
            zmx_comment="M1",
            is_fold_mirror=True,
            fold_angle_deg=45.0,
        )
        
        generator = CodeGenerator([converted])
        code = generator.generate(include_imports=False)
        
        # 验证元件创建语句为多行格式
        assert "FlatMirror(" in code
        assert "    thickness=" in code
        assert "    semi_aperture=" in code
        assert ")" in code
    
    def test_format_elements_separated_by_blank_line(self):
        """测试元件之间用空行分隔
        
        **Validates: Requirements 9.6**
        """
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        mirror1 = FlatMirror(thickness=100.0, semi_aperture=25.0)
        mirror2 = FlatMirror(thickness=150.0, semi_aperture=30.0)
        
        converted_elements = [
            ConvertedElement(element=mirror1, zmx_surface_index=1),
            ConvertedElement(element=mirror2, zmx_surface_index=2),
        ]
        
        generator = CodeGenerator(converted_elements)
        code = generator.generate(include_imports=False)
        
        # 验证元件之间有空行
        # 第一个元件的结束 ")" 和第二个元件的注释之间应该有空行
        assert ")\n\n#" in code
    
    # =========================================================================
    # 变量名生成测试
    # =========================================================================
    
    def test_variable_name_from_comment(self):
        """测试从注释生成变量名"""
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        mirror = FlatMirror(thickness=100.0, semi_aperture=25.0)
        converted = ConvertedElement(
            element=mirror,
            zmx_surface_index=1,
            zmx_comment="Primary Mirror",
        )
        
        generator = CodeGenerator([converted])
        code = generator.generate(include_imports=False)
        
        # 验证变量名从注释生成（小写，空格转下划线）
        assert "primary_mirror = FlatMirror(" in code
    
    def test_variable_name_default_when_no_comment(self):
        """测试无注释时使用默认变量名"""
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        mirror = FlatMirror(thickness=100.0, semi_aperture=25.0)
        converted = ConvertedElement(
            element=mirror,
            zmx_surface_index=5,
            zmx_comment="",
        )
        
        generator = CodeGenerator([converted])
        code = generator.generate(include_imports=False)
        
        # 验证使用默认变量名
        assert "element_5 = FlatMirror(" in code
    
    def test_variable_name_special_characters_removed(self):
        """测试特殊字符被移除"""
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        mirror = FlatMirror(thickness=100.0, semi_aperture=25.0)
        converted = ConvertedElement(
            element=mirror,
            zmx_surface_index=1,
            zmx_comment="M1 - Fold (45°)",
        )
        
        generator = CodeGenerator([converted])
        code = generator.generate(include_imports=False)
        
        # 验证特殊字符被移除，生成有效的变量名
        # 应该生成类似 "m1_fold_45" 的变量名
        lines = code.split('\n')
        var_line = [l for l in lines if "FlatMirror(" in l][0]
        var_name = var_line.split('=')[0].strip()
        
        # 验证变量名是有效的 Python 标识符
        assert var_name.isidentifier(), f"变量名 '{var_name}' 不是有效的 Python 标识符"

    
    # =========================================================================
    # 类型特定参数测试
    # =========================================================================
    
    def test_parabolic_mirror_parent_focal_length(self):
        """测试 ParabolicMirror 的 parent_focal_length 参数
        
        **Validates: Requirements 9.3**
        """
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        mirror = ParabolicMirror(
            thickness=150.0,
            semi_aperture=30.0,
            parent_focal_length=200.0,
        )
        converted = ConvertedElement(
            element=mirror,
            zmx_surface_index=1,
        )
        
        generator = CodeGenerator([converted])
        code = generator.generate(include_imports=False)
        
        # 验证 parent_focal_length 参数存在
        assert "parent_focal_length=200.0" in code
    
    def test_spherical_mirror_radius_of_curvature(self):
        """测试 SphericalMirror 的 radius_of_curvature 参数
        
        **Validates: Requirements 9.3**
        """
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        mirror = SphericalMirror(
            thickness=120.0,
            semi_aperture=20.0,
            radius_of_curvature=500.0,
        )
        converted = ConvertedElement(
            element=mirror,
            zmx_surface_index=1,
        )
        
        generator = CodeGenerator([converted])
        code = generator.generate(include_imports=False)
        
        # 验证 radius_of_curvature 参数存在
        assert "radius_of_curvature=500.0" in code
    
    # =========================================================================
    # 边界情况测试
    # =========================================================================
    
    def test_element_with_all_params(self):
        """测试包含所有参数的元件
        
        **Validates: Requirements 9.3**
        """
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        mirror = FlatMirror(
            thickness=100.0,
            semi_aperture=25.0,
            tilt_x=np.pi / 4,
            tilt_y=np.pi / 6,
            decenter_x=5.0,
            decenter_y=3.0,
            is_fold=True,
        )
        converted = ConvertedElement(
            element=mirror,
            zmx_surface_index=3,
            zmx_comment="M1",
            is_fold_mirror=True,
            fold_angle_deg=45.0,
        )
        
        generator = CodeGenerator([converted])
        code = generator.generate(include_imports=False)
        
        # 验证所有参数都存在
        assert "thickness=100.0" in code
        assert "semi_aperture=25.0" in code
        assert "tilt_x=" in code
        assert "tilt_y=" in code
        assert "decenter_x=5.0" in code
        assert "decenter_y=3.0" in code
        assert "is_fold=True" in code
    
    def test_element_with_minimal_params(self):
        """测试只有必需参数的元件
        
        **Validates: Requirements 9.3**
        
        注意：OpticalElement 基类中 is_fold 默认为 True，
        所以需要显式设置 is_fold=False 来测试"最小参数"情况。
        """
        from sequential_system.zmx_converter import CodeGenerator, ConvertedElement
        
        # 显式设置 is_fold=False，因为基类默认值是 True
        mirror = FlatMirror(
            thickness=100.0,
            semi_aperture=25.0,
            is_fold=False,  # 显式设置为 False
        )
        converted = ConvertedElement(
            element=mirror,
            zmx_surface_index=1,
        )
        
        generator = CodeGenerator([converted])
        code = generator.generate(include_imports=False)
        
        # 验证只有必需参数
        assert "thickness=100.0" in code
        assert "semi_aperture=25.0" in code
        # 可选参数不应该存在
        assert "tilt_x=" not in code
        assert "tilt_y=" not in code
        assert "decenter_x=" not in code
        assert "decenter_y=" not in code
        assert "is_fold=" not in code
    
    def test_code_generator_indent_constant(self):
        """测试 INDENT 常量为 4 空格"""
        from sequential_system.zmx_converter import CodeGenerator
        
        assert CodeGenerator.INDENT == "    "
        assert len(CodeGenerator.INDENT) == 4
    
    def test_format_float_method(self):
        """测试 _format_float 方法"""
        from sequential_system.zmx_converter import CodeGenerator
        
        generator = CodeGenerator([])
        
        # 测试整数值
        assert generator._format_float(100.0) == "100.0"
        
        # 测试小数值
        assert generator._format_float(3.14159) == "3.14159"
        
        # 测试无穷大
        assert generator._format_float(float('inf')) == "float('inf')"
        assert generator._format_float(float('-inf')) == "float('-inf')"
        
        # 测试 NaN
        assert generator._format_float(float('nan')) == "float('nan')"
    
    def test_format_value_method(self):
        """测试 _format_value 方法"""
        from sequential_system.zmx_converter import CodeGenerator
        
        generator = CodeGenerator([])
        
        # 测试布尔值
        assert generator._format_value(True) == "True"
        assert generator._format_value(False) == "False"
        
        # 测试浮点数
        assert generator._format_value(100.0) == "100.0"
        
        # 测试字符串
        assert generator._format_value("test") == "'test'"
