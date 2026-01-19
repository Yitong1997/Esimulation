"""
ZMX 文件读取器属性基测试

本模块包含 ZMX 文件读取器的属性基测试（Property-Based Testing），
使用 hypothesis 库验证系统的正确性属性。

属性测试列表：
- Property 1: 表面数据提取完整性
- Property 2: 坐标断点参数提取
- Property 3: 反射镜类型分类
- Property 4: 折叠镜检测和配置
- Property 7: 代码生成往返测试

作者：混合光学仿真项目
"""

import sys
from pathlib import Path
import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from sequential_system import (
    ZmxParser,
    ZmxDataModel,
    ZmxSurfaceData,
    ElementConverter,
    ConvertedElement,
    CoordinateTransform,
    CodeGenerator,
)
from gaussian_beam_simulation.optical_elements import (
    FlatMirror,
    ParabolicMirror,
    SphericalMirror,
)


# =============================================================================
# 测试策略定义
# =============================================================================

# 曲率半径策略：正值、负值或无穷大
radius_strategy = st.one_of(
    st.floats(min_value=10.0, max_value=10000.0),  # 正曲率半径
    st.floats(min_value=-10000.0, max_value=-10.0),  # 负曲率半径
    st.just(np.inf),  # 平面
)

# 厚度策略：正值
thickness_strategy = st.floats(min_value=0.1, max_value=1000.0)

# 圆锥常数策略
conic_strategy = st.floats(min_value=-10.0, max_value=10.0)

# 倾斜角度策略（度）
tilt_deg_strategy = st.floats(min_value=-90.0, max_value=90.0)

# 偏心策略（mm）
decenter_strategy = st.floats(min_value=-100.0, max_value=100.0)

# 半口径策略（mm）
semi_diameter_strategy = st.floats(min_value=1.0, max_value=100.0)


# =============================================================================
# Property 1: 表面数据提取完整性
# =============================================================================

class TestProperty1SurfaceDataExtraction:
    """Property 1: 表面数据提取完整性
    
    对于任何有效的 ZMX 表面数据，解析后访问表面数据应返回所有预期参数
    （radius, thickness, conic, material, is_mirror, is_stop, semi_diameter）
    且值正确。
    
    **Validates: Requirements 2.1, 2.3, 2.4, 2.5, 2.6**
    """
    
    @given(
        radius=radius_strategy,
        thickness=thickness_strategy,
        conic=conic_strategy,
        semi_diameter=semi_diameter_strategy,
    )
    @settings(max_examples=100)
    def test_surface_data_completeness(
        self,
        radius: float,
        thickness: float,
        conic: float,
        semi_diameter: float,
    ):
        """
        **Feature: zmx-file-reader, Property 1: Surface Data Extraction Completeness**
        **Validates: Requirements 2.1, 2.3, 2.4, 2.5, 2.6**
        
        测试 ZmxSurfaceData 能正确存储和返回所有参数。
        """
        # 创建表面数据
        surface = ZmxSurfaceData(
            index=1,
            surface_type='standard',
            radius=radius,
            thickness=thickness,
            conic=conic,
            semi_diameter=semi_diameter,
            material='air',
            is_mirror=False,
            is_stop=False,
        )
        
        # 验证所有参数都被正确存储
        assert surface.index == 1
        assert surface.surface_type == 'standard'
        
        # 处理 NaN 和 inf 的比较
        if np.isfinite(radius):
            assert abs(surface.radius - radius) < 1e-10
        else:
            assert np.isinf(surface.radius)
        
        assert abs(surface.thickness - thickness) < 1e-10
        assert abs(surface.conic - conic) < 1e-10
        assert abs(surface.semi_diameter - semi_diameter) < 1e-10
        assert surface.material == 'air'
        assert surface.is_mirror == False
        assert surface.is_stop == False
    
    @given(
        is_mirror=st.booleans(),
        is_stop=st.booleans(),
    )
    @settings(max_examples=50)
    def test_boolean_flags(self, is_mirror: bool, is_stop: bool):
        """
        **Feature: zmx-file-reader, Property 1: Surface Data Extraction Completeness**
        **Validates: Requirements 2.4, 2.5**
        
        测试布尔标志（is_mirror, is_stop）的正确存储。
        """
        surface = ZmxSurfaceData(
            index=1,
            surface_type='standard',
            is_mirror=is_mirror,
            is_stop=is_stop,
        )
        
        assert surface.is_mirror == is_mirror
        assert surface.is_stop == is_stop


# =============================================================================
# Property 2: 坐标断点参数提取
# =============================================================================

class TestProperty2CoordinateBreakExtraction:
    """Property 2: 坐标断点参数提取
    
    对于任何 COORDBRK 表面，解析应提取所有坐标断点参数
    （decenter_x, decenter_y, tilt_x, tilt_y, tilt_z, thickness）
    并正确将倾斜值从度转换为弧度。
    
    **Validates: Requirements 3.1, 3.2, 3.3**
    """
    
    @given(
        decenter_x=decenter_strategy,
        decenter_y=decenter_strategy,
        tilt_x_deg=tilt_deg_strategy,
        tilt_y_deg=tilt_deg_strategy,
        tilt_z_deg=tilt_deg_strategy,
    )
    @settings(max_examples=100)
    def test_coordinate_break_parameters(
        self,
        decenter_x: float,
        decenter_y: float,
        tilt_x_deg: float,
        tilt_y_deg: float,
        tilt_z_deg: float,
    ):
        """
        **Feature: zmx-file-reader, Property 2: Coordinate Break Parameter Extraction**
        **Validates: Requirements 3.1, 3.2, 3.3**
        
        测试坐标断点参数的正确存储。
        """
        surface = ZmxSurfaceData(
            index=1,
            surface_type='coordinate_break',
            decenter_x=decenter_x,
            decenter_y=decenter_y,
            tilt_x_deg=tilt_x_deg,
            tilt_y_deg=tilt_y_deg,
            tilt_z_deg=tilt_z_deg,
        )
        
        assert surface.surface_type == 'coordinate_break'
        assert abs(surface.decenter_x - decenter_x) < 1e-10
        assert abs(surface.decenter_y - decenter_y) < 1e-10
        assert abs(surface.tilt_x_deg - tilt_x_deg) < 1e-10
        assert abs(surface.tilt_y_deg - tilt_y_deg) < 1e-10
        assert abs(surface.tilt_z_deg - tilt_z_deg) < 1e-10
    
    @given(
        tilt_deg=tilt_deg_strategy,
    )
    @settings(max_examples=100)
    def test_tilt_conversion_to_radians(self, tilt_deg: float):
        """
        **Feature: zmx-file-reader, Property 2: Coordinate Break Parameter Extraction**
        **Validates: Requirements 3.2**
        
        测试倾斜角度从度到弧度的转换。
        """
        transform = CoordinateTransform()
        transform.apply_coordinate_break(
            dx=0, dy=0, dz=0,
            rx_deg=tilt_deg, ry_deg=0, rz_deg=0,
        )
        
        expected_rad = np.deg2rad(tilt_deg)
        assert abs(transform.tilt_x_rad - expected_rad) < 1e-10


# =============================================================================
# Property 3: 反射镜类型分类
# =============================================================================

class TestProperty3MirrorTypeClassification:
    """Property 3: 反射镜类型分类
    
    对于任何反射镜表面（GLAS MIRROR），ElementConverter 应创建正确的元件类型：
    - radius = inf → FlatMirror
    - conic = -1 → ParabolicMirror
    - 其他 → SphericalMirror
    
    **Validates: Requirements 5.1, 5.2, 5.3**
    """
    
    def _create_mirror_data_model(
        self,
        radius: float,
        conic: float,
        semi_diameter: float = 10.0,
    ) -> ZmxDataModel:
        """创建包含单个反射镜的数据模型"""
        data_model = ZmxDataModel()
        
        # 物面
        data_model.surfaces[0] = ZmxSurfaceData(
            index=0,
            surface_type='standard',
            thickness=np.inf,
        )
        
        # 反射镜
        data_model.surfaces[1] = ZmxSurfaceData(
            index=1,
            surface_type='standard',
            radius=radius,
            conic=conic,
            thickness=100.0,
            is_mirror=True,
            semi_diameter=semi_diameter,
        )
        
        # 像面
        data_model.surfaces[2] = ZmxSurfaceData(
            index=2,
            surface_type='standard',
        )
        
        return data_model
    
    @given(
        semi_diameter=semi_diameter_strategy,
    )
    @settings(max_examples=50)
    def test_flat_mirror_classification(self, semi_diameter: float):
        """
        **Feature: zmx-file-reader, Property 3: Mirror Type Classification**
        **Validates: Requirements 5.1**
        
        测试平面镜（radius=inf）被正确分类为 FlatMirror。
        """
        data_model = self._create_mirror_data_model(
            radius=np.inf,
            conic=0.0,
            semi_diameter=semi_diameter,
        )
        
        converter = ElementConverter(data_model)
        elements = converter.convert()
        
        assert len(elements) == 1
        assert isinstance(elements[0], FlatMirror)
    
    @given(
        radius=st.floats(min_value=10.0, max_value=10000.0),
        semi_diameter=semi_diameter_strategy,
    )
    @settings(max_examples=50)
    def test_parabolic_mirror_classification(
        self,
        radius: float,
        semi_diameter: float,
    ):
        """
        **Feature: zmx-file-reader, Property 3: Mirror Type Classification**
        **Validates: Requirements 5.2**
        
        测试抛物面镜（conic=-1）被正确分类为 ParabolicMirror。
        """
        data_model = self._create_mirror_data_model(
            radius=radius,
            conic=-1.0,
            semi_diameter=semi_diameter,
        )
        
        converter = ElementConverter(data_model)
        elements = converter.convert()
        
        assert len(elements) == 1
        assert isinstance(elements[0], ParabolicMirror)
        
        # 验证焦距计算正确
        expected_focal_length = radius / 2.0
        assert abs(elements[0].parent_focal_length - expected_focal_length) < 1e-6
    
    @given(
        radius=st.floats(min_value=10.0, max_value=10000.0),
        conic=st.floats(min_value=-10.0, max_value=10.0).filter(lambda x: abs(x + 1.0) > 0.001),
        semi_diameter=semi_diameter_strategy,
    )
    @settings(max_examples=50)
    def test_spherical_mirror_classification(
        self,
        radius: float,
        conic: float,
        semi_diameter: float,
    ):
        """
        **Feature: zmx-file-reader, Property 3: Mirror Type Classification**
        **Validates: Requirements 5.3**
        
        测试球面镜（conic != -1 且 radius != inf）被正确分类为 SphericalMirror。
        """
        data_model = self._create_mirror_data_model(
            radius=radius,
            conic=conic,
            semi_diameter=semi_diameter,
        )
        
        converter = ElementConverter(data_model)
        elements = converter.convert()
        
        assert len(elements) == 1
        assert isinstance(elements[0], SphericalMirror)
        
        # 验证曲率半径正确
        assert abs(elements[0].radius_of_curvature - radius) < 1e-6


# =============================================================================
# Property 4: 折叠镜检测和配置
# =============================================================================

class TestProperty4FoldMirrorDetection:
    """Property 4: 折叠镜检测和配置
    
    对于任何倾斜角度 >= 5 度的反射镜，ElementConverter 应设置 is_fold=True
    并正确应用倾斜角度。对于倾斜角度 < 5 度的反射镜，is_fold 应为 False。
    
    **Validates: Requirements 5.7, 5.8, 7.3**
    """
    
    def _create_tilted_mirror_data_model(
        self,
        tilt_x_deg: float,
        tilt_y_deg: float = 0.0,
    ) -> ZmxDataModel:
        """创建包含倾斜反射镜的数据模型"""
        data_model = ZmxDataModel()
        
        # 物面
        data_model.surfaces[0] = ZmxSurfaceData(
            index=0,
            surface_type='standard',
            thickness=np.inf,
        )
        
        # 坐标断点（定义倾斜）
        data_model.surfaces[1] = ZmxSurfaceData(
            index=1,
            surface_type='coordinate_break',
            tilt_x_deg=tilt_x_deg,
            tilt_y_deg=tilt_y_deg,
            thickness=0.0,
        )
        
        # 反射镜
        data_model.surfaces[2] = ZmxSurfaceData(
            index=2,
            surface_type='standard',
            radius=np.inf,
            thickness=0.0,
            is_mirror=True,
            semi_diameter=10.0,
        )
        
        # 恢复坐标断点
        data_model.surfaces[3] = ZmxSurfaceData(
            index=3,
            surface_type='coordinate_break',
            tilt_x_deg=tilt_x_deg,  # 相同角度恢复
            thickness=-50.0,  # 负厚度表示反射方向传播
        )
        
        # 像面
        data_model.surfaces[4] = ZmxSurfaceData(
            index=4,
            surface_type='standard',
        )
        
        return data_model
    
    @given(
        tilt_deg=st.floats(min_value=5.0, max_value=89.0),
    )
    @settings(max_examples=50)
    def test_fold_mirror_detection_large_tilt(self, tilt_deg: float):
        """
        **Feature: zmx-file-reader, Property 4: Fold Mirror Detection and Configuration**
        **Validates: Requirements 5.7**
        
        测试大倾斜角度（>= 5°）的反射镜被识别为折叠镜。
        """
        data_model = self._create_tilted_mirror_data_model(tilt_x_deg=tilt_deg)
        
        converter = ElementConverter(data_model)
        converter.convert()
        converted_elements = converter.get_converted_elements()
        
        assert len(converted_elements) == 1
        ce = converted_elements[0]
        
        # 应该被识别为折叠镜
        assert ce.is_fold_mirror, f"倾斜 {tilt_deg}° 应该被识别为折叠镜"
        assert ce.element.is_fold, f"元件的 is_fold 应该为 True"
        
        # 折叠角度应该正确
        assert abs(ce.fold_angle_deg - tilt_deg) < 0.01
    
    @given(
        tilt_deg=st.floats(min_value=0.0, max_value=4.9),
    )
    @settings(max_examples=50)
    def test_misalignment_detection_small_tilt(self, tilt_deg: float):
        """
        **Feature: zmx-file-reader, Property 4: Fold Mirror Detection and Configuration**
        **Validates: Requirements 5.8**
        
        测试小倾斜角度（< 5°）的反射镜被识别为失调（非折叠镜）。
        """
        data_model = self._create_tilted_mirror_data_model(tilt_x_deg=tilt_deg)
        
        converter = ElementConverter(data_model)
        converter.convert()
        converted_elements = converter.get_converted_elements()
        
        assert len(converted_elements) == 1
        ce = converted_elements[0]
        
        # 不应该被识别为折叠镜
        assert not ce.is_fold_mirror, f"倾斜 {tilt_deg}° 不应该被识别为折叠镜"
        assert not ce.element.is_fold, f"元件的 is_fold 应该为 False"
    
    @given(
        tilt_deg=st.floats(min_value=5.0, max_value=89.0),
    )
    @settings(max_examples=50)
    def test_tilt_angle_application(self, tilt_deg: float):
        """
        **Feature: zmx-file-reader, Property 4: Fold Mirror Detection and Configuration**
        **Validates: Requirements 7.3**
        
        测试倾斜角度被正确应用到元件。
        """
        data_model = self._create_tilted_mirror_data_model(tilt_x_deg=tilt_deg)
        
        converter = ElementConverter(data_model)
        elements = converter.convert()
        
        assert len(elements) == 1
        mirror = elements[0]
        
        # 验证倾斜角度（弧度）
        expected_tilt_rad = np.deg2rad(tilt_deg)
        actual_tilt = max(abs(mirror.tilt_x), abs(mirror.tilt_y))
        assert abs(actual_tilt - expected_tilt_rad) < 0.01


# =============================================================================
# Property 7: 代码生成往返测试
# =============================================================================

class TestProperty7CodeGenerationRoundTrip:
    """Property 7: 代码生成往返测试
    
    对于任何转换后的 OpticalElements 集合，生成代码然后执行该代码
    应产生具有相同参数的等效 OpticalElements 集合。
    
    **Validates: Requirements 9.1, 9.2, 9.3**
    """
    
    @given(
        thickness=thickness_strategy,
        semi_aperture=semi_diameter_strategy,
        tilt_x_deg=st.floats(min_value=5.0, max_value=89.0),
    )
    @settings(max_examples=50)
    def test_flat_mirror_round_trip(
        self,
        thickness: float,
        semi_aperture: float,
        tilt_x_deg: float,
    ):
        """
        **Feature: zmx-file-reader, Property 7: Code Generation Round-Trip**
        **Validates: Requirements 9.1, 9.2, 9.3**
        
        测试 FlatMirror 的代码生成往返。
        """
        # 创建原始元件
        tilt_x_rad = np.deg2rad(tilt_x_deg)
        original = FlatMirror(
            thickness=thickness,
            semi_aperture=semi_aperture,
            tilt_x=tilt_x_rad,
            is_fold=True,
        )
        
        # 创建 ConvertedElement
        converted = ConvertedElement(
            element=original,
            zmx_surface_index=1,
            zmx_comment="test_mirror",
            is_fold_mirror=True,
            fold_angle_deg=tilt_x_deg,
        )
        
        # 生成代码
        generator = CodeGenerator([converted])
        code = generator.generate()
        
        # 执行代码
        exec_globals = {}
        exec(code, exec_globals)
        
        # 验证生成的元件
        assert "test_mirror" in exec_globals
        recreated = exec_globals["test_mirror"]
        
        assert isinstance(recreated, FlatMirror)
        assert abs(recreated.thickness - thickness) < 1e-6
        assert abs(recreated.semi_aperture - semi_aperture) < 1e-6
        assert abs(recreated.tilt_x - tilt_x_rad) < 1e-6
        assert recreated.is_fold == True
    
    @given(
        thickness=thickness_strategy,
        semi_aperture=semi_diameter_strategy,
        focal_length=st.floats(min_value=10.0, max_value=1000.0),
    )
    @settings(max_examples=50)
    def test_parabolic_mirror_round_trip(
        self,
        thickness: float,
        semi_aperture: float,
        focal_length: float,
    ):
        """
        **Feature: zmx-file-reader, Property 7: Code Generation Round-Trip**
        **Validates: Requirements 9.1, 9.2, 9.3**
        
        测试 ParabolicMirror 的代码生成往返。
        """
        # 创建原始元件
        original = ParabolicMirror(
            thickness=thickness,
            semi_aperture=semi_aperture,
            parent_focal_length=focal_length,
        )
        
        # 创建 ConvertedElement
        converted = ConvertedElement(
            element=original,
            zmx_surface_index=2,
            zmx_comment="parabolic",
        )
        
        # 生成代码
        generator = CodeGenerator([converted])
        code = generator.generate()
        
        # 执行代码
        exec_globals = {}
        exec(code, exec_globals)
        
        # 验证生成的元件
        assert "parabolic" in exec_globals
        recreated = exec_globals["parabolic"]
        
        assert isinstance(recreated, ParabolicMirror)
        assert abs(recreated.thickness - thickness) < 1e-6
        assert abs(recreated.semi_aperture - semi_aperture) < 1e-6
        assert abs(recreated.parent_focal_length - focal_length) < 1e-6
    
    @given(
        thickness=thickness_strategy,
        semi_aperture=semi_diameter_strategy,
        radius=st.floats(min_value=10.0, max_value=1000.0),
    )
    @settings(max_examples=50)
    def test_spherical_mirror_round_trip(
        self,
        thickness: float,
        semi_aperture: float,
        radius: float,
    ):
        """
        **Feature: zmx-file-reader, Property 7: Code Generation Round-Trip**
        **Validates: Requirements 9.1, 9.2, 9.3**
        
        测试 SphericalMirror 的代码生成往返。
        """
        # 创建原始元件
        original = SphericalMirror(
            thickness=thickness,
            semi_aperture=semi_aperture,
            radius_of_curvature=radius,
        )
        
        # 创建 ConvertedElement
        converted = ConvertedElement(
            element=original,
            zmx_surface_index=3,
            zmx_comment="spherical",
        )
        
        # 生成代码
        generator = CodeGenerator([converted])
        code = generator.generate()
        
        # 执行代码
        exec_globals = {}
        exec(code, exec_globals)
        
        # 验证生成的元件
        assert "spherical" in exec_globals
        recreated = exec_globals["spherical"]
        
        assert isinstance(recreated, SphericalMirror)
        assert abs(recreated.thickness - thickness) < 1e-6
        assert abs(recreated.semi_aperture - semi_aperture) < 1e-6
        assert abs(recreated.radius_of_curvature - radius) < 1e-6


# =============================================================================
# 运行测试
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
