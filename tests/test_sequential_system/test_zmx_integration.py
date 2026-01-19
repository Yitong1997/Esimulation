"""
ZMX 文件读取器集成测试

本模块包含 ZMX 文件读取器的集成测试，验证完整的解析和转换流程。

测试文件：
- complicated_fold_mirrors_setup_v2.zmx: 复杂折叠光路系统
- one_mirror_up_45deg.zmx: 单个 45 度折叠镜

作者：混合光学仿真项目
"""

import os
import sys
import pytest
import numpy as np
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from sequential_system import (
    ZmxParser,
    ZmxDataModel,
    ElementConverter,
    ConvertedElement,
    load_zmx_file,
    load_zmx_and_generate_code,
    ZmxParseError,
    ZmxUnsupportedError,
)
from gaussian_beam_simulation.optical_elements import (
    FlatMirror,
    ParabolicMirror,
    SphericalMirror,
)


# =============================================================================
# 测试数据路径
# =============================================================================

# ZMX 测试文件目录
ZMX_FILES_DIR = Path(__file__).parent.parent.parent / "optiland-master" / "tests" / "zemax_files"


def get_zmx_file_path(filename: str) -> str:
    """获取 ZMX 测试文件的完整路径"""
    return str(ZMX_FILES_DIR / filename)


# =============================================================================
# Task 8.1: complicated_fold_mirrors_setup_v2.zmx 集成测试
# =============================================================================

class TestComplicatedFoldMirrorsIntegration:
    """complicated_fold_mirrors_setup_v2.zmx 集成测试
    
    验证需求:
    - Requirements 10.1: 所有反射镜被正确识别
    - Requirements 10.2: 所有坐标断点被正确提取
    - Requirements 10.3: 折叠镜序列的 is_fold 标志正确
    - Requirements 10.6: 光程长度计算正确
    """
    
    @pytest.fixture
    def zmx_file_path(self) -> str:
        """获取测试文件路径"""
        return get_zmx_file_path("complicated_fold_mirrors_setup_v2.zmx")
    
    @pytest.fixture
    def parsed_data(self, zmx_file_path: str) -> ZmxDataModel:
        """解析 ZMX 文件"""
        parser = ZmxParser(zmx_file_path)
        return parser.parse()
    
    @pytest.fixture
    def converted_elements(self, parsed_data: ZmxDataModel) -> list:
        """转换为光学元件"""
        converter = ElementConverter(parsed_data)
        converter.convert()
        return converter.get_converted_elements()
    
    def test_file_exists(self, zmx_file_path: str):
        """测试文件存在"""
        assert os.path.exists(zmx_file_path), f"测试文件不存在: {zmx_file_path}"
    
    def test_parse_success(self, parsed_data: ZmxDataModel):
        """测试解析成功
        
        **Validates: Requirements 1.1, 1.2**
        """
        assert parsed_data is not None
        assert len(parsed_data.surfaces) > 0
    
    def test_entrance_pupil_diameter(self, parsed_data: ZmxDataModel):
        """测试入瞳直径提取
        
        **Validates: Requirements 4.4**
        """
        # 文件中 ENPD 20
        assert parsed_data.entrance_pupil_diameter == 20.0
    
    def test_wavelength_extraction(self, parsed_data: ZmxDataModel):
        """测试波长提取
        
        **Validates: Requirements 4.3**
        """
        # 文件中有多个 WAVM，波长为 0.55 μm
        assert len(parsed_data.wavelengths) > 0
        assert any(abs(w - 0.55) < 0.01 for w in parsed_data.wavelengths)

    def test_mirror_count(self, parsed_data: ZmxDataModel):
        """测试反射镜数量
        
        文件中有 5 个反射镜：
        - Surface 4: M1
        - Surface 7: 无名称
        - Surface 10: 无名称
        - Surface 13: 无名称
        - Surface 16: 无名称
        
        **Validates: Requirements 10.1**
        """
        mirrors = parsed_data.get_mirror_surfaces()
        assert len(mirrors) == 5, f"期望 5 个反射镜，实际 {len(mirrors)} 个"
    
    def test_mirror_indices(self, parsed_data: ZmxDataModel):
        """测试反射镜索引
        
        **Validates: Requirements 10.1**
        """
        mirrors = parsed_data.get_mirror_surfaces()
        mirror_indices = [m.index for m in mirrors]
        expected_indices = [4, 7, 10, 13, 16]
        assert mirror_indices == expected_indices, f"期望索引 {expected_indices}，实际 {mirror_indices}"
    
    def test_coordinate_break_count(self, parsed_data: ZmxDataModel):
        """测试坐标断点数量
        
        文件中有多个坐标断点：
        - Surface 3, 5, 6, 8, 9, 11, 12, 14, 15, 17
        
        **Validates: Requirements 10.2**
        """
        coord_breaks = parsed_data.get_coordinate_break_surfaces()
        assert len(coord_breaks) >= 10, f"期望至少 10 个坐标断点，实际 {len(coord_breaks)} 个"
    
    def test_coordinate_break_tilt_extraction(self, parsed_data: ZmxDataModel):
        """测试坐标断点倾斜角度提取
        
        Surface 3 应该有 tilt_x = 45 度
        
        **Validates: Requirements 3.1, 3.2**
        """
        surface_3 = parsed_data.get_surface(3)
        assert surface_3 is not None
        assert surface_3.surface_type == 'coordinate_break'
        assert abs(surface_3.tilt_x_deg - 45.0) < 0.01, f"期望 tilt_x=45°，实际 {surface_3.tilt_x_deg}°"
    
    def test_m1_comment(self, parsed_data: ZmxDataModel):
        """测试 M1 注释提取
        
        Surface 4 应该有注释 "M1"
        
        **Validates: Requirements 2.6**
        """
        surface_4 = parsed_data.get_surface(4)
        assert surface_4 is not None
        assert surface_4.comment == "M1"
    
    def test_converted_elements_count(self, converted_elements: list):
        """测试转换后元件数量
        
        应该有 5 个反射镜元件
        
        **Validates: Requirements 7.1**
        """
        # 只计算反射镜元件
        mirror_elements = [
            ce for ce in converted_elements
            if isinstance(ce.element, (FlatMirror, ParabolicMirror, SphericalMirror))
        ]
        assert len(mirror_elements) == 5, f"期望 5 个反射镜元件，实际 {len(mirror_elements)} 个"
    
    def test_all_mirrors_are_flat(self, converted_elements: list):
        """测试所有反射镜都是平面镜
        
        文件中所有反射镜的曲率半径都是无穷大
        
        **Validates: Requirements 5.1**
        """
        for ce in converted_elements:
            if isinstance(ce.element, (FlatMirror, ParabolicMirror, SphericalMirror)):
                assert isinstance(ce.element, FlatMirror), \
                    f"期望 FlatMirror，实际 {type(ce.element).__name__}"
    
    def test_fold_mirror_detection(self, converted_elements: list):
        """测试折叠镜检测
        
        所有反射镜都应该被识别为折叠镜（倾斜角度 >= 5°）
        
        **Validates: Requirements 10.3, 5.7**
        """
        for ce in converted_elements:
            if isinstance(ce.element, (FlatMirror, ParabolicMirror, SphericalMirror)):
                assert ce.is_fold_mirror, \
                    f"Surface {ce.zmx_surface_index} 应该被识别为折叠镜"
                assert ce.element.is_fold, \
                    f"Surface {ce.zmx_surface_index} 的 is_fold 应该为 True"
    
    def test_fold_angles(self, converted_elements: list):
        """测试折叠角度
        
        所有折叠镜的折叠角度应该是 45 度
        
        **Validates: Requirements 10.3**
        """
        for ce in converted_elements:
            if ce.is_fold_mirror:
                assert abs(ce.fold_angle_deg - 45.0) < 0.01, \
                    f"Surface {ce.zmx_surface_index} 折叠角度应该是 45°，实际 {ce.fold_angle_deg}°"
    
    def test_tilt_values(self, converted_elements: list):
        """测试倾斜值
        
        折叠镜的 tilt_x 或 tilt_y 应该是 π/4 (45°)
        
        **Validates: Requirements 7.4**
        """
        for ce in converted_elements:
            if ce.is_fold_mirror:
                tilt_x = ce.element.tilt_x
                tilt_y = ce.element.tilt_y
                max_tilt = max(abs(tilt_x), abs(tilt_y))
                expected_tilt = np.pi / 4  # 45 度
                assert abs(max_tilt - expected_tilt) < 0.01, \
                    f"Surface {ce.zmx_surface_index} 倾斜角度应该是 π/4，实际 {max_tilt}"

    def test_thickness_values_positive(self, converted_elements: list):
        """测试厚度值为正
        
        所有元件的 thickness 应该是正值
        
        **Validates: Requirements 10.6, 6.4**
        """
        for ce in converted_elements:
            assert ce.element.thickness >= 0, \
                f"Surface {ce.zmx_surface_index} 的 thickness 应该 >= 0，实际 {ce.element.thickness}"
    
    def test_m1_zmx_index(self, converted_elements: list):
        """测试 M1 的 ZMX 索引
        
        第一个反射镜应该来自 Surface 4
        
        **Validates: Requirements 9.4**
        """
        mirror_elements = [
            ce for ce in converted_elements
            if isinstance(ce.element, (FlatMirror, ParabolicMirror, SphericalMirror))
        ]
        assert len(mirror_elements) > 0
        first_mirror = mirror_elements[0]
        assert first_mirror.zmx_surface_index == 4
        assert first_mirror.zmx_comment == "M1"
    
    def test_load_zmx_file_convenience(self, zmx_file_path: str):
        """测试便捷函数 load_zmx_file
        
        **Validates: Requirements 7.1**
        """
        elements = load_zmx_file(zmx_file_path)
        assert len(elements) == 5  # 5 个反射镜
        for elem in elements:
            assert isinstance(elem, FlatMirror)
    
    def test_load_zmx_and_generate_code_convenience(self, zmx_file_path: str):
        """测试便捷函数 load_zmx_and_generate_code
        
        **Validates: Requirements 7.1, 9.1**
        """
        elements, code = load_zmx_and_generate_code(zmx_file_path)
        
        # 验证元件
        assert len(elements) == 5
        
        # 验证代码
        assert "from gaussian_beam_simulation.optical_elements import" in code
        assert "FlatMirror" in code
        assert "is_fold=True" in code
    
    def test_generated_code_executable(self, zmx_file_path: str):
        """测试生成的代码可执行
        
        **Validates: Requirements 9.1**
        """
        elements, code = load_zmx_and_generate_code(zmx_file_path)
        
        # 尝试执行生成的代码
        exec_globals = {}
        try:
            exec(code, exec_globals)
        except Exception as e:
            pytest.fail(f"生成的代码执行失败: {e}")
        
        # 验证代码中定义了变量
        # 第一个反射镜应该有变量名 m1
        assert "m1" in exec_globals, "生成的代码应该定义变量 m1"


# =============================================================================
# Task 8.2: one_mirror_up_45deg.zmx 集成测试
# =============================================================================

class TestOneMirrorUp45DegIntegration:
    """one_mirror_up_45deg.zmx 集成测试
    
    验证需求:
    - Requirements 10.4: 单个 45 度折叠镜被正确识别
    """
    
    @pytest.fixture
    def zmx_file_path(self) -> str:
        """获取测试文件路径"""
        return get_zmx_file_path("one_mirror_up_45deg.zmx")
    
    @pytest.fixture
    def parsed_data(self, zmx_file_path: str) -> ZmxDataModel:
        """解析 ZMX 文件"""
        parser = ZmxParser(zmx_file_path)
        return parser.parse()
    
    @pytest.fixture
    def converted_elements(self, parsed_data: ZmxDataModel) -> list:
        """转换为光学元件"""
        converter = ElementConverter(parsed_data)
        converter.convert()
        return converter.get_converted_elements()
    
    def test_file_exists(self, zmx_file_path: str):
        """测试文件存在"""
        assert os.path.exists(zmx_file_path), f"测试文件不存在: {zmx_file_path}"
    
    def test_parse_success(self, parsed_data: ZmxDataModel):
        """测试解析成功
        
        **Validates: Requirements 1.1, 1.2**
        """
        assert parsed_data is not None
        assert len(parsed_data.surfaces) > 0
    
    def test_single_mirror(self, parsed_data: ZmxDataModel):
        """测试单个反射镜
        
        **Validates: Requirements 10.4**
        """
        mirrors = parsed_data.get_mirror_surfaces()
        assert len(mirrors) == 1, f"期望 1 个反射镜，实际 {len(mirrors)} 个"
    
    def test_converted_single_element(self, converted_elements: list):
        """测试转换后单个元件
        
        **Validates: Requirements 10.4**
        """
        mirror_elements = [
            ce for ce in converted_elements
            if isinstance(ce.element, (FlatMirror, ParabolicMirror, SphericalMirror))
        ]
        assert len(mirror_elements) == 1, f"期望 1 个反射镜元件，实际 {len(mirror_elements)} 个"
    
    def test_45_degree_tilt(self, converted_elements: list):
        """测试 45 度倾斜
        
        tilt_x 或 tilt_y 应该是 π/4 (45°)
        
        **Validates: Requirements 10.4**
        """
        mirror_elements = [
            ce for ce in converted_elements
            if isinstance(ce.element, (FlatMirror, ParabolicMirror, SphericalMirror))
        ]
        assert len(mirror_elements) == 1
        
        mirror = mirror_elements[0]
        tilt_x = mirror.element.tilt_x
        tilt_y = mirror.element.tilt_y
        max_tilt = max(abs(tilt_x), abs(tilt_y))
        expected_tilt = np.pi / 4  # 45 度
        
        assert abs(max_tilt - expected_tilt) < 0.01, \
            f"倾斜角度应该是 π/4 (45°)，实际 {max_tilt} rad ({np.rad2deg(max_tilt)}°)"
    
    def test_is_fold_mirror(self, converted_elements: list):
        """测试折叠镜标志
        
        **Validates: Requirements 10.4, 5.7**
        """
        mirror_elements = [
            ce for ce in converted_elements
            if isinstance(ce.element, (FlatMirror, ParabolicMirror, SphericalMirror))
        ]
        assert len(mirror_elements) == 1
        
        mirror = mirror_elements[0]
        assert mirror.is_fold_mirror, "应该被识别为折叠镜"
        assert mirror.element.is_fold, "is_fold 应该为 True"
        assert abs(mirror.fold_angle_deg - 45.0) < 0.01, \
            f"折叠角度应该是 45°，实际 {mirror.fold_angle_deg}°"


# =============================================================================
# Task 8.3: 所有 ZMX 文件的有效性测试
# =============================================================================

class TestAllZmxFilesValidity:
    """所有 ZMX 文件的有效性测试
    
    验证需求:
    - Requirements 10.5: 所有 ZMX 文件都能成功解析
    """
    
    @pytest.fixture
    def all_zmx_files(self) -> list:
        """获取所有 ZMX 测试文件"""
        if not ZMX_FILES_DIR.exists():
            pytest.skip(f"ZMX 文件目录不存在: {ZMX_FILES_DIR}")
        
        zmx_files = list(ZMX_FILES_DIR.glob("*.zmx"))
        if not zmx_files:
            pytest.skip("没有找到 ZMX 测试文件")
        
        return zmx_files
    
    def test_all_files_parseable(self, all_zmx_files: list):
        """测试所有文件都能解析
        
        注意：包含不支持表面类型的文件会被跳过（如 DGRATING, TOROIDAL）
        
        **Validates: Requirements 10.5**
        """
        failed_files = []
        skipped_files = []
        
        for zmx_file in all_zmx_files:
            try:
                parser = ZmxParser(str(zmx_file))
                data_model = parser.parse()
                assert data_model is not None
                assert len(data_model.surfaces) > 0
            except ZmxUnsupportedError as e:
                # 不支持的表面类型，跳过
                skipped_files.append((zmx_file.name, str(e)))
            except Exception as e:
                failed_files.append((zmx_file.name, str(e)))
        
        if skipped_files:
            print(f"\n跳过 {len(skipped_files)} 个包含不支持表面类型的文件")
        
        if failed_files:
            error_msg = "以下文件解析失败:\n"
            for filename, error in failed_files:
                error_msg += f"  - {filename}: {error}\n"
            pytest.fail(error_msg)
    
    def test_all_files_convertible(self, all_zmx_files: list):
        """测试所有文件都能转换
        
        注意：包含不支持表面类型的文件会被跳过（如 DGRATING, TOROIDAL）
        
        **Validates: Requirements 10.5**
        """
        failed_files = []
        skipped_files = []
        
        for zmx_file in all_zmx_files:
            try:
                parser = ZmxParser(str(zmx_file))
                data_model = parser.parse()
                converter = ElementConverter(data_model)
                elements = converter.convert()
                # 元件列表可以为空（如果没有反射镜）
                assert isinstance(elements, list)
            except ZmxUnsupportedError as e:
                # 不支持的表面类型，跳过
                skipped_files.append((zmx_file.name, str(e)))
            except Exception as e:
                failed_files.append((zmx_file.name, str(e)))
        
        if skipped_files:
            print(f"\n跳过 {len(skipped_files)} 个包含不支持表面类型的文件")
        
        if failed_files:
            error_msg = "以下文件转换失败:\n"
            for filename, error in failed_files:
                error_msg += f"  - {filename}: {error}\n"
            pytest.fail(error_msg)
    
    def test_all_files_code_generation(self, all_zmx_files: list):
        """测试所有文件都能生成代码
        
        注意：包含不支持表面类型的文件会被跳过（如 DGRATING, TOROIDAL）
        
        **Validates: Requirements 10.5, 9.1**
        """
        failed_files = []
        skipped_files = []
        
        for zmx_file in all_zmx_files:
            try:
                parser = ZmxParser(str(zmx_file))
                data_model = parser.parse()
                converter = ElementConverter(data_model)
                converter.convert()
                code = converter.generate_code()
                assert isinstance(code, str)
                
                # 尝试执行生成的代码
                exec_globals = {}
                exec(code, exec_globals)
            except ZmxUnsupportedError as e:
                # 不支持的表面类型，跳过
                skipped_files.append((zmx_file.name, str(e)))
            except Exception as e:
                failed_files.append((zmx_file.name, str(e)))
        
        if skipped_files:
            print(f"\n跳过 {len(skipped_files)} 个包含不支持表面类型的文件")
        
        if failed_files:
            error_msg = "以下文件代码生成或执行失败:\n"
            for filename, error in failed_files:
                error_msg += f"  - {filename}: {error}\n"
            pytest.fail(error_msg)
    
    @pytest.mark.parametrize("zmx_filename", [
        "complicated_fold_mirrors_setup_v2.zmx",
        "one_mirror_up_45deg.zmx",
    ])
    def test_specific_files(self, zmx_filename: str):
        """测试特定文件
        
        **Validates: Requirements 10.5**
        """
        zmx_file_path = get_zmx_file_path(zmx_filename)
        
        if not os.path.exists(zmx_file_path):
            pytest.skip(f"文件不存在: {zmx_file_path}")
        
        # 解析
        parser = ZmxParser(zmx_file_path)
        data_model = parser.parse()
        assert data_model is not None
        
        # 转换
        converter = ElementConverter(data_model)
        elements = converter.convert()
        assert isinstance(elements, list)
        
        # 生成代码
        code = converter.generate_code()
        assert isinstance(code, str)
        
        # 执行代码
        exec_globals = {}
        exec(code, exec_globals)


# =============================================================================
# 运行测试
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
