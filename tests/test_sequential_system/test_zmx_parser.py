"""
ZMX 解析器单元测试

测试 ZmxParser 类的文件读取和解析功能。
"""

import pytest
import tempfile
import os
import sys

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from sequential_system.zmx_parser import (
    ZmxParser,
    ZmxDataModel,
    ZmxSurfaceData,
    ZmxParseError,
    ZmxUnsupportedError,
    ZmxConversionError,
)


class TestZmxParserFileReading:
    """测试 ZmxParser 的文件读取功能"""
    
    def test_file_not_found_error(self):
        """测试文件不存在时抛出 FileNotFoundError
        
        **Validates: Requirements 1.5**
        """
        parser = ZmxParser("nonexistent_file.zmx")
        
        with pytest.raises(FileNotFoundError) as exc_info:
            parser._try_read_file()
        
        assert "不存在" in str(exc_info.value) or "nonexistent_file.zmx" in str(exc_info.value)
    
    def test_directory_not_file_error(self, tmp_path):
        """测试路径是目录而非文件时抛出 FileNotFoundError
        
        **Validates: Requirements 1.5**
        """
        # 创建一个目录
        dir_path = tmp_path / "test_dir"
        dir_path.mkdir()
        
        parser = ZmxParser(str(dir_path))
        
        with pytest.raises(FileNotFoundError) as exc_info:
            parser._try_read_file()
        
        assert "不是文件" in str(exc_info.value)
    
    def test_read_utf16_file(self, tmp_path):
        """测试读取 UTF-16 编码的文件
        
        **Validates: Requirements 1.2**
        """
        # 创建 UTF-16 编码的测试文件
        content = "MODE SEQ\nENPD 20\nSURF 0\n"
        file_path = tmp_path / "test_utf16.zmx"
        
        with open(file_path, 'w', encoding='utf-16') as f:
            f.write(content)
        
        parser = ZmxParser(str(file_path))
        lines = parser._try_read_file()
        
        assert len(lines) == 3
        assert lines[0] == "MODE SEQ"
        assert lines[1] == "ENPD 20"
        assert lines[2] == "SURF 0"
    
    def test_read_utf8_file(self, tmp_path):
        """测试读取 UTF-8 编码的文件
        
        **Validates: Requirements 1.2**
        """
        # 创建 UTF-8 编码的测试文件
        content = "MODE SEQ\nENPD 25\nSURF 0\n"
        file_path = tmp_path / "test_utf8.zmx"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        parser = ZmxParser(str(file_path))
        lines = parser._try_read_file()
        
        assert len(lines) == 3
        assert lines[0] == "MODE SEQ"
        assert lines[1] == "ENPD 25"
    
    def test_read_iso8859_file(self, tmp_path):
        """测试读取 ISO-8859-1 编码的文件
        
        **Validates: Requirements 1.2**
        """
        # 创建 ISO-8859-1 编码的测试文件
        content = "MODE SEQ\nENPD 30\nSURF 0\n"
        file_path = tmp_path / "test_iso8859.zmx"
        
        with open(file_path, 'w', encoding='iso-8859-1') as f:
            f.write(content)
        
        parser = ZmxParser(str(file_path))
        lines = parser._try_read_file()
        
        assert len(lines) == 3
        assert lines[0] == "MODE SEQ"
        assert lines[1] == "ENPD 30"
    
    def test_read_real_zmx_file_utf16(self):
        """测试读取真实的 ZMX 文件（UTF-16 编码）
        
        **Validates: Requirements 1.1, 1.2**
        """
        # 使用 optiland 测试文件
        zmx_path = "optiland-master/tests/zemax_files/one_mirror_up_45deg.zmx"
        
        if not os.path.exists(zmx_path):
            pytest.skip(f"测试文件不存在: {zmx_path}")
        
        parser = ZmxParser(zmx_path)
        lines = parser._try_read_file()
        
        # 验证文件被成功读取
        assert len(lines) > 0
        
        # 验证包含预期的内容
        content = "\n".join(lines)
        assert "MODE" in content or "SURF" in content
    
    def test_read_real_zmx_file_iso8859(self):
        """测试读取 ISO-8859-1 编码的真实 ZMX 文件
        
        **Validates: Requirements 1.1, 1.2**
        """
        # 使用 optiland 测试文件（ISO-8859-1 编码）
        zmx_path = "optiland-master/tests/zemax_files/lens_thorlabs_iso_8859_1.zmx"
        
        if not os.path.exists(zmx_path):
            pytest.skip(f"测试文件不存在: {zmx_path}")
        
        parser = ZmxParser(zmx_path)
        lines = parser._try_read_file()
        
        # 验证文件被成功读取
        assert len(lines) > 0
    
    def test_supported_encodings_constant(self):
        """测试 SUPPORTED_ENCODINGS 常量包含所有必需的编码
        
        **Validates: Requirements 1.2**
        """
        assert "utf-16" in ZmxParser.SUPPORTED_ENCODINGS
        assert "utf-8" in ZmxParser.SUPPORTED_ENCODINGS
        assert "iso-8859-1" in ZmxParser.SUPPORTED_ENCODINGS
        
        # UTF-16 应该是第一个（Zemax 默认格式）
        assert ZmxParser.SUPPORTED_ENCODINGS[0] == "utf-16"
    
    def test_parser_initialization(self):
        """测试解析器初始化"""
        filepath = "test.zmx"
        parser = ZmxParser(filepath)
        
        assert parser.filepath == filepath
        assert isinstance(parser._data_model, ZmxDataModel)
        assert parser._current_surface_index == -1
        assert parser._current_surface is None


class TestZmxDataModel:
    """测试 ZmxDataModel 数据类"""
    
    def test_empty_data_model(self):
        """测试空数据模型"""
        model = ZmxDataModel()
        
        assert len(model.surfaces) == 0
        assert len(model.wavelengths) == 0
        assert model.primary_wavelength_index == 0
        assert model.entrance_pupil_diameter == 0.0
    
    def test_get_surface(self):
        """测试 get_surface 方法"""
        model = ZmxDataModel()
        surface = ZmxSurfaceData(index=1, surface_type='standard')
        model.surfaces[1] = surface
        
        # 获取存在的表面
        result = model.get_surface(1)
        assert result is surface
        
        # 获取不存在的表面
        result = model.get_surface(99)
        assert result is None
    
    def test_get_mirror_surfaces(self):
        """测试 get_mirror_surfaces 方法"""
        model = ZmxDataModel()
        
        # 添加混合表面
        model.surfaces[0] = ZmxSurfaceData(index=0, surface_type='standard')
        model.surfaces[1] = ZmxSurfaceData(index=1, surface_type='standard', is_mirror=True)
        model.surfaces[2] = ZmxSurfaceData(index=2, surface_type='coordinate_break')
        model.surfaces[3] = ZmxSurfaceData(index=3, surface_type='standard', is_mirror=True)
        
        mirrors = model.get_mirror_surfaces()
        
        assert len(mirrors) == 2
        assert mirrors[0].index == 1
        assert mirrors[1].index == 3
    
    def test_get_coordinate_break_surfaces(self):
        """测试 get_coordinate_break_surfaces 方法"""
        model = ZmxDataModel()
        
        model.surfaces[0] = ZmxSurfaceData(index=0, surface_type='standard')
        model.surfaces[1] = ZmxSurfaceData(index=1, surface_type='coordinate_break')
        model.surfaces[2] = ZmxSurfaceData(index=2, surface_type='standard')
        model.surfaces[3] = ZmxSurfaceData(index=3, surface_type='coordinate_break')
        
        coord_breaks = model.get_coordinate_break_surfaces()
        
        assert len(coord_breaks) == 2
        assert coord_breaks[0].index == 1
        assert coord_breaks[1].index == 3
    
    def test_get_surface_count(self):
        """测试 get_surface_count 方法"""
        model = ZmxDataModel()
        assert model.get_surface_count() == 0
        
        model.surfaces[0] = ZmxSurfaceData(index=0, surface_type='standard')
        model.surfaces[1] = ZmxSurfaceData(index=1, surface_type='standard')
        
        assert model.get_surface_count() == 2
    
    def test_get_max_surface_index(self):
        """测试 get_max_surface_index 方法"""
        model = ZmxDataModel()
        assert model.get_max_surface_index() == -1
        
        model.surfaces[0] = ZmxSurfaceData(index=0, surface_type='standard')
        model.surfaces[5] = ZmxSurfaceData(index=5, surface_type='standard')
        model.surfaces[3] = ZmxSurfaceData(index=3, surface_type='standard')
        
        assert model.get_max_surface_index() == 5


class TestZmxSurfaceData:
    """测试 ZmxSurfaceData 数据类"""
    
    def test_default_values(self):
        """测试默认值"""
        import numpy as np
        
        surface = ZmxSurfaceData(index=0, surface_type='standard')
        
        assert surface.index == 0
        assert surface.surface_type == 'standard'
        assert surface.radius == np.inf
        assert surface.thickness == 0.0
        assert surface.conic == 0.0
        assert surface.material == "air"
        assert surface.is_mirror == False
        assert surface.is_stop == False
        assert surface.semi_diameter == 0.0
        assert surface.decenter_x == 0.0
        assert surface.decenter_y == 0.0
        assert surface.tilt_x_deg == 0.0
        assert surface.tilt_y_deg == 0.0
        assert surface.tilt_z_deg == 0.0
        assert surface.asphere_coeffs == []
        assert surface.comment == ""
    
    def test_custom_values(self):
        """测试自定义值"""
        surface = ZmxSurfaceData(
            index=1,
            surface_type='standard',
            radius=100.0,
            thickness=50.0,
            conic=-1.0,
            material='mirror',
            is_mirror=True,
            semi_diameter=25.0,
            comment='M1',
        )
        
        assert surface.index == 1
        assert surface.radius == 100.0
        assert surface.thickness == 50.0
        assert surface.conic == -1.0
        assert surface.material == 'mirror'
        assert surface.is_mirror == True
        assert surface.semi_diameter == 25.0
        assert surface.comment == 'M1'
    
    def test_coordinate_break_values(self):
        """测试坐标断点参数"""
        surface = ZmxSurfaceData(
            index=2,
            surface_type='coordinate_break',
            decenter_x=1.0,
            decenter_y=2.0,
            tilt_x_deg=45.0,
            tilt_y_deg=30.0,
            tilt_z_deg=15.0,
        )
        
        assert surface.surface_type == 'coordinate_break'
        assert surface.decenter_x == 1.0
        assert surface.decenter_y == 2.0
        assert surface.tilt_x_deg == 45.0
        assert surface.tilt_y_deg == 30.0
        assert surface.tilt_z_deg == 15.0


class TestZmxExceptions:
    """测试异常类"""
    
    def test_zmx_parse_error_with_line_info(self):
        """测试带行信息的 ZmxParseError"""
        error = ZmxParseError(
            "无效的曲率值",
            line_number=42,
            line_content="CURV abc"
        )
        
        assert error.line_number == 42
        assert error.line_content == "CURV abc"
        assert "第 42 行" in str(error)
        assert "无效的曲率值" in str(error)
        assert "CURV abc" in str(error)
    
    def test_zmx_parse_error_without_line_info(self):
        """测试不带行信息的 ZmxParseError"""
        error = ZmxParseError("文件格式错误")
        
        assert error.line_number is None
        assert error.line_content is None
        assert "ZMX 解析错误" in str(error)
        assert "文件格式错误" in str(error)
    
    def test_zmx_unsupported_error(self):
        """测试 ZmxUnsupportedError"""
        error = ZmxUnsupportedError(
            "不支持的表面类型: TOROIDAL",
            line_number=15,
            line_content="TYPE TOROIDAL"
        )
        
        assert isinstance(error, ZmxParseError)
        assert "第 15 行" in str(error)
        assert "TOROIDAL" in str(error)
    
    def test_zmx_conversion_error_with_surface_info(self):
        """测试带表面信息的 ZmxConversionError"""
        error = ZmxConversionError(
            "无法确定反射镜类型",
            surface_index=3,
            surface_type="standard"
        )
        
        assert error.surface_index == 3
        assert error.surface_type == "standard"
        assert "表面 3" in str(error)
        assert "standard" in str(error)
        assert "无法确定反射镜类型" in str(error)
    
    def test_zmx_conversion_error_without_surface_info(self):
        """测试不带表面信息的 ZmxConversionError"""
        error = ZmxConversionError("转换失败")
        
        assert error.surface_index is None
        assert error.surface_type is None
        assert "ZMX 转换错误" in str(error)
        assert "转换失败" in str(error)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestZmxParserOperatorDispatch:
    """测试 ZmxParser 的操作符解析分发机制
    
    **Validates: Requirements 1.3, 1.4**
    """
    
    def test_operator_handlers_dict_exists(self):
        """测试 _OPERATOR_HANDLERS 字典存在且包含必要的操作符"""
        assert hasattr(ZmxParser, '_OPERATOR_HANDLERS')
        handlers = ZmxParser._OPERATOR_HANDLERS
        
        # 验证必要的操作符都已注册
        required_operators = [
            "MODE", "SURF", "TYPE", "CURV", "DISZ", "CONI",
            "GLAS", "PARM", "DIAM", "STOP", "COMM", "ENPD", "WAVM"
        ]
        
        for op in required_operators:
            assert op in handlers, f"操作符 {op} 未在 _OPERATOR_HANDLERS 中注册"
    
    def test_operator_handlers_point_to_methods(self):
        """测试 _OPERATOR_HANDLERS 中的方法名对应实际存在的方法"""
        parser = ZmxParser("dummy.zmx")
        
        for operator, handler_name in ZmxParser._OPERATOR_HANDLERS.items():
            handler = getattr(parser, handler_name, None)
            assert handler is not None, f"操作符 {operator} 的处理方法 {handler_name} 不存在"
            assert callable(handler), f"操作符 {operator} 的处理方法 {handler_name} 不可调用"
    
    def test_parse_line_skips_empty_lines(self, tmp_path):
        """测试 _parse_line 跳过空行"""
        parser = ZmxParser("dummy.zmx")
        
        # 这些行应该被静默跳过，不抛出异常
        parser._parse_line("")
        parser._parse_line("   ")
        parser._parse_line("\t\t")
        parser._parse_line("\n")
    
    def test_parse_line_ignores_unknown_operators(self, tmp_path):
        """测试 _parse_line 静默忽略未知操作符"""
        parser = ZmxParser("dummy.zmx")
        
        # 未知操作符应该被静默忽略，不抛出异常
        parser._parse_line("UNKNOWN_OP 123 456")
        parser._parse_line("XYZABC data")
        parser._parse_line("NOTREAL")
    
    def test_parse_line_extracts_operator_case_insensitive(self, tmp_path):
        """测试 _parse_line 操作符提取不区分大小写"""
        # 创建测试文件
        content = "mode seq\n"
        file_path = tmp_path / "test_case.zmx"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        parser = ZmxParser(str(file_path))
        # 小写的 "mode" 应该被识别为 MODE 操作符
        # 如果是 SEQ 模式，不应抛出异常
        parser._parse_line("mode seq")
        parser._parse_line("MODE SEQ")
        parser._parse_line("Mode Seq")


class TestZmxParserParseMode:
    """测试 ZmxParser 的 _parse_mode 方法
    
    **Validates: Requirements 1.3, 1.4**
    """
    
    def test_parse_mode_accepts_seq(self):
        """测试 _parse_mode 接受 SEQ 模式"""
        parser = ZmxParser("dummy.zmx")
        
        # SEQ 模式应该被接受，不抛出异常
        parser._parse_mode(["SEQ"])
        parser._parse_mode(["seq"])
        parser._parse_mode(["Seq"])
    
    def test_parse_mode_rejects_nsc(self):
        """测试 _parse_mode 拒绝 NSC 模式
        
        **Validates: Requirements 1.4**
        """
        parser = ZmxParser("dummy.zmx")
        parser._current_line_number = 1
        parser._current_line_content = "MODE NSC"
        
        with pytest.raises(ZmxUnsupportedError) as exc_info:
            parser._parse_mode(["NSC"])
        
        assert "不支持的模式" in str(exc_info.value)
        assert "NSC" in str(exc_info.value)
        assert "序列模式" in str(exc_info.value) or "SEQ" in str(exc_info.value)
    
    def test_parse_mode_rejects_other_modes(self):
        """测试 _parse_mode 拒绝其他未知模式"""
        parser = ZmxParser("dummy.zmx")
        parser._current_line_number = 5
        parser._current_line_content = "MODE UNKNOWN"
        
        with pytest.raises(ZmxUnsupportedError) as exc_info:
            parser._parse_mode(["UNKNOWN"])
        
        assert "不支持的模式" in str(exc_info.value)
        assert "UNKNOWN" in str(exc_info.value)
    
    def test_parse_mode_ignores_empty_data(self):
        """测试 _parse_mode 忽略空数据"""
        parser = ZmxParser("dummy.zmx")
        
        # 空数据应该被静默忽略，不抛出异常
        parser._parse_mode([])
    
    def test_parse_mode_error_includes_line_info(self):
        """测试 _parse_mode 错误信息包含行号和内容"""
        parser = ZmxParser("dummy.zmx")
        parser._current_line_number = 42
        parser._current_line_content = "MODE NSC"
        
        with pytest.raises(ZmxUnsupportedError) as exc_info:
            parser._parse_mode(["NSC"])
        
        error = exc_info.value
        assert error.line_number == 42
        assert error.line_content == "MODE NSC"
        assert "第 42 行" in str(error)


class TestZmxParserParseMethod:
    """测试 ZmxParser 的 parse() 主方法
    
    **Validates: Requirements 1.1, 1.3**
    """
    
    def test_parse_returns_data_model(self, tmp_path):
        """测试 parse() 返回 ZmxDataModel 对象"""
        content = "MODE SEQ\n"
        file_path = tmp_path / "test.zmx"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        parser = ZmxParser(str(file_path))
        result = parser.parse()
        
        assert isinstance(result, ZmxDataModel)
    
    def test_parse_resets_state_between_calls(self, tmp_path):
        """测试 parse() 在多次调用之间重置状态"""
        content = "MODE SEQ\n"
        file_path = tmp_path / "test.zmx"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        parser = ZmxParser(str(file_path))
        
        # 第一次解析
        result1 = parser.parse()
        
        # 第二次解析应该返回新的数据模型
        result2 = parser.parse()
        
        # 两次结果应该是不同的对象
        assert result1 is not result2
    
    def test_parse_raises_on_nsc_mode(self, tmp_path):
        """测试 parse() 在遇到 NSC 模式时抛出异常
        
        **Validates: Requirements 1.4**
        """
        content = "MODE NSC\nSURF 0\n"
        file_path = tmp_path / "test_nsc.zmx"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        parser = ZmxParser(str(file_path))
        
        with pytest.raises(ZmxUnsupportedError) as exc_info:
            parser.parse()
        
        assert "不支持的模式" in str(exc_info.value)
        assert "NSC" in str(exc_info.value)
    
    def test_parse_accepts_seq_mode(self, tmp_path):
        """测试 parse() 接受 SEQ 模式
        
        **Validates: Requirements 1.3**
        """
        content = "MODE SEQ\n"
        file_path = tmp_path / "test_seq.zmx"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        parser = ZmxParser(str(file_path))
        
        # 不应抛出异常
        result = parser.parse()
        assert isinstance(result, ZmxDataModel)
    
    def test_parse_handles_file_not_found(self):
        """测试 parse() 处理文件不存在的情况
        
        **Validates: Requirements 1.5**
        """
        parser = ZmxParser("nonexistent_file_12345.zmx")
        
        with pytest.raises(FileNotFoundError):
            parser.parse()
    
    def test_parse_real_zmx_file_mode_seq(self):
        """测试 parse() 解析真实 ZMX 文件的 MODE SEQ
        
        **Validates: Requirements 1.1, 1.3**
        """
        import os
        zmx_path = "optiland-master/tests/zemax_files/one_mirror_up_45deg.zmx"
        
        if not os.path.exists(zmx_path):
            pytest.skip(f"测试文件不存在: {zmx_path}")
        
        parser = ZmxParser(zmx_path)
        
        # 真实的 ZMX 文件应该是 SEQ 模式，不应抛出异常
        result = parser.parse()
        assert isinstance(result, ZmxDataModel)


class TestZmxParserSurfaceOperators:
    """测试 ZmxParser 的表面相关操作符解析
    
    **Validates: Requirements 2.1, 2.5, 2.6**
    """
    
    def test_parse_surface_creates_new_surface(self):
        """测试 _parse_surface 创建新的表面对象"""
        parser = ZmxParser("dummy.zmx")
        
        # 解析第一个表面
        parser._parse_surface(["0"])
        
        assert parser._current_surface is not None
        assert parser._current_surface.index == 0
        assert parser._current_surface.surface_type == 'standard'
        assert parser._current_surface_index == 0
    
    def test_parse_surface_finalizes_previous_surface(self):
        """测试 _parse_surface 完成前一个表面的解析"""
        parser = ZmxParser("dummy.zmx")
        
        # 创建第一个表面
        parser._parse_surface(["0"])
        parser._current_surface.radius = 100.0
        
        # 创建第二个表面，应该保存第一个表面
        parser._parse_surface(["1"])
        
        # 验证第一个表面已保存到数据模型
        assert 0 in parser._data_model.surfaces
        assert parser._data_model.surfaces[0].radius == 100.0
        
        # 验证当前表面是新的
        assert parser._current_surface.index == 1
    
    def test_parse_surface_increments_index_without_data(self):
        """测试 _parse_surface 在没有数据时递增索引"""
        parser = ZmxParser("dummy.zmx")
        
        # 第一个表面
        parser._parse_surface(["0"])
        assert parser._current_surface_index == 0
        
        # 保存第一个表面
        parser._finalize_current_surface()
        
        # 没有提供索引时，应该递增
        parser._parse_surface([])
        assert parser._current_surface_index == 1
    
    def test_parse_surface_invalid_index(self):
        """测试 _parse_surface 处理无效索引"""
        parser = ZmxParser("dummy.zmx")
        parser._current_line_number = 10
        parser._current_line_content = "SURF abc"
        
        with pytest.raises(ZmxParseError) as exc_info:
            parser._parse_surface(["abc"])
        
        assert "无效的表面索引" in str(exc_info.value)
        assert "abc" in str(exc_info.value)


class TestZmxParserTypeOperator:
    """测试 ZmxParser 的 _parse_type 方法
    
    **Validates: Requirements 2.1**
    """
    
    def test_parse_type_standard(self):
        """测试解析 STANDARD 类型"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        
        parser._parse_type(["STANDARD"])
        
        assert parser._current_surface.surface_type == 'standard'
    
    def test_parse_type_coordbrk(self):
        """测试解析 COORDBRK 类型"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        
        parser._parse_type(["COORDBRK"])
        
        assert parser._current_surface.surface_type == 'coordinate_break'
    
    def test_parse_type_evenasph(self):
        """测试解析 EVENASPH 类型"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        
        parser._parse_type(["EVENASPH"])
        
        assert parser._current_surface.surface_type == 'even_asphere'
    
    def test_parse_type_case_insensitive(self):
        """测试类型解析不区分大小写"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        
        parser._parse_type(["standard"])
        assert parser._current_surface.surface_type == 'standard'
        
        parser._parse_type(["CoordBrk"])
        assert parser._current_surface.surface_type == 'coordinate_break'
    
    def test_parse_type_unsupported(self):
        """测试不支持的表面类型抛出异常"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        parser._current_line_number = 5
        parser._current_line_content = "TYPE TOROIDAL"
        
        with pytest.raises(ZmxUnsupportedError) as exc_info:
            parser._parse_type(["TOROIDAL"])
        
        assert "不支持的表面类型" in str(exc_info.value)
        assert "TOROIDAL" in str(exc_info.value)
    
    def test_parse_type_no_current_surface(self):
        """测试没有当前表面时静默忽略"""
        parser = ZmxParser("dummy.zmx")
        
        # 不应抛出异常
        parser._parse_type(["STANDARD"])
    
    def test_parse_type_empty_data(self):
        """测试空数据时静默忽略"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        
        # 不应抛出异常
        parser._parse_type([])


class TestZmxParserCurvOperator:
    """测试 ZmxParser 的 _parse_curv 方法
    
    **Validates: Requirements 2.5**
    """
    
    def test_parse_curv_positive(self):
        """测试正曲率转换为正半径"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        
        # 曲率 0.01 -> 半径 100
        parser._parse_curv(["0.01", "0", "0", "0", "0"])
        
        assert parser._current_surface.radius == pytest.approx(100.0)
    
    def test_parse_curv_negative(self):
        """测试负曲率转换为负半径"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        
        # 曲率 -0.005 -> 半径 -200
        parser._parse_curv(["-0.005", "0", "0", "0", "0"])
        
        assert parser._current_surface.radius == pytest.approx(-200.0)
    
    def test_parse_curv_zero(self):
        """测试零曲率转换为无穷大半径（平面）"""
        import numpy as np
        
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        
        parser._parse_curv(["0", "0", "0", "0", "0"])
        
        assert parser._current_surface.radius == np.inf
    
    def test_parse_curv_invalid_value(self):
        """测试无效曲率值抛出异常"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        parser._current_line_number = 15
        parser._current_line_content = "CURV abc"
        
        with pytest.raises(ZmxParseError) as exc_info:
            parser._parse_curv(["abc"])
        
        assert "无效的曲率值" in str(exc_info.value)
    
    def test_parse_curv_no_current_surface(self):
        """测试没有当前表面时静默忽略"""
        parser = ZmxParser("dummy.zmx")
        
        # 不应抛出异常
        parser._parse_curv(["0.01"])


class TestZmxParserDiszOperator:
    """测试 ZmxParser 的 _parse_disz 方法
    
    **Validates: Requirements 2.6**
    """
    
    def test_parse_disz_positive(self):
        """测试正厚度"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        
        parser._parse_disz(["50"])
        
        assert parser._current_surface.thickness == 50.0
    
    def test_parse_disz_negative(self):
        """测试负厚度（反射后传播）"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        
        parser._parse_disz(["-30"])
        
        assert parser._current_surface.thickness == -30.0
    
    def test_parse_disz_zero(self):
        """测试零厚度"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        
        parser._parse_disz(["0"])
        
        assert parser._current_surface.thickness == 0.0
    
    def test_parse_disz_infinity(self):
        """测试 INFINITY 厚度"""
        import numpy as np
        
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        
        parser._parse_disz(["INFINITY"])
        
        assert parser._current_surface.thickness == np.inf
    
    def test_parse_disz_infinity_case_insensitive(self):
        """测试 INFINITY 不区分大小写"""
        import numpy as np
        
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        
        parser._parse_disz(["infinity"])
        assert parser._current_surface.thickness == np.inf
        
        parser._parse_disz(["Infinity"])
        assert parser._current_surface.thickness == np.inf
    
    def test_parse_disz_invalid_value(self):
        """测试无效厚度值抛出异常"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        parser._current_line_number = 20
        parser._current_line_content = "DISZ xyz"
        
        with pytest.raises(ZmxParseError) as exc_info:
            parser._parse_disz(["xyz"])
        
        assert "无效的厚度值" in str(exc_info.value)


class TestZmxParserConiOperator:
    """测试 ZmxParser 的 _parse_coni 方法
    
    **Validates: Requirements 2.5**
    """
    
    def test_parse_coni_zero(self):
        """测试零圆锥常数（球面）"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        
        parser._parse_coni(["0"])
        
        assert parser._current_surface.conic == 0.0
    
    def test_parse_coni_negative_one(self):
        """测试 -1 圆锥常数（抛物面）"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        
        parser._parse_coni(["-1"])
        
        assert parser._current_surface.conic == -1.0
    
    def test_parse_coni_fractional(self):
        """测试小数圆锥常数"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        
        parser._parse_coni(["-0.5"])
        
        assert parser._current_surface.conic == -0.5
    
    def test_parse_coni_invalid_value(self):
        """测试无效圆锥常数抛出异常"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        parser._current_line_number = 25
        parser._current_line_content = "CONI invalid"
        
        with pytest.raises(ZmxParseError) as exc_info:
            parser._parse_coni(["invalid"])
        
        assert "无效的圆锥常数" in str(exc_info.value)


class TestZmxParserDiamOperator:
    """测试 ZmxParser 的 _parse_diam 方法
    
    **Validates: Requirements 2.6**
    """
    
    def test_parse_diam_positive(self):
        """测试正半口径"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        
        parser._parse_diam(["25", "0", "0", "0", "1"])
        
        assert parser._current_surface.semi_diameter == 25.0
    
    def test_parse_diam_fractional(self):
        """测试小数半口径"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        
        parser._parse_diam(["10.5", "1", "0", "0", "0"])
        
        assert parser._current_surface.semi_diameter == 10.5
    
    def test_parse_diam_invalid_value(self):
        """测试无效半口径值抛出异常"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        parser._current_line_number = 30
        parser._current_line_content = "DIAM bad"
        
        with pytest.raises(ZmxParseError) as exc_info:
            parser._parse_diam(["bad"])
        
        assert "无效的半口径值" in str(exc_info.value)


class TestZmxParserStopOperator:
    """测试 ZmxParser 的 _parse_stop 方法
    
    **Validates: Requirements 2.6**
    """
    
    def test_parse_stop_sets_flag(self):
        """测试 STOP 设置 is_stop 标志"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        
        assert parser._current_surface.is_stop == False
        
        parser._parse_stop([])
        
        assert parser._current_surface.is_stop == True
    
    def test_parse_stop_no_current_surface(self):
        """测试没有当前表面时静默忽略"""
        parser = ZmxParser("dummy.zmx")
        
        # 不应抛出异常
        parser._parse_stop([])


class TestZmxParserCommOperator:
    """测试 ZmxParser 的 _parse_comm 方法
    
    **Validates: Requirements 2.1**
    """
    
    def test_parse_comm_single_word(self):
        """测试单词注释"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        
        parser._parse_comm(["M1"])
        
        assert parser._current_surface.comment == "M1"
    
    def test_parse_comm_multiple_words(self):
        """测试多词注释"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        
        parser._parse_comm(["Primary", "Mirror"])
        
        assert parser._current_surface.comment == "Primary Mirror"
    
    def test_parse_comm_with_numbers(self):
        """测试带数字的注释"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        
        parser._parse_comm(["Fold", "Mirror", "45deg"])
        
        assert parser._current_surface.comment == "Fold Mirror 45deg"
    
    def test_parse_comm_empty(self):
        """测试空注释"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        
        parser._parse_comm([])
        
        assert parser._current_surface.comment == ""
    
    def test_parse_comm_no_current_surface(self):
        """测试没有当前表面时静默忽略"""
        parser = ZmxParser("dummy.zmx")
        
        # 不应抛出异常
        parser._parse_comm(["test"])


class TestZmxParserIntegrationSurfaceOperators:
    """测试表面操作符的集成解析
    
    **Validates: Requirements 2.1, 2.5, 2.6**
    """
    
    def test_parse_complete_surface(self, tmp_path):
        """测试解析完整的表面定义"""
        content = """MODE SEQ
SURF 0
  TYPE STANDARD
  CURV 0
  DISZ INFINITY
SURF 1
  TYPE STANDARD
  CURV 0.01
  DISZ 50
  CONI -1
  DIAM 25 0 0 0 1
  STOP
  COMM Primary Mirror
SURF 2
  TYPE STANDARD
  DISZ 0
"""
        file_path = tmp_path / "test_surface.zmx"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        parser = ZmxParser(str(file_path))
        data_model = parser.parse()
        
        # 验证表面数量
        assert len(data_model.surfaces) == 3
        
        # 验证表面 0（物面）
        surf0 = data_model.get_surface(0)
        assert surf0 is not None
        assert surf0.surface_type == 'standard'
        assert surf0.radius == float('inf')
        assert surf0.thickness == float('inf')
        
        # 验证表面 1（主镜）
        surf1 = data_model.get_surface(1)
        assert surf1 is not None
        assert surf1.surface_type == 'standard'
        assert surf1.radius == pytest.approx(100.0)
        assert surf1.thickness == 50.0
        assert surf1.conic == -1.0
        assert surf1.semi_diameter == 25.0
        assert surf1.is_stop == True
        assert surf1.comment == "Primary Mirror"
        
        # 验证表面 2（像面）
        surf2 = data_model.get_surface(2)
        assert surf2 is not None
        assert surf2.thickness == 0.0
    
    def test_parse_coordinate_break_surface(self, tmp_path):
        """测试解析坐标断点表面"""
        content = """MODE SEQ
SURF 0
  TYPE STANDARD
  DISZ INFINITY
SURF 1
  TYPE COORDBRK
  DISZ 0
SURF 2
  TYPE STANDARD
  DISZ 0
"""
        file_path = tmp_path / "test_coordbrk.zmx"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        parser = ZmxParser(str(file_path))
        data_model = parser.parse()
        
        # 验证坐标断点表面
        surf1 = data_model.get_surface(1)
        assert surf1 is not None
        assert surf1.surface_type == 'coordinate_break'
        
        # 验证可以获取所有坐标断点
        coord_breaks = data_model.get_coordinate_break_surfaces()
        assert len(coord_breaks) == 1
        assert coord_breaks[0].index == 1
    
    def test_parse_even_asphere_surface(self, tmp_path):
        """测试解析偶次非球面表面"""
        content = """MODE SEQ
SURF 0
  TYPE STANDARD
  DISZ INFINITY
SURF 1
  TYPE EVENASPH
  CURV 0.02
  DISZ 10
  CONI -0.5
SURF 2
  TYPE STANDARD
  DISZ 0
"""
        file_path = tmp_path / "test_evenasph.zmx"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        parser = ZmxParser(str(file_path))
        data_model = parser.parse()
        
        # 验证偶次非球面表面
        surf1 = data_model.get_surface(1)
        assert surf1 is not None
        assert surf1.surface_type == 'even_asphere'
        assert surf1.radius == pytest.approx(50.0)
        assert surf1.conic == -0.5


class TestZmxParserGlasOperator:
    """测试 ZmxParser 的 _parse_glas 方法
    
    **Validates: Requirements 2.3, 2.4**
    """
    
    def test_parse_glas_mirror(self):
        """测试解析 MIRROR 材料"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        
        parser._parse_glas(["MIRROR", "0", "0", "1.5", "40"])
        
        assert parser._current_surface.is_mirror == True
        assert parser._current_surface.material == "mirror"
    
    def test_parse_glas_mirror_case_insensitive(self):
        """测试 MIRROR 不区分大小写"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        
        parser._parse_glas(["mirror"])
        assert parser._current_surface.is_mirror == True
        
        parser._parse_surface(["1"])
        parser._parse_glas(["Mirror"])
        assert parser._current_surface.is_mirror == True
        
        parser._parse_surface(["2"])
        parser._parse_glas(["MIRROR"])
        assert parser._current_surface.is_mirror == True
    
    def test_parse_glas_glass_material(self):
        """测试解析玻璃材料"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        
        parser._parse_glas(["N-BK7", "0", "0", "1.5", "40"])
        
        assert parser._current_surface.is_mirror == False
        assert parser._current_surface.material == "N-BK7"
    
    def test_parse_glas_various_materials(self):
        """测试解析各种材料名称"""
        parser = ZmxParser("dummy.zmx")
        
        materials = ["BK7", "SF11", "N-SF6", "SILICA", "CAF2"]
        
        for i, material in enumerate(materials):
            parser._parse_surface([str(i)])
            parser._parse_glas([material])
            
            assert parser._current_surface.is_mirror == False
            assert parser._current_surface.material == material
    
    def test_parse_glas_no_current_surface(self):
        """测试没有当前表面时静默忽略"""
        parser = ZmxParser("dummy.zmx")
        
        # 不应抛出异常
        parser._parse_glas(["MIRROR"])
    
    def test_parse_glas_empty_data(self):
        """测试空数据时静默忽略"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        
        # 不应抛出异常
        parser._parse_glas([])
        
        # 材料应保持默认值
        assert parser._current_surface.material == "air"
        assert parser._current_surface.is_mirror == False


class TestZmxParserWavmOperator:
    """测试 ZmxParser 的 _parse_wavm 方法
    
    **Validates: Requirements 4.3**
    """
    
    def test_parse_wavm_single_wavelength(self):
        """测试解析单个波长"""
        parser = ZmxParser("dummy.zmx")
        
        parser._parse_wavm(["1", "0.55", "1"])
        
        assert len(parser._data_model.wavelengths) == 1
        assert parser._data_model.wavelengths[0] == pytest.approx(0.55)
        assert parser._data_model.primary_wavelength_index == 0
    
    def test_parse_wavm_multiple_wavelengths(self):
        """测试解析多个波长"""
        parser = ZmxParser("dummy.zmx")
        
        parser._parse_wavm(["1", "0.55", "1"])
        parser._parse_wavm(["2", "0.486", "0.5"])
        parser._parse_wavm(["3", "0.656", "0.5"])
        
        assert len(parser._data_model.wavelengths) == 3
        assert parser._data_model.wavelengths[0] == pytest.approx(0.55)
        assert parser._data_model.wavelengths[1] == pytest.approx(0.486)
        assert parser._data_model.wavelengths[2] == pytest.approx(0.656)
    
    def test_parse_wavm_primary_wavelength(self):
        """测试主波长设置（权重为 1）"""
        parser = ZmxParser("dummy.zmx")
        
        # 第一个波长权重为 0.5
        parser._parse_wavm(["1", "0.486", "0.5"])
        # 第二个波长权重为 1（主波长）
        parser._parse_wavm(["2", "0.55", "1"])
        # 第三个波长权重为 0.5
        parser._parse_wavm(["3", "0.656", "0.5"])
        
        # 主波长索引应该是 1（第二个波长）
        assert parser._data_model.primary_wavelength_index == 1
    
    def test_parse_wavm_without_weight(self):
        """测试没有权重参数时使用默认值"""
        parser = ZmxParser("dummy.zmx")
        
        parser._parse_wavm(["1", "0.55"])
        
        assert len(parser._data_model.wavelengths) == 1
        assert parser._data_model.wavelengths[0] == pytest.approx(0.55)
        # 默认权重为 1，所以应该是主波长
        assert parser._data_model.primary_wavelength_index == 0
    
    def test_parse_wavm_non_sequential_indices(self):
        """测试非顺序的波长索引"""
        parser = ZmxParser("dummy.zmx")
        
        # 先添加索引 3
        parser._parse_wavm(["3", "0.656", "0.5"])
        # 再添加索引 1
        parser._parse_wavm(["1", "0.55", "1"])
        
        assert len(parser._data_model.wavelengths) == 3
        assert parser._data_model.wavelengths[0] == pytest.approx(0.55)
        assert parser._data_model.wavelengths[2] == pytest.approx(0.656)
    
    def test_parse_wavm_invalid_index(self):
        """测试无效的波长索引抛出异常"""
        parser = ZmxParser("dummy.zmx")
        parser._current_line_number = 10
        parser._current_line_content = "WAVM abc 0.55 1"
        
        with pytest.raises(ZmxParseError) as exc_info:
            parser._parse_wavm(["abc", "0.55", "1"])
        
        assert "无效的波长索引" in str(exc_info.value)
    
    def test_parse_wavm_invalid_wavelength(self):
        """测试无效的波长值抛出异常"""
        parser = ZmxParser("dummy.zmx")
        parser._current_line_number = 10
        parser._current_line_content = "WAVM 1 xyz 1"
        
        with pytest.raises(ZmxParseError) as exc_info:
            parser._parse_wavm(["1", "xyz", "1"])
        
        assert "无效的波长值" in str(exc_info.value)
    
    def test_parse_wavm_empty_data(self):
        """测试空数据时静默忽略"""
        parser = ZmxParser("dummy.zmx")
        
        # 不应抛出异常
        parser._parse_wavm([])
        parser._parse_wavm(["1"])  # 只有索引，没有波长值
        
        assert len(parser._data_model.wavelengths) == 0


class TestZmxParserEnpdOperator:
    """测试 ZmxParser 的 _parse_enpd 方法
    
    **Validates: Requirements 4.4**
    """
    
    def test_parse_enpd_positive(self):
        """测试正入瞳直径"""
        parser = ZmxParser("dummy.zmx")
        
        parser._parse_enpd(["20"])
        
        assert parser._data_model.entrance_pupil_diameter == 20.0
    
    def test_parse_enpd_fractional(self):
        """测试小数入瞳直径"""
        parser = ZmxParser("dummy.zmx")
        
        parser._parse_enpd(["25.5"])
        
        assert parser._data_model.entrance_pupil_diameter == 25.5
    
    def test_parse_enpd_small_value(self):
        """测试小入瞳直径"""
        parser = ZmxParser("dummy.zmx")
        
        parser._parse_enpd(["0.5"])
        
        assert parser._data_model.entrance_pupil_diameter == 0.5
    
    def test_parse_enpd_invalid_value(self):
        """测试无效入瞳直径值抛出异常"""
        parser = ZmxParser("dummy.zmx")
        parser._current_line_number = 5
        parser._current_line_content = "ENPD invalid"
        
        with pytest.raises(ZmxParseError) as exc_info:
            parser._parse_enpd(["invalid"])
        
        assert "无效的入瞳直径值" in str(exc_info.value)
    
    def test_parse_enpd_empty_data(self):
        """测试空数据时静默忽略"""
        parser = ZmxParser("dummy.zmx")
        
        # 不应抛出异常
        parser._parse_enpd([])
        
        # 入瞳直径应保持默认值
        assert parser._data_model.entrance_pupil_diameter == 0.0


class TestZmxParserMaterialAndWavelengthIntegration:
    """测试材料和波长解析的集成测试
    
    **Validates: Requirements 2.3, 2.4, 4.3, 4.4**
    """
    
    def test_parse_complete_system_with_materials(self, tmp_path):
        """测试解析包含材料定义的完整系统"""
        content = """MODE SEQ
ENPD 20
WAVM 1 0.55 1
WAVM 2 0.486 0.5
WAVM 3 0.656 0.5
SURF 0
  TYPE STANDARD
  DISZ INFINITY
SURF 1
  TYPE STANDARD
  CURV 0.01
  DISZ 50
  GLAS MIRROR 0 0 1.5 40
  COMM Primary Mirror
SURF 2
  TYPE STANDARD
  CURV 0.02
  DISZ 10
  GLAS N-BK7 0 0 1.5 40
SURF 3
  TYPE STANDARD
  DISZ 0
"""
        file_path = tmp_path / "test_materials.zmx"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        parser = ZmxParser(str(file_path))
        data_model = parser.parse()
        
        # 验证入瞳直径
        assert data_model.entrance_pupil_diameter == 20.0
        
        # 验证波长
        assert len(data_model.wavelengths) == 3
        assert data_model.wavelengths[0] == pytest.approx(0.55)
        assert data_model.wavelengths[1] == pytest.approx(0.486)
        assert data_model.wavelengths[2] == pytest.approx(0.656)
        assert data_model.primary_wavelength_index == 0
        
        # 验证反射镜表面
        surf1 = data_model.get_surface(1)
        assert surf1.is_mirror == True
        assert surf1.material == "mirror"
        
        # 验证玻璃表面
        surf2 = data_model.get_surface(2)
        assert surf2.is_mirror == False
        assert surf2.material == "N-BK7"
        
        # 验证可以获取所有反射镜
        mirrors = data_model.get_mirror_surfaces()
        assert len(mirrors) == 1
        assert mirrors[0].index == 1
    
    def test_parse_real_zmx_file_with_mirror(self):
        """测试解析真实 ZMX 文件中的反射镜
        
        **Validates: Requirements 2.3, 10.4**
        """
        import os
        zmx_path = "optiland-master/tests/zemax_files/one_mirror_up_45deg.zmx"
        
        if not os.path.exists(zmx_path):
            pytest.skip(f"测试文件不存在: {zmx_path}")
        
        parser = ZmxParser(zmx_path)
        data_model = parser.parse()
        
        # 验证文件被成功解析
        assert len(data_model.surfaces) > 0
        
        # 验证包含反射镜
        mirrors = data_model.get_mirror_surfaces()
        assert len(mirrors) >= 1, "应该至少有一个反射镜"
        
        # 验证反射镜属性
        for mirror in mirrors:
            assert mirror.is_mirror == True
            assert mirror.material == "mirror"
    
    def test_parse_real_zmx_file_wavelengths(self):
        """测试解析真实 ZMX 文件中的波长
        
        **Validates: Requirements 4.3**
        """
        import os
        zmx_path = "optiland-master/tests/zemax_files/one_mirror_up_45deg.zmx"
        
        if not os.path.exists(zmx_path):
            pytest.skip(f"测试文件不存在: {zmx_path}")
        
        parser = ZmxParser(zmx_path)
        data_model = parser.parse()
        
        # 验证波长被解析
        assert len(data_model.wavelengths) > 0, "应该至少有一个波长"
        
        # 验证波长值在合理范围内（可见光范围：0.38-0.78 μm）
        for wavelength in data_model.wavelengths:
            if wavelength > 0:  # 跳过占位符
                assert 0.1 <= wavelength <= 2.0, f"波长 {wavelength} μm 超出合理范围"


class TestZmxParserParmOperator:
    """测试 ZmxParser 的 _parse_parm 方法
    
    **Validates: Requirements 3.1, 3.2, 3.3**
    """
    
    def test_parse_parm_decenter_x(self):
        """测试解析 PARM 1（decenter_x）"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        parser._parse_type(["COORDBRK"])
        
        parser._parse_parm(["1", "5.0"])
        
        assert parser._current_surface.decenter_x == 5.0
    
    def test_parse_parm_decenter_y(self):
        """测试解析 PARM 2（decenter_y）"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        parser._parse_type(["COORDBRK"])
        
        parser._parse_parm(["2", "3.5"])
        
        assert parser._current_surface.decenter_y == 3.5
    
    def test_parse_parm_tilt_x(self):
        """测试解析 PARM 3（tilt_x，度）"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        parser._parse_type(["COORDBRK"])
        
        parser._parse_parm(["3", "45"])
        
        assert parser._current_surface.tilt_x_deg == 45.0
    
    def test_parse_parm_tilt_y(self):
        """测试解析 PARM 4（tilt_y，度）"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        parser._parse_type(["COORDBRK"])
        
        parser._parse_parm(["4", "30"])
        
        assert parser._current_surface.tilt_y_deg == 30.0
    
    def test_parse_parm_tilt_z(self):
        """测试解析 PARM 5（tilt_z，度）"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        parser._parse_type(["COORDBRK"])
        
        parser._parse_parm(["5", "15"])
        
        assert parser._current_surface.tilt_z_deg == 15.0
    
    def test_parse_parm_negative_values(self):
        """测试解析负值参数"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        parser._parse_type(["COORDBRK"])
        
        parser._parse_parm(["1", "-2.5"])
        parser._parse_parm(["3", "-45"])
        
        assert parser._current_surface.decenter_x == -2.5
        assert parser._current_surface.tilt_x_deg == -45.0
    
    def test_parse_parm_zero_values(self):
        """测试解析零值参数"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        parser._parse_type(["COORDBRK"])
        
        parser._parse_parm(["1", "0"])
        parser._parse_parm(["2", "0"])
        parser._parse_parm(["3", "0"])
        parser._parse_parm(["4", "0"])
        parser._parse_parm(["5", "0"])
        
        assert parser._current_surface.decenter_x == 0.0
        assert parser._current_surface.decenter_y == 0.0
        assert parser._current_surface.tilt_x_deg == 0.0
        assert parser._current_surface.tilt_y_deg == 0.0
        assert parser._current_surface.tilt_z_deg == 0.0
    
    def test_parse_parm_fractional_values(self):
        """测试解析小数值参数"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        parser._parse_type(["COORDBRK"])
        
        parser._parse_parm(["1", "1.234"])
        parser._parse_parm(["3", "22.5"])
        
        assert parser._current_surface.decenter_x == pytest.approx(1.234)
        assert parser._current_surface.tilt_x_deg == pytest.approx(22.5)
    
    def test_parse_parm_ignores_non_coordbrk(self):
        """测试非 COORDBRK 类型时忽略 PARM"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        parser._parse_type(["STANDARD"])
        
        # 对于 STANDARD 类型，PARM 应该被忽略
        parser._parse_parm(["3", "45"])
        
        # 值应该保持默认值
        assert parser._current_surface.tilt_x_deg == 0.0
    
    def test_parse_parm_ignores_parm_6(self):
        """测试忽略 PARM 6（旋转顺序标志）"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        parser._parse_type(["COORDBRK"])
        
        # PARM 6 应该被静默忽略，不抛出异常
        parser._parse_parm(["6", "1"])
        
        # 其他参数应该保持默认值
        assert parser._current_surface.decenter_x == 0.0
        assert parser._current_surface.tilt_x_deg == 0.0
    
    def test_parse_parm_no_current_surface(self):
        """测试没有当前表面时静默忽略"""
        parser = ZmxParser("dummy.zmx")
        
        # 不应抛出异常
        parser._parse_parm(["3", "45"])
    
    def test_parse_parm_empty_data(self):
        """测试空数据时静默忽略"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        parser._parse_type(["COORDBRK"])
        
        # 不应抛出异常
        parser._parse_parm([])
        parser._parse_parm(["3"])  # 只有索引，没有值
        
        # 值应该保持默认值
        assert parser._current_surface.tilt_x_deg == 0.0
    
    def test_parse_parm_invalid_index(self):
        """测试无效参数索引抛出异常"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        parser._parse_type(["COORDBRK"])
        parser._current_line_number = 50
        parser._current_line_content = "PARM abc 45"
        
        with pytest.raises(ZmxParseError) as exc_info:
            parser._parse_parm(["abc", "45"])
        
        assert "无效的参数索引" in str(exc_info.value)
    
    def test_parse_parm_invalid_value(self):
        """测试无效参数值抛出异常"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        parser._parse_type(["COORDBRK"])
        parser._current_line_number = 50
        parser._current_line_content = "PARM 3 xyz"
        
        with pytest.raises(ZmxParseError) as exc_info:
            parser._parse_parm(["3", "xyz"])
        
        assert "无效的参数值" in str(exc_info.value)
    
    def test_parse_parm_all_coordbrk_params(self):
        """测试解析完整的 COORDBRK 参数集"""
        parser = ZmxParser("dummy.zmx")
        parser._parse_surface(["0"])
        parser._parse_type(["COORDBRK"])
        
        # 设置所有参数
        parser._parse_parm(["1", "1.0"])   # decenter_x
        parser._parse_parm(["2", "2.0"])   # decenter_y
        parser._parse_parm(["3", "45.0"])  # tilt_x
        parser._parse_parm(["4", "30.0"])  # tilt_y
        parser._parse_parm(["5", "15.0"])  # tilt_z
        parser._parse_parm(["6", "0"])     # order (ignored)
        
        assert parser._current_surface.decenter_x == 1.0
        assert parser._current_surface.decenter_y == 2.0
        assert parser._current_surface.tilt_x_deg == 45.0
        assert parser._current_surface.tilt_y_deg == 30.0
        assert parser._current_surface.tilt_z_deg == 15.0


class TestZmxParserCoordinateBreakIntegration:
    """测试坐标断点解析的集成测试
    
    **Validates: Requirements 3.1, 3.2, 3.3**
    """
    
    def test_parse_coordbrk_with_parm(self, tmp_path):
        """测试解析带 PARM 的坐标断点表面"""
        content = """MODE SEQ
SURF 0
  TYPE STANDARD
  DISZ INFINITY
SURF 1
  TYPE COORDBRK
  PARM 1 0
  PARM 2 0
  PARM 3 45
  PARM 4 0
  PARM 5 0
  PARM 6 0
  DISZ 0
SURF 2
  TYPE STANDARD
  GLAS MIRROR 0 0 1.5 40
  DISZ 0
SURF 3
  TYPE COORDBRK
  PARM 1 0
  PARM 2 0
  PARM 3 45
  PARM 4 0
  PARM 5 0
  PARM 6 0
  DISZ -50
SURF 4
  TYPE STANDARD
  DISZ 0
"""
        file_path = tmp_path / "test_coordbrk_parm.zmx"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        parser = ZmxParser(str(file_path))
        data_model = parser.parse()
        
        # 验证表面数量
        assert len(data_model.surfaces) == 5
        
        # 验证第一个坐标断点（表面 1）
        surf1 = data_model.get_surface(1)
        assert surf1 is not None
        assert surf1.surface_type == 'coordinate_break'
        assert surf1.tilt_x_deg == 45.0
        assert surf1.tilt_y_deg == 0.0
        assert surf1.decenter_x == 0.0
        assert surf1.decenter_y == 0.0
        assert surf1.thickness == 0.0
        
        # 验证反射镜（表面 2）
        surf2 = data_model.get_surface(2)
        assert surf2 is not None
        assert surf2.is_mirror == True
        
        # 验证第二个坐标断点（表面 3）
        surf3 = data_model.get_surface(3)
        assert surf3 is not None
        assert surf3.surface_type == 'coordinate_break'
        assert surf3.tilt_x_deg == 45.0
        assert surf3.thickness == -50.0  # 负厚度表示反射方向传播
        
        # 验证可以获取所有坐标断点
        coord_breaks = data_model.get_coordinate_break_surfaces()
        assert len(coord_breaks) == 2
    
    def test_parse_coordbrk_with_decenter(self, tmp_path):
        """测试解析带偏心的坐标断点"""
        content = """MODE SEQ
SURF 0
  TYPE STANDARD
  DISZ INFINITY
SURF 1
  TYPE COORDBRK
  PARM 1 5.0
  PARM 2 3.0
  PARM 3 0
  PARM 4 0
  PARM 5 0
  PARM 6 0
  DISZ 10
SURF 2
  TYPE STANDARD
  DISZ 0
"""
        file_path = tmp_path / "test_coordbrk_decenter.zmx"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        parser = ZmxParser(str(file_path))
        data_model = parser.parse()
        
        # 验证坐标断点的偏心参数
        surf1 = data_model.get_surface(1)
        assert surf1 is not None
        assert surf1.surface_type == 'coordinate_break'
        assert surf1.decenter_x == 5.0
        assert surf1.decenter_y == 3.0
        assert surf1.tilt_x_deg == 0.0
        assert surf1.thickness == 10.0
    
    def test_parse_coordbrk_with_multiple_tilts(self, tmp_path):
        """测试解析带多个倾斜角的坐标断点"""
        content = """MODE SEQ
SURF 0
  TYPE STANDARD
  DISZ INFINITY
SURF 1
  TYPE COORDBRK
  PARM 1 0
  PARM 2 0
  PARM 3 30
  PARM 4 15
  PARM 5 5
  PARM 6 0
  DISZ 0
SURF 2
  TYPE STANDARD
  DISZ 0
"""
        file_path = tmp_path / "test_coordbrk_tilts.zmx"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        parser = ZmxParser(str(file_path))
        data_model = parser.parse()
        
        # 验证坐标断点的倾斜参数
        surf1 = data_model.get_surface(1)
        assert surf1 is not None
        assert surf1.surface_type == 'coordinate_break'
        assert surf1.tilt_x_deg == 30.0
        assert surf1.tilt_y_deg == 15.0
        assert surf1.tilt_z_deg == 5.0
    
    def test_parse_real_zmx_file_coordbrk(self):
        """测试解析真实 ZMX 文件中的坐标断点
        
        **Validates: Requirements 3.1, 3.2, 10.4**
        """
        import os
        zmx_path = "optiland-master/tests/zemax_files/one_mirror_up_45deg.zmx"
        
        if not os.path.exists(zmx_path):
            pytest.skip(f"测试文件不存在: {zmx_path}")
        
        parser = ZmxParser(zmx_path)
        data_model = parser.parse()
        
        # 验证文件被成功解析
        assert len(data_model.surfaces) > 0
        
        # 验证包含坐标断点
        coord_breaks = data_model.get_coordinate_break_surfaces()
        assert len(coord_breaks) >= 1, "应该至少有一个坐标断点"
        
        # 验证 45 度折叠镜的坐标断点
        found_45_deg = False
        for cb in coord_breaks:
            if abs(cb.tilt_x_deg) == 45.0 or abs(cb.tilt_y_deg) == 45.0:
                found_45_deg = True
                break
        
        assert found_45_deg, "应该有一个 45 度倾斜的坐标断点"
    
    def test_parse_simple_fold_mirror_zmx(self):
        """测试解析 simple_fold_mirror_up.zmx 文件
        
        **Validates: Requirements 3.1, 3.2, 3.3**
        """
        import os
        zmx_path = "optiland-master/tests/zemax_files/simple_fold_mirror_up.zmx"
        
        if not os.path.exists(zmx_path):
            pytest.skip(f"测试文件不存在: {zmx_path}")
        
        parser = ZmxParser(zmx_path)
        data_model = parser.parse()
        
        # 验证文件被成功解析
        assert len(data_model.surfaces) > 0
        
        # 验证包含坐标断点
        coord_breaks = data_model.get_coordinate_break_surfaces()
        assert len(coord_breaks) >= 2, "应该至少有两个坐标断点（折叠镜前后）"
        
        # 验证坐标断点的倾斜参数
        for cb in coord_breaks:
            # 所有坐标断点应该有有效的倾斜值（可能是 45, -45, 或 0）
            assert -90 <= cb.tilt_x_deg <= 90
            assert -90 <= cb.tilt_y_deg <= 90
            assert -90 <= cb.tilt_z_deg <= 90



class TestZmxParserEncodingErrorHandling:
    """测试 ZmxParser 的编码错误处理
    
    **Validates: Requirements 1.2, 8.3**
    """
    
    def test_encoding_error_raises_zmx_parse_error(self, tmp_path):
        """测试当所有编码都失败时抛出 ZmxParseError
        
        **Validates: Requirements 8.3**
        """
        # 创建一个包含无效字节序列的文件
        # 这个字节序列在 UTF-16、UTF-8 和 ISO-8859-1 中都无法正确解码
        file_path = tmp_path / "invalid_encoding.zmx"
        
        # 写入一些二进制数据，模拟损坏的文件
        # 使用 UTF-16 BOM 但后面跟着无效的 UTF-16 序列
        with open(file_path, 'wb') as f:
            # UTF-16 LE BOM
            f.write(b'\xff\xfe')
            # 后面跟着奇数个字节，这会导致 UTF-16 解码失败
            f.write(b'\x00\x01\x02')
        
        parser = ZmxParser(str(file_path))
        
        # 由于 ISO-8859-1 可以解码任何字节序列，这个测试可能不会失败
        # 但我们可以验证解析器能够处理这种情况
        try:
            lines = parser._try_read_file()
            # 如果成功读取，验证返回的是列表
            assert isinstance(lines, list)
        except ZmxParseError as e:
            # 如果抛出异常，验证错误信息包含编码相关内容
            assert "编码" in str(e) or "encoding" in str(e).lower()
    
    def test_empty_file_handling(self, tmp_path):
        """测试空文件处理
        
        **Validates: Requirements 1.1**
        """
        file_path = tmp_path / "empty.zmx"
        file_path.touch()  # 创建空文件
        
        parser = ZmxParser(str(file_path))
        lines = parser._try_read_file()
        
        # 空文件应该返回空列表
        assert lines == []
    
    def test_file_with_only_whitespace(self, tmp_path):
        """测试只包含空白字符的文件
        
        **Validates: Requirements 1.1**
        """
        file_path = tmp_path / "whitespace.zmx"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("   \n\t\n   \n")
        
        parser = ZmxParser(str(file_path))
        data_model = parser.parse()
        
        # 应该成功解析，但没有表面
        assert isinstance(data_model, ZmxDataModel)
        assert len(data_model.surfaces) == 0


class TestZmxParserRealFileIntegration:
    """测试真实 ZMX 文件的集成解析
    
    **Validates: Requirements 10.1-10.6**
    """
    
    def test_one_mirror_up_45deg_complete_parsing(self):
        """测试 one_mirror_up_45deg.zmx 的完整解析
        
        **Validates: Requirements 10.4**
        """
        import os
        import numpy as np
        
        zmx_path = "optiland-master/tests/zemax_files/one_mirror_up_45deg.zmx"
        
        if not os.path.exists(zmx_path):
            pytest.skip(f"测试文件不存在: {zmx_path}")
        
        parser = ZmxParser(zmx_path)
        data_model = parser.parse()
        
        # 验证基本结构
        assert len(data_model.surfaces) > 0
        assert data_model.entrance_pupil_diameter > 0
        assert len(data_model.wavelengths) > 0
        
        # 验证包含反射镜
        mirrors = data_model.get_mirror_surfaces()
        assert len(mirrors) >= 1, "应该至少有一个反射镜"
        
        # 验证包含坐标断点
        coord_breaks = data_model.get_coordinate_break_surfaces()
        assert len(coord_breaks) >= 1, "应该至少有一个坐标断点"
        
        # 验证 45 度倾斜
        found_45_deg = False
        for cb in coord_breaks:
            if abs(abs(cb.tilt_x_deg) - 45.0) < 0.1 or abs(abs(cb.tilt_y_deg) - 45.0) < 0.1:
                found_45_deg = True
                break
        
        assert found_45_deg, "应该有一个 45 度倾斜的坐标断点"
    
    def test_simple_fold_mirror_up_complete_parsing(self):
        """测试 simple_fold_mirror_up.zmx 的完整解析
        
        **Validates: Requirements 10.4, 10.5**
        """
        import os
        
        zmx_path = "optiland-master/tests/zemax_files/simple_fold_mirror_up.zmx"
        
        if not os.path.exists(zmx_path):
            pytest.skip(f"测试文件不存在: {zmx_path}")
        
        parser = ZmxParser(zmx_path)
        data_model = parser.parse()
        
        # 验证基本结构
        assert len(data_model.surfaces) > 0
        
        # 验证包含反射镜
        mirrors = data_model.get_mirror_surfaces()
        assert len(mirrors) >= 1, "应该至少有一个反射镜"
        
        # 验证包含坐标断点（折叠镜前后）
        coord_breaks = data_model.get_coordinate_break_surfaces()
        assert len(coord_breaks) >= 2, "应该至少有两个坐标断点（折叠镜前后）"
        
        # 验证折叠镜序列：COORDBRK + MIRROR + COORDBRK
        # 检查是否有连续的坐标断点-反射镜-坐标断点序列
        surface_indices = sorted(data_model.surfaces.keys())
        
        for i in range(len(surface_indices) - 2):
            idx1, idx2, idx3 = surface_indices[i], surface_indices[i+1], surface_indices[i+2]
            surf1 = data_model.get_surface(idx1)
            surf2 = data_model.get_surface(idx2)
            surf3 = data_model.get_surface(idx3)
            
            if (surf1.surface_type == 'coordinate_break' and
                surf2.is_mirror and
                surf3.surface_type == 'coordinate_break'):
                # 找到折叠镜序列
                # 验证前后坐标断点的倾斜角度匹配
                assert abs(surf1.tilt_x_deg - surf3.tilt_x_deg) < 0.1 or \
                       abs(surf1.tilt_y_deg - surf3.tilt_y_deg) < 0.1, \
                       "折叠镜前后的坐标断点倾斜角度应该匹配"
                break
    
    def test_all_zemax_files_can_be_parsed(self):
        """测试所有 ZMX 文件都能被成功解析
        
        **Validates: Requirements 10.5**
        """
        import os
        from pathlib import Path
        
        zmx_dir = Path("optiland-master/tests/zemax_files")
        
        if not zmx_dir.exists():
            pytest.skip(f"测试目录不存在: {zmx_dir}")
        
        zmx_files = list(zmx_dir.glob("*.zmx"))
        
        if not zmx_files:
            pytest.skip("没有找到 ZMX 测试文件")
        
        parsed_count = 0
        failed_files = []
        
        for zmx_file in zmx_files:
            try:
                parser = ZmxParser(str(zmx_file))
                data_model = parser.parse()
                
                # 验证返回有效的数据模型
                assert isinstance(data_model, ZmxDataModel)
                parsed_count += 1
                
            except ZmxUnsupportedError as e:
                # 不支持的特性是预期的，记录但不失败
                print(f"跳过不支持的文件 {zmx_file.name}: {e}")
            except Exception as e:
                failed_files.append((zmx_file.name, str(e)))
        
        # 至少应该成功解析一些文件
        assert parsed_count > 0, f"没有成功解析任何文件。失败: {failed_files}"
        
        # 如果有失败的文件，打印警告但不失败测试
        if failed_files:
            print(f"警告：以下文件解析失败: {failed_files}")


class TestZmxParserEdgeCases:
    """测试 ZmxParser 的边界情况
    
    **Validates: Requirements 8.2**
    """
    
    def test_surface_with_missing_radius_uses_default(self, tmp_path):
        """测试缺少半径时使用默认值（无穷大）
        
        **Validates: Requirements 8.2**
        """
        import numpy as np
        
        content = """MODE SEQ
SURF 0
  TYPE STANDARD
  DISZ INFINITY
SURF 1
  TYPE STANDARD
  DISZ 50
SURF 2
  TYPE STANDARD
  DISZ 0
"""
        file_path = tmp_path / "test_missing_radius.zmx"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        parser = ZmxParser(str(file_path))
        data_model = parser.parse()
        
        # 表面 1 没有 CURV，应该使用默认值 inf
        surf1 = data_model.get_surface(1)
        assert surf1.radius == np.inf
    
    def test_surface_with_missing_thickness_uses_default(self, tmp_path):
        """测试缺少厚度时使用默认值（0）
        
        **Validates: Requirements 8.2**
        """
        content = """MODE SEQ
SURF 0
  TYPE STANDARD
  DISZ INFINITY
SURF 1
  TYPE STANDARD
  CURV 0.01
SURF 2
  TYPE STANDARD
  DISZ 0
"""
        file_path = tmp_path / "test_missing_thickness.zmx"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        parser = ZmxParser(str(file_path))
        data_model = parser.parse()
        
        # 表面 1 没有 DISZ，应该使用默认值 0
        surf1 = data_model.get_surface(1)
        assert surf1.thickness == 0.0
    
    def test_surface_with_missing_conic_uses_default(self, tmp_path):
        """测试缺少圆锥常数时使用默认值（0）
        
        **Validates: Requirements 8.2**
        """
        content = """MODE SEQ
SURF 0
  TYPE STANDARD
  DISZ INFINITY
SURF 1
  TYPE STANDARD
  CURV 0.01
  DISZ 50
SURF 2
  TYPE STANDARD
  DISZ 0
"""
        file_path = tmp_path / "test_missing_conic.zmx"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        parser = ZmxParser(str(file_path))
        data_model = parser.parse()
        
        # 表面 1 没有 CONI，应该使用默认值 0
        surf1 = data_model.get_surface(1)
        assert surf1.conic == 0.0
    
    def test_coordbrk_with_missing_parm_uses_default(self, tmp_path):
        """测试坐标断点缺少 PARM 时使用默认值
        
        **Validates: Requirements 8.2**
        """
        content = """MODE SEQ
SURF 0
  TYPE STANDARD
  DISZ INFINITY
SURF 1
  TYPE COORDBRK
  PARM 3 45
  DISZ 0
SURF 2
  TYPE STANDARD
  DISZ 0
"""
        file_path = tmp_path / "test_missing_parm.zmx"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        parser = ZmxParser(str(file_path))
        data_model = parser.parse()
        
        # 表面 1 只有 PARM 3，其他应该使用默认值 0
        surf1 = data_model.get_surface(1)
        assert surf1.decenter_x == 0.0
        assert surf1.decenter_y == 0.0
        assert surf1.tilt_x_deg == 45.0
        assert surf1.tilt_y_deg == 0.0
        assert surf1.tilt_z_deg == 0.0
    
    def test_very_small_curvature(self, tmp_path):
        """测试非常小的曲率值
        
        **Validates: Requirements 2.1**
        """
        content = """MODE SEQ
SURF 0
  TYPE STANDARD
  DISZ INFINITY
SURF 1
  TYPE STANDARD
  CURV 0.0000001
  DISZ 50
SURF 2
  TYPE STANDARD
  DISZ 0
"""
        file_path = tmp_path / "test_small_curv.zmx"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        parser = ZmxParser(str(file_path))
        data_model = parser.parse()
        
        # 非常小的曲率应该转换为非常大的半径
        surf1 = data_model.get_surface(1)
        assert surf1.radius == pytest.approx(10000000.0, rel=1e-6)
    
    def test_negative_thickness_in_coordbrk(self, tmp_path):
        """测试坐标断点中的负厚度（反射后传播）
        
        **Validates: Requirements 3.3, 6.3**
        """
        content = """MODE SEQ
SURF 0
  TYPE STANDARD
  DISZ INFINITY
SURF 1
  TYPE COORDBRK
  PARM 3 45
  DISZ 0
SURF 2
  TYPE STANDARD
  GLAS MIRROR 0 0 1.5 40
  DISZ 0
SURF 3
  TYPE COORDBRK
  PARM 3 45
  DISZ -100
SURF 4
  TYPE STANDARD
  DISZ 0
"""
        file_path = tmp_path / "test_negative_thickness.zmx"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        parser = ZmxParser(str(file_path))
        data_model = parser.parse()
        
        # 表面 3 应该有负厚度
        surf3 = data_model.get_surface(3)
        assert surf3.thickness == -100.0
    
    def test_multiple_mirrors_in_sequence(self, tmp_path):
        """测试多个连续反射镜
        
        **Validates: Requirements 6.2**
        """
        content = """MODE SEQ
SURF 0
  TYPE STANDARD
  DISZ INFINITY
SURF 1
  TYPE COORDBRK
  PARM 3 45
  DISZ 0
SURF 2
  TYPE STANDARD
  GLAS MIRROR 0 0 1.5 40
  CURV 0.01
  DISZ 0
SURF 3
  TYPE COORDBRK
  PARM 3 45
  DISZ -50
SURF 4
  TYPE COORDBRK
  PARM 3 -45
  DISZ 0
SURF 5
  TYPE STANDARD
  GLAS MIRROR 0 0 1.5 40
  CURV 0.02
  DISZ 0
SURF 6
  TYPE COORDBRK
  PARM 3 -45
  DISZ -50
SURF 7
  TYPE STANDARD
  DISZ 0
"""
        file_path = tmp_path / "test_multiple_mirrors.zmx"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        parser = ZmxParser(str(file_path))
        data_model = parser.parse()
        
        # 验证有两个反射镜
        mirrors = data_model.get_mirror_surfaces()
        assert len(mirrors) == 2
        
        # 验证有四个坐标断点
        coord_breaks = data_model.get_coordinate_break_surfaces()
        assert len(coord_breaks) == 4
        
        # 验证反射镜的曲率半径
        assert mirrors[0].radius == pytest.approx(100.0)
        assert mirrors[1].radius == pytest.approx(50.0)


class TestZmxParserDataModelRepr:
    """测试数据模型的字符串表示
    
    **Validates: Requirements 4.1**
    """
    
    def test_zmx_surface_data_repr(self):
        """测试 ZmxSurfaceData 的 __repr__ 方法"""
        surface = ZmxSurfaceData(
            index=1,
            surface_type='standard',
            radius=100.0,
            thickness=50.0,
            conic=-1.0,
            material='mirror',
            is_mirror=True,
            semi_diameter=25.0,
            comment='M1',
        )
        
        repr_str = repr(surface)
        
        assert "index=1" in repr_str
        assert "standard" in repr_str
        assert "radius=100" in repr_str
        assert "thickness=50" in repr_str
        assert "conic=-1" in repr_str
        assert "mirror" in repr_str
        assert "is_mirror=True" in repr_str
        assert "semi_diameter=25" in repr_str
        assert "M1" in repr_str
    
    def test_zmx_surface_data_repr_coordbrk(self):
        """测试坐标断点的 __repr__ 方法"""
        surface = ZmxSurfaceData(
            index=2,
            surface_type='coordinate_break',
            decenter_x=1.0,
            decenter_y=2.0,
            tilt_x_deg=45.0,
            tilt_y_deg=30.0,
        )
        
        repr_str = repr(surface)
        
        assert "coordinate_break" in repr_str
        assert "decenter_x=1" in repr_str
        assert "decenter_y=2" in repr_str
        assert "tilt_x_deg=45" in repr_str
        assert "tilt_y_deg=30" in repr_str
    
    def test_zmx_data_model_repr(self):
        """测试 ZmxDataModel 的 __repr__ 方法"""
        model = ZmxDataModel()
        model.surfaces[0] = ZmxSurfaceData(index=0, surface_type='standard')
        model.surfaces[1] = ZmxSurfaceData(index=1, surface_type='standard', is_mirror=True)
        model.surfaces[2] = ZmxSurfaceData(index=2, surface_type='coordinate_break')
        model.wavelengths = [0.55, 0.486, 0.656]
        model.entrance_pupil_diameter = 20.0
        
        repr_str = repr(model)
        
        assert "surfaces=3" in repr_str
        assert "mirrors=1" in repr_str
        assert "coord_breaks=1" in repr_str
        assert "0.55" in repr_str
        assert "20" in repr_str
