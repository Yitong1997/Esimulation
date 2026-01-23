"""
bts I/O 函数属性基测试

使用 hypothesis 库验证 load_zmx 函数的正确性属性。

**Feature: matlab-style-api**
**Validates: Requirements 1.1, 2.1, 2.2**
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import List

import pytest
from hypothesis import given, strategies as st, settings, assume

sys.path.insert(0, 'src')

from bts import load_zmx
from bts.optical_system import OpticalSystem


# ============================================================================
# 全局设置：增加 deadline 以适应 ZMX 文件加载时间
# ============================================================================

# ZMX 文件加载可能需要较长时间（首次加载时需要初始化模块）
# 设置 deadline=None 禁用超时检查
ZMX_LOAD_SETTINGS = settings(max_examples=100, deadline=None)


# ============================================================================
# 测试数据：工作区中可用的 ZMX 文件
# ============================================================================

# 工作区中已知存在的 ZMX 文件路径（相对于工作区根目录）
AVAILABLE_ZMX_FILES: List[str] = [
    "optiland-master/tests/zemax_files/simple_fold_mirror_up.zmx",
    "optiland-master/tests/zemax_files/one_mirror_up_45deg.zmx",
    "optiland-master/tests/zemax_files/complicated_fold_mirrors_setup_v2.zmx",
    "optiland-master/tests/zemax_files/lens1.zmx",
    "optiland-master/tests/zemax_files/lens2.zmx",
]


def get_valid_zmx_files() -> List[str]:
    """获取工作区中实际存在的 ZMX 文件列表"""
    valid_files = []
    for path in AVAILABLE_ZMX_FILES:
        if Path(path).exists():
            valid_files.append(path)
    return valid_files


# ============================================================================
# 测试策略定义
# ============================================================================

# 有效 ZMX 文件路径策略
@st.composite
def valid_zmx_path_strategy(draw):
    """生成有效的 ZMX 文件路径"""
    valid_files = get_valid_zmx_files()
    assume(len(valid_files) > 0)  # 确保至少有一个有效文件
    return draw(st.sampled_from(valid_files))


# 不存在的文件路径策略
@st.composite
def nonexistent_path_strategy(draw):
    """生成不存在的文件路径字符串"""
    # 生成随机文件名组件
    filename_chars = st.characters(
        whitelist_categories=('L', 'N'),  # 字母和数字
        whitelist_characters='_-'
    )
    
    # 生成随机目录名
    dir_name = draw(st.text(
        alphabet=filename_chars,
        min_size=1,
        max_size=20
    ))
    
    # 生成随机文件名
    file_name = draw(st.text(
        alphabet=filename_chars,
        min_size=1,
        max_size=20
    ))
    
    # 组合成路径
    path = f"nonexistent_dir_{dir_name}/nonexistent_file_{file_name}.zmx"
    
    # 确保路径确实不存在
    assume(not Path(path).exists())
    
    return path


# 随机字符串路径策略（用于测试各种不存在的路径）
@st.composite
def random_nonexistent_path_strategy(draw):
    """生成各种形式的不存在的文件路径"""
    path_type = draw(st.integers(min_value=0, max_value=4))
    
    if path_type == 0:
        # 简单的不存在文件名
        name = draw(st.text(
            alphabet=st.characters(whitelist_categories=('L', 'N')),
            min_size=5,
            max_size=30
        ))
        path = f"nonexistent_{name}.zmx"
    elif path_type == 1:
        # 带有不存在目录的路径
        dir_name = draw(st.text(
            alphabet=st.characters(whitelist_categories=('L', 'N')),
            min_size=3,
            max_size=15
        ))
        file_name = draw(st.text(
            alphabet=st.characters(whitelist_categories=('L', 'N')),
            min_size=3,
            max_size=15
        ))
        path = f"fake_dir_{dir_name}/fake_file_{file_name}.zmx"
    elif path_type == 2:
        # 深层嵌套的不存在路径
        depth = draw(st.integers(min_value=2, max_value=5))
        parts = []
        for _ in range(depth):
            part = draw(st.text(
                alphabet=st.characters(whitelist_categories=('L', 'N')),
                min_size=2,
                max_size=10
            ))
            parts.append(f"fake_{part}")
        path = "/".join(parts) + ".zmx"
    elif path_type == 3:
        # 带有时间戳的唯一路径
        import time
        timestamp = int(time.time() * 1000000)
        random_suffix = draw(st.integers(min_value=0, max_value=999999))
        path = f"nonexistent_{timestamp}_{random_suffix}.zmx"
    else:
        # 带有特殊字符的路径（但仍然是有效的文件名字符）
        name = draw(st.text(
            alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='_-'),
            min_size=5,
            max_size=25
        ))
        path = f"test_nonexistent_{name}.zmx"
    
    # 确保路径确实不存在
    assume(not Path(path).exists())
    assume(len(path) > 0)
    
    return path


# ============================================================================
# Property 1: load_zmx 返回正确类型
# ============================================================================

@ZMX_LOAD_SETTINGS
@given(zmx_path=valid_zmx_path_strategy())
def test_property_1_load_zmx_returns_optical_system(zmx_path: str):
    """
    **Feature: matlab-style-api, Property 1: load_zmx 返回正确类型**
    **Validates: Requirements 1.1, 2.1**
    
    *For any* 有效的 ZMX 文件路径，调用 `bts.load_zmx(path)` 应该返回 
    `OpticalSystem` 类型的对象，且该对象的表面数量应该大于 0。
    """
    # 调用 load_zmx
    system = load_zmx(zmx_path)
    
    # 验证返回类型
    assert isinstance(system, OpticalSystem), (
        f"load_zmx 返回类型不正确：\n"
        f"  期望类型: OpticalSystem\n"
        f"  实际类型: {type(system).__name__}\n"
        f"  文件路径: {zmx_path}"
    )
    
    # 验证表面数量大于 0
    num_surfaces = len(system)
    assert num_surfaces > 0, (
        f"load_zmx 返回的 OpticalSystem 表面数量为 0：\n"
        f"  文件路径: {zmx_path}\n"
        f"  表面数量: {num_surfaces}"
    )


# 有效的表面类型列表（包括 ZMX 解析器可能返回的所有类型）
VALID_SURFACE_TYPES = [
    'standard',      # 标准球面
    'paraxial',      # 近轴透镜
    'coordbrk',      # 坐标断点
    'flat',          # 平面
    'evenasph',      # 偶次非球面
    'even_asphere',  # 偶次非球面（另一种命名）
    'biconic',       # 双圆锥面
    'toroidal',      # 环面
    'sphere',        # 球面
    'mirror',        # 反射镜
]


@ZMX_LOAD_SETTINGS
@given(zmx_path=valid_zmx_path_strategy())
def test_property_1_load_zmx_system_has_valid_surfaces(zmx_path: str):
    """
    **Feature: matlab-style-api, Property 1: load_zmx 返回正确类型**
    **Validates: Requirements 1.1, 2.1**
    
    验证加载的 OpticalSystem 中的每个表面都有有效的属性。
    """
    # 调用 load_zmx
    system = load_zmx(zmx_path)
    
    # 验证每个表面都有有效的属性
    for i, surface in enumerate(system._surfaces):
        # 验证表面索引
        assert surface.index >= 0, (
            f"表面索引无效：\n"
            f"  文件路径: {zmx_path}\n"
            f"  表面 {i}: index = {surface.index}"
        )
        
        # 验证表面类型（包括所有可能的类型）
        assert surface.surface_type in VALID_SURFACE_TYPES, (
            f"表面类型无效：\n"
            f"  文件路径: {zmx_path}\n"
            f"  表面 {i}: surface_type = {surface.surface_type}\n"
            f"  有效类型: {VALID_SURFACE_TYPES}"
        )
        
        # 验证半口径为非负数（某些虚拟表面可能为 0）
        assert surface.semi_aperture >= 0, (
            f"半口径无效：\n"
            f"  文件路径: {zmx_path}\n"
            f"  表面 {i}: semi_aperture = {surface.semi_aperture}"
        )


@ZMX_LOAD_SETTINGS
@given(zmx_path=valid_zmx_path_strategy())
def test_property_1_load_zmx_system_name_from_filename(zmx_path: str):
    """
    **Feature: matlab-style-api, Property 1: load_zmx 返回正确类型**
    **Validates: Requirements 1.1, 2.1**
    
    验证加载的 OpticalSystem 的名称来自文件名（不含扩展名）。
    """
    # 调用 load_zmx
    system = load_zmx(zmx_path)
    
    # 获取期望的系统名称（文件名不含扩展名）
    expected_name = Path(zmx_path).stem
    
    # 验证系统名称
    assert system.name == expected_name, (
        f"系统名称不正确：\n"
        f"  文件路径: {zmx_path}\n"
        f"  期望名称: {expected_name}\n"
        f"  实际名称: {system.name}"
    )


@ZMX_LOAD_SETTINGS
@given(zmx_path=valid_zmx_path_strategy())
def test_property_1_load_zmx_preserves_source_path(zmx_path: str):
    """
    **Feature: matlab-style-api, Property 1: load_zmx 返回正确类型**
    **Validates: Requirements 1.1, 2.1**
    
    验证加载的 OpticalSystem 保存了源文件路径。
    """
    # 调用 load_zmx
    system = load_zmx(zmx_path)
    
    # 验证源文件路径被保存
    assert hasattr(system, '_source_path'), (
        f"OpticalSystem 缺少 _source_path 属性：\n"
        f"  文件路径: {zmx_path}"
    )
    
    # 规范化路径进行比较（处理 Windows/Unix 路径差异）
    expected_path = Path(zmx_path).as_posix()
    actual_path = Path(system._source_path).as_posix()
    
    assert actual_path == expected_path, (
        f"源文件路径不正确：\n"
        f"  期望路径: {expected_path}\n"
        f"  实际路径: {actual_path}"
    )


# ============================================================================
# Property 2: 不存在的文件抛出 FileNotFoundError
# ============================================================================

@settings(max_examples=100, deadline=None)
@given(nonexistent_path=random_nonexistent_path_strategy())
def test_property_2_nonexistent_file_raises_error(nonexistent_path: str):
    """
    **Feature: matlab-style-api, Property 2: 不存在的文件抛出 FileNotFoundError**
    **Validates: Requirements 2.2**
    
    *For any* 不存在的文件路径字符串，调用 `bts.load_zmx(path)` 
    应该抛出 `FileNotFoundError` 异常。
    """
    # 确保路径确实不存在
    assert not Path(nonexistent_path).exists(), (
        f"测试前提条件失败：路径意外存在：{nonexistent_path}"
    )
    
    # 验证抛出 FileNotFoundError
    with pytest.raises(FileNotFoundError) as exc_info:
        load_zmx(nonexistent_path)
    
    # 验证错误信息包含文件路径
    error_message = str(exc_info.value)
    assert nonexistent_path in error_message or "不存在" in error_message, (
        f"FileNotFoundError 错误信息不包含文件路径：\n"
        f"  文件路径: {nonexistent_path}\n"
        f"  错误信息: {error_message}"
    )


@settings(max_examples=100, deadline=None)
@given(nonexistent_path=nonexistent_path_strategy())
def test_property_2_structured_nonexistent_path_raises_error(nonexistent_path: str):
    """
    **Feature: matlab-style-api, Property 2: 不存在的文件抛出 FileNotFoundError**
    **Validates: Requirements 2.2**
    
    使用结构化生成的不存在路径测试 FileNotFoundError。
    """
    # 确保路径确实不存在
    assert not Path(nonexistent_path).exists(), (
        f"测试前提条件失败：路径意外存在：{nonexistent_path}"
    )
    
    # 验证抛出 FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_zmx(nonexistent_path)


def test_property_2_empty_path_raises_error():
    """
    **Feature: matlab-style-api, Property 2: 不存在的文件抛出 FileNotFoundError**
    **Validates: Requirements 2.2**
    
    测试空路径字符串抛出 FileNotFoundError。
    """
    # 空路径应该抛出 FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_zmx("")


def test_property_2_directory_path_raises_error():
    """
    **Feature: matlab-style-api, Property 2: 不存在的文件抛出 FileNotFoundError**
    **Validates: Requirements 2.2**
    
    测试目录路径（而非文件路径）抛出 FileNotFoundError。
    """
    # 使用一个存在的目录路径
    dir_path = "tests/property"
    
    # 确保这是一个目录
    assert Path(dir_path).is_dir(), f"测试前提条件失败：{dir_path} 不是目录"
    
    # 目录路径应该抛出 FileNotFoundError（因为不是文件）
    with pytest.raises(FileNotFoundError):
        load_zmx(dir_path)


# ============================================================================
# 额外的边界条件测试
# ============================================================================

def test_load_zmx_with_absolute_path():
    """
    测试使用绝对路径加载 ZMX 文件。
    """
    valid_files = get_valid_zmx_files()
    if not valid_files:
        pytest.skip("没有可用的 ZMX 文件")
    
    # 使用第一个有效文件的绝对路径
    relative_path = valid_files[0]
    absolute_path = str(Path(relative_path).resolve())
    
    # 加载文件
    system = load_zmx(absolute_path)
    
    # 验证返回类型
    assert isinstance(system, OpticalSystem)
    assert len(system) > 0


def test_load_zmx_multiple_times_same_file():
    """
    测试多次加载同一个 ZMX 文件返回相同结构。
    """
    valid_files = get_valid_zmx_files()
    if not valid_files:
        pytest.skip("没有可用的 ZMX 文件")
    
    zmx_path = valid_files[0]
    
    # 加载两次
    system1 = load_zmx(zmx_path)
    system2 = load_zmx(zmx_path)
    
    # 验证两次加载的结果结构相同
    assert len(system1) == len(system2), (
        f"多次加载同一文件返回不同的表面数量：\n"
        f"  第一次: {len(system1)}\n"
        f"  第二次: {len(system2)}"
    )
    
    assert system1.name == system2.name, (
        f"多次加载同一文件返回不同的系统名称：\n"
        f"  第一次: {system1.name}\n"
        f"  第二次: {system2.name}"
    )


@settings(max_examples=50, deadline=None)
@given(zmx_path=valid_zmx_path_strategy())
def test_load_zmx_num_surfaces_property(zmx_path: str):
    """
    验证 num_surfaces 属性与 __len__ 返回相同的值。
    """
    system = load_zmx(zmx_path)
    
    assert system.num_surfaces == len(system), (
        f"num_surfaces 与 __len__ 不一致：\n"
        f"  num_surfaces: {system.num_surfaces}\n"
        f"  __len__: {len(system)}"
    )


# ============================================================================
# 特定 ZMX 文件测试（确保测试覆盖）
# ============================================================================

def test_load_simple_fold_mirror():
    """
    测试加载 simple_fold_mirror_up.zmx 文件。
    """
    zmx_path = "optiland-master/tests/zemax_files/simple_fold_mirror_up.zmx"
    if not Path(zmx_path).exists():
        pytest.skip(f"ZMX 文件不存在: {zmx_path}")
    
    system = load_zmx(zmx_path)
    
    assert isinstance(system, OpticalSystem)
    assert len(system) > 0
    assert system.name == "simple_fold_mirror_up"


def test_load_complicated_fold_mirrors():
    """
    测试加载 complicated_fold_mirrors_setup_v2.zmx 文件。
    """
    zmx_path = "optiland-master/tests/zemax_files/complicated_fold_mirrors_setup_v2.zmx"
    if not Path(zmx_path).exists():
        pytest.skip(f"ZMX 文件不存在: {zmx_path}")
    
    system = load_zmx(zmx_path)
    
    assert isinstance(system, OpticalSystem)
    assert len(system) > 0
    assert system.name == "complicated_fold_mirrors_setup_v2"


def test_load_lens_file():
    """
    测试加载透镜 ZMX 文件。
    """
    zmx_path = "optiland-master/tests/zemax_files/lens1.zmx"
    if not Path(zmx_path).exists():
        pytest.skip(f"ZMX 文件不存在: {zmx_path}")
    
    system = load_zmx(zmx_path)
    
    assert isinstance(system, OpticalSystem)
    assert len(system) > 0
    assert system.name == "lens1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
