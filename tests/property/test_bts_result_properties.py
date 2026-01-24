"""
SimulationResult 保存/加载属性基测试

使用 hypothesis 库验证 SimulationResult 的保存/加载往返一致性。

**Feature: matlab-style-api**
**Validates: Requirements 7.2, 7.3, 7.4**
"""

import tempfile
import shutil
import os
import numpy as np
from numpy.testing import assert_allclose
import pytest
from hypothesis import given, strategies as st, settings, assume

import sys
sys.path.insert(0, 'src')

from bts import simulate, OpticalSystem, GaussianSource, SimulationResult


# ============================================================================
# 测试策略定义
# ============================================================================

# 波长策略（可见光到近红外，单位 μm）
wavelength_strategy = st.floats(
    min_value=0.4, max_value=2.0,
    allow_nan=False, allow_infinity=False
)

# 束腰半径策略（单位 mm）
waist_radius_strategy = st.floats(
    min_value=1.0, max_value=20.0,
    allow_nan=False, allow_infinity=False
)

# 网格大小策略（使用较小的值以加快测试速度）
grid_size_strategy = st.sampled_from([64, 128])

# 表面位置策略（单位 mm）
surface_z_strategy = st.floats(
    min_value=10.0, max_value=200.0,
    allow_nan=False, allow_infinity=False
)

# 倾斜角度策略（度）
tilt_angle_strategy = st.floats(
    min_value=0.0, max_value=45.0,
    allow_nan=False, allow_infinity=False
)


# 曲率半径策略（单位 mm）
radius_strategy = st.floats(
    min_value=100.0, max_value=1000.0,
    allow_nan=False, allow_infinity=False
)


# ============================================================================
# 辅助函数
# ============================================================================

def create_simple_flat_mirror_system(
    z: float = 50.0,
    tilt_x: float = 0.0,
) -> OpticalSystem:
    """创建简单的平面镜系统
    
    参数:
        z: 镜面位置 (mm)
        tilt_x: 绕 X 轴倾斜角度（度）
    
    返回:
        OpticalSystem 对象
    
    注意:
        🚫 禁止设置 semi_aperture 参数！
    """
    system = OpticalSystem("Test Flat Mirror")
    system.add_flat_mirror(z=z, tilt_x=tilt_x)
    return system


def create_simple_spherical_mirror_system(
    z: float = 100.0,
    radius: float = 200.0,
) -> OpticalSystem:
    """创建简单的球面镜系统
    
    参数:
        z: 镜面位置 (mm)
        radius: 曲率半径 (mm)
    
    返回:
        OpticalSystem 对象
    
    注意:
        🚫 禁止设置 semi_aperture 参数！
    """
    system = OpticalSystem("Test Spherical Mirror")
    system.add_spherical_mirror(z=z, radius=radius)
    return system


def create_valid_source(
    wavelength_um: float = 0.633,
    w0_mm: float = 5.0,
    grid_size: int = 64,
) -> GaussianSource:
    """创建有效的高斯光源
    
    参数:
        wavelength_um: 波长 (μm)
        w0_mm: 束腰半径 (mm)
        grid_size: 网格大小
    
    返回:
        GaussianSource 对象
    """
    return GaussianSource(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        grid_size=grid_size,
    )


# ============================================================================
# Property 8: 结果保存/加载往返一致性
# ============================================================================

@settings(max_examples=100, deadline=None)
@given(
    z=surface_z_strategy,
    tilt_x=tilt_angle_strategy,
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    grid_size=grid_size_strategy,
)
def test_property_8_save_load_roundtrip_flat_mirror(
    z: float,
    tilt_x: float,
    wavelength_um: float,
    w0_mm: float,
    grid_size: int,
):
    """
    **Feature: matlab-style-api, Property 8: 结果保存/加载往返一致性**
    **Validates: Requirements 7.2, 7.3, 7.4**
    
    *For any* 有效的 `SimulationResult` 对象，保存到目录后再加载回来，
    加载的结果应该与原始结果在关键属性上一致（success、wavelength_um、grid_size、表面数量）。
    
    测试场景：单个平面镜系统
    """
    # 创建光学系统
    system = create_simple_flat_mirror_system(z=z, tilt_x=tilt_x)
    
    # 创建光源
    source = create_valid_source(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        grid_size=grid_size,
    )
    
    # 执行仿真
    result = simulate(system, source, verbose=False)
    
    # 验证仿真成功
    assert result.success, f"仿真失败: {result.error_message}"
    
    # 使用临时目录进行保存/加载测试
    temp_dir = tempfile.mkdtemp()
    try:
        save_path = os.path.join(temp_dir, "test_result")
        
        # 保存结果
        result.save(save_path)
        
        # 加载结果
        loaded_result = SimulationResult.load(save_path)
        
        # 验证关键属性一致性
        assert loaded_result.success == result.success, (
            f"success 不一致：原始 {result.success}，加载 {loaded_result.success}"
        )
        
        assert_allclose(
            loaded_result.config.wavelength_um,
            result.config.wavelength_um,
            rtol=1e-10,
            err_msg=f"wavelength_um 不一致：原始 {result.config.wavelength_um}，"
                    f"加载 {loaded_result.config.wavelength_um}",
        )
        
        assert loaded_result.config.grid_size == result.config.grid_size, (
            f"grid_size 不一致：原始 {result.config.grid_size}，"
            f"加载 {loaded_result.config.grid_size}"
        )
        
        assert len(loaded_result.surfaces) == len(result.surfaces), (
            f"表面数量不一致：原始 {len(result.surfaces)}，"
            f"加载 {len(loaded_result.surfaces)}"
        )
        
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir)


@settings(max_examples=100, deadline=None)
@given(
    z=surface_z_strategy,
    radius=radius_strategy,
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    grid_size=grid_size_strategy,
)
def test_property_8_save_load_roundtrip_spherical_mirror(
    z: float,
    radius: float,
    wavelength_um: float,
    w0_mm: float,
    grid_size: int,
):
    """
    **Feature: matlab-style-api, Property 8: 结果保存/加载往返一致性**
    **Validates: Requirements 7.2, 7.3, 7.4**
    
    测试场景：单个球面镜系统
    """
    # 创建光学系统
    system = create_simple_spherical_mirror_system(z=z, radius=radius)
    
    # 创建光源
    source = create_valid_source(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        grid_size=grid_size,
    )
    
    # 执行仿真
    result = simulate(system, source, verbose=False)
    
    # 验证仿真成功
    assert result.success, f"仿真失败: {result.error_message}"
    
    # 使用临时目录进行保存/加载测试
    temp_dir = tempfile.mkdtemp()
    try:
        save_path = os.path.join(temp_dir, "test_result")
        
        # 保存结果
        result.save(save_path)
        
        # 加载结果
        loaded_result = SimulationResult.load(save_path)
        
        # 验证关键属性一致性
        assert loaded_result.success == result.success, (
            f"success 不一致：原始 {result.success}，加载 {loaded_result.success}"
        )
        
        assert_allclose(
            loaded_result.config.wavelength_um,
            result.config.wavelength_um,
            rtol=1e-10,
            err_msg=f"wavelength_um 不一致",
        )
        
        assert loaded_result.config.grid_size == result.config.grid_size, (
            f"grid_size 不一致：原始 {result.config.grid_size}，"
            f"加载 {loaded_result.config.grid_size}"
        )
        
        assert len(loaded_result.surfaces) == len(result.surfaces), (
            f"表面数量不一致：原始 {len(result.surfaces)}，"
            f"加载 {len(loaded_result.surfaces)}"
        )
        
    finally:
        shutil.rmtree(temp_dir)


@settings(max_examples=50, deadline=None)
@given(
    num_surfaces=st.integers(min_value=1, max_value=3),
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    grid_size=grid_size_strategy,
)
def test_property_8_save_load_roundtrip_multiple_surfaces(
    num_surfaces: int,
    wavelength_um: float,
    w0_mm: float,
    grid_size: int,
):
    """
    **Feature: matlab-style-api, Property 8: 结果保存/加载往返一致性**
    **Validates: Requirements 7.2, 7.3, 7.4**
    
    测试场景：多表面系统
    """
    # 创建光学系统（多个平面镜）
    system = OpticalSystem("Test Multiple Surfaces")
    
    for i in range(num_surfaces):
        z = 50.0 + i * 100.0  # 每个表面间隔 100mm
        system.add_flat_mirror(z=z, tilt_x=0.0)
    
    # 创建光源
    source = create_valid_source(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        grid_size=grid_size,
    )
    
    # 执行仿真
    result = simulate(system, source, verbose=False)
    
    # 验证仿真成功
    assert result.success, f"仿真失败: {result.error_message}"
    
    # 使用临时目录进行保存/加载测试
    temp_dir = tempfile.mkdtemp()
    try:
        save_path = os.path.join(temp_dir, "test_result")
        
        # 保存结果
        result.save(save_path)
        
        # 加载结果
        loaded_result = SimulationResult.load(save_path)
        
        # 验证关键属性一致性
        assert loaded_result.success == result.success
        
        assert_allclose(
            loaded_result.config.wavelength_um,
            result.config.wavelength_um,
            rtol=1e-10,
        )
        
        assert loaded_result.config.grid_size == result.config.grid_size
        
        assert len(loaded_result.surfaces) == len(result.surfaces), (
            f"表面数量不一致：原始 {len(result.surfaces)}，"
            f"加载 {len(loaded_result.surfaces)}"
        )
        
    finally:
        shutil.rmtree(temp_dir)


@settings(max_examples=100, deadline=None)
@given(
    z=surface_z_strategy,
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    grid_size=grid_size_strategy,
)
def test_property_8_save_load_preserves_source_params(
    z: float,
    wavelength_um: float,
    w0_mm: float,
    grid_size: int,
):
    """
    **Feature: matlab-style-api, Property 8: 结果保存/加载往返一致性**
    **Validates: Requirements 7.2, 7.3, 7.4**
    
    验证保存/加载后光源参数保持一致。
    """
    # 创建光学系统
    system = create_simple_flat_mirror_system(z=z)
    
    # 创建光源
    source = create_valid_source(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        grid_size=grid_size,
    )
    
    # 执行仿真
    result = simulate(system, source, verbose=False)
    
    # 验证仿真成功
    assert result.success, f"仿真失败: {result.error_message}"
    
    # 使用临时目录进行保存/加载测试
    temp_dir = tempfile.mkdtemp()
    try:
        save_path = os.path.join(temp_dir, "test_result")
        
        # 保存结果
        result.save(save_path)
        
        # 加载结果
        loaded_result = SimulationResult.load(save_path)
        
        # 验证光源参数一致性
        assert_allclose(
            loaded_result.source_params.wavelength_um,
            result.source_params.wavelength_um,
            rtol=1e-10,
            err_msg="光源波长不一致",
        )
        
        assert_allclose(
            loaded_result.source_params.w0_mm,
            result.source_params.w0_mm,
            rtol=1e-10,
            err_msg="光源束腰半径不一致",
        )
        
        assert loaded_result.source_params.grid_size == result.source_params.grid_size, (
            f"光源网格大小不一致：原始 {result.source_params.grid_size}，"
            f"加载 {loaded_result.source_params.grid_size}"
        )
        
        assert_allclose(
            loaded_result.source_params.physical_size_mm,
            result.source_params.physical_size_mm,
            rtol=1e-10,
            err_msg="光源物理尺寸不一致",
        )
        
    finally:
        shutil.rmtree(temp_dir)


@settings(max_examples=100, deadline=None)
@given(
    z=surface_z_strategy,
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    grid_size=grid_size_strategy,
)
def test_property_8_save_load_preserves_total_path_length(
    z: float,
    wavelength_um: float,
    w0_mm: float,
    grid_size: int,
):
    """
    **Feature: matlab-style-api, Property 8: 结果保存/加载往返一致性**
    **Validates: Requirements 7.2, 7.3, 7.4**
    
    验证保存/加载后总光程保持一致。
    """
    # 创建光学系统
    system = create_simple_flat_mirror_system(z=z)
    
    # 创建光源
    source = create_valid_source(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        grid_size=grid_size,
    )
    
    # 执行仿真
    result = simulate(system, source, verbose=False)
    
    # 验证仿真成功
    assert result.success, f"仿真失败: {result.error_message}"
    
    # 使用临时目录进行保存/加载测试
    temp_dir = tempfile.mkdtemp()
    try:
        save_path = os.path.join(temp_dir, "test_result")
        
        # 保存结果
        result.save(save_path)
        
        # 加载结果
        loaded_result = SimulationResult.load(save_path)
        
        # 验证总光程一致性
        assert_allclose(
            loaded_result.total_path_length,
            result.total_path_length,
            rtol=1e-10,
            err_msg=f"总光程不一致：原始 {result.total_path_length}，"
                    f"加载 {loaded_result.total_path_length}",
        )
        
    finally:
        shutil.rmtree(temp_dir)


@settings(max_examples=100, deadline=None)
@given(
    z=surface_z_strategy,
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    grid_size=grid_size_strategy,
)
def test_property_8_save_load_preserves_wavefront_data(
    z: float,
    wavelength_um: float,
    w0_mm: float,
    grid_size: int,
):
    """
    **Feature: matlab-style-api, Property 8: 结果保存/加载往返一致性**
    **Validates: Requirements 7.2, 7.3, 7.4**
    
    验证保存/加载后波前数据（振幅和相位）保持一致。
    """
    # 创建光学系统
    system = create_simple_flat_mirror_system(z=z)
    
    # 创建光源
    source = create_valid_source(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        grid_size=grid_size,
    )
    
    # 执行仿真
    result = simulate(system, source, verbose=False)
    
    # 验证仿真成功
    assert result.success, f"仿真失败: {result.error_message}"
    
    # 使用临时目录进行保存/加载测试
    temp_dir = tempfile.mkdtemp()
    try:
        save_path = os.path.join(temp_dir, "test_result")
        
        # 保存结果
        result.save(save_path)
        
        # 加载结果
        loaded_result = SimulationResult.load(save_path)
        
        # 验证每个表面的波前数据
        for i, (orig_surf, loaded_surf) in enumerate(
            zip(result.surfaces, loaded_result.surfaces)
        ):
            # 验证表面索引和名称
            assert loaded_surf.index == orig_surf.index, (
                f"表面 {i} 索引不一致"
            )
            assert loaded_surf.name == orig_surf.name, (
                f"表面 {i} 名称不一致"
            )
            
            # 验证入射波前数据
            if orig_surf.entrance is not None:
                assert loaded_surf.entrance is not None, (
                    f"表面 {i} 入射波前数据丢失"
                )
                assert_allclose(
                    loaded_surf.entrance.amplitude,
                    orig_surf.entrance.amplitude,
                    rtol=1e-10,
                    err_msg=f"表面 {i} 入射振幅不一致",
                )
                assert_allclose(
                    loaded_surf.entrance.phase,
                    orig_surf.entrance.phase,
                    rtol=1e-10,
                    err_msg=f"表面 {i} 入射相位不一致",
                )
            
            # 验证出射波前数据
            if orig_surf.exit is not None:
                assert loaded_surf.exit is not None, (
                    f"表面 {i} 出射波前数据丢失"
                )
                assert_allclose(
                    loaded_surf.exit.amplitude,
                    orig_surf.exit.amplitude,
                    rtol=1e-10,
                    err_msg=f"表面 {i} 出射振幅不一致",
                )
                assert_allclose(
                    loaded_surf.exit.phase,
                    orig_surf.exit.phase,
                    rtol=1e-10,
                    err_msg=f"表面 {i} 出射相位不一致",
                )
        
    finally:
        shutil.rmtree(temp_dir)


@settings(max_examples=100, deadline=None)
@given(
    z=surface_z_strategy,
    wavelength_um=wavelength_strategy,
    w0_mm=waist_radius_strategy,
    grid_size=grid_size_strategy,
)
def test_property_8_save_load_preserves_surface_geometry(
    z: float,
    wavelength_um: float,
    w0_mm: float,
    grid_size: int,
):
    """
    **Feature: matlab-style-api, Property 8: 结果保存/加载往返一致性**
    **Validates: Requirements 7.2, 7.3, 7.4**
    
    验证保存/加载后表面几何信息保持一致。
    """
    # 创建光学系统
    system = create_simple_flat_mirror_system(z=z)
    
    # 创建光源
    source = create_valid_source(
        wavelength_um=wavelength_um,
        w0_mm=w0_mm,
        grid_size=grid_size,
    )
    
    # 执行仿真
    result = simulate(system, source, verbose=False)
    
    # 验证仿真成功
    assert result.success, f"仿真失败: {result.error_message}"
    
    # 使用临时目录进行保存/加载测试
    temp_dir = tempfile.mkdtemp()
    try:
        save_path = os.path.join(temp_dir, "test_result")
        
        # 保存结果
        result.save(save_path)
        
        # 加载结果
        loaded_result = SimulationResult.load(save_path)
        
        # 验证每个表面的几何信息
        for i, (orig_surf, loaded_surf) in enumerate(
            zip(result.surfaces, loaded_result.surfaces)
        ):
            if orig_surf.geometry is not None:
                assert loaded_surf.geometry is not None, (
                    f"表面 {i} 几何信息丢失"
                )
                
                assert_allclose(
                    loaded_surf.geometry.vertex_position,
                    orig_surf.geometry.vertex_position,
                    rtol=1e-10,
                    err_msg=f"表面 {i} 顶点位置不一致",
                )
                
                assert_allclose(
                    loaded_surf.geometry.surface_normal,
                    orig_surf.geometry.surface_normal,
                    rtol=1e-10,
                    err_msg=f"表面 {i} 法向量不一致",
                )
                
                # 处理无穷大曲率半径
                if np.isinf(orig_surf.geometry.radius):
                    assert np.isinf(loaded_surf.geometry.radius), (
                        f"表面 {i} 曲率半径应为无穷大"
                    )
                else:
                    assert_allclose(
                        loaded_surf.geometry.radius,
                        orig_surf.geometry.radius,
                        rtol=1e-10,
                        err_msg=f"表面 {i} 曲率半径不一致",
                    )
                
                assert loaded_surf.geometry.is_mirror == orig_surf.geometry.is_mirror, (
                    f"表面 {i} is_mirror 标志不一致"
                )
        
    finally:
        shutil.rmtree(temp_dir)


# ============================================================================
# 边界条件测试
# ============================================================================

def test_save_load_creates_directory():
    """
    测试保存时自动创建目录。
    """
    # 创建光学系统
    system = create_simple_flat_mirror_system()
    source = create_valid_source()
    
    # 执行仿真
    result = simulate(system, source, verbose=False)
    assert result.success
    
    # 使用临时目录
    temp_dir = tempfile.mkdtemp()
    try:
        # 使用嵌套路径
        save_path = os.path.join(temp_dir, "nested", "path", "result")
        
        # 保存结果（应自动创建目录）
        result.save(save_path)
        
        # 验证目录已创建
        assert os.path.exists(save_path)
        
        # 验证可以加载
        loaded_result = SimulationResult.load(save_path)
        assert loaded_result.success == result.success
        
    finally:
        shutil.rmtree(temp_dir)


def test_save_load_with_special_characters_in_surface_name():
    """
    测试表面名称包含特殊字符时的保存/加载。
    """
    # 创建光学系统
    system = OpticalSystem("Test System with Special Name")
    system.add_flat_mirror(z=50.0)
    
    source = create_valid_source()
    
    # 执行仿真
    result = simulate(system, source, verbose=False)
    assert result.success
    
    # 使用临时目录
    temp_dir = tempfile.mkdtemp()
    try:
        save_path = os.path.join(temp_dir, "test_result")
        
        # 保存结果
        result.save(save_path)
        
        # 加载结果
        loaded_result = SimulationResult.load(save_path)
        
        # 验证表面数量一致
        assert len(loaded_result.surfaces) == len(result.surfaces)
        
    finally:
        shutil.rmtree(temp_dir)


def test_save_load_preserves_error_message():
    """
    测试保存/加载保留错误信息（对于成功的仿真，错误信息应为空）。
    """
    # 创建光学系统
    system = create_simple_flat_mirror_system()
    source = create_valid_source()
    
    # 执行仿真
    result = simulate(system, source, verbose=False)
    assert result.success
    
    # 使用临时目录
    temp_dir = tempfile.mkdtemp()
    try:
        save_path = os.path.join(temp_dir, "test_result")
        
        # 保存结果
        result.save(save_path)
        
        # 加载结果
        loaded_result = SimulationResult.load(save_path)
        
        # 验证错误信息一致
        assert loaded_result.error_message == result.error_message
        
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
