"""ElementRaytracer 光线位置传递测试

测试 ElementRaytracer 类中 get_input_positions() 和 get_output_positions() 方法的功能。

测试内容：
1. 正常情况：输入/输出位置正确保存和返回
2. 边界情况：空输入光线
3. 有效光线掩模正确
4. OPD 计算正确
5. RuntimeError 检查（未调用 trace() 时）

验证需求：
- 需求 1.1: 输出每条光线的 OPD（波长数）
- 需求 1.2: 保存输入光线位置，并提供 get_input_positions() 方法
- 需求 1.3: 提供 get_output_positions() 方法获取出射光线位置
- 需求 1.4: 保持现有 get_relative_opd_waves() 方法的功能
- 需求 1.5: 提供 get_valid_ray_mask() 方法获取有效光线掩模

作者：混合光学仿真项目
"""

import sys
import os

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from optiland.rays import RealRays

from wavefront_to_rays.element_raytracer import (
    SurfaceDefinition,
    ElementRaytracer,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def concave_mirror():
    """凹面镜表面定义
    
    曲率半径 200mm，焦距 100mm
    半口径 15mm
    """
    return SurfaceDefinition(
        surface_type='mirror',
        radius=200.0,
        thickness=0.0,
        material='mirror',
        semi_aperture=15.0
    )


@pytest.fixture
def flat_mirror():
    """平面镜表面定义"""
    return SurfaceDefinition(
        surface_type='mirror',
        radius=np.inf,
        thickness=0.0,
        material='mirror',
        semi_aperture=15.0
    )


@pytest.fixture
def raytracer_concave(concave_mirror):
    """使用凹面镜的光线追迹器（正入射配置）"""
    return ElementRaytracer(
        surfaces=[concave_mirror],
        wavelength=0.55,
        chief_ray_direction=(0, 0, 1),
        entrance_position=(0, 0, 0),
    )


@pytest.fixture
def raytracer_flat(flat_mirror):
    """使用平面镜的光线追迹器（正入射配置）"""
    return ElementRaytracer(
        surfaces=[flat_mirror],
        wavelength=0.55,
        chief_ray_direction=(0, 0, 1),
        entrance_position=(0, 0, 0),
    )


@pytest.fixture
def sample_rays_grid():
    """网格采样的输入光线
    
    5x5 网格，覆盖 [-5, 5] mm 范围
    所有光线沿 +Z 方向传播
    """
    n = 5
    x_coords = np.linspace(-5, 5, n)
    y_coords = np.linspace(-5, 5, n)
    X, Y = np.meshgrid(x_coords, y_coords)
    x = X.flatten()
    y = Y.flatten()
    n_rays = len(x)
    
    return RealRays(
        x=x,
        y=y,
        z=np.zeros(n_rays),
        L=np.zeros(n_rays),
        M=np.zeros(n_rays),
        N=np.ones(n_rays),
        intensity=np.ones(n_rays),
        wavelength=np.full(n_rays, 0.55),
    )


@pytest.fixture
def sample_rays_simple():
    """简单的输入光线
    
    5 条光线：中心 + 四个方向
    所有光线沿 +Z 方向传播
    """
    return RealRays(
        x=np.array([0.0, 1.0, -1.0, 0.0, 0.0]),
        y=np.array([0.0, 0.0, 0.0, 1.0, -1.0]),
        z=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        L=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        M=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        N=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        intensity=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        wavelength=np.array([0.55, 0.55, 0.55, 0.55, 0.55]),
    )


@pytest.fixture
def empty_rays():
    """空光线集合"""
    return RealRays(
        x=np.array([]),
        y=np.array([]),
        z=np.array([]),
        L=np.array([]),
        M=np.array([]),
        N=np.array([]),
        intensity=np.array([]),
        wavelength=np.array([]),
    )


# =============================================================================
# 测试类：RuntimeError 检查（未调用 trace() 时）
# =============================================================================

class TestRuntimeErrorBeforeTrace:
    """测试在调用 trace() 之前访问方法时的 RuntimeError
    
    验证需求：
    - 需求 1.2: get_input_positions() 在未追迹时抛出 RuntimeError
    - 需求 1.3: get_output_positions() 在未追迹时抛出 RuntimeError
    """
    
    def test_get_input_positions_before_trace(self, raytracer_concave):
        """测试在追迹前调用 get_input_positions() 抛出 RuntimeError
        
        Validates: 需求 1.2
        """
        with pytest.raises(RuntimeError, match="尚未执行光线追迹"):
            raytracer_concave.get_input_positions()
    
    def test_get_output_positions_before_trace(self, raytracer_concave):
        """测试在追迹前调用 get_output_positions() 抛出 RuntimeError
        
        Validates: 需求 1.3
        """
        with pytest.raises(RuntimeError, match="尚未执行光线追迹"):
            raytracer_concave.get_output_positions()
    
    def test_get_valid_ray_mask_before_trace(self, raytracer_concave):
        """测试在追迹前调用 get_valid_ray_mask() 抛出 RuntimeError
        
        Validates: 需求 1.5
        """
        with pytest.raises(RuntimeError, match="尚未执行光线追迹"):
            raytracer_concave.get_valid_ray_mask()
    
    def test_get_relative_opd_waves_before_trace(self, raytracer_concave):
        """测试在追迹前调用 get_relative_opd_waves() 抛出 RuntimeError
        
        Validates: 需求 1.4
        """
        with pytest.raises(RuntimeError, match="尚未执行光线追迹"):
            raytracer_concave.get_relative_opd_waves()


# =============================================================================
# 测试类：输入位置保存和返回
# =============================================================================

class TestInputPositions:
    """测试输入光线位置的保存和返回
    
    验证需求：
    - 需求 1.2: 保存输入光线位置，并提供 get_input_positions() 方法
    """
    
    def test_input_positions_saved_correctly(
        self, raytracer_concave, sample_rays_simple
    ):
        """测试输入光线位置正确保存
        
        Validates: 需求 1.2
        """
        # 执行追迹
        raytracer_concave.trace(sample_rays_simple)
        
        # 获取输入位置
        x_in, y_in = raytracer_concave.get_input_positions()
        
        # 验证输入位置与原始光线一致
        assert_allclose(x_in, sample_rays_simple.x, atol=1e-10)
        assert_allclose(y_in, sample_rays_simple.y, atol=1e-10)
    
    def test_input_positions_return_type(
        self, raytracer_concave, sample_rays_simple
    ):
        """测试 get_input_positions() 返回类型正确
        
        Validates: 需求 1.2
        """
        raytracer_concave.trace(sample_rays_simple)
        x_in, y_in = raytracer_concave.get_input_positions()
        
        # 验证返回类型为 numpy 数组
        assert isinstance(x_in, np.ndarray)
        assert isinstance(y_in, np.ndarray)
        
        # 验证数组长度
        assert len(x_in) == len(sample_rays_simple.x)
        assert len(y_in) == len(sample_rays_simple.y)

    
    def test_input_positions_grid_pattern(
        self, raytracer_concave, sample_rays_grid
    ):
        """测试网格采样的输入位置正确保存
        
        Validates: 需求 1.2
        """
        raytracer_concave.trace(sample_rays_grid)
        x_in, y_in = raytracer_concave.get_input_positions()
        
        # 验证输入位置与原始光线一致
        assert_allclose(x_in, sample_rays_grid.x, atol=1e-10)
        assert_allclose(y_in, sample_rays_grid.y, atol=1e-10)
    
    def test_input_positions_not_modified_by_trace(
        self, raytracer_concave, sample_rays_simple
    ):
        """测试输入位置在追迹过程中不被修改
        
        Validates: 需求 1.2
        """
        # 保存原始输入位置
        original_x = sample_rays_simple.x.copy()
        original_y = sample_rays_simple.y.copy()
        
        # 执行追迹
        raytracer_concave.trace(sample_rays_simple)
        
        # 获取保存的输入位置
        x_in, y_in = raytracer_concave.get_input_positions()
        
        # 验证输入位置与原始值一致
        assert_allclose(x_in, original_x, atol=1e-10)
        assert_allclose(y_in, original_y, atol=1e-10)


# =============================================================================
# 测试类：输出位置保存和返回
# =============================================================================

class TestOutputPositions:
    """测试输出光线位置的保存和返回
    
    验证需求：
    - 需求 1.3: 提供 get_output_positions() 方法获取出射光线位置
    """
    
    def test_output_positions_return_type(
        self, raytracer_concave, sample_rays_simple
    ):
        """测试 get_output_positions() 返回类型正确
        
        Validates: 需求 1.3
        """
        raytracer_concave.trace(sample_rays_simple)
        x_out, y_out = raytracer_concave.get_output_positions()
        
        # 验证返回类型为 numpy 数组
        assert isinstance(x_out, np.ndarray)
        assert isinstance(y_out, np.ndarray)
        
        # 验证数组长度
        assert len(x_out) == len(sample_rays_simple.x)
        assert len(y_out) == len(sample_rays_simple.y)
    
    def test_output_positions_flat_mirror(
        self, raytracer_flat, sample_rays_simple
    ):
        """测试平面镜的输出位置（应与输入位置相近）
        
        对于正入射的平面镜，输出位置应与输入位置相近
        （在出射面局部坐标系中）
        
        Validates: 需求 1.3
        """
        raytracer_flat.trace(sample_rays_simple)
        
        x_in, y_in = raytracer_flat.get_input_positions()
        x_out, y_out = raytracer_flat.get_output_positions()
        
        # 对于平面镜正入射，输出位置应与输入位置相近
        # 注意：由于坐标系变换，可能有符号变化
        assert_allclose(np.abs(x_out), np.abs(x_in), atol=0.1)
        assert_allclose(np.abs(y_out), np.abs(y_in), atol=0.1)

    
    def test_output_positions_concave_mirror(
        self, raytracer_concave, sample_rays_simple
    ):
        """测试凹面镜的输出位置
        
        对于凹面镜，输出位置应该有变化（由于曲面反射）
        
        Validates: 需求 1.3
        """
        raytracer_concave.trace(sample_rays_simple)
        
        x_out, y_out = raytracer_concave.get_output_positions()
        
        # 验证输出位置是有限值
        valid_mask = raytracer_concave.get_valid_ray_mask()
        assert np.all(np.isfinite(x_out[valid_mask]))
        assert np.all(np.isfinite(y_out[valid_mask]))
    
    def test_output_positions_grid_pattern(
        self, raytracer_concave, sample_rays_grid
    ):
        """测试网格采样的输出位置
        
        Validates: 需求 1.3
        """
        raytracer_concave.trace(sample_rays_grid)
        x_out, y_out = raytracer_concave.get_output_positions()
        
        # 验证输出位置数量与输入一致
        assert len(x_out) == len(sample_rays_grid.x)
        assert len(y_out) == len(sample_rays_grid.y)


# =============================================================================
# 测试类：空输入光线边界情况
# =============================================================================

class TestEmptyRays:
    """测试空输入光线的边界情况
    
    验证需求：
    - 需求 1.2, 1.3: 空输入时返回空数组
    """
    
    def test_empty_input_positions(self, raytracer_concave, empty_rays):
        """测试空输入光线的输入位置
        
        Validates: 需求 1.2
        """
        raytracer_concave.trace(empty_rays)
        x_in, y_in = raytracer_concave.get_input_positions()
        
        # 验证返回空数组
        assert len(x_in) == 0
        assert len(y_in) == 0
    
    def test_empty_output_positions(self, raytracer_concave, empty_rays):
        """测试空输入光线的输出位置
        
        Validates: 需求 1.3
        """
        raytracer_concave.trace(empty_rays)
        x_out, y_out = raytracer_concave.get_output_positions()
        
        # 验证返回空数组
        assert len(x_out) == 0
        assert len(y_out) == 0
    
    def test_empty_valid_mask(self, raytracer_concave, empty_rays):
        """测试空输入光线的有效掩模
        
        Validates: 需求 1.5
        """
        raytracer_concave.trace(empty_rays)
        valid_mask = raytracer_concave.get_valid_ray_mask()
        
        # 验证返回空数组
        assert len(valid_mask) == 0


# =============================================================================
# 测试类：有效光线掩模
# =============================================================================

class TestValidRayMask:
    """测试有效光线掩模功能
    
    验证需求：
    - 需求 1.5: 提供 get_valid_ray_mask() 方法获取有效光线掩模
    """
    
    def test_valid_mask_return_type(
        self, raytracer_concave, sample_rays_simple
    ):
        """测试 get_valid_ray_mask() 返回类型正确
        
        Validates: 需求 1.5
        """
        raytracer_concave.trace(sample_rays_simple)
        valid_mask = raytracer_concave.get_valid_ray_mask()
        
        # 验证返回类型为布尔数组
        assert isinstance(valid_mask, np.ndarray)
        assert valid_mask.dtype == bool
        
        # 验证数组长度
        assert len(valid_mask) == len(sample_rays_simple.x)
    
    def test_valid_mask_all_valid(
        self, raytracer_concave, sample_rays_simple
    ):
        """测试所有光线都有效的情况
        
        对于小光瞳的正入射，所有光线应该都有效
        
        Validates: 需求 1.5
        """
        raytracer_concave.trace(sample_rays_simple)
        valid_mask = raytracer_concave.get_valid_ray_mask()
        
        # 对于小光瞳，所有光线应该都有效
        assert np.all(valid_mask)
    
    def test_valid_mask_grid_pattern(
        self, raytracer_concave, sample_rays_grid
    ):
        """测试网格采样的有效掩模
        
        Validates: 需求 1.5
        """
        raytracer_concave.trace(sample_rays_grid)
        valid_mask = raytracer_concave.get_valid_ray_mask()
        
        # 验证掩模长度
        assert len(valid_mask) == len(sample_rays_grid.x)
        
        # 验证有效光线数量大于 0
        assert np.sum(valid_mask) > 0
    
    def test_valid_mask_consistency_with_positions(
        self, raytracer_concave, sample_rays_grid
    ):
        """测试有效掩模与位置数据的一致性
        
        有效光线的位置应该是有限值
        
        Validates: 需求 1.5
        """
        raytracer_concave.trace(sample_rays_grid)
        
        valid_mask = raytracer_concave.get_valid_ray_mask()
        x_out, y_out = raytracer_concave.get_output_positions()
        
        # 有效光线的位置应该是有限值
        assert np.all(np.isfinite(x_out[valid_mask]))
        assert np.all(np.isfinite(y_out[valid_mask]))


# =============================================================================
# 测试类：OPD 计算
# =============================================================================

class TestOPDCalculation:
    """测试 OPD 计算功能
    
    验证需求：
    - 需求 1.1: 输出每条光线的 OPD（波长数）
    - 需求 1.4: 保持现有 get_relative_opd_waves() 方法的功能
    """
    
    def test_opd_return_type(
        self, raytracer_concave, sample_rays_simple
    ):
        """测试 get_relative_opd_waves() 返回类型正确
        
        Validates: 需求 1.1, 1.4
        """
        raytracer_concave.trace(sample_rays_simple)
        opd_waves = raytracer_concave.get_relative_opd_waves()
        
        # 验证返回类型为 numpy 数组
        assert isinstance(opd_waves, np.ndarray)
        
        # 验证数组长度
        assert len(opd_waves) == len(sample_rays_simple.x)
    
    def test_opd_chief_ray_zero(
        self, raytracer_concave, sample_rays_simple
    ):
        """测试主光线的相对 OPD 为零
        
        主光线（中心光线）的相对 OPD 应该为零
        
        Validates: 需求 1.1, 1.4
        """
        raytracer_concave.trace(sample_rays_simple)
        opd_waves = raytracer_concave.get_relative_opd_waves()
        
        # 找到中心光线（x=0, y=0）
        x_in, y_in = raytracer_concave.get_input_positions()
        center_idx = np.argmin(x_in**2 + y_in**2)
        
        # 主光线的相对 OPD 应该接近零
        assert_allclose(opd_waves[center_idx], 0.0, atol=1e-6)
    
    def test_opd_flat_mirror_constant(
        self, raytracer_flat, sample_rays_simple
    ):
        """测试平面镜的 OPD 为常数
        
        对于平面镜正入射，所有光线的相对 OPD 应该接近零
        
        Validates: 需求 1.1, 1.4
        """
        raytracer_flat.trace(sample_rays_simple)
        opd_waves = raytracer_flat.get_relative_opd_waves()
        
        valid_mask = raytracer_flat.get_valid_ray_mask()
        valid_opd = opd_waves[valid_mask]
        
        # 平面镜的 OPD 应该接近常数（相对 OPD 接近零）
        opd_std = np.std(valid_opd)
        assert opd_std < 0.01, f"平面镜 OPD 标准差过大: {opd_std}"
    
    def test_opd_concave_mirror_pattern(
        self, raytracer_concave, sample_rays_grid
    ):
        """测试凹面镜的 OPD 分布模式
        
        对于凹面镜，边缘光线的 OPD 应该与中心光线不同
        
        Validates: 需求 1.1, 1.4
        """
        raytracer_concave.trace(sample_rays_grid)
        opd_waves = raytracer_concave.get_relative_opd_waves()
        
        valid_mask = raytracer_concave.get_valid_ray_mask()
        valid_opd = opd_waves[valid_mask]
        
        # 验证 OPD 是有限值
        assert np.all(np.isfinite(valid_opd))

    
    def test_opd_invalid_rays_nan(
        self, raytracer_concave, sample_rays_grid
    ):
        """测试无效光线的 OPD 为 NaN
        
        无效光线的 OPD 应该被设置为 NaN
        
        Validates: 需求 1.1, 1.4
        """
        raytracer_concave.trace(sample_rays_grid)
        opd_waves = raytracer_concave.get_relative_opd_waves()
        valid_mask = raytracer_concave.get_valid_ray_mask()
        
        # 如果有无效光线，其 OPD 应该为 NaN
        if not np.all(valid_mask):
            invalid_opd = opd_waves[~valid_mask]
            assert np.all(np.isnan(invalid_opd))


# =============================================================================
# 测试类：输入/输出位置与雅可比矩阵计算的兼容性
# =============================================================================

class TestJacobianCompatibility:
    """测试输入/输出位置与雅可比矩阵计算的兼容性
    
    验证输入/输出位置数据可以用于雅可比矩阵计算
    """
    
    def test_positions_same_length(
        self, raytracer_concave, sample_rays_grid
    ):
        """测试输入/输出位置数组长度相同
        
        雅可比矩阵计算需要输入/输出位置一一对应
        """
        raytracer_concave.trace(sample_rays_grid)
        
        x_in, y_in = raytracer_concave.get_input_positions()
        x_out, y_out = raytracer_concave.get_output_positions()
        
        # 验证长度相同
        assert len(x_in) == len(x_out)
        assert len(y_in) == len(y_out)
    
    def test_positions_valid_for_interpolation(
        self, raytracer_concave, sample_rays_grid
    ):
        """测试有效光线的位置可用于插值
        
        雅可比矩阵计算需要使用 RBF 插值，
        有效光线的位置应该是有限值
        """
        raytracer_concave.trace(sample_rays_grid)
        
        valid_mask = raytracer_concave.get_valid_ray_mask()
        x_in, y_in = raytracer_concave.get_input_positions()
        x_out, y_out = raytracer_concave.get_output_positions()
        
        # 有效光线的位置应该是有限值
        assert np.all(np.isfinite(x_in[valid_mask]))
        assert np.all(np.isfinite(y_in[valid_mask]))
        assert np.all(np.isfinite(x_out[valid_mask]))
        assert np.all(np.isfinite(y_out[valid_mask]))
    
    def test_sufficient_valid_rays_for_jacobian(
        self, raytracer_concave, sample_rays_grid
    ):
        """测试有足够的有效光线用于雅可比矩阵计算
        
        雅可比矩阵计算至少需要 4 条有效光线
        """
        raytracer_concave.trace(sample_rays_grid)
        
        valid_mask = raytracer_concave.get_valid_ray_mask()
        n_valid = np.sum(valid_mask)
        
        # 至少需要 4 条有效光线
        assert n_valid >= 4, f"有效光线数量不足: {n_valid} < 4"


# =============================================================================
# 测试类：倾斜入射情况
# =============================================================================

class TestTiltedIncidence:
    """测试倾斜入射情况下的位置传递
    
    验证倾斜入射时输入/输出位置正确保存
    """
    
    @pytest.fixture
    def tilted_raytracer(self, concave_mirror):
        """倾斜入射的光线追迹器"""
        angle = np.pi / 6  # 30 度
        direction = (0, np.sin(angle), np.cos(angle))
        
        return ElementRaytracer(
            surfaces=[concave_mirror],
            wavelength=0.55,
            chief_ray_direction=direction,
            entrance_position=(0, 0, 0),
        )
    
    def test_tilted_input_positions(
        self, tilted_raytracer, sample_rays_simple
    ):
        """测试倾斜入射时输入位置正确保存
        
        Validates: 需求 1.2
        """
        tilted_raytracer.trace(sample_rays_simple)
        x_in, y_in = tilted_raytracer.get_input_positions()
        
        # 验证输入位置与原始光线一致
        assert_allclose(x_in, sample_rays_simple.x, atol=1e-10)
        assert_allclose(y_in, sample_rays_simple.y, atol=1e-10)
    
    def test_tilted_output_positions(
        self, tilted_raytracer, sample_rays_simple
    ):
        """测试倾斜入射时输出位置正确返回
        
        Validates: 需求 1.3
        """
        tilted_raytracer.trace(sample_rays_simple)
        x_out, y_out = tilted_raytracer.get_output_positions()
        
        valid_mask = tilted_raytracer.get_valid_ray_mask()
        
        # 验证有效光线的输出位置是有限值
        assert np.all(np.isfinite(x_out[valid_mask]))
        assert np.all(np.isfinite(y_out[valid_mask]))
    
    def test_tilted_opd_calculation(
        self, tilted_raytracer, sample_rays_simple
    ):
        """测试倾斜入射时 OPD 计算正确
        
        Validates: 需求 1.1, 1.4
        """
        tilted_raytracer.trace(sample_rays_simple)
        opd_waves = tilted_raytracer.get_relative_opd_waves()
        
        valid_mask = tilted_raytracer.get_valid_ray_mask()
        valid_opd = opd_waves[valid_mask]
        
        # 验证 OPD 是有限值
        assert np.all(np.isfinite(valid_opd))


# =============================================================================
# 测试类：多次追迹
# =============================================================================

class TestMultipleTraces:
    """测试多次追迹时位置数据的更新
    
    验证每次追迹后位置数据正确更新
    """
    
    def test_positions_updated_on_retrace(
        self, raytracer_concave, sample_rays_simple, sample_rays_grid
    ):
        """测试重新追迹时位置数据更新
        
        Validates: 需求 1.2, 1.3
        """
        # 第一次追迹
        raytracer_concave.trace(sample_rays_simple)
        x_in_1, y_in_1 = raytracer_concave.get_input_positions()
        
        # 第二次追迹（使用不同的光线）
        raytracer_concave.trace(sample_rays_grid)
        x_in_2, y_in_2 = raytracer_concave.get_input_positions()
        
        # 验证位置数据已更新
        assert len(x_in_2) == len(sample_rays_grid.x)
        assert len(y_in_2) == len(sample_rays_grid.y)
        
        # 验证新的输入位置与新光线一致
        assert_allclose(x_in_2, sample_rays_grid.x, atol=1e-10)
        assert_allclose(y_in_2, sample_rays_grid.y, atol=1e-10)


# =============================================================================
# 运行测试
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
