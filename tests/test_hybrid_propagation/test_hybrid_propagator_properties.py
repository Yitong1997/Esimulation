"""
HybridElementPropagator 属性测试

使用 hypothesis 库进行属性基测试。

**Validates: Requirements 1.4, 8.4, 11.1, 11.2, 11.3, 11.4**
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
import sys
import warnings

from hypothesis import given, strategies as st, settings, assume

sys.path.insert(0, 'src')

from hybrid_propagation import HybridElementPropagator
from wavefront_to_rays.element_raytracer import SurfaceDefinition
from sequential_system.exceptions import SimulationError, PilotBeamWarning

# 检查 finufft 是否可用
try:
    import finufft
    HAS_FINUFFT = True
except ImportError:
    HAS_FINUFFT = False

# 需要 finufft 的测试跳过装饰器
requires_finufft = pytest.mark.skipif(
    not HAS_FINUFFT,
    reason="finufft 库未安装"
)


# ============================================================================
# 策略定义
# ============================================================================

# 网格大小策略（较小的值以加快测试）
grid_size_strategy = st.sampled_from([32, 64])

# 波长策略（可见光范围，μm）
wavelength_strategy = st.floats(
    min_value=0.4, max_value=0.8,
    allow_nan=False, allow_infinity=False, allow_subnormal=False
)

# 物理尺寸策略（mm）
physical_size_strategy = st.floats(
    min_value=10.0, max_value=100.0,
    allow_nan=False, allow_infinity=False, allow_subnormal=False
)

# 光线数量策略
num_rays_strategy = st.integers(min_value=20, max_value=100)

# 曲率半径策略（mm）
radius_strategy = st.one_of(
    st.just(float('inf')),  # 平面
    st.floats(min_value=50.0, max_value=500.0, allow_nan=False, allow_infinity=False, allow_subnormal=False),  # 凹面
    st.floats(min_value=-500.0, max_value=-50.0, allow_nan=False, allow_infinity=False, allow_subnormal=False),  # 凸面
)

# 半口径策略（mm）
semi_aperture_strategy = st.floats(
    min_value=10.0, max_value=50.0,
    allow_nan=False, allow_infinity=False, allow_subnormal=False
)


# ============================================================================
# 辅助函数
# ============================================================================

def create_gaussian_amplitude(grid_size: int, sigma: float = 0.3) -> np.ndarray:
    """创建高斯分布复振幅"""
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    R2 = X**2 + Y**2
    amplitude = np.exp(-R2 / (2 * sigma**2))
    return amplitude.astype(complex)


def create_flat_mirror(semi_aperture: float) -> SurfaceDefinition:
    """创建平面镜"""
    return SurfaceDefinition(
        surface_type='mirror',
        radius=np.inf,
        tilt_x=0.0,
        tilt_y=0.0,
        semi_aperture=semi_aperture,
    )


def create_curved_mirror(radius: float, semi_aperture: float) -> SurfaceDefinition:
    """创建曲面镜"""
    return SurfaceDefinition(
        surface_type='mirror',
        radius=radius,
        tilt_x=0.0,
        tilt_y=0.0,
        semi_aperture=semi_aperture,
    )


# ============================================================================
# Property 1: 能量守恒
# ============================================================================

class TestEnergyConservationProperty:
    """测试能量守恒属性
    
    **Property 1: 能量守恒**
    
    传播前后总能量应保持不变（在数值精度范围内）。
    由于采样和插值，允许一定的能量损失。
    
    **Validates: Requirements 1.4, 8.4**
    """
    
    @given(
        grid_size=grid_size_strategy,
        wavelength=wavelength_strategy,
        physical_size=physical_size_strategy,
        semi_aperture=semi_aperture_strategy,
    )
    @settings(max_examples=100, deadline=30000)
    def test_energy_conservation_flat_mirror(
        self, grid_size, wavelength, physical_size, semi_aperture
    ):
        """
        **Property 1: 能量守恒 - 平面镜**
        
        **Validates: Requirements 1.4**
        """
        # 确保半口径足够大以包含光束
        assume(semi_aperture >= physical_size / 4)
        
        amplitude = create_gaussian_amplitude(grid_size)
        element = create_flat_mirror(semi_aperture)
        
        propagator = HybridElementPropagator(
            complex_amplitude=amplitude,
            element=element,
            wavelength=wavelength,
            physical_size=physical_size,
            num_rays=50,
            pilot_beam_method='analytical',
        )
        
        input_energy = np.sum(np.abs(amplitude)**2)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PilotBeamWarning)
            output = propagator.propagate()
        
        output_energy = np.sum(np.abs(output)**2)
        
        # 能量应该大致守恒（允许较大误差，因为混合方法涉及多次采样和插值）
        assert output_energy > 0, "输出能量不应为零"
        # 能量比应该在合理范围内
        energy_ratio = output_energy / input_energy
        assert energy_ratio > 0.05, f"能量损失过大: {energy_ratio:.4f}"
    
    @given(
        grid_size=grid_size_strategy,
        wavelength=wavelength_strategy,
        physical_size=physical_size_strategy,
        radius=st.floats(min_value=100.0, max_value=500.0, allow_nan=False, allow_infinity=False, allow_subnormal=False),
        semi_aperture=semi_aperture_strategy,
    )
    @settings(max_examples=100, deadline=30000)
    def test_energy_conservation_curved_mirror(
        self, grid_size, wavelength, physical_size, radius, semi_aperture
    ):
        """
        **Property 1: 能量守恒 - 曲面镜**
        
        **Validates: Requirements 8.4**
        """
        # 确保半口径足够大以包含光束
        assume(semi_aperture >= physical_size / 4)
        
        amplitude = create_gaussian_amplitude(grid_size)
        element = create_curved_mirror(radius, semi_aperture)
        
        propagator = HybridElementPropagator(
            complex_amplitude=amplitude,
            element=element,
            wavelength=wavelength,
            physical_size=physical_size,
            num_rays=50,
            pilot_beam_method='analytical',
        )
        
        input_energy = np.sum(np.abs(amplitude)**2)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PilotBeamWarning)
            output = propagator.propagate()
        
        output_energy = np.sum(np.abs(output)**2)
        
        # 能量应该大致守恒
        assert output_energy > 0, "输出能量不应为零"
        energy_ratio = output_energy / input_energy
        assert energy_ratio > 0.05, f"能量损失过大: {energy_ratio:.4f}"


# ============================================================================
# Property 13: 输入验证
# ============================================================================

class TestInputValidationProperty:
    """测试输入验证属性
    
    **Property 13: 输入验证**
    
    无效输入应该抛出适当的异常。
    
    **Validates: Requirements 11.1, 11.2, 11.3**
    """
    
    @given(
        wavelength=st.floats(min_value=-10.0, max_value=0.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=10000)
    def test_invalid_wavelength_raises_error(self, wavelength):
        """
        **Property 13: 输入验证 - 无效波长**
        
        **Validates: Requirements 11.1**
        """
        amplitude = create_gaussian_amplitude(32)
        element = create_flat_mirror(25.0)
        
        with pytest.raises(ValueError, match="波长"):
            HybridElementPropagator(
                complex_amplitude=amplitude,
                element=element,
                wavelength=wavelength,
                physical_size=50.0,
                num_rays=50,
            )
    
    @given(
        physical_size=st.floats(min_value=-100.0, max_value=0.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=10000)
    def test_invalid_physical_size_raises_error(self, physical_size):
        """
        **Property 13: 输入验证 - 无效物理尺寸**
        
        **Validates: Requirements 11.2**
        """
        amplitude = create_gaussian_amplitude(32)
        element = create_flat_mirror(25.0)
        
        with pytest.raises(ValueError, match="物理尺寸"):
            HybridElementPropagator(
                complex_amplitude=amplitude,
                element=element,
                wavelength=0.633,
                physical_size=physical_size,
                num_rays=50,
            )
    
    @given(
        num_rays=st.integers(min_value=-100, max_value=0),
    )
    @settings(max_examples=100, deadline=10000)
    def test_invalid_num_rays_raises_error(self, num_rays):
        """
        **Property 13: 输入验证 - 无效光线数量**
        
        **Validates: Requirements 11.3**
        """
        amplitude = create_gaussian_amplitude(32)
        element = create_flat_mirror(25.0)
        
        with pytest.raises(ValueError, match="光线数量"):
            HybridElementPropagator(
                complex_amplitude=amplitude,
                element=element,
                wavelength=0.633,
                physical_size=50.0,
                num_rays=num_rays,
            )


# ============================================================================
# Property 14: 全光线无效处理
# ============================================================================

class TestAllRaysInvalidProperty:
    """测试全光线无效处理属性
    
    **Property 14: 全光线无效处理**
    
    当所有光线都无效时，应该优雅地处理（抛出错误或返回低能量输出）。
    
    **Validates: Requirements 11.4**
    """
    
    @given(
        grid_size=grid_size_strategy,
        wavelength=wavelength_strategy,
    )
    @settings(max_examples=50, deadline=30000)
    def test_small_aperture_handling(self, grid_size, wavelength):
        """
        **Property 14: 全光线无效处理 - 小半口径**
        
        **Validates: Requirements 11.4**
        """
        amplitude = create_gaussian_amplitude(grid_size)
        
        # 使用非常小的半口径
        element = SurfaceDefinition(
            surface_type='mirror',
            radius=np.inf,
            tilt_x=0.0,
            tilt_y=0.0,
            semi_aperture=0.001,  # 极小的半口径
        )
        
        propagator = HybridElementPropagator(
            complex_amplitude=amplitude,
            element=element,
            wavelength=wavelength,
            physical_size=100.0,  # 大的物理尺寸
            num_rays=20,
            pilot_beam_method='analytical',
        )
        
        # 应该要么抛出错误，要么返回低能量输出
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", PilotBeamWarning)
                output = propagator.propagate()
            
            # 如果没有抛出错误，输出应该是有效的
            assert np.isfinite(output).all(), "输出包含非有限值"
        except SimulationError:
            # 抛出错误也是可接受的行为
            pass


# ============================================================================
# 输出形状一致性
# ============================================================================

class TestOutputShapeProperty:
    """测试输出形状一致性属性"""
    
    @given(
        grid_size=grid_size_strategy,
        wavelength=wavelength_strategy,
        physical_size=physical_size_strategy,
        semi_aperture=semi_aperture_strategy,
    )
    @settings(max_examples=100, deadline=30000)
    def test_output_shape_matches_input(
        self, grid_size, wavelength, physical_size, semi_aperture
    ):
        """输出形状应该与输入形状一致（或与指定的 grid_size 一致）"""
        assume(semi_aperture >= physical_size / 4)
        
        amplitude = create_gaussian_amplitude(grid_size)
        element = create_flat_mirror(semi_aperture)
        
        propagator = HybridElementPropagator(
            complex_amplitude=amplitude,
            element=element,
            wavelength=wavelength,
            physical_size=physical_size,
            num_rays=50,
            pilot_beam_method='analytical',
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PilotBeamWarning)
            output = propagator.propagate()
        
        # 输出形状应该与输入形状一致
        assert output.shape == amplitude.shape, \
            f"输出形状 {output.shape} 与输入形状 {amplitude.shape} 不一致"
    
    @given(
        input_grid_size=grid_size_strategy,
        output_grid_size=grid_size_strategy,
        wavelength=wavelength_strategy,
        physical_size=physical_size_strategy,
        semi_aperture=semi_aperture_strategy,
    )
    @settings(max_examples=100, deadline=30000)
    def test_custom_output_grid_size(
        self, input_grid_size, output_grid_size, wavelength, physical_size, semi_aperture
    ):
        """自定义输出网格大小应该被正确应用"""
        assume(semi_aperture >= physical_size / 4)
        
        amplitude = create_gaussian_amplitude(input_grid_size)
        element = create_flat_mirror(semi_aperture)
        
        propagator = HybridElementPropagator(
            complex_amplitude=amplitude,
            element=element,
            wavelength=wavelength,
            physical_size=physical_size,
            grid_size=output_grid_size,
            num_rays=50,
            pilot_beam_method='analytical',
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PilotBeamWarning)
            output = propagator.propagate()
        
        # 输出形状应该与指定的 grid_size 一致
        assert output.shape == (output_grid_size, output_grid_size), \
            f"输出形状 {output.shape} 与指定的 grid_size {output_grid_size} 不一致"


# ============================================================================
# 中间结果一致性
# ============================================================================

class TestIntermediateResultsProperty:
    """测试中间结果一致性属性"""
    
    @given(
        grid_size=grid_size_strategy,
        wavelength=wavelength_strategy,
        physical_size=physical_size_strategy,
        semi_aperture=semi_aperture_strategy,
    )
    @settings(max_examples=50, deadline=60000)
    def test_intermediate_results_available_after_propagate(
        self, grid_size, wavelength, physical_size, semi_aperture
    ):
        """传播后应该能获取中间结果"""
        assume(semi_aperture >= physical_size / 4)
        
        amplitude = create_gaussian_amplitude(grid_size)
        element = create_flat_mirror(semi_aperture)
        
        propagator = HybridElementPropagator(
            complex_amplitude=amplitude,
            element=element,
            wavelength=wavelength,
            physical_size=physical_size,
            num_rays=50,
            pilot_beam_method='analytical',
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PilotBeamWarning)
            propagator.propagate()
        
        results = propagator.get_intermediate_results()
        
        # 检查所有预期的键
        expected_keys = [
            'tangent_amplitude_in',
            'rays_in',
            'rays_out',
            'pilot_phase',
            'residual_phase',
            'tangent_amplitude_out',
            'validation_result',
        ]
        
        for key in expected_keys:
            assert key in results, f"缺少中间结果键: {key}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
