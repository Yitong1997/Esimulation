"""
SequentialOpticalSystem 混合传播集成属性测试

使用 hypothesis 进行属性基测试，验证集成的正确性。

**Validates: Requirements 10.1, 10.2, 10.3, 10.4**

作者：混合光学仿真项目
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from hypothesis import given, strategies as st, settings, assume
import sys

sys.path.insert(0, 'src')

from sequential_system import (
    SequentialOpticalSystem,
    GaussianBeamSource,
)
from sequential_system.coordinate_tracking import (
    RayDirection,
    Position3D,
    OpticalAxisTracker,
    calculate_reflection_direction,
)
from sequential_system.exceptions import PilotBeamWarning


# ============================================================================
# 策略定义
# ============================================================================

# 波长策略（可见光范围，单位 μm）
wavelength_strategy = st.floats(
    min_value=0.4, max_value=0.8, allow_nan=False, allow_infinity=False,
    allow_subnormal=False
)

# 束腰半径策略（单位 mm）
beam_waist_strategy = st.floats(
    min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False,
    allow_subnormal=False
)

# 倾斜角度策略（弧度，限制在合理范围内）
tilt_angle_strategy = st.floats(
    min_value=-np.pi/4, max_value=np.pi/4, allow_nan=False, allow_infinity=False,
    allow_subnormal=False
)

# 小倾斜角度策略（用于测试正入射附近的行为）
small_tilt_strategy = st.floats(
    min_value=-0.1, max_value=0.1, allow_nan=False, allow_infinity=False,
    allow_subnormal=False
)

# 方向余弦策略
direction_component_strategy = st.floats(
    min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False,
    allow_subnormal=False
)


# ============================================================================
# Property 16: 坐标系转换往返一致性
# ============================================================================

class TestCoordinateTransformRoundTrip:
    """测试坐标系转换的往返一致性
    
    **Property 16: 坐标系转换往返一致性**
    **Validates: Requirements 10.3**
    """
    
    @given(
        tilt_x=small_tilt_strategy,
        tilt_y=small_tilt_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_reflection_direction_is_unit_vector(self, tilt_x, tilt_y):
        """反射后的方向向量应该是单位向量
        
        **Validates: Requirements 10.3**
        """
        incident = RayDirection(0, 0, 1)
        reflected = calculate_reflection_direction(incident, tilt_x, tilt_y)
        
        # 验证是单位向量
        norm = np.sqrt(reflected.L**2 + reflected.M**2 + reflected.N**2)
        assert_allclose(norm, 1.0, rtol=1e-10)
    
    @given(
        tilt_x=small_tilt_strategy,
        tilt_y=small_tilt_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_double_reflection_returns_to_original(self, tilt_x, tilt_y):
        """两次相同的反射应该返回原方向
        
        对于同一个表面，入射光线反射两次应该回到原方向。
        
        **Validates: Requirements 10.3**
        """
        incident = RayDirection(0, 0, 1)
        
        # 第一次反射
        reflected1 = calculate_reflection_direction(incident, tilt_x, tilt_y)
        
        # 第二次反射（使用相同的倾斜角度）
        # 注意：第二次反射时，入射方向是 reflected1
        # 但表面法向量的定义需要考虑入射方向的变化
        # 这里我们简化为：对于平面镜，两次反射应该回到原方向
        
        # 计算表面法向量
        normal = RayDirection(0, 0, -1)
        if tilt_x != 0:
            normal = normal.rotate_x(tilt_x)
        if tilt_y != 0:
            normal = normal.rotate_y(tilt_y)
        
        # 第二次反射
        reflected2 = reflected1.reflect(normal)
        
        # 验证回到原方向
        assert_allclose(
            reflected2.to_array(), incident.to_array(),
            rtol=1e-10, atol=1e-10
        )
    
    @given(
        L=direction_component_strategy,
        M=direction_component_strategy,
        N=direction_component_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_ray_direction_normalization(self, L, M, N):
        """RayDirection 应该自动归一化
        
        **Validates: Requirements 10.3**
        """
        # 跳过零向量
        norm = np.sqrt(L**2 + M**2 + N**2)
        assume(norm > 1e-10)
        
        direction = RayDirection(L, M, N)
        
        # 验证是单位向量
        result_norm = np.sqrt(direction.L**2 + direction.M**2 + direction.N**2)
        assert_allclose(result_norm, 1.0, rtol=1e-10)


class TestOpticalAxisTrackerProperties:
    """测试光轴跟踪器的属性
    
    **Validates: Requirements 10.3**
    """
    
    @given(
        distance=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, 
                          allow_infinity=False, allow_subnormal=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_propagation_preserves_direction(self, distance):
        """传播不改变方向
        
        在没有光学元件的情况下，传播只改变位置，不改变方向。
        
        **Validates: Requirements 10.3**
        """
        from sequential_system.coordinate_tracking import OpticalAxisState
        
        initial_state = OpticalAxisState(
            position=Position3D(0, 0, 0),
            direction=RayDirection(0, 0, 1),
            path_length=0.0,
        )
        
        propagated = initial_state.propagate(distance)
        
        # 方向应该不变
        assert_allclose(
            propagated.direction.to_array(),
            initial_state.direction.to_array(),
            rtol=1e-10
        )
        
        # 位置应该沿方向移动
        expected_position = np.array([0, 0, distance])
        assert_allclose(
            propagated.position.to_array(),
            expected_position,
            rtol=1e-10
        )
        
        # 光程应该增加
        assert_allclose(propagated.path_length, distance, rtol=1e-10)
    
    @given(
        tilt_x=tilt_angle_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_reflection_changes_direction(self, tilt_x):
        """反射改变方向
        
        对于非零倾斜的反射镜，反射应该改变光轴方向。
        
        **Validates: Requirements 10.3**
        """
        # 跳过接近零的倾斜（此时反射方向接近原方向）
        assume(abs(tilt_x) > 0.01)
        
        incident = RayDirection(0, 0, 1)
        reflected = calculate_reflection_direction(incident, tilt_x, 0)
        
        # 方向应该改变
        angle = incident.angle_with(reflected)
        assert angle > 0.01  # 方向应该有明显变化


class TestHybridModeProperties:
    """测试混合模式的属性
    
    **Validates: Requirements 10.1, 10.2, 10.4**
    """
    
    @given(
        wavelength=wavelength_strategy,
        beam_waist=beam_waist_strategy,
    )
    @settings(max_examples=50, deadline=None)
    def test_hybrid_mode_initialization_valid(self, wavelength, beam_waist):
        """混合模式应该能用有效参数初始化
        
        **Validates: Requirements 10.1**
        """
        source = GaussianBeamSource(
            wavelength=wavelength,
            w0=beam_waist,
            z0=0.0,
        )
        
        system = SequentialOpticalSystem(
            source,
            grid_size=32,
            use_hybrid_propagation=True,
            hybrid_num_rays=20,
        )
        
        assert system.use_hybrid_propagation == True
        assert system._hybrid_num_rays == 20
    
    @given(
        num_rays=st.integers(min_value=10, max_value=200),
    )
    @settings(max_examples=50, deadline=None)
    def test_hybrid_num_rays_stored_correctly(self, num_rays):
        """混合模式的光线数量应该正确存储
        
        **Validates: Requirements 10.1**
        """
        source = GaussianBeamSource(
            wavelength=0.633,
            w0=1.0,
            z0=0.0,
        )
        
        system = SequentialOpticalSystem(
            source,
            use_hybrid_propagation=True,
            hybrid_num_rays=num_rays,
        )
        
        assert system._hybrid_num_rays == num_rays


class TestBackwardCompatibilityProperties:
    """测试向后兼容性的属性
    
    **Validates: Requirements 10.4**
    """
    
    @given(
        grid_size=st.sampled_from([32, 64, 128]),
        beam_ratio=st.floats(min_value=0.1, max_value=0.9, allow_nan=False,
                            allow_infinity=False, allow_subnormal=False),
    )
    @settings(max_examples=30, deadline=None)
    def test_pure_proper_mode_parameters_preserved(self, grid_size, beam_ratio):
        """纯 PROPER 模式的参数应该被保留
        
        **Validates: Requirements 10.4**
        """
        source = GaussianBeamSource(
            wavelength=0.633,
            w0=1.0,
            z0=0.0,
        )
        
        system = SequentialOpticalSystem(
            source,
            grid_size=grid_size,
            beam_ratio=beam_ratio,
            use_hybrid_propagation=False,
        )
        
        assert system.grid_size == grid_size
        assert_allclose(system.beam_ratio, beam_ratio, rtol=1e-10)
        assert system.use_hybrid_propagation == False
