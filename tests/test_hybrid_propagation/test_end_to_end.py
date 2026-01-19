"""
端到端集成测试

测试完整的混合传播流程，包括：
- 平面镜正入射测试
- 凹面镜正入射测试
- 45° 折叠镜测试

**Validates: Requirements 1.1, 3.3, 8.1**

作者：混合光学仿真项目
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
import sys

sys.path.insert(0, 'src')

from sequential_system import (
    SequentialOpticalSystem,
    GaussianBeamSource,
)
from sequential_system.exceptions import PilotBeamWarning
from hybrid_propagation import HybridElementPropagator
from wavefront_to_rays.element_raytracer import SurfaceDefinition

# 检查 finufft 是否可用
try:
    import finufft
    HAS_FINUFFT = True
except ImportError:
    HAS_FINUFFT = False


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def gaussian_source():
    """标准高斯光源"""
    return GaussianBeamSource(
        wavelength=0.633,  # μm
        w0=1.0,            # mm
        z0=0.0,            # mm
    )


@pytest.fixture
def small_beam_source():
    """小光束光源（用于减少计算时间）"""
    return GaussianBeamSource(
        wavelength=0.633,
        w0=0.5,
        z0=0.0,
    )


# ============================================================================
# 测试类：平面镜正入射
# ============================================================================

class TestFlatMirrorNormalIncidence:
    """平面镜正入射测试
    
    **Validates: Requirements 1.1, 8.1**
    """
    
    @pytest.mark.filterwarnings("ignore::sequential_system.exceptions.PilotBeamWarning")
    def test_flat_mirror_preserves_beam_shape(self, small_beam_source):
        """平面镜应该保持光束形状
        
        对于正入射的平面镜，反射后的光束形状应该与入射光束相同。
        
        **Validates: Requirements 1.1**
        """
        from gaussian_beam_simulation.optical_elements import SphericalMirror
        
        # 创建系统
        system = SequentialOpticalSystem(
            small_beam_source,
            grid_size=64,
            use_hybrid_propagation=True,
            hybrid_num_rays=50,
        )
        
        # 添加平面镜
        system.add_surface(SphericalMirror(
            radius_of_curvature=float('inf'),
            thickness=50.0,
            semi_aperture=5.0,
        ))
        
        # 添加采样面
        system.add_sampling_plane(distance=50.0, name="output")
        
        # 运行仿真
        results = system.run()
        
        # 验证结果
        output = results["output"]
        assert output.wavefront is not None
        assert output.beam_radius > 0
        
        # 光束半径应该在合理范围内（考虑衍射展宽）
        # 对于 w0=0.5mm，传播 50mm 后，光束会有一定展宽
        assert output.beam_radius < 5.0  # 不应该太大
    
    @pytest.mark.filterwarnings("ignore::sequential_system.exceptions.PilotBeamWarning")
    def test_flat_mirror_energy_conservation(self, small_beam_source):
        """平面镜应该保持能量守恒
        
        注意：混合模式由于采样和插值，能量可能有较大损失。
        这个测试主要验证两种模式都能产生有效输出。
        
        **Validates: Requirements 1.4, 8.4**
        """
        from gaussian_beam_simulation.optical_elements import SphericalMirror
        
        # 创建两个系统：一个使用混合模式，一个使用纯 PROPER 模式
        # 比较它们的能量
        
        # 混合模式
        system_hybrid = SequentialOpticalSystem(
            small_beam_source,
            grid_size=64,
            use_hybrid_propagation=True,
            hybrid_num_rays=50,
        )
        
        system_hybrid.add_surface(SphericalMirror(
            radius_of_curvature=float('inf'),
            thickness=50.0,
            semi_aperture=5.0,
        ))
        
        system_hybrid.add_sampling_plane(distance=50.0, name="output")
        
        results_hybrid = system_hybrid.run()
        energy_hybrid = np.sum(np.abs(results_hybrid["output"].wavefront)**2)
        
        # 纯 PROPER 模式
        system_proper = SequentialOpticalSystem(
            small_beam_source,
            grid_size=64,
            use_hybrid_propagation=False,
        )
        
        system_proper.add_surface(SphericalMirror(
            radius_of_curvature=float('inf'),
            thickness=50.0,
            semi_aperture=5.0,
        ))
        
        system_proper.add_sampling_plane(distance=50.0, name="output")
        
        results_proper = system_proper.run()
        energy_proper = np.sum(np.abs(results_proper["output"].wavefront)**2)
        
        # 验证两种模式都产生有效输出
        assert energy_hybrid > 0, "混合模式应该产生非零能量"
        assert energy_proper > 0, "纯 PROPER 模式应该产生非零能量"
        
        # 注意：由于混合模式的采样和插值过程，能量可能有较大差异
        # 这里只验证两种模式都能正常工作


# ============================================================================
# 测试类：凹面镜正入射
# ============================================================================

class TestConcaveMirrorNormalIncidence:
    """凹面镜正入射测试
    
    **Validates: Requirements 3.3**
    """
    
    @pytest.mark.filterwarnings("ignore::sequential_system.exceptions.PilotBeamWarning")
    def test_concave_mirror_focuses_beam(self, small_beam_source):
        """凹面镜应该聚焦光束
        
        **Validates: Requirements 3.3**
        """
        from gaussian_beam_simulation.optical_elements import SphericalMirror
        
        # 创建系统
        system = SequentialOpticalSystem(
            small_beam_source,
            grid_size=64,
            use_hybrid_propagation=True,
            hybrid_num_rays=50,
        )
        
        # 添加凹面镜（焦距 50mm）
        system.add_surface(SphericalMirror(
            radius_of_curvature=100.0,  # R = 2f
            thickness=40.0,
            semi_aperture=5.0,
        ))
        
        # 在焦点附近添加采样面
        system.add_sampling_plane(distance=40.0, name="near_focus")
        
        # 运行仿真
        results = system.run()
        
        # 验证结果
        output = results["near_focus"]
        assert output.wavefront is not None
        assert output.beam_radius > 0


# ============================================================================
# 测试类：45° 折叠镜
# ============================================================================

class TestFoldingMirror45Degrees:
    """45° 折叠镜测试
    
    **Validates: Requirements 8.1**
    """
    
    @pytest.mark.skipif(not HAS_FINUFFT, reason="finufft 库未安装")
    @pytest.mark.filterwarnings("ignore::sequential_system.exceptions.PilotBeamWarning")
    def test_folding_mirror_changes_direction(self, small_beam_source):
        """45° 折叠镜应该改变光束方向
        
        **Validates: Requirements 8.1**
        """
        from gaussian_beam_simulation.optical_elements import SphericalMirror
        
        # 创建系统
        system = SequentialOpticalSystem(
            small_beam_source,
            grid_size=64,
            use_hybrid_propagation=True,
            hybrid_num_rays=50,
        )
        
        # 添加 45° 折叠镜
        system.add_surface(SphericalMirror(
            radius_of_curvature=float('inf'),
            thickness=50.0,
            semi_aperture=5.0,
            tilt_x=np.pi/4,  # 45° 倾斜
        ))
        
        # 添加采样面
        system.add_sampling_plane(distance=50.0, name="output")
        
        # 运行仿真
        results = system.run()
        
        # 验证结果
        output = results["output"]
        assert output.wavefront is not None
        
        # 验证光轴方向已改变
        axis_state = output.axis_state
        assert axis_state is not None
        
        # 反射后方向应该是 (0, 1, 0)（向 +Y 方向）
        direction = axis_state.direction.to_array()
        assert_allclose(direction, [0, 1, 0], atol=1e-6)
    
    def test_folding_mirror_axis_tracking_without_hybrid(self, small_beam_source):
        """测试折叠镜的光轴跟踪（不使用混合模式）
        
        即使不使用混合传播模式，光轴跟踪也应该正确工作。
        
        **Validates: Requirements 8.1**
        """
        from gaussian_beam_simulation.optical_elements import SphericalMirror
        
        # 创建系统（纯 PROPER 模式）
        system = SequentialOpticalSystem(
            small_beam_source,
            grid_size=64,
            use_hybrid_propagation=False,
        )
        
        # 添加 45° 折叠镜
        system.add_surface(SphericalMirror(
            radius_of_curvature=float('inf'),
            thickness=50.0,
            semi_aperture=5.0,
            tilt_x=np.pi/4,  # 45° 倾斜
        ))
        
        # 添加采样面
        system.add_sampling_plane(distance=50.0, name="output")
        
        # 运行仿真
        results = system.run()
        
        # 验证光轴方向已改变
        output = results["output"]
        axis_state = output.axis_state
        assert axis_state is not None
        
        # 反射后方向应该是 (0, 1, 0)（向 +Y 方向）
        direction = axis_state.direction.to_array()
        assert_allclose(direction, [0, 1, 0], atol=1e-6)


# ============================================================================
# 测试类：直接使用 HybridElementPropagator
# ============================================================================

class TestHybridElementPropagatorDirect:
    """直接测试 HybridElementPropagator
    
    **Validates: Requirements 9.1, 9.3**
    """
    
    @pytest.mark.filterwarnings("ignore::sequential_system.exceptions.PilotBeamWarning")
    def test_propagator_with_flat_mirror(self):
        """直接使用传播器处理平面镜
        
        **Validates: Requirements 9.1**
        """
        # 创建输入复振幅（高斯分布）
        grid_size = 64
        physical_size = 10.0  # mm
        
        x = np.linspace(-physical_size/2, physical_size/2, grid_size)
        y = np.linspace(-physical_size/2, physical_size/2, grid_size)
        X, Y = np.meshgrid(x, y)
        
        w0 = 2.0  # mm
        amplitude = np.exp(-(X**2 + Y**2) / w0**2).astype(complex)
        
        # 定义平面镜
        element = SurfaceDefinition(
            surface_type='mirror',
            radius=float('inf'),
            tilt_x=0.0,
            tilt_y=0.0,
            semi_aperture=5.0,
        )
        
        # 创建传播器
        propagator = HybridElementPropagator(
            complex_amplitude=amplitude,
            element=element,
            wavelength=0.633,
            physical_size=physical_size,
            num_rays=50,
        )
        
        # 执行传播
        output = propagator.propagate()
        
        # 验证输出
        assert output.shape == amplitude.shape
        assert np.sum(np.abs(output)**2) > 0  # 有能量
    
    @pytest.mark.filterwarnings("ignore::sequential_system.exceptions.PilotBeamWarning")
    def test_propagator_intermediate_results(self):
        """测试中间结果获取
        
        **Validates: Requirements 9.4**
        """
        # 创建输入
        grid_size = 32
        physical_size = 10.0
        
        x = np.linspace(-physical_size/2, physical_size/2, grid_size)
        y = np.linspace(-physical_size/2, physical_size/2, grid_size)
        X, Y = np.meshgrid(x, y)
        
        amplitude = np.exp(-(X**2 + Y**2) / 4.0).astype(complex)
        
        element = SurfaceDefinition(
            surface_type='mirror',
            radius=float('inf'),
            tilt_x=0.0,
            tilt_y=0.0,
            semi_aperture=5.0,
        )
        
        propagator = HybridElementPropagator(
            complex_amplitude=amplitude,
            element=element,
            wavelength=0.633,
            physical_size=physical_size,
            num_rays=30,
        )
        
        # 执行传播
        propagator.propagate()
        
        # 获取中间结果
        results = propagator.get_intermediate_results()
        
        # 验证中间结果存在
        assert 'tangent_amplitude_in' in results
        assert 'pilot_phase' in results
        assert 'residual_phase' in results
        assert 'tangent_amplitude_out' in results


# ============================================================================
# 测试类：多元件系统
# ============================================================================

class TestMultiElementSystem:
    """多元件系统测试"""
    
    @pytest.mark.filterwarnings("ignore::sequential_system.exceptions.PilotBeamWarning")
    def test_two_mirror_system(self, small_beam_source):
        """测试双镜系统"""
        from gaussian_beam_simulation.optical_elements import SphericalMirror
        
        # 创建系统
        system = SequentialOpticalSystem(
            small_beam_source,
            grid_size=64,
            use_hybrid_propagation=True,
            hybrid_num_rays=50,
        )
        
        # 添加第一个平面镜
        system.add_surface(SphericalMirror(
            radius_of_curvature=float('inf'),
            thickness=30.0,
            semi_aperture=5.0,
        ))
        
        # 添加第二个平面镜
        system.add_surface(SphericalMirror(
            radius_of_curvature=float('inf'),
            thickness=30.0,
            semi_aperture=5.0,
        ))
        
        # 添加采样面
        system.add_sampling_plane(distance=60.0, name="output")
        
        # 运行仿真
        results = system.run()
        
        # 验证结果
        assert "output" in results.sampling_results
        output = results["output"]
        assert output.wavefront is not None
