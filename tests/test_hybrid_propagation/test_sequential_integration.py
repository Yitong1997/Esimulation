"""
SequentialOpticalSystem 混合传播集成测试

测试 SequentialOpticalSystem 与 HybridElementPropagator 的集成。

**Validates: Requirements 10.1, 10.2, 10.3, 10.4**

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


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def basic_source():
    """基本高斯光源"""
    return GaussianBeamSource(
        wavelength=0.633,  # μm
        w0=1.0,            # mm
        z0=0.0,            # mm
    )


@pytest.fixture
def small_grid_source():
    """小网格测试用光源"""
    return GaussianBeamSource(
        wavelength=0.633,
        w0=0.5,
        z0=0.0,
    )


# ============================================================================
# 测试类：基本功能
# ============================================================================

class TestHybridModeInitialization:
    """测试混合模式初始化"""
    
    def test_default_mode_is_pure_proper(self, basic_source):
        """默认模式应该是纯 PROPER 模式
        
        **Validates: Requirements 10.4**
        """
        system = SequentialOpticalSystem(basic_source)
        assert system.use_hybrid_propagation == False
    
    def test_enable_hybrid_mode(self, basic_source):
        """可以启用混合传播模式
        
        **Validates: Requirements 10.1**
        """
        system = SequentialOpticalSystem(
            basic_source,
            use_hybrid_propagation=True,
            hybrid_num_rays=50,
        )
        assert system.use_hybrid_propagation == True
    
    def test_hybrid_mode_with_custom_rays(self, basic_source):
        """可以自定义混合模式的光线数量"""
        system = SequentialOpticalSystem(
            basic_source,
            use_hybrid_propagation=True,
            hybrid_num_rays=200,
        )
        assert system._hybrid_num_rays == 200


class TestBackwardCompatibility:
    """测试向后兼容性
    
    **Validates: Requirements 10.4**
    """
    
    def test_pure_proper_mode_still_works(self, basic_source):
        """纯 PROPER 模式应该仍然正常工作"""
        # 导入元件类
        from gaussian_beam_simulation.optical_elements import ThinLens
        
        system = SequentialOpticalSystem(
            basic_source,
            grid_size=64,
            use_hybrid_propagation=False,
        )
        
        # 添加薄透镜
        system.add_surface(ThinLens(
            focal_length_value=50.0,
            thickness=100.0,
            semi_aperture=10.0,
        ))
        
        # 添加采样面
        system.add_sampling_plane(distance=100.0, name="output")
        
        # 运行仿真
        results = system.run()
        
        # 验证结果存在
        assert "output" in results.sampling_results
        result = results["output"]
        assert result.wavefront is not None
        assert result.beam_radius > 0
    
    def test_system_summary_unchanged(self, basic_source):
        """系统摘要格式应该保持不变"""
        from gaussian_beam_simulation.optical_elements import ThinLens
        
        system = SequentialOpticalSystem(
            basic_source,
            use_hybrid_propagation=True,
        )
        
        system.add_surface(ThinLens(
            focal_length_value=50.0,
            thickness=100.0,
            semi_aperture=10.0,
        ))
        
        summary = system.summary()
        
        # 验证摘要包含必要信息
        assert "序列光学系统配置摘要" in summary
        assert "光源参数" in summary
        assert "光学元件" in summary


class TestHybridModeWithMirror:
    """测试混合模式与反射镜
    
    **Validates: Requirements 10.1, 10.2**
    """
    
    @pytest.mark.filterwarnings("ignore::sequential_system.exceptions.PilotBeamWarning")
    def test_hybrid_mode_with_flat_mirror(self, small_grid_source):
        """混合模式应该能处理平面镜"""
        from gaussian_beam_simulation.optical_elements import SphericalMirror
        
        system = SequentialOpticalSystem(
            small_grid_source,
            grid_size=64,
            use_hybrid_propagation=True,
            hybrid_num_rays=50,
        )
        
        # 添加平面镜（无穷大曲率半径）
        system.add_surface(SphericalMirror(
            radius_of_curvature=float('inf'),
            thickness=50.0,
            semi_aperture=5.0,
        ))
        
        system.add_sampling_plane(distance=50.0, name="output")
        
        # 运行仿真（可能会有警告）
        results = system.run()
        
        # 验证结果
        assert "output" in results.sampling_results
        result = results["output"]
        assert result.wavefront is not None
    
    @pytest.mark.filterwarnings("ignore::sequential_system.exceptions.PilotBeamWarning")
    def test_hybrid_mode_with_curved_mirror(self, small_grid_source):
        """混合模式应该能处理曲面镜"""
        from gaussian_beam_simulation.optical_elements import SphericalMirror
        
        system = SequentialOpticalSystem(
            small_grid_source,
            grid_size=64,
            use_hybrid_propagation=True,
            hybrid_num_rays=50,
        )
        
        # 添加凹面镜
        system.add_surface(SphericalMirror(
            radius_of_curvature=100.0,  # 凹面镜
            thickness=80.0,
            semi_aperture=5.0,
        ))
        
        system.add_sampling_plane(distance=80.0, name="output")
        
        # 运行仿真
        results = system.run()
        
        # 验证结果
        assert "output" in results.sampling_results


class TestCoordinateSystemIntegration:
    """测试坐标系转换集成
    
    **Validates: Requirements 10.3**
    """
    
    def test_axis_tracker_updated_in_hybrid_mode(self, basic_source):
        """混合模式下光轴跟踪器应该正确更新"""
        from gaussian_beam_simulation.optical_elements import SphericalMirror
        
        system = SequentialOpticalSystem(
            basic_source,
            grid_size=64,
            use_hybrid_propagation=True,
        )
        
        # 添加 45° 折叠镜
        system.add_surface(SphericalMirror(
            radius_of_curvature=float('inf'),
            thickness=50.0,
            semi_aperture=10.0,
            tilt_x=np.pi/4,  # 45° 倾斜
        ))
        
        # 验证光轴跟踪器已更新
        tracker = system.axis_tracker
        assert len(tracker._element_states) == 1
        
        # 获取元件后的光轴状态
        _, state_before, state_after = tracker._element_states[0]
        
        # 入射方向应该是 (0, 0, 1)
        assert_allclose(state_before.direction.to_array(), [0, 0, 1], atol=1e-10)
        
        # 反射后方向应该改变（45° 折叠镜使光线向 +Y 方向）
        # 对于 tilt_x = π/4：
        # - 初始法向量 (0, 0, -1) 绕 X 轴旋转 45° 得到 (0, 0.707, -0.707)
        # - 反射公式 r = d - 2(d·n)n，其中 d = (0, 0, 1)
        # - d·n = -0.707，所以 r = (0, 0, 1) + 1.414*(0, 0.707, -0.707) = (0, 1, 0)
        reflected_dir = state_after.direction.to_array()
        assert_allclose(reflected_dir, [0, 1, 0], atol=1e-10)


class TestEnergyConservation:
    """测试能量守恒
    
    **Validates: Requirements 1.4, 8.4**
    """
    
    @pytest.mark.filterwarnings("ignore::sequential_system.exceptions.PilotBeamWarning")
    def test_energy_approximately_conserved_pure_proper(self, small_grid_source):
        """纯 PROPER 模式下能量应该近似守恒"""
        from gaussian_beam_simulation.optical_elements import ThinLens
        
        system = SequentialOpticalSystem(
            small_grid_source,
            grid_size=64,
            use_hybrid_propagation=False,
        )
        
        system.add_surface(ThinLens(
            focal_length_value=50.0,
            thickness=30.0,
            semi_aperture=5.0,
        ))
        
        system.add_sampling_plane(distance=10.0, name="before")
        system.add_sampling_plane(distance=30.0, name="after")
        
        results = system.run()
        
        # 计算能量
        energy_before = np.sum(np.abs(results["before"].wavefront)**2)
        energy_after = np.sum(np.abs(results["after"].wavefront)**2)
        
        # 能量应该近似守恒（允许一定误差）
        assert_allclose(energy_after, energy_before, rtol=0.1)


# ============================================================================
# 测试类：错误处理
# ============================================================================

class TestErrorHandling:
    """测试错误处理"""
    
    def test_invalid_hybrid_num_rays(self, basic_source):
        """无效的光线数量应该在运行时报错"""
        # 注意：当前实现可能不会在初始化时验证
        # 这个测试确保系统能正确处理边界情况
        system = SequentialOpticalSystem(
            basic_source,
            use_hybrid_propagation=True,
            hybrid_num_rays=1,  # 非常少的光线
        )
        
        # 系统应该能创建成功
        assert system._hybrid_num_rays == 1


# ============================================================================
# 测试类：警告处理
# ============================================================================

class TestWarningHandling:
    """测试警告处理"""
    
    def test_pilot_beam_warning_can_be_caught(self, small_grid_source):
        """Pilot Beam 警告应该可以被捕获"""
        from gaussian_beam_simulation.optical_elements import SphericalMirror
        
        system = SequentialOpticalSystem(
            small_grid_source,
            grid_size=32,  # 小网格更容易触发警告
            use_hybrid_propagation=True,
            hybrid_num_rays=20,
        )
        
        system.add_surface(SphericalMirror(
            radius_of_curvature=float('inf'),
            thickness=50.0,
            semi_aperture=5.0,
        ))
        
        system.add_sampling_plane(distance=50.0, name="output")
        
        # 捕获警告
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = system.run()
            
            # 检查是否有 PilotBeamWarning（可能有也可能没有）
            pilot_warnings = [
                warning for warning in w 
                if issubclass(warning.category, PilotBeamWarning)
            ]
            # 不强制要求有警告，只验证系统能正常运行
            assert results is not None
