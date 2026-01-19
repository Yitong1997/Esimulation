"""
可视化单元测试

测试 LayoutVisualizer 类。

**Validates: Requirements 7.6, 7.7**
"""

import sys
sys.path.insert(0, 'src')

import pytest
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

from sequential_system.system import SequentialOpticalSystem
from sequential_system.source import GaussianBeamSource
from sequential_system.visualization import LayoutVisualizer
from gaussian_beam_simulation.optical_elements import SphericalMirror, ThinLens


class TestLayoutVisualizer:
    """测试 LayoutVisualizer"""
    
    @pytest.fixture
    def simple_system(self):
        """创建简单系统"""
        source = GaussianBeamSource(wavelength=0.633, w0=1.0, z0=-50.0)
        system = SequentialOpticalSystem(source)
        
        system.add_surface(SphericalMirror(
            thickness=100.0, semi_aperture=15.0, radius_of_curvature=200.0
        ))
        system.add_sampling_plane(distance=150.0, name="focus")
        
        return system
    
    def test_draw_returns_figure_and_axes(self, simple_system):
        """
        **Validates: Requirements 7.7**
        
        验证 draw 返回 figure 和 axes
        """
        visualizer = LayoutVisualizer(simple_system)
        fig, ax = visualizer.draw(show=False)
        
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.axes import Axes
        
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        
        plt.close(fig)
    
    def test_draw_show_false_does_not_block(self, simple_system):
        """
        **Validates: Requirements 7.6**
        
        验证 show=False 不阻塞
        """
        import matplotlib.pyplot as plt
        
        visualizer = LayoutVisualizer(simple_system)
        
        # 这应该立即返回，不阻塞
        fig, ax = visualizer.draw(show=False)
        
        assert fig is not None
        assert ax is not None
        
        plt.close(fig)
    
    def test_draw_with_custom_figsize(self, simple_system):
        """验证自定义图形大小"""
        import matplotlib.pyplot as plt
        
        visualizer = LayoutVisualizer(simple_system)
        fig, ax = visualizer.draw(show=False, figsize=(8, 4))
        
        # 检查图形大小
        width, height = fig.get_size_inches()
        assert abs(width - 8) < 0.1
        assert abs(height - 4) < 0.1
        
        plt.close(fig)
    
    def test_system_draw_layout_method(self, simple_system):
        """验证系统的 draw_layout 方法"""
        import matplotlib.pyplot as plt
        
        fig, ax = simple_system.draw_layout(show=False)
        
        assert fig is not None
        assert ax is not None
        
        plt.close(fig)


class TestVisualizationWithElements:
    """测试带元件的可视化"""
    
    def test_draw_with_lens(self):
        """测试带透镜的可视化"""
        import matplotlib.pyplot as plt
        
        source = GaussianBeamSource(wavelength=0.633, w0=1.0)
        system = SequentialOpticalSystem(source)
        
        system.add_surface(ThinLens(
            thickness=100.0, semi_aperture=10.0, focal_length_value=50.0
        ))
        
        fig, ax = system.draw_layout(show=False)
        
        assert fig is not None
        plt.close(fig)
    
    def test_draw_with_multiple_elements(self):
        """测试带多个元件的可视化"""
        import matplotlib.pyplot as plt
        
        source = GaussianBeamSource(wavelength=0.633, w0=1.0)
        system = SequentialOpticalSystem(source)
        
        system.add_surface(ThinLens(
            thickness=50.0, semi_aperture=10.0, focal_length_value=50.0
        ))
        system.add_surface(SphericalMirror(
            thickness=100.0, semi_aperture=15.0, radius_of_curvature=200.0
        ))
        system.add_sampling_plane(distance=75.0, name="mid")
        system.add_sampling_plane(distance=150.0, name="end")
        
        fig, ax = system.draw_layout(show=False)
        
        assert fig is not None
        plt.close(fig)
    
    def test_draw_empty_system(self):
        """测试空系统的可视化"""
        import matplotlib.pyplot as plt
        
        source = GaussianBeamSource(wavelength=0.633, w0=1.0)
        system = SequentialOpticalSystem(source)
        
        fig, ax = system.draw_layout(show=False)
        
        assert fig is not None
        plt.close(fig)


class TestSpatialVisualization:
    """测试空间坐标可视化"""
    
    def test_draw_spatial_mode(self):
        """测试空间模式可视化"""
        import matplotlib.pyplot as plt
        import numpy as np
        from sequential_system import FlatMirror
        
        source = GaussianBeamSource(wavelength=0.633, w0=1.0)
        system = SequentialOpticalSystem(source)
        
        # 添加 45° 折叠镜
        system.add_surface(FlatMirror(
            thickness=50.0,
            semi_aperture=10.0,
            tilt_x=np.pi/4,  # 45° 倾斜
        ))
        system.add_sampling_plane(distance=50.0, name="after_fold")
        
        visualizer = LayoutVisualizer(system)
        fig, ax = visualizer.draw(show=False, mode="spatial")
        
        assert fig is not None
        assert ax is not None
        
        plt.close(fig)
    
    def test_draw_spatial_with_multiple_folds(self):
        """测试多次折叠的空间可视化"""
        import matplotlib.pyplot as plt
        import numpy as np
        from sequential_system import FlatMirror
        
        source = GaussianBeamSource(wavelength=0.633, w0=1.0)
        system = SequentialOpticalSystem(source)
        
        # 第一个折叠镜
        system.add_surface(FlatMirror(
            thickness=50.0,
            semi_aperture=10.0,
            tilt_x=np.pi/4,
        ))
        
        # 第二个折叠镜
        system.add_surface(FlatMirror(
            thickness=50.0,
            semi_aperture=10.0,
            tilt_x=np.pi/4,
        ))
        
        system.add_sampling_plane(distance=100.0, name="output")
        
        visualizer = LayoutVisualizer(system)
        fig, ax = visualizer.draw(show=False, mode="spatial")
        
        assert fig is not None
        plt.close(fig)
    
    def test_axis_tracker_integration(self):
        """测试光轴跟踪器集成"""
        import numpy as np
        from sequential_system import FlatMirror
        
        source = GaussianBeamSource(wavelength=0.633, w0=1.0)
        system = SequentialOpticalSystem(source)
        
        # 添加 45° 折叠镜
        system.add_surface(FlatMirror(
            thickness=50.0,
            semi_aperture=10.0,
            tilt_x=np.pi/4,
        ))
        
        # 验证光轴跟踪器存在
        assert system.axis_tracker is not None
        
        # 验证可以获取光轴状态
        state = system.axis_tracker.get_state_at_distance(50.0)
        assert state is not None
        
        # 验证折叠后光轴方向改变
        # 45° 折叠镜（tilt_x=π/4）将光束从 +Z 方向折叠
        # 根据反射定律，出射方向取决于表面法向量
        # 验证方向已经改变（不再沿 +Z）
        assert abs(state.direction.N) < 0.1  # N ≈ 0，不再沿 Z 方向
        # 验证光束现在主要沿 Y 方向
        assert abs(state.direction.M) > 0.9  # M ≈ ±1


class TestSamplingResultAxisState:
    """测试采样结果中的光轴状态"""
    
    def test_sampling_result_has_axis_state(self):
        """测试采样结果包含光轴状态"""
        import numpy as np
        from sequential_system import FlatMirror
        
        source = GaussianBeamSource(wavelength=0.633, w0=1.0)
        system = SequentialOpticalSystem(source)
        
        system.add_surface(FlatMirror(
            thickness=50.0,
            semi_aperture=10.0,
            tilt_x=np.pi/4,
        ))
        system.add_sampling_plane(distance=50.0, name="after_fold")
        
        results = system.run()
        result = results["after_fold"]
        
        # 验证采样结果包含光轴状态
        assert result.axis_state is not None
        
        # 验证光轴状态的位置和方向
        assert result.axis_state.position is not None
        assert result.axis_state.direction is not None
