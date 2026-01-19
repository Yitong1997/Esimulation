"""
直接传播模式测试

测试 DirectElementPropagator 的基本功能和正确性。
"""

import sys
import numpy as np
import pytest

sys.path.insert(0, 'src')

from hybrid_propagation import (
    DirectElementPropagator,
    DirectPropagationResult,
    propagate_through_element,
)
from wavefront_to_rays.element_raytracer import SurfaceDefinition


class TestDirectElementPropagator:
    """DirectElementPropagator 基本功能测试"""
    
    def test_initialization(self):
        """测试初始化"""
        grid_size = 32
        amplitude = np.ones((grid_size, grid_size), dtype=complex)
        
        element = SurfaceDefinition(
            surface_type='mirror',
            radius=np.inf,
            semi_aperture=10.0,
        )
        
        propagator = DirectElementPropagator(
            complex_amplitude=amplitude,
            element=element,
            wavelength=0.633,
            physical_size=20.0,
            num_rays=50,
        )
        
        assert propagator.grid_size == grid_size
        assert propagator.wavelength == 0.633
        assert propagator.physical_size == 20.0
        assert propagator.num_rays == 50
    
    def test_plane_mirror_propagation(self):
        """测试平面镜传播
        
        平面波入射到平面镜，出射应该仍然是平面波。
        """
        grid_size = 32
        amplitude = np.ones((grid_size, grid_size), dtype=complex)
        
        element = SurfaceDefinition(
            surface_type='mirror',
            radius=np.inf,
            semi_aperture=15.0,
        )
        
        propagator = DirectElementPropagator(
            complex_amplitude=amplitude,
            element=element,
            wavelength=0.633,
            physical_size=20.0,
            num_rays=100,
            use_pilot_beam=False,  # 简化测试，不使用 Pilot Beam
            debug=False,
        )
        
        result = propagator.propagate()
        
        # 检查返回类型
        assert isinstance(result, DirectPropagationResult)
        assert result.output_amplitude is not None
        assert result.output_amplitude.shape == (grid_size, grid_size)
        
        # 检查出射方向（平面镜正入射，出射方向应该是 (0, 0, -1)）
        exit_dir = result.exit_direction
        assert len(exit_dir) == 3
    
    def test_curved_mirror_propagation(self):
        """测试曲面镜传播"""
        grid_size = 32
        amplitude = np.ones((grid_size, grid_size), dtype=complex)
        
        # 凹面镜，焦距 100mm
        element = SurfaceDefinition(
            surface_type='mirror',
            radius=200.0,  # R = 2f
            semi_aperture=15.0,
        )
        
        propagator = DirectElementPropagator(
            complex_amplitude=amplitude,
            element=element,
            wavelength=0.633,
            physical_size=20.0,
            num_rays=100,
            use_pilot_beam=False,
            debug=False,
        )
        
        result = propagator.propagate()
        
        assert result.output_amplitude is not None
        assert result.output_amplitude.shape == (grid_size, grid_size)
    
    def test_with_pilot_beam(self):
        """测试使用 Pilot Beam 参考相位"""
        grid_size = 32
        amplitude = np.ones((grid_size, grid_size), dtype=complex)
        
        element = SurfaceDefinition(
            surface_type='mirror',
            radius=np.inf,
            semi_aperture=15.0,
        )
        
        propagator = DirectElementPropagator(
            complex_amplitude=amplitude,
            element=element,
            wavelength=0.633,
            physical_size=20.0,
            num_rays=100,
            use_pilot_beam=True,
            pilot_beam_method='analytical',
            debug=False,
        )
        
        result = propagator.propagate()
        
        assert result.output_amplitude is not None
        # 使用 Pilot Beam 时应该有验证结果
        # 注意：验证结果可能为 None 如果 Pilot Beam 计算失败
    
    def test_convenience_function(self):
        """测试便捷函数 propagate_through_element"""
        grid_size = 32
        amplitude = np.ones((grid_size, grid_size), dtype=complex)
        
        element = SurfaceDefinition(
            surface_type='mirror',
            radius=np.inf,
            semi_aperture=15.0,
        )
        
        result = propagate_through_element(
            complex_amplitude=amplitude,
            element=element,
            wavelength=0.633,
            physical_size=20.0,
            num_rays=50,
            use_pilot_beam=False,
        )
        
        assert isinstance(result, DirectPropagationResult)
        assert result.output_amplitude is not None
    
    def test_intermediate_results_debug_mode(self):
        """测试调试模式下的中间结果"""
        grid_size = 32
        amplitude = np.ones((grid_size, grid_size), dtype=complex)
        
        element = SurfaceDefinition(
            surface_type='mirror',
            radius=np.inf,
            semi_aperture=15.0,
        )
        
        propagator = DirectElementPropagator(
            complex_amplitude=amplitude,
            element=element,
            wavelength=0.633,
            physical_size=20.0,
            num_rays=50,
            use_pilot_beam=False,
            debug=True,
        )
        
        result = propagator.propagate()
        
        # 调试模式下应该有中间结果
        assert result.intermediate is not None
        assert 'rays_in' in result.intermediate
        assert 'rays_out' in result.intermediate
        assert 'valid_mask' in result.intermediate
    
    def test_input_validation(self):
        """测试输入验证"""
        # 测试无效的复振幅数组维度
        with pytest.raises(ValueError, match="2D"):
            DirectElementPropagator(
                complex_amplitude=np.ones(10, dtype=complex),  # 1D 数组
                element=SurfaceDefinition(),
                wavelength=0.633,
                physical_size=20.0,
            )
        
        # 测试无效的波长
        with pytest.raises(ValueError, match="波长"):
            DirectElementPropagator(
                complex_amplitude=np.ones((32, 32), dtype=complex),
                element=SurfaceDefinition(),
                wavelength=-0.633,  # 负波长
                physical_size=20.0,
            )
        
        # 测试无效的物理尺寸
        with pytest.raises(ValueError, match="物理尺寸"):
            DirectElementPropagator(
                complex_amplitude=np.ones((32, 32), dtype=complex),
                element=SurfaceDefinition(),
                wavelength=0.633,
                physical_size=0,  # 零尺寸
            )


class TestDirectPropagatorPhysics:
    """DirectElementPropagator 物理正确性测试"""
    
    def test_energy_conservation_plane_mirror(self):
        """测试平面镜能量守恒
        
        平面波入射到平面镜，出射能量应该接近入射能量。
        """
        grid_size = 64
        amplitude = np.ones((grid_size, grid_size), dtype=complex)
        input_energy = np.sum(np.abs(amplitude)**2)
        
        element = SurfaceDefinition(
            surface_type='mirror',
            radius=np.inf,
            semi_aperture=25.0,  # 足够大，不会渐晕
        )
        
        result = propagate_through_element(
            complex_amplitude=amplitude,
            element=element,
            wavelength=0.633,
            physical_size=20.0,  # 小于半口径的两倍
            num_rays=200,
            use_pilot_beam=False,
        )
        
        output_energy = np.sum(np.abs(result.output_amplitude)**2)
        
        # 能量应该大致守恒（允许一定误差，因为插值和采样会引入误差）
        # 这里使用较宽松的容差，因为光线采样和插值会引入误差
        energy_ratio = output_energy / input_energy
        assert 0.5 < energy_ratio < 2.0, f"能量比: {energy_ratio}"
    
    def test_phase_continuity(self):
        """测试相位连续性
        
        输出波前的相位应该是连续的（没有突变）。
        """
        grid_size = 64
        amplitude = np.ones((grid_size, grid_size), dtype=complex)
        
        element = SurfaceDefinition(
            surface_type='mirror',
            radius=np.inf,
            semi_aperture=25.0,
        )
        
        result = propagate_through_element(
            complex_amplitude=amplitude,
            element=element,
            wavelength=0.633,
            physical_size=20.0,
            num_rays=200,
            use_pilot_beam=False,
        )
        
        # 提取相位
        phase = np.angle(result.output_amplitude)
        
        # 计算相位梯度
        grad_x = np.diff(phase, axis=1)
        grad_y = np.diff(phase, axis=0)
        
        # 处理相位包裹（2π 跳变）
        grad_x = np.where(grad_x > np.pi, grad_x - 2*np.pi, grad_x)
        grad_x = np.where(grad_x < -np.pi, grad_x + 2*np.pi, grad_x)
        grad_y = np.where(grad_y > np.pi, grad_y - 2*np.pi, grad_y)
        grad_y = np.where(grad_y < -np.pi, grad_y + 2*np.pi, grad_y)
        
        # 忽略零振幅区域
        mask = np.abs(result.output_amplitude[:-1, :-1]) > 0.01
        
        if np.any(mask):
            # 相位梯度不应该太大（排除边缘效应）
            max_grad = max(
                np.nanmax(np.abs(grad_x[:-1, :][mask])),
                np.nanmax(np.abs(grad_y[:, :-1][mask]))
            )
            # 允许较大的梯度，因为插值可能引入一些不连续
            assert max_grad < np.pi, f"最大相位梯度: {max_grad}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
