"""
混合传播模式光线追迹 OPD 重构测试

本测试文件验证重构后的混合传播模式的正确性。

测试内容：
1. 核心方法单元测试
2. ElementRaytracer OPD 验证测试
3. 集成测试 - 简单光路
4. 集成测试 - 复杂光路
5. 属性测试

作者：混合光学仿真项目
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pytest
import warnings

from sequential_system import SequentialOpticalSystem, GaussianBeamSource
from gaussian_beam_simulation.optical_elements import (
    SphericalMirror,
    ParabolicMirror,
    FlatMirror,
)
from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
from optiland.rays import RealRays


# =============================================================================
# 3. 单元测试
# =============================================================================

class TestUpdateGaussianParamsOnly:
    """测试 _update_gaussian_params_only 方法
    
    **Validates: Property 2 - 高斯光束参数更新正确性**
    """
    
    def test_positive_focal_length(self):
        """测试正焦距（凹面镜）的参数更新"""
        import proper
        
        # 创建两个相同的波前对象
        wavelength_m = 0.633e-6
        beam_diameter_m = 0.01
        grid_size = 64
        
        wfo1 = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, 0.5)
        wfo2 = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, 0.5)
        
        # 使用 prop_lens 更新 wfo1
        focal_length_m = 0.1
        proper.prop_lens(wfo1, focal_length_m)
        
        # 使用 _update_gaussian_params_only 更新 wfo2
        source = GaussianBeamSource(wavelength=0.633, w0=1.0, z0=0.0)
        system = SequentialOpticalSystem(source, grid_size=64)
        system._update_gaussian_params_only(wfo2, focal_length_m)
        
        # 比较参数
        assert np.isclose(wfo1.z_w0, wfo2.z_w0, rtol=1e-6)
        assert np.isclose(wfo1.w0, wfo2.w0, rtol=1e-6)
        assert np.isclose(wfo1.z_Rayleigh, wfo2.z_Rayleigh, rtol=1e-6)
        assert wfo1.beam_type_old == wfo2.beam_type_old
        assert wfo1.reference_surface == wfo2.reference_surface
    
    def test_negative_focal_length(self):
        """测试负焦距（凸面镜）的参数更新"""
        import proper
        
        wavelength_m = 0.633e-6
        beam_diameter_m = 0.01
        grid_size = 64
        
        wfo1 = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, 0.5)
        wfo2 = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, 0.5)
        
        focal_length_m = -0.05
        proper.prop_lens(wfo1, focal_length_m)
        
        source = GaussianBeamSource(wavelength=0.633, w0=1.0, z0=0.0)
        system = SequentialOpticalSystem(source, grid_size=64)
        system._update_gaussian_params_only(wfo2, focal_length_m)
        
        assert np.isclose(wfo1.z_w0, wfo2.z_w0, rtol=1e-6)
        assert np.isclose(wfo1.w0, wfo2.w0, rtol=1e-6)
        assert np.isclose(wfo1.z_Rayleigh, wfo2.z_Rayleigh, rtol=1e-6)


class TestComputeReferencePhase:
    """测试 _compute_reference_phase 方法
    
    **Validates: Requirements 2.1, 2.2, 2.3, 2.4**
    """
    
    def test_planar_reference(self):
        """测试 PLANAR 参考面（应返回零）"""
        import proper
        
        wavelength_m = 0.633e-6
        beam_diameter_m = 0.01
        grid_size = 64
        
        wfo = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, 0.5)
        # 初始状态是 PLANAR 参考面
        assert wfo.reference_surface == "PLANAR"
        
        source = GaussianBeamSource(wavelength=0.633, w0=1.0, z0=0.0)
        system = SequentialOpticalSystem(source, grid_size=64)
        
        x_mm = np.array([0.0, 1.0, 2.0])
        y_mm = np.array([0.0, 0.0, 0.0])
        
        phase_ref = system._compute_reference_phase(wfo, x_mm, y_mm)
        
        # PLANAR 参考面应返回零
        np.testing.assert_allclose(phase_ref, 0.0, atol=1e-10)
    
    def test_spherical_reference(self):
        """测试 SPHERI 参考面（应返回二次相位）"""
        import proper
        
        wavelength_m = 0.633e-6
        beam_diameter_m = 0.01
        grid_size = 64
        
        wfo = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, 0.5)
        
        # 应用透镜使参考面变为 SPHERI
        focal_length_m = 0.1
        proper.prop_lens(wfo, focal_length_m)
        assert wfo.reference_surface == "SPHERI"
        
        source = GaussianBeamSource(wavelength=0.633, w0=1.0, z0=0.0)
        system = SequentialOpticalSystem(source, grid_size=64)
        
        x_mm = np.array([0.0, 5.0])
        y_mm = np.array([0.0, 0.0])
        
        phase_ref = system._compute_reference_phase(wfo, x_mm, y_mm)
        
        # 中心应为零
        assert np.isclose(phase_ref[0], 0.0, atol=1e-10)
        
        # 边缘应为非零
        assert abs(phase_ref[1]) > 0


class TestCheckPhaseSampling:
    """测试 _check_phase_sampling 方法
    
    **Validates: Requirements 7.1, 7.2, 7.3**
    """
    
    def test_normal_gradient_no_warning(self):
        """测试正常相位梯度（不应警告）"""
        source = GaussianBeamSource(wavelength=0.633, w0=1.0, z0=0.0)
        system = SequentialOpticalSystem(source, grid_size=64)
        
        # 创建小梯度的相位网格
        n = 64
        phase_grid = np.zeros((n, n))
        for i in range(n):
            phase_grid[:, i] = i * 0.01  # 小梯度
        
        # 不应发出警告
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            system._check_phase_sampling(phase_grid, 0.1)
            assert len(w) == 0
    
    def test_large_gradient_warning(self):
        """测试过大相位梯度（应警告）"""
        source = GaussianBeamSource(wavelength=0.633, w0=1.0, z0=0.0)
        system = SequentialOpticalSystem(source, grid_size=64)
        
        # 创建大梯度的相位网格
        n = 64
        phase_grid = np.zeros((n, n))
        for i in range(n):
            phase_grid[:, i] = i * 5.0  # 大梯度（每像素 5 弧度，超过 π）
        
        # 应发出警告
        with pytest.warns(UserWarning, match="相位采样不足"):
            system._check_phase_sampling(phase_grid, 0.1)


# =============================================================================
# 4. ElementRaytracer OPD 验证测试
# =============================================================================

class TestElementRaytracerOPD:
    """ElementRaytracer OPD 验证测试
    
    **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**
    """
    
    def test_flat_mirror_opd_constant(self):
        """平面镜 OPD 常数性测试
        
        **Validates: Requirements 4.4, Property 5**
        """
        surface_def = SurfaceDefinition(
            surface_type='mirror',
            radius=np.inf,
            thickness=0.0,
            material='mirror',
            semi_aperture=15.0,
        )
        
        wavelength_um = 0.633
        
        # 创建采样光线
        n_rays_1d = 11
        half_size = 5.0
        ray_coords = np.linspace(-half_size, half_size, n_rays_1d)
        ray_X, ray_Y = np.meshgrid(ray_coords, ray_coords)
        ray_x = ray_X.flatten()
        ray_y = ray_Y.flatten()
        n_rays = len(ray_x)
        
        rays_in = RealRays(
            x=ray_x,
            y=ray_y,
            z=np.zeros(n_rays),
            L=np.zeros(n_rays),
            M=np.zeros(n_rays),
            N=np.ones(n_rays),
            intensity=np.ones(n_rays),
            wavelength=np.full(n_rays, wavelength_um),
        )
        
        raytracer = ElementRaytracer(
            surfaces=[surface_def],
            wavelength=wavelength_um,
        )
        
        rays_out = raytracer.trace(rays_in)
        opd_waves = raytracer.get_relative_opd_waves()
        valid_mask = raytracer.get_valid_ray_mask()
        
        # 平面镜 OPD RMS 应 < 0.001 波
        opd_valid = opd_waves[valid_mask]
        opd_rms = np.std(opd_valid)
        assert opd_rms < 0.001, f"平面镜 OPD RMS = {opd_rms:.6f} 波，超过 0.001 波"
    
    def test_parabolic_mirror_opd_constant(self):
        """抛物面镜 OPD 常数性测试
        
        **Validates: Requirements 4.2, Property 3**
        
        注意：此测试验证抛物面镜的像差（相对于理想聚焦）应该很小。
        由于数值精度和采样限制，允许 0.2 波的 RMS 误差。
        """
        surface_def = SurfaceDefinition(
            surface_type='mirror',
            radius=200.0,  # f = 100mm
            thickness=0.0,
            material='mirror',
            semi_aperture=15.0,
            conic=-1.0,  # 抛物面
        )
        
        wavelength_um = 0.633
        
        n_rays_1d = 21  # 增加采样点数
        half_size = 5.0
        ray_coords = np.linspace(-half_size, half_size, n_rays_1d)
        ray_X, ray_Y = np.meshgrid(ray_coords, ray_coords)
        ray_x = ray_X.flatten()
        ray_y = ray_Y.flatten()
        n_rays = len(ray_x)
        
        rays_in = RealRays(
            x=ray_x,
            y=ray_y,
            z=np.zeros(n_rays),
            L=np.zeros(n_rays),
            M=np.zeros(n_rays),
            N=np.ones(n_rays),
            intensity=np.ones(n_rays),
            wavelength=np.full(n_rays, wavelength_um),
        )
        
        raytracer = ElementRaytracer(
            surfaces=[surface_def],
            wavelength=wavelength_um,
        )
        
        rays_out = raytracer.trace(rays_in)
        opd_waves = raytracer.get_relative_opd_waves()
        valid_mask = raytracer.get_valid_ray_mask()
        
        # 计算像差（减去理想聚焦 OPD）
        focal_length_mm = 100.0
        wavelength_mm = wavelength_um * 1e-3
        ray_r_sq = ray_x**2 + ray_y**2
        ideal_opd_waves = ray_r_sq / (2 * focal_length_mm * wavelength_mm)
        aberration_waves = opd_waves - ideal_opd_waves
        
        # 抛物面镜像差 RMS 应 < 0.2 波（考虑数值精度）
        aberration_valid = aberration_waves[valid_mask]
        aberration_rms = np.nanstd(aberration_valid)
        assert aberration_rms < 0.2, f"抛物面镜像差 RMS = {aberration_rms:.6f} 波，超过 0.2 波"


# =============================================================================
# 5. 集成测试 - 简单光路
# =============================================================================

class TestSimpleOpticalPath:
    """简单光路集成测试
    
    **Validates: Requirements 5.1, 5.2, 5.3, 5.4, Property 7**
    """
    
    def test_single_spherical_mirror(self):
        """单凹面镜集成测试
        
        **Validates: Requirements 5.1**
        """
        source = GaussianBeamSource(
            wavelength=0.633,
            w0=1.0,
            z0=0.0,
        )
        
        system = SequentialOpticalSystem(
            source,
            grid_size=256,
            use_hybrid_propagation=True,
            hybrid_num_rays=100,
        )
        
        system.add_surface(SphericalMirror(
            radius_of_curvature=200.0,
            thickness=150.0,
            semi_aperture=15.0,
        ))
        
        system.add_sampling_plane(distance=150.0, name='focus')
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = system.run()
        
        focus_result = results['focus']
        abcd_result = system.get_abcd_result(distance=150.0)
        
        # 光束半径误差应 < 1%
        error = abs(focus_result.beam_radius - abcd_result.w) / abcd_result.w
        assert error < 0.01, f"光束半径误差 = {error*100:.2f}%，超过 1%"
    
    def test_single_parabolic_mirror(self):
        """单抛物面镜集成测试
        
        **Validates: Requirements 5.2**
        """
        source = GaussianBeamSource(
            wavelength=0.633,
            w0=1.0,
            z0=0.0,
        )
        
        system = SequentialOpticalSystem(
            source,
            grid_size=256,
            use_hybrid_propagation=True,
            hybrid_num_rays=100,
        )
        
        system.add_surface(ParabolicMirror(
            parent_focal_length=100.0,
            thickness=150.0,
            semi_aperture=15.0,
        ))
        
        system.add_sampling_plane(distance=150.0, name='focus')
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = system.run()
        
        focus_result = results['focus']
        abcd_result = system.get_abcd_result(distance=150.0)
        
        # 光束半径误差应 < 1%
        error = abs(focus_result.beam_radius - abcd_result.w) / abcd_result.w
        assert error < 0.01, f"光束半径误差 = {error*100:.2f}%，超过 1%"
    
    def test_single_flat_mirror(self):
        """单平面镜集成测试
        
        **Validates: Requirements 5.3**
        """
        source = GaussianBeamSource(
            wavelength=0.633,
            w0=1.0,
            z0=0.0,
        )
        
        system = SequentialOpticalSystem(
            source,
            grid_size=256,
            use_hybrid_propagation=True,
            hybrid_num_rays=100,
        )
        
        system.add_surface(FlatMirror(
            thickness=100.0,
            semi_aperture=15.0,
        ))
        
        system.add_sampling_plane(distance=100.0, name='output')
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = system.run()
        
        output_result = results['output']
        abcd_result = system.get_abcd_result(distance=100.0)
        
        # 光束半径误差应 < 1%
        error = abs(output_result.beam_radius - abcd_result.w) / abcd_result.w
        assert error < 0.01, f"光束半径误差 = {error*100:.2f}%，超过 1%"


# =============================================================================
# 6. 集成测试 - 复杂光路
# =============================================================================

class TestComplexOpticalPath:
    """复杂光路集成测试
    
    **Validates: Requirements 6.1, 6.2, 6.3, 6.4, Property 8**
    """
    
    def test_galilean_oap_expander(self):
        """伽利略式扩束镜集成测试
        
        **Validates: Requirements 6.1, 6.2**
        """
        # 光源参数
        wavelength = 10.64
        w0_input = 10.0
        
        # 扩束镜焦距设计
        f1 = -50.0
        f2 = 150.0
        magnification = -f2 / f1
        
        # 倾斜角度
        theta = np.radians(45.0)
        
        # 几何参数
        d_oap1_to_fold = 50.0
        d_fold_to_oap2 = 50.0
        d_oap2_to_output = 100.0
        total_path = d_oap1_to_fold + d_fold_to_oap2 + d_oap2_to_output
        
        source = GaussianBeamSource(
            wavelength=wavelength,
            w0=w0_input,
            z0=0.0,
        )
        
        system = SequentialOpticalSystem(
            source=source,
            grid_size=512,
            beam_ratio=0.25,
            use_hybrid_propagation=True,
            hybrid_num_rays=100,
        )
        
        system.add_sampling_plane(distance=0.0, name='Input')
        
        system.add_surface(ParabolicMirror(
            parent_focal_length=f1,
            thickness=d_oap1_to_fold,
            semi_aperture=20.0,
            off_axis_distance=100.0,
            tilt_x=theta,
        ))
        
        system.add_surface(FlatMirror(
            thickness=d_fold_to_oap2,
            semi_aperture=30.0,
            tilt_x=theta,
        ))
        
        system.add_surface(ParabolicMirror(
            parent_focal_length=f2,
            thickness=d_oap2_to_output,
            semi_aperture=50.0,
            off_axis_distance=300.0,
            tilt_x=theta,
        ))
        
        system.add_sampling_plane(distance=total_path, name='Output')
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = system.run()
        
        input_result = results['Input']
        output_result = results['Output']
        
        # 放大倍率验证
        measured_mag = output_result.beam_radius / input_result.beam_radius
        mag_error = abs(measured_mag - magnification) / magnification
        assert mag_error < 0.01, f"放大倍率误差 = {mag_error*100:.2f}%，超过 1%"
        
        # ABCD 对比
        abcd_result = system.get_abcd_result(distance=total_path)
        beam_error = abs(output_result.beam_radius - abcd_result.w) / abcd_result.w
        assert beam_error < 0.01, f"光束半径误差 = {beam_error*100:.2f}%，超过 1%"


# =============================================================================
# 运行测试
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
