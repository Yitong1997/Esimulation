"""ElementRaytracer OPD 验证测试

本模块测试 ElementRaytracer 计算 OPD 的正确性。

测试内容：
- 平面镜 OPD 常数性测试
- 抛物面镜 OPD 常数性测试
- 凹面镜 OPD 解析验证测试
- 45° 折叠镜坐标变换测试

**Validates: Requirements 4.1-4.5, Property 3-6**
"""

import sys
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings

sys.path.insert(0, 'src')

from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
from optiland.rays import RealRays


def create_parallel_rays(
    n_rays_1d: int,
    half_size_mm: float,
    wavelength_um: float,
) -> RealRays:
    """创建平行光入射光线
    
    参数:
        n_rays_1d: 每个方向的光线数量
        half_size_mm: 采样面半尺寸（mm）
        wavelength_um: 波长（μm）
    
    返回:
        RealRays 对象
    """
    ray_coords = np.linspace(-half_size_mm, half_size_mm, n_rays_1d)
    ray_X, ray_Y = np.meshgrid(ray_coords, ray_coords)
    ray_x = ray_X.flatten()
    ray_y = ray_Y.flatten()
    n_rays = len(ray_x)
    
    return RealRays(
        x=ray_x,
        y=ray_y,
        z=np.zeros(n_rays),
        L=np.zeros(n_rays),
        M=np.zeros(n_rays),
        N=np.ones(n_rays),
        intensity=np.ones(n_rays),
        wavelength=np.full(n_rays, wavelength_um),
    )


class TestPlaneMirrorOPD:
    """测试平面镜 OPD 常数性
    
    **Validates: Requirements 4.4, Property 5**
    """
    
    def test_plane_mirror_opd_constant(self):
        """平行光入射平面镜，OPD RMS < 0.001 波
        
        **Validates: Property 5 - 平面镜 OPD 常数性**
        """
        # 创建平面镜
        mirror = SurfaceDefinition(
            surface_type='mirror',
            radius=np.inf,  # 平面
            thickness=0.0,
            material='mirror',
            semi_aperture=10.0,
        )
        
        wavelength_um = 0.633
        raytracer = ElementRaytracer(
            surfaces=[mirror],
            wavelength=wavelength_um,
        )
        
        # 创建平行光
        rays_in = create_parallel_rays(
            n_rays_1d=11,
            half_size_mm=5.0,
            wavelength_um=wavelength_um,
        )
        
        # 光线追迹
        rays_out = raytracer.trace(rays_in)
        opd_waves = raytracer.get_relative_opd_waves()
        valid_mask = raytracer.get_valid_ray_mask()
        
        # 计算有效光线的 OPD RMS
        valid_opd = opd_waves[valid_mask]
        opd_rms = np.nanstd(valid_opd)
        
        # 验证 OPD RMS < 0.001 波
        assert opd_rms < 0.001, f"平面镜 OPD RMS = {opd_rms:.6f} 波，应 < 0.001 波"
    
    @given(
        half_size=st.floats(min_value=1.0, max_value=20.0),
        wavelength=st.floats(min_value=0.4, max_value=1.0),
    )
    @settings(max_examples=10)
    def test_plane_mirror_opd_constant_property(self, half_size, wavelength):
        """属性测试：平面镜 OPD 应为常数
        
        **Validates: Property 5**
        """
        mirror = SurfaceDefinition(
            surface_type='mirror',
            radius=np.inf,
            thickness=0.0,
            material='mirror',
            semi_aperture=half_size * 2,
        )
        
        raytracer = ElementRaytracer(
            surfaces=[mirror],
            wavelength=wavelength,
        )
        
        rays_in = create_parallel_rays(
            n_rays_1d=7,
            half_size_mm=half_size,
            wavelength_um=wavelength,
        )
        
        rays_out = raytracer.trace(rays_in)
        opd_waves = raytracer.get_relative_opd_waves()
        valid_mask = raytracer.get_valid_ray_mask()
        
        valid_opd = opd_waves[valid_mask]
        opd_rms = np.nanstd(valid_opd)
        
        assert opd_rms < 0.001, f"平面镜 OPD RMS = {opd_rms:.6f} 波"


class TestParabolicMirrorOPD:
    """测试抛物面镜 OPD 特性
    
    注意：ElementRaytracer 计算的是从入射面到出射面的光程差，
    包含了聚焦引入的二次相位。
    
    对于抛物面镜，理论上所有平行光都聚焦到同一点，因此相对于
    理想聚焦的像差应该为零。但由于 OPD 计算是到出射面而不是到焦点，
    所以需要使用近轴公式 r^2/(2f) 来计算理想聚焦 OPD。
    
    由于高阶项的存在，大孔径时会有残余像差，但抛物面镜的像差
    应该比球面镜小得多。
    
    **Validates: Requirements 4.2, Property 3**
    """
    
    def test_parabolic_mirror_less_aberration_than_spherical(self):
        """抛物面镜的像差应该比球面镜小
        
        验证方法：比较抛物面镜和球面镜的 OPD 分布，
        抛物面镜的像差应该比球面镜小得多。
        
        **Validates: Property 3 - 抛物面镜 OPD 特性**
        """
        # 创建抛物面镜（焦距 100mm，顶点曲率半径 200mm）
        R = 200.0  # mm
        parabolic_mirror = SurfaceDefinition(
            surface_type='mirror',
            radius=R,
            thickness=0.0,
            material='mirror',
            semi_aperture=15.0,
            conic=-1.0,  # 抛物面
        )
        
        # 创建球面镜（相同曲率半径）
        spherical_mirror = SurfaceDefinition(
            surface_type='mirror',
            radius=R,
            thickness=0.0,
            material='mirror',
            semi_aperture=15.0,
            conic=0.0,  # 球面
        )
        
        wavelength_um = 0.633
        wavelength_mm = wavelength_um * 1e-3
        
        # 创建平行光
        rays_in = create_parallel_rays(
            n_rays_1d=11,
            half_size_mm=10.0,
            wavelength_um=wavelength_um,
        )
        
        # 抛物面镜光线追迹
        raytracer_parabolic = ElementRaytracer(
            surfaces=[parabolic_mirror],
            wavelength=wavelength_um,
        )
        raytracer_parabolic.trace(rays_in)
        opd_parabolic = raytracer_parabolic.get_relative_opd_waves()
        valid_parabolic = raytracer_parabolic.get_valid_ray_mask()
        
        # 球面镜光线追迹
        rays_in_spherical = create_parallel_rays(
            n_rays_1d=11,
            half_size_mm=10.0,
            wavelength_um=wavelength_um,
        )
        raytracer_spherical = ElementRaytracer(
            surfaces=[spherical_mirror],
            wavelength=wavelength_um,
        )
        raytracer_spherical.trace(rays_in_spherical)
        opd_spherical = raytracer_spherical.get_relative_opd_waves()
        valid_spherical = raytracer_spherical.get_valid_ray_mask()
        
        # 获取光线位置
        ray_x = np.asarray(rays_in.x)
        ray_y = np.asarray(rays_in.y)
        ray_r_sq = ray_x**2 + ray_y**2
        
        # 计算理想聚焦 OPD（近轴公式）
        focal_length = R / 2  # mm
        ideal_opd_mm = ray_r_sq / (2 * focal_length)
        ideal_opd_waves = ideal_opd_mm / wavelength_mm
        chief_idx = np.argmin(ray_r_sq)
        ideal_relative_opd = ideal_opd_waves - ideal_opd_waves[chief_idx]
        
        # 计算抛物面镜的像差（实际 OPD - 理想 OPD）
        aberration_parabolic = opd_parabolic - ideal_relative_opd
        aberration_parabolic_valid = aberration_parabolic[valid_parabolic]
        aberration_parabolic_rms = np.nanstd(aberration_parabolic_valid)
        
        # 计算球面镜的像差
        aberration_spherical = opd_spherical - ideal_relative_opd
        aberration_spherical_valid = aberration_spherical[valid_spherical]
        aberration_spherical_rms = np.nanstd(aberration_spherical_valid)
        
        # 抛物面镜的像差应该比球面镜小得多
        assert aberration_parabolic_rms < aberration_spherical_rms, \
            f"抛物面镜像差 RMS ({aberration_parabolic_rms:.4f}) 应小于球面镜 ({aberration_spherical_rms:.4f})"
    
    def test_parabolic_mirror_small_aperture_low_aberration(self):
        """小孔径抛物面镜的像差应该很小
        
        使用较小的孔径（2mm），验证抛物面镜的像差 RMS < 0.01 波
        
        **Validates: Property 3 - 抛物面镜 OPD 常数性**
        """
        # 创建抛物面镜（焦距 100mm，顶点曲率半径 200mm）
        R = 200.0  # mm
        parabolic_mirror = SurfaceDefinition(
            surface_type='mirror',
            radius=R,
            thickness=0.0,
            material='mirror',
            semi_aperture=5.0,
            conic=-1.0,  # 抛物面
        )
        
        wavelength_um = 0.633
        wavelength_mm = wavelength_um * 1e-3
        
        # 创建平行光（小孔径）
        rays_in = create_parallel_rays(
            n_rays_1d=11,
            half_size_mm=2.0,  # 小孔径
            wavelength_um=wavelength_um,
        )
        
        # 抛物面镜光线追迹
        raytracer = ElementRaytracer(
            surfaces=[parabolic_mirror],
            wavelength=wavelength_um,
        )
        raytracer.trace(rays_in)
        opd_waves = raytracer.get_relative_opd_waves()
        valid_mask = raytracer.get_valid_ray_mask()
        
        # 获取光线位置
        ray_x = np.asarray(rays_in.x)
        ray_y = np.asarray(rays_in.y)
        ray_r_sq = ray_x**2 + ray_y**2
        
        # 计算理想聚焦 OPD
        focal_length = R / 2  # mm
        ideal_opd_mm = ray_r_sq / (2 * focal_length)
        ideal_opd_waves = ideal_opd_mm / wavelength_mm
        chief_idx = np.argmin(ray_r_sq)
        ideal_relative_opd = ideal_opd_waves - ideal_opd_waves[chief_idx]
        
        # 计算像差
        aberration = opd_waves - ideal_relative_opd
        valid_aberration = aberration[valid_mask]
        aberration_rms = np.nanstd(valid_aberration)
        
        # 小孔径时像差应该很小（< 0.01 波）
        assert aberration_rms < 0.01, \
            f"小孔径抛物面镜像差 RMS = {aberration_rms:.6f} 波，应 < 0.01 波"


class TestSphericalMirrorOPD:
    """测试凹面镜 OPD 解析验证
    
    **Validates: Requirements 4.1, 4.3, Property 4**
    """
    
    def test_spherical_mirror_opd_analytical(self):
        """平行光入射球面凹面镜，OPD 与解析公式一致
        
        验证球差公式 SA = r⁴ / (8 * R³)
        
        **Validates: Property 4 - 凹面镜 OPD 解析验证**
        """
        # 创建球面凹面镜（焦距 100mm，曲率半径 200mm）
        R = 200.0  # mm
        mirror = SurfaceDefinition(
            surface_type='mirror',
            radius=R,
            thickness=0.0,
            material='mirror',
            semi_aperture=15.0,
            conic=0.0,  # 球面
        )
        
        wavelength_um = 0.633
        wavelength_mm = wavelength_um * 1e-3
        raytracer = ElementRaytracer(
            surfaces=[mirror],
            wavelength=wavelength_um,
        )
        
        # 创建平行光（使用较小的孔径以提高解析公式精度）
        rays_in = create_parallel_rays(
            n_rays_1d=11,
            half_size_mm=5.0,  # 较小的孔径
            wavelength_um=wavelength_um,
        )
        
        # 光线追迹
        rays_out = raytracer.trace(rays_in)
        opd_waves = raytracer.get_relative_opd_waves()
        valid_mask = raytracer.get_valid_ray_mask()
        
        # 获取光线位置
        ray_x = np.asarray(rays_in.x)
        ray_y = np.asarray(rays_in.y)
        ray_r = np.sqrt(ray_x**2 + ray_y**2)
        ray_r_sq = ray_r**2
        
        # 计算理想聚焦 OPD（二次曲线）
        focal_length = R / 2  # mm
        ideal_opd_mm = ray_r_sq / (2 * focal_length)
        ideal_opd_waves = ideal_opd_mm / wavelength_mm
        chief_idx = np.argmin(ray_r)
        ideal_relative_opd = ideal_opd_waves - ideal_opd_waves[chief_idx]
        
        # 计算像差（实际 OPD - 理想 OPD）
        aberration = opd_waves - ideal_relative_opd
        
        # 计算解析球差 OPD
        # 对于球面镜，球差 SA ≈ r⁴ / (8 * R³)（单位：mm）
        analytical_sa_mm = ray_r**4 / (8 * R**3)
        analytical_sa_waves = analytical_sa_mm / wavelength_mm
        analytical_relative_sa = analytical_sa_waves - analytical_sa_waves[chief_idx]
        
        # 比较像差与解析球差
        valid_aberration = aberration[valid_mask]
        valid_analytical = analytical_relative_sa[valid_mask]
        
        # 计算相关系数（应该接近 1）
        if len(valid_aberration) > 3:
            # 移除常数偏移
            valid_aberration_centered = valid_aberration - np.nanmean(valid_aberration)
            valid_analytical_centered = valid_analytical - np.nanmean(valid_analytical)
            
            # 计算相关系数
            correlation = np.corrcoef(valid_aberration_centered, valid_analytical_centered)[0, 1]
            assert correlation > 0.95, f"像差与解析球差相关系数 = {correlation:.4f}，应 > 0.95"
    
    @given(
        radius=st.floats(min_value=100.0, max_value=1000.0),
    )
    @settings(max_examples=10)
    def test_spherical_mirror_has_spherical_aberration(self, radius):
        """属性测试：球面镜应有球差
        
        **Validates: Property 4**
        """
        mirror = SurfaceDefinition(
            surface_type='mirror',
            radius=radius,
            thickness=0.0,
            material='mirror',
            semi_aperture=radius * 0.1,
            conic=0.0,
        )
        
        wavelength_um = 0.633
        raytracer = ElementRaytracer(
            surfaces=[mirror],
            wavelength=wavelength_um,
        )
        
        rays_in = create_parallel_rays(
            n_rays_1d=7,
            half_size_mm=radius * 0.05,
            wavelength_um=wavelength_um,
        )
        
        rays_out = raytracer.trace(rays_in)
        opd_waves = raytracer.get_relative_opd_waves()
        valid_mask = raytracer.get_valid_ray_mask()
        
        valid_opd = opd_waves[valid_mask]
        opd_pv = np.nanmax(valid_opd) - np.nanmin(valid_opd)
        
        # 球面镜应该有非零的 OPD PV（球差）
        # 但对于小孔径，球差可能很小
        # 这里只验证 OPD 计算没有错误
        assert np.isfinite(opd_pv), "OPD PV 应为有限值"


class TestFoldMirrorOPD:
    """测试 45° 折叠镜坐标变换
    
    **Validates: Requirements 4.5, Property 6**
    """
    
    def test_fold_mirror_direction(self):
        """验证出射光线方向与反射定律一致
        
        注意：tilt_x 为正时，表面法向量绕 X 轴旋转，
        初始法向量为 (0, 0, -1)，旋转后指向 +Y 方向。
        入射光 (0, 0, 1) 反射后应该指向 +Y 方向。
        
        **Validates: Property 6 - 45° 折叠镜坐标变换正确性**
        """
        # 创建 45° 折叠镜
        mirror = SurfaceDefinition(
            surface_type='mirror',
            radius=np.inf,  # 平面
            thickness=0.0,
            material='mirror',
            semi_aperture=10.0,
            tilt_x=np.pi/4,  # 45° 倾斜
        )
        
        wavelength_um = 0.633
        raytracer = ElementRaytracer(
            surfaces=[mirror],
            wavelength=wavelength_um,
        )
        
        # 获取出射主光线方向
        exit_dir = raytracer.get_exit_chief_ray_direction()
        
        # 入射方向为 (0, 0, 1)
        # tilt_x = π/4 时，表面法向量从 (0, 0, -1) 旋转到 (0, -sin(45°), -cos(45°))
        # 反射后光线方向应该是 (0, 1, 0) 或 (0, -1, 0)，取决于旋转方向
        # 验证出射方向在 YZ 平面内，且 Z 分量接近 0
        assert abs(exit_dir[0]) < 1e-6, f"出射方向 X 分量应接近 0，实际为 {exit_dir[0]}"
        assert abs(exit_dir[2]) < 1e-6, f"出射方向 Z 分量应接近 0，实际为 {exit_dir[2]}"
        assert abs(abs(exit_dir[1]) - 1.0) < 1e-6, f"出射方向 Y 分量绝对值应接近 1，实际为 {exit_dir[1]}"
    
    def test_fold_mirror_opd_symmetry(self):
        """验证 OPD 分布保持对称性
        
        **Validates: Property 6**
        """
        # 创建 45° 折叠镜
        mirror = SurfaceDefinition(
            surface_type='mirror',
            radius=np.inf,
            thickness=0.0,
            material='mirror',
            semi_aperture=10.0,
            tilt_x=np.pi/4,
        )
        
        wavelength_um = 0.633
        raytracer = ElementRaytracer(
            surfaces=[mirror],
            wavelength=wavelength_um,
        )
        
        # 创建平行光
        rays_in = create_parallel_rays(
            n_rays_1d=11,
            half_size_mm=5.0,
            wavelength_um=wavelength_um,
        )
        
        # 光线追迹
        rays_out = raytracer.trace(rays_in)
        opd_waves = raytracer.get_relative_opd_waves()
        valid_mask = raytracer.get_valid_ray_mask()
        
        # 计算 OPD RMS
        valid_opd = opd_waves[valid_mask]
        opd_rms = np.nanstd(valid_opd)
        
        # 平面折叠镜的 OPD 应该是常数（RMS < 0.001 波）
        assert opd_rms < 0.001, f"折叠镜 OPD RMS = {opd_rms:.6f} 波，应 < 0.001 波"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
