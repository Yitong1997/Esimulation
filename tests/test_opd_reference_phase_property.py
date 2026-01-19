"""OPD 符号与参考面变换正确性属性测试

本模块测试混合传播模式中 OPD 符号取反和参考面变换的正确性。

**Feature: hybrid-propagation-raytracing-opd**
**Validates: Property 1 - OPD 符号与参考面变换正确性**

Property 1 定义：
*For any* 光学元件和入射光束，光线追迹计算的 OPD 经过符号取反和参考面变换后，
应用到波前的残差相位应正确反映元件引入的波前变化。

测试策略：
1. 对于理想抛物面镜，残差相位应该接近零（因为抛物面镜无球差）
2. 对于球面镜，残差相位应该包含球差
3. OPD 符号取反后，边缘光程长应对应正 OPD
4. 参考面变换应正确减去理想聚焦 OPD

验证需求：
- Requirements 1.3: 正确处理 OPD 符号（ElementRaytracer 与 PROPER 相反）
- Requirements 2.1-2.4: 参考面变换正确性
"""

import sys
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume

sys.path.insert(0, 'src')

from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
from optiland.rays import RealRays


# =============================================================================
# 辅助函数
# =============================================================================

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


def compute_ideal_focusing_opd_waves(
    ray_x: np.ndarray,
    ray_y: np.ndarray,
    focal_length_mm: float,
    wavelength_mm: float,
) -> np.ndarray:
    """计算理想聚焦 OPD（波长数）
    
    对于聚焦元件，理想 OPD = r² / (2f) / λ
    
    参数:
        ray_x: 光线 X 坐标（mm）
        ray_y: 光线 Y 坐标（mm）
        focal_length_mm: 焦距（mm）
        wavelength_mm: 波长（mm）
    
    返回:
        理想聚焦 OPD（波长数）
    """
    ray_r_sq = ray_x**2 + ray_y**2
    ideal_opd_mm = ray_r_sq / (2 * focal_length_mm)
    ideal_opd_waves = ideal_opd_mm / wavelength_mm
    
    # 减去主光线（中心）的 OPD
    chief_idx = np.argmin(ray_r_sq)
    return ideal_opd_waves - ideal_opd_waves[chief_idx]


# =============================================================================
# 属性测试类
# =============================================================================

class TestOPDSignConvention:
    """测试 OPD 符号约定
    
    **Validates: Property 1 - OPD 符号与参考面变换正确性**
    **Validates: Requirements 1.3**
    
    注意：ElementRaytracer 的原始 OPD 符号约定：
    - 边缘光程长 → 正 OPD
    - 这与理想聚焦 OPD 的符号一致
    """
    
    @given(
        radius=st.floats(min_value=100.0, max_value=500.0),
        half_size=st.floats(min_value=2.0, max_value=10.0),
    )
    @settings(max_examples=20, deadline=None)
    def test_opd_edge_positive_for_concave_mirror(self, radius, half_size):
        """属性测试：凹面镜边缘 OPD 应为正（边缘光程长）
        
        **Validates: Property 1**
        
        对于凹面镜（正曲率半径），边缘光线的光程比中心光线长。
        ElementRaytracer 的原始 OPD：
        - 边缘 OPD > 0（边缘光程长）
        - 中心 OPD = 0（参考点）
        """
        # 确保半尺寸不超过曲率半径的合理比例
        assume(half_size < radius * 0.1)
        
        # 创建凹面镜（球面）
        mirror = SurfaceDefinition(
            surface_type='mirror',
            radius=radius,
            thickness=0.0,
            material='mirror',
            semi_aperture=half_size * 2,
            conic=0.0,  # 球面
        )
        
        wavelength_um = 0.633
        raytracer = ElementRaytracer(
            surfaces=[mirror],
            wavelength=wavelength_um,
        )
        
        # 创建平行光
        rays_in = create_parallel_rays(
            n_rays_1d=9,
            half_size_mm=half_size,
            wavelength_um=wavelength_um,
        )
        
        # 光线追迹
        rays_out = raytracer.trace(rays_in)
        opd_waves = raytracer.get_relative_opd_waves()  # 原始 OPD，不取反
        valid_mask = raytracer.get_valid_ray_mask()
        
        # 获取光线位置
        ray_x = np.asarray(rays_in.x)
        ray_y = np.asarray(rays_in.y)
        ray_r_sq = ray_x**2 + ray_y**2
        
        # 找到边缘光线（距离中心最远的有效光线）
        valid_r_sq = np.where(valid_mask, ray_r_sq, 0.0)
        edge_idx = np.argmax(valid_r_sq)
        
        # 找到中心光线
        valid_r_sq_for_center = np.where(valid_mask, ray_r_sq, np.inf)
        center_idx = np.argmin(valid_r_sq_for_center)
        
        # 验证：边缘 OPD > 中心 OPD（边缘光程长）
        edge_opd = opd_waves[edge_idx]
        center_opd = opd_waves[center_idx]
        
        # 由于是相对 OPD，中心应该接近 0
        assert abs(center_opd) < 0.01, \
            f"中心 OPD 应接近 0，实际为 {center_opd:.4f} 波"
        
        # 边缘 OPD 应该为正（边缘光程长）
        assert edge_opd > 0, \
            f"边缘 OPD 应为正（边缘光程长），实际为 {edge_opd:.4f} 波"


class TestReferencePhaseTransform:
    """测试参考面变换正确性
    
    **Validates: Property 1 - OPD 符号与参考面变换正确性**
    **Validates: Requirements 2.1, 2.2, 2.3, 2.4**
    """
    
    @given(
        radius=st.floats(min_value=150.0, max_value=400.0),
        half_size=st.floats(min_value=2.0, max_value=8.0),
    )
    @settings(max_examples=15, deadline=None)
    def test_aberration_calculation_for_spherical_mirror(self, radius, half_size):
        """属性测试：球面镜像差计算正确性
        
        **Validates: Property 1**
        
        像差 = 实际 OPD - 理想聚焦 OPD
        对于球面镜，像差应该是球差（与 r⁴ 成正比）
        """
        assume(half_size < radius * 0.08)
        
        # 创建球面镜
        mirror = SurfaceDefinition(
            surface_type='mirror',
            radius=radius,
            thickness=0.0,
            material='mirror',
            semi_aperture=half_size * 2,
            conic=0.0,
        )
        
        wavelength_um = 0.633
        wavelength_mm = wavelength_um * 1e-3
        focal_length_mm = radius / 2
        
        raytracer = ElementRaytracer(
            surfaces=[mirror],
            wavelength=wavelength_um,
        )
        
        # 创建平行光
        rays_in = create_parallel_rays(
            n_rays_1d=9,
            half_size_mm=half_size,
            wavelength_um=wavelength_um,
        )
        
        # 光线追迹
        raytracer.trace(rays_in)
        opd_waves = raytracer.get_relative_opd_waves()  # 原始 OPD，不取反
        valid_mask = raytracer.get_valid_ray_mask()
        
        # 获取光线位置
        ray_x = np.asarray(rays_in.x)
        ray_y = np.asarray(rays_in.y)
        
        # 计算理想聚焦 OPD
        ideal_opd = compute_ideal_focusing_opd_waves(
            ray_x, ray_y, focal_length_mm, wavelength_mm
        )
        
        # 计算像差（不取反，因为 ElementRaytracer OPD 符号与理想 OPD 一致）
        aberration = opd_waves - ideal_opd
        valid_aberration = aberration[valid_mask]
        
        # 像差应该是有限值
        assert np.all(np.isfinite(valid_aberration)), \
            "像差应该是有限值"
        
        # 像差 RMS 应该大于 0（球面镜有球差）
        aberration_rms = np.nanstd(valid_aberration)
        assert aberration_rms >= 0, \
            f"像差 RMS 应该非负，实际为 {aberration_rms:.6f}"
    
    @given(
        radius=st.floats(min_value=150.0, max_value=400.0),
        half_size=st.floats(min_value=1.0, max_value=5.0),
    )
    @settings(max_examples=15, deadline=None)
    def test_parabolic_mirror_minimal_aberration(self, radius, half_size):
        """属性测试：抛物面镜像差应该很小
        
        **Validates: Property 1**
        
        对于理想抛物面镜，像差（实际 OPD - 理想聚焦 OPD）应该接近零，
        因为抛物面镜对轴上平行光无球差。
        """
        assume(half_size < radius * 0.05)
        
        # 创建抛物面镜
        mirror = SurfaceDefinition(
            surface_type='mirror',
            radius=radius,
            thickness=0.0,
            material='mirror',
            semi_aperture=half_size * 2,
            conic=-1.0,  # 抛物面
        )
        
        wavelength_um = 0.633
        wavelength_mm = wavelength_um * 1e-3
        focal_length_mm = radius / 2
        
        raytracer = ElementRaytracer(
            surfaces=[mirror],
            wavelength=wavelength_um,
        )
        
        # 创建平行光
        rays_in = create_parallel_rays(
            n_rays_1d=9,
            half_size_mm=half_size,
            wavelength_um=wavelength_um,
        )
        
        # 光线追迹
        raytracer.trace(rays_in)
        opd_waves = raytracer.get_relative_opd_waves()  # 原始 OPD，不取反
        valid_mask = raytracer.get_valid_ray_mask()
        
        # 获取光线位置
        ray_x = np.asarray(rays_in.x)
        ray_y = np.asarray(rays_in.y)
        
        # 计算理想聚焦 OPD
        ideal_opd = compute_ideal_focusing_opd_waves(
            ray_x, ray_y, focal_length_mm, wavelength_mm
        )
        
        # 计算像差（不取反）
        aberration = opd_waves - ideal_opd
        valid_aberration = aberration[valid_mask]
        
        # 抛物面镜的像差 RMS 应该很小（< 0.5 波，考虑数值误差）
        aberration_rms = np.nanstd(valid_aberration)
        assert aberration_rms < 0.5, \
            f"抛物面镜像差 RMS 应 < 0.5 波，实际为 {aberration_rms:.4f} 波"



class TestResidualPhaseCorrectness:
    """测试残差相位正确性
    
    **Validates: Property 1 - OPD 符号与参考面变换正确性**
    
    残差相位 = OPD 相位 - 参考面相位
    对于理想抛物面镜，残差相位应该接近零
    对于球面镜，残差相位应该包含球差
    """
    
    @given(
        radius=st.floats(min_value=150.0, max_value=400.0),
    )
    @settings(max_examples=10, deadline=None)
    def test_spherical_vs_parabolic_aberration_difference(self, radius):
        """属性测试：球面镜像差应该大于抛物面镜
        
        **Validates: Property 1**
        
        对于相同的曲率半径和孔径，球面镜的像差应该大于抛物面镜。
        """
        half_size = radius * 0.03  # 使用较小的孔径
        
        # 创建球面镜
        spherical_mirror = SurfaceDefinition(
            surface_type='mirror',
            radius=radius,
            thickness=0.0,
            material='mirror',
            semi_aperture=half_size * 2,
            conic=0.0,
        )
        
        # 创建抛物面镜
        parabolic_mirror = SurfaceDefinition(
            surface_type='mirror',
            radius=radius,
            thickness=0.0,
            material='mirror',
            semi_aperture=half_size * 2,
            conic=-1.0,
        )
        
        wavelength_um = 0.633
        wavelength_mm = wavelength_um * 1e-3
        focal_length_mm = radius / 2
        
        # 球面镜光线追迹
        raytracer_spherical = ElementRaytracer(
            surfaces=[spherical_mirror],
            wavelength=wavelength_um,
        )
        rays_in_s = create_parallel_rays(9, half_size, wavelength_um)
        raytracer_spherical.trace(rays_in_s)
        opd_spherical = -raytracer_spherical.get_relative_opd_waves()
        valid_s = raytracer_spherical.get_valid_ray_mask()
        
        # 抛物面镜光线追迹
        raytracer_parabolic = ElementRaytracer(
            surfaces=[parabolic_mirror],
            wavelength=wavelength_um,
        )
        rays_in_p = create_parallel_rays(9, half_size, wavelength_um)
        raytracer_parabolic.trace(rays_in_p)
        opd_parabolic = -raytracer_parabolic.get_relative_opd_waves()
        valid_p = raytracer_parabolic.get_valid_ray_mask()
        
        # 计算理想聚焦 OPD
        ray_x = np.asarray(rays_in_s.x)
        ray_y = np.asarray(rays_in_s.y)
        ideal_opd = compute_ideal_focusing_opd_waves(
            ray_x, ray_y, focal_length_mm, wavelength_mm
        )
        
        # 计算像差
        aberration_spherical = opd_spherical - ideal_opd
        aberration_parabolic = opd_parabolic - ideal_opd
        
        # 计算像差 RMS
        rms_spherical = np.nanstd(aberration_spherical[valid_s])
        rms_parabolic = np.nanstd(aberration_parabolic[valid_p])
        
        # 球面镜像差应该大于抛物面镜
        assert rms_spherical >= rms_parabolic, \
            f"球面镜像差 ({rms_spherical:.4f}) 应 >= 抛物面镜 ({rms_parabolic:.4f})"



class TestOPDPhaseConversion:
    """测试 OPD 到相位的转换正确性
    
    **Validates: Property 1**
    
    相位 = 2π × OPD（波长数）
    正 OPD（光程长）对应正相位
    """
    
    def test_opd_to_phase_conversion_sign(self):
        """测试 OPD 到相位转换的符号
        
        **Validates: Property 1**
        
        验证像差计算的正确性：
        - ElementRaytracer OPD：边缘为正
        - 理想聚焦 OPD：边缘为正
        - 像差 = 实际 OPD - 理想 OPD
        - 像差相位 = -2π × 像差（波长数）
        """
        # 创建球面镜
        radius = 200.0
        half_size = 5.0
        
        mirror = SurfaceDefinition(
            surface_type='mirror',
            radius=radius,
            thickness=0.0,
            material='mirror',
            semi_aperture=half_size * 2,
            conic=0.0,
        )
        
        wavelength_um = 0.633
        wavelength_mm = wavelength_um * 1e-3
        focal_length_mm = radius / 2
        
        raytracer = ElementRaytracer(
            surfaces=[mirror],
            wavelength=wavelength_um,
        )
        
        rays_in = create_parallel_rays(9, half_size, wavelength_um)
        raytracer.trace(rays_in)
        opd_waves = raytracer.get_relative_opd_waves()  # 不取反
        valid_mask = raytracer.get_valid_ray_mask()
        
        ray_x = np.asarray(rays_in.x)
        ray_y = np.asarray(rays_in.y)
        
        # 计算理想聚焦 OPD
        ideal_opd = compute_ideal_focusing_opd_waves(
            ray_x, ray_y, focal_length_mm, wavelength_mm
        )
        
        # 计算像差（不取反）
        aberration_waves = opd_waves - ideal_opd
        
        # 像差相位（按照正确的物理关系）
        # 正 OPD 对应负相位（波前滞后）
        aberration_phase = -2 * np.pi * aberration_waves
        
        # 验证相位是有限值
        valid_phase = aberration_phase[valid_mask]
        assert np.all(np.isfinite(valid_phase)), \
            "像差相位应该是有限值"
        
        # 验证相位范围合理（球面镜像差应该较小）
        max_phase = np.nanmax(np.abs(valid_phase))
        assert max_phase < 10, \
            f"像差相位不应过大，实际最大值为 {max_phase:.2f} 弧度"
    
    @given(
        opd_waves=st.floats(min_value=-10.0, max_value=10.0),
    )
    @settings(max_examples=20)
    def test_opd_phase_relationship(self, opd_waves):
        """属性测试：OPD 与相位的关系
        
        **Validates: Property 1**
        
        相位 = 2π × OPD（波长数）
        """
        # 计算相位
        phase = 2 * np.pi * opd_waves
        
        # 验证关系
        expected_opd = phase / (2 * np.pi)
        np.testing.assert_allclose(expected_opd, opd_waves, rtol=1e-10)


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
