"""混合传播模式 OPD 重构单元测试

本模块测试混合传播模式的核心方法实现。

测试内容：
- _update_gaussian_params_only 方法
- _compute_reference_phase 方法
- _check_phase_sampling 方法
- 属性测试（Property-Based Testing）

**Validates: Requirements 2.1-2.4, 3.1-3.4, 7.1-7.3, Property 2**
"""

import sys
import numpy as np
import pytest
import warnings
from hypothesis import given, strategies as st, settings, assume

sys.path.insert(0, 'src')

from sequential_system.system import SequentialOpticalSystem
from sequential_system.source import GaussianBeamSource


class TestUpdateGaussianParamsOnly:
    """测试 _update_gaussian_params_only 方法
    
    **Validates: Property 2 - 高斯光束参数更新正确性**
    """
    
    @pytest.fixture
    def system(self):
        """创建测试用的光学系统"""
        source = GaussianBeamSource(
            wavelength=0.633,  # μm
            w0=1.0,            # mm
            z0=0.0,            # mm
        )
        return SequentialOpticalSystem(source, grid_size=128)
    
    @pytest.fixture
    def wfo(self, system):
        """创建测试用的 PROPER 波前对象"""
        import proper
        
        beam = system._source.to_gaussian_beam()
        wavelength_m = system._source.wavelength * 1e-6
        w_init = system._source.w(0.0)
        beam_diameter_m = 4 * w_init * 1e-3
        
        wfo = proper.prop_begin(
            beam_diameter_m, wavelength_m, system._grid_size, system._beam_ratio
        )
        return wfo
    
    def test_positive_focal_length(self, system, wfo):
        """测试正焦距（凹面镜）的参数更新
        
        **Validates: Requirements 3.1, 3.2, 3.3**
        """
        import proper
        
        # 保存原始参数
        z_w0_before = wfo.z_w0
        w0_before = wfo.w0
        
        # 应用正焦距
        focal_length_m = 0.1  # 100mm
        system._update_gaussian_params_only(wfo, focal_length_m)
        
        # 验证参数已更新
        assert wfo.z_w0 != z_w0_before, "z_w0 应该被更新"
        assert wfo.w0 != w0_before or wfo.z_w0 != z_w0_before, "至少一个参数应该被更新"
        
        # 验证参考面类型
        assert wfo.reference_surface in ["PLANAR", "SPHERI"], "参考面类型应该是 PLANAR 或 SPHERI"
    
    def test_negative_focal_length(self, system, wfo):
        """测试负焦距（凸面镜）的参数更新
        
        **Validates: Requirements 3.1, 3.2, 3.3**
        """
        # 保存原始参数
        z_w0_before = wfo.z_w0
        
        # 应用负焦距
        focal_length_m = -0.1  # -100mm
        system._update_gaussian_params_only(wfo, focal_length_m)
        
        # 验证参数已更新
        assert wfo.z_w0 != z_w0_before, "z_w0 应该被更新"
    
    def test_consistency_with_prop_lens(self, system, wfo):
        """测试与 prop_lens 参数更新结果的一致性
        
        **Validates: Property 2**
        """
        import proper
        
        # 创建两个相同的波前对象
        beam = system._source.to_gaussian_beam()
        wavelength_m = system._source.wavelength * 1e-6
        w_init = system._source.w(0.0)
        beam_diameter_m = 4 * w_init * 1e-3
        
        wfo1 = proper.prop_begin(
            beam_diameter_m, wavelength_m, system._grid_size, system._beam_ratio
        )
        wfo2 = proper.prop_begin(
            beam_diameter_m, wavelength_m, system._grid_size, system._beam_ratio
        )
        
        focal_length_m = 0.1  # 100mm
        
        # 使用 _update_gaussian_params_only
        system._update_gaussian_params_only(wfo1, focal_length_m)
        
        # 使用 prop_lens（会同时更新参数和 wfarr）
        proper.prop_lens(wfo2, focal_length_m)
        
        # 验证参数一致性
        np.testing.assert_allclose(wfo1.z_w0, wfo2.z_w0, rtol=1e-10,
                                   err_msg="z_w0 应该与 prop_lens 结果一致")
        np.testing.assert_allclose(wfo1.w0, wfo2.w0, rtol=1e-10,
                                   err_msg="w0 应该与 prop_lens 结果一致")
        np.testing.assert_allclose(wfo1.z_Rayleigh, wfo2.z_Rayleigh, rtol=1e-10,
                                   err_msg="z_Rayleigh 应该与 prop_lens 结果一致")
        assert wfo1.reference_surface == wfo2.reference_surface, \
            "reference_surface 应该与 prop_lens 结果一致"
        assert wfo1.beam_type_old == wfo2.beam_type_old, \
            "beam_type_old 应该与 prop_lens 结果一致"


class TestComputeReferencePhase:
    """测试 _compute_reference_phase 方法
    
    **Validates: Requirements 2.1, 2.2, 2.3, 2.4**
    """
    
    @pytest.fixture
    def system(self):
        """创建测试用的光学系统"""
        source = GaussianBeamSource(
            wavelength=0.633,  # μm
            w0=1.0,            # mm
            z0=0.0,            # mm
        )
        return SequentialOpticalSystem(source, grid_size=128)
    
    @pytest.fixture
    def wfo(self, system):
        """创建测试用的 PROPER 波前对象"""
        import proper
        
        beam = system._source.to_gaussian_beam()
        wavelength_m = system._source.wavelength * 1e-6
        w_init = system._source.w(0.0)
        beam_diameter_m = 4 * w_init * 1e-3
        
        wfo = proper.prop_begin(
            beam_diameter_m, wavelength_m, system._grid_size, system._beam_ratio
        )
        return wfo
    
    def test_planar_reference_returns_zero(self, system, wfo):
        """测试 PLANAR 参考面返回零相位
        
        **Validates: Requirements 2.1**
        """
        # 设置为 PLANAR 参考面
        wfo.reference_surface = "PLANAR"
        
        # 创建测试坐标
        x_mm = np.array([0, 1, 2, -1, -2])
        y_mm = np.array([0, 0, 0, 0, 0])
        
        # 计算参考相位
        phase_ref = system._compute_reference_phase(wfo, x_mm, y_mm)
        
        # 验证返回零
        np.testing.assert_array_equal(phase_ref, np.zeros_like(x_mm),
                                      err_msg="PLANAR 参考面应返回零相位")
    
    def test_spheri_reference_returns_quadratic_phase(self, system, wfo):
        """测试 SPHERI 参考面返回二次相位
        
        **Validates: Requirements 2.2, 2.3**
        """
        # 设置为 SPHERI 参考面
        wfo.reference_surface = "SPHERI"
        wfo.z_w0 = 0.0  # 束腰在原点
        wfo.z = 0.1     # 当前位置在 100mm 处
        
        # 创建测试坐标
        x_mm = np.array([0, 1, 2, -1, -2])
        y_mm = np.array([0, 0, 0, 0, 0])
        
        # 计算参考相位
        phase_ref = system._compute_reference_phase(wfo, x_mm, y_mm)
        
        # 验证中心点相位为零
        assert phase_ref[0] == 0.0, "中心点相位应为零"
        
        # 验证相位与 r² 成正比
        r_sq = x_mm**2 + y_mm**2
        # 相位应该与 r² 成正比（除了中心点）
        non_zero_mask = r_sq > 0
        if np.any(non_zero_mask):
            ratio = phase_ref[non_zero_mask] / r_sq[non_zero_mask]
            # 所有比值应该相同
            np.testing.assert_allclose(ratio, ratio[0], rtol=1e-10,
                                       err_msg="相位应与 r² 成正比")
    
    def test_spheri_reference_formula(self, system, wfo):
        """测试 SPHERI 参考面相位公式正确性
        
        验证公式：φ_ref = -k * r² / (2 * R_ref)
        
        **Validates: Requirements 2.4**
        """
        # 设置为 SPHERI 参考面
        wfo.reference_surface = "SPHERI"
        R_ref_m = 0.1  # 参考球面曲率半径 100mm
        wfo.z_w0 = 0.0
        wfo.z = R_ref_m
        
        # 创建测试坐标
        r_mm = 5.0  # 5mm
        x_mm = np.array([r_mm])
        y_mm = np.array([0.0])
        
        # 计算参考相位
        phase_ref = system._compute_reference_phase(wfo, x_mm, y_mm)
        
        # 计算期望值
        wavelength_m = wfo.lamda
        k = 2 * np.pi / wavelength_m
        r_m = r_mm * 1e-3
        expected_phase = -k * r_m**2 / (2 * R_ref_m)
        
        np.testing.assert_allclose(phase_ref[0], expected_phase, rtol=1e-10,
                                   err_msg="参考相位公式计算错误")


class TestCheckPhaseSampling:
    """测试 _check_phase_sampling 方法
    
    **Validates: Requirements 7.1, 7.2, 7.3**
    """
    
    @pytest.fixture
    def system(self):
        """创建测试用的光学系统"""
        source = GaussianBeamSource(
            wavelength=0.633,  # μm
            w0=1.0,            # mm
            z0=0.0,            # mm
        )
        return SequentialOpticalSystem(source, grid_size=128)
    
    def test_no_warning_for_small_gradient(self, system):
        """测试正常相位梯度不发出警告
        
        **Validates: Requirements 7.1**
        """
        # 创建小梯度相位网格
        n = 64
        x = np.linspace(-10, 10, n)
        X, Y = np.meshgrid(x, x)
        # 小梯度相位（最大梯度 < π）
        phase_grid = 0.1 * (X + Y)  # 梯度约 0.1 * 20/64 ≈ 0.03 弧度/像素
        
        sampling_mm = 20.0 / n
        
        # 不应该发出警告
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            system._check_phase_sampling(phase_grid, sampling_mm)
            assert len(w) == 0, "小梯度相位不应发出警告"
    
    def test_warning_for_large_gradient(self, system):
        """测试过大相位梯度发出警告
        
        **Validates: Requirements 7.2, 7.3**
        """
        # 创建大梯度相位网格
        n = 64
        x = np.linspace(-10, 10, n)
        X, Y = np.meshgrid(x, x)
        # 大梯度相位（最大梯度 > π）
        phase_grid = 10.0 * X  # 梯度约 10 * 20/64 ≈ 3.1 弧度/像素 > π
        
        sampling_mm = 20.0 / n
        
        # 应该发出警告
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            system._check_phase_sampling(phase_grid, sampling_mm)
            assert len(w) == 1, "大梯度相位应发出警告"
            assert "相位采样不足" in str(w[0].message), "警告信息应包含'相位采样不足'"


class TestGetSamplingHalfSizeMm:
    """测试 _get_sampling_half_size_mm 方法
    
    **Validates: Requirements 1.1**
    """
    
    @pytest.fixture
    def system(self):
        """创建测试用的光学系统"""
        source = GaussianBeamSource(
            wavelength=0.633,  # μm
            w0=1.0,            # mm
            z0=0.0,            # mm
        )
        return SequentialOpticalSystem(source, grid_size=128)
    
    @pytest.fixture
    def wfo(self, system):
        """创建测试用的 PROPER 波前对象"""
        import proper
        
        beam = system._source.to_gaussian_beam()
        wavelength_m = system._source.wavelength * 1e-6
        w_init = system._source.w(0.0)
        beam_diameter_m = 4 * w_init * 1e-3
        
        wfo = proper.prop_begin(
            beam_diameter_m, wavelength_m, system._grid_size, system._beam_ratio
        )
        return wfo
    
    def test_returns_half_grid_size(self, system, wfo):
        """测试返回正确的半尺寸
        
        **Validates: Requirements 1.1**
        """
        import proper
        
        half_size_mm = system._get_sampling_half_size_mm(wfo)
        
        # 计算期望值
        n = proper.prop_get_gridsize(wfo)
        sampling_m = proper.prop_get_sampling(wfo)
        sampling_mm = sampling_m * 1e3
        expected_half_size = sampling_mm * n / 2
        
        np.testing.assert_allclose(half_size_mm, expected_half_size, rtol=1e-10,
                                   err_msg="采样面半尺寸计算错误")


class TestCreateSamplingRays:
    """测试 _create_sampling_rays 方法
    
    **Validates: Requirements 1.1**
    """
    
    @pytest.fixture
    def system(self):
        """创建测试用的光学系统"""
        source = GaussianBeamSource(
            wavelength=0.633,  # μm
            w0=1.0,            # mm
            z0=0.0,            # mm
        )
        return SequentialOpticalSystem(source, grid_size=128, hybrid_num_rays=100)
    
    def test_creates_uniform_grid(self, system):
        """测试创建均匀分布的采样点
        
        **Validates: Requirements 1.1**
        """
        half_size_mm = 10.0
        ray_x, ray_y = system._create_sampling_rays(half_size_mm)
        
        # 验证光线数量
        n_rays_1d = int(np.sqrt(system._hybrid_num_rays))
        expected_n_rays = n_rays_1d ** 2
        assert len(ray_x) == expected_n_rays, f"光线数量应为 {expected_n_rays}"
        assert len(ray_y) == expected_n_rays, f"光线数量应为 {expected_n_rays}"
        
        # 验证范围
        assert np.min(ray_x) == -half_size_mm, "X 最小值应为 -half_size_mm"
        assert np.max(ray_x) == half_size_mm, "X 最大值应为 half_size_mm"
        assert np.min(ray_y) == -half_size_mm, "Y 最小值应为 -half_size_mm"
        assert np.max(ray_y) == half_size_mm, "Y 最大值应为 half_size_mm"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# =============================================================================
# 属性测试（Property-Based Testing）
# =============================================================================

class TestGaussianParamsUpdateProperty:
    """高斯光束参数更新正确性属性测试
    
    **Feature: hybrid-propagation-raytracing-opd, Property 2: 高斯光束参数更新正确性**
    
    *For any* 有效的焦距值和初始高斯光束参数，`_update_gaussian_params_only` 方法
    更新后的参数（z_w0, w0, z_Rayleigh, beam_type_old, reference_surface）应与 
    `prop_lens` 的更新结果一致。
    
    **Validates: Requirements 3.1, 3.2, 3.3, 3.4**
    """
    
    # 定义策略：焦距范围（正负焦距，避免接近零的值）
    focal_length_strategy = st.one_of(
        st.floats(min_value=0.01, max_value=1.0),   # 正焦距：10mm 到 1000mm
        st.floats(min_value=-1.0, max_value=-0.01)  # 负焦距：-1000mm 到 -10mm
    )
    
    # 定义策略：束腰半径（0.1mm 到 10mm）
    w0_strategy = st.floats(min_value=0.1, max_value=10.0)
    
    # 定义策略：波长（可见光范围，0.4μm 到 1.0μm）
    wavelength_strategy = st.floats(min_value=0.4, max_value=1.0)
    
    # 定义策略：初始传播距离（0 到 100mm）
    propagation_distance_strategy = st.floats(min_value=0.0, max_value=0.1)
    
    @given(
        focal_length_m=focal_length_strategy,
        w0_mm=w0_strategy,
        wavelength_um=wavelength_strategy,
        prop_distance_m=propagation_distance_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_params_consistency_with_prop_lens(
        self,
        focal_length_m: float,
        w0_mm: float,
        wavelength_um: float,
        prop_distance_m: float,
    ):
        """
        **Validates: Property 2 - 高斯光束参数更新正确性**
        
        验证 _update_gaussian_params_only 与 prop_lens 的参数更新结果一致。
        
        测试策略：
        1. 使用随机生成的焦距、束腰半径、波长和传播距离
        2. 创建两个相同的波前对象
        3. 分别使用 _update_gaussian_params_only 和 prop_lens 更新参数
        4. 验证所有高斯光束参数一致
        """
        import proper
        
        # 过滤掉可能导致数值问题的边界情况
        # 避免焦距与当前曲率半径相等的情况（会导致无穷大）
        assume(abs(focal_length_m) > 0.005)
        
        # 创建光学系统
        source = GaussianBeamSource(
            wavelength=wavelength_um,
            w0=w0_mm,
            z0=0.0,
        )
        system = SequentialOpticalSystem(source, grid_size=64)
        
        # 计算光束直径
        wavelength_m = wavelength_um * 1e-6
        beam_diameter_m = 4 * w0_mm * 1e-3
        
        # 创建两个相同的波前对象
        wfo1 = proper.prop_begin(beam_diameter_m, wavelength_m, 64, 0.5)
        wfo2 = proper.prop_begin(beam_diameter_m, wavelength_m, 64, 0.5)
        
        # 如果有传播距离，先传播
        if prop_distance_m > 0:
            proper.prop_propagate(wfo1, prop_distance_m)
            proper.prop_propagate(wfo2, prop_distance_m)
        
        # 检查是否会导致数值问题
        # 当 gR_beam_old == focal_length_m 时会导致除零
        if (wfo1.z - wfo1.z_w0) != 0.0:
            z_Rayleigh = np.pi * wfo1.w0**2 / wfo1.lamda
            gR_beam_old = (wfo1.z - wfo1.z_w0) + z_Rayleigh**2 / (wfo1.z - wfo1.z_w0)
            # 避免 gR_beam_old 接近 focal_length_m 的情况
            assume(abs(gR_beam_old - focal_length_m) > 1e-6)
        
        # 使用 _update_gaussian_params_only 更新 wfo1
        system._update_gaussian_params_only(wfo1, focal_length_m)
        
        # 使用 prop_lens 更新 wfo2（会同时更新参数和 wfarr）
        proper.prop_lens(wfo2, focal_length_m)
        
        # 验证参数一致性
        # z_w0: 虚拟束腰位置
        np.testing.assert_allclose(
            wfo1.z_w0, wfo2.z_w0, rtol=1e-10,
            err_msg=f"z_w0 不一致: {wfo1.z_w0} vs {wfo2.z_w0}"
        )
        
        # w0: 束腰半径
        np.testing.assert_allclose(
            wfo1.w0, wfo2.w0, rtol=1e-10,
            err_msg=f"w0 不一致: {wfo1.w0} vs {wfo2.w0}"
        )
        
        # z_Rayleigh: 瑞利距离
        np.testing.assert_allclose(
            wfo1.z_Rayleigh, wfo2.z_Rayleigh, rtol=1e-10,
            err_msg=f"z_Rayleigh 不一致: {wfo1.z_Rayleigh} vs {wfo2.z_Rayleigh}"
        )
        
        # beam_type_old: 光束类型
        assert wfo1.beam_type_old == wfo2.beam_type_old, \
            f"beam_type_old 不一致: {wfo1.beam_type_old} vs {wfo2.beam_type_old}"
        
        # reference_surface: 参考面类型
        assert wfo1.reference_surface == wfo2.reference_surface, \
            f"reference_surface 不一致: {wfo1.reference_surface} vs {wfo2.reference_surface}"
    
    @given(
        focal_length_m=st.floats(min_value=0.01, max_value=0.5),
        w0_mm=st.floats(min_value=0.5, max_value=5.0),
    )
    @settings(max_examples=50, deadline=None)
    def test_inside_beam_type_transition(
        self,
        focal_length_m: float,
        w0_mm: float,
    ):
        """
        **Validates: Property 2 - INSIDE_ 光束类型转换正确性**
        
        验证当光束在瑞利距离内时，beam_type_old 应为 "INSIDE_"，
        reference_surface 应为 "PLANAR"。
        """
        import proper
        
        # 创建光学系统
        source = GaussianBeamSource(
            wavelength=0.633,
            w0=w0_mm,
            z0=0.0,
        )
        system = SequentialOpticalSystem(source, grid_size=64)
        
        # 创建波前对象
        wavelength_m = 0.633e-6
        beam_diameter_m = 4 * w0_mm * 1e-3
        
        wfo1 = proper.prop_begin(beam_diameter_m, wavelength_m, 64, 0.5)
        wfo2 = proper.prop_begin(beam_diameter_m, wavelength_m, 64, 0.5)
        
        # 使用两种方法更新参数
        system._update_gaussian_params_only(wfo1, focal_length_m)
        proper.prop_lens(wfo2, focal_length_m)
        
        # 验证光束类型一致
        assert wfo1.beam_type_old == wfo2.beam_type_old, \
            f"beam_type_old 不一致: {wfo1.beam_type_old} vs {wfo2.beam_type_old}"
        
        # 验证参考面类型一致
        assert wfo1.reference_surface == wfo2.reference_surface, \
            f"reference_surface 不一致: {wfo1.reference_surface} vs {wfo2.reference_surface}"
        
        # 验证逻辑一致性：INSIDE_ 对应 PLANAR，OUTSIDE 对应 SPHERI
        if wfo1.beam_type_old == "INSIDE_":
            assert wfo1.reference_surface == "PLANAR", \
                "INSIDE_ 光束类型应对应 PLANAR 参考面"
        else:
            assert wfo1.reference_surface == "SPHERI", \
                "OUTSIDE 光束类型应对应 SPHERI 参考面"
    
    @given(
        focal_length_m=st.floats(min_value=-0.5, max_value=-0.01),
        w0_mm=st.floats(min_value=0.5, max_value=5.0),
    )
    @settings(max_examples=50, deadline=None)
    def test_negative_focal_length_consistency(
        self,
        focal_length_m: float,
        w0_mm: float,
    ):
        """
        **Validates: Property 2 - 负焦距（凸面镜/发散透镜）参数更新正确性**
        
        验证负焦距情况下参数更新的一致性。
        """
        import proper
        
        # 创建光学系统
        source = GaussianBeamSource(
            wavelength=0.633,
            w0=w0_mm,
            z0=0.0,
        )
        system = SequentialOpticalSystem(source, grid_size=64)
        
        # 创建波前对象
        wavelength_m = 0.633e-6
        beam_diameter_m = 4 * w0_mm * 1e-3
        
        wfo1 = proper.prop_begin(beam_diameter_m, wavelength_m, 64, 0.5)
        wfo2 = proper.prop_begin(beam_diameter_m, wavelength_m, 64, 0.5)
        
        # 使用两种方法更新参数
        system._update_gaussian_params_only(wfo1, focal_length_m)
        proper.prop_lens(wfo2, focal_length_m)
        
        # 验证所有参数一致
        np.testing.assert_allclose(wfo1.z_w0, wfo2.z_w0, rtol=1e-10)
        np.testing.assert_allclose(wfo1.w0, wfo2.w0, rtol=1e-10)
        np.testing.assert_allclose(wfo1.z_Rayleigh, wfo2.z_Rayleigh, rtol=1e-10)
        assert wfo1.beam_type_old == wfo2.beam_type_old
        assert wfo1.reference_surface == wfo2.reference_surface
    
    @given(
        focal_length_m=focal_length_strategy,
        w0_mm=w0_strategy,
    )
    @settings(max_examples=50, deadline=None)
    def test_wfarr_not_modified(
        self,
        focal_length_m: float,
        w0_mm: float,
    ):
        """
        **Validates: Property 2 - _update_gaussian_params_only 不修改 wfarr**
        
        验证 _update_gaussian_params_only 方法只更新参数，不修改波前数组。
        """
        import proper
        
        assume(abs(focal_length_m) > 0.005)
        
        # 创建光学系统
        source = GaussianBeamSource(
            wavelength=0.633,
            w0=w0_mm,
            z0=0.0,
        )
        system = SequentialOpticalSystem(source, grid_size=64)
        
        # 创建波前对象
        wavelength_m = 0.633e-6
        beam_diameter_m = 4 * w0_mm * 1e-3
        
        wfo = proper.prop_begin(beam_diameter_m, wavelength_m, 64, 0.5)
        
        # 保存原始 wfarr
        wfarr_before = wfo.wfarr.copy()
        
        # 使用 _update_gaussian_params_only 更新参数
        system._update_gaussian_params_only(wfo, focal_length_m)
        
        # 验证 wfarr 未被修改
        np.testing.assert_array_equal(
            wfo.wfarr, wfarr_before,
            err_msg="_update_gaussian_params_only 不应修改 wfarr"
        )
    
    @given(
        focal_length_m=focal_length_strategy,
        w0_mm=w0_strategy,
    )
    @settings(max_examples=50, deadline=None)
    def test_current_fratio_consistency(
        self,
        focal_length_m: float,
        w0_mm: float,
    ):
        """
        **Validates: Property 2 - current_fratio 更新正确性**
        
        验证 current_fratio 参数更新的一致性。
        """
        import proper
        
        assume(abs(focal_length_m) > 0.005)
        
        # 创建光学系统
        source = GaussianBeamSource(
            wavelength=0.633,
            w0=w0_mm,
            z0=0.0,
        )
        system = SequentialOpticalSystem(source, grid_size=64)
        
        # 创建波前对象
        wavelength_m = 0.633e-6
        beam_diameter_m = 4 * w0_mm * 1e-3
        
        wfo1 = proper.prop_begin(beam_diameter_m, wavelength_m, 64, 0.5)
        wfo2 = proper.prop_begin(beam_diameter_m, wavelength_m, 64, 0.5)
        
        # 使用两种方法更新参数
        system._update_gaussian_params_only(wfo1, focal_length_m)
        proper.prop_lens(wfo2, focal_length_m)
        
        # 验证 current_fratio 一致
        np.testing.assert_allclose(
            wfo1.current_fratio, wfo2.current_fratio, rtol=1e-10,
            err_msg=f"current_fratio 不一致: {wfo1.current_fratio} vs {wfo2.current_fratio}"
        )


class TestPhaseSamplingCheckProperty:
    """相位采样检查正确性属性测试
    
    **Feature: hybrid-propagation-raytracing-opd, Property 9: 相位采样检查正确性**
    
    *For any* 相位网格，当相邻像素间相位差超过 π 时，`_check_phase_sampling` 方法
    应该发出警告。
    
    **Validates: Requirements 7.1, 7.2, 7.3**
    """
    
    # 定义策略：网格大小（32 到 128）
    grid_size_strategy = st.sampled_from([32, 64, 128])
    
    # 定义策略：采样间隔（0.1mm 到 1.0mm）
    sampling_strategy = st.floats(min_value=0.1, max_value=1.0)
    
    # 定义策略：相位梯度系数（控制相位变化速率）
    # 小系数：不应警告；大系数：应警告
    small_gradient_strategy = st.floats(min_value=0.01, max_value=0.1)
    large_gradient_strategy = st.floats(min_value=5.0, max_value=20.0)
    
    @given(
        grid_size=grid_size_strategy,
        sampling_mm=sampling_strategy,
        gradient_coeff=small_gradient_strategy,
    )
    @settings(max_examples=50, deadline=None)
    def test_no_warning_for_small_gradient(
        self,
        grid_size: int,
        sampling_mm: float,
        gradient_coeff: float,
    ):
        """
        **Validates: Property 9 - 小相位梯度不应发出警告**
        
        验证当相邻像素间相位差小于 π 时，不发出警告。
        
        测试策略：
        1. 使用随机生成的网格大小和采样间隔
        2. 创建小梯度相位网格（梯度系数 < 0.1）
        3. 验证不发出警告
        """
        # 创建光学系统
        source = GaussianBeamSource(
            wavelength=0.633,
            w0=1.0,
            z0=0.0,
        )
        system = SequentialOpticalSystem(source, grid_size=64)
        
        # 创建小梯度相位网格
        half_size = sampling_mm * grid_size / 2
        x = np.linspace(-half_size, half_size, grid_size)
        X, Y = np.meshgrid(x, x)
        
        # 相位梯度 = gradient_coeff * sampling_mm
        # 确保梯度 < π
        phase_grid = gradient_coeff * X
        
        # 计算实际最大梯度
        max_grad = gradient_coeff * sampling_mm
        
        # 只有当最大梯度 < π 时才测试
        assume(max_grad < np.pi * 0.9)  # 留一些余量
        
        # 不应该发出警告
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            system._check_phase_sampling(phase_grid, sampling_mm)
            
            # 过滤出相位采样相关的警告
            phase_warnings = [
                warning for warning in w 
                if "相位采样不足" in str(warning.message)
            ]
            
            assert len(phase_warnings) == 0, \
                f"小梯度相位（max_grad={max_grad:.3f}）不应发出警告"
    
    @given(
        grid_size=grid_size_strategy,
        sampling_mm=sampling_strategy,
        gradient_coeff=large_gradient_strategy,
    )
    @settings(max_examples=50, deadline=None)
    def test_warning_for_large_gradient(
        self,
        grid_size: int,
        sampling_mm: float,
        gradient_coeff: float,
    ):
        """
        **Validates: Property 9 - 大相位梯度应发出警告**
        
        验证当相邻像素间相位差超过 π 时，发出警告。
        
        测试策略：
        1. 使用随机生成的网格大小和采样间隔
        2. 创建大梯度相位网格（梯度系数 > 5.0）
        3. 验证发出警告
        """
        # 创建光学系统
        source = GaussianBeamSource(
            wavelength=0.633,
            w0=1.0,
            z0=0.0,
        )
        system = SequentialOpticalSystem(source, grid_size=64)
        
        # 创建大梯度相位网格
        half_size = sampling_mm * grid_size / 2
        x = np.linspace(-half_size, half_size, grid_size)
        X, Y = np.meshgrid(x, x)
        
        # 相位梯度 = gradient_coeff * sampling_mm
        # 确保梯度 > π
        phase_grid = gradient_coeff * X
        
        # 计算实际最大梯度
        max_grad = gradient_coeff * sampling_mm
        
        # 只有当最大梯度 > π 时才测试
        assume(max_grad > np.pi * 1.1)  # 留一些余量
        
        # 应该发出警告
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            system._check_phase_sampling(phase_grid, sampling_mm)
            
            # 过滤出相位采样相关的警告
            phase_warnings = [
                warning for warning in w 
                if "相位采样不足" in str(warning.message)
            ]
            
            assert len(phase_warnings) == 1, \
                f"大梯度相位（max_grad={max_grad:.3f}）应发出警告"
    
    @given(
        grid_size=grid_size_strategy,
        sampling_mm=sampling_strategy,
    )
    @settings(max_examples=30, deadline=None)
    def test_boundary_gradient_near_pi(
        self,
        grid_size: int,
        sampling_mm: float,
    ):
        """
        **Validates: Property 9 - 边界情况：梯度接近 π**
        
        验证当相邻像素间相位差接近 π 时的行为。
        
        测试策略：
        1. 创建梯度略小于 π 的相位网格，验证不警告
        2. 创建梯度略大于 π 的相位网格，验证警告
        """
        # 创建光学系统
        source = GaussianBeamSource(
            wavelength=0.633,
            w0=1.0,
            z0=0.0,
        )
        system = SequentialOpticalSystem(source, grid_size=64)
        
        # 创建相位网格
        half_size = sampling_mm * grid_size / 2
        x = np.linspace(-half_size, half_size, grid_size)
        X, Y = np.meshgrid(x, x)
        
        # 测试 1：梯度略小于 π（0.9π）
        gradient_below = 0.9 * np.pi / sampling_mm
        phase_below = gradient_below * X
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            system._check_phase_sampling(phase_below, sampling_mm)
            phase_warnings = [
                warning for warning in w 
                if "相位采样不足" in str(warning.message)
            ]
            assert len(phase_warnings) == 0, \
                "梯度 0.9π 不应发出警告"
        
        # 测试 2：梯度略大于 π（1.1π）
        gradient_above = 1.1 * np.pi / sampling_mm
        phase_above = gradient_above * X
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            system._check_phase_sampling(phase_above, sampling_mm)
            phase_warnings = [
                warning for warning in w 
                if "相位采样不足" in str(warning.message)
            ]
            assert len(phase_warnings) == 1, \
                "梯度 1.1π 应发出警告"
    
    @given(
        grid_size=grid_size_strategy,
        sampling_mm=sampling_strategy,
    )
    @settings(max_examples=30, deadline=None)
    def test_2d_gradient_detection(
        self,
        grid_size: int,
        sampling_mm: float,
    ):
        """
        **Validates: Property 9 - 2D 梯度检测**
        
        验证方法能检测 X 和 Y 方向的大梯度。
        
        测试策略：
        1. 创建只有 X 方向大梯度的相位网格，验证警告
        2. 创建只有 Y 方向大梯度的相位网格，验证警告
        """
        # 创建光学系统
        source = GaussianBeamSource(
            wavelength=0.633,
            w0=1.0,
            z0=0.0,
        )
        system = SequentialOpticalSystem(source, grid_size=64)
        
        # 创建相位网格
        half_size = sampling_mm * grid_size / 2
        x = np.linspace(-half_size, half_size, grid_size)
        X, Y = np.meshgrid(x, x)
        
        # 大梯度系数
        large_gradient = 2.0 * np.pi / sampling_mm
        
        # 测试 1：只有 X 方向大梯度
        phase_x = large_gradient * X
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            system._check_phase_sampling(phase_x, sampling_mm)
            phase_warnings = [
                warning for warning in w 
                if "相位采样不足" in str(warning.message)
            ]
            assert len(phase_warnings) == 1, \
                "X 方向大梯度应发出警告"
        
        # 测试 2：只有 Y 方向大梯度
        phase_y = large_gradient * Y
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            system._check_phase_sampling(phase_y, sampling_mm)
            phase_warnings = [
                warning for warning in w 
                if "相位采样不足" in str(warning.message)
            ]
            assert len(phase_warnings) == 1, \
                "Y 方向大梯度应发出警告"
