"""
采样面和仿真结果属性测试

使用 Hypothesis 进行属性基测试，验证：
- Property 5: 采样面数据完整性
- Property 7: 仿真结果完整性

**Validates: Requirements 4.3, 4.4, 4.5, 4.6, 5.6**
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings

from sequential_system.sampling import SamplingPlane, SamplingResult, SimulationResults
from sequential_system.source import GaussianBeamSource
from sequential_system.exceptions import SamplingError


# ============================================================================
# SamplingPlane 测试
# ============================================================================

class TestSamplingPlaneCreation:
    """测试 SamplingPlane 创建"""
    
    def test_create_basic_sampling_plane(self):
        """测试创建基本采样面"""
        plane = SamplingPlane(distance=100.0)
        
        assert plane.distance == 100.0
        assert plane.name is None
        assert plane.result is None
    
    def test_create_named_sampling_plane(self):
        """测试创建命名采样面"""
        plane = SamplingPlane(distance=150.0, name="focus")
        
        assert plane.distance == 150.0
        assert plane.name == "focus"
    
    def test_create_zero_distance_sampling_plane(self):
        """测试创建零距离采样面"""
        plane = SamplingPlane(distance=0.0, name="entrance")
        
        assert plane.distance == 0.0


class TestSamplingPlaneValidation:
    """测试 SamplingPlane 参数验证"""
    
    def test_reject_negative_distance(self):
        """验证拒绝负距离"""
        with pytest.raises(SamplingError) as exc_info:
            SamplingPlane(distance=-10.0)
        
        assert 'distance' in str(exc_info.value).lower()
    
    def test_reject_nan_distance(self):
        """验证拒绝 NaN 距离"""
        with pytest.raises(SamplingError) as exc_info:
            SamplingPlane(distance=float('nan'))
        
        assert 'distance' in str(exc_info.value).lower()
    
    def test_reject_inf_distance(self):
        """验证拒绝无穷大距离"""
        with pytest.raises(SamplingError) as exc_info:
            SamplingPlane(distance=float('inf'))
        
        assert 'distance' in str(exc_info.value).lower()


# ============================================================================
# SamplingResult 测试
# ============================================================================

class TestSamplingResultProperties:
    """测试 SamplingResult 属性"""
    
    @pytest.fixture
    def sample_wavefront(self):
        """创建示例波前"""
        n = 64
        x = np.linspace(-5, 5, n)
        X, Y = np.meshgrid(x, x)
        R_sq = X**2 + Y**2
        
        # 高斯振幅
        w = 2.0
        amplitude = np.exp(-R_sq / w**2)
        
        # 球面相位
        phase = -0.1 * R_sq
        
        return amplitude * np.exp(1j * phase)
    
    def test_amplitude_extraction(self, sample_wavefront):
        """
        **Validates: Requirements 8.7**
        
        验证振幅提取
        """
        result = SamplingResult(
            distance=100.0,
            z_position=100.0,
            wavefront=sample_wavefront,
            sampling=0.15625,
            beam_radius=2.0,
        )
        
        amplitude = result.amplitude
        
        assert amplitude.shape == sample_wavefront.shape
        assert np.all(amplitude >= 0)
        np.testing.assert_allclose(amplitude, np.abs(sample_wavefront))
    
    def test_phase_extraction(self, sample_wavefront):
        """
        **Validates: Requirements 8.6**
        
        验证相位提取
        """
        result = SamplingResult(
            distance=100.0,
            z_position=100.0,
            wavefront=sample_wavefront,
            sampling=0.15625,
            beam_radius=2.0,
        )
        
        phase = result.phase
        
        assert phase.shape == sample_wavefront.shape
        np.testing.assert_allclose(phase, np.angle(sample_wavefront))
    
    def test_grid_size(self, sample_wavefront):
        """验证网格大小"""
        result = SamplingResult(
            distance=100.0,
            z_position=100.0,
            wavefront=sample_wavefront,
            sampling=0.15625,
            beam_radius=2.0,
        )
        
        assert result.grid_size == 64
    
    def test_physical_size(self, sample_wavefront):
        """验证物理尺寸"""
        result = SamplingResult(
            distance=100.0,
            z_position=100.0,
            wavefront=sample_wavefront,
            sampling=0.15625,
            beam_radius=2.0,
        )
        
        expected_size = 0.15625 * 64
        assert result.physical_size == expected_size
    
    def test_wavefront_rms(self, sample_wavefront):
        """
        **Validates: Requirements 8.5**
        
        验证波前 RMS 计算
        """
        result = SamplingResult(
            distance=100.0,
            z_position=100.0,
            wavefront=sample_wavefront,
            sampling=0.15625,
            beam_radius=2.0,
        )
        
        rms = result.wavefront_rms
        
        assert isinstance(rms, float)
        assert rms >= 0
    
    def test_wavefront_pv(self, sample_wavefront):
        """
        **Validates: Requirements 8.5**
        
        验证波前 PV 计算
        """
        result = SamplingResult(
            distance=100.0,
            z_position=100.0,
            wavefront=sample_wavefront,
            sampling=0.15625,
            beam_radius=2.0,
        )
        
        pv = result.wavefront_pv
        
        assert isinstance(pv, float)
        assert pv >= 0
        assert pv >= result.wavefront_rms  # PV 应该 >= RMS
    
    def test_compute_m2(self, sample_wavefront):
        """
        **Validates: Requirements 8.4**
        
        验证 M² 计算
        """
        result = SamplingResult(
            distance=100.0,
            z_position=100.0,
            wavefront=sample_wavefront,
            sampling=0.15625,
            beam_radius=2.0,
        )
        
        m2 = result.compute_m2()
        
        assert isinstance(m2, float)
        assert m2 >= 1.0  # M² 必须 >= 1


# ============================================================================
# Property 5: 采样面数据完整性
# ============================================================================

@given(
    distance=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    grid_size=st.sampled_from([32, 64, 128, 256]),
    sampling=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
    beam_radius=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50)
def test_sampling_result_data_integrity(distance, grid_size, sampling, beam_radius):
    """
    **Feature: sequential-optical-system, Property 5: 采样面数据完整性**
    **Validates: Requirements 4.3, 4.4, 4.5, 4.6**
    
    验证采样结果包含完整的数据
    """
    # 创建测试波前
    n = grid_size
    x = np.linspace(-5, 5, n)
    X, Y = np.meshgrid(x, x)
    R_sq = X**2 + Y**2
    amplitude = np.exp(-R_sq / 4.0)
    phase = -0.05 * R_sq
    wavefront = amplitude * np.exp(1j * phase)
    
    result = SamplingResult(
        distance=distance,
        z_position=distance,
        wavefront=wavefront,
        sampling=sampling,
        beam_radius=beam_radius,
    )
    
    # 验证数据完整性
    assert result.wavefront is not None
    assert result.wavefront.shape == (grid_size, grid_size)
    assert result.sampling > 0
    assert result.beam_radius > 0
    assert result.grid_size == grid_size


# ============================================================================
# SimulationResults 测试
# ============================================================================

class TestSimulationResults:
    """测试 SimulationResults 容器类"""
    
    @pytest.fixture
    def sample_results(self):
        """创建示例仿真结果"""
        source = GaussianBeamSource(wavelength=0.633, w0=1.0, z0=-50.0)
        
        # 创建示例波前
        n = 64
        x = np.linspace(-5, 5, n)
        X, Y = np.meshgrid(x, x)
        R_sq = X**2 + Y**2
        wavefront = np.exp(-R_sq / 4.0) * np.exp(-0.05j * R_sq)
        
        result1 = SamplingResult(
            distance=100.0,
            z_position=100.0,
            wavefront=wavefront,
            sampling=0.15625,
            beam_radius=2.0,
            name="plane1",
        )
        
        result2 = SamplingResult(
            distance=200.0,
            z_position=200.0,
            wavefront=wavefront,
            sampling=0.15625,
            beam_radius=3.0,
            name="plane2",
        )
        
        return SimulationResults(
            sampling_results={"plane1": result1, "plane2": result2},
            source=source,
            surfaces=[],
        )
    
    def test_access_by_name(self, sample_results):
        """
        **Validates: Requirements 5.6**
        
        验证通过名称访问
        """
        result = sample_results["plane1"]
        
        assert result.name == "plane1"
        assert result.distance == 100.0
    
    def test_access_by_index(self, sample_results):
        """
        **Validates: Requirements 5.6**
        
        验证通过索引访问
        """
        result = sample_results[0]
        
        assert result.name == "plane1"
    
    def test_iteration(self, sample_results):
        """
        **Validates: Requirements 5.6**
        
        验证迭代
        """
        results_list = list(sample_results)
        
        assert len(results_list) == 2
        assert results_list[0].name == "plane1"
        assert results_list[1].name == "plane2"
    
    def test_length(self, sample_results):
        """验证长度"""
        assert len(sample_results) == 2
    
    def test_invalid_name_raises_key_error(self, sample_results):
        """验证无效名称抛出 KeyError"""
        with pytest.raises(KeyError):
            _ = sample_results["nonexistent"]
    
    def test_invalid_index_raises_index_error(self, sample_results):
        """验证无效索引抛出 IndexError"""
        with pytest.raises(IndexError):
            _ = sample_results[10]


# ============================================================================
# Property 7: 仿真结果完整性
# ============================================================================

@given(
    n_planes=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=20)
def test_simulation_results_completeness(n_planes):
    """
    **Feature: sequential-optical-system, Property 7: 仿真结果完整性**
    **Validates: Requirements 5.6, 8.1, 8.2, 8.3**
    
    验证仿真结果包含所有采样面的完整数据
    """
    source = GaussianBeamSource(wavelength=0.633, w0=1.0, z0=-50.0)
    
    # 创建示例波前
    n = 64
    x = np.linspace(-5, 5, n)
    X, Y = np.meshgrid(x, x)
    R_sq = X**2 + Y**2
    wavefront = np.exp(-R_sq / 4.0) * np.exp(-0.05j * R_sq)
    
    # 创建多个采样结果
    sampling_results = {}
    for i in range(n_planes):
        name = f"plane{i}"
        result = SamplingResult(
            distance=100.0 * (i + 1),
            z_position=100.0 * (i + 1),
            wavefront=wavefront.copy(),
            sampling=0.15625,
            beam_radius=2.0 + 0.5 * i,
            name=name,
        )
        sampling_results[name] = result
    
    results = SimulationResults(
        sampling_results=sampling_results,
        source=source,
        surfaces=[],
    )
    
    # 验证完整性
    assert len(results) == n_planes
    
    for i, result in enumerate(results):
        assert result.wavefront is not None
        assert result.amplitude is not None
        assert result.phase is not None
        assert result.beam_radius > 0
        assert result.sampling > 0
