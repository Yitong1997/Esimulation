"""
PilotBeamCalculator 单元测试

测试 Pilot Beam 参考相位计算器的各项功能：
- 两种方法的输出格式
- 参考相位的合理性
- 方法切换功能

作者：混合光学仿真项目
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import sys
sys.path.insert(0, 'src')

from hybrid_propagation.pilot_beam import PilotBeamCalculator, PilotBeamValidator
from hybrid_propagation import PilotBeamValidationResult


class TestPilotBeamCalculatorInit:
    """测试 PilotBeamCalculator 初始化"""
    
    def test_init_with_default_params(self):
        """测试使用默认参数初始化"""
        calculator = PilotBeamCalculator(
            wavelength=0.633,
            beam_waist=5.0,
        )
        
        assert calculator.wavelength == 0.633
        assert calculator.beam_waist == 5.0
        assert calculator.beam_waist_position == 0.0
        assert calculator.element_focal_length == float('inf')
        assert calculator.method == 'analytical'
        assert calculator.grid_size == 64
        assert calculator.physical_size == 20.0
    
    def test_init_with_all_params(self):
        """测试使用所有参数初始化"""
        calculator = PilotBeamCalculator(
            wavelength=0.55,
            beam_waist=3.0,
            beam_waist_position=10.0,
            element_focal_length=100.0,
            method='proper',
            grid_size=128,
            physical_size=30.0,
        )
        
        assert calculator.wavelength == 0.55
        assert calculator.beam_waist == 3.0
        assert calculator.beam_waist_position == 10.0
        assert calculator.element_focal_length == 100.0
        assert calculator.method == 'proper'
        assert calculator.grid_size == 128
        assert calculator.physical_size == 30.0
    
    def test_wavelength_conversion(self):
        """测试波长单位转换"""
        calculator = PilotBeamCalculator(
            wavelength=0.633,  # μm
            beam_waist=5.0,
        )
        
        # 波长应该被转换为 mm
        assert calculator.wavelength_mm == pytest.approx(0.633e-3, rel=1e-6)
    
    def test_rayleigh_length_calculation(self):
        """测试瑞利长度计算"""
        wavelength = 0.633  # μm
        beam_waist = 5.0    # mm
        
        calculator = PilotBeamCalculator(
            wavelength=wavelength,
            beam_waist=beam_waist,
        )
        
        # 瑞利长度 = π * w0^2 / λ
        wavelength_mm = wavelength * 1e-3
        expected_rayleigh = np.pi * beam_waist**2 / wavelength_mm
        
        assert calculator.rayleigh_length == pytest.approx(expected_rayleigh, rel=1e-6)


class TestComputeReferencePhaseAnalytical:
    """测试解析方法计算参考相位
    
    **Validates: Requirements 4.3**
    """
    
    def test_output_shape(self):
        """测试输出形状"""
        calculator = PilotBeamCalculator(
            wavelength=0.633,
            beam_waist=5.0,
            method='analytical',
        )
        
        n = 64
        x = np.linspace(-10, 10, n)
        y = np.linspace(-10, 10, n)
        
        phase = calculator.compute_reference_phase_analytical(x, y)
        
        assert phase.shape == (n, n)
        assert phase.dtype == np.float64
    
    def test_at_beam_waist_flat_phase(self):
        """测试在束腰位置相位平坦"""
        calculator = PilotBeamCalculator(
            wavelength=0.633,
            beam_waist=5.0,
            beam_waist_position=0.0,  # 元件在束腰位置
            element_focal_length=float('inf'),  # 无焦距
            method='analytical',
        )
        
        n = 64
        x = np.linspace(-10, 10, n)
        y = np.linspace(-10, 10, n)
        
        phase = calculator.compute_reference_phase_analytical(x, y)
        
        # 在束腰位置，相位应该为零（平面波前）
        assert_allclose(phase, 0.0, atol=1e-10)
    
    def test_away_from_waist_curved_phase(self):
        """测试远离束腰位置相位弯曲"""
        calculator = PilotBeamCalculator(
            wavelength=0.633,
            beam_waist=5.0,
            beam_waist_position=100.0,  # 束腰在元件后方 100mm
            element_focal_length=float('inf'),
            method='analytical',
        )
        
        n = 64
        x = np.linspace(-10, 10, n)
        y = np.linspace(-10, 10, n)
        
        phase = calculator.compute_reference_phase_analytical(x, y)
        
        # 远离束腰，相位应该有曲率
        # 中心相位应该接近零
        center = n // 2
        assert abs(phase[center, center]) < 0.1
        
        # 边缘相位应该不为零（可能很小但不为零）
        assert abs(phase[0, 0]) > 1e-6 or abs(phase[-1, -1]) > 1e-6
    
    def test_with_lens_phase(self):
        """测试带透镜相位"""
        calculator = PilotBeamCalculator(
            wavelength=0.633,
            beam_waist=5.0,
            beam_waist_position=0.0,
            element_focal_length=100.0,  # 100mm 焦距
            method='analytical',
        )
        
        n = 64
        x = np.linspace(-10, 10, n)
        y = np.linspace(-10, 10, n)
        
        phase = calculator.compute_reference_phase_analytical(x, y)
        
        # 透镜相位应该是二次的
        # 中心相位可能不为零（取决于网格是否包含原点）
        # 但边缘相位应该不为零（透镜引入的相位）
        assert abs(phase[0, 0]) > 0.1
    
    def test_phase_symmetry(self):
        """测试相位对称性"""
        calculator = PilotBeamCalculator(
            wavelength=0.633,
            beam_waist=5.0,
            beam_waist_position=50.0,
            element_focal_length=float('inf'),
            method='analytical',
        )
        
        n = 64
        x = np.linspace(-10, 10, n)
        y = np.linspace(-10, 10, n)
        
        phase = calculator.compute_reference_phase_analytical(x, y)
        
        # 相位应该关于中心对称
        assert_allclose(phase, phase[::-1, ::-1], rtol=1e-6)


def _proper_available():
    """检查 PROPER 库是否可用"""
    try:
        import proper
        return True
    except ImportError:
        return False


class TestComputeReferencePhaseProper:
    """测试 PROPER 方法计算参考相位
    
    **Validates: Requirements 4.2**
    """
    
    @pytest.mark.skipif(
        not _proper_available(),
        reason="PROPER 库未安装"
    )
    def test_output_shape(self):
        """测试输出形状"""
        calculator = PilotBeamCalculator(
            wavelength=0.633,
            beam_waist=5.0,
            method='proper',
        )
        
        n = 64
        x = np.linspace(-10, 10, n)
        y = np.linspace(-10, 10, n)
        
        phase = calculator.compute_reference_phase_proper(x, y)
        
        assert phase.shape == (n, n)
    
    def test_fallback_to_analytical(self):
        """测试 PROPER 不可用时回退到解析方法"""
        calculator = PilotBeamCalculator(
            wavelength=0.633,
            beam_waist=5.0,
            method='proper',
        )
        
        n = 64
        x = np.linspace(-10, 10, n)
        y = np.linspace(-10, 10, n)
        
        # 即使 PROPER 不可用，也应该返回结果（回退到解析方法）
        phase = calculator.compute_reference_phase_proper(x, y)
        
        assert phase.shape == (n, n)


class TestComputeReferencePhase:
    """测试统一接口
    
    **Validates: Requirements 4.4, 4.5**
    """
    
    def test_analytical_method_selection(self):
        """测试选择解析方法"""
        calculator = PilotBeamCalculator(
            wavelength=0.633,
            beam_waist=5.0,
            method='analytical',
        )
        
        n = 64
        x = np.linspace(-10, 10, n)
        y = np.linspace(-10, 10, n)
        
        phase = calculator.compute_reference_phase(x, y)
        
        # 应该与直接调用解析方法结果相同
        phase_analytical = calculator.compute_reference_phase_analytical(x, y)
        assert_allclose(phase, phase_analytical)
    
    def test_proper_method_selection(self):
        """测试选择 PROPER 方法"""
        calculator = PilotBeamCalculator(
            wavelength=0.633,
            beam_waist=5.0,
            method='proper',
        )
        
        n = 64
        x = np.linspace(-10, 10, n)
        y = np.linspace(-10, 10, n)
        
        # 应该不抛出异常
        phase = calculator.compute_reference_phase(x, y)
        assert phase.shape == (n, n)
    
    def test_invalid_method_raises_error(self):
        """测试无效方法抛出错误"""
        calculator = PilotBeamCalculator(
            wavelength=0.633,
            beam_waist=5.0,
            method='invalid_method',
        )
        
        n = 64
        x = np.linspace(-10, 10, n)
        y = np.linspace(-10, 10, n)
        
        with pytest.raises(ValueError, match="未知的计算方法"):
            calculator.compute_reference_phase(x, y)


class TestValidate:
    """测试验证功能"""
    
    def test_validate_returns_result(self):
        """测试验证返回结果"""
        calculator = PilotBeamCalculator(
            wavelength=0.633,
            beam_waist=5.0,
            grid_size=64,
            physical_size=20.0,
        )
        
        result = calculator.validate(actual_beam_size=8.0)
        
        assert isinstance(result, PilotBeamValidationResult)
    
    def test_validate_with_matching_beam_size(self):
        """测试匹配的光束尺寸"""
        calculator = PilotBeamCalculator(
            wavelength=0.633,
            beam_waist=5.0,  # Pilot Beam 直径 = 10mm
            grid_size=64,
            physical_size=20.0,
        )
        
        result = calculator.validate(actual_beam_size=10.0)
        
        # 尺寸匹配应该通过
        assert result.beam_size_match.passed == True
    
    def test_validate_with_mismatched_beam_size(self):
        """测试不匹配的光束尺寸"""
        calculator = PilotBeamCalculator(
            wavelength=0.633,
            beam_waist=5.0,  # Pilot Beam 直径 = 10mm
            grid_size=64,
            physical_size=20.0,
        )
        
        result = calculator.validate(actual_beam_size=3.0)  # 差异很大
        
        # 尺寸匹配应该失败
        assert result.beam_size_match.passed == False
    
    def test_validate_without_actual_beam_size(self):
        """测试不提供实际光束尺寸"""
        calculator = PilotBeamCalculator(
            wavelength=0.633,
            beam_waist=5.0,
            grid_size=64,
            physical_size=20.0,
        )
        
        result = calculator.validate()
        
        # 应该跳过尺寸匹配检查
        assert "跳过" in result.beam_size_match.message


class TestPhasePhysics:
    """测试相位物理特性"""
    
    def test_quadratic_phase_for_curved_wavefront(self):
        """测试弯曲波前的二次相位"""
        # 远离束腰时，波前应该是球面的，相位是二次的
        calculator = PilotBeamCalculator(
            wavelength=0.633,
            beam_waist=5.0,
            beam_waist_position=1000.0,  # 远离束腰
            element_focal_length=float('inf'),
            method='analytical',
        )
        
        n = 64
        x = np.linspace(-5, 5, n)
        y = np.linspace(-5, 5, n)
        
        phase = calculator.compute_reference_phase_analytical(x, y)
        
        # 沿 x 轴的相位应该近似二次
        center = n // 2
        phase_x = phase[center, :]
        
        # 拟合二次多项式
        coeffs = np.polyfit(x, phase_x, 2)
        
        # 二次项系数应该不为零
        assert abs(coeffs[0]) > 1e-6
    
    def test_lens_phase_sign(self):
        """测试透镜相位符号"""
        # 正焦距透镜应该引入负的二次相位（会聚）
        calculator = PilotBeamCalculator(
            wavelength=0.633,
            beam_waist=5.0,
            beam_waist_position=0.0,
            element_focal_length=100.0,  # 正焦距
            method='analytical',
        )
        
        n = 64
        x = np.linspace(-5, 5, n)
        y = np.linspace(-5, 5, n)
        
        phase = calculator.compute_reference_phase_analytical(x, y)
        
        # 边缘相位应该是负的（会聚透镜）
        assert phase[0, 0] < 0
        assert phase[-1, -1] < 0
    
    def test_divergence_calculation(self):
        """测试发散角计算"""
        wavelength = 0.633  # μm
        beam_waist = 5.0    # mm
        
        calculator = PilotBeamCalculator(
            wavelength=wavelength,
            beam_waist=beam_waist,
        )
        
        # 验证时会计算发散角
        result = calculator.validate()
        
        # 发散角 = λ / (π * w0)
        wavelength_mm = wavelength * 1e-3
        expected_divergence = wavelength_mm / (np.pi * beam_waist)
        
        assert result.beam_divergence.value == pytest.approx(expected_divergence, rel=1e-6)
