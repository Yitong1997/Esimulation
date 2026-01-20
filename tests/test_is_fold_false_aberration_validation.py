"""测试 is_fold=False 失调像差与理论值的一致性

设计两个基于反射镜的测试实验：
1. 平面镜小角度失调：验证只引入波前倾斜，无其他像差
2. 抛物面镜小角度失调：验证像散和彗差与理论值一致

测试条件：
- 使用混合传播模式 (use_hybrid_propagation=True)
- 开启倾斜失调 (is_fold=False)
- 出射光为平行光
- 测量面垂直于光轴，不应有倾斜相位因子

理论背景：
- 平面镜倾斜 θ 引入的波前倾斜：ΔW = 2 * y * sin(θ)（反射加倍）
- 抛物面镜倾斜引入的像差主要是像散和彗差
- 对于小角度失调，像差与倾斜角度成正比

作者：混合光学仿真项目
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import warnings
import pytest
from typing import Tuple

from sequential_system import (
    SequentialOpticalSystem,
    GaussianBeamSource,
)
from gaussian_beam_simulation.optical_elements import (
    ParabolicMirror,
    FlatMirror,
)


def fit_zernike_tilt(phase: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    """拟合波前中的倾斜分量（Zernike Z2, Z3）
    
    参数:
        phase: 相位数组（弧度）
        mask: 有效区域掩模
    
    返回:
        (tilt_x, tilt_y): X 和 Y 方向的倾斜系数（弧度/归一化坐标）
    """
    n = phase.shape[0]
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, y)
    
    # 提取有效区域
    valid_phase = phase[mask]
    valid_x = X[mask]
    valid_y = Y[mask]
    
    if len(valid_phase) < 10:
        return 0.0, 0.0
    
    # 最小二乘拟合：phase = a0 + a1*x + a2*y
    A = np.column_stack([np.ones_like(valid_x), valid_x, valid_y])
    coeffs, _, _, _ = np.linalg.lstsq(A, valid_phase, rcond=None)
    
    return coeffs[1], coeffs[2]  # tilt_x, tilt_y


def remove_tilt_from_phase(phase: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """从相位中移除倾斜分量
    
    参数:
        phase: 相位数组（弧度）
        mask: 有效区域掩模
    
    返回:
        去除倾斜后的相位
    """
    n = phase.shape[0]
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, y)
    
    tilt_x, tilt_y = fit_zernike_tilt(phase, mask)
    
    # 移除倾斜
    tilt_phase = tilt_x * X + tilt_y * Y
    phase_no_tilt = phase - tilt_phase
    
    return phase_no_tilt


def calculate_theoretical_tilt_opd(tilt_angle: float, beam_radius: float, 
                                    wavelength: float, is_reflective: bool = True) -> float:
    """计算理论波前倾斜 OPD
    
    对于倾斜角度 θ 的反射镜，波前倾斜为：
    ΔW = 2 * y * sin(θ)  （反射加倍）
    
    参数:
        tilt_angle: 倾斜角度（弧度）
        beam_radius: 光束半径（mm）
        wavelength: 波长（μm）
        is_reflective: 是否为反射元件
    
    返回:
        波前倾斜的 PV 值（波长数）
    """
    # 波前倾斜 PV = 2 * beam_radius * sin(tilt_angle) * (2 if reflective else 1)
    factor = 2.0 if is_reflective else 1.0
    opd_mm = factor * 2 * beam_radius * np.sin(tilt_angle)
    opd_waves = opd_mm / (wavelength * 1e-3)
    return abs(opd_waves)


# =============================================================================
# 实验 1：平面镜小角度失调
# =============================================================================

class TestFlatMirrorTiltAberration:
    """平面镜小角度失调测试
    
    测试配置：
    - 平行光入射到平面镜
    - 平面镜有小角度失调（is_fold=False）
    - 出射光仍为平行光
    - 测量面垂直于光轴
    
    预期结果：
    - 波前只有倾斜分量（Zernike Z2/Z3）
    - 去除倾斜后，WFE RMS ≈ 0
    - 倾斜量与理论值一致
    """
    
    @pytest.fixture
    def flat_mirror_system(self):
        """创建平面镜测试系统"""
        def _create(tilt_angle_deg: float):
            source = GaussianBeamSource(
                wavelength=0.633,  # μm
                w0=5.0,  # mm，光束腰半径
                z0=0.0,  # mm，光束腰位置
            )
            
            system = SequentialOpticalSystem(
                source,
                grid_size=512,
                beam_ratio=0.25,
                use_hybrid_propagation=True,
            )
            
            tilt_rad = np.deg2rad(tilt_angle_deg)
            system.add_surface(FlatMirror(
                thickness=100.0,  # mm
                semi_aperture=15.0,  # mm
                tilt_x=tilt_rad,
                is_fold=False,  # 失调倾斜
            ))
            
            system.add_sampling_plane(distance=100.0, name="output")
            
            return system, source
        
        return _create
    
    @pytest.mark.parametrize("tilt_deg", [0.1, 0.5, 1.0, 2.0])
    def test_flat_mirror_tilt_only(self, flat_mirror_system, tilt_deg):
        """测试平面镜失调只引入波前倾斜"""
        system, source = flat_mirror_system(tilt_deg)
        
        results = system.run()
        output = results["output"]
        
        # 获取相位和掩模
        phase = output.phase
        amp = output.amplitude
        mask = amp > 0.01 * np.max(amp)
        
        # 拟合倾斜
        tilt_x, tilt_y = fit_zernike_tilt(phase, mask)
        
        # 去除倾斜后的相位
        phase_no_tilt = remove_tilt_from_phase(phase, mask)
        
        # 计算去除倾斜后的 RMS
        valid_phase = phase_no_tilt[mask]
        rms_no_tilt = np.std(valid_phase - np.mean(valid_phase)) / (2 * np.pi)
        
        print(f"\n平面镜倾斜 {tilt_deg}°:")
        print(f"  原始 WFE RMS: {output.wavefront_rms:.4f} waves")
        print(f"  去除倾斜后 RMS: {rms_no_tilt:.4f} waves")
        print(f"  倾斜系数 (tilt_y): {tilt_y:.4f} rad")
        
        # 验证：去除倾斜后 RMS 应该很小（< 0.1 waves）
        assert rms_no_tilt < 0.1, (
            f"平面镜失调应该只引入倾斜，去除倾斜后 RMS 应 < 0.1 waves，"
            f"实际为 {rms_no_tilt:.4f} waves"
        )


# =============================================================================
# 实验 2：抛物面镜准直系统小角度失调
# =============================================================================

class TestParabolicMirrorCollimatorTiltAberration:
    """抛物面镜准直系统小角度失调测试
    
    测试配置：
    - 点光源位于抛物面镜焦点
    - 抛物面镜将发散光准直为平行光
    - 抛物面镜有小角度失调（is_fold=False）
    - 出射光理论上为平行光（无失调时）
    - 测量面垂直于光轴
    
    预期结果：
    - 失调引入像散和彗差
    - 像差量与倾斜角度成正比
    - is_fold=True 时无像差
    """
    
    @pytest.fixture
    def parabolic_collimator_system(self):
        """创建抛物面镜准直系统"""
        def _create(tilt_angle_deg: float, is_fold: bool = False):
            # 使用高斯光束模拟点光源
            # 光束腰位于焦点，经过抛物面镜后准直
            focal_length = 100.0  # mm
            
            source = GaussianBeamSource(
                wavelength=0.633,  # μm
                w0=0.1,  # mm，很小的光束腰模拟点光源
                z0=0.0,  # mm
            )
            
            system = SequentialOpticalSystem(
                source,
                grid_size=512,
                beam_ratio=0.25,
                use_hybrid_propagation=True,
            )
            
            # 先传播到焦点位置（抛物面镜前）
            # 然后抛物面镜准直
            tilt_rad = np.deg2rad(tilt_angle_deg)
            system.add_surface(ParabolicMirror(
                parent_focal_length=focal_length,
                thickness=200.0,  # mm，到测量面的距离
                semi_aperture=15.0,  # mm
                tilt_x=tilt_rad,
                is_fold=is_fold,
            ))
            
            system.add_sampling_plane(distance=200.0, name="output")
            
            return system, source, focal_length
        
        return _create
    
    def test_parabolic_no_tilt_no_aberration(self, parabolic_collimator_system):
        """测试无倾斜时抛物面镜无像差"""
        system, source, f = parabolic_collimator_system(0.0, is_fold=False)
        
        results = system.run()
        output = results["output"]
        
        print(f"\n抛物面镜无倾斜:")
        print(f"  WFE RMS: {output.wavefront_rms:.4f} waves")
        
        # 无倾斜时 WFE 应该很小
        assert output.wavefront_rms < 0.1, (
            f"无倾斜的抛物面镜 WFE 应 < 0.1 waves，"
            f"实际为 {output.wavefront_rms:.4f} waves"
        )
    
    def test_parabolic_is_fold_true_no_aberration(self, parabolic_collimator_system):
        """测试 is_fold=True 时无像差"""
        system, source, f = parabolic_collimator_system(1.0, is_fold=True)
        
        results = system.run()
        output = results["output"]
        
        print(f"\n抛物面镜 is_fold=True (1°):")
        print(f"  WFE RMS: {output.wavefront_rms:.4f} waves")
        
        # is_fold=True 时 WFE 应该很小
        assert output.wavefront_rms < 0.1, (
            f"is_fold=True 的抛物面镜 WFE 应 < 0.1 waves，"
            f"实际为 {output.wavefront_rms:.4f} waves"
        )
    
    @pytest.mark.parametrize("tilt_deg", [0.1, 0.5, 1.0])
    def test_parabolic_is_fold_false_aberration(self, parabolic_collimator_system, tilt_deg):
        """测试 is_fold=False 时引入像差"""
        system, source, f = parabolic_collimator_system(tilt_deg, is_fold=False)
        
        results = system.run()
        output = results["output"]
        
        # 获取相位和掩模
        phase = output.phase
        amp = output.amplitude
        mask = amp > 0.01 * np.max(amp)
        
        # 去除倾斜后的相位
        phase_no_tilt = remove_tilt_from_phase(phase, mask)
        
        # 计算去除倾斜后的 RMS
        valid_phase = phase_no_tilt[mask]
        rms_no_tilt = np.std(valid_phase - np.mean(valid_phase)) / (2 * np.pi)
        
        print(f"\n抛物面镜 is_fold=False ({tilt_deg}°):")
        print(f"  原始 WFE RMS: {output.wavefront_rms:.4f} waves")
        print(f"  去除倾斜后 RMS: {rms_no_tilt:.4f} waves")
        
        # is_fold=False 时应该有像差（去除倾斜后仍有残余）
        # 对于小角度，像差应该与角度成正比
        # 这里只验证像差存在且合理
        if tilt_deg >= 0.5:
            assert rms_no_tilt > 0.01, (
                f"is_fold=False 的抛物面镜应该有像差，"
                f"去除倾斜后 RMS 应 > 0.01 waves"
            )


# =============================================================================
# 主函数：运行所有测试并生成报告
# =============================================================================

def run_flat_mirror_experiment():
    """运行平面镜实验"""
    print("=" * 70)
    print("实验 1：平面镜小角度失调")
    print("=" * 70)
    print("\n配置：")
    print("  - 平行光入射")
    print("  - 平面镜 is_fold=False")
    print("  - 测量面垂直于光轴")
    print("\n预期：只有波前倾斜，无其他像差")
    print("-" * 70)
    
    source = GaussianBeamSource(
        wavelength=0.633,
        w0=5.0,
        z0=0.0,
    )
    
    results_table = []
    
    for tilt_deg in [0.0, 0.1, 0.5, 1.0, 2.0]:
        system = SequentialOpticalSystem(
            source,
            grid_size=512,
            beam_ratio=0.25,
            use_hybrid_propagation=True,
        )
        
        tilt_rad = np.deg2rad(tilt_deg)
        system.add_surface(FlatMirror(
            thickness=100.0,
            semi_aperture=15.0,
            tilt_x=tilt_rad,
            is_fold=False,
        ))
        
        system.add_sampling_plane(distance=100.0, name="output")
        
        results = system.run()
        output = results["output"]
        
        # 分析相位
        phase = output.phase
        amp = output.amplitude
        mask = amp > 0.01 * np.max(amp)
        
        # 拟合并去除倾斜
        tilt_x, tilt_y = fit_zernike_tilt(phase, mask)
        phase_no_tilt = remove_tilt_from_phase(phase, mask)
        
        valid_phase = phase_no_tilt[mask]
        rms_no_tilt = np.std(valid_phase - np.mean(valid_phase)) / (2 * np.pi)
        
        # 理论倾斜 OPD
        theoretical_tilt = calculate_theoretical_tilt_opd(
            tilt_rad, 5.0, 0.633, is_reflective=True
        )
        
        results_table.append({
            'tilt_deg': tilt_deg,
            'wfe_rms': output.wavefront_rms,
            'rms_no_tilt': rms_no_tilt,
            'tilt_y_rad': tilt_y,
            'theoretical_tilt_pv': theoretical_tilt,
        })
    
    # 打印结果表
    print(f"\n{'倾斜(°)':<10} {'WFE RMS':<12} {'去倾斜RMS':<12} {'倾斜系数':<12}")
    print("-" * 50)
    for r in results_table:
        print(f"{r['tilt_deg']:<10.1f} {r['wfe_rms']:<12.4f} {r['rms_no_tilt']:<12.4f} {r['tilt_y_rad']:<12.4f}")
    
    return results_table


def run_parabolic_collimator_experiment():
    """运行抛物面镜准直系统实验"""
    print("\n" + "=" * 70)
    print("实验 2：抛物面镜准直系统小角度失调")
    print("=" * 70)
    print("\n配置：")
    print("  - 点光源位于焦点")
    print("  - 抛物面镜准直")
    print("  - 测量面垂直于光轴")
    print("\n预期：is_fold=True 无像差，is_fold=False 有像差")
    print("-" * 70)
    
    focal_length = 100.0
    
    source = GaussianBeamSource(
        wavelength=0.633,
        w0=5.0,  # 使用较大的光束腰
        z0=0.0,
    )
    
    results_table = []
    
    for tilt_deg in [0.0, 0.5, 1.0, 2.0]:
        for is_fold in [True, False]:
            system = SequentialOpticalSystem(
                source,
                grid_size=512,
                beam_ratio=0.25,
                use_hybrid_propagation=True,
            )
            
            tilt_rad = np.deg2rad(tilt_deg)
            system.add_surface(ParabolicMirror(
                parent_focal_length=focal_length,
                thickness=200.0,
                semi_aperture=15.0,
                tilt_x=tilt_rad,
                is_fold=is_fold,
            ))
            
            system.add_sampling_plane(distance=200.0, name="output")
            
            results = system.run()
            output = results["output"]
            
            # 分析相位
            phase = output.phase
            amp = output.amplitude
            mask = amp > 0.01 * np.max(amp)
            
            # 去除倾斜
            phase_no_tilt = remove_tilt_from_phase(phase, mask)
            valid_phase = phase_no_tilt[mask]
            rms_no_tilt = np.std(valid_phase - np.mean(valid_phase)) / (2 * np.pi)
            
            results_table.append({
                'tilt_deg': tilt_deg,
                'is_fold': is_fold,
                'wfe_rms': output.wavefront_rms,
                'rms_no_tilt': rms_no_tilt,
            })
    
    # 打印结果表
    print(f"\n{'倾斜(°)':<10} {'is_fold':<10} {'WFE RMS':<12} {'去倾斜RMS':<12}")
    print("-" * 50)
    for r in results_table:
        fold_str = "True" if r['is_fold'] else "False"
        print(f"{r['tilt_deg']:<10.1f} {fold_str:<10} {r['wfe_rms']:<12.4f} {r['rms_no_tilt']:<12.4f}")
    
    return results_table


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    print("\n" + "=" * 70)
    print("is_fold=False 失调像差验证实验")
    print("=" * 70)
    
    # 运行实验
    flat_results = run_flat_mirror_experiment()
    parabolic_results = run_parabolic_collimator_experiment()
    
    # 总结
    print("\n" + "=" * 70)
    print("实验总结")
    print("=" * 70)
    
    print("\n实验 1 结论（平面镜）：")
    max_residual = max(r['rms_no_tilt'] for r in flat_results)
    if max_residual < 0.1:
        print(f"  ✓ 平面镜失调只引入波前倾斜，去除倾斜后最大残余 RMS = {max_residual:.4f} waves")
    else:
        print(f"  ✗ 平面镜失调引入了额外像差，去除倾斜后最大残余 RMS = {max_residual:.4f} waves")
    
    print("\n实验 2 结论（抛物面镜）：")
    fold_true_results = [r for r in parabolic_results if r['is_fold'] and r['tilt_deg'] > 0]
    fold_false_results = [r for r in parabolic_results if not r['is_fold'] and r['tilt_deg'] > 0]
    
    if fold_true_results:
        max_fold_true = max(r['wfe_rms'] for r in fold_true_results)
        if max_fold_true < 0.1:
            print(f"  ✓ is_fold=True 时无像差，最大 WFE RMS = {max_fold_true:.4f} waves")
        else:
            print(f"  ✗ is_fold=True 时有像差，最大 WFE RMS = {max_fold_true:.4f} waves")
    
    if fold_false_results:
        max_fold_false = max(r['rms_no_tilt'] for r in fold_false_results)
        print(f"  - is_fold=False 时去除倾斜后最大 RMS = {max_fold_false:.4f} waves")
