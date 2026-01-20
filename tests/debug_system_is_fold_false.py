"""调试 SequentialOpticalSystem 中 is_fold=False 的像差计算

问题：抛物面镜 is_fold=False 时，所有倾斜角度都显示 ~0.29 waves 残余
预期：像差应该随倾斜角度变化

分析步骤：
1. 检查 _apply_element_hybrid 方法的执行流程
2. 验证像差是否被正确计算和应用
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import warnings

from sequential_system import (
    SequentialOpticalSystem,
    GaussianBeamSource,
)
from gaussian_beam_simulation.optical_elements import (
    ParabolicMirror,
    FlatMirror,
)


def debug_parabolic_system():
    """调试抛物面镜系统"""
    print("=" * 70)
    print("调试抛物面镜 is_fold=False 系统")
    print("=" * 70)
    
    focal_length = 100.0  # mm
    
    source = GaussianBeamSource(
        wavelength=0.633,
        w0=5.0,
        z0=0.0,
    )
    
    for tilt_deg in [0.0, 0.5, 1.0, 2.0]:
        print(f"\n倾斜角度: {tilt_deg}°")
        print("-" * 50)
        
        # 创建系统
        system = SequentialOpticalSystem(
            source,
            grid_size=512,
            beam_ratio=0.25,
            use_hybrid_propagation=True,
            hybrid_num_rays=100,  # 10x10 网格
        )
        
        tilt_rad = np.deg2rad(tilt_deg)
        system.add_surface(ParabolicMirror(
            parent_focal_length=focal_length,
            thickness=200.0,
            semi_aperture=15.0,
            tilt_x=tilt_rad,
            is_fold=False,
        ))
        
        system.add_sampling_plane(distance=200.0, name="output")
        
        # 运行仿真
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = system.run()
            
            # 打印警告
            for warning in w:
                print(f"  警告: {warning.message}")
        
        output = results["output"]
        
        print(f"  WFE RMS: {output.wavefront_rms:.4f} waves")
        
        # 分析相位
        phase = output.phase
        amp = output.amplitude
        mask = amp > 0.01 * np.max(amp)
        
        # 去除倾斜
        n = phase.shape[0]
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        X, Y = np.meshgrid(x, y)
        
        valid_phase = phase[mask]
        valid_x = X[mask]
        valid_y = Y[mask]
        
        if len(valid_phase) > 10:
            A = np.column_stack([np.ones_like(valid_x), valid_x, valid_y])
            coeffs, _, _, _ = np.linalg.lstsq(A, valid_phase, rcond=None)
            
            tilt_phase = coeffs[0] + coeffs[1] * X + coeffs[2] * Y
            phase_no_tilt = phase - tilt_phase
            
            valid_no_tilt = phase_no_tilt[mask]
            rms_no_tilt = np.std(valid_no_tilt - np.mean(valid_no_tilt)) / (2 * np.pi)
            
            print(f"  去除倾斜后 RMS: {rms_no_tilt:.4f} waves")
            print(f"  倾斜系数: a0={coeffs[0]:.4f}, a1={coeffs[1]:.4f}, a2={coeffs[2]:.4f}")


def debug_with_more_rays():
    """使用更多光线调试"""
    print("\n" + "=" * 70)
    print("使用更多光线调试")
    print("=" * 70)
    
    focal_length = 100.0
    
    source = GaussianBeamSource(
        wavelength=0.633,
        w0=5.0,
        z0=0.0,
    )
    
    tilt_deg = 1.0
    
    for num_rays in [100, 400, 900]:
        print(f"\n光线数量: {num_rays}")
        print("-" * 50)
        
        system = SequentialOpticalSystem(
            source,
            grid_size=512,
            beam_ratio=0.25,
            use_hybrid_propagation=True,
            hybrid_num_rays=num_rays,
        )
        
        tilt_rad = np.deg2rad(tilt_deg)
        system.add_surface(ParabolicMirror(
            parent_focal_length=focal_length,
            thickness=200.0,
            semi_aperture=15.0,
            tilt_x=tilt_rad,
            is_fold=False,
        ))
        
        system.add_sampling_plane(distance=200.0, name="output")
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = system.run()
        
        output = results["output"]
        print(f"  WFE RMS: {output.wavefront_rms:.4f} waves")


if __name__ == "__main__":
    debug_parabolic_system()
    debug_with_more_rays()
