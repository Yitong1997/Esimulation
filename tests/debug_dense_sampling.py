"""调试密集采样的效果

验证密集采样是否被正确应用
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
)


def test_dense_sampling():
    """测试密集采样"""
    print("=" * 70)
    print("测试密集采样")
    print("=" * 70)
    
    focal_length = 100.0
    tilt_deg = 1.0
    tilt_rad = np.deg2rad(tilt_deg)
    
    source = GaussianBeamSource(
        wavelength=0.633,
        w0=5.0,
        z0=0.0,
    )
    
    # 计算预期的采样参数
    beam_radius_mm = source.w(0.0) * 1.5
    print(f"\n光束参数:")
    print(f"  w0 = {source.w0} mm")
    print(f"  采样范围 = ±{beam_radius_mm:.2f} mm")
    
    for grid_size in [256, 512, 1024]:
        print(f"\n网格大小: {grid_size}")
        print("-" * 50)
        
        n_rays_1d_dense = min(grid_size // 4, 128)
        print(f"  密集采样点数: {n_rays_1d_dense}x{n_rays_1d_dense} = {n_rays_1d_dense**2}")
        
        # 计算采样间隔
        sampling_interval = 2 * beam_radius_mm / (n_rays_1d_dense - 1)
        print(f"  采样间隔: {sampling_interval:.4f} mm")
        
        system = SequentialOpticalSystem(
            source,
            grid_size=grid_size,
            beam_ratio=0.25,
            use_hybrid_propagation=True,
        )
        
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
            
            phase_warnings = [x for x in w if "相位采样不足" in str(x.message)]
        
        output = results["output"]
        
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
        else:
            rms_no_tilt = 0.0
        
        warning_str = "⚠️" if phase_warnings else "✓"
        print(f"  WFE RMS = {output.wavefront_rms:.4f}, 去倾斜 RMS = {rms_no_tilt:.4f} waves  {warning_str}")


if __name__ == "__main__":
    test_dense_sampling()
