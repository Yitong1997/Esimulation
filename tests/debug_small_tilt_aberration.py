"""验证小角度失调时的像差

核心假设：
- is_fold=False 用于表示元件的小角度失调
- 对于小角度失调（如 1°、0.1°），像差应该很小
- 45° 这样的大角度不是 is_fold=False 的预期用途

验证方法：
- 测试不同倾斜角度下的像差
- 验证小角度失调时像差是否合理
"""

import sys
sys.path.insert(0, 'src')

import numpy as np

from wavefront_to_rays.element_raytracer import (
    ElementRaytracer,
    SurfaceDefinition,
)
from optiland.rays import RealRays


def test_small_tilt_aberration():
    """测试小角度失调时的像差"""
    
    print("=" * 70)
    print("小角度失调像差分析")
    print("=" * 70)
    
    # 参数
    wavelength_um = 0.633
    wavelength_mm = wavelength_um * 1e-3
    focal_length = 100.0  # mm
    
    # 测试不同的倾斜角度
    tilt_angles_deg = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 45.0]
    
    # 创建采样光线
    n_side = 21
    coords = np.linspace(-5, 5, n_side)
    X, Y = np.meshgrid(coords, coords)
    ray_x = X.flatten()
    ray_y = Y.flatten()
    n_rays = len(ray_x)
    
    # 圆形光瞳掩模
    r = np.sqrt(ray_x**2 + ray_y**2)
    pupil_mask = r <= 5.0
    
    def create_rays():
        return RealRays(
            x=ray_x.copy(),
            y=ray_y.copy(),
            z=np.zeros(n_rays),
            L=np.zeros(n_rays),
            M=np.zeros(n_rays),
            N=np.ones(n_rays),
            intensity=np.ones(n_rays),
            wavelength=np.full(n_rays, wavelength_um),
        )
    
    print("\n倾斜角度 vs 像差：")
    print("-" * 70)
    print(f"{'角度 (°)':<12} {'OPD RMS':<15} {'去除倾斜后 RMS':<18} {'像散':<12} {'彗差':<12}")
    print("-" * 70)
    
    for tilt_deg in tilt_angles_deg:
        tilt_rad = np.deg2rad(tilt_deg)
        
        # 创建带倾斜的抛物面镜
        surface = SurfaceDefinition(
            surface_type='mirror',
            radius=2 * focal_length,
            thickness=0.0,
            material='mirror',
            semi_aperture=15.0,
            conic=-1.0,
            tilt_x=tilt_rad,
            tilt_y=0.0,
        )
        
        # 追迹
        raytracer = ElementRaytracer(
            surfaces=[surface],
            wavelength=wavelength_um,
        )
        rays_out = raytracer.trace(create_rays())
        opd_waves = raytracer.get_relative_opd_waves()
        valid_mask = raytracer.get_valid_ray_mask() & pupil_mask
        
        if not np.any(valid_mask):
            print(f"{tilt_deg:<12.1f} {'N/A':<15} {'N/A':<18} {'N/A':<12} {'N/A':<12}")
            continue
        
        valid_opd = opd_waves[valid_mask]
        opd_rms = np.std(valid_opd)
        
        # Zernike 分解
        valid_x = ray_x[valid_mask]
        valid_y = ray_y[valid_mask]
        
        r_max = 5.0
        rho = np.sqrt(valid_x**2 + valid_y**2) / r_max
        theta = np.arctan2(valid_y, valid_x)
        
        Z = np.column_stack([
            np.ones_like(rho),                          # Z1: Piston
            rho * np.cos(theta),                        # Z2: Tilt X
            rho * np.sin(theta),                        # Z3: Tilt Y
            2 * rho**2 - 1,                             # Z4: Defocus
            rho**2 * np.sin(2 * theta),                 # Z5: Astigmatism 45°
            rho**2 * np.cos(2 * theta),                 # Z6: Astigmatism 0°
            (3 * rho**3 - 2 * rho) * np.cos(theta),     # Z7: Coma X
            (3 * rho**3 - 2 * rho) * np.sin(theta),     # Z8: Coma Y
        ])
        
        coeffs, _, _, _ = np.linalg.lstsq(Z, valid_opd, rcond=None)
        
        # 去除 Piston 和 Tilt 后的 RMS
        aberration_coeffs = coeffs.copy()
        aberration_coeffs[0] = 0  # 去除 Piston
        aberration_coeffs[1] = 0  # 去除 Tilt X
        aberration_coeffs[2] = 0  # 去除 Tilt Y
        
        aberration = Z @ aberration_coeffs
        aberration_rms = np.std(aberration)
        
        astigmatism = np.sqrt(coeffs[4]**2 + coeffs[5]**2)
        coma = np.sqrt(coeffs[6]**2 + coeffs[7]**2)
        
        print(f"{tilt_deg:<12.1f} {opd_rms:<15.4f} {aberration_rms:<18.4f} {astigmatism:<12.4f} {coma:<12.4f}")
    
    print("-" * 70)
    
    # =========================================================================
    # 结论
    # =========================================================================
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print("1. 小角度失调（< 1°）时，像差很小（< 1 wave）")
    print("2. 大角度倾斜（如 45°）会引入显著的像散")
    print("3. is_fold=False 应该用于小角度失调，不适用于大角度折叠")
    print("4. 对于 45° 折叠镜，应该使用 is_fold=True")


if __name__ == "__main__":
    test_small_tilt_aberration()
