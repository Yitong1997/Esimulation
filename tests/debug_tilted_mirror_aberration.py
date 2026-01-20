"""验证倾斜抛物面镜的像差

核心问题：
- 当抛物面镜倾斜时，入射光相对于镜面光轴有角度
- 这相当于"离轴点源"，会引入彗差等像差
- 这是物理上正确的行为

验证方法：
1. 分析倾斜抛物面镜的 OPD 分布
2. 检查是否包含彗差成分
3. 验证这是真实的像差，不是计算错误
"""

import sys
sys.path.insert(0, 'src')

import numpy as np

from wavefront_to_rays.element_raytracer import (
    ElementRaytracer,
    SurfaceDefinition,
)
from optiland.rays import RealRays


def analyze_tilted_mirror_aberration():
    """分析倾斜抛物面镜的像差"""
    
    print("=" * 70)
    print("倾斜抛物面镜像差分析")
    print("=" * 70)
    
    # 参数
    wavelength_um = 0.633
    wavelength_mm = wavelength_um * 1e-3
    tilt_x = np.pi / 4  # 45°
    focal_length = 100.0  # mm
    
    # 创建带倾斜的抛物面镜
    surface_with_tilt = SurfaceDefinition(
        surface_type='mirror',
        radius=2 * focal_length,
        thickness=0.0,
        material='mirror',
        semi_aperture=15.0,
        conic=-1.0,
        tilt_x=tilt_x,
        tilt_y=0.0,
    )
    
    # 创建采样光线
    n_side = 21
    coords = np.linspace(-5, 5, n_side)
    X, Y = np.meshgrid(coords, coords)
    ray_x = X.flatten()
    ray_y = Y.flatten()
    n_rays = len(ray_x)
    
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
    
    # =========================================================================
    # 追迹带倾斜的表面
    # =========================================================================
    print("\n1. 追迹带倾斜的抛物面镜：")
    print("-" * 50)
    
    raytracer = ElementRaytracer(
        surfaces=[surface_with_tilt],
        wavelength=wavelength_um,
    )
    rays_out = raytracer.trace(create_rays())
    opd_waves = raytracer.get_relative_opd_waves()
    valid_mask = raytracer.get_valid_ray_mask()
    
    print(f"   有效光线数: {np.sum(valid_mask)}/{n_rays}")
    print(f"   OPD 范围: {np.min(opd_waves[valid_mask]):.2f} ~ "
          f"{np.max(opd_waves[valid_mask]):.2f} waves")
    
    # =========================================================================
    # Zernike 分解
    # =========================================================================
    print("\n2. Zernike 分解：")
    print("-" * 50)
    
    # 使用有效光线进行 Zernike 分解
    valid_x = ray_x[valid_mask]
    valid_y = ray_y[valid_mask]
    valid_opd = opd_waves[valid_mask]
    
    # 归一化坐标
    r_max = 5.0  # mm
    rho = np.sqrt(valid_x**2 + valid_y**2) / r_max
    theta = np.arctan2(valid_y, valid_x)
    
    # 只保留圆形光瞳内的光线
    pupil_mask = rho <= 1.0
    rho = rho[pupil_mask]
    theta = theta[pupil_mask]
    opd_pupil = valid_opd[pupil_mask]
    
    # 计算 Zernike 多项式（前几项）
    # Z1: Piston
    # Z2: Tilt X (rho * cos(theta))
    # Z3: Tilt Y (rho * sin(theta))
    # Z4: Defocus (2*rho^2 - 1)
    # Z5: Astigmatism 45° (rho^2 * sin(2*theta))
    # Z6: Astigmatism 0° (rho^2 * cos(2*theta))
    # Z7: Coma X (3*rho^3 - 2*rho) * cos(theta)
    # Z8: Coma Y (3*rho^3 - 2*rho) * sin(theta)
    # Z11: Spherical (6*rho^4 - 6*rho^2 + 1)
    
    Z = np.column_stack([
        np.ones_like(rho),                          # Z1: Piston
        rho * np.cos(theta),                        # Z2: Tilt X
        rho * np.sin(theta),                        # Z3: Tilt Y
        2 * rho**2 - 1,                             # Z4: Defocus
        rho**2 * np.sin(2 * theta),                 # Z5: Astigmatism 45°
        rho**2 * np.cos(2 * theta),                 # Z6: Astigmatism 0°
        (3 * rho**3 - 2 * rho) * np.cos(theta),     # Z7: Coma X
        (3 * rho**3 - 2 * rho) * np.sin(theta),     # Z8: Coma Y
        6 * rho**4 - 6 * rho**2 + 1,                # Z11: Spherical
    ])
    
    # 最小二乘拟合
    coeffs, residuals, rank, s = np.linalg.lstsq(Z, opd_pupil, rcond=None)
    
    zernike_names = [
        "Z1 (Piston)",
        "Z2 (Tilt X)",
        "Z3 (Tilt Y)",
        "Z4 (Defocus)",
        "Z5 (Astig 45°)",
        "Z6 (Astig 0°)",
        "Z7 (Coma X)",
        "Z8 (Coma Y)",
        "Z11 (Spherical)",
    ]
    
    print("   Zernike 系数（波长数）：")
    for i, (name, coeff) in enumerate(zip(zernike_names, coeffs)):
        print(f"     {name}: {coeff:.4f}")
    
    # 计算拟合残差
    fitted = Z @ coeffs
    residual = opd_pupil - fitted
    
    print(f"\n   拟合残差 RMS: {np.std(residual):.4f} waves")
    
    # =========================================================================
    # 分析主要像差成分
    # =========================================================================
    print("\n3. 主要像差成分分析：")
    print("-" * 50)
    
    # 去除 Piston、Tilt、Defocus 后的像差
    aberration_coeffs = coeffs.copy()
    aberration_coeffs[0] = 0  # 去除 Piston
    aberration_coeffs[1] = 0  # 去除 Tilt X
    aberration_coeffs[2] = 0  # 去除 Tilt Y
    aberration_coeffs[3] = 0  # 去除 Defocus
    
    aberration = Z @ aberration_coeffs
    
    print(f"   去除 Piston/Tilt/Defocus 后的像差：")
    print(f"     RMS: {np.std(aberration):.4f} waves")
    print(f"     PV: {np.max(aberration) - np.min(aberration):.4f} waves")
    
    # 主要像差成分
    print(f"\n   主要像差成分：")
    print(f"     Astigmatism: {np.sqrt(coeffs[4]**2 + coeffs[5]**2):.4f} waves")
    print(f"     Coma: {np.sqrt(coeffs[6]**2 + coeffs[7]**2):.4f} waves")
    print(f"     Spherical: {abs(coeffs[8]):.4f} waves")
    
    # =========================================================================
    # 结论
    # =========================================================================
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    
    coma = np.sqrt(coeffs[6]**2 + coeffs[7]**2)
    
    if coma > 1.0:
        print(f"✓ 倾斜抛物面镜引入了显著的彗差: {coma:.2f} waves")
        print("  这是物理上正确的行为：")
        print("  - 入射光相对于镜面光轴有 45° 角度")
        print("  - 相当于离轴点源，会引入彗差")
        print("  - 这不是计算错误，而是真实的像差")
    else:
        print(f"✗ 彗差很小: {coma:.4f} waves")
        print("  需要进一步分析")


if __name__ == "__main__":
    analyze_tilted_mirror_aberration()
