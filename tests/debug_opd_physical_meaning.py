"""理解 OPD 的物理意义

核心问题：
- 对于倾斜的抛物面镜，ElementRaytracer 计算的 OPD 是什么？
- 这个 OPD 应该如何与 PROPER 的相位结合？

物理分析：
1. OPD（光程差）是光线从入射面到出射面的几何光程
2. 对于反射镜，OPD = 入射路径 + 反射路径
3. 相对 OPD = 光线 OPD - 主光线 OPD

关键问题：
- 当抛物面镜倾斜时，入射光相对于镜面光轴有角度
- 这相当于"离轴点源"，会引入彗差等像差
- 但在我们的应用中，我们只关心"波前变换"，不关心像差

正确的理解：
- is_fold=True：倾斜用于折叠光路，PROPER 在展开的光路上传播
  - 不应该追迹倾斜的表面，而是追迹不倾斜的表面
  - 这样 OPD 只包含聚焦效果
  
- is_fold=False：倾斜表示元件失调
  - 应该追迹倾斜的表面
  - OPD 包含倾斜引入的像差
  - 但这个像差应该是"真实的"，不是计算错误

验证方法：
- 对于 is_fold=True，使用不倾斜的表面追迹，OPD 应该只有聚焦效果
- 对于 is_fold=False，使用倾斜的表面追迹，OPD 应该包含真实的像差
"""

import sys
sys.path.insert(0, 'src')

import numpy as np

from wavefront_to_rays.element_raytracer import (
    ElementRaytracer,
    SurfaceDefinition,
)
from optiland.rays import RealRays


def test_opd_physical_meaning():
    """测试 OPD 的物理意义"""
    
    print("=" * 70)
    print("OPD 物理意义验证")
    print("=" * 70)
    
    # 参数
    wavelength_um = 0.633
    wavelength_mm = wavelength_um * 1e-3
    focal_length = 100.0  # mm
    
    # 创建不带倾斜的抛物面镜
    surface_no_tilt = SurfaceDefinition(
        surface_type='mirror',
        radius=2 * focal_length,
        thickness=0.0,
        material='mirror',
        semi_aperture=10.0,
        conic=-1.0,
        tilt_x=0.0,
        tilt_y=0.0,
    )
    
    # 创建采样光线
    n_side = 11
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
    # 追迹不带倾斜的表面
    # =========================================================================
    print("\n1. 不带倾斜的抛物面镜：")
    print("-" * 50)
    
    raytracer = ElementRaytracer(
        surfaces=[surface_no_tilt],
        wavelength=wavelength_um,
    )
    rays_out = raytracer.trace(create_rays())
    opd_waves = raytracer.get_relative_opd_waves()
    valid_mask = raytracer.get_valid_ray_mask()
    
    print(f"   有效光线数: {np.sum(valid_mask)}/{n_rays}")
    print(f"   OPD 范围: {np.min(opd_waves[valid_mask]):.2f} ~ "
          f"{np.max(opd_waves[valid_mask]):.2f} waves")
    
    # =========================================================================
    # 计算理想 OPD（使用精确公式）
    # =========================================================================
    print("\n2. 理想 OPD 计算：")
    print("-" * 50)
    
    r_sq = ray_x**2 + ray_y**2
    
    def calculate_exact_mirror_opd(r_sq, f):
        """计算理想抛物面镜的精确 OPD
        
        对于抛物面镜，从入射面到出射面的 OPD：
        - 入射路径：从入射面到表面的距离 = sag
        - 反射路径：从表面到出射面的距离
        
        对于抛物面 z = r²/(4f)，表面法向量为：
        n = (-∂z/∂x, -∂z/∂y, 1) / |n|
        n = (-x/(2f), -y/(2f), 1) / sqrt(1 + r²/(4f²))
        
        入射光沿 +z 方向，反射后方向为：
        r = d - 2(d·n)n
        
        对于轴上平行光入射抛物面镜，所有光线汇聚到焦点 (0, 0, f)
        """
        # 表面矢高
        sag = r_sq / (4 * f)
        
        # 表面法向量的 z 分量
        n_mag_sq = 1 + r_sq / (4 * f**2)
        nz = 1 / np.sqrt(n_mag_sq)
        
        # 入射光方向 d = (0, 0, 1)
        # d·n = nz
        # 反射方向 rz = 1 - 2*nz*nz = 1 - 2/n_mag_sq
        rz = 1 - 2 / n_mag_sq
        
        # 入射路径（从 z=0 到表面）
        incident_path = sag
        
        # 反射路径（从表面到 z=0 平面）
        # 反射光线从 (x, y, sag) 出发，方向为 (rx, ry, rz)
        # 到达 z=0 平面时：sag + rz * t = 0 => t = -sag / rz
        # 反射路径长度 = |t| * sqrt(rx² + ry² + rz²) = |t| * 1 = -sag / rz
        reflected_path = -sag / rz
        
        # 总 OPD
        total_opd = incident_path + reflected_path
        
        return total_opd
    
    ideal_opd_mm = calculate_exact_mirror_opd(r_sq, focal_length)
    ideal_opd_waves = ideal_opd_mm / wavelength_mm
    
    # 计算相对理想 OPD
    center_idx = n_rays // 2
    ideal_opd_waves_relative = ideal_opd_waves - ideal_opd_waves[center_idx]
    
    print(f"   理想 OPD 范围: {np.min(ideal_opd_waves_relative):.2f} ~ "
          f"{np.max(ideal_opd_waves_relative):.2f} waves")
    
    # =========================================================================
    # 比较实际 OPD 和理想 OPD
    # =========================================================================
    print("\n3. 像差分析：")
    print("-" * 50)
    
    aberration = opd_waves - ideal_opd_waves_relative
    valid_aberration = aberration[valid_mask]
    
    print(f"   像差范围: {np.min(valid_aberration):.6f} ~ "
          f"{np.max(valid_aberration):.6f} waves")
    print(f"   像差 RMS: {np.std(valid_aberration):.6f} waves")
    print(f"   像差 PV: {np.max(valid_aberration) - np.min(valid_aberration):.6f} waves")
    
    # =========================================================================
    # 结论
    # =========================================================================
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    
    if np.std(valid_aberration) < 0.01:
        print("✓ 不带倾斜的抛物面镜像差 < 0.01 waves")
        print("  ElementRaytracer 的 OPD 计算与理论公式一致")
    else:
        print(f"✗ 不带倾斜的抛物面镜像差 = {np.std(valid_aberration):.4f} waves")
        print("  需要检查 ElementRaytracer 的 OPD 计算")


if __name__ == "__main__":
    test_opd_physical_meaning()
