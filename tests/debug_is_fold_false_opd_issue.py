"""调试 is_fold=False 的 OPD 计算问题

问题：平面镜 is_fold=False 时，去除倾斜后仍有 ~0.29 waves 的残余
预期：平面镜失调应该只引入波前倾斜，无其他像差

分析步骤：
1. 检查 ElementRaytracer 对平面镜的 OPD 计算
2. 检查带倾斜和不带倾斜的 OPD 差异
3. 验证差分方法是否正确
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition
from optiland.rays import RealRays


def analyze_flat_mirror_opd():
    """分析平面镜的 OPD 计算"""
    print("=" * 70)
    print("平面镜 OPD 分析")
    print("=" * 70)
    
    wavelength_um = 0.633
    
    # 创建光线网格
    n_side = 11
    half_size = 5.0  # mm
    x = np.linspace(-half_size, half_size, n_side)
    y = np.linspace(-half_size, half_size, n_side)
    X, Y = np.meshgrid(x, y)
    ray_x = X.flatten()
    ray_y = Y.flatten()
    n_rays = len(ray_x)
    
    # 测试不同倾斜角度
    for tilt_deg in [0.0, 0.5, 1.0, 2.0]:
        tilt_rad = np.deg2rad(tilt_deg)
        
        print(f"\n倾斜角度: {tilt_deg}°")
        print("-" * 50)
        
        # 带倾斜的平面镜
        surface_tilted = SurfaceDefinition(
            surface_type='mirror',
            radius=np.inf,
            thickness=0.0,
            material='mirror',
            semi_aperture=15.0,
            conic=0.0,
            tilt_x=tilt_rad,
            tilt_y=0.0,
        )
        
        raytracer_tilted = ElementRaytracer(
            surfaces=[surface_tilted],
            wavelength=wavelength_um,
        )
        
        rays_in = RealRays(
            x=ray_x.copy(),
            y=ray_y.copy(),
            z=np.zeros(n_rays),
            L=np.zeros(n_rays),
            M=np.zeros(n_rays),
            N=np.ones(n_rays),
            intensity=np.ones(n_rays),
            wavelength=np.full(n_rays, wavelength_um),
        )
        
        rays_out_tilted = raytracer_tilted.trace(rays_in)
        opd_tilted = raytracer_tilted.get_relative_opd_waves()
        
        print(f"  带倾斜 OPD: min={np.min(opd_tilted):.4f}, max={np.max(opd_tilted):.4f}, "
              f"PV={np.max(opd_tilted)-np.min(opd_tilted):.4f} waves")

        # 不带倾斜的平面镜
        surface_no_tilt = SurfaceDefinition(
            surface_type='mirror',
            radius=np.inf,
            thickness=0.0,
            material='mirror',
            semi_aperture=15.0,
            conic=0.0,
            tilt_x=0.0,
            tilt_y=0.0,
        )
        
        raytracer_no_tilt = ElementRaytracer(
            surfaces=[surface_no_tilt],
            wavelength=wavelength_um,
        )
        
        rays_in_ref = RealRays(
            x=ray_x.copy(),
            y=ray_y.copy(),
            z=np.zeros(n_rays),
            L=np.zeros(n_rays),
            M=np.zeros(n_rays),
            N=np.ones(n_rays),
            intensity=np.ones(n_rays),
            wavelength=np.full(n_rays, wavelength_um),
        )
        
        rays_out_no_tilt = raytracer_no_tilt.trace(rays_in_ref)
        opd_no_tilt = raytracer_no_tilt.get_relative_opd_waves()
        
        print(f"  不带倾斜 OPD: min={np.min(opd_no_tilt):.4f}, max={np.max(opd_no_tilt):.4f}, "
              f"PV={np.max(opd_no_tilt)-np.min(opd_no_tilt):.4f} waves")
        
        # 差分
        center_idx = n_rays // 2
        opd_tilted_aligned = opd_tilted - opd_tilted[center_idx]
        opd_no_tilt_aligned = opd_no_tilt - opd_no_tilt[center_idx]
        diff = opd_tilted_aligned - opd_no_tilt_aligned
        
        print(f"  差分 OPD: min={np.min(diff):.4f}, max={np.max(diff):.4f}, "
              f"PV={np.max(diff)-np.min(diff):.4f} waves")
        print(f"  差分 RMS: {np.std(diff):.4f} waves")
        
        # 理论倾斜 OPD
        # 对于反射镜，倾斜 θ 引入的 OPD = 2 * y * sin(θ)
        theoretical_opd = 2 * ray_y * np.sin(tilt_rad)
        theoretical_opd_waves = theoretical_opd / (wavelength_um * 1e-3)
        theoretical_opd_waves_aligned = theoretical_opd_waves - theoretical_opd_waves[center_idx]
        
        print(f"  理论倾斜 OPD PV: {np.max(theoretical_opd_waves_aligned)-np.min(theoretical_opd_waves_aligned):.4f} waves")
        
        # 检查出射光线方向
        L_out = np.asarray(rays_out_tilted.L)
        M_out = np.asarray(rays_out_tilted.M)
        N_out = np.asarray(rays_out_tilted.N)
        print(f"  出射光线方向 (中心): L={L_out[center_idx]:.4f}, M={M_out[center_idx]:.4f}, N={N_out[center_idx]:.4f}")


def analyze_raytracer_opd_calculation():
    """分析 ElementRaytracer 的 OPD 计算方法"""
    print("\n" + "=" * 70)
    print("ElementRaytracer OPD 计算分析")
    print("=" * 70)
    
    # 查看 ElementRaytracer 的 get_relative_opd_waves 方法
    from wavefront_to_rays.element_raytracer import ElementRaytracer
    import inspect
    
    print("\nget_relative_opd_waves 方法源码：")
    print("-" * 50)
    try:
        source = inspect.getsource(ElementRaytracer.get_relative_opd_waves)
        print(source[:2000])  # 只打印前 2000 字符
    except Exception as e:
        print(f"无法获取源码: {e}")


if __name__ == "__main__":
    analyze_flat_mirror_opd()
    analyze_raytracer_opd_calculation()
