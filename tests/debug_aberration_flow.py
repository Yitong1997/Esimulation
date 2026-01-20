"""调试像差计算和应用流程

问题：像差计算正确，但应用到 PROPER 波前时结果不对
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


class DebugSequentialOpticalSystem(SequentialOpticalSystem):
    """调试版本的 SequentialOpticalSystem"""
    
    def _apply_element_hybrid(self, wfo, element):
        """带调试输出的 _apply_element_hybrid"""
        import proper
        
        is_fold = getattr(element, 'is_fold', True)
        has_tilt = (element.tilt_x != 0 or element.tilt_y != 0)
        is_plane_mirror = np.isinf(element.focal_length)
        
        print(f"\n  [DEBUG] _apply_element_hybrid:")
        print(f"    is_fold={is_fold}, has_tilt={has_tilt}, is_plane_mirror={is_plane_mirror}")
        
        # 获取 SurfaceDefinition
        surface_def = element.get_surface_definition()
        
        if surface_def is None:
            print(f"    surface_def is None, using prop_lens")
            if not is_plane_mirror:
                proper.prop_lens(wfo, element.focal_length * 1e-3)
            return
        
        # 调用父类方法的核心逻辑
        # 这里我们手动复制关键部分来添加调试输出
        
        from wavefront_to_rays.element_raytracer import ElementRaytracer, SurfaceDefinition as SD
        from optiland.rays import RealRays
        from scipy.interpolate import griddata
        
        n = proper.prop_get_gridsize(wfo)
        sampling_m = proper.prop_get_sampling(wfo)
        sampling_mm = sampling_m * 1e3
        wavelength_um = wfo.lamda * 1e6
        wavelength_mm = wfo.lamda * 1e3
        
        # 采样范围
        if not is_fold and has_tilt:
            beam_radius_mm = self._source.w(0.0) * 3
            element_aperture = surface_def.semi_aperture if surface_def.semi_aperture else 15.0
            half_size_mm = min(beam_radius_mm, element_aperture)
        else:
            half_size_mm = self._get_sampling_half_size_mm(wfo)
        
        print(f"    half_size_mm={half_size_mm:.2f}")
        
        # 创建采样光线
        ray_x, ray_y = self._create_sampling_rays(half_size_mm)
        n_rays = len(ray_x)
        print(f"    n_rays={n_rays}")
        
        if n_rays == 0:
            print(f"    No rays, returning")
            return
        
        # 创建 surface_def_for_trace
        if is_fold and has_tilt:
            surface_def_for_trace = SD(
                surface_type=surface_def.surface_type,
                radius=surface_def.radius,
                thickness=surface_def.thickness,
                material=surface_def.material,
                semi_aperture=surface_def.semi_aperture,
                conic=surface_def.conic,
                tilt_x=0.0,
                tilt_y=0.0,
            )
        else:
            surface_def_for_trace = surface_def
        
        # 更新高斯光束参数
        if not is_plane_mirror:
            focal_length_m = element.focal_length * 1e-3
            self._update_gaussian_params_only(wfo, focal_length_m)
        
        # 光线追迹
        rays_in = RealRays(
            x=ray_x,
            y=ray_y,
            z=np.zeros(n_rays),
            L=np.zeros(n_rays),
            M=np.zeros(n_rays),
            N=np.ones(n_rays),
            intensity=np.ones(n_rays),
            wavelength=np.full(n_rays, wavelength_um),
        )
        
        raytracer = ElementRaytracer(
            surfaces=[surface_def_for_trace],
            wavelength=wavelength_um,
        )
        
        rays_out = raytracer.trace(rays_in)
        opd_waves = raytracer.get_relative_opd_waves()
        valid_mask = raytracer.get_valid_ray_mask()
        
        print(f"    opd_waves: min={np.min(opd_waves[valid_mask]):.4f}, max={np.max(opd_waves[valid_mask]):.4f}")
        
        # is_fold=False 且有倾斜：使用差分方法
        if not is_fold and has_tilt:
            # 追迹不带倾斜的表面
            surface_no_tilt = SD(
                surface_type=surface_def.surface_type,
                radius=surface_def.radius,
                thickness=surface_def.thickness,
                material=surface_def.material,
                semi_aperture=surface_def.semi_aperture,
                conic=surface_def.conic,
                tilt_x=0.0,
                tilt_y=0.0,
            )
            
            raytracer_ref = ElementRaytracer(
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
            
            rays_out_ref = raytracer_ref.trace(rays_in_ref)
            opd_waves_ref = raytracer_ref.get_relative_opd_waves()
            valid_mask_ref = raytracer_ref.get_valid_ray_mask()
            
            # 差分
            center_idx = n_rays // 2
            opd_waves_aligned = opd_waves - opd_waves[center_idx]
            opd_waves_ref_aligned = opd_waves_ref - opd_waves_ref[center_idx]
            diff_opd_waves = opd_waves_aligned - opd_waves_ref_aligned
            
            valid_mask = valid_mask & valid_mask_ref
            
            print(f"    diff_opd: min={np.min(diff_opd_waves[valid_mask]):.4f}, max={np.max(diff_opd_waves[valid_mask]):.4f}")
            print(f"    diff_opd RMS={np.std(diff_opd_waves[valid_mask]):.4f}")
            
            # 去除倾斜
            valid_x = ray_x[valid_mask]
            valid_y = ray_y[valid_mask]
            valid_diff = diff_opd_waves[valid_mask]
            
            if len(valid_x) > 3:
                max_r = max(np.max(np.abs(valid_x)), np.max(np.abs(valid_y)))
                if max_r > 0:
                    norm_x = valid_x / max_r
                    norm_y = valid_y / max_r
                else:
                    norm_x = valid_x
                    norm_y = valid_y
                
                A = np.column_stack([np.ones_like(norm_x), norm_x, norm_y])
                coeffs, _, _, _ = np.linalg.lstsq(A, valid_diff, rcond=None)
                
                tilt_component = coeffs[0] + coeffs[1] * (ray_x / max_r if max_r > 0 else ray_x) + \
                                 coeffs[2] * (ray_y / max_r if max_r > 0 else ray_y)
                
                aberration_waves = diff_opd_waves - tilt_component
                
                print(f"    tilt coeffs: a0={coeffs[0]:.4f}, a1={coeffs[1]:.4f}, a2={coeffs[2]:.4f}")
                print(f"    aberration after tilt removal: RMS={np.std(aberration_waves[valid_mask]):.4f}")
            else:
                aberration_waves = diff_opd_waves
        else:
            # is_fold=True 或无倾斜
            ray_r_sq = ray_x**2 + ray_y**2
            
            if is_plane_mirror:
                ideal_opd_waves = np.zeros_like(ray_r_sq)
            else:
                focal_length_mm = element.focal_length
                ideal_opd_mm = self._calculate_exact_mirror_opd(ray_r_sq, focal_length_mm)
                ideal_opd_waves = ideal_opd_mm / wavelength_mm
            
            aberration_waves = opd_waves - ideal_opd_waves
        
        # 检查像差
        aberration_waves = np.where(valid_mask, aberration_waves, 0.0)
        valid_aberration = aberration_waves[valid_mask]
        
        if len(valid_aberration) > 0:
            aberration_rms = np.std(valid_aberration)
            aberration_pv = np.max(valid_aberration) - np.min(valid_aberration)
            
            print(f"    Final aberration: RMS={aberration_rms:.4f}, PV={aberration_pv:.4f}")
            
            if aberration_rms < 0.01:
                print(f"    Aberration too small, returning early")
                return
        
        # 应用像差
        aberration_phase = -2 * np.pi * aberration_waves
        
        valid_x = ray_x[valid_mask]
        valid_y = ray_y[valid_mask]
        valid_phase = aberration_phase[valid_mask]
        
        print(f"    Applying phase: min={np.min(valid_phase):.4f}, max={np.max(valid_phase):.4f} rad")
        
        if len(valid_x) > 3:
            coords_mm = np.linspace(-half_size_mm, half_size_mm, n)
            X_mm, Y_mm = np.meshgrid(coords_mm, coords_mm)
            
            points = np.column_stack([valid_x, valid_y])
            phase_grid = griddata(
                points,
                valid_phase,
                (X_mm, Y_mm),
                method='cubic',
                fill_value=0.0,
            )
            
            phase_grid = np.nan_to_num(phase_grid, nan=0.0)
            
            # 检查相位梯度
            grad_x = np.diff(phase_grid, axis=1)
            grad_y = np.diff(phase_grid, axis=0)
            max_grad = max(np.nanmax(np.abs(grad_x)), np.nanmax(np.abs(grad_y)))
            
            print(f"    Phase grid: min={np.nanmin(phase_grid):.4f}, max={np.nanmax(phase_grid):.4f}")
            print(f"    Max phase gradient: {max_grad:.4f} rad/pixel")
            
            # 应用相位
            phase_field = np.exp(1j * phase_grid)
            phase_field_fft = proper.prop_shift_center(phase_field)
            wfo.wfarr = wfo.wfarr * phase_field_fft
            
            print(f"    Phase applied successfully")


def debug_parabolic_system():
    """调试抛物面镜系统"""
    print("=" * 70)
    print("调试抛物面镜 is_fold=False 系统（详细流程）")
    print("=" * 70)
    
    focal_length = 100.0
    
    source = GaussianBeamSource(
        wavelength=0.633,
        w0=5.0,
        z0=0.0,
    )
    
    for tilt_deg in [0.5, 1.0]:
        print(f"\n{'='*70}")
        print(f"倾斜角度: {tilt_deg}°")
        print("=" * 70)
        
        system = DebugSequentialOpticalSystem(
            source,
            grid_size=512,
            beam_ratio=0.25,
            use_hybrid_propagation=True,
            hybrid_num_rays=100,
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
        print(f"\n  最终 WFE RMS: {output.wavefront_rms:.4f} waves")


if __name__ == "__main__":
    debug_parabolic_system()
