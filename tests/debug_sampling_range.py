"""调试采样范围"""

import sys
sys.path.insert(0, 'src')

import numpy as np

from sequential_system import (
    SequentialOpticalSystem,
    GaussianBeamSource,
)
from gaussian_beam_simulation.optical_elements import (
    ParabolicMirror,
)


def test_sampling_range():
    """测试采样范围"""
    source = GaussianBeamSource(
        wavelength=0.633,
        w0=5.0,
        z0=0.0,
    )
    
    print(f"光束腰半径 w0 = {source.w(0.0)} mm")
    print(f"2 * w0 = {source.w(0.0) * 2} mm")
    print(f"3 * w0 = {source.w(0.0) * 3} mm")
    
    system = SequentialOpticalSystem(
        source,
        grid_size=512,
        beam_ratio=0.25,
        use_hybrid_propagation=True,
        hybrid_num_rays=100,
    )
    
    tilt_rad = np.deg2rad(1.0)
    element = ParabolicMirror(
        parent_focal_length=100.0,
        thickness=200.0,
        semi_aperture=15.0,
        tilt_x=tilt_rad,
        is_fold=False,
    )
    
    system.add_surface(element)
    
    # 检查 surface_def
    surface_def = element.get_surface_definition()
    print(f"\nsurface_def.semi_aperture = {surface_def.semi_aperture}")
    
    # 计算采样范围
    is_fold = getattr(element, 'is_fold', True)
    has_tilt = (element.tilt_x != 0 or element.tilt_y != 0)
    
    print(f"\nis_fold = {is_fold}")
    print(f"has_tilt = {has_tilt}")
    
    if not is_fold and has_tilt:
        beam_radius_mm = source.w(0.0) * 2
        element_aperture = surface_def.semi_aperture if surface_def.semi_aperture else 15.0
        half_size_mm = min(beam_radius_mm, element_aperture)
        print(f"\nbeam_radius_mm = {beam_radius_mm}")
        print(f"element_aperture = {element_aperture}")
        print(f"half_size_mm = {half_size_mm}")


if __name__ == "__main__":
    test_sampling_range()
