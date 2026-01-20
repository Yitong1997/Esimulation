"""测试 is_fold=False 的行为

验证：
1. 小角度失调（< 1°）时，像差很小
2. 大角度倾斜（45°）时，is_fold=False 正常工作（不产生警告）
3. is_fold=True 时不引入像差

注意：
- is_fold=False 是默认行为，适用于所有情况
- 入射面和出射面垂直于各自的光轴，OPD 不包含整体倾斜
- 对于抛物面镜，无论倾斜角度如何，都不会引入像差（抛物面的定义特性）
- 对于球面镜，倾斜会引入真实的像差（像散、彗差等）
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import warnings
import pytest

from sequential_system import (
    SequentialOpticalSystem,
    GaussianBeamSource,
)
from gaussian_beam_simulation.optical_elements import ParabolicMirror


def test_small_tilt_is_fold_false():
    """测试小角度失调时 is_fold=False 的行为"""
    
    # 创建光源
    source = GaussianBeamSource(
        wavelength=0.633,
        w0=5.0,
        z0=0.0,
    )
    
    # 创建系统
    system = SequentialOpticalSystem(
        source,
        grid_size=256,
        beam_ratio=0.25,
        use_hybrid_propagation=True,
    )
    
    # 添加带小角度失调的抛物面镜
    small_tilt = np.deg2rad(0.5)  # 0.5°
    system.add_surface(ParabolicMirror(
        parent_focal_length=100.0,
        thickness=100.0,
        semi_aperture=15.0,
        tilt_x=small_tilt,
        is_fold=False,  # 失调倾斜
    ))
    
    # 添加采样面
    system.add_sampling_plane(distance=100.0, name="output")
    
    # 运行仿真（不应该有警告）
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        results = system.run()
        
        # 检查是否有警告
        tilt_warnings = [x for x in w if "is_fold=False" in str(x.message)]
        assert len(tilt_warnings) == 0, "小角度失调不应该产生警告"
    
    # 检查 WFE
    output_result = results["output"]
    wfe_rms = output_result.wavefront_rms
    
    print(f"小角度失调 (0.5°) WFE RMS: {wfe_rms:.4f} waves")
    
    # 小角度失调的 WFE 应该很小（< 1 wave）
    assert wfe_rms < 1.0, f"小角度失调的 WFE 应该 < 1 wave，实际为 {wfe_rms:.4f}"


def test_large_tilt_is_fold_false_no_warning():
    """测试大角度倾斜时 is_fold=False 正常工作（不产生警告）
    
    设计原理：
    - is_fold=False 是默认行为，适用于所有情况
    - 入射面和出射面垂直于各自的光轴
    - OPD 不包含整体倾斜（因为参考面垂直于光轴）
    - 对于抛物面镜，无论倾斜角度如何，都不会引入像差
    """
    
    # 创建光源
    source = GaussianBeamSource(
        wavelength=0.633,
        w0=5.0,
        z0=0.0,
    )
    
    # 创建系统
    system = SequentialOpticalSystem(
        source,
        grid_size=256,
        beam_ratio=0.25,
        use_hybrid_propagation=True,
    )
    
    # 添加带大角度倾斜的抛物面镜
    large_tilt = np.deg2rad(45.0)  # 45°
    system.add_surface(ParabolicMirror(
        parent_focal_length=100.0,
        thickness=100.0,
        semi_aperture=15.0,
        tilt_x=large_tilt,
        is_fold=False,  # 默认行为
    ))
    
    # 添加采样面
    system.add_sampling_plane(distance=100.0, name="output")
    
    # 运行仿真（不应该有 is_fold=False 警告）
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        results = system.run()
        
        # 检查是否有 is_fold=False 警告（不应该有）
        tilt_warnings = [x for x in w if "is_fold=False" in str(x.message)]
        assert len(tilt_warnings) == 0, "is_fold=False 是默认行为，不应该产生警告"
    
    # 检查 WFE
    output_result = results["output"]
    wfe_rms = output_result.wavefront_rms
    
    print(f"大角度倾斜 (45°, is_fold=False) WFE RMS: {wfe_rms:.4f} waves")
    
    # 对于抛物面镜，即使大角度倾斜，WFE 也应该很小
    # 因为抛物面对轴上平行光是无像差的
    # 注意：由于数值精度和采样限制，可能有少量残余
    assert wfe_rms < 0.5, f"抛物面镜的 WFE 应该 < 0.5 waves，实际为 {wfe_rms:.4f}"


def test_is_fold_true_no_aberration():
    """测试 is_fold=True 时不引入像差"""
    
    # 创建光源
    source = GaussianBeamSource(
        wavelength=0.633,
        w0=5.0,
        z0=0.0,
    )
    
    # 创建系统
    system = SequentialOpticalSystem(
        source,
        grid_size=256,
        beam_ratio=0.25,
        use_hybrid_propagation=True,
    )
    
    # 添加带 45° 倾斜的抛物面镜（折叠倾斜）
    large_tilt = np.deg2rad(45.0)
    system.add_surface(ParabolicMirror(
        parent_focal_length=100.0,
        thickness=100.0,
        semi_aperture=15.0,
        tilt_x=large_tilt,
        is_fold=True,  # 折叠倾斜
    ))
    
    # 添加采样面
    system.add_sampling_plane(distance=100.0, name="output")
    
    # 运行仿真
    results = system.run()
    
    # 检查 WFE
    output_result = results["output"]
    wfe_rms = output_result.wavefront_rms
    
    print(f"折叠倾斜 (45°, is_fold=True) WFE RMS: {wfe_rms:.4f} waves")
    
    # 折叠倾斜的 WFE 应该很小（< 0.05 waves）
    # 注意：由于数值精度，可能有少量残余像差
    assert wfe_rms < 0.05, f"折叠倾斜的 WFE 应该 < 0.05 waves，实际为 {wfe_rms:.4f}"


if __name__ == "__main__":
    print("=" * 70)
    print("is_fold=False 行为测试")
    print("=" * 70)
    
    print("\n测试 1: 小角度失调")
    print("-" * 50)
    test_small_tilt_is_fold_false()
    print("✓ 通过")
    
    print("\n测试 2: 大角度倾斜（无警告）")
    print("-" * 50)
    test_large_tilt_is_fold_false_no_warning()
    print("✓ 通过")
    
    print("\n测试 3: is_fold=True 无像差")
    print("-" * 50)
    test_is_fold_true_no_aberration()
    print("✓ 通过")
    
    print("\n" + "=" * 70)
    print("所有测试通过！")
    print("=" * 70)
