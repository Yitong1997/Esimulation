"""
验证 45° 修复是否正确应用
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
from wavefront_to_rays.element_raytracer import (
    ElementRaytracer, 
    SurfaceDefinition,
    _avoid_exact_45_degrees,
)


def test_avoid_exact_45_degrees():
    """测试 _avoid_exact_45_degrees 函数"""
    
    print("=" * 60)
    print("测试 _avoid_exact_45_degrees 函数")
    print("=" * 60)
    
    test_angles = [
        0.0,
        np.pi/4,           # 45°
        np.pi/4 + 1e-15,   # 接近 45°
        np.pi/2,           # 90°
        3*np.pi/4,         # 135°
        np.pi,             # 180°
        5*np.pi/4,         # 225°
        -np.pi/4,          # -45°
        -3*np.pi/4,        # -135°
    ]
    
    for angle in test_angles:
        result = _avoid_exact_45_degrees(angle)
        diff = result - angle
        print(f"  输入: {np.degrees(angle):8.3f}° -> 输出: {np.degrees(result):12.9f}° (差值: {diff:+.2e})")


def test_elementraytracer_with_45deg():
    """测试 ElementRaytracer 是否正确处理 45°"""
    
    print("\n" + "=" * 60)
    print("测试 ElementRaytracer 是否正确处理 45°")
    print("=" * 60)
    
    # 创建 45° 倾斜的平面镜
    mirror = SurfaceDefinition(
        surface_type='mirror',
        radius=np.inf,
        thickness=0.0,
        material='mirror',
        semi_aperture=20.0,
        tilt_x=np.pi/4,  # 精确 45°
    )
    
    print(f"\n输入 tilt_x: {np.degrees(mirror.tilt_x):.10f}°")
    
    # 创建 ElementRaytracer
    raytracer = ElementRaytracer(
        surfaces=[mirror],
        wavelength=0.633,
    )
    
    # 检查 optic 中的表面旋转角度
    surface = raytracer.optic.surface_group.surfaces[1]
    # optiland 的 Surface 对象通过 geometry.cs 访问坐标系
    if hasattr(surface, 'geometry') and hasattr(surface.geometry, 'cs'):
        actual_rx = surface.geometry.cs.rx
        print(f"optic 中的 rx: {np.degrees(actual_rx):.10f}°")
        print(f"差值: {actual_rx - np.pi/4:.2e} rad")
    else:
        print("无法访问表面的坐标系信息")
    
    # 测试光线追迹
    from optiland.rays import RealRays
    
    input_rays = RealRays(
        x=np.array([0.0, 5.0, 10.0]),
        y=np.array([0.0, 0.0, 0.0]),
        z=np.array([0.0, 0.0, 0.0]),
        L=np.array([0.0, 0.0, 0.0]),
        M=np.array([0.0, 0.0, 0.0]),
        N=np.array([1.0, 1.0, 1.0]),
        intensity=np.array([1.0, 1.0, 1.0]),
        wavelength=np.array([0.633, 0.633, 0.633]),
    )
    input_rays.opd = np.array([0.0, 0.0, 0.0])
    
    output_rays = raytracer.trace(input_rays)
    
    print("\n追迹结果:")
    for i in range(3):
        x = np.asarray(output_rays.x)[i]
        y = np.asarray(output_rays.y)[i]
        z = np.asarray(output_rays.z)[i]
        valid = np.isfinite(x)
        print(f"  光线 {i}: pos=({x:.3f}, {y:.3f}, {z:.3f}), valid={valid}")


def test_opd_calculation():
    """测试 OPD 计算"""
    
    print("\n" + "=" * 60)
    print("测试 OPD 计算")
    print("=" * 60)
    
    # 创建简单的凸面镜（无倾斜）
    f = -50.0  # mm
    radius = 2 * f  # -100 mm
    
    mirror = SurfaceDefinition(
        surface_type='mirror',
        radius=radius,
        thickness=0.0,
        material='mirror',
        semi_aperture=20.0,
        conic=-1.0,  # 抛物面
        tilt_x=0.0,  # 无倾斜
    )
    
    raytracer = ElementRaytracer(
        surfaces=[mirror],
        wavelength=10.64,  # μm
    )
    
    from optiland.rays import RealRays
    
    # 创建简单的输入光线
    input_rays = RealRays(
        x=np.array([0.0, 5.0, 10.0]),
        y=np.array([0.0, 0.0, 0.0]),
        z=np.array([0.0, 0.0, 0.0]),
        L=np.array([0.0, 0.0, 0.0]),
        M=np.array([0.0, 0.0, 0.0]),
        N=np.array([1.0, 1.0, 1.0]),
        intensity=np.array([1.0, 1.0, 1.0]),
        wavelength=np.array([10.64, 10.64, 10.64]),
    )
    input_rays.opd = np.array([0.0, 0.0, 0.0])
    
    output_rays = raytracer.trace(input_rays)
    opd_waves = raytracer.get_relative_opd_waves()
    
    print("\n无倾斜凸面镜的 OPD:")
    for i in range(3):
        x_in = np.array([0.0, 5.0, 10.0])[i]
        opd = opd_waves[i]
        print(f"  x={x_in:.1f} mm: OPD={opd:.6f} waves")
    
    # 对于抛物面镜，从焦点发出的球面波应该变成平面波
    # 但这里是平行光入射，所以会有 OPD 变化
    # OPD 应该与 x² 成正比（对于小角度近似）
    
    # 计算理论 OPD
    # 对于抛物面 z = r²/(4f)，反射后的 OPD 变化为 2z（反射加倍）
    # OPD = 2 * r² / (4f) = r² / (2f)
    # 单位：mm
    # 转换为波长数：OPD_waves = OPD_mm / (wavelength_um * 1e-3)
    
    wavelength_mm = 10.64 * 1e-3
    for i, x in enumerate([0.0, 5.0, 10.0]):
        r = x  # 假设 y=0
        z_sag = r**2 / (4 * abs(f))  # 矢高（注意 f 是负的）
        opd_mm = 2 * z_sag  # 反射加倍
        opd_waves_theory = opd_mm / wavelength_mm
        print(f"  x={x:.1f} mm: 理论 OPD={opd_waves_theory:.6f} waves")


if __name__ == "__main__":
    test_avoid_exact_45_degrees()
    test_elementraytracer_with_45deg()
    test_opd_calculation()
