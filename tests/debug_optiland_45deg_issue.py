"""
深入调试 optiland 在精确 45° 时的问题
"""
import numpy as np
from optiland.optic import Optic
from optiland.rays import RealRays


def test_45_degree_variants():
    """测试 45° 附近的各种角度"""
    
    print("=" * 70)
    print("测试 45° 附近的各种角度")
    print("=" * 70)
    
    # 测试各种接近 45° 的角度
    angles = [
        np.pi/4,                    # 精确 45°
        np.pi/4 + 1e-10,           # 45° + 极小值
        np.pi/4 - 1e-10,           # 45° - 极小值
        np.pi/4 + 1e-8,            # 45° + 小值
        np.pi/4 - 1e-8,            # 45° - 小值
        np.pi/4 + 1e-6,            # 45° + 较大值
        np.pi/4 - 1e-6,            # 45° - 较大值
        0.7853981633974483,        # np.pi/4 的精确值
        0.7853981633974484,        # 稍大
        0.7853981633974482,        # 稍小
    ]
    
    for angle in angles:
        optic = Optic()
        optic.set_aperture(aperture_type='EPD', value=40.0)
        optic.set_field_type(field_type='angle')
        optic.add_field(y=0, x=0)
        optic.add_wavelength(value=0.633, is_primary=True)
        
        optic.add_surface(index=0, radius=np.inf, thickness=np.inf)
        optic.add_surface(
            index=1,
            radius=np.inf,
            thickness=0.0,
            material='mirror',
            is_stop=True,
            rx=angle,
        )
        optic.add_surface(index=2, radius=np.inf, thickness=0.0)
        
        ray = RealRays(
            x=np.array([0.0]),
            y=np.array([0.0]),
            z=np.array([0.0]),
            L=np.array([0.0]),
            M=np.array([0.0]),
            N=np.array([1.0]),
            intensity=np.array([1.0]),
            wavelength=np.array([0.633]),
        )
        ray.opd = np.array([0.0])
        
        optic.surface_group.trace(ray, skip=1)
        
        valid = np.isfinite(ray.x[0])
        diff_from_45 = angle - np.pi/4
        
        print(f"  角度={angle:.16f} (差值={diff_from_45:+.2e}): {'有效' if valid else 'NaN'}")


def test_reflection_direction_issue():
    """测试反射方向的符号问题"""
    
    print("\n" + "=" * 70)
    print("测试反射方向的符号问题")
    print("=" * 70)
    
    # 入射方向
    d = np.array([0.0, 0.0, 1.0])
    
    # 根据 coordinate_conventions.md：
    # 初始法向量沿入射光轴的反方向（指向入射侧）= (0, 0, -1)
    # 绕 X 轴旋转 45° 后
    
    # 旋转矩阵 Rx(45°)
    rx = np.pi / 4
    c, s = np.cos(rx), np.sin(rx)
    Rx = np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])
    
    # 初始法向量（指向入射侧，即 -Z）
    n0 = np.array([0.0, 0.0, -1.0])
    
    # 旋转后的法向量
    n = Rx @ n0
    print(f"初始法向量: {n0}")
    print(f"旋转矩阵 Rx(45°):\n{Rx}")
    print(f"旋转后法向量: {n}")
    
    # 反射公式：r = d - 2(d·n)n
    dot = np.dot(d, n)
    r = d - 2 * dot * n
    
    print(f"\n入射方向 d = {d}")
    print(f"d·n = {dot:.6f}")
    print(f"反射方向 r = {r}")
    
    # 另一种约定：法向量指向 +Z 侧
    print("\n--- 另一种约定（法向量指向 +Z）---")
    n0_alt = np.array([0.0, 0.0, 1.0])
    n_alt = Rx @ n0_alt
    print(f"初始法向量: {n0_alt}")
    print(f"旋转后法向量: {n_alt}")
    
    dot_alt = np.dot(d, n_alt)
    r_alt = d - 2 * dot_alt * n_alt
    print(f"d·n = {dot_alt:.6f}")
    print(f"反射方向 r = {r_alt}")


def test_workaround_with_small_offset():
    """测试使用小偏移量作为临时解决方案"""
    
    print("\n" + "=" * 70)
    print("测试使用小偏移量作为临时解决方案")
    print("=" * 70)
    
    # 使用 45° + 1e-10 代替精确的 45°
    angle = np.pi/4 + 1e-10
    
    optic = Optic()
    optic.set_aperture(aperture_type='EPD', value=40.0)
    optic.set_field_type(field_type='angle')
    optic.add_field(y=0, x=0)
    optic.add_wavelength(value=0.633, is_primary=True)
    
    optic.add_surface(index=0, radius=np.inf, thickness=np.inf)
    optic.add_surface(
        index=1,
        radius=np.inf,
        thickness=0.0,
        material='mirror',
        is_stop=True,
        rx=angle,
    )
    optic.add_surface(index=2, radius=np.inf, thickness=0.0)
    
    # 测试多条光线
    n_rays = 5
    x_vals = np.linspace(-10, 10, n_rays)
    
    ray = RealRays(
        x=x_vals,
        y=np.zeros(n_rays),
        z=np.zeros(n_rays),
        L=np.zeros(n_rays),
        M=np.zeros(n_rays),
        N=np.ones(n_rays),
        intensity=np.ones(n_rays),
        wavelength=np.full(n_rays, 0.633),
    )
    ray.opd = np.zeros(n_rays)
    
    optic.surface_group.trace(ray, skip=1)
    
    print(f"\n使用角度 {np.degrees(angle):.10f}° 的追迹结果:")
    for i in range(n_rays):
        print(f"  光线 {i}: x={x_vals[i]:6.1f} -> "
              f"位置=({ray.x[i]:8.3f}, {ray.y[i]:8.3f}, {ray.z[i]:8.3f}), "
              f"方向=({ray.L[i]:8.5f}, {ray.M[i]:8.5f}, {ray.N[i]:8.5f})")


def test_curved_mirror_at_45():
    """测试 45° 倾斜的曲面镜"""
    
    print("\n" + "=" * 70)
    print("测试 45° 倾斜的曲面镜（OAP）")
    print("=" * 70)
    
    # 使用小偏移量避免精确 45° 的问题
    angle = np.pi/4 + 1e-10
    
    # 创建 OAP（抛物面镜）
    # 焦距 50mm，曲率半径 100mm
    radius = -100.0  # 凸面镜（发散）
    
    optic = Optic()
    optic.set_aperture(aperture_type='EPD', value=40.0)
    optic.set_field_type(field_type='angle')
    optic.add_field(y=0, x=0)
    optic.add_wavelength(value=10.64, is_primary=True)
    
    optic.add_surface(index=0, radius=np.inf, thickness=np.inf)
    optic.add_surface(
        index=1,
        radius=radius,
        thickness=0.0,
        material='mirror',
        is_stop=True,
        rx=angle,
        conic=-1.0,  # 抛物面
    )
    optic.add_surface(index=2, radius=np.inf, thickness=0.0)
    
    # 测试中心光线
    ray = RealRays(
        x=np.array([0.0]),
        y=np.array([0.0]),
        z=np.array([0.0]),
        L=np.array([0.0]),
        M=np.array([0.0]),
        N=np.array([1.0]),
        intensity=np.array([1.0]),
        wavelength=np.array([10.64]),
    )
    ray.opd = np.array([0.0])
    
    optic.surface_group.trace(ray, skip=1)
    
    valid = np.isfinite(ray.x[0])
    print(f"\n45° 倾斜抛物面镜追迹结果: {'有效' if valid else 'NaN'}")
    if valid:
        print(f"  位置: ({ray.x[0]:.6f}, {ray.y[0]:.6f}, {ray.z[0]:.6f})")
        print(f"  方向: ({ray.L[0]:.6f}, {ray.M[0]:.6f}, {ray.N[0]:.6f})")
        print(f"  OPD: {ray.opd[0]:.6f}")


if __name__ == "__main__":
    test_45_degree_variants()
    test_reflection_direction_issue()
    test_workaround_with_small_offset()
    test_curved_mirror_at_45()
