"""
详细调试 OAP 光线追迹问题

核心问题：
1. optiland 对 45° 倾斜表面的中心光线返回 NaN
2. OPD 计算可能有问题
3. 坐标系变换可能有问题
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
from optiland.optic import Optic
from optiland.rays import RealRays


def test_optiland_tilted_mirror_detailed():
    """详细测试 optiland 对倾斜镜的处理"""
    
    print("=" * 70)
    print("测试 optiland 对 45° 倾斜平面镜的处理")
    print("=" * 70)
    
    # 创建简单的 45° 倾斜平面镜系统
    optic = Optic()
    optic.set_aperture(aperture_type='EPD', value=40.0)
    optic.set_field_type(field_type='angle')
    optic.add_field(y=0, x=0)
    optic.add_wavelength(value=0.633, is_primary=True)
    
    # 物面
    optic.add_surface(index=0, radius=np.inf, thickness=np.inf)
    
    # 45° 倾斜平面镜
    optic.add_surface(
        index=1,
        radius=np.inf,
        thickness=0.0,
        material='mirror',
        is_stop=True,
        rx=np.pi/4,  # 45° 绕 X 轴旋转
    )
    
    # 像面
    optic.add_surface(index=2, radius=np.inf, thickness=0.0)
    
    # 检查表面坐标系
    print("\n表面坐标系信息:")
    for i, surf in enumerate(optic.surface_group.surfaces):
        print(f"  表面 {i}: {type(surf).__name__}")
        if hasattr(surf, 'cs'):
            cs = surf.cs
            print(f"    位置: ({cs.x}, {cs.y}, {cs.z})")
            print(f"    旋转: rx={np.degrees(cs.rx):.1f}°, ry={np.degrees(cs.ry):.1f}°, rz={np.degrees(cs.rz):.1f}°")
    
    # 测试不同位置的光线
    print("\n" + "-" * 70)
    print("测试不同位置的入射光线:")
    print("-" * 70)
    
    test_positions = [
        (0.0, 0.0),    # 中心
        (0.001, 0.0),  # 接近中心
        (1.0, 0.0),    # 偏离中心
        (5.0, 0.0),    # 更远
        (0.0, 1.0),    # Y 方向偏离
        (1.0, 1.0),    # 对角
    ]
    
    for x, y in test_positions:
        ray = RealRays(
            x=np.array([x]),
            y=np.array([y]),
            z=np.array([0.0]),
            L=np.array([0.0]),
            M=np.array([0.0]),
            N=np.array([1.0]),
            intensity=np.array([1.0]),
            wavelength=np.array([0.633]),
        )
        ray.opd = np.array([0.0])
        
        # 追迹
        optic.surface_group.trace(ray, skip=1)
        
        pos_valid = np.isfinite(ray.x[0])
        print(f"  入射 ({x:6.3f}, {y:6.3f}) -> 出射 ({ray.x[0]:8.3f}, {ray.y[0]:8.3f}, {ray.z[0]:8.3f})")
        print(f"    方向: ({ray.L[0]:8.5f}, {ray.M[0]:8.5f}, {ray.N[0]:8.5f}), OPD={ray.opd[0]:.4f}, valid={pos_valid}")


def test_reflection_direction():
    """测试反射方向计算"""
    
    print("\n" + "=" * 70)
    print("验证反射方向计算")
    print("=" * 70)
    
    # 入射方向
    d = np.array([0.0, 0.0, 1.0])
    
    # 45° 倾斜镜的法向量
    # 初始法向量 (0, 0, -1)，绕 X 轴旋转 45°
    # 旋转矩阵 Rx(45°) 作用于 (0, 0, -1)
    # Rx = [[1, 0, 0], [0, cos, -sin], [0, sin, cos]]
    # n = Rx @ (0, 0, -1) = (0, sin(45°), -cos(45°)) = (0, 0.707, -0.707)
    # 但这是指向 +Y, -Z 方向，应该指向入射侧
    # 实际上，optiland 的约定可能不同
    
    # 按照 coordinate_conventions.md 的约定：
    # 初始法向量沿入射光轴的反方向（指向入射侧）= (0, 0, -1)
    # 绕 X 轴旋转 45° 后：(0, -sin(45°), -cos(45°)) = (0, -0.707, -0.707)
    n = np.array([0.0, -np.sin(np.pi/4), -np.cos(np.pi/4)])
    
    # 反射公式：r = d - 2(d·n)n
    dot = np.dot(d, n)
    r = d - 2 * dot * n
    
    print(f"入射方向 d = {d}")
    print(f"法向量 n = {n}")
    print(f"d·n = {dot:.6f}")
    print(f"反射方向 r = {r}")
    print(f"预期: (0, -1, 0)")
    
    # 另一种约定：法向量指向 +Z 侧
    n2 = np.array([0.0, np.sin(np.pi/4), np.cos(np.pi/4)])
    dot2 = np.dot(d, n2)
    r2 = d - 2 * dot2 * n2
    
    print(f"\n另一种约定（法向量指向 +Z）:")
    print(f"法向量 n2 = {n2}")
    print(f"d·n2 = {dot2:.6f}")
    print(f"反射方向 r2 = {r2}")


def test_proper_vs_hybrid():
    """对比纯 PROPER 和混合模式"""
    
    print("\n" + "=" * 70)
    print("对比纯 PROPER 和混合模式")
    print("=" * 70)
    
    import proper
    
    # 参数
    wavelength_m = 10.64e-6
    beam_diameter_m = 0.04  # 40 mm
    grid_size = 128
    beam_ratio = 0.5
    f1 = -50.0  # mm, 凸面镜焦距
    
    # 纯 PROPER 模式
    print("\n纯 PROPER 模式（使用 prop_lens）:")
    wfo1 = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, beam_ratio)
    
    # 初始高斯光束
    w0 = 10.0  # mm
    n = proper.prop_get_gridsize(wfo1)
    sampling = proper.prop_get_sampling(wfo1) * 1e3  # mm
    half_size = sampling * n / 2
    coords = np.linspace(-half_size, half_size, n)
    X, Y = np.meshgrid(coords, coords)
    R_sq = X**2 + Y**2
    
    amplitude = np.exp(-R_sq / w0**2)
    gaussian_field = proper.prop_shift_center(amplitude)
    wfo1.wfarr = wfo1.wfarr * gaussian_field
    
    # 应用透镜相位（模拟凸面镜）
    proper.prop_lens(wfo1, f1 * 1e-3)
    
    # 传播 50 mm
    proper.prop_propagate(wfo1, 0.05)
    
    # 获取光束半径
    amp1 = proper.prop_get_amplitude(wfo1)
    intensity1 = amp1**2
    total1 = np.sum(intensity1)
    x_var1 = np.sum(X**2 * intensity1) / total1
    y_var1 = np.sum(Y**2 * intensity1) / total1
    w1 = np.sqrt(2 * (x_var1 + y_var1))
    
    print(f"  传播 50mm 后光束半径: {w1:.3f} mm")
    
    # 计算 ABCD 预期值
    # 凸面镜 ABCD 矩阵：[[1, 0], [-1/f, 1]]
    # 传播 d 的 ABCD 矩阵：[[1, d], [0, 1]]
    # 组合：[[1, d], [0, 1]] @ [[1, 0], [-1/f, 1]] = [[1-d/f, d], [-1/f, 1]]
    d = 50.0  # mm
    A = 1 - d/f1
    B = d
    C = -1/f1
    D = 1
    
    # 高斯光束 q 参数变换
    # q_in = j * z_R，其中 z_R = pi * w0^2 / lambda
    z_R = np.pi * w0**2 / (wavelength_m * 1e6)  # mm
    q_in = 1j * z_R
    
    # q_out = (A*q_in + B) / (C*q_in + D)
    q_out = (A * q_in + B) / (C * q_in + D)
    
    # w_out = sqrt(-lambda / (pi * Im(1/q_out)))
    inv_q_out = 1 / q_out
    w_abcd = np.sqrt(-(wavelength_m * 1e6) / (np.pi * np.imag(inv_q_out)))
    
    print(f"  ABCD 预期光束半径: {w_abcd:.3f} mm")
    print(f"  误差: {abs(w1 - w_abcd) / w_abcd * 100:.2f}%")


if __name__ == "__main__":
    test_optiland_tilted_mirror_detailed()
    test_reflection_direction()
    test_proper_vs_hybrid()
