"""
详细调试 optiland 对倾斜表面的光线追迹

核心问题：optiland 对 45° 倾斜平面镜返回 NaN
原因分析：在局部坐标系中，光线方向的 N 分量可能为 0，导致除以零
"""
import numpy as np
from optiland.optic import Optic
from optiland.rays import RealRays


def test_coordinate_transform():
    """测试坐标变换对光线方向的影响"""
    
    print("=" * 70)
    print("测试坐标变换对光线方向的影响")
    print("=" * 70)
    
    # 入射光线方向（全局坐标系）
    L, M, N = 0.0, 0.0, 1.0
    print(f"\n入射光线方向（全局）: ({L}, {M}, {N})")
    
    # 45° 绕 X 轴旋转的逆变换（localize 使用 -rx）
    rx = np.pi / 4  # 45°
    
    # 旋转矩阵 Rx(-45°)
    c, s = np.cos(-rx), np.sin(-rx)
    Rx_inv = np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])
    
    # 变换光线方向
    dir_global = np.array([L, M, N])
    dir_local = Rx_inv @ dir_global
    
    print(f"旋转矩阵 Rx(-45°):\n{Rx_inv}")
    print(f"变换后光线方向（局部）: ({dir_local[0]:.6f}, {dir_local[1]:.6f}, {dir_local[2]:.6f})")
    print(f"N 分量: {dir_local[2]:.6f}")
    
    # 计算 t = -z / N
    z = 0.0  # 假设光线起点在 z=0
    if abs(dir_local[2]) < 1e-10:
        print(f"警告：N 分量接近零，t = -z/N 会产生无穷大或 NaN！")
    else:
        t = -z / dir_local[2]
        print(f"t = -z/N = {t}")


def test_optiland_with_different_tilts():
    """测试不同倾斜角度下 optiland 的行为"""
    
    print("\n" + "=" * 70)
    print("测试不同倾斜角度下 optiland 的行为")
    print("=" * 70)
    
    tilt_angles = [0, 15, 30, 44, 44.9, 45, 45.1, 46, 60, 90]
    
    for angle_deg in tilt_angles:
        angle_rad = np.radians(angle_deg)
        
        # 创建光学系统
        optic = Optic()
        optic.set_aperture(aperture_type='EPD', value=40.0)
        optic.set_field_type(field_type='angle')
        optic.add_field(y=0, x=0)
        optic.add_wavelength(value=0.633, is_primary=True)
        
        # 物面
        optic.add_surface(index=0, radius=np.inf, thickness=np.inf)
        
        # 倾斜平面镜
        optic.add_surface(
            index=1,
            radius=np.inf,
            thickness=0.0,
            material='mirror',
            is_stop=True,
            rx=angle_rad,
        )
        
        # 像面
        optic.add_surface(index=2, radius=np.inf, thickness=0.0)
        
        # 创建测试光线
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
        
        # 追迹
        optic.surface_group.trace(ray, skip=1)
        
        # 检查结果
        x_valid = np.isfinite(ray.x[0])
        result = "有效" if x_valid else "NaN"
        
        # 计算局部坐标系中的 N 分量
        c, s = np.cos(-angle_rad), np.sin(-angle_rad)
        N_local = s * 0.0 + c * 1.0  # 简化：M=0, N=1
        
        print(f"  倾斜 {angle_deg:5.1f}°: N_local={N_local:8.5f}, 结果={result}")
        if x_valid:
            print(f"           出射方向: ({ray.L[0]:.5f}, {ray.M[0]:.5f}, {ray.N[0]:.5f})")


def test_alternative_approach():
    """测试替代方案：不使用 optiland 的倾斜，而是手动处理坐标变换"""
    
    print("\n" + "=" * 70)
    print("替代方案：手动坐标变换 + 非倾斜表面追迹")
    print("=" * 70)
    
    # 方案：
    # 1. 将入射光线从全局坐标系变换到元件局部坐标系
    # 2. 在局部坐标系中，表面是非倾斜的（z=0 平面）
    # 3. 追迹光线
    # 4. 将出射光线变换回全局坐标系
    
    # 入射光线（全局坐标系）
    x_g, y_g, z_g = 0.0, 0.0, 0.0
    L_g, M_g, N_g = 0.0, 0.0, 1.0
    
    print(f"\n入射光线（全局）:")
    print(f"  位置: ({x_g}, {y_g}, {z_g})")
    print(f"  方向: ({L_g}, {M_g}, {N_g})")
    
    # 45° 倾斜镜的旋转矩阵
    rx = np.pi / 4
    c, s = np.cos(rx), np.sin(rx)
    R = np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])
    R_inv = R.T  # 正交矩阵的逆等于转置
    
    # 变换到局部坐标系
    pos_g = np.array([x_g, y_g, z_g])
    dir_g = np.array([L_g, M_g, N_g])
    
    pos_l = R_inv @ pos_g
    dir_l = R_inv @ dir_g
    
    print(f"\n变换到局部坐标系:")
    print(f"  位置: ({pos_l[0]:.6f}, {pos_l[1]:.6f}, {pos_l[2]:.6f})")
    print(f"  方向: ({dir_l[0]:.6f}, {dir_l[1]:.6f}, {dir_l[2]:.6f})")
    
    # 在局部坐标系中，表面是 z=0 平面
    # 计算光线到平面的距离
    if abs(dir_l[2]) < 1e-10:
        print(f"\n警告：光线平行于表面，无法求交！")
        return
    
    t = -pos_l[2] / dir_l[2]
    print(f"\n传播距离 t = {t:.6f}")
    
    # 传播到表面
    pos_l_surface = pos_l + t * dir_l
    print(f"表面交点（局部）: ({pos_l_surface[0]:.6f}, {pos_l_surface[1]:.6f}, {pos_l_surface[2]:.6f})")
    
    # 反射（表面法向量在局部坐标系中是 (0, 0, 1)）
    n_l = np.array([0.0, 0.0, 1.0])
    dir_l_out = dir_l - 2 * np.dot(dir_l, n_l) * n_l
    print(f"反射后方向（局部）: ({dir_l_out[0]:.6f}, {dir_l_out[1]:.6f}, {dir_l_out[2]:.6f})")
    
    # 变换回全局坐标系
    pos_g_out = R @ pos_l_surface
    dir_g_out = R @ dir_l_out
    
    print(f"\n变换回全局坐标系:")
    print(f"  位置: ({pos_g_out[0]:.6f}, {pos_g_out[1]:.6f}, {pos_g_out[2]:.6f})")
    print(f"  方向: ({dir_g_out[0]:.6f}, {dir_g_out[1]:.6f}, {dir_g_out[2]:.6f})")
    print(f"  预期方向: (0, -1, 0)")


def test_optiland_non_tilted_mirror():
    """测试 optiland 对非倾斜镜的追迹（作为基准）"""
    
    print("\n" + "=" * 70)
    print("测试 optiland 对非倾斜镜的追迹（基准）")
    print("=" * 70)
    
    # 创建非倾斜平面镜系统
    optic = Optic()
    optic.set_aperture(aperture_type='EPD', value=40.0)
    optic.set_field_type(field_type='angle')
    optic.add_field(y=0, x=0)
    optic.add_wavelength(value=0.633, is_primary=True)
    
    # 物面
    optic.add_surface(index=0, radius=np.inf, thickness=np.inf)
    
    # 非倾斜平面镜
    optic.add_surface(
        index=1,
        radius=np.inf,
        thickness=0.0,
        material='mirror',
        is_stop=True,
    )
    
    # 像面
    optic.add_surface(index=2, radius=np.inf, thickness=0.0)
    
    # 创建测试光线
    ray = RealRays(
        x=np.array([0.0, 1.0, 5.0]),
        y=np.array([0.0, 0.0, 0.0]),
        z=np.array([0.0, 0.0, 0.0]),
        L=np.array([0.0, 0.0, 0.0]),
        M=np.array([0.0, 0.0, 0.0]),
        N=np.array([1.0, 1.0, 1.0]),
        intensity=np.array([1.0, 1.0, 1.0]),
        wavelength=np.array([0.633, 0.633, 0.633]),
    )
    ray.opd = np.array([0.0, 0.0, 0.0])
    
    # 追迹
    optic.surface_group.trace(ray, skip=1)
    
    print(f"\n非倾斜平面镜追迹结果:")
    for i in range(3):
        print(f"  光线 {i}: 位置=({ray.x[i]:.3f}, {ray.y[i]:.3f}, {ray.z[i]:.3f}), "
              f"方向=({ray.L[i]:.5f}, {ray.M[i]:.5f}, {ray.N[i]:.5f})")


if __name__ == "__main__":
    test_coordinate_transform()
    test_optiland_with_different_tilts()
    test_alternative_approach()
    test_optiland_non_tilted_mirror()
