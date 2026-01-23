"""
调试 optiland 表面法向量定义

问题：optiland 计算的反射方向 M 分量符号相反
- 期望：[0, -0.17364818, -0.98480775]
- 实际：[0, +0.17364818, -0.98480775]

这说明 optiland 的表面法向量定义可能与我们的预期不同。
"""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'optiland-master'))
sys.path.insert(0, str(project_root / 'proper_v3.3.4_python'))

import warnings
warnings.filterwarnings('ignore')


def analyze_optiland_tilt_convention():
    """分析 optiland 的倾斜约定"""
    from optiland.optic import Optic
    from optiland.rays import RealRays
    
    print("=" * 70)
    print("分析 optiland 的倾斜约定")
    print("=" * 70)
    
    # 测试不同的倾斜角度
    for tilt_deg in [5, 10, 22.5, 45]:
        tilt_rad = np.radians(tilt_deg)
        
        print(f"\n--- 倾斜角度: {tilt_deg}° ---")
        
        # 创建一个简单的倾斜平面镜系统
        optic = Optic()
        optic.set_aperture(aperture_type='EPD', value=20.0)
        optic.set_field_type(field_type='angle')
        optic.add_field(y=0, x=0)
        optic.add_wavelength(value=0.633, is_primary=True)
        
        # 物面
        optic.add_surface(index=0, radius=np.inf, thickness=np.inf)
        
        # 倾斜的平面镜
        optic.add_surface(
            index=1,
            radius=np.inf,
            thickness=0.0,
            material='mirror',
            is_stop=True,
            rx=tilt_rad,  # 绕 X 轴旋转
            ry=0.0,
        )
        
        # 出射面（透明平面）
        optic.add_surface(
            index=2,
            radius=np.inf,
            thickness=0.0,
            material='air',
        )
        
        # 创建测试光线
        rays = RealRays(
            x=np.array([0.0]),
            y=np.array([0.0]),
            z=np.array([0.0]),
            L=np.array([0.0]),
            M=np.array([0.0]),
            N=np.array([1.0]),
            intensity=np.array([1.0]),
            wavelength=np.array([0.633]),
        )
        
        # 追迹
        optic.surface_group.trace(rays, skip=1)
        
        # 输出结果
        L_out = float(rays.L[0])
        M_out = float(rays.M[0])
        N_out = float(rays.N[0])
        
        print(f"  optiland 输出方向: L={L_out:.6f}, M={M_out:.6f}, N={N_out:.6f}")
        
        # 计算期望的反射方向
        # 入射方向: (0, 0, 1)
        # 表面法向量（指向入射侧）: (0, sin(tilt), -cos(tilt))
        # 反射方向: d - 2(d·n)n
        
        d_in = np.array([0, 0, 1])
        n = np.array([0, np.sin(tilt_rad), -np.cos(tilt_rad)])
        d_out_expected = d_in - 2 * np.dot(d_in, n) * n
        
        print(f"  期望方向（法向量指向入射侧）: L={d_out_expected[0]:.6f}, M={d_out_expected[1]:.6f}, N={d_out_expected[2]:.6f}")
        
        # 另一种可能：法向量指向出射侧
        n_alt = np.array([0, -np.sin(tilt_rad), np.cos(tilt_rad)])
        d_out_alt = d_in - 2 * np.dot(d_in, n_alt) * n_alt
        
        print(f"  期望方向（法向量指向出射侧）: L={d_out_alt[0]:.6f}, M={d_out_alt[1]:.6f}, N={d_out_alt[2]:.6f}")
        
        # 检查哪个匹配
        if np.allclose([L_out, M_out, N_out], d_out_expected, atol=1e-6):
            print(f"  ✓ 匹配：法向量指向入射侧")
        elif np.allclose([L_out, M_out, N_out], d_out_alt, atol=1e-6):
            print(f"  ✓ 匹配：法向量指向出射侧")
        else:
            print(f"  ✗ 不匹配任何预期")
            
            # 尝试找出 optiland 使用的法向量
            # 反射公式: d_out = d_in - 2(d_in·n)n
            # 已知 d_in 和 d_out，求 n
            # 2(d_in·n)n = d_in - d_out
            # 设 k = 2(d_in·n)，则 kn = d_in - d_out
            # n = (d_in - d_out) / |d_in - d_out|
            
            d_out_actual = np.array([L_out, M_out, N_out])
            diff = d_in - d_out_actual
            if np.linalg.norm(diff) > 1e-10:
                n_inferred = diff / np.linalg.norm(diff)
                print(f"  推断的法向量: {n_inferred}")


def analyze_optiland_rx_convention():
    """分析 optiland 的 rx 参数约定"""
    from optiland.optic import Optic
    from optiland.rays import RealRays
    
    print("\n" + "=" * 70)
    print("分析 optiland 的 rx 参数约定")
    print("=" * 70)
    
    # 测试正负 rx 的效果
    for rx_deg in [-5, 5]:
        rx_rad = np.radians(rx_deg)
        
        print(f"\n--- rx = {rx_deg}° ---")
        
        # 创建系统
        optic = Optic()
        optic.set_aperture(aperture_type='EPD', value=20.0)
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
            rx=rx_rad,
            ry=0.0,
        )
        optic.add_surface(
            index=2,
            radius=np.inf,
            thickness=0.0,
            material='air',
        )
        
        # 追迹
        rays = RealRays(
            x=np.array([0.0]),
            y=np.array([0.0]),
            z=np.array([0.0]),
            L=np.array([0.0]),
            M=np.array([0.0]),
            N=np.array([1.0]),
            intensity=np.array([1.0]),
            wavelength=np.array([0.633]),
        )
        
        optic.surface_group.trace(rays, skip=1)
        
        L_out = float(rays.L[0])
        M_out = float(rays.M[0])
        N_out = float(rays.N[0])
        
        print(f"  输出方向: L={L_out:.6f}, M={M_out:.6f}, N={N_out:.6f}")
        
        # 分析 M 分量的符号
        # 如果 rx > 0，表面绕 X 轴正向旋转
        # 右手定则：X 轴指向右，正向旋转使 Y 轴向 Z 轴方向转
        # 这意味着表面法向量的 Y 分量变为正
        # 反射后，光线的 M 分量应该变为正
        
        if rx_deg > 0:
            if M_out > 0:
                print(f"  rx > 0 时，M_out > 0：符合右手定则")
            else:
                print(f"  rx > 0 时，M_out < 0：不符合右手定则")
        else:
            if M_out < 0:
                print(f"  rx < 0 时，M_out < 0：符合右手定则")
            else:
                print(f"  rx < 0 时，M_out > 0：不符合右手定则")


def main():
    analyze_optiland_tilt_convention()
    analyze_optiland_rx_convention()
    
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print("""
关键发现：

1. optiland 的 rx 参数定义：
   - rx > 0 时，表面绕 X 轴正向旋转（右手定则）
   - 这导致反射光线的 M 分量为正

2. 我们的预期：
   - 表面法向量指向入射侧
   - 对于 rx > 0，法向量应该是 (0, sin(rx), -cos(rx))
   - 反射后，M 分量应该为正

3. 问题根源：
   - 在 ElementRaytracer 中，我们计算出射方向时使用的法向量定义
     可能与 optiland 的定义不一致
   - 需要检查 _compute_exit_chief_direction 方法

4. 解决方案：
   - 确保 ElementRaytracer 中的出射方向计算与 optiland 一致
   - 或者在坐标变换时考虑这个差异
""")


if __name__ == "__main__":
    main()
