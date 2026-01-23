"""
详细调试坐标变换问题

问题总结：
1. 出射光线方向始终是 (0, 0, 1)，而不是反射后的方向
2. 出射光线 z 坐标不为 0（应该在出射面上，z=0）
3. OPD 与 z_out 高度相关，说明 OPD 包含了错误的 z 偏移

根本原因分析：
- ElementRaytracer.trace() 方法中的坐标变换逻辑有问题
- 需要检查 R_entrance_to_exit 矩阵的计算是否正确
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


def debug_rotation_matrices(tilt_deg: float):
    """调试旋转矩阵计算"""
    from wavefront_to_rays.element_raytracer import (
        ElementRaytracer, SurfaceDefinition, compute_rotation_matrix
    )
    
    print(f"\n{'='*70}")
    print(f"调试旋转矩阵: {tilt_deg}°")
    print(f"{'='*70}")
    
    tilt_rad = np.radians(tilt_deg)
    
    # 入射方向
    d_in = np.array([0, 0, 1])
    
    # 计算出射方向（反射）
    n = np.array([0, np.sin(tilt_rad), -np.cos(tilt_rad)])
    d_out = d_in - 2 * np.dot(d_in, n) * n
    d_out = d_out / np.linalg.norm(d_out)
    
    print(f"\n入射方向 d_in: {d_in}")
    print(f"表面法向量 n: {n}")
    print(f"出射方向 d_out: {d_out}")
    
    # 计算旋转矩阵
    R_entrance = compute_rotation_matrix(tuple(d_in))
    R_exit = compute_rotation_matrix(tuple(d_out))
    
    print(f"\n入射面旋转矩阵 R_entrance:")
    print(R_entrance)
    
    print(f"\n出射面旋转矩阵 R_exit:")
    print(R_exit)
    
    # 验证旋转矩阵
    # R_entrance 应该将 (0,0,1) 变换为 d_in
    z_axis = np.array([0, 0, 1])
    d_in_computed = R_entrance @ z_axis
    print(f"\nR_entrance @ [0,0,1] = {d_in_computed}")
    print(f"期望: {d_in}")
    print(f"匹配: {np.allclose(d_in_computed, d_in)}")
    
    # R_exit 应该将 (0,0,1) 变换为 d_out
    d_out_computed = R_exit @ z_axis
    print(f"\nR_exit @ [0,0,1] = {d_out_computed}")
    print(f"期望: {d_out}")
    print(f"匹配: {np.allclose(d_out_computed, d_out)}")
    
    # 计算从入射面到出射面的变换矩阵
    R_entrance_to_exit = R_exit.T @ R_entrance
    
    print(f"\n从入射面到出射面的变换矩阵 R_entrance_to_exit = R_exit.T @ R_entrance:")
    print(R_entrance_to_exit)
    
    # 验证：入射面局部坐标系中的 (0,0,1) 应该变换为出射面局部坐标系中的什么？
    # 入射面局部坐标系中的 (0,0,1) 在全局坐标系中是 d_in
    # 在出射面局部坐标系中，d_in 应该是什么？
    
    # 全局坐标系中的 d_in 变换到出射面局部坐标系
    d_in_in_exit_local = R_exit.T @ d_in
    print(f"\n入射方向在出射面局部坐标系中: {d_in_in_exit_local}")
    
    # 使用 R_entrance_to_exit 变换 (0,0,1)
    z_transformed = R_entrance_to_exit @ z_axis
    print(f"R_entrance_to_exit @ [0,0,1] = {z_transformed}")
    print(f"匹配: {np.allclose(z_transformed, d_in_in_exit_local)}")
    
    # 关键问题：出射光线在入射面局部坐标系中的方向是什么？
    # optiland 追迹后，光线方向应该是反射后的方向
    # 但在入射面局部坐标系中，反射后的方向是什么？
    
    # 全局坐标系中的 d_out 变换到入射面局部坐标系
    d_out_in_entrance_local = R_entrance.T @ d_out
    print(f"\n出射方向在入射面局部坐标系中: {d_out_in_entrance_local}")
    
    # 这个方向经过 R_entrance_to_exit 变换后应该是 (0,0,1)
    d_out_transformed = R_entrance_to_exit @ d_out_in_entrance_local
    print(f"R_entrance_to_exit @ d_out_in_entrance_local = {d_out_transformed}")
    print(f"期望: [0, 0, 1]")
    print(f"匹配: {np.allclose(d_out_transformed, z_axis)}")
    
    return {
        'd_in': d_in,
        'd_out': d_out,
        'R_entrance': R_entrance,
        'R_exit': R_exit,
        'R_entrance_to_exit': R_entrance_to_exit,
    }


def debug_optiland_trace(tilt_deg: float):
    """调试 optiland 追迹过程"""
    from wavefront_to_rays import WavefrontToRaysSampler
    from wavefront_to_rays.element_raytracer import (
        ElementRaytracer, SurfaceDefinition, compute_rotation_matrix
    )
    from optiland.rays import RealRays
    
    print(f"\n{'='*70}")
    print(f"调试 optiland 追迹: {tilt_deg}°")
    print(f"{'='*70}")
    
    # 参数
    wavelength_um = 0.633
    tilt_rad = np.radians(tilt_deg)
    
    # 计算出射方向
    d_in = np.array([0, 0, 1])
    n = np.array([0, np.sin(tilt_rad), -np.cos(tilt_rad)])
    d_out = d_in - 2 * np.dot(d_in, n) * n
    d_out = d_out / np.linalg.norm(d_out)
    
    # 创建简单的输入光线（只有几条）
    input_rays = RealRays(
        x=np.array([0.0, 5.0, -5.0, 0.0, 0.0]),
        y=np.array([0.0, 0.0, 0.0, 5.0, -5.0]),
        z=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        L=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        M=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        N=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        intensity=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        wavelength=np.array([wavelength_um] * 5),
    )
    input_rays.opd = np.zeros(5)
    
    print(f"\n输入光线（入射面局部坐标系）:")
    print(f"  x: {input_rays.x}")
    print(f"  y: {input_rays.y}")
    print(f"  z: {input_rays.z}")
    print(f"  L: {input_rays.L}")
    print(f"  M: {input_rays.M}")
    print(f"  N: {input_rays.N}")
    
    # 创建表面定义
    surface_def = SurfaceDefinition(
        surface_type='mirror',
        radius=np.inf,
        thickness=0.0,
        material='mirror',
        semi_aperture=30.0,
        conic=0.0,
        tilt_x=tilt_rad,
        tilt_y=0.0,
    )
    
    # 创建光线追迹器
    raytracer = ElementRaytracer(
        surfaces=[surface_def],
        wavelength=wavelength_um,
        chief_ray_direction=(0, 0, 1),
        entrance_position=(0, 0, 0),
        exit_chief_direction=tuple(d_out),
    )
    
    print(f"\n旋转矩阵:")
    print(f"  R_entrance:\n{raytracer.rotation_matrix}")
    print(f"  R_exit:\n{raytracer.exit_rotation_matrix}")
    
    # 手动执行追迹过程，观察中间结果
    
    # 1. 复制光线
    traced_rays = RealRays(
        x=np.asarray(input_rays.x).copy(),
        y=np.asarray(input_rays.y).copy(),
        z=np.asarray(input_rays.z).copy(),
        L=np.asarray(input_rays.L).copy(),
        M=np.asarray(input_rays.M).copy(),
        N=np.asarray(input_rays.N).copy(),
        intensity=np.asarray(input_rays.i).copy(),
        wavelength=np.asarray(input_rays.w).copy(),
    )
    traced_rays.opd = np.asarray(input_rays.opd).copy()
    
    # 2. 调用 _trace_with_signed_opd
    raytracer._trace_with_signed_opd(traced_rays, skip=1)
    
    print(f"\noptiland 追迹后（入射面局部坐标系）:")
    print(f"  x: {np.asarray(traced_rays.x)}")
    print(f"  y: {np.asarray(traced_rays.y)}")
    print(f"  z: {np.asarray(traced_rays.z)}")
    print(f"  L: {np.asarray(traced_rays.L)}")
    print(f"  M: {np.asarray(traced_rays.M)}")
    print(f"  N: {np.asarray(traced_rays.N)}")
    print(f"  OPD: {np.asarray(traced_rays.opd)}")
    
    # 3. 坐标变换
    R_entrance_to_exit = raytracer.exit_rotation_matrix.T @ raytracer.rotation_matrix
    
    x_entrance = np.asarray(traced_rays.x)
    y_entrance = np.asarray(traced_rays.y)
    z_entrance = np.asarray(traced_rays.z)
    L_entrance = np.asarray(traced_rays.L)
    M_entrance = np.asarray(traced_rays.M)
    N_entrance = np.asarray(traced_rays.N)
    
    # 位置转换
    pos_entrance = np.stack([x_entrance, y_entrance, z_entrance], axis=0)
    pos_exit = R_entrance_to_exit @ pos_entrance
    
    # 方向转换
    dir_entrance = np.stack([L_entrance, M_entrance, N_entrance], axis=0)
    dir_exit = R_entrance_to_exit @ dir_entrance
    
    print(f"\n坐标变换后（出射面局部坐标系）:")
    print(f"  x: {pos_exit[0]}")
    print(f"  y: {pos_exit[1]}")
    print(f"  z: {pos_exit[2]}")
    print(f"  L: {dir_exit[0]}")
    print(f"  M: {dir_exit[1]}")
    print(f"  N: {dir_exit[2]}")
    
    # 分析问题
    print(f"\n问题分析:")
    
    # 检查 optiland 追迹后的光线方向
    # 对于反射镜，光线方向应该是反射后的方向
    # 在入射面局部坐标系中，反射后的方向应该是什么？
    
    # 入射方向在入射面局部坐标系中是 (0, 0, 1)
    # 表面法向量在入射面局部坐标系中是什么？
    # 表面倾斜 tilt_x 弧度，法向量初始为 (0, 0, -1)
    # 绕 X 轴旋转 tilt_x 后：
    c, s = np.cos(tilt_rad), np.sin(tilt_rad)
    n_local = np.array([0, -s, -c])  # 注意：初始法向量是 (0, 0, -1)
    
    print(f"  表面法向量（入射面局部坐标系）: {n_local}")
    
    # 反射方向
    d_in_local = np.array([0, 0, 1])
    d_out_local = d_in_local - 2 * np.dot(d_in_local, n_local) * n_local
    d_out_local = d_out_local / np.linalg.norm(d_out_local)
    
    print(f"  期望的反射方向（入射面局部坐标系）: {d_out_local}")
    print(f"  实际的光线方向（入射面局部坐标系）: [{L_entrance[0]:.6f}, {M_entrance[0]:.6f}, {N_entrance[0]:.6f}]")
    
    # 检查 optiland 是否正确计算了反射方向
    if np.allclose([L_entrance[0], M_entrance[0], N_entrance[0]], d_out_local, atol=1e-6):
        print("  ✓ optiland 正确计算了反射方向")
    else:
        print("  ✗ optiland 没有正确计算反射方向！")
        print("    这可能是因为 optiland 的表面定义或追迹逻辑有问题")


def main():
    print("=" * 70)
    print("详细调试坐标变换问题")
    print("=" * 70)
    
    # 测试几个角度
    for angle in [5, 22.5, 45]:
        debug_rotation_matrices(angle)
        debug_optiland_trace(angle)


if __name__ == "__main__":
    main()
