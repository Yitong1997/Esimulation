"""调试 Surface 8 的曲率处理"""
import sys
sys.path.insert(0, 'src')
import numpy as np
from bts import load_zmx, GaussianSource

# 加载系统
system = load_zmx('optiland-master/tests/zemax_files/complicated_fold_mirrors_setup_v2.zmx')

# 创建光源
source = GaussianSource(
    wavelength_um=0.633,  # 633nm
    w0=2.0,            # 光腰 2mm
    beam_diam_fraction=0.5
)

# 只传播到 Surface 8 和 9（透镜）
# 先获取表面信息
surfaces = system._global_surfaces
print(f"总表面数: {len(surfaces)}")
print()

# Surface 8 详细输出
s8 = surfaces[8]
print("=== Surface 8 (BK7 入射面) ===")
print(f"  位置: {s8.vertex_position}")
print(f"  曲率半径: {s8.radius}")
print(f"  材料: {s8.material}")
print(f"  法向 (Z-axis): {s8.orientation[:, 2]}")

# 模拟 _create_optic_base 中的计算
# 假设入射方向是从 S7 到 S8
s7 = surfaces[7]
delta = s8.vertex_position - s7.vertex_position
entrance_dir = delta / np.linalg.norm(delta) if np.linalg.norm(delta) > 0 else np.array([0, 0, 1])
print(f"  入射方向 (估算): {entrance_dir}")

# 计算 R_rel
from wavefront_to_rays.element_raytracer import compute_rotation_matrix
rotation_matrix = compute_rotation_matrix(tuple(entrance_dir))
R_rel = rotation_matrix.T @ s8.orientation
normal_local = R_rel[:, 2]
L, M, N = normal_local
print(f"  R_rel Z-axis (法向在入射坐标系): ({L:.4f}, {M:.4f}, {N:.4f})")
print(f"  Nz = {N:.4f}")

# 检查曲率翻转
radius = s8.radius
if N < 0:
    print(f"  -> N < 0, 需要翻转曲率半径: {radius} -> {-radius}")
else:
    print(f"  -> N >= 0, 保持曲率半径: {radius}")
    
print()
print("=== 运行完整仿真（只到 Surface 9）===")

# 运行仿真看 Surface 8/9 的结果
result = system.trace(source, num_surfaces=9)  # 只追迹到 Surface 9

# 检查残差
final_wf = result.get_final_wavefront()
if final_wf is not None:
    rms = final_wf.get_residual_rms_waves()
    print(f"波前 RMS 残差: {rms:.4f} waves")
