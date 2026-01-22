"""
详细分析 Surface 3（45度折叠镜）从入射面到出射面的全流程误差
添加更多调试输出来定位问题
"""

import sys
print("Script starting...", flush=True)

from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'optiland-master'))

print("Paths configured", flush=True)

import numpy as np
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

print("Numpy imported", flush=True)

try:
    from hybrid_optical_propagation import (
        SourceDefinition,
        HybridOpticalPropagator,
        load_optical_system_from_zmx,
    )
    print("Hybrid imports done", flush=True)
except Exception as e:
    print(f"Error importing hybrid: {e}", flush=True)
    raise

try:
    from wavefront_to_rays import WavefrontToRaysSampler
    print("WavefrontToRaysSampler imported", flush=True)
except Exception as e:
    print(f"Error importing sampler: {e}", flush=True)
    raise

from scipy.interpolate import RegularGridInterpolator
print("All imports done", flush=True)

# 加载光学系统
zmx_file = "optiland-master/tests/zemax_files/complicated_fold_mirrors_setup_v2.zmx"
print(f"Loading ZMX file: {zmx_file}", flush=True)
optical_system = load_optical_system_from_zmx(zmx_file)
print(f"Loaded {len(optical_system)} surfaces", flush=True)

# 创建源
source = SourceDefinition(
    wavelength_um=0.55,
    w0_mm=5.0,
    z0_mm=0.0,
    grid_size=256,
    physical_size_mm=40.0,
)
print("Source created", flush=True)

# 创建传播器
print("Creating propagator...", flush=True)
propagator = HybridOpticalPropagator(
    optical_system=optical_system,
    source=source,
    wavelength_um=0.55,
    grid_size=256,
    num_rays=150,
)
print("Propagator created", flush=True)

# 初始化传播
print("Initializing propagation...", flush=True)
propagator._current_state = propagator._initialize_propagation()
propagator._surface_states = [propagator._current_state]
print("Propagation initialized", flush=True)

# 传播到 Surface 3
print("Propagating to surfaces 0-2...", flush=True)
for i in range(3):
    print(f"  Propagating to surface {i}...", flush=True)
    propagator._propagate_to_surface(i)
print("Propagated to surfaces 0-2", flush=True)

# 传播到 Surface 3 入射面
print("Propagating to surface 3...", flush=True)
propagator._propagate_to_surface(3)
print("Propagated to surface 3", flush=True)

# 找到 Surface 3 入射面状态
print("Finding Surface 3 entrance state...", flush=True)
state_entrance = None
for state in propagator._surface_states:
    print(f"  State: surface_index={state.surface_index}, position={state.position}", flush=True)
    if state.surface_index == 3 and state.position == 'entrance':
        state_entrance = state
        break

if state_entrance is None:
    print("ERROR: Surface 3 entrance state not found!", flush=True)
    sys.exit(1)

print("Found Surface 3 entrance state", flush=True)

wavelength_mm = 0.55 * 1e-3
k = 2 * np.pi / wavelength_mm

print("=" * 70)
print("Surface 3（45度折叠镜）全流程误差分析")
print("=" * 70)

# =========================================================================
# 步骤 1: 入射面相位分析
# =========================================================================
print("\n【步骤 1】入射面相位分析")
print("-" * 50)

grid_size = state_entrance.grid_sampling.grid_size
physical_size_mm = state_entrance.grid_sampling.physical_size_mm

print(f"Grid size: {grid_size}, Physical size: {physical_size_mm} mm", flush=True)

# 计算 Pilot Beam 相位
pilot_phase_entrance = state_entrance.pilot_beam_params.compute_phase_grid(
    grid_size, physical_size_mm
)
print(f"Pilot phase computed", flush=True)

# 仿真相位
sim_phase_entrance = state_entrance.phase
print(f"Simulation phase shape: {sim_phase_entrance.shape}", flush=True)

# 有效区域
mask = state_entrance.amplitude > 0.01 * np.max(state_entrance.amplitude)
print(f"Valid pixels: {np.sum(mask)}", flush=True)

# 误差
diff_entrance = sim_phase_entrance - pilot_phase_entrance
rms_entrance = np.std(diff_entrance[mask]) / (2 * np.pi)

print(f"Pilot Beam 曲率半径: {state_entrance.pilot_beam_params.curvature_radius_mm:.2f} mm")
print(f"仿真相位范围: [{np.min(sim_phase_entrance[mask]):.6f}, {np.max(sim_phase_entrance[mask]):.6f}] rad")
print(f"Pilot Beam 相位范围: [{np.min(pilot_phase_entrance[mask]):.6f}, {np.max(pilot_phase_entrance[mask]):.6f}] rad")
print(f"入射面相位 RMS 误差: {rms_entrance:.6f} waves")

# =========================================================================
# 步骤 2: 光线采样分析
# =========================================================================
print("\n【步骤 2】光线采样分析")
print("-" * 50)

print("Creating WavefrontToRaysSampler...", flush=True)
sampler = WavefrontToRaysSampler(
    amplitude=state_entrance.amplitude,
    phase=sim_phase_entrance,
    physical_size=state_entrance.grid_sampling.physical_size_mm,
    wavelength=0.55,
    num_rays=150,
)
print("Sampler created", flush=True)

# 获取采样光线
output_rays = sampler.get_output_rays()
ray_x, ray_y = sampler.get_ray_positions()
print(f"采样光线数量: {len(ray_x)}", flush=True)

# 光线位置处的 Pilot Beam 相位
r_sq_rays = ray_x**2 + ray_y**2
R_entrance = state_entrance.pilot_beam_params.curvature_radius_mm
if np.isinf(R_entrance):
    pilot_phase_at_rays = np.zeros_like(r_sq_rays)
else:
    pilot_phase_at_rays = k * r_sq_rays / (2 * R_entrance)

# 光线的 OPD 转换为相位
ray_opd_mm = np.asarray(output_rays.opd)
ray_phase = k * ray_opd_mm

# 比较光线相位与 Pilot Beam 相位
diff_rays = ray_phase - pilot_phase_at_rays
rms_rays = np.std(diff_rays) / (2 * np.pi)

print(f"光线 OPD 范围: [{np.min(ray_opd_mm):.6f}, {np.max(ray_opd_mm):.6f}] mm")
print(f"光线相位范围: [{np.min(ray_phase):.6f}, {np.max(ray_phase):.6f}] rad")
print(f"Pilot Beam 相位范围: [{np.min(pilot_phase_at_rays):.6f}, {np.max(pilot_phase_at_rays):.6f}] rad")
print(f"光线相位 vs Pilot Beam RMS 误差: {rms_rays:.6f} waves")

# 额外分析：光线相位与网格相位的对比
print("\n  [额外分析] 光线相位与网格相位插值对比:")
half_size = physical_size_mm / 2
coords = np.linspace(-half_size, half_size, grid_size)
interpolator = RegularGridInterpolator(
    (coords, coords),
    sim_phase_entrance,
    method='linear',
    bounds_error=False,
    fill_value=0.0,
)
points = np.column_stack([ray_y, ray_x])
grid_phase_at_rays = interpolator(points)
diff_grid_vs_ray = ray_phase - grid_phase_at_rays
rms_grid_vs_ray = np.std(diff_grid_vs_ray) / (2 * np.pi)
print(f"  网格相位插值范围: [{np.min(grid_phase_at_rays):.6f}, {np.max(grid_phase_at_rays):.6f}] rad")
print(f"  光线相位 vs 网格插值 RMS 误差: {rms_grid_vs_ray:.6f} waves")

print("\n脚本执行完成！", flush=True)
