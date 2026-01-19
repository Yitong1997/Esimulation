"""
WavefrontToRaysSampler + ElementRaytracer 联合测试可视化

验证两个模块联合使用时的正确性：
1. 使用非常平坦的长焦球面波（小 NA）
2. 球面波经凹面镜反射后应变为平面波
3. 可视化各个阶段的数据，便于观察问题

理论基础：
- 从凹面镜焦点发出的球面波，经凹面镜反射后变为平面波
- 平面波的 OPD 应为常数（标准差接近 0）

作者：混合光学仿真项目
"""

import sys
import os

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from optiland.rays import RealRays
from wavefront_to_rays import WavefrontToRaysSampler
from wavefront_to_rays.element_raytracer import (
    ElementRaytracer,
    SurfaceDefinition,
    create_concave_mirror_for_spherical_wave,
)


def create_spherical_wave_from_focus(
    grid_size: int,
    physical_size: float,
    focal_distance: float,
    wavelength: float,
) -> np.ndarray:
    """创建从焦点发出的球面波波前
    
    球面波从距离入射面 focal_distance 处的点源发出。
    
    参数:
        grid_size: 网格大小
        physical_size: 物理尺寸（直径），单位：mm
        focal_distance: 焦点到入射面的距离，单位：mm
        wavelength: 波长，单位：μm
    
    返回:
        波前复振幅数组
    
    注意:
        为了避免相位包裹，需要确保最大相位变化小于 π。
        最大相位变化 ≈ k * (r_max - focal_distance)
        其中 r_max = sqrt((physical_size/2)^2 + focal_distance^2)
        
        对于小 NA 情况（physical_size << focal_distance）：
        r_max - focal_distance ≈ (physical_size/2)^2 / (2 * focal_distance)
        
        要求相位变化 < π：
        k * (physical_size/2)^2 / (2 * focal_distance) < π
        即 focal_distance > (physical_size/2)^2 / wavelength
    """
    half_size = physical_size / 2.0
    x = np.linspace(-half_size, half_size, grid_size)
    y = np.linspace(-half_size, half_size, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # 计算从焦点到入射面各点的距离
    # 焦点位于 (0, 0, -focal_distance)
    # 入射面位于 z = 0
    r = np.sqrt(X**2 + Y**2 + focal_distance**2)
    
    # 球面波相位：k * (r - focal_distance)
    # 其中 focal_distance 是主光线的光程（参考）
    wavelength_mm = wavelength * 1e-3  # μm -> mm
    k = 2 * np.pi / wavelength_mm
    
    # 相对于主光线的相位差（OPD）
    opd_mm = r - focal_distance
    
    # 检查最大 OPD 是否会导致相位包裹
    max_opd_mm = np.max(np.abs(opd_mm))
    max_phase = k * max_opd_mm
    print(f"  [调试] 最大 OPD: {max_opd_mm:.6f} mm = {max_opd_mm/wavelength_mm:.4f} waves")
    print(f"  [调试] 最大相位: {max_phase:.4f} rad = {max_phase/np.pi:.4f} π")
    
    if max_phase > np.pi:
        print(f"  [警告] 相位变化超过 π，可能发生相位包裹！")
        print(f"  [建议] 增加焦距或减小光瞳尺寸")
    
    # 相位
    phase = k * opd_mm
    
    # 创建复振幅
    amplitude = np.ones_like(phase)
    wavefront = amplitude * np.exp(1j * phase)
    
    return wavefront


def visualize_combined_test():
    """可视化联合测试
    
    测试流程：
    1. 创建长焦球面波（从凹面镜焦点发出）
    2. 使用 WavefrontToRaysSampler 将波前转换为光线
    3. 使用 ElementRaytracer 对光线进行凹面镜反射追迹
    4. 验证出射光线是否为平面波（OPD 为常数）
    """
    
    print("=" * 70)
    print("WavefrontToRaysSampler + ElementRaytracer 联合测试")
    print("=" * 70)
    
    # ==========================================================================
    # 测试参数（使用非常平坦的长焦配置）
    # ==========================================================================
    
    # 光学参数
    # 为了避免相位包裹，需要满足：
    # focal_distance > (physical_size/2)^2 / wavelength
    # 对于 physical_size=20mm, wavelength=0.55μm=0.00055mm:
    # focal_distance > 100^2 / 0.00055 = 18181818 mm
    # 这太大了！所以我们需要减小光瞳尺寸
    
    # 使用更小的光瞳和更长的焦距
    physical_size = 2.0  # mm，小光瞳直径
    focal_distance = 10000.0  # mm，非常长的焦距
    mirror_radius = 2 * focal_distance  # 曲率半径 = 2 * 焦距
    wavelength = 0.55  # μm
    
    # 检查相位包裹条件
    wavelength_mm = wavelength * 1e-3
    min_focal_for_no_wrap = (physical_size / 2) ** 2 / wavelength_mm
    print(f"\n相位包裹检查:")
    print(f"  避免相位包裹的最小焦距: {min_focal_for_no_wrap:.2f} mm")
    print(f"  实际焦距: {focal_distance:.2f} mm")
    if focal_distance > min_focal_for_no_wrap:
        print(f"  ✓ 焦距足够长，不会发生相位包裹")
    else:
        print(f"  ✗ 焦距不够长，可能发生相位包裹！")
    
    # 计算 NA（数值孔径）
    na = (physical_size / 2) / focal_distance
    print(f"\n光学参数:")
    print(f"  焦距: {focal_distance} mm")
    print(f"  曲率半径: {mirror_radius} mm")
    print(f"  光瞳直径: {physical_size} mm")
    print(f"  波长: {wavelength} μm")
    print(f"  数值孔径 NA: {na:.4f} (非常小，波前很平坦)")
    
    # 采样参数
    grid_size = 512
    num_rays = 200
    
    print(f"\n采样参数:")
    print(f"  网格大小: {grid_size} x {grid_size}")
    print(f"  光线数量: {num_rays}")
    
    # ==========================================================================
    # 步骤 1：创建球面波波前
    # ==========================================================================
    
    print("\n步骤 1: 创建球面波波前...")
    
    wavefront = create_spherical_wave_from_focus(
        grid_size=grid_size,
        physical_size=physical_size,
        focal_distance=focal_distance,
        wavelength=wavelength,
    )
    
    # 提取相位
    input_phase = np.angle(wavefront)
    input_opd_waves = input_phase / (2 * np.pi)
    
    print(f"  输入波前相位范围: [{np.min(input_phase):.4f}, {np.max(input_phase):.4f}] rad")
    print(f"  输入波前 OPD 范围: [{np.min(input_opd_waves):.4f}, {np.max(input_opd_waves):.4f}] waves")
    
    # ==========================================================================
    # 步骤 2：使用 WavefrontToRaysSampler 将波前转换为光线
    # ==========================================================================
    
    print("\n步骤 2: 将波前转换为光线...")
    
    sampler = WavefrontToRaysSampler(
        wavefront_amplitude=wavefront,
        physical_size=physical_size,
        wavelength=wavelength,
        num_rays=num_rays,
        distribution='hexapolar',
    )
    
    # 获取采样后的光线
    input_rays = sampler.get_output_rays()
    x_sampled, y_sampled = sampler.get_ray_positions()
    L_sampled, M_sampled, N_sampled = sampler.get_ray_directions()
    
    print(f"  采样光线数量: {len(x_sampled)}")
    print(f"  光线位置范围 X: [{np.min(x_sampled):.2f}, {np.max(x_sampled):.2f}] mm")
    print(f"  光线位置范围 Y: [{np.min(y_sampled):.2f}, {np.max(y_sampled):.2f}] mm")
    print(f"  方向余弦 L 范围: [{np.min(L_sampled):.6f}, {np.max(L_sampled):.6f}]")
    print(f"  方向余弦 M 范围: [{np.min(M_sampled):.6f}, {np.max(M_sampled):.6f}]")
    print(f"  方向余弦 N 范围: [{np.min(N_sampled):.6f}, {np.max(N_sampled):.6f}]")
    
    # ==========================================================================
    # 步骤 3：使用 ElementRaytracer 进行凹面镜反射追迹
    # ==========================================================================
    
    print("\n步骤 3: 凹面镜反射追迹...")
    
    # 创建凹面镜
    mirror = create_concave_mirror_for_spherical_wave(
        source_distance=focal_distance,
        semi_aperture=physical_size / 2 + 5,  # 稍大于光瞳
    )
    
    print(f"  凹面镜曲率半径: {mirror.radius} mm")
    print(f"  凹面镜焦距: {mirror.focal_length} mm")
    print(f"  凹面镜半口径: {mirror.semi_aperture} mm")
    
    # 创建光线追迹器
    raytracer = ElementRaytracer(
        surfaces=[mirror],
        wavelength=wavelength,
        chief_ray_direction=(0, 0, 1),
        entrance_position=(0, 0, 0),
    )
    
    # 执行追迹
    output_rays = raytracer.trace(input_rays)
    
    # 获取输出数据
    x_out = np.asarray(output_rays.x)
    y_out = np.asarray(output_rays.y)
    z_out = np.asarray(output_rays.z)
    L_out = np.asarray(output_rays.L)
    M_out = np.asarray(output_rays.M)
    N_out = np.asarray(output_rays.N)
    
    print(f"  输出光线数量: {len(x_out)}")
    print(f"  输出位置范围 X: [{np.min(x_out):.2f}, {np.max(x_out):.2f}] mm")
    print(f"  输出位置范围 Y: [{np.min(y_out):.2f}, {np.max(y_out):.2f}] mm")
    print(f"  输出位置范围 Z: [{np.min(z_out):.4f}, {np.max(z_out):.4f}] mm")
    print(f"  输出方向 L 范围: [{np.min(L_out):.6f}, {np.max(L_out):.6f}]")
    print(f"  输出方向 M 范围: [{np.min(M_out):.6f}, {np.max(M_out):.6f}]")
    print(f"  输出方向 N 范围: [{np.min(N_out):.6f}, {np.max(N_out):.6f}]")
    
    # ==========================================================================
    # 步骤 4：分析结果
    # ==========================================================================
    
    print("\n步骤 4: 分析结果...")
    
    # 获取有效光线掩模
    valid_mask = raytracer.get_valid_ray_mask()
    n_valid = np.sum(valid_mask)
    print(f"  有效光线数量: {n_valid}/{len(valid_mask)}")
    
    # 获取相对 OPD
    opd_waves = raytracer.get_relative_opd_waves()
    valid_opd = opd_waves[valid_mask]
    
    print(f"  输出 OPD 范围: [{np.nanmin(valid_opd):.6f}, {np.nanmax(valid_opd):.6f}] waves")
    print(f"  输出 OPD 均值: {np.nanmean(valid_opd):.6f} waves")
    print(f"  输出 OPD 标准差: {np.nanstd(valid_opd):.6f} waves")
    print(f"  输出 OPD PV: {np.nanmax(valid_opd) - np.nanmin(valid_opd):.6f} waves")
    
    # 理论上，平面波的 OPD 应该是常数（标准差 ≈ 0）
    # 检查是否满足要求
    opd_std = np.nanstd(valid_opd)
    tolerance = 0.01  # 0.01 波长
    
    if opd_std < tolerance:
        print(f"\n✓ 测试通过！OPD 标准差 ({opd_std:.6f}) < 容差 ({tolerance})")
    else:
        print(f"\n✗ 测试失败！OPD 标准差 ({opd_std:.6f}) > 容差 ({tolerance})")
    
    # ==========================================================================
    # 可视化
    # ==========================================================================
    
    print("\n生成可视化图...")
    
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # 坐标网格
    half_size = physical_size / 2.0
    x_coords = np.linspace(-half_size, half_size, grid_size)
    y_coords = np.linspace(-half_size, half_size, grid_size)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # ----- 第一行：输入波前 -----
    
    # 1.1 输入波前相位
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.pcolormesh(X, Y, input_phase, cmap='RdBu_r', shading='auto')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_title('输入波前相位 (rad)')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1)
    
    # 1.2 输入波前 OPD
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.pcolormesh(X, Y, input_opd_waves, cmap='RdBu_r', shading='auto')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title('输入波前 OPD (waves)')
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2)
    
    # 1.3 采样光线位置
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(x_sampled, y_sampled, c='blue', s=5, alpha=0.5)
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Y (mm)')
    ax3.set_title(f'采样光线位置 ({len(x_sampled)} rays)')
    ax3.set_aspect('equal')
    ax3.set_xlim(-half_size * 1.1, half_size * 1.1)
    ax3.set_ylim(-half_size * 1.1, half_size * 1.1)
    ax3.grid(True, alpha=0.3)
    
    # 1.4 采样光线方向（L, M 分量）
    ax4 = fig.add_subplot(gs[0, 3])
    # 用箭头表示方向
    skip = max(1, len(x_sampled) // 100)  # 只显示部分箭头
    ax4.quiver(x_sampled[::skip], y_sampled[::skip], 
               L_sampled[::skip], M_sampled[::skip],
               scale=0.5, scale_units='xy', angles='xy')
    ax4.set_xlabel('X (mm)')
    ax4.set_ylabel('Y (mm)')
    ax4.set_title('采样光线方向 (L, M)')
    ax4.set_aspect('equal')
    ax4.set_xlim(-half_size * 1.1, half_size * 1.1)
    ax4.set_ylim(-half_size * 1.1, half_size * 1.1)
    ax4.grid(True, alpha=0.3)
    
    # ----- 第二行：输出光线 -----
    
    # 2.1 输出光线位置
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.scatter(x_out[valid_mask], y_out[valid_mask], c='green', s=5, alpha=0.5)
    ax5.set_xlabel('X (mm)')
    ax5.set_ylabel('Y (mm)')
    ax5.set_title(f'输出光线位置 ({n_valid} valid rays)')
    ax5.set_aspect('equal')
    ax5.set_xlim(-half_size * 1.1, half_size * 1.1)
    ax5.set_ylim(-half_size * 1.1, half_size * 1.1)
    ax5.grid(True, alpha=0.3)
    
    # 2.2 输出光线方向（L, M 分量）
    ax6 = fig.add_subplot(gs[1, 1])
    valid_indices = np.where(valid_mask)[0]
    skip_valid = max(1, len(valid_indices) // 100)
    idx = valid_indices[::skip_valid]
    ax6.quiver(x_out[idx], y_out[idx], 
               L_out[idx], M_out[idx],
               scale=0.5, scale_units='xy', angles='xy')
    ax6.set_xlabel('X (mm)')
    ax6.set_ylabel('Y (mm)')
    ax6.set_title('输出光线方向 (L, M)')
    ax6.set_aspect('equal')
    ax6.set_xlim(-half_size * 1.1, half_size * 1.1)
    ax6.set_ylim(-half_size * 1.1, half_size * 1.1)
    ax6.grid(True, alpha=0.3)
    
    # 2.3 输出 OPD 分布
    ax7 = fig.add_subplot(gs[1, 2])
    scatter7 = ax7.scatter(x_out[valid_mask], y_out[valid_mask], 
                           c=valid_opd, cmap='RdBu_r', s=10)
    ax7.set_xlabel('X (mm)')
    ax7.set_ylabel('Y (mm)')
    ax7.set_title(f'输出 OPD (waves)\nstd={opd_std:.6f}')
    ax7.set_aspect('equal')
    ax7.set_xlim(-half_size * 1.1, half_size * 1.1)
    ax7.set_ylim(-half_size * 1.1, half_size * 1.1)
    plt.colorbar(scatter7, ax=ax7)
    
    # 2.4 输出 OPD 直方图
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.hist(valid_opd, bins=50, edgecolor='black', alpha=0.7)
    ax8.axvline(np.nanmean(valid_opd), color='r', linestyle='--', 
                label=f'均值: {np.nanmean(valid_opd):.4f}')
    ax8.set_xlabel('OPD (waves)')
    ax8.set_ylabel('频数')
    ax8.set_title(f'输出 OPD 分布\nPV={np.nanmax(valid_opd)-np.nanmin(valid_opd):.4f}')
    ax8.legend()
    
    # ----- 第三行：详细分析 -----
    
    # 3.1 输出 N 分量分布（应该接近 -1，表示反射后向 -Z 方向）
    ax9 = fig.add_subplot(gs[2, 0])
    ax9.hist(N_out[valid_mask], bins=50, edgecolor='black', alpha=0.7)
    ax9.axvline(np.mean(N_out[valid_mask]), color='r', linestyle='--',
                label=f'均值: {np.mean(N_out[valid_mask]):.6f}')
    ax9.set_xlabel('N (方向余弦)')
    ax9.set_ylabel('频数')
    ax9.set_title('输出光线 N 分量分布\n(平面波应接近 -1)')
    ax9.legend()
    
    # 3.2 输出 Z 位置分布
    ax10 = fig.add_subplot(gs[2, 1])
    ax10.hist(z_out[valid_mask], bins=50, edgecolor='black', alpha=0.7)
    ax10.set_xlabel('Z (mm)')
    ax10.set_ylabel('频数')
    ax10.set_title('输出光线 Z 位置分布')
    
    # 3.3 OPD vs 半径
    ax11 = fig.add_subplot(gs[2, 2])
    r_out = np.sqrt(x_out**2 + y_out**2)
    ax11.scatter(r_out[valid_mask], valid_opd, s=5, alpha=0.5)
    ax11.set_xlabel('半径 r (mm)')
    ax11.set_ylabel('OPD (waves)')
    ax11.set_title('OPD vs 半径\n(平面波应为水平线)')
    ax11.grid(True, alpha=0.3)
    
    # 3.4 原始 OPD 数据
    ax12 = fig.add_subplot(gs[2, 3])
    raw_opd = np.asarray(output_rays.opd)
    ax12.hist(raw_opd[valid_mask], bins=50, edgecolor='black', alpha=0.7)
    ax12.set_xlabel('原始 OPD (mm)')
    ax12.set_ylabel('频数')
    ax12.set_title('原始 OPD 分布 (optiland 输出)')
    
    plt.suptitle(f'球面波 → 凹面镜 → 平面波 验证测试\n'
                 f'焦距={focal_distance}mm, NA={na:.4f}, '
                 f'OPD std={opd_std:.6f} waves',
                 fontsize=14, fontweight='bold')
    
    plt.savefig('combined_wavefront_raytracer_test.png', dpi=150, bbox_inches='tight')
    print(f"\n可视化结果已保存到: combined_wavefront_raytracer_test.png")
    plt.show()
    
    return opd_std < tolerance


if __name__ == "__main__":
    visualize_combined_test()
