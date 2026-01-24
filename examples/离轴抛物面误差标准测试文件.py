"""
离轴抛物面误差标准测试文件

测试在不同初始束腰位置的情境中（包括近场与远场情景），
高斯光束入射至具有不同离轴量与曲率半径的离轴抛物镜时，
仿真结果相对于 Pilot Beam 理论相位的精度差异（PV, RMS）。

离轴抛物面（OAP）的正确定义：
- 抛物面沿 Y 方向偏移（off-axis distance）
- 入射光束仍然沿原始光轴方向（Z 轴）入射，无倾斜
- 出射光线方向由抛物面的局部法向量决定

⚠️ 重要说明：
当前 BTS API 的 OpticalSystem.add_surface 方法不支持 off_axis_distance 参数。
本测试文件直接使用底层的 ElementRaytracer 来验证离轴抛物面的处理精度。

测试参数范围：
- 曲率半径：~300 mm 到 ~2000 mm
- 离轴量：不同的离轴距离（d/f = 0.5, 1.0, 2.0）
- 束腰位置：近场（束腰在镜面附近）和远场（束腰远离镜面）

输出：
- 振幅误差（PV, RMS）
- 相位误差（PV, RMS）单位：milli-waves
- 误差分析和可能原因

作者：混合光学仿真项目
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from wavefront_to_rays.element_raytracer import (
    ElementRaytracer,
    SurfaceDefinition,
)
from hybrid_optical_propagation.data_models import PilotBeamParams
from optiland.rays import RealRays


# ============================================================
# 数据结构定义
# ============================================================

@dataclass
class TestCase:
    """测试用例定义
    
    离轴抛物面参数说明：
    - radius_mm: 母抛物面的顶点曲率半径 R = 2f
    - off_axis_distance_mm: 离轴距离（Y 方向偏移）
    - 无倾斜角度！入射光沿 Z 轴方向
    """
    name: str
    radius_mm: float           # 曲率半径 (mm)
    off_axis_distance_mm: float  # 离轴距离 (mm)，Y 方向偏移
    w0_mm: float               # 束腰半径 (mm)
    z_rayleigh_mm: float       # 瑞利距离 (mm)
    wavelength_um: float       # 波长 (μm)
    is_near_field: bool        # 是否为近场情景


@dataclass
class TestResult:
    """测试结果"""
    test_case: TestCase
    opd_pv_mwaves: float       # OPD PV (milli-waves)
    opd_rms_mwaves: float      # OPD RMS (milli-waves)
    exit_angle_deg: float      # 出射角度（度）
    success: bool              # 测试是否成功
    error_message: str = ""    # 错误信息


# ============================================================
# 测试用例生成
# ============================================================

def generate_test_cases() -> List[TestCase]:
    """生成测试用例列表
    
    离轴抛物面的关键参数：
    - 曲率半径 R = 2f（f 为焦距）
    - 离轴距离 d：光束中心相对于母抛物面光轴的偏移
    - 出射角度 θ = arctan(d / f) = arctan(2d / R)
    
    注意：不引入任何表面倾斜！
    """
    test_cases = []
    wavelength_um = 0.633  # He-Ne 激光
    
    # 曲率半径范围：300 mm 到 2000 mm
    radii = [300.0, 500.0, 800.0, 1200.0, 2000.0]
    
    # 离轴距离比例（相对于焦距）
    # off_axis = ratio * f = ratio * R/2
    off_axis_ratios = [0.5, 1.0, 2.0]  # 对应出射角约 27°, 45°, 63°
    
    for radius in radii:
        focal_length = radius / 2  # 抛物面焦距
        
        for ratio in off_axis_ratios:
            off_axis = ratio * focal_length
            
            # 束腰半径（根据曲率半径调整，确保光束不会太大）
            w0 = min(3.0, radius / 200)
            
            # 瑞利距离 z_R = π * w0² / λ
            z_R = np.pi * w0**2 / (wavelength_um * 1e-3)
            
            # 添加近场测试用例
            test_cases.append(TestCase(
                name=f"OAP R={radius:.0f}mm d/f={ratio:.1f} 近场",
                radius_mm=radius,
                off_axis_distance_mm=off_axis,
                w0_mm=w0,
                z_rayleigh_mm=z_R,
                wavelength_um=wavelength_um,
                is_near_field=True,
            ))
            
            # 添加远场测试用例
            test_cases.append(TestCase(
                name=f"OAP R={radius:.0f}mm d/f={ratio:.1f} 远场",
                radius_mm=radius,
                off_axis_distance_mm=off_axis,
                w0_mm=w0,
                z_rayleigh_mm=z_R,
                wavelength_um=wavelength_um,
                is_near_field=False,
            ))
    
    return test_cases


# ============================================================
# 辅助函数
# ============================================================

def create_test_rays(
    num_rays: int,
    beam_radius_mm: float,
    wavelength_um: float,
) -> RealRays:
    """创建测试光线网格
    
    参数:
        num_rays: 每个方向的光线数量
        beam_radius_mm: 光束半径 (mm)
        wavelength_um: 波长 (μm)
    
    返回:
        RealRays 对象
    """
    # 创建均匀网格
    x = np.linspace(-beam_radius_mm, beam_radius_mm, num_rays)
    y = np.linspace(-beam_radius_mm, beam_radius_mm, num_rays)
    xx, yy = np.meshgrid(x, y)
    
    # 只保留圆形区域内的光线
    r = np.sqrt(xx**2 + yy**2)
    mask = r <= beam_radius_mm
    
    x_flat = xx[mask].flatten()
    y_flat = yy[mask].flatten()
    n_rays = len(x_flat)
    
    # 所有光线沿 Z 轴方向（正入射）
    rays = RealRays(
        x=x_flat,
        y=y_flat,
        z=np.zeros(n_rays),
        L=np.zeros(n_rays),
        M=np.zeros(n_rays),
        N=np.ones(n_rays),
        intensity=np.ones(n_rays),
        wavelength=np.full(n_rays, wavelength_um),
    )
    
    # 初始化 OPD 为 0
    rays.opd = np.zeros(n_rays)
    
    return rays


def compute_theoretical_exit_direction(
    radius_mm: float,
    off_axis_distance_mm: float,
) -> tuple:
    """计算理论出射方向
    
    对于离轴抛物面，入射平行光在离轴位置处的反射方向
    由抛物面的局部法向量决定。
    
    抛物面方程（顶点在原点，开口朝 +Z）：z = (x² + y²) / (2R)
    其中 R = 2f 是顶点曲率半径
    
    在 (0, d) 处：
    - z = d² / (2R)
    - 梯度：∇z = (x/R, y/R) = (0, d/R)
    - 表面法向量（指向 -Z 侧，即入射侧）：
      n = (-∂z/∂x, -∂z/∂y, 1) / |n| = (0, -d/R, 1) / sqrt(1 + (d/R)²)
    
    入射方向：(0, 0, 1)
    反射方向：r = i - 2(i·n)n
    
    参数:
        radius_mm: 曲率半径 (mm)
        off_axis_distance_mm: 离轴距离 (mm)
    
    返回:
        (L, M, N) 出射方向余弦
    """
    d = off_axis_distance_mm
    R = radius_mm
    
    # 表面法向量（指向入射侧，即 -Z 方向）
    # 抛物面 z = r²/(2R)
    # 梯度 ∇z = (x/R, y/R)
    # 在 (0, d) 处：∇z = (0, d/R)
    # 法向量（指向 -Z 侧）：n = (-∂z/∂x, -∂z/∂y, 1) / |n| = (0, -d/R, 1) / |n|
    grad_y = d / R
    norm_factor = np.sqrt(1 + grad_y**2)
    nx, ny, nz = 0, -grad_y / norm_factor, 1 / norm_factor
    
    # 入射方向（沿 +Z）
    ix, iy, iz = 0.0, 0.0, 1.0
    
    # 反射公式：r = i - 2(i·n)n
    dot = ix*nx + iy*ny + iz*nz  # = nz = 1/norm_factor
    rx = ix - 2*dot*nx
    ry = iy - 2*dot*ny
    rz = iz - 2*dot*nz
    
    # 归一化（应该已经是归一化的，但为了安全）
    r_norm = np.sqrt(rx**2 + ry**2 + rz**2)
    rx, ry, rz = rx/r_norm, ry/r_norm, rz/r_norm
    
    return (rx, ry, rz)


# ============================================================
# 单个测试用例执行
# ============================================================

def run_single_test(test_case: TestCase, verbose: bool = False) -> TestResult:
    """执行单个测试用例
    
    使用 ElementRaytracer 直接测试离轴抛物面的光线追迹精度。
    
    参数:
        test_case: 测试用例
        verbose: 是否输出详细信息
    
    返回:
        TestResult 对象
    """
    try:
        if verbose:
            print(f"\n{'='*60}")
            print(f"测试: {test_case.name}")
            print(f"{'='*60}")
            print(f"  曲率半径: {test_case.radius_mm:.1f} mm")
            print(f"  离轴距离: {test_case.off_axis_distance_mm:.1f} mm")
            print(f"  束腰半径: {test_case.w0_mm:.3f} mm")
            print(f"  场景类型: {'近场' if test_case.is_near_field else '远场'}")
        
        # 计算理论出射方向
        exit_dir = compute_theoretical_exit_direction(
            test_case.radius_mm,
            test_case.off_axis_distance_mm,
        )
        exit_angle_deg = np.degrees(np.arctan2(
            np.sqrt(exit_dir[0]**2 + exit_dir[1]**2),
            exit_dir[2]
        ))
        
        if verbose:
            print(f"  理论出射角: {exit_angle_deg:.2f}°")
            print(f"  理论出射方向: ({exit_dir[0]:.4f}, {exit_dir[1]:.4f}, {exit_dir[2]:.4f})")
        
        # 创建离轴抛物面定义
        # 关键：使用 off_axis_distance 参数，不使用 tilt_x/tilt_y
        surface_def = SurfaceDefinition(
            surface_type='mirror',
            radius=test_case.radius_mm,
            thickness=0.0,
            material='mirror',
            semi_aperture=test_case.w0_mm * 4,
            conic=-1.0,  # 抛物面
            tilt_x=0.0,  # 无倾斜！
            tilt_y=0.0,  # 无倾斜！
            off_axis_distance=test_case.off_axis_distance_mm,  # 离轴距离
        )
        
        # 创建光线追迹器
        # 入射方向沿 Z 轴（正入射）
        # 不提供 exit_chief_direction，让 ElementRaytracer 使用 optiland 自动计算
        raytracer = ElementRaytracer(
            surfaces=[surface_def],
            wavelength=test_case.wavelength_um,
            chief_ray_direction=(0, 0, 1),  # 正入射
            entrance_position=(0, 0, 0),
        )
        
        # 创建测试光线
        beam_radius = test_case.w0_mm * 2  # 使用 2 倍束腰半径
        input_rays = create_test_rays(
            num_rays=21,  # 21x21 网格
            beam_radius_mm=beam_radius,
            wavelength_um=test_case.wavelength_um,
        )
        
        if verbose:
            print(f"  光线数量: {len(input_rays.x)}")
        
        # 执行光线追迹
        output_rays = raytracer.trace(input_rays)
        
        # 计算 OPD 统计
        opd_mm = np.asarray(output_rays.opd)
        wavelength_mm = test_case.wavelength_um * 1e-3
        opd_waves = opd_mm / wavelength_mm
        
        # 过滤有效光线（强度 > 0）
        valid_mask = np.asarray(output_rays.i) > 0.01
        if np.sum(valid_mask) < 10:
            return TestResult(
                test_case=test_case,
                opd_pv_mwaves=np.nan,
                opd_rms_mwaves=np.nan,
                exit_angle_deg=exit_angle_deg,
                success=False,
                error_message="有效光线数量不足",
            )
        
        opd_valid = opd_waves[valid_mask]
        
        # 计算 PV 和 RMS（相对于主光线，即中心光线）
        # 对于理想离轴抛物面，所有光线的 OPD 应该相同（聚焦到焦点）
        opd_pv = np.ptp(opd_valid)
        opd_rms = np.std(opd_valid)
        
        # 转换为 milli-waves
        opd_pv_mwaves = opd_pv * 1000
        opd_rms_mwaves = opd_rms * 1000
        
        if verbose:
            print(f"\n  结果:")
            print(f"    OPD PV: {opd_pv_mwaves:.3f} milli-waves")
            print(f"    OPD RMS: {opd_rms_mwaves:.3f} milli-waves")
        
        return TestResult(
            test_case=test_case,
            opd_pv_mwaves=opd_pv_mwaves,
            opd_rms_mwaves=opd_rms_mwaves,
            exit_angle_deg=exit_angle_deg,
            success=True,
        )
        
    except Exception as e:
        if verbose:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
        return TestResult(
            test_case=test_case,
            opd_pv_mwaves=np.nan,
            opd_rms_mwaves=np.nan,
            exit_angle_deg=0.0,
            success=False,
            error_message=str(e),
        )


# ============================================================
# 结果可视化
# ============================================================

def plot_results(results: List[TestResult], save_path: str = None):
    """绘制测试结果
    
    参数:
        results: 测试结果列表
        save_path: 保存路径（可选）
    """
    # 过滤成功的结果
    successful = [r for r in results if r.success]
    
    if not successful:
        print("没有成功的测试结果可供绘制")
        return
    
    # 提取数据
    names = [r.test_case.name for r in successful]
    radii = [r.test_case.radius_mm for r in successful]
    off_axis = [r.test_case.off_axis_distance_mm for r in successful]
    opd_pv = [r.opd_pv_mwaves for r in successful]
    opd_rms = [r.opd_rms_mwaves for r in successful]
    exit_angles = [r.exit_angle_deg for r in successful]
    is_near_field = [r.test_case.is_near_field for r in successful]
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 图1：OPD RMS vs 出射角度
    ax1 = axes[0, 0]
    near_mask = np.array(is_near_field)
    far_mask = ~near_mask
    
    ax1.scatter(np.array(exit_angles)[near_mask], np.array(opd_rms)[near_mask], 
                c='blue', marker='o', label='近场', alpha=0.7)
    ax1.scatter(np.array(exit_angles)[far_mask], np.array(opd_rms)[far_mask], 
                c='red', marker='s', label='远场', alpha=0.7)
    ax1.set_xlabel('出射角度 (°)')
    ax1.set_ylabel('OPD RMS (milli-waves)')
    ax1.set_title('OPD RMS vs 出射角度')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 图2：OPD PV vs 出射角度
    ax2 = axes[0, 1]
    ax2.scatter(np.array(exit_angles)[near_mask], np.array(opd_pv)[near_mask], 
                c='blue', marker='o', label='近场', alpha=0.7)
    ax2.scatter(np.array(exit_angles)[far_mask], np.array(opd_pv)[far_mask], 
                c='red', marker='s', label='远场', alpha=0.7)
    ax2.set_xlabel('出射角度 (°)')
    ax2.set_ylabel('OPD PV (milli-waves)')
    ax2.set_title('OPD PV vs 出射角度')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 图3：OPD RMS vs 曲率半径
    ax3 = axes[1, 0]
    ax3.scatter(np.array(radii)[near_mask], np.array(opd_rms)[near_mask], 
                c='blue', marker='o', label='近场', alpha=0.7)
    ax3.scatter(np.array(radii)[far_mask], np.array(opd_rms)[far_mask], 
                c='red', marker='s', label='远场', alpha=0.7)
    ax3.set_xlabel('曲率半径 (mm)')
    ax3.set_ylabel('OPD RMS (milli-waves)')
    ax3.set_title('OPD RMS vs 曲率半径')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 图4：OPD RMS vs 离轴距离
    ax4 = axes[1, 1]
    ax4.scatter(np.array(off_axis)[near_mask], np.array(opd_rms)[near_mask], 
                c='blue', marker='o', label='近场', alpha=0.7)
    ax4.scatter(np.array(off_axis)[far_mask], np.array(opd_rms)[far_mask], 
                c='red', marker='s', label='远场', alpha=0.7)
    ax4.set_xlabel('离轴距离 (mm)')
    ax4.set_ylabel('OPD RMS (milli-waves)')
    ax4.set_title('OPD RMS vs 离轴距离')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    
    plt.show()


def print_summary(results: List[TestResult]):
    """打印测试结果摘要
    
    参数:
        results: 测试结果列表
    """
    print("\n" + "="*80)
    print("测试结果摘要")
    print("="*80)
    
    # 统计
    total = len(results)
    successful = sum(1 for r in results if r.success)
    failed = total - successful
    
    print(f"\n总测试数: {total}")
    print(f"成功: {successful}")
    print(f"失败: {failed}")
    
    if successful > 0:
        # 成功测试的统计
        opd_rms_values = [r.opd_rms_mwaves for r in results if r.success]
        opd_pv_values = [r.opd_pv_mwaves for r in results if r.success]
        
        print(f"\nOPD RMS 统计 (milli-waves):")
        print(f"  最小值: {min(opd_rms_values):.3f}")
        print(f"  最大值: {max(opd_rms_values):.3f}")
        print(f"  平均值: {np.mean(opd_rms_values):.3f}")
        
        print(f"\nOPD PV 统计 (milli-waves):")
        print(f"  最小值: {min(opd_pv_values):.3f}")
        print(f"  最大值: {max(opd_pv_values):.3f}")
        print(f"  平均值: {np.mean(opd_pv_values):.3f}")
    
    # 打印详细结果表格
    print("\n" + "-"*80)
    print(f"{'测试用例':<40} {'出射角(°)':<10} {'RMS(mw)':<12} {'PV(mw)':<12} {'状态':<8}")
    print("-"*80)
    
    for r in results:
        status = "✓" if r.success else "✗"
        if r.success:
            print(f"{r.test_case.name:<40} {r.exit_angle_deg:<10.1f} "
                  f"{r.opd_rms_mwaves:<12.3f} {r.opd_pv_mwaves:<12.3f} {status:<8}")
        else:
            print(f"{r.test_case.name:<40} {'-':<10} {'-':<12} {'-':<12} {status:<8}")
            if r.error_message:
                print(f"  错误: {r.error_message}")
    
    print("-"*80)


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数"""
    print("="*80)
    print("离轴抛物面误差标准测试")
    print("="*80)
    print("\n测试目的：验证 ElementRaytracer 对离轴抛物面的处理精度")
    print("测试方法：使用 optiland 的实际光线追迹计算主光线出射方向")
    print()
    
    # 生成测试用例
    test_cases = generate_test_cases()
    print(f"生成了 {len(test_cases)} 个测试用例")
    
    # 执行测试
    results = []
    for i, test_case in enumerate(test_cases):
        print(f"\r执行测试 {i+1}/{len(test_cases)}: {test_case.name[:30]}...", end="")
        result = run_single_test(test_case, verbose=False)
        results.append(result)
    
    print("\n")
    
    # 打印摘要
    print_summary(results)
    
    # 绘制结果
    save_path = os.path.join(os.path.dirname(__file__), '离轴抛物面误差测试结果.png')
    plot_results(results, save_path)
    
    return results


if __name__ == "__main__":
    results = main()
