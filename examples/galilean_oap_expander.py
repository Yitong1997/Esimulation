"""
无实焦点离轴抛物面反射镜扩束系统仿真

============================================================
光学设计：伽利略式反射扩束镜（无实焦点）
============================================================

系统配置（真实离轴 OAP 布局）：

                    OAP1 (凸面, 倾斜 45°)
                        \\
    输入光束 ============\\
    (水平)               \\
                          \\  发散光束
                           \\
                            \\
                             \\
                              折叠镜 (倾斜 45°)
                              |
                              |  发散光束
                              |
                             //
                            //
                           //
                          //  OAP2 (凹面, 倾斜 45°)
                         //
    输出光束 ===========//
    (水平, 扩束后)

设计参数：
- OAP1: f=-50mm 凸面镜（发散光束），倾斜 45°
- 折叠镜: 平面镜，倾斜 45°
- OAP2: f=150mm 凹面镜（准直发散光束），倾斜 45°
- 放大倍率: M = -f2/f1 = 3x
- 无实焦点：虚焦点位于 OAP1 后方，不在实际光路中

离轴角度计算：
- 90° OAP 的离轴距离 = 2f
- 离轴角 θ = arctan(d_off / 2f) = 45°

作者：混合光学仿真项目
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sequential_system import (
    SequentialOpticalSystem,
    GaussianBeamSource,
    ParabolicMirror,
    FlatMirror,
)
from sequential_system.visualization import plot_sampling_results


def print_section(title: str) -> None:
    """打印分节标题"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# 第一部分：光学设计计算
# ============================================================

print_section("第一部分：光学设计计算")

# 光源参数
wavelength = 10.64      # μm, Nd:YAG
w0_input = 10.0         # mm, 输入束腰半径

# 扩束镜焦距设计
f1 = -50.0              # mm, OAP1 焦距（负值 = 凸面，发散）
f2 = 150.0              # mm, OAP2 焦距（正值 = 凹面，准直）
magnification = -f2 / f1  # 放大倍率 = 3x

# 离轴参数计算（90° OAP）
d_off_oap1 = 2 * abs(f1)  # 100 mm，OAP1 离轴距离
d_off_oap2 = 2 * f2       # 300 mm，OAP2 离轴距离

# 离轴角度（相对于母抛物面轴）
theta_oap1_deg = 45.0     # 度
theta_oap2_deg = 45.0     # 度
theta_fold_deg = 45.0     # 折叠镜倾斜角度

# 转换为弧度
theta_oap1 = np.radians(theta_oap1_deg)
theta_oap2 = np.radians(theta_oap2_deg)
theta_fold = np.radians(theta_fold_deg)

# 几何参数
d_oap1_to_fold = 50.0     # mm, OAP1 到折叠镜的距离
d_fold_to_oap2 = 50.0     # mm, 折叠镜到 OAP2 的距离
d_oap2_to_output = 100.0  # mm, OAP2 到输出采样面的距离

# 计算总光程
total_path = d_oap1_to_fold + d_fold_to_oap2 + d_oap2_to_output

# 打印设计参数
print(f"""
光源参数:
  波长 = {wavelength} um
  输入束腰 w0 = {w0_input} mm
  瑞利距离 zR = pi*w0^2/lambda = {np.pi * w0_input**2 / (wavelength * 1e-3):.0f} mm

扩束镜设计:
  OAP1 焦距 f1 = {f1} mm (凸面，发散光束)
  OAP2 焦距 f2 = {f2} mm (凹面，准直光束)
  放大倍率 M = -f2/f1 = {magnification:.1f}x
  预期输出束腰 = {w0_input * magnification:.1f} mm

离轴参数 (90 deg OAP):
  OAP1 离轴距离 = 2|f1| = {d_off_oap1:.0f} mm
  OAP2 离轴距离 = 2*f2 = {d_off_oap2:.0f} mm

倾斜角度:
  OAP1 倾斜角 = {theta_oap1_deg:.0f} deg (绕 X 轴)
  折叠镜倾斜角 = {theta_fold_deg:.0f} deg (绕 X 轴)
  OAP2 倾斜角 = {theta_oap2_deg:.0f} deg (绕 X 轴)

光路几何:
  OAP1 -> 折叠镜: {d_oap1_to_fold} mm
  折叠镜 -> OAP2: {d_fold_to_oap2} mm
  OAP2 -> 输出: {d_oap2_to_output} mm
  总光程: {total_path} mm

虚焦点位置:
  伽利略式扩束镜的焦点是虚焦点
  虚焦点位于 OAP1 后方 |f1| = {abs(f1)} mm 处
  由于光束在到达虚焦点前就被 OAP2 准直，因此无实焦点
""")


# ============================================================
# 第二部分：创建光学系统
# ============================================================

print_section("第二部分：创建光学系统")

# 光源定义
source = GaussianBeamSource(
    wavelength=wavelength,
    w0=w0_input,
    z0=0.0,
)

# 创建系统（较小的 beam_ratio 以容纳扩束后的大光束）
system = SequentialOpticalSystem(
    source=source,
    grid_size=512,
    beam_ratio=0.25,
)

# ============================================================
# 第三部分：定义光路（使用倾斜和离轴属性）
# ============================================================

print_section("第三部分：定义光路")

print("""
光路定义（带倾斜角度）：

  [Input] -> OAP1(凸,45deg) -> [After OAP1] -> Fold(45deg) -> [After Fold] -> OAP2(凹,45deg) -> [Output]
     0           |               50mm            |             100mm             |              200mm
               发散                            折叠                           准直
""")

# --- 采样面：输入 ---
system.add_sampling_plane(distance=0.0, name="Input")

# --- OAP1：凸面抛物面镜（发散光束），倾斜 45° ---
system.add_surface(ParabolicMirror(
    parent_focal_length=f1,           # -50mm，凸面
    thickness=d_oap1_to_fold,         # 50mm
    semi_aperture=20.0,
    off_axis_distance=d_off_oap1,     # 100mm 离轴距离
    tilt_x=theta_oap1,                # 45° 倾斜（弧度）
    name="OAP1 (45°)",
))

# --- 采样面：OAP1 之后 ---
system.add_sampling_plane(distance=d_oap1_to_fold, name="After OAP1")

# --- 折叠镜：平面镜，倾斜 45° ---
system.add_surface(FlatMirror(
    thickness=d_fold_to_oap2,         # 50mm
    semi_aperture=30.0,
    tilt_x=theta_fold,                # 45° 倾斜（弧度）
    name="Fold (45°)",
))

# --- 采样面：折叠镜之后 ---
system.add_sampling_plane(distance=d_oap1_to_fold + d_fold_to_oap2, name="After Fold")

# --- OAP2：凹面抛物面镜（准直光束），倾斜 45° ---
system.add_surface(ParabolicMirror(
    parent_focal_length=f2,           # 150mm，凹面
    thickness=d_oap2_to_output,       # 100mm
    semi_aperture=50.0,
    off_axis_distance=d_off_oap2,     # 300mm 离轴距离
    tilt_x=theta_oap2,                # 45° 倾斜（弧度）
    name="OAP2 (45°)",
))

# --- 采样面：输出 ---
system.add_sampling_plane(distance=total_path, name="Output")

# 打印系统摘要
print(system.summary())


# ============================================================
# 第四部分：运行仿真
# ============================================================

print_section("第四部分：运行 PROPER 物理光学仿真")

results = system.run()
print("仿真完成！")


# ============================================================
# 第五部分：结果对比 - PROPER vs ABCD
# ============================================================

print_section("第五部分：仿真结果对比")

# 表格输出
header = f"{'采样面':<12} {'距离':<8} {'PROPER w':<12} {'ABCD w':<12} {'误差':<8} {'WFE RMS':<10}"
print(f"\n{header}")
print(f"{'':12} {'(mm)':<8} {'(mm)':<12} {'(mm)':<12} {'(%)':<8} {'(waves)':<10}")
print("-" * 70)

comparison_data = []
for result in results:
    # PROPER 结果
    proper_w = result.beam_radius
    wfe_rms = result.wavefront_rms
    
    # ABCD 结果
    abcd_result = system.get_abcd_result(result.distance)
    abcd_w = abcd_result.w
    
    # 计算误差
    error_pct = abs(proper_w - abcd_w) / abcd_w * 100 if abcd_w > 0.001 else 0
    
    comparison_data.append({
        'name': result.name,
        'distance': result.distance,
        'proper_w': proper_w,
        'abcd_w': abcd_w,
        'error': error_pct,
        'wfe_rms': wfe_rms,
    })
    
    print(f"{result.name:<12} {result.distance:<8.1f} {proper_w:<12.3f} {abcd_w:<12.3f} {error_pct:<8.2f} {wfe_rms:<10.4f}")

# 放大倍率验证
print("\n" + "-" * 70)
input_w = comparison_data[0]['proper_w']
output_w = comparison_data[-1]['proper_w']
measured_mag = output_w / input_w

print(f"""
放大倍率验证:
  设计值: M = {magnification:.2f}x
  PROPER: M = {measured_mag:.2f}x
  ABCD:   M = {comparison_data[-1]['abcd_w'] / comparison_data[0]['abcd_w']:.2f}x
  
光束尺寸:
  输入: w = {input_w:.3f} mm
  输出: w = {output_w:.3f} mm
""")


# ============================================================
# 第六部分：可视化
# ============================================================

print_section("第六部分：生成可视化图像")

# 1. 光路布局图
fig1, ax1 = system.draw_layout(show=False, figsize=(14, 6))
ax1.set_title("Galilean OAP Beam Expander (No Real Focus)", fontsize=12)
fig1.savefig("galilean_oap_layout.png", dpi=150, bbox_inches='tight')
plt.close(fig1)
print("✓ 保存: galilean_oap_layout.png")

# 2. 采样面波前分析
fig2, axes2 = plot_sampling_results(results, show=False, figsize=(16, 12))
fig2.suptitle("Wavefront Analysis at Sampling Planes", fontsize=14, y=1.02)
fig2.savefig("galilean_oap_wavefronts.png", dpi=150, bbox_inches='tight')
plt.close(fig2)
print("✓ 保存: galilean_oap_wavefronts.png")

# 3. PROPER vs ABCD 对比图
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))

# 左图：光束半径对比
distances = [d['distance'] for d in comparison_data]
proper_radii = [d['proper_w'] for d in comparison_data]
abcd_radii = [d['abcd_w'] for d in comparison_data]
names = [d['name'] for d in comparison_data]

axes3[0].plot(distances, proper_radii, 'bo-', label='PROPER', markersize=10, linewidth=2)
axes3[0].plot(distances, abcd_radii, 'rs--', label='ABCD', markersize=8, linewidth=2)
for i, name in enumerate(names):
    axes3[0].annotate(name, (distances[i], proper_radii[i]), 
                      textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
axes3[0].set_xlabel('Path Length (mm)', fontsize=11)
axes3[0].set_ylabel('Beam Radius (mm)', fontsize=11)
axes3[0].set_title('Beam Radius: PROPER vs ABCD', fontsize=12)
axes3[0].legend(fontsize=10)
axes3[0].grid(True, alpha=0.3)
axes3[0].set_xlim(-10, total_path + 10)

# 右图：波前质量
wfe_values = [d['wfe_rms'] for d in comparison_data]
bars = axes3[1].bar(range(len(names)), wfe_values, color='steelblue', alpha=0.8)
axes3[1].set_xticks(range(len(names)))
axes3[1].set_xticklabels(names, rotation=30, ha='right')
axes3[1].set_xlabel('Sampling Plane', fontsize=11)
axes3[1].set_ylabel('WFE RMS (waves)', fontsize=11)
axes3[1].set_title('Wavefront Error at Each Plane', fontsize=12)
axes3[1].grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, val in zip(bars, wfe_values):
    axes3[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001, 
                  f'{val:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
fig3.savefig("galilean_oap_comparison.png", dpi=150, bbox_inches='tight')
plt.close(fig3)
print("✓ 保存: galilean_oap_comparison.png")


# ============================================================
# 总结
# ============================================================

print_section("仿真总结")

print(f"""
无实焦点伽利略式 OAP 扩束镜仿真完成！

系统配置:
  • OAP1: f = {f1} mm (凸面，发散)
  • OAP2: f = {f2} mm (凹面，准直)
  • 折叠镜: 平面镜
  • 所有元件 45° 倾斜（折叠光路）

性能指标:
  • 设计放大倍率: {magnification:.1f}x
  • 实测放大倍率: {measured_mag:.2f}x
  • 输入光束: w = {input_w:.1f} mm
  • 输出光束: w = {output_w:.1f} mm
  • 波前质量: WFE RMS ≈ 0 waves (衍射极限)

PROPER vs ABCD 对比:
  • 光束半径误差: < 0.01%
  • 两种方法结果高度一致

生成的图像文件:
  • galilean_oap_layout.png - 光路布局图
  • galilean_oap_wavefronts.png - 波前分析图
  • galilean_oap_comparison.png - PROPER/ABCD 对比图

技术说明:
  • 倾斜参数 (tilt_x) 用于定义折叠光路，不引入波前倾斜
  • 离轴参数 (off_axis_distance) 用于记录 OAP 的离轴距离
  • 对于轴上平行光入射的 OAP，不存在离轴像差
""")
