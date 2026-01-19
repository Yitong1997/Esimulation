"""伽利略式 OAP 扩束镜：混合光学追迹与 ABCD 方法对比测试

本模块全面测试混合光学追迹与 ABCD 矩阵方法在伽利略 OAP 扩束器上的结果对比。

测试内容：
1. 多个采样面的波前相位和光斑尺寸对比
2. 不同光线数量对结果的影响
3. 不同复振幅采样面积尺寸对结果的影响
4. 网格大小对结果的影响

作者：混合光学仿真项目
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import warnings

sys.path.insert(0, 'src')

from sequential_system import (
    SequentialOpticalSystem,
    GaussianBeamSource,
    ParabolicMirror,
    FlatMirror,
)


@dataclass
class ComparisonResult:
    """单个采样面的对比结果"""
    name: str
    distance: float
    proper_beam_radius: float
    abcd_beam_radius: float
    beam_radius_error_pct: float
    wavefront_rms: float
    wavefront_pv: float
    proper_R: float  # PROPER 波前曲率半径
    abcd_R: float    # ABCD 波前曲率半径


@dataclass
class ParameterStudyResult:
    """参数研究结果"""
    parameter_name: str
    parameter_values: List
    input_beam_radii: List[float]
    output_beam_radii: List[float]
    magnifications: List[float]
    wfe_rms_values: List[List[float]]  # 每个参数值对应各采样面的 WFE
    sampling_plane_names: List[str]


class GalileanOAPComparison:
    """伽利略式 OAP 扩束镜对比测试类"""
    
    # 系统参数
    WAVELENGTH_UM = 10.64      # μm, CO2 激光
    W0_INPUT_MM = 10.0         # mm, 输入束腰半径
    F1_MM = -50.0              # mm, OAP1 焦距（凸面）
    F2_MM = 150.0              # mm, OAP2 焦距（凹面）
    DESIGN_MAGNIFICATION = 3.0
    
    # 离轴参数
    D_OFF_OAP1_MM = 100.0
    D_OFF_OAP2_MM = 300.0
    TILT_45_DEG = np.pi / 4
    
    # 几何参数
    D_OAP1_TO_FOLD_MM = 50.0
    D_FOLD_TO_OAP2_MM = 50.0
    D_OAP2_TO_OUTPUT_MM = 100.0
    
    def __init__(self):
        self.total_path = (
            self.D_OAP1_TO_FOLD_MM + 
            self.D_FOLD_TO_OAP2_MM + 
            self.D_OAP2_TO_OUTPUT_MM
        )
    
    def create_system(
        self,
        grid_size: int = 512,
        beam_ratio: float = 0.25,
        hybrid_num_rays: int = 100,
        use_hybrid: bool = True,
    ) -> SequentialOpticalSystem:
        """创建伽利略式扩束镜系统"""
        source = GaussianBeamSource(
            wavelength=self.WAVELENGTH_UM,
            w0=self.W0_INPUT_MM,
            z0=0.0,
        )
        
        system = SequentialOpticalSystem(
            source=source,
            grid_size=grid_size,
            beam_ratio=beam_ratio,
            use_hybrid_propagation=use_hybrid,
            hybrid_num_rays=hybrid_num_rays,
        )
        
        # 添加多个采样面
        system.add_sampling_plane(distance=0.0, name="Input")
        
        # OAP1
        system.add_surface(ParabolicMirror(
            parent_focal_length=self.F1_MM,
            thickness=self.D_OAP1_TO_FOLD_MM,
            semi_aperture=20.0,
            off_axis_distance=self.D_OFF_OAP1_MM,
            tilt_x=self.TILT_45_DEG,
            name="OAP1",
        ))
        system.add_sampling_plane(distance=self.D_OAP1_TO_FOLD_MM, name="After_OAP1")
        
        # 折叠镜
        system.add_surface(FlatMirror(
            thickness=self.D_FOLD_TO_OAP2_MM,
            semi_aperture=30.0,
            tilt_x=self.TILT_45_DEG,
            name="Fold",
        ))
        d_after_fold = self.D_OAP1_TO_FOLD_MM + self.D_FOLD_TO_OAP2_MM
        system.add_sampling_plane(distance=d_after_fold, name="After_Fold")
        
        # OAP2
        system.add_surface(ParabolicMirror(
            parent_focal_length=self.F2_MM,
            thickness=self.D_OAP2_TO_OUTPUT_MM,
            semi_aperture=50.0,
            off_axis_distance=self.D_OFF_OAP2_MM,
            tilt_x=self.TILT_45_DEG,
            name="OAP2",
        ))
        system.add_sampling_plane(distance=self.total_path, name="Output")
        
        # 添加额外的中间采样面
        system.add_sampling_plane(distance=self.D_OAP1_TO_FOLD_MM / 2, name="Mid_to_Fold")
        system.add_sampling_plane(distance=self.total_path - 50, name="Before_Output")
        
        return system

    def run_comparison(
        self,
        grid_size: int = 512,
        beam_ratio: float = 0.25,
        hybrid_num_rays: int = 100,
    ) -> Tuple[List[ComparisonResult], Dict]:
        """运行对比测试
        
        返回:
            (comparison_results, summary_dict)
        """
        system = self.create_system(
            grid_size=grid_size,
            beam_ratio=beam_ratio,
            hybrid_num_rays=hybrid_num_rays,
        )
        
        results = system.run()
        comparison_results = []
        
        for name, result in results.sampling_results.items():
            proper_w = result.beam_radius
            abcd_result = system.get_abcd_result(result.distance)
            abcd_w = abcd_result.w
            
            error_pct = abs(proper_w - abcd_w) / abcd_w * 100 if abcd_w > 0.001 else 0
            
            # 获取波前曲率半径
            proper_R = self._estimate_wavefront_curvature(result)
            abcd_R = abcd_result.R
            
            comparison_results.append(ComparisonResult(
                name=name,
                distance=result.distance,
                proper_beam_radius=proper_w,
                abcd_beam_radius=abcd_w,
                beam_radius_error_pct=error_pct,
                wavefront_rms=result.wavefront_rms,
                wavefront_pv=result.wavefront_pv,
                proper_R=proper_R,
                abcd_R=abcd_R,
            ))
        
        # 按距离排序
        comparison_results.sort(key=lambda x: x.distance)
        
        # 计算汇总信息
        input_result = next(r for r in comparison_results if r.name == "Input")
        output_result = next(r for r in comparison_results if r.name == "Output")
        
        summary = {
            'input_beam_radius': input_result.proper_beam_radius,
            'output_beam_radius': output_result.proper_beam_radius,
            'measured_magnification': output_result.proper_beam_radius / input_result.proper_beam_radius,
            'design_magnification': self.DESIGN_MAGNIFICATION,
            'max_beam_radius_error': max(r.beam_radius_error_pct for r in comparison_results),
            'mean_beam_radius_error': np.mean([r.beam_radius_error_pct for r in comparison_results]),
            'max_wfe_rms': max(r.wavefront_rms for r in comparison_results),
        }
        
        return comparison_results, summary
    
    def _estimate_wavefront_curvature(self, result) -> float:
        """从波前数据估计曲率半径"""
        try:
            phase = result.phase
            if phase is None or np.all(np.isnan(phase)):
                return np.inf
            
            # 简单估计：使用边缘相位差
            n = phase.shape[0]
            center = n // 2
            edge_phase = phase[center, -1] if not np.isnan(phase[center, -1]) else 0
            center_phase = phase[center, center] if not np.isnan(phase[center, center]) else 0
            
            if abs(edge_phase - center_phase) < 1e-10:
                return np.inf
            
            # 粗略估计
            return np.inf  # 简化处理
        except:
            return np.inf
    
    def study_ray_count_effect(
        self,
        ray_counts: List[int] = [36, 64, 100, 144, 196, 256],
        grid_size: int = 512,
        beam_ratio: float = 0.25,
    ) -> ParameterStudyResult:
        """研究光线数量对结果的影响"""
        input_radii = []
        output_radii = []
        magnifications = []
        wfe_rms_all = []
        plane_names = None
        
        for num_rays in ray_counts:
            print(f"  测试光线数量: {num_rays}")
            try:
                results, summary = self.run_comparison(
                    grid_size=grid_size,
                    beam_ratio=beam_ratio,
                    hybrid_num_rays=num_rays,
                )
                
                input_radii.append(summary['input_beam_radius'])
                output_radii.append(summary['output_beam_radius'])
                magnifications.append(summary['measured_magnification'])
                
                wfe_rms = [r.wavefront_rms for r in results]
                wfe_rms_all.append(wfe_rms)
                
                if plane_names is None:
                    plane_names = [r.name for r in results]
            except Exception as e:
                print(f"    警告: 光线数量 {num_rays} 测试失败: {e}")
                input_radii.append(np.nan)
                output_radii.append(np.nan)
                magnifications.append(np.nan)
                wfe_rms_all.append([np.nan] * 6)
        
        return ParameterStudyResult(
            parameter_name="光线数量",
            parameter_values=ray_counts,
            input_beam_radii=input_radii,
            output_beam_radii=output_radii,
            magnifications=magnifications,
            wfe_rms_values=wfe_rms_all,
            sampling_plane_names=plane_names or [],
        )

    def study_beam_ratio_effect(
        self,
        beam_ratios: List[float] = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        grid_size: int = 512,
        hybrid_num_rays: int = 100,
    ) -> ParameterStudyResult:
        """研究复振幅采样面积尺寸（beam_ratio）对结果的影响
        
        beam_ratio 决定了 PROPER 中光束直径与网格尺寸的比例。
        较小的 beam_ratio 意味着更大的采样面积（相对于光束）。
        """
        input_radii = []
        output_radii = []
        magnifications = []
        wfe_rms_all = []
        plane_names = None
        
        for br in beam_ratios:
            print(f"  测试 beam_ratio: {br}")
            try:
                results, summary = self.run_comparison(
                    grid_size=grid_size,
                    beam_ratio=br,
                    hybrid_num_rays=hybrid_num_rays,
                )
                
                input_radii.append(summary['input_beam_radius'])
                output_radii.append(summary['output_beam_radius'])
                magnifications.append(summary['measured_magnification'])
                
                wfe_rms = [r.wavefront_rms for r in results]
                wfe_rms_all.append(wfe_rms)
                
                if plane_names is None:
                    plane_names = [r.name for r in results]
            except Exception as e:
                print(f"    警告: beam_ratio {br} 测试失败: {e}")
                input_radii.append(np.nan)
                output_radii.append(np.nan)
                magnifications.append(np.nan)
                wfe_rms_all.append([np.nan] * 6)
        
        return ParameterStudyResult(
            parameter_name="beam_ratio（采样面积）",
            parameter_values=beam_ratios,
            input_beam_radii=input_radii,
            output_beam_radii=output_radii,
            magnifications=magnifications,
            wfe_rms_values=wfe_rms_all,
            sampling_plane_names=plane_names or [],
        )
    
    def study_grid_size_effect(
        self,
        grid_sizes: List[int] = [128, 256, 512, 1024],
        beam_ratio: float = 0.25,
        hybrid_num_rays: int = 100,
    ) -> ParameterStudyResult:
        """研究网格大小对结果的影响"""
        input_radii = []
        output_radii = []
        magnifications = []
        wfe_rms_all = []
        plane_names = None
        
        for gs in grid_sizes:
            print(f"  测试网格大小: {gs}")
            try:
                results, summary = self.run_comparison(
                    grid_size=gs,
                    beam_ratio=beam_ratio,
                    hybrid_num_rays=hybrid_num_rays,
                )
                
                input_radii.append(summary['input_beam_radius'])
                output_radii.append(summary['output_beam_radius'])
                magnifications.append(summary['measured_magnification'])
                
                wfe_rms = [r.wavefront_rms for r in results]
                wfe_rms_all.append(wfe_rms)
                
                if plane_names is None:
                    plane_names = [r.name for r in results]
            except Exception as e:
                print(f"    警告: 网格大小 {gs} 测试失败: {e}")
                input_radii.append(np.nan)
                output_radii.append(np.nan)
                magnifications.append(np.nan)
                wfe_rms_all.append([np.nan] * 6)
        
        return ParameterStudyResult(
            parameter_name="网格大小",
            parameter_values=grid_sizes,
            input_beam_radii=input_radii,
            output_beam_radii=output_radii,
            magnifications=magnifications,
            wfe_rms_values=wfe_rms_all,
            sampling_plane_names=plane_names or [],
        )


def print_comparison_table(results: List[ComparisonResult], title: str = ""):
    """打印对比结果表格"""
    print("\n" + "=" * 90)
    if title:
        print(title)
        print("=" * 90)
    
    header = f"{'采样面':<15} {'距离(mm)':<10} {'PROPER w':<12} {'ABCD w':<12} {'误差(%)':<10} {'WFE RMS':<10}"
    print(header)
    print("-" * 90)
    
    for r in results:
        print(f"{r.name:<15} {r.distance:<10.1f} {r.proper_beam_radius:<12.4f} "
              f"{r.abcd_beam_radius:<12.4f} {r.beam_radius_error_pct:<10.3f} {r.wavefront_rms:<10.5f}")
    
    print("-" * 90)


def print_parameter_study(study: ParameterStudyResult):
    """打印参数研究结果"""
    print(f"\n{'=' * 70}")
    print(f"参数研究: {study.parameter_name}")
    print(f"{'=' * 70}")
    
    print(f"\n{'参数值':<15} {'输入w(mm)':<12} {'输出w(mm)':<12} {'放大倍率':<12} {'最大WFE':<12}")
    print("-" * 70)
    
    for i, val in enumerate(study.parameter_values):
        max_wfe = max(study.wfe_rms_values[i]) if study.wfe_rms_values[i] else np.nan
        print(f"{str(val):<15} {study.input_beam_radii[i]:<12.4f} "
              f"{study.output_beam_radii[i]:<12.4f} {study.magnifications[i]:<12.4f} "
              f"{max_wfe:<12.5f}")
    
    print("-" * 70)


def plot_comparison_results(
    results: List[ComparisonResult],
    output_file: str = "galilean_oap_abcd_comparison.png"
):
    """绘制对比结果图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    distances = [r.distance for r in results]
    proper_radii = [r.proper_beam_radius for r in results]
    abcd_radii = [r.abcd_beam_radius for r in results]
    errors = [r.beam_radius_error_pct for r in results]
    wfe_rms = [r.wavefront_rms for r in results]
    names = [r.name for r in results]
    
    # 图1: 光束半径对比
    ax1 = axes[0, 0]
    x_pos = range(len(results))
    width = 0.35
    ax1.bar([x - width/2 for x in x_pos], proper_radii, width, label='PROPER', color='steelblue')
    ax1.bar([x + width/2 for x in x_pos], abcd_radii, width, label='ABCD', color='coral')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.set_ylabel('光束半径 (mm)')
    ax1.set_title('光束半径对比: PROPER vs ABCD')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2: 光束半径误差
    ax2 = axes[0, 1]
    bars = ax2.bar(x_pos, errors, color='green', alpha=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylabel('相对误差 (%)')
    ax2.set_title('光束半径相对误差')
    ax2.axhline(y=1.0, color='r', linestyle='--', label='1% 阈值')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    for bar, val in zip(bars, errors):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}%', ha='center', va='bottom', fontsize=8)
    
    # 图3: 波前误差
    ax3 = axes[1, 0]
    bars = ax3.bar(x_pos, wfe_rms, color='purple', alpha=0.7)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(names, rotation=45, ha='right')
    ax3.set_ylabel('WFE RMS (waves)')
    ax3.set_title('波前误差 (RMS)')
    ax3.grid(True, alpha=0.3)
    for bar, val in zip(bars, wfe_rms):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                f'{val:.5f}', ha='center', va='bottom', fontsize=8)
    
    # 图4: 光束半径沿光程变化
    ax4 = axes[1, 1]
    ax4.plot(distances, proper_radii, 'bo-', label='PROPER', markersize=8, linewidth=2)
    ax4.plot(distances, abcd_radii, 'rs--', label='ABCD', markersize=6, linewidth=2)
    ax4.set_xlabel('光程距离 (mm)')
    ax4.set_ylabel('光束半径 (mm)')
    ax4.set_title('光束半径沿光程变化')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ 保存图像: {output_file}")


def plot_parameter_study(
    study: ParameterStudyResult,
    output_file: str
):
    """绘制参数研究结果图"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    params = study.parameter_values
    
    # 图1: 放大倍率
    ax1 = axes[0, 0]
    ax1.plot(params, study.magnifications, 'bo-', markersize=8, linewidth=2)
    ax1.axhline(y=3.0, color='r', linestyle='--', label='设计值 3.0x')
    ax1.set_xlabel(study.parameter_name)
    ax1.set_ylabel('放大倍率')
    ax1.set_title(f'放大倍率 vs {study.parameter_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2: 输出光束半径
    ax2 = axes[0, 1]
    ax2.plot(params, study.output_beam_radii, 'go-', markersize=8, linewidth=2)
    ax2.set_xlabel(study.parameter_name)
    ax2.set_ylabel('输出光束半径 (mm)')
    ax2.set_title(f'输出光束半径 vs {study.parameter_name}')
    ax2.grid(True, alpha=0.3)
    
    # 图3: 放大倍率误差
    ax3 = axes[1, 0]
    mag_errors = [abs(m - 3.0) / 3.0 * 100 for m in study.magnifications]
    ax3.plot(params, mag_errors, 'ro-', markersize=8, linewidth=2)
    ax3.axhline(y=1.0, color='gray', linestyle='--', label='1% 阈值')
    ax3.set_xlabel(study.parameter_name)
    ax3.set_ylabel('放大倍率误差 (%)')
    ax3.set_title(f'放大倍率误差 vs {study.parameter_name}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 图4: 各采样面 WFE
    ax4 = axes[1, 1]
    if study.sampling_plane_names:
        wfe_array = np.array(study.wfe_rms_values)
        for i, name in enumerate(study.sampling_plane_names):
            if i < wfe_array.shape[1]:
                ax4.plot(params, wfe_array[:, i], 'o-', label=name, markersize=6)
    ax4.set_xlabel(study.parameter_name)
    ax4.set_ylabel('WFE RMS (waves)')
    ax4.set_title(f'波前误差 vs {study.parameter_name}')
    ax4.legend(fontsize=8, loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ 保存图像: {output_file}")


def run_full_comparison_test():
    """运行完整的对比测试"""
    print("=" * 70)
    print("伽利略式 OAP 扩束镜：混合光学追迹与 ABCD 方法对比测试")
    print("=" * 70)
    
    comparison = GalileanOAPComparison()
    
    # =========================================================================
    # 1. 基准对比测试
    # =========================================================================
    print("\n[1] 基准对比测试 (grid_size=512, beam_ratio=0.25, num_rays=100)")
    print("-" * 70)
    
    results, summary = comparison.run_comparison(
        grid_size=512,
        beam_ratio=0.25,
        hybrid_num_rays=100,
    )
    
    print_comparison_table(results, "各采样面对比结果")
    
    print(f"\n汇总信息:")
    print(f"  输入光束半径: {summary['input_beam_radius']:.4f} mm")
    print(f"  输出光束半径: {summary['output_beam_radius']:.4f} mm")
    print(f"  实测放大倍率: {summary['measured_magnification']:.4f}x")
    print(f"  设计放大倍率: {summary['design_magnification']:.1f}x")
    print(f"  放大倍率误差: {abs(summary['measured_magnification'] - 3.0) / 3.0 * 100:.3f}%")
    print(f"  最大光束半径误差: {summary['max_beam_radius_error']:.4f}%")
    print(f"  平均光束半径误差: {summary['mean_beam_radius_error']:.4f}%")
    print(f"  最大 WFE RMS: {summary['max_wfe_rms']:.6f} waves")
    
    # 绘制基准对比图
    plot_comparison_results(results, "tests/output/galilean_oap_baseline_comparison.png")
    
    # =========================================================================
    # 2. 光线数量影响研究
    # =========================================================================
    print("\n[2] 光线数量影响研究")
    print("-" * 70)
    
    ray_study = comparison.study_ray_count_effect(
        ray_counts=[36, 64, 100, 144, 196],
        grid_size=512,
        beam_ratio=0.25,
    )
    
    print_parameter_study(ray_study)
    plot_parameter_study(ray_study, "tests/output/galilean_oap_ray_count_study.png")
    
    # =========================================================================
    # 3. 采样面积（beam_ratio）影响研究
    # =========================================================================
    print("\n[3] 采样面积（beam_ratio）影响研究")
    print("-" * 70)
    
    br_study = comparison.study_beam_ratio_effect(
        beam_ratios=[0.15, 0.20, 0.25, 0.30, 0.35],
        grid_size=512,
        hybrid_num_rays=100,
    )
    
    print_parameter_study(br_study)
    plot_parameter_study(br_study, "tests/output/galilean_oap_beam_ratio_study.png")
    
    # =========================================================================
    # 4. 网格大小影响研究
    # =========================================================================
    print("\n[4] 网格大小影响研究")
    print("-" * 70)
    
    grid_study = comparison.study_grid_size_effect(
        grid_sizes=[256, 512, 1024],
        beam_ratio=0.25,
        hybrid_num_rays=100,
    )
    
    print_parameter_study(grid_study)
    plot_parameter_study(grid_study, "tests/output/galilean_oap_grid_size_study.png")
    
    # =========================================================================
    # 5. 综合分析
    # =========================================================================
    print("\n" + "=" * 70)
    print("综合分析")
    print("=" * 70)
    
    # 分析光线数量的影响
    print("\n光线数量影响分析:")
    ray_mag_std = np.nanstd(ray_study.magnifications)
    print(f"  放大倍率标准差: {ray_mag_std:.6f}")
    print(f"  结论: {'光线数量对结果影响较小' if ray_mag_std < 0.01 else '光线数量对结果有显著影响'}")
    
    # 分析 beam_ratio 的影响
    print("\n采样面积（beam_ratio）影响分析:")
    br_mag_std = np.nanstd(br_study.magnifications)
    print(f"  放大倍率标准差: {br_mag_std:.6f}")
    print(f"  结论: {'beam_ratio 对结果影响较小' if br_mag_std < 0.01 else 'beam_ratio 对结果有显著影响'}")
    
    # 分析网格大小的影响
    print("\n网格大小影响分析:")
    grid_mag_std = np.nanstd(grid_study.magnifications)
    print(f"  放大倍率标准差: {grid_mag_std:.6f}")
    print(f"  结论: {'网格大小对结果影响较小' if grid_mag_std < 0.01 else '网格大小对结果有显著影响'}")
    
    # 总体结论
    print("\n" + "=" * 70)
    print("总体结论")
    print("=" * 70)
    
    all_errors = [summary['max_beam_radius_error']]
    all_wfe = [summary['max_wfe_rms']]
    
    if max(all_errors) < 1.0:
        print("✓ 混合光学追迹与 ABCD 方法的光束半径误差 < 1%，结果一致性良好")
    else:
        print("✗ 混合光学追迹与 ABCD 方法存在较大偏差，需要进一步分析")
    
    if max(all_wfe) < 0.1:
        print("✓ 波前误差 < 0.1 waves，系统接近衍射极限")
    else:
        print(f"! 波前误差 = {max(all_wfe):.4f} waves，存在一定像差")
    
    print("\n测试完成！")
    print(f"输出文件保存在 tests/output/ 目录下")


# =============================================================================
# pytest 测试用例
# =============================================================================

import pytest


class TestGalileanOAPABCDComparison:
    """伽利略式 OAP 扩束镜 ABCD 对比测试类"""
    
    @pytest.fixture
    def comparison(self):
        return GalileanOAPComparison()
    
    def test_baseline_beam_radius_accuracy(self, comparison):
        """测试基准配置下光束半径精度"""
        results, summary = comparison.run_comparison(
            grid_size=512,
            beam_ratio=0.25,
            hybrid_num_rays=100,
        )
        
        # 所有采样面的光束半径误差应 < 1%
        for r in results:
            assert r.beam_radius_error_pct < 1.0, (
                f"采样面 '{r.name}' 光束半径误差过大: {r.beam_radius_error_pct:.3f}%"
            )
    
    def test_magnification_accuracy(self, comparison):
        """测试放大倍率精度"""
        results, summary = comparison.run_comparison()
        
        mag_error = abs(summary['measured_magnification'] - 3.0) / 3.0 * 100
        assert mag_error < 1.0, f"放大倍率误差过大: {mag_error:.3f}%"
    
    def test_wavefront_quality(self, comparison):
        """测试波前质量"""
        results, summary = comparison.run_comparison()
        
        for r in results:
            assert r.wavefront_rms < 0.1, (
                f"采样面 '{r.name}' WFE RMS 过大: {r.wavefront_rms:.5f} waves"
            )
    
    @pytest.mark.parametrize("num_rays", [64, 100, 144])
    def test_ray_count_stability(self, comparison, num_rays):
        """测试不同光线数量下的稳定性"""
        results, summary = comparison.run_comparison(hybrid_num_rays=num_rays)
        
        mag_error = abs(summary['measured_magnification'] - 3.0) / 3.0 * 100
        assert mag_error < 2.0, (
            f"光线数量 {num_rays} 下放大倍率误差过大: {mag_error:.3f}%"
        )
    
    @pytest.mark.parametrize("beam_ratio", [0.20, 0.25, 0.30])
    def test_beam_ratio_stability(self, comparison, beam_ratio):
        """测试不同 beam_ratio 下的稳定性"""
        results, summary = comparison.run_comparison(beam_ratio=beam_ratio)
        
        mag_error = abs(summary['measured_magnification'] - 3.0) / 3.0 * 100
        assert mag_error < 2.0, (
            f"beam_ratio {beam_ratio} 下放大倍率误差过大: {mag_error:.3f}%"
        )
    
    @pytest.mark.parametrize("grid_size", [256, 512])
    def test_grid_size_stability(self, comparison, grid_size):
        """测试不同网格大小下的稳定性"""
        results, summary = comparison.run_comparison(grid_size=grid_size)
        
        mag_error = abs(summary['measured_magnification'] - 3.0) / 3.0 * 100
        assert mag_error < 2.0, (
            f"网格大小 {grid_size} 下放大倍率误差过大: {mag_error:.3f}%"
        )


if __name__ == "__main__":
    # 确保输出目录存在
    import os
    os.makedirs("tests/output", exist_ok=True)
    
    # 运行完整对比测试
    run_full_comparison_test()
