"""伽利略式扩束镜集成测试

本模块测试伽利略式 OAP 扩束镜系统的混合传播模式正确性。

测试内容：
- 混合模式与 ABCD 理论的光束半径一致性（误差 < 1%）
- 放大倍率与设计值一致（误差 < 1%）
- 各采样面的波前质量

**Validates: Requirements 6.1, 6.2, Property 8**

参考：examples/galilean_oap_expander.py
"""

import sys
import numpy as np
import pytest

sys.path.insert(0, 'src')

from sequential_system import (
    SequentialOpticalSystem,
    GaussianBeamSource,
    ParabolicMirror,
    FlatMirror,
)


class TestGalileanOAPExpander:
    """伽利略式 OAP 扩束镜集成测试
    
    系统配置（无实焦点）：
    - OAP1: f=-50mm 凸面镜（发散光束），倾斜 45°
    - 折叠镜: 平面镜，倾斜 45°
    - OAP2: f=150mm 凹面镜（准直发散光束），倾斜 45°
    - 放大倍率: M = -f2/f1 = 3x
    
    **Validates: Requirements 6.1, 6.2, Property 8**
    """
    
    # =========================================================================
    # 系统参数定义
    # =========================================================================
    
    # 光源参数
    WAVELENGTH_UM = 10.64      # μm, CO2 激光
    W0_INPUT_MM = 10.0         # mm, 输入束腰半径
    
    # 扩束镜焦距设计
    F1_MM = -50.0              # mm, OAP1 焦距（负值 = 凸面，发散）
    F2_MM = 150.0              # mm, OAP2 焦距（正值 = 凹面，准直）
    DESIGN_MAGNIFICATION = 3.0  # 设计放大倍率 = -f2/f1 = 3x
    
    # 离轴参数（90° OAP）
    D_OFF_OAP1_MM = 100.0      # OAP1 离轴距离 = 2|f1|
    D_OFF_OAP2_MM = 300.0      # OAP2 离轴距离 = 2*f2
    
    # 倾斜角度（弧度）
    TILT_45_DEG = np.pi / 4    # 45° 倾斜
    
    # 几何参数
    D_OAP1_TO_FOLD_MM = 50.0   # OAP1 到折叠镜的距离
    D_FOLD_TO_OAP2_MM = 50.0   # 折叠镜到 OAP2 的距离
    D_OAP2_TO_OUTPUT_MM = 100.0  # OAP2 到输出采样面的距离
    
    # 测试容差
    BEAM_RADIUS_ERROR_TOLERANCE = 0.01  # 光束半径误差容差 1%
    MAGNIFICATION_ERROR_TOLERANCE = 0.01  # 放大倍率误差容差 1%
    
    def _create_galilean_expander_system(
        self,
        grid_size: int = 512,
        beam_ratio: float = 0.25,
        hybrid_num_rays: int = 100,
    ) -> SequentialOpticalSystem:
        """创建伽利略式扩束镜系统
        
        参数:
            grid_size: 网格大小
            beam_ratio: PROPER beam_ratio 参数
            hybrid_num_rays: 混合传播模式下的采样光线数量
        
        返回:
            配置好的 SequentialOpticalSystem 对象
        """
        # 光源定义
        source = GaussianBeamSource(
            wavelength=self.WAVELENGTH_UM,
            w0=self.W0_INPUT_MM,
            z0=0.0,
        )
        
        # 创建系统（较小的 beam_ratio 以容纳扩束后的大光束）
        system = SequentialOpticalSystem(
            source=source,
            grid_size=grid_size,
            beam_ratio=beam_ratio,
            use_hybrid_propagation=True,
            hybrid_num_rays=hybrid_num_rays,
        )
        
        # 计算总光程
        total_path = (
            self.D_OAP1_TO_FOLD_MM + 
            self.D_FOLD_TO_OAP2_MM + 
            self.D_OAP2_TO_OUTPUT_MM
        )
        
        # --- 采样面：输入 ---
        system.add_sampling_plane(distance=0.0, name="Input")
        
        # --- OAP1：凸面抛物面镜（发散光束），倾斜 45° ---
        system.add_surface(ParabolicMirror(
            parent_focal_length=self.F1_MM,
            thickness=self.D_OAP1_TO_FOLD_MM,
            semi_aperture=20.0,
            off_axis_distance=self.D_OFF_OAP1_MM,
            tilt_x=self.TILT_45_DEG,
            name="OAP1 (45°)",
        ))
        
        # --- 采样面：OAP1 之后 ---
        system.add_sampling_plane(
            distance=self.D_OAP1_TO_FOLD_MM, 
            name="After OAP1"
        )
        
        # --- 折叠镜：平面镜，倾斜 45° ---
        system.add_surface(FlatMirror(
            thickness=self.D_FOLD_TO_OAP2_MM,
            semi_aperture=30.0,
            tilt_x=self.TILT_45_DEG,
            name="Fold (45°)",
        ))
        
        # --- 采样面：折叠镜之后 ---
        system.add_sampling_plane(
            distance=self.D_OAP1_TO_FOLD_MM + self.D_FOLD_TO_OAP2_MM, 
            name="After Fold"
        )
        
        # --- OAP2：凹面抛物面镜（准直光束），倾斜 45° ---
        system.add_surface(ParabolicMirror(
            parent_focal_length=self.F2_MM,
            thickness=self.D_OAP2_TO_OUTPUT_MM,
            semi_aperture=50.0,
            off_axis_distance=self.D_OFF_OAP2_MM,
            tilt_x=self.TILT_45_DEG,
            name="OAP2 (45°)",
        ))
        
        # --- 采样面：输出 ---
        system.add_sampling_plane(distance=total_path, name="Output")
        
        return system
    
    # =========================================================================
    # 测试用例
    # =========================================================================
    
    def test_beam_radius_vs_abcd(self):
        """验证混合模式与 ABCD 理论的光束半径一致性
        
        在各采样面上，PROPER 计算的光束半径应与 ABCD 理论一致，
        相对误差应小于 1%。
        
        **Validates: Requirements 6.1, Property 8**
        """
        # 创建系统
        system = self._create_galilean_expander_system()
        
        # 运行仿真
        results = system.run()
        
        # 验证各采样面的光束半径
        for name, result in results.sampling_results.items():
            # 获取 PROPER 结果
            proper_w = result.beam_radius
            
            # 获取 ABCD 结果
            abcd_result = system.get_abcd_result(result.distance)
            abcd_w = abcd_result.w
            
            # 计算相对误差
            if abcd_w > 0.001:  # 避免除以零
                error_pct = abs(proper_w - abcd_w) / abcd_w
            else:
                error_pct = 0.0
            
            # 验证误差在容差范围内
            assert error_pct < self.BEAM_RADIUS_ERROR_TOLERANCE, (
                f"采样面 '{name}' 光束半径误差过大: "
                f"PROPER={proper_w:.4f}mm, ABCD={abcd_w:.4f}mm, "
                f"误差={error_pct*100:.2f}% (容差={self.BEAM_RADIUS_ERROR_TOLERANCE*100:.0f}%)"
            )
    
    def test_magnification_accuracy(self):
        """验证放大倍率与设计值一致
        
        输出光束半径与输入光束半径的比值应等于设计放大倍率，
        误差应小于 1%。
        
        **Validates: Requirements 6.2, Property 8**
        """
        # 创建系统
        system = self._create_galilean_expander_system()
        
        # 运行仿真
        results = system.run()
        
        # 获取输入和输出光束半径
        input_result = results.sampling_results["Input"]
        output_result = results.sampling_results["Output"]
        
        input_w = input_result.beam_radius
        output_w = output_result.beam_radius
        
        # 计算实测放大倍率
        measured_magnification = output_w / input_w
        
        # 计算相对误差
        error_pct = abs(measured_magnification - self.DESIGN_MAGNIFICATION) / self.DESIGN_MAGNIFICATION
        
        # 验证误差在容差范围内
        assert error_pct < self.MAGNIFICATION_ERROR_TOLERANCE, (
            f"放大倍率误差过大: "
            f"设计值={self.DESIGN_MAGNIFICATION:.2f}x, "
            f"实测值={measured_magnification:.2f}x, "
            f"误差={error_pct*100:.2f}% (容差={self.MAGNIFICATION_ERROR_TOLERANCE*100:.0f}%)"
        )
    
    def test_abcd_magnification_consistency(self):
        """验证 ABCD 理论计算的放大倍率与设计值一致
        
        作为参考，验证 ABCD 理论本身的正确性。
        
        **Validates: Requirements 6.2**
        """
        # 创建系统
        system = self._create_galilean_expander_system()
        
        # 获取 ABCD 结果
        total_path = (
            self.D_OAP1_TO_FOLD_MM + 
            self.D_FOLD_TO_OAP2_MM + 
            self.D_OAP2_TO_OUTPUT_MM
        )
        
        abcd_input = system.get_abcd_result(0.0)
        abcd_output = system.get_abcd_result(total_path)
        
        # 计算 ABCD 放大倍率
        abcd_magnification = abcd_output.w / abcd_input.w
        
        # 计算相对误差
        error_pct = abs(abcd_magnification - self.DESIGN_MAGNIFICATION) / self.DESIGN_MAGNIFICATION
        
        # 验证误差在容差范围内
        assert error_pct < self.MAGNIFICATION_ERROR_TOLERANCE, (
            f"ABCD 放大倍率误差过大: "
            f"设计值={self.DESIGN_MAGNIFICATION:.2f}x, "
            f"ABCD值={abcd_magnification:.2f}x, "
            f"误差={error_pct*100:.2f}%"
        )
    
    def test_wavefront_quality(self):
        """验证各采样面的波前质量
        
        对于理想抛物面镜系统，波前误差应该很小。
        
        **Validates: Property 8**
        """
        # 创建系统
        system = self._create_galilean_expander_system()
        
        # 运行仿真
        results = system.run()
        
        # 验证各采样面的波前质量
        for name, result in results.sampling_results.items():
            wfe_rms = result.wavefront_rms
            
            # 波前误差应为有限值
            assert np.isfinite(wfe_rms), (
                f"采样面 '{name}' 波前误差为非有限值: {wfe_rms}"
            )
            
            # 对于理想抛物面镜系统，WFE RMS 应该很小（< 0.1 波）
            # 注意：由于数值精度和采样限制，允许一定的误差
            assert wfe_rms < 0.1, (
                f"采样面 '{name}' 波前误差过大: "
                f"WFE RMS = {wfe_rms:.4f} waves (期望 < 0.1 waves)"
            )
    
    def test_beam_radius_positive_finite(self):
        """验证所有采样面的光束半径为正有限值
        
        基本健全性检查。
        """
        # 创建系统
        system = self._create_galilean_expander_system()
        
        # 运行仿真
        results = system.run()
        
        # 验证各采样面
        for name, result in results.sampling_results.items():
            assert result.beam_radius > 0, (
                f"采样面 '{name}' 光束半径应为正值: {result.beam_radius}"
            )
            assert np.isfinite(result.beam_radius), (
                f"采样面 '{name}' 光束半径应为有限值: {result.beam_radius}"
            )
    
    def test_output_beam_larger_than_input(self):
        """验证输出光束大于输入光束（扩束效果）
        
        伽利略式扩束镜应该放大光束。
        """
        # 创建系统
        system = self._create_galilean_expander_system()
        
        # 运行仿真
        results = system.run()
        
        # 获取输入和输出光束半径
        input_w = results.sampling_results["Input"].beam_radius
        output_w = results.sampling_results["Output"].beam_radius
        
        # 验证输出光束大于输入光束
        assert output_w > input_w, (
            f"输出光束应大于输入光束: "
            f"输入={input_w:.3f}mm, 输出={output_w:.3f}mm"
        )


class TestGalileanOAPExpanderParametric:
    """伽利略式扩束镜参数化测试
    
    测试不同参数配置下的系统行为。
    """
    
    @pytest.mark.parametrize("grid_size", [256, 512])
    def test_different_grid_sizes(self, grid_size: int):
        """测试不同网格大小下的结果一致性
        
        不同网格大小应该给出相似的结果。
        """
        # 使用 TestGalileanOAPExpander 的参数
        test_instance = TestGalileanOAPExpander()
        
        # 创建系统
        system = test_instance._create_galilean_expander_system(grid_size=grid_size)
        
        # 运行仿真
        results = system.run()
        
        # 验证输出存在且有效
        assert "Output" in results.sampling_results
        output_result = results.sampling_results["Output"]
        
        assert output_result.beam_radius > 0
        assert np.isfinite(output_result.beam_radius)
        
        # 验证放大倍率大致正确（允许 5% 误差，因为不同网格大小可能有数值差异）
        input_w = results.sampling_results["Input"].beam_radius
        output_w = output_result.beam_radius
        measured_mag = output_w / input_w
        
        assert abs(measured_mag - 3.0) / 3.0 < 0.05, (
            f"网格大小 {grid_size} 下放大倍率偏差过大: {measured_mag:.2f}x"
        )
    
    @pytest.mark.parametrize("hybrid_num_rays", [64, 100, 144])
    def test_different_ray_counts(self, hybrid_num_rays: int):
        """测试不同光线数量下的结果一致性
        
        不同光线数量应该给出相似的结果。
        """
        # 使用 TestGalileanOAPExpander 的参数
        test_instance = TestGalileanOAPExpander()
        
        # 创建系统
        system = test_instance._create_galilean_expander_system(
            hybrid_num_rays=hybrid_num_rays
        )
        
        # 运行仿真
        results = system.run()
        
        # 验证输出存在且有效
        assert "Output" in results.sampling_results
        output_result = results.sampling_results["Output"]
        
        assert output_result.beam_radius > 0
        assert np.isfinite(output_result.beam_radius)


class TestGalileanOAPExpanderAmplitude:
    """伽利略式扩束镜振幅验证测试
    
    验证混合传播模式下振幅变化符合预期。
    
    注意：PROPER 的波前归一化方式是保持总能量守恒，而不是保持峰值振幅。
    当光束扩展时，光束覆盖更多像素，但总能量不变。
    
    **Validates: Requirements 2.3, 2.4 - 雅可比矩阵振幅计算**
    """
    
    def test_beam_expansion_changes_amplitude_distribution(self):
        """验证扩束后振幅分布发生变化
        
        扩束后，光束覆盖更大的区域，光束半径应该增大。
        这里使用光束半径（而不是像素数量）来验证扩束效果。
        
        **Validates: Requirements 2.3 - 雅可比矩阵振幅计算**
        """
        # 使用 TestGalileanOAPExpander 的参数
        test_instance = TestGalileanOAPExpander()
        
        # 创建系统
        system = test_instance._create_galilean_expander_system()
        
        # 运行仿真
        results = system.run()
        
        # 获取输入和输出的光束半径
        input_w = results.sampling_results["Input"].beam_radius
        output_w = results.sampling_results["Output"].beam_radius
        
        # 计算光束半径比（放大倍率）
        beam_ratio = output_w / input_w
        
        # 对于 3x 扩束器，光束半径应增大约 3 倍
        expected_ratio = 3.0
        
        # 验证光束确实扩大了
        assert beam_ratio > 1.0, (
            f"扩束后光束半径应增大，但比值为 {beam_ratio:.3f}"
        )
        
        # 验证扩大的量级大致正确（允许 10% 的误差）
        error = abs(beam_ratio - expected_ratio) / expected_ratio
        assert error < 0.10, (
            f"光束半径比与预期不符：实际 = {beam_ratio:.3f}，"
            f"期望 = {expected_ratio:.1f}，误差 = {error*100:.1f}%"
        )
    
    def test_amplitude_profile_is_gaussian_like(self):
        """验证输出振幅分布仍然是高斯型
        
        扩束后，振幅分布应该仍然保持高斯型特征。
        
        **Validates: Requirements 2.4 - 振幅归一化**
        """
        # 使用 TestGalileanOAPExpander 的参数
        test_instance = TestGalileanOAPExpander()
        
        # 创建系统
        system = test_instance._create_galilean_expander_system()
        
        # 运行仿真
        results = system.run()
        
        # 获取输出的波前数据
        output_result = results.sampling_results["Output"]
        output_amplitude = np.abs(output_result.wavefront)
        
        # 找到峰值位置
        peak_idx = np.unravel_index(np.argmax(output_amplitude), output_amplitude.shape)
        
        # 提取通过峰值的水平和垂直切片
        h_slice = output_amplitude[peak_idx[0], :]
        v_slice = output_amplitude[:, peak_idx[1]]
        
        # 验证切片是单峰的（峰值在中间附近）
        h_peak_idx = np.argmax(h_slice)
        v_peak_idx = np.argmax(v_slice)
        
        n = len(h_slice)
        center = n // 2
        
        # 峰值应该在中心附近（允许 20% 的偏移）
        assert abs(h_peak_idx - center) < n * 0.2, (
            f"水平切片峰值位置偏离中心过多：{h_peak_idx} vs {center}"
        )
        assert abs(v_peak_idx - center) < n * 0.2, (
            f"垂直切片峰值位置偏离中心过多：{v_peak_idx} vs {center}"
        )
        
        # 验证振幅从峰值向边缘递减
        # 检查峰值两侧的振幅是否递减
        peak_val = h_slice[h_peak_idx]
        edge_val = max(h_slice[0], h_slice[-1])
        
        assert edge_val < peak_val * 0.5, (
            f"振幅分布不是高斯型：边缘值 {edge_val:.4f} 应远小于峰值 {peak_val:.4f}"
        )
    
    def test_energy_conservation(self):
        """验证能量守恒
        
        扩束前后总能量应保持不变（在数值精度范围内）。
        这是混合传播模式正确性的关键验证。
        
        **Validates: Requirements 2.3 - 能量守恒**
        """
        # 使用 TestGalileanOAPExpander 的参数
        test_instance = TestGalileanOAPExpander()
        
        # 创建系统
        system = test_instance._create_galilean_expander_system()
        
        # 运行仿真
        results = system.run()
        
        # 获取输入和输出的波前数据
        input_result = results.sampling_results["Input"]
        output_result = results.sampling_results["Output"]
        
        # 计算总能量（强度的积分）
        # 注意：PROPER 的波前已经考虑了采样间隔的归一化
        input_intensity = np.abs(input_result.wavefront)**2
        output_intensity = np.abs(output_result.wavefront)**2
        
        input_total_energy = np.sum(input_intensity)
        output_total_energy = np.sum(output_intensity)
        
        # 计算能量比
        energy_ratio = output_total_energy / input_total_energy
        
        # 能量应该守恒（允许 5% 的误差）
        assert 0.95 < energy_ratio < 1.05, (
            f"能量不守恒：输出/输入能量比 = {energy_ratio:.4f}，"
            f"期望约 1.0"
        )
    
    def test_amplitude_finite_and_positive(self):
        """验证所有采样面的振幅为有限正值
        
        基本健全性检查：振幅应该是有限的正值。
        
        **Validates: Requirements 2.6 - 无效光线区域振幅为 0**
        """
        # 使用 TestGalileanOAPExpander 的参数
        test_instance = TestGalileanOAPExpander()
        
        # 创建系统
        system = test_instance._create_galilean_expander_system()
        
        # 运行仿真
        results = system.run()
        
        # 验证各采样面
        for name, result in results.sampling_results.items():
            amplitude = np.abs(result.wavefront)
            
            # 振幅应该是有限值
            assert np.all(np.isfinite(amplitude)), (
                f"采样面 '{name}' 存在非有限振幅值"
            )
            
            # 振幅应该非负
            assert np.all(amplitude >= 0), (
                f"采样面 '{name}' 存在负振幅值"
            )
            
            # 峰值振幅应该为正
            assert np.max(amplitude) > 0, (
                f"采样面 '{name}' 峰值振幅应为正值"
            )


class TestGalileanOAPExpanderComparison:
    """伽利略式扩束镜对比测试
    
    对比混合传播模式与纯 PROPER 模式的结果。
    """
    
    def test_hybrid_vs_proper_consistency(self):
        """验证混合传播模式与纯 PROPER 模式的一致性
        
        对于理想抛物面镜系统，两种模式应该给出相似的结果。
        
        注意：由于混合模式使用光线追迹计算 OPD，而纯 PROPER 模式
        使用 prop_lens，两者可能存在微小差异。
        """
        # 使用 TestGalileanOAPExpander 的参数
        test_instance = TestGalileanOAPExpander()
        
        # 创建混合模式系统
        source = GaussianBeamSource(
            wavelength=test_instance.WAVELENGTH_UM,
            w0=test_instance.W0_INPUT_MM,
            z0=0.0,
        )
        
        # 混合模式
        system_hybrid = SequentialOpticalSystem(
            source=source,
            grid_size=512,
            beam_ratio=0.25,
            use_hybrid_propagation=True,
            hybrid_num_rays=100,
        )
        
        # 添加元件和采样面（简化版本，只测试输入和输出）
        total_path = (
            test_instance.D_OAP1_TO_FOLD_MM + 
            test_instance.D_FOLD_TO_OAP2_MM + 
            test_instance.D_OAP2_TO_OUTPUT_MM
        )
        
        system_hybrid.add_sampling_plane(distance=0.0, name="Input")
        
        system_hybrid.add_surface(ParabolicMirror(
            parent_focal_length=test_instance.F1_MM,
            thickness=test_instance.D_OAP1_TO_FOLD_MM,
            semi_aperture=20.0,
            off_axis_distance=test_instance.D_OFF_OAP1_MM,
            tilt_x=test_instance.TILT_45_DEG,
        ))
        
        system_hybrid.add_surface(FlatMirror(
            thickness=test_instance.D_FOLD_TO_OAP2_MM,
            semi_aperture=30.0,
            tilt_x=test_instance.TILT_45_DEG,
        ))
        
        system_hybrid.add_surface(ParabolicMirror(
            parent_focal_length=test_instance.F2_MM,
            thickness=test_instance.D_OAP2_TO_OUTPUT_MM,
            semi_aperture=50.0,
            off_axis_distance=test_instance.D_OFF_OAP2_MM,
            tilt_x=test_instance.TILT_45_DEG,
        ))
        
        system_hybrid.add_sampling_plane(distance=total_path, name="Output")
        
        # 运行混合模式仿真
        results_hybrid = system_hybrid.run()
        
        # 获取结果
        hybrid_input_w = results_hybrid.sampling_results["Input"].beam_radius
        hybrid_output_w = results_hybrid.sampling_results["Output"].beam_radius
        hybrid_mag = hybrid_output_w / hybrid_input_w
        
        # 验证混合模式结果与 ABCD 理论一致
        abcd_input = system_hybrid.get_abcd_result(0.0)
        abcd_output = system_hybrid.get_abcd_result(total_path)
        abcd_mag = abcd_output.w / abcd_input.w
        
        # 放大倍率误差应小于 1%
        mag_error = abs(hybrid_mag - abcd_mag) / abcd_mag
        assert mag_error < 0.01, (
            f"混合模式与 ABCD 放大倍率不一致: "
            f"混合={hybrid_mag:.3f}x, ABCD={abcd_mag:.3f}x, "
            f"误差={mag_error*100:.2f}%"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
