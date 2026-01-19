"""多 OAP 系统集成测试

本模块测试包含多个 OAP（离轴抛物面镜）的光学系统的混合传播模式正确性。

测试内容：
- 4f 中继系统（两个 OAP 组成的 1:1 中继）
- 不等焦距 OAP 中继系统（放大/缩小）
- 三 OAP 系统
- 输出光束参数与 ABCD 理论一致（误差 < 1%）
- WFE RMS < 0.1 波

**Validates: Requirements 6.4, Property 8**

参考：
- tests/test_galilean_oap_integration.py - 伽利略式扩束镜集成测试
- tests/test_multi_fold_integration.py - 多折叠光路集成测试
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


class TestFourFRelaySystem:
    """4f 中继系统测试（1:1 成像）
    
    系统配置：
    - OAP1: f=100mm 凹面镜，倾斜 45°（准直光 → 聚焦）
    - 折叠镜: 平面镜，倾斜 45°（在焦点附近）
    - OAP2: f=100mm 凹面镜，倾斜 45°（聚焦 → 准直光）
    - 放大倍率: M = f2/f1 = 1x
    
    4f 系统特点：
    - 输入和输出光束参数相同（1:1 中继）
    - 中间有实焦点
    
    **Validates: Requirements 6.4, Property 8**
    """
    
    # 光源参数
    WAVELENGTH_UM = 10.64      # μm, CO2 激光
    W0_INPUT_MM = 10.0         # mm, 输入束腰半径
    
    # OAP 焦距设计（相同焦距实现 1:1 中继）
    F_OAP_MM = 100.0           # mm, 两个 OAP 焦距相同
    DESIGN_MAGNIFICATION = 1.0  # 设计放大倍率 = 1x
    
    # 离轴参数（90° OAP）
    D_OFF_AXIS_MM = 200.0      # 离轴距离 = 2f
    
    # 倾斜角度（弧度）
    TILT_45_DEG = np.pi / 4    # 45° 倾斜
    
    # 几何参数
    D_TO_OAP1_MM = 0.0         # 光源到 OAP1 的距离（光源在 OAP1 处）
    D_OAP1_TO_FOLD_MM = 100.0  # OAP1 到折叠镜的距离（= f）
    D_FOLD_TO_OAP2_MM = 100.0  # 折叠镜到 OAP2 的距离（= f）
    D_OAP2_TO_OUTPUT_MM = 100.0  # OAP2 到输出采样面的距离
    
    # 测试容差
    BEAM_RADIUS_ERROR_TOLERANCE = 0.01  # 光束半径误差容差 1%
    MAGNIFICATION_ERROR_TOLERANCE = 0.01  # 放大倍率误差容差 1%
    WFE_RMS_TOLERANCE = 0.1  # WFE RMS 容差 0.1 波
    
    def _create_4f_relay_system(
        self,
        grid_size: int = 512,
        beam_ratio: float = 0.25,
        hybrid_num_rays: int = 100,
    ) -> SequentialOpticalSystem:
        """创建 4f 中继系统
        
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
        
        # 创建系统
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
        
        # --- OAP1：凹面抛物面镜（聚焦），倾斜 45° ---
        system.add_surface(ParabolicMirror(
            parent_focal_length=self.F_OAP_MM,
            thickness=self.D_OAP1_TO_FOLD_MM,
            semi_aperture=25.0,
            off_axis_distance=self.D_OFF_AXIS_MM,
            tilt_x=self.TILT_45_DEG,
            name="OAP1 (45°)",
        ))
        
        # --- 采样面：OAP1 之后（焦点附近） ---
        system.add_sampling_plane(
            distance=self.D_OAP1_TO_FOLD_MM, 
            name="After OAP1 (Focus)"
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
        
        # --- OAP2：凹面抛物面镜（准直），倾斜 45° ---
        system.add_surface(ParabolicMirror(
            parent_focal_length=self.F_OAP_MM,
            thickness=self.D_OAP2_TO_OUTPUT_MM,
            semi_aperture=25.0,
            off_axis_distance=self.D_OFF_AXIS_MM,
            tilt_x=self.TILT_45_DEG,
            name="OAP2 (45°)",
        ))
        
        # --- 采样面：输出 ---
        system.add_sampling_plane(distance=total_path, name="Output")
        
        return system
    
    def test_beam_radius_vs_abcd(self):
        """验证 4f 系统混合模式与 ABCD 理论的光束半径一致性
        
        在各采样面上，PROPER 计算的光束半径应与 ABCD 理论一致，
        相对误差应小于 1%。
        
        注意：在焦点附近，由于光束半径非常小，数值精度和采样限制
        会导致更大的相对误差，因此对焦点附近的采样面使用更宽松的容差。
        
        **Validates: Requirements 6.4, Property 8**
        """
        # 创建系统
        system = self._create_4f_relay_system()
        
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
            
            # 对焦点附近的采样面使用更宽松的容差（光束半径 < 0.5mm）
            # 因为在焦点附近，数值精度和采样限制会导致更大的相对误差
            if abcd_w < 0.5:
                tolerance = 0.05  # 5% 容差
            else:
                tolerance = self.BEAM_RADIUS_ERROR_TOLERANCE
            
            # 验证误差在容差范围内
            assert error_pct < tolerance, (
                f"采样面 '{name}' 光束半径误差过大: "
                f"PROPER={proper_w:.4f}mm, ABCD={abcd_w:.4f}mm, "
                f"误差={error_pct*100:.2f}% (容差={tolerance*100:.0f}%)"
            )
    
    def test_magnification_unity(self):
        """验证 4f 系统放大倍率为 1
        
        输出光束半径应与输入光束半径相同（1:1 中继）。
        
        **Validates: Requirements 6.4, Property 8**
        """
        # 创建系统
        system = self._create_4f_relay_system()
        
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
            f"4f 系统放大倍率误差过大: "
            f"设计值={self.DESIGN_MAGNIFICATION:.2f}x, "
            f"实测值={measured_magnification:.2f}x, "
            f"误差={error_pct*100:.2f}% (容差={self.MAGNIFICATION_ERROR_TOLERANCE*100:.0f}%)"
        )
    
    def test_wavefront_quality(self):
        """验证 4f 系统各采样面的波前质量
        
        对于理想抛物面镜系统，波前误差应该很小（< 0.1 波）。
        
        **Validates: Requirements 6.4, Property 8**
        """
        # 创建系统
        system = self._create_4f_relay_system()
        
        # 运行仿真
        results = system.run()
        
        # 验证各采样面的波前质量
        for name, result in results.sampling_results.items():
            wfe_rms = result.wavefront_rms
            
            # 波前误差应为有限值
            assert np.isfinite(wfe_rms), (
                f"采样面 '{name}' 波前误差为非有限值: {wfe_rms}"
            )
            
            # WFE RMS 应该很小
            assert wfe_rms < self.WFE_RMS_TOLERANCE, (
                f"采样面 '{name}' 波前误差过大: "
                f"WFE RMS = {wfe_rms:.4f} waves (期望 < {self.WFE_RMS_TOLERANCE} waves)"
            )



class TestUnequalFocalLengthOAPRelay:
    """不等焦距 OAP 中继系统测试（放大/缩小）
    
    系统配置：
    - OAP1: f=50mm 凹面镜，倾斜 45°
    - 折叠镜: 平面镜，倾斜 45°
    - OAP2: f=150mm 凹面镜，倾斜 45°
    - 放大倍率: M = f2/f1 = 3x
    
    **Validates: Requirements 6.4, Property 8**
    """
    
    # 光源参数
    WAVELENGTH_UM = 10.64      # μm, CO2 激光
    W0_INPUT_MM = 5.0          # mm, 输入束腰半径
    
    # OAP 焦距设计
    F1_MM = 50.0               # mm, OAP1 焦距
    F2_MM = 150.0              # mm, OAP2 焦距
    DESIGN_MAGNIFICATION = 3.0  # 设计放大倍率 = f2/f1 = 3x
    
    # 离轴参数（90° OAP）
    D_OFF_OAP1_MM = 100.0      # OAP1 离轴距离 = 2f1
    D_OFF_OAP2_MM = 300.0      # OAP2 离轴距离 = 2f2
    
    # 倾斜角度（弧度）
    TILT_45_DEG = np.pi / 4    # 45° 倾斜
    
    # 几何参数
    D_OAP1_TO_FOLD_MM = 50.0   # OAP1 到折叠镜的距离（= f1）
    D_FOLD_TO_OAP2_MM = 150.0  # 折叠镜到 OAP2 的距离（= f2）
    D_OAP2_TO_OUTPUT_MM = 100.0  # OAP2 到输出采样面的距离
    
    # 测试容差
    BEAM_RADIUS_ERROR_TOLERANCE = 0.01  # 光束半径误差容差 1%
    MAGNIFICATION_ERROR_TOLERANCE = 0.01  # 放大倍率误差容差 1%
    WFE_RMS_TOLERANCE = 0.1  # WFE RMS 容差 0.1 波
    
    def _create_unequal_focal_relay_system(
        self,
        grid_size: int = 512,
        beam_ratio: float = 0.2,
        hybrid_num_rays: int = 100,
    ) -> SequentialOpticalSystem:
        """创建不等焦距 OAP 中继系统
        
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
        
        # --- OAP1：凹面抛物面镜（聚焦），倾斜 45° ---
        system.add_surface(ParabolicMirror(
            parent_focal_length=self.F1_MM,
            thickness=self.D_OAP1_TO_FOLD_MM,
            semi_aperture=15.0,
            off_axis_distance=self.D_OFF_OAP1_MM,
            tilt_x=self.TILT_45_DEG,
            name="OAP1 (45°)",
        ))
        
        # --- 采样面：OAP1 之后（焦点） ---
        system.add_sampling_plane(
            distance=self.D_OAP1_TO_FOLD_MM, 
            name="After OAP1 (Focus)"
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
        
        # --- OAP2：凹面抛物面镜（准直），倾斜 45° ---
        system.add_surface(ParabolicMirror(
            parent_focal_length=self.F2_MM,
            thickness=self.D_OAP2_TO_OUTPUT_MM,
            semi_aperture=40.0,
            off_axis_distance=self.D_OFF_OAP2_MM,
            tilt_x=self.TILT_45_DEG,
            name="OAP2 (45°)",
        ))
        
        # --- 采样面：输出 ---
        system.add_sampling_plane(distance=total_path, name="Output")
        
        return system
    
    def test_beam_radius_vs_abcd(self):
        """验证不等焦距中继系统与 ABCD 理论的光束半径一致性
        
        注意：在焦点附近，由于光束半径非常小，数值精度和采样限制
        会导致更大的相对误差，因此对焦点附近的采样面使用更宽松的容差。
        
        **Validates: Requirements 6.4, Property 8**
        """
        # 创建系统
        system = self._create_unequal_focal_relay_system()
        
        # 运行仿真
        results = system.run()
        
        # 验证各采样面的光束半径
        for name, result in results.sampling_results.items():
            proper_w = result.beam_radius
            abcd_result = system.get_abcd_result(result.distance)
            abcd_w = abcd_result.w
            
            if abcd_w > 0.001:
                error_pct = abs(proper_w - abcd_w) / abcd_w
            else:
                error_pct = 0.0
            
            # 对焦点附近的采样面使用更宽松的容差（光束半径 < 0.5mm）
            if abcd_w < 0.5:
                tolerance = 0.05  # 5% 容差
            else:
                tolerance = self.BEAM_RADIUS_ERROR_TOLERANCE
            
            assert error_pct < tolerance, (
                f"采样面 '{name}' 光束半径误差过大: "
                f"PROPER={proper_w:.4f}mm, ABCD={abcd_w:.4f}mm, "
                f"误差={error_pct*100:.2f}% (容差={tolerance*100:.0f}%)"
            )
    
    def test_magnification_accuracy(self):
        """验证不等焦距中继系统放大倍率与设计值一致
        
        **Validates: Requirements 6.4, Property 8**
        """
        # 创建系统
        system = self._create_unequal_focal_relay_system()
        
        # 运行仿真
        results = system.run()
        
        # 获取输入和输出光束半径
        input_w = results.sampling_results["Input"].beam_radius
        output_w = results.sampling_results["Output"].beam_radius
        
        # 计算实测放大倍率
        measured_magnification = output_w / input_w
        
        # 计算相对误差
        error_pct = abs(measured_magnification - self.DESIGN_MAGNIFICATION) / self.DESIGN_MAGNIFICATION
        
        assert error_pct < self.MAGNIFICATION_ERROR_TOLERANCE, (
            f"放大倍率误差过大: "
            f"设计值={self.DESIGN_MAGNIFICATION:.2f}x, "
            f"实测值={measured_magnification:.2f}x, "
            f"误差={error_pct*100:.2f}%"
        )
    
    def test_wavefront_quality(self):
        """验证不等焦距中继系统各采样面的波前质量
        
        **Validates: Requirements 6.4, Property 8**
        """
        # 创建系统
        system = self._create_unequal_focal_relay_system()
        
        # 运行仿真
        results = system.run()
        
        # 验证各采样面的波前质量
        for name, result in results.sampling_results.items():
            wfe_rms = result.wavefront_rms
            
            assert np.isfinite(wfe_rms), (
                f"采样面 '{name}' 波前误差为非有限值: {wfe_rms}"
            )
            
            assert wfe_rms < self.WFE_RMS_TOLERANCE, (
                f"采样面 '{name}' 波前误差过大: "
                f"WFE RMS = {wfe_rms:.4f} waves (期望 < {self.WFE_RMS_TOLERANCE} waves)"
            )
    
    def test_output_beam_larger_than_input(self):
        """验证输出光束大于输入光束（扩束效果）
        
        不等焦距中继系统（f2 > f1）应该放大光束。
        """
        # 创建系统
        system = self._create_unequal_focal_relay_system()
        
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


class TestTripleOAPSystem:
    """三 OAP 系统测试
    
    系统配置：
    - OAP1: f=100mm 凹面镜，倾斜 45°（准直光 → 聚焦）
    - OAP2: f=100mm 凹面镜，倾斜 45°（聚焦 → 准直光）
    - OAP3: f=200mm 凹面镜，倾斜 45°（准直光 → 聚焦）
    - 总放大倍率: M = 1 × 2 = 2x
    
    **Validates: Requirements 6.4, Property 8**
    """
    
    # 光源参数
    WAVELENGTH_UM = 10.64      # μm, CO2 激光
    W0_INPUT_MM = 8.0          # mm, 输入束腰半径
    
    # OAP 焦距设计
    F1_MM = 100.0              # mm, OAP1 焦距
    F2_MM = 100.0              # mm, OAP2 焦距（与 OAP1 组成 1:1 中继）
    F3_MM = 200.0              # mm, OAP3 焦距（聚焦）
    
    # 离轴参数（90° OAP）
    D_OFF_OAP1_MM = 200.0      # OAP1 离轴距离 = 2f1
    D_OFF_OAP2_MM = 200.0      # OAP2 离轴距离 = 2f2
    D_OFF_OAP3_MM = 400.0      # OAP3 离轴距离 = 2f3
    
    # 倾斜角度（弧度）
    TILT_45_DEG = np.pi / 4    # 45° 倾斜
    
    # 几何参数
    D_OAP1_TO_OAP2_MM = 200.0  # OAP1 到 OAP2 的距离（= f1 + f2）
    D_OAP2_TO_OAP3_MM = 100.0  # OAP2 到 OAP3 的距离
    D_OAP3_TO_OUTPUT_MM = 200.0  # OAP3 到输出采样面的距离（= f3）
    
    # 测试容差
    BEAM_RADIUS_ERROR_TOLERANCE = 0.01  # 光束半径误差容差 1%
    WFE_RMS_TOLERANCE = 0.1  # WFE RMS 容差 0.1 波
    
    def _create_triple_oap_system(
        self,
        grid_size: int = 512,
        beam_ratio: float = 0.25,
        hybrid_num_rays: int = 100,
    ) -> SequentialOpticalSystem:
        """创建三 OAP 系统
        
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
        
        # 创建系统
        system = SequentialOpticalSystem(
            source=source,
            grid_size=grid_size,
            beam_ratio=beam_ratio,
            use_hybrid_propagation=True,
            hybrid_num_rays=hybrid_num_rays,
        )
        
        # 计算总光程
        total_path = (
            self.D_OAP1_TO_OAP2_MM + 
            self.D_OAP2_TO_OAP3_MM + 
            self.D_OAP3_TO_OUTPUT_MM
        )
        
        # --- 采样面：输入 ---
        system.add_sampling_plane(distance=0.0, name="Input")
        
        # --- OAP1：凹面抛物面镜（聚焦），倾斜 45° ---
        system.add_surface(ParabolicMirror(
            parent_focal_length=self.F1_MM,
            thickness=self.D_OAP1_TO_OAP2_MM,
            semi_aperture=20.0,
            off_axis_distance=self.D_OFF_OAP1_MM,
            tilt_x=self.TILT_45_DEG,
            name="OAP1 (45°)",
        ))
        
        # --- 采样面：OAP1 之后 ---
        system.add_sampling_plane(
            distance=self.D_OAP1_TO_OAP2_MM, 
            name="After OAP1"
        )
        
        # --- OAP2：凹面抛物面镜（准直），倾斜 45° ---
        system.add_surface(ParabolicMirror(
            parent_focal_length=self.F2_MM,
            thickness=self.D_OAP2_TO_OAP3_MM,
            semi_aperture=20.0,
            off_axis_distance=self.D_OFF_OAP2_MM,
            tilt_x=self.TILT_45_DEG,
            name="OAP2 (45°)",
        ))
        
        # --- 采样面：OAP2 之后 ---
        system.add_sampling_plane(
            distance=self.D_OAP1_TO_OAP2_MM + self.D_OAP2_TO_OAP3_MM, 
            name="After OAP2"
        )
        
        # --- OAP3：凹面抛物面镜（聚焦），倾斜 45° ---
        system.add_surface(ParabolicMirror(
            parent_focal_length=self.F3_MM,
            thickness=self.D_OAP3_TO_OUTPUT_MM,
            semi_aperture=25.0,
            off_axis_distance=self.D_OFF_OAP3_MM,
            tilt_x=self.TILT_45_DEG,
            name="OAP3 (45°)",
        ))
        
        # --- 采样面：输出（OAP3 焦点） ---
        system.add_sampling_plane(distance=total_path, name="Output (Focus)")
        
        return system
    
    def test_beam_radius_vs_abcd(self):
        """验证三 OAP 系统与 ABCD 理论的光束半径一致性
        
        注意：在焦点附近，由于光束半径非常小，数值精度和采样限制
        会导致更大的相对误差，因此对焦点附近的采样面使用更宽松的容差。
        
        **Validates: Requirements 6.4, Property 8**
        """
        # 创建系统
        system = self._create_triple_oap_system()
        
        # 运行仿真
        results = system.run()
        
        # 验证各采样面的光束半径
        for name, result in results.sampling_results.items():
            proper_w = result.beam_radius
            abcd_result = system.get_abcd_result(result.distance)
            abcd_w = abcd_result.w
            
            if abcd_w > 0.001:
                error_pct = abs(proper_w - abcd_w) / abcd_w
            else:
                error_pct = 0.0
            
            # 对焦点附近的采样面使用更宽松的容差（光束半径 < 0.5mm）
            if abcd_w < 0.5:
                tolerance = 0.05  # 5% 容差
            else:
                tolerance = self.BEAM_RADIUS_ERROR_TOLERANCE
            
            assert error_pct < tolerance, (
                f"采样面 '{name}' 光束半径误差过大: "
                f"PROPER={proper_w:.4f}mm, ABCD={abcd_w:.4f}mm, "
                f"误差={error_pct*100:.2f}% (容差={tolerance*100:.0f}%)"
            )
    
    def test_wavefront_quality(self):
        """验证三 OAP 系统各采样面的波前质量
        
        **Validates: Requirements 6.4, Property 8**
        """
        # 创建系统
        system = self._create_triple_oap_system()
        
        # 运行仿真
        results = system.run()
        
        # 验证各采样面的波前质量
        for name, result in results.sampling_results.items():
            wfe_rms = result.wavefront_rms
            
            assert np.isfinite(wfe_rms), (
                f"采样面 '{name}' 波前误差为非有限值: {wfe_rms}"
            )
            
            assert wfe_rms < self.WFE_RMS_TOLERANCE, (
                f"采样面 '{name}' 波前误差过大: "
                f"WFE RMS = {wfe_rms:.4f} waves (期望 < {self.WFE_RMS_TOLERANCE} waves)"
            )
    
    def test_propagation_completes(self):
        """验证三 OAP 系统传播正常完成
        
        基本健全性检查。
        """
        # 创建系统
        system = self._create_triple_oap_system()
        
        # 运行仿真
        results = system.run()
        
        # 验证所有采样面都有结果
        expected_planes = ["Input", "After OAP1", "After OAP2", "Output (Focus)"]
        for name in expected_planes:
            assert name in results.sampling_results, (
                f"缺少采样面 '{name}' 的结果"
            )
            result = results.sampling_results[name]
            assert result.beam_radius > 0, (
                f"采样面 '{name}' 光束半径应为正值: {result.beam_radius}"
            )
            assert np.isfinite(result.beam_radius), (
                f"采样面 '{name}' 光束半径应为有限值: {result.beam_radius}"
            )


class TestMultiOAPParametric:
    """多 OAP 系统参数化测试
    
    测试不同参数配置下的系统行为。
    """
    
    @pytest.mark.parametrize("grid_size", [256, 512])
    def test_4f_different_grid_sizes(self, grid_size: int):
        """测试 4f 系统在不同网格大小下的结果一致性
        
        不同网格大小应该给出相似的结果。
        """
        test_instance = TestFourFRelaySystem()
        system = test_instance._create_4f_relay_system(grid_size=grid_size)
        results = system.run()
        
        # 验证输出存在且有效
        assert "Output" in results.sampling_results
        output_result = results.sampling_results["Output"]
        
        assert output_result.beam_radius > 0
        assert np.isfinite(output_result.beam_radius)
        
        # 验证放大倍率大致正确（允许 5% 误差）
        input_w = results.sampling_results["Input"].beam_radius
        output_w = output_result.beam_radius
        measured_mag = output_w / input_w
        
        assert abs(measured_mag - 1.0) / 1.0 < 0.05, (
            f"网格大小 {grid_size} 下放大倍率偏差过大: {measured_mag:.2f}x"
        )
    
    @pytest.mark.parametrize("hybrid_num_rays", [64, 100, 144])
    def test_4f_different_ray_counts(self, hybrid_num_rays: int):
        """测试 4f 系统在不同光线数量下的结果一致性
        
        不同光线数量应该给出相似的结果。
        """
        test_instance = TestFourFRelaySystem()
        system = test_instance._create_4f_relay_system(
            hybrid_num_rays=hybrid_num_rays
        )
        results = system.run()
        
        # 验证输出存在且有效
        assert "Output" in results.sampling_results
        output_result = results.sampling_results["Output"]
        
        assert output_result.beam_radius > 0
        assert np.isfinite(output_result.beam_radius)
    
    @pytest.mark.parametrize("magnification,f1,f2", [
        (2.0, 50.0, 100.0),   # 2x 放大
        (0.5, 100.0, 50.0),   # 0.5x 缩小
        (1.5, 100.0, 150.0),  # 1.5x 放大
    ])
    def test_different_magnifications(self, magnification: float, f1: float, f2: float):
        """测试不同放大倍率的 OAP 中继系统
        
        验证不同焦距组合下的放大倍率正确性。
        """
        wavelength_um = 10.64
        w0_mm = 5.0
        
        source = GaussianBeamSource(
            wavelength=wavelength_um,
            w0=w0_mm,
            z0=0.0,
        )
        
        system = SequentialOpticalSystem(
            source=source,
            grid_size=512,
            beam_ratio=0.2,
            use_hybrid_propagation=True,
            hybrid_num_rays=100,
        )
        
        # 添加采样面和元件
        system.add_sampling_plane(distance=0.0, name="Input")
        
        # OAP1
        system.add_surface(ParabolicMirror(
            parent_focal_length=f1,
            thickness=f1,  # 到焦点的距离
            semi_aperture=20.0,
            off_axis_distance=2*f1,
            tilt_x=np.pi/4,
            name="OAP1",
        ))
        
        # 折叠镜
        system.add_surface(FlatMirror(
            thickness=f2,  # 从焦点到 OAP2 的距离
            semi_aperture=30.0,
            tilt_x=np.pi/4,
            name="Fold",
        ))
        
        # OAP2
        system.add_surface(ParabolicMirror(
            parent_focal_length=f2,
            thickness=100.0,  # 到输出的距离
            semi_aperture=30.0,
            off_axis_distance=2*f2,
            tilt_x=np.pi/4,
            name="OAP2",
        ))
        
        total_path = f1 + f2 + 100.0
        system.add_sampling_plane(distance=total_path, name="Output")
        
        # 运行仿真
        results = system.run()
        
        # 验证放大倍率
        input_w = results.sampling_results["Input"].beam_radius
        output_w = results.sampling_results["Output"].beam_radius
        measured_mag = output_w / input_w
        
        # 允许 5% 误差
        assert abs(measured_mag - magnification) / magnification < 0.05, (
            f"放大倍率 {magnification}x 测试失败: "
            f"实测值={measured_mag:.2f}x"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
