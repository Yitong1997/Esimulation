"""
状态转换器模块

本模块实现振幅/相位、PROPER 复振幅和 Pilot Beam 参考之间的转换。

核心功能：
1. 从 PROPER 提取振幅和相位（使用 PROPER 参考面相位重建绝对相位）
2. 将振幅和相位写入 PROPER 对象（计算相对于 PROPER 参考面的残差）
3. 计算 PROPER 参考面相位
4. 计算 Pilot Beam 参考相位（用于几何光线追迹时的解包裹）

重要：仿真波前使用振幅和相位分离存储，相位为非折叠实数。
这避免了复数形式 exp(1j*φ) 隐含的相位折叠问题。

**Validates: Requirements 5.1-5.5, 9.1-9.5**
"""

from __future__ import annotations

from typing import Any, Tuple, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
import warnings

from .data_models import PilotBeamParams, GridSampling
from .exceptions import PhaseUnwrappingError

if TYPE_CHECKING:
    pass


class StateConverter:
    """状态转换器
    
    在振幅/相位（分离存储）和 PROPER 复振幅（相对于参考面的残差）
    之间进行转换。
    
    关键概念：
    - PROPER 参考面相位：φ_ref = +k × r² / (2 × R_ref)（正号！）
    - Pilot Beam 相位：φ_pilot = +k × r² / (2 × R_pilot)（正号！）
    - 两者符号相同，但曲率半径公式不同：
      - PROPER: R_ref = z - z_w0（远场近似）
      - Pilot Beam: R = z × (1 + (z_R/z)²)（严格公式）
    
    数据流：
    - 写入 PROPER (SPHERI): wfarr = 仿真复振幅 × exp(-i × φ_ref)
    - 读取 PROPER (SPHERI): 完整相位 = PROPER相位 + φ_ref
    
    属性:
        wavelength_um: 波长 (μm)
    
    **Validates: Requirements 5.1-5.5, 9.1-9.5**
    """
    
    def __init__(self, wavelength_um: float) -> None:
        """初始化状态转换器
        
        参数:
            wavelength_um: 波长 (μm)
        """
        self._wavelength_um = wavelength_um
    
    @property
    def wavelength_um(self) -> float:
        """波长 (μm)"""
        return self._wavelength_um
    
    @property
    def wavelength_mm(self) -> float:
        """波长 (mm)"""
        return self._wavelength_um * 1e-3

    # =========================================================================
    # PROPER 参考面相位计算
    # =========================================================================
    
    def compute_proper_reference_phase(
        self,
        wfo: Any,
        grid_sampling: GridSampling,
    ) -> NDArray[np.floating]:
        """计算 PROPER 参考面相位
        
        PROPER 参考面类型：
        - "PLANAR"：平面参考，相位为零（在瑞利距离内）
        - "SPHERI"：球面参考，φ_ref = +k × r² / (2 × R_ref)（在瑞利距离外）
        
        注意：PROPER 参考面相位使用正号！
        - R_ref = z - z_w0（PROPER 远场近似曲率半径）
        
        参数:
            wfo: PROPER 波前对象
            grid_sampling: 网格采样信息
        
        返回:
            PROPER 参考面相位网格 (弧度)
        
        **Validates: Requirements 5.1, 9.1**
        """
        n = grid_sampling.grid_size
        
        if wfo.reference_surface == "PLANAR":
            return np.zeros((n, n))
        
        R_ref_m = wfo.z - wfo.z_w0
        
        if abs(R_ref_m) < 1e-10:
            return np.zeros((n, n))
        
        X_mm, Y_mm = grid_sampling.get_coordinate_arrays()
        r_sq_m = (X_mm * 1e-3)**2 + (Y_mm * 1e-3)**2
        
        k = 2 * np.pi / wfo.lamda
        # 正号！参考球面相位
        proper_ref_phase = k * r_sq_m / (2 * R_ref_m)
        
        return proper_ref_phase

    # =========================================================================
    # Pilot Beam 参考相位计算
    # =========================================================================
    
    def compute_pilot_beam_phase(
        self,
        pilot_beam_params: PilotBeamParams,
        grid_sampling: GridSampling,
    ) -> NDArray[np.floating]:
        """计算 Pilot Beam 参考相位
        
        公式: φ_pilot(r) = k × r² / (2 × R)（正号！）
        
        参数:
            pilot_beam_params: Pilot Beam 参数
            grid_sampling: 网格采样信息
        
        返回:
            Pilot Beam 参考相位网格 (弧度)
        
        **Validates: Requirements 8.5, 8.6, 8.7**
        """
        return pilot_beam_params.compute_phase_grid(
            grid_sampling.grid_size,
            grid_sampling.physical_size_mm,
        )
    
    # =========================================================================
    # PROPER → 振幅/相位（从 PROPER 提取）
    # =========================================================================
    
    def proper_to_amplitude_phase(
        self,
        wfo: Any,
        grid_sampling: GridSampling,
        pilot_beam_params: PilotBeamParams = None,
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """从 PROPER 提取振幅和相位
        
        流程:
        1. 从 PROPER 提取残差相位（相对于 PROPER 参考面，折叠的）
        2. 计算 PROPER 参考面相位
        3. 重建绝对相位 = PROPER 参考面相位 + 残差相位（仍然是折叠的）
        4. 使用 Pilot Beam 解包裹得到非折叠相位
        
        参数:
            wfo: PROPER 波前对象
            grid_sampling: 网格采样信息
            pilot_beam_params: Pilot Beam 参数（必须提供，用于解包裹）
        
        返回:
            (amplitude, phase) 元组
            - amplitude: 振幅网格（实数，非负）
            - phase: 相位网格（实数，非折叠，弧度）
        
        **Validates: Requirements 5.1, 5.2, 5.3**
        """
        import proper
        
        amplitude = proper.prop_get_amplitude(wfo)
        residual_phase = proper.prop_get_phase(wfo)  # 折叠相位 [-π, π]
        
        proper_ref_phase = self.compute_proper_reference_phase(wfo, grid_sampling)
        wrapped_phase = proper_ref_phase + residual_phase  # 仍然是折叠的
        
        # 使用 Pilot Beam 解包裹
        if pilot_beam_params is not None:
            # 计算 Pilot Beam 参考相位（非折叠）
            pilot_phase = self.compute_pilot_beam_phase(pilot_beam_params, grid_sampling)
            # 解包裹公式: T_unwrapped = T_pilot + angle(exp(1j * (T - T_pilot)))
            phase_diff = wrapped_phase - pilot_phase
            unwrapped_phase = pilot_phase + np.angle(np.exp(1j * phase_diff))
            
            self._check_residual_phase_range(residual_phase, amplitude)
            
            return amplitude, unwrapped_phase
        else:
            # 如果没有提供 Pilot Beam 参数，返回折叠相位（不推荐）
            warnings.warn(
                "未提供 pilot_beam_params，返回的相位可能是折叠的。"
                "建议始终提供 pilot_beam_params 以获得正确的非折叠相位。",
                UserWarning,
            )
            return amplitude, wrapped_phase

    # 向后兼容方法
    def proper_to_simulation(
        self,
        wfo: Any,
        grid_sampling: GridSampling,
        pilot_beam_params: PilotBeamParams = None,
    ) -> NDArray[np.complexfloating]:
        """从 PROPER 提取仿真复振幅（向后兼容）
        
        警告：此方法已废弃，请使用 proper_to_amplitude_phase()。
        
        返回:
            仿真复振幅（复数形式，会有相位折叠）
        """
        amplitude, phase = self.proper_to_amplitude_phase(
            wfo, grid_sampling, pilot_beam_params
        )
        return amplitude * np.exp(1j * phase)

    # =========================================================================
    # 振幅/相位 → PROPER（写入 PROPER）
    # =========================================================================
    
    def amplitude_phase_to_proper(
        self,
        amplitude: NDArray[np.floating],
        phase: NDArray[np.floating],
        grid_sampling: GridSampling,
        pilot_beam_params: PilotBeamParams = None,
    ) -> Any:
        """将振幅和相位写入 PROPER 对象
        
        流程:
        1. 使用 prop_begin 正确初始化 PROPER 对象
        2. 计算 PROPER 参考面相位
        3. 计算残差相位 = 输入相位 - PROPER 参考面相位
        4. 将残差写入 PROPER（移到 FFT 坐标系）
        
        参数:
            amplitude: 振幅网格（实数，非负）
            phase: 相位网格（实数，非折叠，弧度）
            grid_sampling: 网格采样信息
            pilot_beam_params: Pilot Beam 参数（可选）
        
        返回:
            PROPER 波前对象
        
        **Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5**
        """
        import proper
        
        beam_diameter_m = grid_sampling.physical_size_mm * 1e-3
        wavelength_m = self._wavelength_um * 1e-6
        
        if pilot_beam_params is not None:
            w0_m = pilot_beam_params.waist_radius_mm * 1e-3
            beam_diam_fraction = (2 * w0_m) / beam_diameter_m
            beam_diam_fraction = max(0.1, min(0.9, beam_diam_fraction))
        else:
            beam_diam_fraction = grid_sampling.beam_ratio
        
        wfo = proper.prop_begin(
            beam_diameter_m,
            wavelength_m,
            grid_sampling.grid_size,
            beam_diam_fraction,
        )
        
        if pilot_beam_params is not None:
            self._sync_gaussian_params(wfo, pilot_beam_params)
        
        proper_ref_phase = self.compute_proper_reference_phase(wfo, grid_sampling)
        residual_phase = phase - proper_ref_phase
        
        self._check_residual_phase_range(residual_phase, amplitude)
        
        residual_field = amplitude * np.exp(1j * residual_phase)
        wfo.wfarr = proper.prop_shift_center(residual_field)
        
        return wfo

    # 向后兼容方法
    def simulation_to_proper(
        self,
        simulation_amplitude: NDArray[np.complexfloating],
        grid_sampling: GridSampling,
        pilot_beam_params: PilotBeamParams = None,
    ) -> Any:
        """将仿真复振幅写入 PROPER 对象（向后兼容）
        
        警告：此方法已废弃，请使用 amplitude_phase_to_proper()。
        """
        amplitude = np.abs(simulation_amplitude)
        phase = np.angle(simulation_amplitude)
        return self.amplitude_phase_to_proper(
            amplitude, phase, grid_sampling, pilot_beam_params
        )
    
    def _sync_gaussian_params(
        self,
        wfo: Any,
        pilot_beam_params: PilotBeamParams,
    ) -> None:
        """同步 PROPER 高斯光束参数与 Pilot Beam 参数
        
        PROPER 坐标约定：
        - wfo.z: 当前位置（PROPER 内部坐标，prop_begin 后为 0）
        - wfo.z_w0: 束腰位置（PROPER 内部坐标）
        - z - z_w0: 当前位置相对于束腰的距离
        
        PilotBeamParams 约定：
        - waist_position_mm: 束腰相对于当前位置的距离
        - 负值表示束腰在当前位置之前
        
        转换关系：
        - 当前位置相对于束腰的距离 = -waist_position_mm
        - z_w0 = z - (-waist_position_mm) = z + waist_position_mm
        """
        import proper
        
        wfo.w0 = pilot_beam_params.waist_radius_mm * 1e-3
        wfo.z_Rayleigh = pilot_beam_params.rayleigh_length_mm * 1e-3
        # 正确的转换：z_w0 = z + waist_position_mm
        # 因为 waist_position_mm 是束腰相对于当前位置的距离（负值表示在之前）
        wfo.z_w0 = wfo.z + pilot_beam_params.waist_position_mm * 1e-3
        
        rayleigh_factor = proper.rayleigh_factor
        
        if abs(wfo.z_w0 - wfo.z) < rayleigh_factor * wfo.z_Rayleigh:
            wfo.beam_type_old = "INSIDE_"
            wfo.reference_surface = "PLANAR"
        else:
            wfo.beam_type_old = "OUTSIDE"
            wfo.reference_surface = "SPHERI"
    
    # =========================================================================
    # Pilot Beam 解包裹（用于几何光线追迹）
    # =========================================================================
    
    def unwrap_with_pilot_beam(
        self,
        wrapped_phase: NDArray[np.floating],
        pilot_beam_params: PilotBeamParams,
        grid_sampling: GridSampling,
    ) -> NDArray[np.floating]:
        """使用 Pilot Beam 参考相位解包裹
        
        公式: T_unwrapped = T_pilot + angle(exp(1j * (T - T_pilot)))
        
        参数:
            wrapped_phase: 可能折叠的相位 (弧度)
            pilot_beam_params: Pilot Beam 参数
            grid_sampling: 网格采样信息
        
        返回:
            解包裹后的相位 (弧度)
        
        **Validates: Requirements 5.1, 5.2**
        """
        pilot_phase = self.compute_pilot_beam_phase(pilot_beam_params, grid_sampling)
        phase_diff = wrapped_phase - pilot_phase
        unwrapped_phase = pilot_phase + np.angle(np.exp(1j * phase_diff))
        return unwrapped_phase

    # =========================================================================
    # 验证方法
    # =========================================================================
    
    def _check_residual_phase_range(
        self,
        residual_phase: NDArray[np.floating],
        amplitude: NDArray[np.floating],
        threshold: float = np.pi / 2,
    ) -> None:
        """检查残差相位是否在合理范围内
        
        参数:
            residual_phase: 残差相位 (弧度)
            amplitude: 振幅（用于确定有效区域）
            threshold: 警告阈值 (弧度)，默认 π/2
        
        **Validates: Requirements 5.4, 9.4**
        """
        valid_mask = amplitude > 0.01 * np.max(amplitude)
        
        if not np.any(valid_mask):
            return
        
        max_residual = np.max(np.abs(residual_phase[valid_mask]))
        
        if max_residual > threshold:
            warnings.warn(
                f"残差相位过大: {max_residual:.2f} rad > {threshold:.2f} rad。"
                f"这可能导致 FFT 传播时的采样问题。",
                UserWarning,
            )
    
    def check_gaussian_params_sync(
        self,
        wfo: Any,
        pilot_beam_params: PilotBeamParams,
        rtol: float = 0.01,
    ) -> bool:
        """检查 PROPER 和 Pilot Beam 的高斯光束参数是否同步
        
        参数:
            wfo: PROPER 波前对象
            pilot_beam_params: Pilot Beam 参数
            rtol: 相对容差，默认 1%
        
        返回:
            True 如果参数同步，False 否则
        
        **Validates: Requirements 5.5, 9.5**
        """
        w0_proper_mm = wfo.w0 * 1e3
        z_R_proper_mm = wfo.z_Rayleigh * 1e3
        
        w0_pilot_mm = pilot_beam_params.waist_radius_mm
        z_R_pilot_mm = pilot_beam_params.rayleigh_length_mm
        
        w0_error = abs(w0_proper_mm - w0_pilot_mm) / w0_pilot_mm if w0_pilot_mm > 0 else 0
        z_R_error = abs(z_R_proper_mm - z_R_pilot_mm) / z_R_pilot_mm if z_R_pilot_mm > 0 else 0
        
        is_synced = w0_error <= rtol and z_R_error <= rtol
        
        if not is_synced:
            warnings.warn(
                f"高斯光束参数不同步: "
                f"w0 误差 {w0_error*100:.1f}%, z_R 误差 {z_R_error*100:.1f}%",
                UserWarning,
            )
        
        return is_synced
    
    # 向后兼容方法
    def unwrap_simulation_amplitude(
        self,
        simulation_amplitude: NDArray[np.complexfloating],
        pilot_beam_params: PilotBeamParams,
        grid_sampling: GridSampling,
    ) -> NDArray[np.complexfloating]:
        """对仿真复振幅进行相位解包裹（向后兼容）
        
        警告：此方法已废弃。新代码应使用分离的振幅和相位。
        """
        amplitude = np.abs(simulation_amplitude)
        phase = np.angle(simulation_amplitude)
        
        unwrapped_phase = self.unwrap_with_pilot_beam(
            phase, pilot_beam_params, grid_sampling
        )
        
        return amplitude * np.exp(1j * unwrapped_phase)
