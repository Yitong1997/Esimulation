"""Source definitions for POP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Any

import numpy as np

from .core import GridSampling, PilotBeamParams


@dataclass
class GaussianSource:
    """Gaussian source definition."""

    wavelength_um: float
    w0_mm: float
    grid_size: int = 256
    physical_size_mm: Optional[float] = None
    z0_mm: float = 0.0
    beam_diam_fraction: Optional[float] = None

    def __post_init__(self) -> None:
        if self.wavelength_um <= 0:
            raise ValueError("wavelength_um must be positive")
        if self.w0_mm <= 0:
            raise ValueError("w0_mm must be positive")
        if not isinstance(self.grid_size, int) or self.grid_size <= 0:
            raise ValueError("grid_size must be a positive integer")
        if self.physical_size_mm is not None and self.physical_size_mm <= 0:
            raise ValueError("physical_size_mm must be positive")
        if self.beam_diam_fraction is not None and self.beam_diam_fraction <= 0:
            raise ValueError("beam_diam_fraction must be positive")

    @property
    def wavelength_mm(self) -> float:
        return self.wavelength_um * 1e-3

    def _resolve_beam_ratio(self) -> Tuple[float, float]:
        if self.physical_size_mm is None:
            beam_ratio = self.beam_diam_fraction if self.beam_diam_fraction is not None else 0.5
            physical_size_mm = 2.0 * self.w0_mm / beam_ratio
            return beam_ratio, physical_size_mm
        beam_ratio = 2.0 * self.w0_mm / self.physical_size_mm
        return beam_ratio, self.physical_size_mm

    def get_grid_sampling(self) -> GridSampling:
        beam_ratio, physical_size_mm = self._resolve_beam_ratio()
        return GridSampling.create(
            grid_size=self.grid_size,
            physical_size_mm=physical_size_mm,
            beam_ratio=beam_ratio,
        )

    def create_initial_wavefront(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, PilotBeamParams, Any]:
        import proper

        beam_ratio, physical_size_mm = self._resolve_beam_ratio()
        wavelength_m = self.wavelength_um * 1e-6
        beam_diameter_m = 2.0 * self.w0_mm * 1e-3

        wfo = proper.prop_begin(
            beam_diameter_m,
            wavelength_m,
            self.grid_size,
            beam_ratio,
        )

        wavelength_mm = self.wavelength_um * 1e-3
        z_r_mm = np.pi * self.w0_mm**2 / wavelength_mm
        z_mm = -self.z0_mm

        wfo.w0 = self.w0_mm * 1e-3
        wfo.z_Rayleigh = z_r_mm * 1e-3
        wfo.z = z_mm * 1e-3
        wfo.z_w0 = 0.0

        rayleigh_factor = proper.rayleigh_factor
        if abs(wfo.z - wfo.z_w0) < rayleigh_factor * wfo.z_Rayleigh:
            wfo.beam_type_old = "INSIDE_"
            wfo.reference_surface = "PLANAR"
        else:
            wfo.beam_type_old = "OUTSIDE"
            wfo.reference_surface = "SPHERI"

        sampling_mm = proper.prop_get_sampling(wfo) * 1e3
        coords = (np.arange(self.grid_size) - self.grid_size // 2) * sampling_mm
        x_grid, y_grid = np.meshgrid(coords, coords)
        r_sq = x_grid**2 + y_grid**2

        if z_r_mm > 0:
            w = self.w0_mm * np.sqrt(1.0 + (z_mm / z_r_mm) ** 2)
        else:
            w = self.w0_mm

        if abs(z_mm) < 1e-15:
            r_curv = np.inf
        else:
            r_curv = z_mm * (1.0 + (z_r_mm / z_mm) ** 2)

        amplitude = np.exp(-r_sq / w**2)

        if np.isinf(r_curv):
            phase = np.zeros_like(r_sq)
        else:
            k = 2.0 * np.pi / wavelength_mm
            phase = k * r_sq / (2.0 * r_curv)

        from pop.propagation.free_space import _compute_proper_reference_phase

        ref_phase = _compute_proper_reference_phase(wfo, GridSampling.from_proper(wfo))
        complex_amplitude = amplitude * np.exp(1j * (phase - ref_phase))

        wfo.wfarr = proper.prop_shift_center(complex_amplitude)

        pilot_beam = PilotBeamParams.from_gaussian_source(
            wavelength_um=self.wavelength_um,
            w0_mm=self.w0_mm,
            z0_mm=self.z0_mm,
        )

        return amplitude, phase, pilot_beam, wfo


def _fit_pilot_from_field(
    amplitude: np.ndarray,
    phase: np.ndarray,
    sampling_mm: float,
    wavelength_um: float,
    fit_threshold: float = 0.01,
) -> Tuple[float, float, float]:
    """从任意振幅/相位场拟合 Pilot Beam 参数 (w, R)。

    步骤:
        1. 从振幅场通过强度加权二阶矩拟合光束尺寸 w (D4σ 定义)
        2. 从相位场通过相邻像素相位梯度拟合曲率半径 R

    参数:
        amplitude: 2D 振幅场 (N×N)
        phase: 2D 相位场 (rad, N×N)
        sampling_mm: 空间采样间隔 (mm/pixel)
        wavelength_um: 波长 (μm)
        fit_threshold: 振幅阈值 — amplitude > threshold * max 的区域参与拟合

    返回:
        (w_fit_mm, R_fit_mm, w0_fit_mm):
        - w_fit_mm: 拟合光束半径 (mm, 1/e² 半径)
        - R_fit_mm: 拟合曲率半径 (mm, inf 表示平面波)
        - w0_fit_mm: 推导束腰半径 (mm)
    """
    from pop.analysis import compute_intensity_moments, _fit_phase_curvature

    grid_size = amplitude.shape[0]
    coords = (np.arange(grid_size) - grid_size // 2) * sampling_mm
    x_grid, y_grid = np.meshgrid(coords, coords)

    intensity = amplitude ** 2
    max_amp = float(np.max(amplitude))
    if max_amp <= 0:
        raise ValueError("振幅场最大值为零或负值，无法拟合 Pilot Beam")
    mask = amplitude > fit_threshold * max_amp

    # ——— Step 1: 从强度矩拟合光束尺寸 w ———
    moments = compute_intensity_moments(intensity, x_grid, y_grid, mask)
    if moments is None:
        raise ValueError("有效区域总强度为零，无法拟合")
    cx, cy, sigma_x, sigma_y, _ = moments

    # D4σ 定义: 光束半径 w = 2σ（1/e² 振幅定义）
    w_fit_mm = float(2.0 * np.sqrt(0.5 * (sigma_x ** 2 + sigma_y ** 2)))
    if w_fit_mm < 1e-12:
        raise ValueError(f"拟合光束尺寸过小 ({w_fit_mm:.2e} mm)，无法构造 Pilot Beam")

    # ——— Step 2: 从相位区域拟合曲率半径 R ———
    wavelength_mm = wavelength_um * 1e-3
    intensity_masked = np.where(mask, intensity, 0.0)

    # 复用 analysis 的鲁棒曲率拟合，直接开启 global_scan 以支持大像差下的远场折叠相位
    R_fit_mm = _fit_phase_curvature(
        phase=phase,
        x_grid=x_grid,
        y_grid=y_grid,
        weights=intensity_masked,
        centroid_x=cx,
        centroid_y=cy,
        wavelength_mm=wavelength_mm,
        refractive_index=1.0,
        enable_global_scan=True,
    )
    if np.isnan(R_fit_mm):
        R_fit_mm = np.inf

    # ——— Step 3: 从 (w, R) 推导 w0 ———
    if np.isinf(R_fit_mm):
        # 平面波 → 当前位置就是束腰
        w0_fit_mm = w_fit_mm
    else:
        # w² = w0² * (1 + (z/z_R)²), R = z * (1 + (z_R/z)²)
        # → z = R / (1 + (z_R/z)²) ... 需要迭代
        # 使用 q 参数直接推导:
        # 1/q = 1/R - j*λ/(π*n*w²)
        n = 1.0  # 默认空气
        inv_q_real = 1.0 / R_fit_mm
        inv_q_imag = -wavelength_mm / (np.pi * n * w_fit_mm ** 2)
        inv_q = complex(inv_q_real, inv_q_imag)
        if abs(inv_q) < 1e-30:
            w0_fit_mm = w_fit_mm
        else:
            q = 1.0 / inv_q
            z_R_fit = abs(np.imag(q))
            if z_R_fit > 0:
                w0_fit_mm = float(np.sqrt(wavelength_mm * z_R_fit / np.pi))
            else:
                w0_fit_mm = w_fit_mm

    print(
        f"[POP][CustomSource] Pilot Beam 拟合结果: "
        f"w={w_fit_mm:.4f} mm, R={'inf' if np.isinf(R_fit_mm) else f'{R_fit_mm:.2f}'} mm, "
        f"w0={w0_fit_mm:.4f} mm"
    )

    return w_fit_mm, R_fit_mm, w0_fit_mm


@dataclass
class CustomSource:
    """自定义复振幅输入源。

    用户提供 2D 振幅场、2D 相位场（完整物理相位）和空间采样间隔，
    框架自动拟合 Pilot Beam 参数并构建 PROPER wfo 对象。

    参数:
        wavelength_um: 波长 (μm) — 必填
        amplitude: 2D 振幅场 (N×N ndarray) — 必填
        phase: 2D 相位场 (rad, N×N ndarray, 完整物理相位) — 必填
        sampling_mm: 空间采样间隔 (mm/pixel) — 必填
        pilot_w0_mm: 手动指定束腰半径 (mm)。None = 自动拟合
        pilot_z0_mm: 手动指定束腰离 z=0 的距离 (mm)。None = 自动推导
        pilot_R_mm: 手动指定波前曲率半径 (mm)。None = 自动拟合
        fit_threshold: 振幅阈值比例，用于拟合时的有效区域识别
    """

    wavelength_um: float
    amplitude: np.ndarray
    phase: np.ndarray
    sampling_mm: float
    pilot_w0_mm: Optional[float] = None
    pilot_z0_mm: Optional[float] = None
    pilot_R_mm: Optional[float] = None
    fit_threshold: float = 0.01
    debug_plot: bool = False
    debug_save_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.wavelength_um <= 0:
            raise ValueError("wavelength_um 必须为正数")
        if self.sampling_mm <= 0:
            raise ValueError("sampling_mm 必须为正数")
        amp = np.asarray(self.amplitude, dtype=float)
        ph = np.asarray(self.phase, dtype=float)
        if amp.ndim != 2:
            raise ValueError(f"amplitude 必须是 2D 数组，当前 ndim={amp.ndim}")
        if ph.ndim != 2:
            raise ValueError(f"phase 必须是 2D 数组，当前 ndim={ph.ndim}")
        if amp.shape[0] != amp.shape[1]:
            raise ValueError(f"amplitude 必须是 N×N 正方阵，当前 shape={amp.shape}")
        if amp.shape != ph.shape:
            raise ValueError(
                f"amplitude 和 phase 形状不一致: {amp.shape} vs {ph.shape}"
            )
        self.amplitude = amp
        self.phase = ph

    @property
    def grid_size(self) -> int:
        return self.amplitude.shape[0]

    @property
    def physical_size_mm(self) -> float:
        return self.grid_size * self.sampling_mm

    @property
    def wavelength_mm(self) -> float:
        return self.wavelength_um * 1e-3

    def create_initial_wavefront(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, PilotBeamParams, Any]:
        """创建初始波前，与 GaussianSource 接口一致。

        返回:
            (amplitude, phase, pilot_beam, wfo)
        """
        import proper

        grid_size = self.grid_size
        wavelength_m = self.wavelength_um * 1e-6
        wavelength_mm = self.wavelength_mm
        sampling_m = self.sampling_mm * 1e-3
        physical_size_mm = self.physical_size_mm

        # ═══════════════════════════════════════════════════════════════
        # Step 1-2: 拟合或使用手动 Pilot Beam 参数
        # ═══════════════════════════════════════════════════════════════
        w_fit, R_fit, w0_fit = _fit_pilot_from_field(
            self.amplitude, self.phase, self.sampling_mm,
            self.wavelength_um, self.fit_threshold,
        )

        # 用户手动覆盖优先
        w0_mm = self.pilot_w0_mm if self.pilot_w0_mm is not None else w0_fit
        R_mm = self.pilot_R_mm if self.pilot_R_mm is not None else R_fit

        # ═══════════════════════════════════════════════════════════════
        # Step 3: 构造 PilotBeamParams (从 w_current 和 R 构造 q)
        # ═══════════════════════════════════════════════════════════════
        w_current = w_fit  # 当前位置的光束半径（拟合值）
        n = 1.0  # 默认空气

        if np.isinf(R_mm):
            # 平面波 → 当前位置就是束腰, q = 0 + j*z_R
            z_R_mm = np.pi * w0_mm ** 2 / wavelength_mm
            q = complex(0.0, z_R_mm)
        else:
            # 1/q = 1/R - j*λ/(π*n*w²)
            inv_q_real = 1.0 / R_mm
            inv_q_imag = -wavelength_mm / (np.pi * n * w_current ** 2)
            inv_q = complex(inv_q_real, inv_q_imag)
            q = 1.0 / inv_q

        pilot_beam = PilotBeamParams.from_q_parameter(
            q, self.wavelength_um, current_refractive_index=n,
        )

        # 用户手动覆盖 z0
        if self.pilot_z0_mm is not None:
            # 重新构造：用户给了 z0_mm,从 GaussianSource 约定出发
            pilot_beam = PilotBeamParams.from_gaussian_source(
                wavelength_um=self.wavelength_um,
                w0_mm=w0_mm,
                z0_mm=self.pilot_z0_mm,
            )

        print(
            f"[POP][CustomSource] 最终 Pilot Beam: "
            f"w0={pilot_beam.waist_radius_mm:.4f} mm, "
            f"w={pilot_beam.spot_size_mm:.4f} mm, "
            f"R={'inf' if np.isinf(pilot_beam.curvature_radius_mm) else f'{pilot_beam.curvature_radius_mm:.2f}'} mm, "
            f"z_R={pilot_beam.rayleigh_length_mm:.4f} mm, "
            f"z_waist={pilot_beam.waist_position_mm:.4f} mm"
        )

        # ═══════════════════════════════════════════════════════════════
        # Step 4: 构建 PROPER wfo 对象
        # ═══════════════════════════════════════════════════════════════
        beam_diameter_m = 2.0 * pilot_beam.waist_radius_mm * 1e-3
        beam_ratio = beam_diameter_m / (grid_size * sampling_m)
        # beam_ratio 可能超出合理范围，clamp 到 (0, 1]
        beam_ratio = max(min(beam_ratio, 1.0), 1e-6)

        wfo = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, beam_ratio)

        # 同步 Pilot Beam 参数到 wfo
        wfo.w0 = pilot_beam.waist_radius_mm * 1e-3
        wfo.z_Rayleigh = pilot_beam.rayleigh_length_mm * 1e-3
        # 用户的复振幅定义在 z=0 平面
        wfo.z = 0.0
        wfo.z_w0 = (pilot_beam.waist_position_mm) * 1e-3

        # 覆盖 dx 为用户提供的精确采样
        wfo._dx = sampling_m

        # 判断参考面
        rayleigh_factor = proper.rayleigh_factor
        if abs(wfo.z - wfo.z_w0) < rayleigh_factor * wfo.z_Rayleigh:
            wfo.beam_type_old = "INSIDE_"
            wfo.reference_surface = "PLANAR"
        else:
            wfo.beam_type_old = "OUTSIDE"
            wfo.reference_surface = "SPHERI"

        # 构建 wfarr
        # 用户输入的 phase 是完整物理相位 (mm 空间 → 需要转换为 m 空间)
        # phase 本身是 rad，与空间单位无关。但参考球面计算需要 m 单位坐标
        from pop.propagation.free_space import _compute_proper_reference_phase

        ref_phase = _compute_proper_reference_phase(wfo, GridSampling.from_proper(wfo))
        complex_amplitude = self.amplitude * np.exp(1j * (self.phase - ref_phase))

        wfo.wfarr = proper.prop_shift_center(complex_amplitude)

        if self.debug_plot or self.debug_save_path:
            _plot_custom_source_diagnostics(
                amplitude=self.amplitude,
                phase=self.phase,
                sampling_mm=self.sampling_mm,
                wavelength_um=self.wavelength_um,
                fit_threshold=self.fit_threshold,
                w_fit=w_fit,
                R_fit=R_fit,
                pilot_beam=pilot_beam,
                wfo=wfo,
                save_path=self.debug_save_path
            )

        return self.amplitude.copy(), self.phase.copy(), pilot_beam, wfo


@dataclass
class ZbfSource:
    """Zemax Beam File source adapter.

    The default mode keeps the ZBF residual field in PROPER while exposing the
    corresponding physical phase on the returned source arrays.  When the ZBF
    plane belongs to a reflected branch, ``coordinate_z_axis`` and
    ``propagation_direction`` define the sign needed to convert the ZBF local
    reference into the beam-following q coordinate.
    """

    zbf_path: str | Path
    reference_mode: str = "reference_relative"
    allow_polarized_ex_only: bool = False
    allow_astigmatic_approximation: bool = False
    radial_rtol: float = 1e-4
    radial_atol: float = 1e-9
    coordinate_z_axis: Optional[np.ndarray] = None
    propagation_direction: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        mode = str(self.reference_mode).strip().lower()
        if mode not in {"reference_relative", "physical"}:
            raise ValueError("reference_mode must be 'reference_relative' or 'physical'")
        self.reference_mode = mode
        self.zbf_path = Path(self.zbf_path)
        if self.coordinate_z_axis is not None:
            axis = np.asarray(self.coordinate_z_axis, dtype=float)
            if axis.shape != (3,):
                raise ValueError("coordinate_z_axis must be shape (3,)")
            norm = np.linalg.norm(axis)
            if norm < 1e-12:
                raise ValueError("coordinate_z_axis cannot be zero")
            self.coordinate_z_axis = axis / norm
        if self.propagation_direction is not None:
            direction = np.asarray(self.propagation_direction, dtype=float)
            if direction.shape != (3,):
                raise ValueError("propagation_direction must be shape (3,)")
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                raise ValueError("propagation_direction cannot be zero")
            self.propagation_direction = direction / norm

    @property
    def wavelength_um(self) -> float:
        from .io.zbf import read_zbf

        return read_zbf(self.zbf_path).wavelength * 1e3

    def create_initial_wavefront(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, PilotBeamParams, Any]:
        import proper

        from .io.zbf import (
            read_zbf,
            zbf_physical_field_pop_convention_for_axis,
            zbf_reference_relative_field_pop_convention,
        )

        zbf = read_zbf(self.zbf_path)
        if zbf.is_polarized and not self.allow_polarized_ex_only:
            raise ValueError("ZBF polarized input is not supported by default")
        self._validate_radial_header(zbf)

        axis_sign = self._coordinate_axis_sign()
        pilot_beam = self._pilot_from_zbf(zbf, axis_sign=axis_sign)
        reference_relative_field = zbf_reference_relative_field_pop_convention(zbf)
        physical_field = zbf_physical_field_pop_convention_for_axis(
            zbf,
            axis_sign=axis_sign,
        )

        if self.reference_mode == "physical":
            wfarr_field = physical_field
        else:
            wfarr_field = reference_relative_field

        amplitude = np.abs(physical_field)
        phase = np.angle(physical_field)

        sampling_m = zbf.dx * 1e-3
        beam_diameter_m = 2.0 * pilot_beam.waist_radius_mm * 1e-3
        beam_ratio = beam_diameter_m / (zbf.nx * sampling_m)
        beam_ratio = max(min(float(beam_ratio), 1.0), 1e-6)

        wfo = proper.prop_begin(
            beam_diameter_m,
            zbf.wavelength * 1e-3,
            zbf.nx,
            beam_ratio,
        )
        wfo.w0 = pilot_beam.waist_radius_mm * 1e-3
        wfo.z_Rayleigh = pilot_beam.rayleigh_length_mm * 1e-3
        wfo.z = float(axis_sign * zbf.zx) * 1e-3
        wfo.z_w0 = 0.0
        wfo._dx = sampling_m
        if abs(zbf.zx) < proper.rayleigh_factor * max(abs(zbf.rx), 1e-15):
            wfo.beam_type_old = "INSIDE_"
            wfo.reference_surface = "PLANAR"
        else:
            wfo.beam_type_old = "OUTSIDE"
            wfo.reference_surface = "SPHERI"
        wfo.wfarr = proper.prop_shift_center(wfarr_field)
        return amplitude, phase, pilot_beam, wfo

    def _validate_radial_header(self, zbf: Any) -> None:
        pairs = (
            ("dx", zbf.dx, zbf.dy),
            ("z", zbf.zx, zbf.zy),
            ("rayleigh", zbf.rx, zbf.ry),
            ("waist", zbf.wx, zbf.wy),
        )
        for name, x_value, y_value in pairs:
            if not np.isclose(
                x_value,
                y_value,
                rtol=self.radial_rtol,
                atol=self.radial_atol,
            ):
                if not self.allow_astigmatic_approximation:
                    raise ValueError(
                        f"ZBF astigmatic header is not supported: {name} differs"
                    )

    def _coordinate_axis_sign(self) -> float:
        if self.coordinate_z_axis is None:
            return 1.0
        direction = self.propagation_direction
        if direction is None:
            direction = np.array([0.0, 0.0, 1.0])
        alignment = float(np.dot(direction, self.coordinate_z_axis))
        if abs(alignment) < 1e-8:
            raise ValueError(
                "ZBF coordinate_z_axis is nearly orthogonal to propagation_direction"
            )
        return 1.0 if alignment > 0.0 else -1.0

    def _pilot_from_zbf(self, zbf: Any, *, axis_sign: float = 1.0) -> PilotBeamParams:
        wavelength_um = zbf.wavelength * 1e3
        waist_radius_mm = float(zbf.wx)
        rayleigh_mm = float(zbf.rx)
        q = complex(float(axis_sign) * float(zbf.zx), rayleigh_mm)
        spot_size_mm = waist_radius_mm
        if abs(rayleigh_mm) > 1e-15:
            spot_size_mm = waist_radius_mm * np.sqrt(
                1.0 + (float(zbf.zx) / rayleigh_mm) ** 2
            )
        curvature_radius_mm = np.inf
        if abs(float(zbf.zx)) > 1e-15:
            curvature_radius_mm = float(axis_sign) * float(zbf.zx) * (
                1.0 + (rayleigh_mm / float(zbf.zx)) ** 2
            )
        return PilotBeamParams(
            wavelength_um=wavelength_um,
            waist_radius_mm=waist_radius_mm,
            waist_position_mm=-float(axis_sign) * float(zbf.zx),
            curvature_radius_mm=curvature_radius_mm,
            spot_size_mm=float(spot_size_mm),
            q_parameter=q,
            current_refractive_index=float(zbf.index),
        )

def _plot_custom_source_diagnostics(
    amplitude: np.ndarray,
    phase: np.ndarray,
    sampling_mm: float,
    wavelength_um: float,
    fit_threshold: float,
    w_fit: float,
    R_fit: float,
    pilot_beam: PilotBeamParams,
    wfo: Any,
    save_path: Optional[str] = None
) -> None:
    """内部辅助: 绘制 CustomSource 初始场诊断画板。"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.patches import Ellipse
    except ImportError:
        print("[POP] 警告: 未安装 matplotlib, 无法生成诊断绘图。")
        return

    # 中文字体设置
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"CustomSource Initial Wavefront Diagnostics\n"
        f"Pilot: w0={pilot_beam.waist_radius_mm:.3f} mm, "
        f"R={'inf' if np.isinf(pilot_beam.curvature_radius_mm) else f'{pilot_beam.curvature_radius_mm:.1f}'} mm | "
        f"PROPER Ref Surface: {wfo.reference_surface}",
        fontsize=16, fontweight='bold'
    )

    grid_size = amplitude.shape[0]
    coords = (np.arange(grid_size) - grid_size // 2) * sampling_mm
    extent = [coords[0], coords[-1], coords[0], coords[-1]]
    x_grid, y_grid = np.meshgrid(coords, coords)

    intensity = amplitude ** 2
    max_amp = float(np.max(amplitude))
    mask = amplitude > fit_threshold * max_amp

    # S1: Amplitude & Waists
    ax = axes[0, 0]
    im = ax.imshow(amplitude, extent=extent, cmap='viridis', origin='lower')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.contour(coords, coords, mask, levels=[0.5], colors='white', linewidths=1, alpha=0.5)
    # 画出拟合得到的 1/e^2 光斑覆盖范围 (w_fit)
    circ = Ellipse((0, 0), width=w_fit*2, height=w_fit*2, edgecolor='red',
                   facecolor='none', linestyle='--', linewidth=2)
    ax.add_patch(circ)
    ax.set_title("S1: Amplitude & Fitting Mask & w_fit(red)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

    # S2: Raw Phase
    ax = axes[0, 1]
    # 强制将原始相位置于 [-pi, pi] 间以便 cyclic colormap 渲染，用户也可能传了未包裹场
    im = ax.imshow(np.angle(np.exp(1j*phase)), extent=extent, cmap='twilight', origin='lower',
                   vmin=-np.pi, vmax=np.pi)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("S2: Raw Wrapped Phase (Input)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

    # S3: Phase Gradient Vs R_fit
    ax = axes[0, 2]
    # 使用后向差分避免 Unwrap
    phasor = np.exp(1j * phase)
    d_phasor_x = phasor[:, 1:] * np.conj(phasor[:, :-1])
    grad_x = np.angle(d_phasor_x) / sampling_mm
    x_mid = 0.5 * (x_grid[:, 1:] + x_grid[:, :-1])
    mask_mid = mask[:, 1:] & mask[:, :-1]
    
    valid_idx = np.where(mask_mid.ravel())[0]
    if len(valid_idx) > 2000:
        valid_idx = np.random.choice(valid_idx, 2000, replace=False)
    
    ax.scatter(x_mid.ravel()[valid_idx], grad_x.ravel()[valid_idx],
               s=2, alpha=0.5, label='Raw dPhi/dx', color='blue')
    
    k = 2.0 * np.pi / (wavelength_um * 1e-3)
    x_line = np.linspace(np.min(x_mid), np.max(x_mid), 100)
    if not np.isinf(R_fit) and R_fit != 0:
        y_line = (k / R_fit) * x_line
    else:
        y_line = np.zeros_like(x_line)
        
    ax.plot(x_line, y_line, color='red', linewidth=2, label=f'Fit R={R_fit:.1f}mm')
    ax.set_title("S3: Gradient Scatter vs Parabolic Fit")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("Phase Gradient (rad/mm)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Calculate what goes into PROPER
    import proper
    wfarr_shifted = proper.prop_shift_center(wfo.wfarr)
    residual_phase = np.angle(wfarr_shifted)

    # S4: PROPER reference phase generated through PROPER's own q-phase routine.
    from pop.propagation.free_space import _compute_proper_reference_phase

    ref_phase_analytical = _compute_proper_reference_phase(
        wfo,
        GridSampling.from_proper(wfo),
    )

    ax = axes[1, 0]
    # S4 (理论参考球面): 必须显示其本来的物理尺度（绝对是不包裹的连续面）
    # 以便直观看出 PROPER 减去了多么庞大的全局曲率包裹
    im = ax.imshow(ref_phase_analytical, extent=extent, cmap='viridis', origin='lower')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("S4: Analytical Ref Phase (UNWRAPPED)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

    # S5: Residual Phase
    ax = axes[1, 1]
    # 不强制指定 vmin, vmax。
    # 如果成功剥离了参考曲面，残差应该非常平缓，不会触碰到 π 的包裹边界，视觉上达到“被解包裹变平坦”的效果。
    # 如果极度失真残差巨大，则能看到残存的包裹断崖。
    im = ax.imshow(residual_phase, extent=extent, cmap='twilight', origin='lower')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("S5: Residual Phase in PROPER wfarr")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

    # S6: Nyquist Aliasing Risk
    ax = axes[1, 2]
    res_phasor = np.exp(1j * residual_phase)
    # 取相邻像素的绝对相位差
    dx_res = np.abs(np.angle(res_phasor[:, 1:] * np.conj(res_phasor[:, :-1])))
    dy_res = np.abs(np.angle(res_phasor[1:, :] * np.conj(res_phasor[:-1, :])))
    risk_x = np.zeros_like(residual_phase)
    risk_y = np.zeros_like(residual_phase)
    risk_x[:, :-1] = dx_res
    risk_y[:-1, :] = dy_res
    
    # 重新添加缺失的 risk_map
    risk_map = np.maximum(risk_x, risk_y) / np.pi
    risk_map[~mask] = np.nan
    valid_risks = risk_map[mask].ravel()

    ax.hist(valid_risks, bins=50, color='teal', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2.0, label='Nyquist Limit')
    
    if len(valid_risks) > 0:
        max_risk = np.max(valid_risks)
        ax.axvline(x=max_risk, color='orange', linestyle=':', linewidth=2.0, label=f'Max Risk: {max_risk:.2f}')
        
    ax.set_title("S6: Aliasing Risk Histogram")
    ax.set_xlabel("Nyquist Limit Fraction (|dPhi| / pi)")
    ax.set_ylabel("Pixel Count")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"[POP][CustomSource] Diagnostic plots saved to {save_path}")
    else:
        plt.show(block=False)
        plt.pause(1.0)
    plt.close(fig)
