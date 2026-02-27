"""
自适应光学（AO）仿真与控制模块

基于 Zemax OpticStudio 的 ZOS-API（通过 zospy 库），提供从系统初始化、
变形镜（DM）控制、光场数据提取到结果可视化的完整工作流。
所有数据接口使用 numpy.ndarray，与 Zemax 底层 COM 对象彻底解耦。
"""

import os
import warnings

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import zospy as zp
from zospy.analyses.physicaloptics import PhysicalOpticsPropagation

# ---------------------------------------------------------------------------
# 配置 matplotlib 中文字体（Windows 使用微软雅黑，回退到 SimHei / STSong）
# ---------------------------------------------------------------------------
for font_name in ["Microsoft YaHei", "SimHei", "STSong"]:
    if font_name in [f.name for f in mpl.font_manager.fontManager.ttflist]:
        mpl.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
        break
mpl.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


# ---------------------------------------------------------------------------
# 1. 系统初始化与连接
# ---------------------------------------------------------------------------

def init_system(zmx_path: str) -> tuple:
    """连接 Zemax 并加载光学系统文件。

    参数：
        zmx_path: .zmx 文件的绝对或相对路径

    返回：
        (zos, oss) 元组，zos 用于后续 cleanup，oss 用于所有操作

    异常：
        FileNotFoundError: 文件路径不存在
    """
    abs_path = os.path.abspath(zmx_path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"找不到 Zemax 文件：{abs_path}")

    zos = zp.ZOS()
    oss = zos.connect()
    oss.load(abs_path)
    return zos, oss


# ---------------------------------------------------------------------------
# 2. 面类型配置（DM 建模）
# ---------------------------------------------------------------------------

def setup_dm_surface(oss, surface_idx: int) -> None:
    """将指定面设置为 Zernike Standard Phase 类型，用于建模变形镜。

    参数：
        oss: OpticStudioSystem 实例
        surface_idx: 面在 LDE 中的索引（0 = Object，最后 = Image）

    异常：
        IndexError: surface_idx 超出 LDE 有效范围
    """
    # 校验面索引范围
    num_surfaces = oss.LDE.NumberOfSurfaces
    if surface_idx < 0 or surface_idx > num_surfaces - 1:
        raise IndexError(
            f"面索引 {surface_idx} 超出有效范围 [0, {num_surfaces - 1}]"
        )

    # 获取面对象并更改类型为 Zernike Standard Phase
    surface = oss.LDE.GetSurfaceAt(surface_idx)
    zp.functions.lde.surface_change_type(
        surface,
        zp.constants.Editors.LDE.SurfaceType.ZernikeStandardPhase,
    )

# ---------------------------------------------------------------------------
# 2b. DM 阵列物理建模
# ---------------------------------------------------------------------------

def create_dm_grid(
    n_actuators: int,
    dm_diameter: float,
    pupil_diameter: float | None = None,
) -> dict:
    """定义可变形镜的离散 actuator 阵列参数。

    参数：
        n_actuators: 单轴 actuator 数量（方阵，如 12 表示 12×12）
        dm_diameter: DM 有效口径（mm）
        pupil_diameter: 光瞳直径（mm）。若为 None 则等于 dm_diameter

    返回：
        DM 网格参数字典，包含：
        - n_actuators: 单轴阵列数
        - dm_diameter: DM 口径（mm）
        - pupil_diameter: 光瞳直径（mm）
        - pitch: actuator 间距（mm）= dm_diameter / n_actuators
        - centers_1d: 各 actuator 中心坐标的一维数组（mm）
        - cx, cy: actuator 中心坐标的二维网格（mm）
    """
    if n_actuators < 2:
        raise ValueError(f"阵列数必须 >= 2，当前值：{n_actuators}")
    if dm_diameter <= 0:
        raise ValueError(f"DM 口径必须为正数，当前值：{dm_diameter}")

    if pupil_diameter is None:
        pupil_diameter = dm_diameter

    pitch = dm_diameter / n_actuators

    # actuator 中心坐标：从 -dm_diameter/2 + pitch/2 到 +dm_diameter/2 - pitch/2
    half = dm_diameter / 2.0
    centers_1d = np.linspace(-half + pitch / 2, half - pitch / 2, n_actuators)

    cx, cy = np.meshgrid(centers_1d, centers_1d)

    return {
        "n_actuators": n_actuators,
        "dm_diameter": dm_diameter,
        "pupil_diameter": pupil_diameter,
        "pitch": pitch,
        "centers_1d": centers_1d,
        "cx": cx,
        "cy": cy,
    }


def _zernike_basis_on_grid(
    n_terms: int,
    x: np.ndarray,
    y: np.ndarray,
    radius: float,
) -> np.ndarray:
    """在给定坐标网格上计算 Zernike Standard 基函数（Noll 序号）。

    参数：
        n_terms: Zernike 项数（从第 1 项 Piston 开始）
        x, y: 坐标数组（任意形状，但须相同）
        radius: 归一化半径（mm）

    返回：
        形状为 (n_terms, *x.shape) 的数组，basis[k] 对应第 k+1 项
    """
    rho = np.sqrt(x**2 + y**2) / radius
    theta = np.arctan2(y, x)

    basis = np.zeros((n_terms, *x.shape), dtype=float)

    # Noll 序号到 (n, m) 的映射表（前 37 项足够覆盖常见 DM）
    # 格式：(n, m)，m > 0 表示 cos 项，m < 0 表示 sin 项
    noll_to_nm = [
        (0, 0),    # 1: Piston
        (1, 1),    # 2: Tip (x-tilt)
        (1, -1),   # 3: Tilt (y-tilt)
        (2, 0),    # 4: Defocus
        (2, -2),   # 5: Astigmatism (oblique)
        (2, 2),    # 6: Astigmatism (vertical)
        (3, -1),   # 7: Coma (vertical)
        (3, 1),    # 8: Coma (horizontal)
        (3, -3),   # 9: Trefoil (oblique)
        (3, 3),    # 10: Trefoil (vertical)
        (4, 0),    # 11: Primary spherical
        (4, 2),    # 12
        (4, -2),   # 13
        (4, 4),    # 14
        (4, -4),   # 15
        (5, 1),    # 16
        (5, -1),   # 17
        (5, 3),    # 18
        (5, -3),   # 19
        (5, 5),    # 20
        (5, -5),   # 21
        (6, 0),    # 22: Secondary spherical
        (6, 2),    # 23
        (6, -2),   # 24
        (6, 4),    # 25
        (6, -4),   # 26
        (6, 6),    # 27
        (6, -6),   # 28
        (7, 1),    # 29
        (7, -1),   # 30
        (7, 3),    # 31
        (7, -3),   # 32
        (7, 5),    # 33
        (7, -5),   # 34
        (7, 7),    # 35
        (7, -7),   # 36
        (8, 0),    # 37: Tertiary spherical
    ]

    if n_terms > len(noll_to_nm):
        raise ValueError(
            f"当前最多支持 {len(noll_to_nm)} 项 Zernike，"
            f"请求了 {n_terms} 项"
        )

    for k in range(n_terms):
        n, m = noll_to_nm[k]
        basis[k] = _zernike_radial(n, abs(m), rho) * _zernike_angular(m, theta)

    return basis


def _zernike_radial(n: int, m_abs: int, rho: np.ndarray) -> np.ndarray:
    """计算 Zernike 径向多项式 R_n^m(rho)。"""
    result = np.zeros_like(rho)
    for s in range((n - m_abs) // 2 + 1):
        num = (-1)**s * _factorial(n - s)
        den = (
            _factorial(s)
            * _factorial((n + m_abs) // 2 - s)
            * _factorial((n - m_abs) // 2 - s)
        )
        result += (num / den) * rho**(n - 2*s)
    return result


def _zernike_angular(m: int, theta: np.ndarray) -> np.ndarray:
    """计算 Zernike 角向部分。m > 0 → cos，m < 0 → sin，m == 0 → 1。"""
    if m > 0:
        return np.cos(m * theta)
    elif m < 0:
        return np.sin(-m * theta)
    else:
        return np.ones_like(theta)


def _factorial(n: int) -> int:
    """简单阶乘（仅用于小整数）。"""
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def zernike_to_actuator_commands(
    coeffs: np.ndarray,
    dm_grid: dict,
    hires_factor: int = 10,
) -> np.ndarray:
    """Zernike 支路：将 Zernike 系数转换为 DM actuator 指令。

    流程：
    1. 在每个 actuator 区域内生成 hires_factor × hires_factor 的子采样点
    2. 在子采样点上计算连续 Zernike 相位面
    3. 对每个 actuator 区域内的子采样值取均值（面积加权最小二乘）

    参数：
        coeffs: 一维 Zernike 系数数组，coeffs[i] 对应第 i+1 项
        dm_grid: 由 create_dm_grid 返回的 DM 网格参数字典
        hires_factor: 每个 actuator 内的子采样倍数（默认 10）

    返回：
        形状为 (n_actuators, n_actuators) 的二维数组，
        每个元素为对应 actuator 的行程指令值
    """
    if coeffs.size == 0:
        n = dm_grid["n_actuators"]
        return np.zeros((n, n))

    n_act = dm_grid["n_actuators"]
    pitch = dm_grid["pitch"]
    centers = dm_grid["centers_1d"]
    radius = dm_grid["pupil_diameter"] / 2.0
    n_terms = len(coeffs)

    # 在每个 actuator 区域内生成子采样偏移量
    sub = np.linspace(
        -pitch / 2 * (1 - 1 / hires_factor),
        pitch / 2 * (1 - 1 / hires_factor),
        hires_factor,
    )

    commands = np.zeros((n_act, n_act))

    for iy in range(n_act):
        for ix in range(n_act):
            # 该 actuator 区域内的子采样坐标
            x_sub = centers[ix] + sub
            y_sub = centers[iy] + sub
            xx, yy = np.meshgrid(x_sub, y_sub)

            # 在子采样点上计算 Zernike 基函数
            basis = _zernike_basis_on_grid(n_terms, xx, yy, radius)

            # 合成连续相位面并取均值
            phase_sub = np.tensordot(coeffs, basis, axes=(0, 0))
            commands[iy, ix] = np.mean(phase_sub)

    return commands


def actuator_commands_to_phase(
    commands: np.ndarray,
    dm_grid: dict,
    output_size: int,
    output_diameter: float | None = None,
) -> np.ndarray:
    """将 actuator 指令阵列转换为高分辨率离散相位面。

    每个 actuator 覆盖的区域内相位为常数（阶梯型），
    这是 DM 实际能产生的相位调制。

    参数：
        commands: 形状为 (n_actuators, n_actuators) 的指令数组
        dm_grid: 由 create_dm_grid 返回的 DM 网格参数字典
        output_size: 输出相位面的像素数（正方形，output_size × output_size）
        output_diameter: 输出相位面覆盖的物理范围（mm）。
                         若为 None 则等于 dm_diameter

    返回：
        形状为 (output_size, output_size) 的二维相位数组
    """
    n_act = dm_grid["n_actuators"]
    dm_d = dm_grid["dm_diameter"]

    if output_diameter is None:
        output_diameter = dm_d

    # 输出网格坐标
    half_out = output_diameter / 2.0
    px = np.linspace(-half_out, half_out, output_size)
    py = np.linspace(-half_out, half_out, output_size)

    phase = np.zeros((output_size, output_size))

    centers = dm_grid["centers_1d"]
    pitch = dm_grid["pitch"]
    half_pitch = pitch / 2.0

    for iy in range(n_act):
        for ix in range(n_act):
            # 该 actuator 覆盖的物理范围
            x_lo = centers[ix] - half_pitch
            x_hi = centers[ix] + half_pitch
            y_lo = centers[iy] - half_pitch
            y_hi = centers[iy] + half_pitch

            # 找到输出网格中落入该区域的像素
            mask_x = (px >= x_lo) & (px < x_hi)
            mask_y = (py >= y_lo) & (py < y_hi)

            # 赋值为该 actuator 的指令值
            phase[np.ix_(mask_y, mask_x)] = commands[iy, ix]

    return phase


# ---------------------------------------------------------------------------
# 3. DM Zernike 系数写入
# ---------------------------------------------------------------------------

def apply_dm_zernike_coeffs(
    oss,
    surface_idx: int,
    coeffs: np.ndarray,
    update: bool = True,
) -> None:
    """写入 Zernike 系数到指定面。

    参数：
        oss: OpticStudioSystem 实例
        surface_idx: Zernike Standard Phase 面的索引
        coeffs: 一维 numpy 数组，coeffs[i] 对应第 i+1 项 Zernike 系数
                注意：coeffs[0]=Piston, coeffs[1]=Tip, coeffs[2]=Tilt，
                在典型 AO 系统中前三项通常由快速倾斜镜（FSM）校正而非 DM
        update: 是否在写入后触发系统更新（默认 True）。
                在批量标定或生成 Jacobian 矩阵等高频调用场景下，
                可设为 False 以避免每次都触发 Zemax 更新，
                待所有操作完成后手动调用 oss.update_status()

    异常：
        无（空数组直接返回，超长数组截断并发出 warning）
    """
    # 空数组检查：不执行任何写入操作
    if coeffs.size == 0:
        return

    # 获取面对象和面数据
    surface = oss.LDE.GetSurfaceAt(surface_idx)
    surf_data = surface.SurfaceData

    # 读取该面支持的最大 Zernike 项数
    max_terms = surf_data.NumberOfTerms

    # 超长截断：仅写入该面支持范围内的系数，并发出警告
    if len(coeffs) > max_terms:
        warnings.warn(
            f"系数数组长度 {len(coeffs)} 超过该面支持的最大项数 {max_terms}，"
            f"多余系数将被截断",
            UserWarning,
            stacklevel=2,
        )
        coeffs = coeffs[:max_terms]

    # 逐项写入 Zernike 系数（Zemax 编号从 1 开始）
    for i in range(len(coeffs)):
        surf_data.SetNthZernikeCoefficient(i + 1, float(coeffs[i]))

    # 根据 update 参数决定是否触发系统更新
    if update:
        oss.update_status()


# ---------------------------------------------------------------------------
# 4. POP 光场数据提取
# ---------------------------------------------------------------------------

def get_pop_field(
    oss,
    start_surf: int = 1,
    end_surf: int | str = "Image",
    sampling: int = 128,
    beam_width: float = 10.0,
    auto_beam_sampling: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """执行 POP 分析，返回振幅、相位和物理坐标信息。

    参数：
        oss: OpticStudioSystem 实例
        start_surf: POP 起始面编号（默认 1）
        end_surf: POP 终止面编号或 "Image"（默认 "Image"）
        sampling: 采样点数（默认 128）
        beam_width: 光束采样区域宽度，单位为系统透镜单位（默认 10.0）
        auto_beam_sampling: 是否让 Zemax 自动计算采样窗口（默认 True）。
                            设为 False 时使用 beam_width 指定的窗口大小。

    返回：
        (amplitude, phase, extent_info) 元组：
        - amplitude: 二维 numpy 数组（辐照度开方）
        - phase: 二维 numpy 数组（未解包裹的原始相位，单位 radians）
          注意：返回的相位为 Zemax POP 原始输出，通常包裹在
          [-π, π] 或 [0, 2π] 范围内。如需解包裹，请使用
          skimage.restoration.unwrap_phase 等工具自行处理。
        - extent_info: 字典，包含物理坐标信息：
          {"x": np.ndarray, "y": np.ndarray,
           "x_width": float, "y_width": float}
          其中 x/y 为坐标轴数组（mm），x_width/y_width 为物理范围

    异常：
        RuntimeError: POP 分析未返回有效数据
    """
    # --- 第一次 POP 调用：获取辐照度（Irradiance） ---
    pop_irr = PhysicalOpticsPropagation(
        start_surface=start_surf,
        end_surface=end_surf,
        x_sampling=sampling,
        y_sampling=sampling,
        x_width=beam_width,
        y_width=beam_width,
        data_type="Irradiance",
        show_as="FalseColor",
        auto_calculate_beam_sampling=auto_beam_sampling,
    )
    result_irr = pop_irr.run(oss)

    # 空数据检查
    if result_irr.data is None:
        raise RuntimeError(
            "POP 辐照度分析未返回有效数据，请检查光学系统配置和面参数"
        )

    # 从 DataFrame 提取辐照度数据并转为 numpy 数组
    df_irr = result_irr.data
    irradiance = df_irr.values.astype(float)

    # 提取物理坐标轴（mm）
    x = df_irr.columns.astype(float).values
    y = df_irr.index.astype(float).values

    # 辐照度开方得到振幅（确保非负）
    amplitude = np.sqrt(irradiance)

    # --- 第二次 POP 调用：获取相位（Phase） ---
    pop_phase = PhysicalOpticsPropagation(
        start_surface=start_surf,
        end_surface=end_surf,
        x_sampling=sampling,
        y_sampling=sampling,
        x_width=beam_width,
        y_width=beam_width,
        data_type="Phase",
        show_as="FalseColor",
        auto_calculate_beam_sampling=auto_beam_sampling,
    )
    result_phase = pop_phase.run(oss)

    # 空数据检查
    if result_phase.data is None:
        raise RuntimeError(
            "POP 相位分析未返回有效数据，请检查光学系统配置和面参数"
        )

    # 从 DataFrame 提取相位数据并转为 numpy 数组
    phase = result_phase.data.values.astype(float)

    # 构建物理坐标信息字典
    extent_info = {
        "x": x,
        "y": y,
        "x_width": float(x.max() - x.min()),
        "y_width": float(y.max() - y.min()),
    }

    return amplitude, phase, extent_info

def get_pop_field_with_beam_file(
    oss,
    beam_file: str,
    start_surf: int = 1,
    end_surf: int | str = "Image",
    sampling: int = 128,
    beam_width: float = 10.0,
    auto_beam_sampling: bool = False,
    total_power: float | None = None,
    save_output_beam: bool = False,
    output_beam_file: str = "",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """执行 POP 分析（使用 ZBF 文件作为输入光束），返回振幅、相位和坐标信息。

    与 get_pop_field 功能相同，但使用 ZBF 文件替代默认高斯光束。
    适用于 GS+SPSA 混合优化：GS 相位写入 ZBF，SPSA 在此基础上微调。

    参数：
        oss: OpticStudioSystem 实例
        beam_file: ZBF 文件名（不含路径，位于 Zemax POP/BEAMFILES 目录）
        start_surf: POP 起始面编号（默认 1）
        end_surf: POP 终止面编号或 "Image"（默认 "Image"）
        sampling: 采样点数（默认 128）
        beam_width: 光束采样区域宽度，单位 mm（默认 10.0）
        auto_beam_sampling: 是否自动计算采样窗口（默认 False）
        total_power: 总功率归一化值；为 None 时使用峰值辐照度归一化
        save_output_beam: 是否保存输出光束 ZBF 文件（默认 False）
        output_beam_file: 输出光束文件名（不含扩展名，仅 save_output_beam=True 时有效）

    返回：
        (amplitude, phase, extent_info) 元组，格式与 get_pop_field 相同
    """
    # 功率归一化设置
    if total_power is not None:
        use_tp, use_pi, tp = True, False, total_power
    else:
        use_tp, use_pi, tp = False, True, 1.0

    # --- 第一次 POP 调用：获取辐照度 ---
    pop_irr = PhysicalOpticsPropagation(
        start_surface=start_surf,
        end_surface=end_surf,
        x_sampling=sampling,
        y_sampling=sampling,
        x_width=beam_width,
        y_width=beam_width,
        data_type="Irradiance",
        show_as="FalseColor",
        beam_type="File",
        beam_file=beam_file,
        auto_calculate_beam_sampling=auto_beam_sampling,
        use_total_power=use_tp,
        use_peak_irradiance=use_pi,
        total_power=tp,
        save_output_beam=save_output_beam,
        output_beam_file=output_beam_file if save_output_beam else "",
    )
    result_irr = pop_irr.run(oss)

    if result_irr.data is None:
        raise RuntimeError(
            "POP 辐照度分析（ZBF 输入）未返回有效数据"
        )

    df_irr = result_irr.data
    irradiance = df_irr.values.astype(float)
    x = df_irr.columns.astype(float).values
    y = df_irr.index.astype(float).values
    amplitude = np.sqrt(irradiance)

    # --- 第二次 POP 调用：获取相位 ---
    pop_phase = PhysicalOpticsPropagation(
        start_surface=start_surf,
        end_surface=end_surf,
        x_sampling=sampling,
        y_sampling=sampling,
        x_width=beam_width,
        y_width=beam_width,
        data_type="Phase",
        show_as="FalseColor",
        beam_type="File",
        beam_file=beam_file,
        auto_calculate_beam_sampling=auto_beam_sampling,
        use_total_power=use_tp,
        use_peak_irradiance=use_pi,
        total_power=tp,
    )
    result_phase = pop_phase.run(oss)

    if result_phase.data is None:
        raise RuntimeError(
            "POP 相位分析（ZBF 输入）未返回有效数据"
        )

    phase = result_phase.data.values.astype(float)

    extent_info = {
        "x": x,
        "y": y,
        "x_width": float(x.max() - x.min()),
        "y_width": float(y.max() - y.min()),
    }

    return amplitude, phase, extent_info



# ---------------------------------------------------------------------------
# 5. 结果可视化
# ---------------------------------------------------------------------------

def plot_wavefront(
    amplitude: np.ndarray,
    phase: np.ndarray,
    title: str = "波前分析",
    extent_info: dict | None = None,
) -> plt.Figure:
    """绘制振幅与相位的并排热力图。

    参数：
        amplitude: 二维振幅数组
        phase: 二维相位数组
        title: 图形总标题（默认"波前分析"）
        extent_info: 可选，来自 get_pop_field 的物理坐标信息字典。
                     提供后坐标轴将标注物理单位（mm）

    返回：
        matplotlib Figure 对象

    异常：
        ValueError: amplitude 和 phase 形状不一致
    """
    # 形状校验
    if amplitude.shape != phase.shape:
        raise ValueError(
            f"振幅与相位数组形状不一致：amplitude={amplitude.shape}, "
            f"phase={phase.shape}"
        )

    # 创建 1×2 子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 计算物理坐标范围（如果提供了 extent_info）
    extent = None
    if extent_info is not None:
        x = extent_info["x"]
        y = extent_info["y"]
        extent = [x.min(), x.max(), y.min(), y.max()]

    # 左侧子图：振幅（hot colormap）
    im1 = ax1.imshow(amplitude, cmap="hot", extent=extent)
    ax1.set_title("振幅")
    fig.colorbar(im1, ax=ax1)

    # 右侧子图：相位（RdBu_r colormap）
    im2 = ax2.imshow(phase, cmap="RdBu_r", extent=extent)
    ax2.set_title("相位")
    fig.colorbar(im2, ax=ax2)

    # 如果有物理坐标，标注坐标轴单位
    if extent_info is not None:
        for ax in (ax1, ax2):
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")

    # 设置总标题
    fig.suptitle(title)

    return fig


# ---------------------------------------------------------------------------
# 5b. POP 结果增强绘图与报告（可复用工具）
# ---------------------------------------------------------------------------

def plot_pop_result(
    zbf_data,
    title: str = "POP 结果",
    save_path: str | None = None,
) -> plt.Figure:
    """增强版单面 POP 结果绘图。

    包含 4 个子图：辐照度热力图、相位热力图、X/Y 截面曲线。
    自动标注物理网格信息和光束宽度（从 ZBFData header 读取）。

    参数：
        zbf_data: ZBFData 对象（Zemax POP 输出）
        title: 图形标题
        save_path: 保存路径（可选，为 None 时不保存）

    返回：
        matplotlib Figure 对象
    """
    irr = zbf_data.irradiance
    ph = zbf_data.phase
    x = zbf_data.x_coords
    y = zbf_data.y_coords
    units_map = {0: "mm", 1: "cm", 2: "in", 3: "m"}
    u = units_map.get(zbf_data.units, "?")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(title, fontsize=14)
    extent = [x[0], x[-1], y[-1], y[0]]

    # (0,0) 辐照度热力图
    ax = axes[0, 0]
    im = ax.imshow(irr, cmap="hot", extent=extent, aspect="equal")
    fig.colorbar(im, ax=ax, shrink=0.8, label="W")
    ax.set_title("辐照度")
    ax.set_xlabel(f"X ({u})")
    ax.set_ylabel(f"Y ({u})")

    # (0,1) 相位热力图
    ax = axes[0, 1]
    im = ax.imshow(ph, cmap="RdBu_r", extent=extent, aspect="equal")
    fig.colorbar(im, ax=ax, shrink=0.8, label="rad")
    ax.set_title("相位")
    ax.set_xlabel(f"X ({u})")
    ax.set_ylabel(f"Y ({u})")

    # (1,0) X/Y 截面
    ax = axes[1, 0]
    cy = irr.shape[0] // 2
    cx = irr.shape[1] // 2
    row = irr[cy, :]
    col = irr[:, cx]
    peak = max(np.max(row), 1e-30)
    ax.plot(x, row / peak, "r-", label="X 截面", linewidth=1.5)
    ax.plot(y, col / peak, "b--", label="Y 截面", linewidth=1.5)
    ax.set_title("归一化辐照度截面")
    ax.set_xlabel(f"坐标 ({u})")
    ax.set_ylabel("归一化辐照度")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.15)

    # (1,1) 物理信息文本
    ax = axes[1, 1]
    ax.axis("off")
    total_power = float(np.sum(irr))
    peak_irr = float(np.max(irr))
    info_text = (
        f"网格: {zbf_data.nx} × {zbf_data.ny}\n"
        f"dx={zbf_data.dx:.4e} {u},  dy={zbf_data.dy:.4e} {u}\n"
        f"物理范围: {zbf_data.x_width:.4f} × {zbf_data.y_width:.4f} {u}\n"
        f"波长: {zbf_data.wavelength:.6e} {u}\n"
        f"折射率: {zbf_data.index:.4f}\n"
        f"─────────────────\n"
        f"光束宽度 (Guide Beam):\n"
        f"  wx = {zbf_data.wx:.6e} {u}\n"
        f"  wy = {zbf_data.wy:.6e} {u}\n"
        f"─────────────────\n"
        f"总功率: {total_power:.6e} W\n"
        f"峰值辐照度: {peak_irr:.6e} W"
    )
    ax.text(
        0.05, 0.95, info_text,
        transform=ax.transAxes, fontsize=10,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                  alpha=0.8),
    )
    ax.set_title("物理参数")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_pop_overview(
    results: list[dict],
    title: str = "多面 POP 汇总",
    save_path: str | None = None,
) -> plt.Figure:
    """多面 POP 结果汇总绘图。

    上排：各面辐照度热力图；下排：各面相位热力图。
    每个子图标注光束宽度。

    参数：
        results: 字典列表，每个字典包含：
            - "label": 面标签（如 "面 3"、"Image"）
            - "zbf": ZBFData 对象
        title: 总标题
        save_path: 保存路径（可选）

    返回：
        matplotlib Figure 对象
    """
    n = len(results)
    if n == 0:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.text(0.5, 0.5, "无数据", ha="center", va="center")
        return fig

    fig, axes = plt.subplots(2, n, figsize=(5 * n, 9), squeeze=False)
    fig.suptitle(title, fontsize=14)

    for i, res in enumerate(results):
        zbf = res["zbf"]
        label = res["label"]
        irr = zbf.irradiance
        ph = zbf.phase
        x = zbf.x_coords
        y = zbf.y_coords
        extent = [x[0], x[-1], y[-1], y[0]]
        units_map = {0: "mm", 1: "cm", 2: "in", 3: "m"}
        u = units_map.get(zbf.units, "?")

        # 上排：辐照度
        ax = axes[0, i]
        im = ax.imshow(irr, cmap="hot", extent=extent, aspect="equal")
        fig.colorbar(im, ax=ax, shrink=0.7)
        ax.set_title(f"{label}\nwx={zbf.wx:.3e} wy={zbf.wy:.3e}",
                     fontsize=9)
        ax.set_xlabel(f"X ({u})")
        ax.set_ylabel(f"Y ({u})")

        # 下排：相位
        ax = axes[1, i]
        im = ax.imshow(ph, cmap="RdBu_r", extent=extent, aspect="equal")
        fig.colorbar(im, ax=ax, shrink=0.7)
        ax.set_title(f"{label} 相位", fontsize=9)
        ax.set_xlabel(f"X ({u})")
        ax.set_ylabel(f"Y ({u})")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def generate_pop_report(
    results: list[dict],
    output_path: str | None = None,
) -> str:
    """生成 POP 仿真文本报告。

    包含每个面的网格参数、光束宽度、功率等信息的汇总表格。

    参数：
        results: 字典列表，每个字典包含：
            - "label": 面标签
            - "zbf": ZBFData 对象
        output_path: 报告文件保存路径（可选，为 None 时仅返回字符串）

    返回：
        报告文本字符串
    """
    lines = []
    lines.append("=" * 70)
    lines.append("POP 仿真报告")
    lines.append("=" * 70)

    for i, res in enumerate(results):
        zbf = res["zbf"]
        label = res["label"]
        units_map = {0: "mm", 1: "cm", 2: "in", 3: "m"}
        u = units_map.get(zbf.units, "?")
        total_power = float(np.sum(zbf.irradiance))
        peak_irr = float(np.max(zbf.irradiance))

        lines.append(f"\n--- {label} ---")
        lines.append(f"  网格:       {zbf.nx} × {zbf.ny}")
        lines.append(f"  dx:         {zbf.dx:.6e} {u}")
        lines.append(f"  dy:         {zbf.dy:.6e} {u}")
        lines.append(f"  物理范围:   {zbf.x_width:.4f} × {zbf.y_width:.4f} {u}")
        lines.append(f"  波长:       {zbf.wavelength:.6e} {u}")
        lines.append(f"  折射率:     {zbf.index:.4f}")
        lines.append(f"  光束宽度:   wx={zbf.wx:.6e} {u},  wy={zbf.wy:.6e} {u}")
        lines.append(f"  总功率:     {total_power:.6e} W")
        lines.append(f"  峰值辐照度: {peak_irr:.6e} W")

    # 汇总表
    lines.append(f"\n{'=' * 70}")
    lines.append("光束宽度汇总")
    lines.append(f"{'=' * 70}")
    header = f"  {'面':>10s}  {'wx':>14s}  {'wy':>14s}  {'总功率':>14s}"
    lines.append(header)
    lines.append("  " + "-" * 60)
    for res in results:
        zbf = res["zbf"]
        u = units_map.get(zbf.units, "?")
        tp = float(np.sum(zbf.irradiance))
        lines.append(
            f"  {res['label']:>10s}"
            f"  {zbf.wx:>14.6e}"
            f"  {zbf.wy:>14.6e}"
            f"  {tp:>14.6e}"
        )

    lines.append("")
    report = "\n".join(lines)

    if output_path:
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

    return report


# ---------------------------------------------------------------------------
# 6. 断开连接
# ---------------------------------------------------------------------------

def cleanup(zos: zp.ZOS) -> None:
    """断开与 Zemax OpticStudio 的连接。"""
    zos.disconnect()


# ---------------------------------------------------------------------------
# 7. 模块入口：演示完整 AO 仿真工作流
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 演示文件路径（相对于工作目录）
    ZMX_FILE = "Zemax_baseline/biconic_focus_test.zmx"

    # 演示用 Zernike 系数（前 10 项，单位为波长数）
    # coeffs[0]=Piston, coeffs[1]=Tip, coeffs[2]=Tilt, ...
    DEMO_COEFFS = np.array([
        0.0,   # Piston
        0.0,   # Tip
        0.0,   # Tilt
        0.5,   # Defocus
        0.3,   # Astigmatism (oblique)
        0.2,   # Astigmatism (vertical)
        0.1,   # Coma (vertical)
        0.1,   # Coma (horizontal)
        0.05,  # Trefoil (oblique)
        0.05,  # Trefoil (vertical)
    ])

    # DM 所在面的索引（根据具体光学系统调整）
    DM_SURFACE_IDX = 2

    zos = None  # 预声明，确保 finally 中可访问

    try:
        # ---- 步骤 1：初始化系统 ----
        print("正在连接 Zemax OpticStudio 并加载光学系统...")
        zos, oss = init_system(ZMX_FILE)
        print(f"系统加载成功：{ZMX_FILE}")

        # ---- 步骤 2：配置 DM 面 ----
        print(f"正在将面 {DM_SURFACE_IDX} 设置为 Zernike Standard Phase 类型...")
        setup_dm_surface(oss, DM_SURFACE_IDX)
        print("DM 面配置完成")

        # ---- 步骤 3：写入 Zernike 系数 ----
        print(f"正在写入 {len(DEMO_COEFFS)} 项 Zernike 系数...")
        apply_dm_zernike_coeffs(oss, DM_SURFACE_IDX, DEMO_COEFFS)
        print("Zernike 系数写入完成")

        # ---- 步骤 4：提取 POP 光场数据 ----
        print("正在执行 POP 分析，提取光场数据...")
        amplitude, phase, extent_info = get_pop_field(oss)
        print(f"光场数据提取完成：振幅形状={amplitude.shape}，相位形状={phase.shape}")

        # ---- 步骤 5：可视化结果 ----
        print("正在绘制波前分析图...")
        fig = plot_wavefront(
            amplitude, phase,
            title="AO 仿真演示 — 波前分析",
            extent_info=extent_info,
        )
        plt.show()
        print("演示流程全部完成！")

    except FileNotFoundError as e:
        print(f"文件错误：{e}")
    except IndexError as e:
        print(f"面索引错误：{e}")
    except RuntimeError as e:
        print(f"运行时错误：{e}")
    except Exception as e:
        print(f"未预期的错误：{type(e).__name__}: {e}")
    finally:
        # 确保无论是否出错都断开 Zemax 连接
        if zos is not None:
            print("正在断开 Zemax 连接...")
            cleanup(zos)
            print("连接已断开")
