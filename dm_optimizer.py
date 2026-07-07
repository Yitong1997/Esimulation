"""DM Zernike 系数优化模块。

在 ao_core.py 基础上，实现可变形镜（DM）Zernike 系数的迭代优化，
目标是将像面强度分布整形为理想平顶分布。

提供两条可切换的优化管线：
  - Pipeline A：基于外部 spsa 库的 SPSA 优化
  - Pipeline B：自定义双边 SPGD 优化

整体架构：扁平过程式风格，纯函数 + numpy 数组，零 OOP。
"""

import numpy as np                  # 数值计算
import matplotlib.pyplot as plt     # 可视化绘图
import ao_core                      # Zemax API 封装（现有模块）
import spsa                         # SPSA 优化库（外部依赖）


# ---------------------------------------------------------------------------
# 物理约束：DM 行程限制
# ---------------------------------------------------------------------------

def apply_stroke_limit(
    zernike_coeffs: np.ndarray,
    max_stroke: float,
) -> np.ndarray:
    """裁剪 Zernike 系数到 DM 行程限制范围。

    参数：
        zernike_coeffs: 一维 Zernike 系数数组
        max_stroke: 最大行程限制（正数）

    返回：
        裁剪后的新数组（不修改输入）
    """
    # 使用 np.clip 裁剪到 [-max_stroke, max_stroke]，返回新数组
    return np.clip(zernike_coeffs, -max_stroke, max_stroke)

# ---------------------------------------------------------------------------
# Zemax API 接口：封装 POP 调用，返回像面强度
# ---------------------------------------------------------------------------

def run_zemax_pop_and_get_intensity(
    zernike_coeffs: np.ndarray,
    oss,
    dm_surface_idx: int,
    pop_params: dict,
) -> np.ndarray:
    """封装 ao_core 的 Zemax POP 调用，返回像面强度分布。

    参数：
        zernike_coeffs: 一维 Zernike 系数数组
        oss: OpticStudioSystem 实例（来自 ao_core.init_system）
        dm_surface_idx: DM 面在 LDE 中的索引
        pop_params: POP 分析参数字典

    返回：
        二维 numpy 数组，像面强度分布（amplitude²，非负）
    """
    # 将 Zernike 系数写入 Zemax DM 面
    ao_core.apply_dm_zernike_coeffs(oss, dm_surface_idx, zernike_coeffs)

    # 执行 POP 分析，获取复数场数据
    amplitude, phase, extent_info = ao_core.get_pop_field(oss, **pop_params)

    # 计算强度：振幅的平方
    intensity = amplitude ** 2

    # 防御性裁剪：浮点误差可能产生极小负值
    intensity = np.clip(intensity, 0, None)

    return intensity


# ---------------------------------------------------------------------------
# 代价函数：平顶分布损失
# ---------------------------------------------------------------------------

def calculate_flat_top_loss(
    intensity_2d: np.ndarray,
    target_radius: float,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """计算平顶分布代价函数。

    参数：
        intensity_2d: 二维强度数组
        target_radius: 目标圆形掩模半径（像素单位）
        alpha: 均匀性项权重
        beta: 能量泄漏项权重

    返回：
        标量损失值 J = alpha * CV + beta * leakage_ratio
        若掩模内均值为零，返回惩罚值 1e6
    """
    # 生成圆形掩模，以数组中心为圆心
    ny, nx = intensity_2d.shape
    cy, cx = ny / 2, nx / 2
    Y, X = np.ogrid[:ny, :nx]
    mask = (X - cx)**2 + (Y - cy)**2 <= target_radius**2

    # 提取掩模内像素
    inside = intensity_2d[mask]

    # 计算掩模内均值
    mean_val = np.mean(inside)

    # 零均值保护：避免除零错误
    if mean_val == 0:
        return 1e6

    # 变异系数 CV = 标准差 / 均值
    cv = np.std(inside) / mean_val

    # 能量泄漏比 = 掩模外能量 / 总能量
    total_energy = np.sum(intensity_2d)
    if total_energy == 0:
        leakage = 0.0
    else:
        leakage = np.sum(intensity_2d[~mask]) / total_energy

    # 加权求和返回损失值
    return alpha * cv + beta * leakage


# ---------------------------------------------------------------------------
# SPSA 优化管线：基于外部 spsa 库
# ---------------------------------------------------------------------------

def run_spsa_optimization(
    initial_coeffs: np.ndarray,
    objective_fn,
    max_iter: int,
    visualize_fn=None,
    vis_interval: int = 10,
) -> tuple[np.ndarray, list[float]]:
    """使用外部 spsa 库执行优化。

    参数：
        initial_coeffs: 初始 Zernike 系数数组
        objective_fn: 目标函数，接受系数数组返回标量损失值
        max_iter: 最大迭代次数
        visualize_fn: 可选的可视化回调函数
        vis_interval: 可视化更新间隔

    返回：
        (optimized_coeffs, loss_history) 元组
    """
    # 记录每次迭代的损失值
    loss_history: list[float] = []

    # 包装目标函数：记录损失并触发可视化
    def wrapper(coeffs):
        loss = objective_fn(coeffs)
        loss_history.append(loss)
        # 每隔 vis_interval 次调用可视化回调
        if visualize_fn is not None and len(loss_history) % vis_interval == 0:
            visualize_fn(loss_history)
        return loss

    # 调用 spsa 库执行优化
    result = spsa.minimize(wrapper, initial_coeffs, max_iter=max_iter)

    # 提取优化后的系数
    optimized_coeffs = np.array(result.x)

    return optimized_coeffs, loss_history


# ---------------------------------------------------------------------------
# SPGD 优化管线：自定义双边 SPGD
# ---------------------------------------------------------------------------

def run_spgd_optimization(
    initial_coeffs: np.ndarray,
    objective_fn,
    max_iter: int,
    perturbation_size: float,
    gain: float,
    max_stroke: float,
    visualize_fn=None,
    vis_interval: int = 10,
) -> tuple[np.ndarray, list[float]]:
    """自定义双边 SPGD 优化。

    参数：
        initial_coeffs: 初始 Zernike 系数数组
        objective_fn: 目标函数，接受系数数组返回标量损失值
        max_iter: 最大迭代次数
        perturbation_size: 扰动幅度
        gain: 学习增益
        max_stroke: DM 行程限制
        visualize_fn: 可选的可视化回调函数
        vis_interval: 可视化更新间隔

    返回：
        (optimized_coeffs, loss_history) 元组
    """
    # 复制初始系数，避免修改输入
    coeffs = initial_coeffs.copy()
    loss_history: list[float] = []

    for i in range(max_iter):
        # 生成随机扰动向量：每个元素从 {-1, +1} 均匀采样后乘以扰动幅度
        delta = np.random.choice([-1, 1], size=len(coeffs)) * perturbation_size

        # 正扰动和负扰动经过行程限制
        c_plus = apply_stroke_limit(coeffs + delta, max_stroke)
        c_minus = apply_stroke_limit(coeffs - delta, max_stroke)

        # 计算双边损失
        j_plus = objective_fn(c_plus)
        j_minus = objective_fn(c_minus)

        # 梯度估计与系数更新
        coeffs = coeffs - gain * (j_plus - j_minus) * delta

        # 记录当前系数经行程限制后的损失值
        current_loss = objective_fn(apply_stroke_limit(coeffs, max_stroke))
        loss_history.append(current_loss)

        # 每隔 vis_interval 次迭代调用可视化回调
        if visualize_fn and (i + 1) % vis_interval == 0:
            visualize_fn(loss_history)

    return coeffs, loss_history


# ---------------------------------------------------------------------------
# 实时可视化：交互模式绘图
# ---------------------------------------------------------------------------

def init_visualization() -> tuple:
    """初始化 matplotlib 交互模式绘图窗口。

    返回：
        (fig, ax_loss, ax_intensity) 元组
    """
    # 开启交互模式，允许实时刷新
    plt.ion()
    # 创建 1×2 子图布局：左侧收敛曲线，右侧强度热图
    fig, (ax_loss, ax_intensity) = plt.subplots(1, 2, figsize=(12, 5))
    return fig, ax_loss, ax_intensity


def update_visualization(fig, ax_loss, ax_intensity, loss_history, intensity_2d):
    """刷新收敛曲线和强度热图。

    参数：
        fig: matplotlib Figure 对象
        ax_loss: 收敛曲线子图 Axes
        ax_intensity: 强度热图子图 Axes
        loss_history: 损失值历史列表
        intensity_2d: 当前像面强度二维数组
    """
    # 清除左侧子图并绘制收敛曲线
    ax_loss.clear()
    ax_loss.plot(loss_history)
    ax_loss.set_xlabel("迭代次数")
    ax_loss.set_ylabel("损失值")
    ax_loss.set_title("收敛曲线")

    # 清除右侧子图并绘制强度热图
    ax_intensity.clear()
    im = ax_intensity.imshow(intensity_2d, cmap='hot')
    ax_intensity.set_title("像面强度分布")

    # 移除旧的 colorbar（避免重复创建）
    while len(fig.axes) > 2:
        fig.axes[-1].remove()
    # 添加新的 colorbar
    fig.colorbar(im, ax=ax_intensity)

    # 刷新画布
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


# ---------------------------------------------------------------------------
# 主程序入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ===== 超参数集中定义 =====
    ZMX_FILE = "Zemax_baseline/biconic_focus_test.zmx"  # Zemax 文件路径
    DM_SURFACE_IDX = 2       # DM 面在 LDE 中的索引
    N_ZERNIKE = 10           # Zernike 项数
    MAX_ITER = 200           # 最大迭代次数
    TARGET_RADIUS = 30.0     # 目标掩模半径（像素）
    MAX_STROKE = 2.0         # DM 行程限制
    ALPHA = 1.0              # 均匀性权重
    BETA = 1.0               # 能量泄漏权重
    PERTURBATION_SIZE = 0.05 # SPGD 扰动幅度
    GAIN = 0.1               # SPGD 学习增益
    VIS_INTERVAL = 5         # 可视化更新间隔（迭代次数）
    USE_PIPELINE = "A"       # "A" = SPSA, "B" = SPGD

    # POP 分析参数
    POP_PARAMS = {
        "start_surf": 1,
        "end_surf": "Image",
        "sampling": 128,
        "beam_width": 10.0,
    }

    # 初始化 Zemax 系统
    zos, oss = ao_core.init_system(ZMX_FILE)

    try:
        ao_core.setup_dm_surface(oss, DM_SURFACE_IDX)

        # 初始系数（全零）
        initial_coeffs = np.zeros(N_ZERNIKE)

        # 构造目标函数闭包（行程限制 → Zemax POP → 代价函数）
        def objective_fn(coeffs):
            clipped = apply_stroke_limit(coeffs, MAX_STROKE)
            intensity = run_zemax_pop_and_get_intensity(
                clipped, oss, DM_SURFACE_IDX, POP_PARAMS)
            return calculate_flat_top_loss(
                intensity, TARGET_RADIUS, ALPHA, BETA)

        # 初始化可视化窗口
        fig, ax_loss, ax_intensity = init_visualization()

        # 可视化回调：获取当前强度并刷新图表
        def visualize_callback(loss_history):
            clipped = apply_stroke_limit(initial_coeffs, MAX_STROKE)
            intensity = run_zemax_pop_and_get_intensity(
                clipped, oss, DM_SURFACE_IDX, POP_PARAMS)
            update_visualization(
                fig, ax_loss, ax_intensity, loss_history, intensity)

        # 根据 USE_PIPELINE 选择管线并执行优化
        if USE_PIPELINE == "A":
            best_coeffs, history = run_spsa_optimization(
                initial_coeffs, objective_fn, MAX_ITER,
                visualize_fn=visualize_callback,
                vis_interval=VIS_INTERVAL)
        else:
            best_coeffs, history = run_spgd_optimization(
                initial_coeffs, objective_fn, MAX_ITER,
                PERTURBATION_SIZE, GAIN, MAX_STROKE,
                visualize_fn=visualize_callback,
                vis_interval=VIS_INTERVAL)

    finally:
        # 确保 Zemax 资源被清理（即使发生异常）
        ao_core.cleanup(zos)
