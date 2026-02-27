"""ZBF → Zemax POP 仿真流水线。

完整流程：读取 ZBF → 部署到 Zemax → 多面 POP 仿真 → 绘图 → 保存（CSV/ZBF）。
100% 复用 zbf_io 和 ao_core 现有模块，主调用约 20 行。
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from zbf_io import read_zbf, write_zbf, ZBFData, print_zbf_info
from ao_core import (
    init_system, get_pop_field_with_beam_file, plot_wavefront, cleanup,
)


# ============================================================
# Zemax POP BEAMFILES 目录（固定位置）
# ============================================================
ZEMAX_POP_DIR = Path(os.path.expanduser("~/Documents/Zemax/POP/BEAMFILES"))


# ============================================================
# 3 个轻量辅助函数（脚本内定义，不重复造轮子）
# ============================================================

def deploy_zbf_to_zemax(zbf_path: str) -> str:
    """将 ZBF 文件复制到 Zemax POP/BEAMFILES 目录。

    参数：
        zbf_path: 源 ZBF 文件路径

    返回：
        文件名（不含路径），供 get_pop_field_with_beam_file 使用
    """
    src = Path(zbf_path)
    ZEMAX_POP_DIR.mkdir(parents=True, exist_ok=True)
    dst = ZEMAX_POP_DIR / src.name
    shutil.copy2(src, dst)
    print(f"  已部署: {src.name} → {dst}")
    return src.name


def save_pop_result_csv(
    amplitude: np.ndarray,
    phase: np.ndarray,
    extent_info: dict,
    output_dir: str,
    label: str,
) -> None:
    """将 POP 仿真结果保存为 CSV 文件。

    生成两个文件：{label}_irradiance.csv 和 {label}_phase.csv，
    首行为 X 坐标，首列为 Y 坐标，数据区为对应物理量。

    参数：
        amplitude: 振幅数组
        phase: 相位数组
        extent_info: 物理坐标信息
        output_dir: 输出目录
        label: 文件名前缀
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    x = extent_info["x"]
    y = extent_info["y"]

    irradiance = amplitude ** 2

    for name, data in [("irradiance", irradiance), ("phase", phase)]:
        filepath = out / f"{label}_{name}.csv"
        # 构建带坐标头的矩阵
        header_row = np.concatenate([[0.0], x])  # 左上角填 0
        rows = np.column_stack([y, data])
        full = np.vstack([header_row, rows])
        np.savetxt(filepath, full, delimiter=",", fmt="%.8e")
        print(f"  CSV 已保存: {filepath}")


def save_pop_result_zbf(
    amplitude: np.ndarray,
    phase: np.ndarray,
    extent_info: dict,
    ref_zbf: ZBFData,
    filepath: str,
) -> None:
    """将 POP 仿真结果保存为 ZBF 文件。

    以 ref_zbf 为模板（复用波长、折射率等元数据），
    将 amplitude 和 phase 构造为复数电场后写入。

    参数：
        amplitude: 振幅数组
        phase: 相位数组
        extent_info: 物理坐标信息
        ref_zbf: 参考 ZBFData（提供波长等元数据）
        filepath: 输出 ZBF 文件路径
    """
    zbf = ref_zbf.copy()
    ny, nx = amplitude.shape
    zbf.nx = nx
    zbf.ny = ny

    x = extent_info["x"]
    y = extent_info["y"]
    zbf.dx = float(x[1] - x[0]) if nx > 1 else ref_zbf.dx
    zbf.dy = float(y[1] - y[0]) if ny > 1 else ref_zbf.dy

    # 构建复数电场
    zbf.Ex = amplitude * np.exp(1j * phase)
    zbf.Ey = None
    zbf.is_polarized = 0

    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_zbf(out_path, zbf)
    print(f"  ZBF 已保存: {out_path}")


# ============================================================
# 主程序配置
# ============================================================

# ===== 用户配置区（根据实际情况修改）=====
ZBF_INPUT     = "test_output.zbf"                          # 输入 ZBF 文件路径
ZMX_FILE      = "Zemax_baseline/biconic_focus_test.zmx"     # Zemax 系统文件
END_SURFACES  = [3, "Image"]                                # 需要仿真的面列表
SAMPLING      = 256                                         # POP 采样点数
BEAM_WIDTH    = 10.0                                        # 采样窗口宽度 (mm)
OUTPUT_DIR    = "pop_results"                               # 输出目录
SAVE_CSV      = True                                        # 是否保存 CSV
SAVE_ZBF      = True                                        # 是否保存 ZBF


# ============================================================
# 主流程
# ============================================================

def main():
    zos = None
    try:
        print("=" * 60)
        print("ZBF → Zemax POP 仿真流水线")
        print("=" * 60)

        # 1. 读取 ZBF 文件
        print("\n[1] 读取 ZBF 输入文件...")
        zbf_data = read_zbf(ZBF_INPUT)
        print_zbf_info(zbf_data, "输入光束")

        # 2. 部署到 Zemax BEAMFILES 目录
        print("\n[2] 部署 ZBF 到 Zemax...")
        beam_filename = deploy_zbf_to_zemax(ZBF_INPUT)

        # 3. 连接 Zemax
        print("\n[3] 连接 Zemax OpticStudio...")
        zos, oss = init_system(ZMX_FILE)
        print(f"  已加载: {ZMX_FILE}")

        # 4. 多面 POP 仿真 + 绘图 + 保存
        print(f"\n[4] 执行 POP 仿真（面: {END_SURFACES}）...")
        for surf in END_SURFACES:
            print(f"\n  --- 面 {surf} ---")
            amp, phase, ext = get_pop_field_with_beam_file(
                oss,
                beam_file=beam_filename,
                start_surf=1,
                end_surf=surf,
                sampling=SAMPLING,
                beam_width=BEAM_WIDTH,
            )
            print(f"  振幅形状: {amp.shape}, 相位形状: {phase.shape}")

            # 绘图
            fig = plot_wavefront(
                amp, phase,
                title=f"POP 仿真 — 面 {surf}",
                extent_info=ext,
            )

            # 保存绘图
            fig_path = Path(OUTPUT_DIR) / f"pop_surf_{surf}.png"
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(fig_path, dpi=150)
            print(f"  图已保存: {fig_path}")

            # 保存 CSV
            if SAVE_CSV:
                save_pop_result_csv(amp, phase, ext, OUTPUT_DIR, f"surf_{surf}")

            # 保存 ZBF
            if SAVE_ZBF:
                zbf_out = Path(OUTPUT_DIR) / f"surf_{surf}.zbf"
                save_pop_result_zbf(amp, phase, ext, zbf_data, str(zbf_out))

        # 5. 完成
        print(f"\n{'=' * 60}")
        print("流水线执行完成！")
        print(f"  输出目录: {os.path.abspath(OUTPUT_DIR)}")
        print(f"{'=' * 60}")

    except Exception as e:
        print(f"\n错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if zos is not None:
            print("\n断开 Zemax 连接...")
            cleanup(zos)
            print("  已断开")
        plt.show()


if __name__ == "__main__":
    main()
