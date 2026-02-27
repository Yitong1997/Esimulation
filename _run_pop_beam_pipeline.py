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
    init_system, get_pop_field_with_beam_file, cleanup,
    parse_pop_header,
    plot_pop_result, plot_pop_overview, generate_pop_report,
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
    zemax_zbf: ZBFData,
    filepath: str,
) -> None:
    """将 Zemax POP 输出的 ZBF 数据另存为文件。

    直接使用 Zemax 原生写出的 ZBFData（含完整复数电场 Ex 及全部
    物理网格参数 dx, dy, wx, wy, Rx, Ry, zx, zy, wavelength 等），
    保证数据零损失。

    参数：
        zemax_zbf: Zemax POP 保存的输出 ZBFData
        filepath: 输出 ZBF 文件路径
    """
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_zbf(out_path, zemax_zbf)
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
START_SURFACE = 1                                           # POP 输入起始面序号
TOTAL_POWER   = None                                        # 输入光束总功率 (W)，None=峰值辐照度归一化


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
        all_results = []  # 收集各面结果用于汇总
        for surf in END_SURFACES:
            print(f"\n  --- 面 {surf} ---")

            # POP 输出 ZBF 文件名（用于 Zemax 保存）
            out_beam_name = f"_pipeline_surf_{surf}"

            amp, phase, ext = get_pop_field_with_beam_file(
                oss,
                beam_file=beam_filename,
                start_surf=START_SURFACE,
                end_surf=surf,
                sampling=SAMPLING,
                beam_width=BEAM_WIDTH,
                total_power=TOTAL_POWER,
                save_output_beam=True,
                output_beam_file=out_beam_name,
            )
            print(f"  振幅形状: {amp.shape}, 相位形状: {phase.shape}")

            # 从 Zemax 保存的输出 ZBF 读取准确的物理网格参数
            out_zbf_path = ZEMAX_POP_DIR / f"{out_beam_name}.ZBF"
            if out_zbf_path.exists():
                output_zbf = read_zbf(out_zbf_path)
                print(f"  输出 ZBF: "
                      f"nx={output_zbf.nx}, ny={output_zbf.ny}, "
                      f"dx={output_zbf.dx:.6f}, dy={output_zbf.dy:.6f}, "
                      f"wx={output_zbf.wx:.6e}, wy={output_zbf.wy:.6e}")
                if amp.shape != (output_zbf.ny, output_zbf.nx):
                    print(f"  注意: DataFrame 维度 {amp.shape} "
                          f"≠ ZBF 维度 ({output_zbf.ny}, {output_zbf.nx})")
            else:
                print(f"  警告: 未找到 Zemax 输出 ZBF ({out_zbf_path})")
                output_zbf = None

            # POP 报告头（含光束宽度等仿真结果）
            pop_hdr = ext.get("header")
            hdr_dict = parse_pop_header(pop_hdr)
            bw_x = hdr_dict.get("Beam Width X", "N/A")
            bw_y = hdr_dict.get("Beam Width Y", "N/A")
            print(f"  光束宽度: X={bw_x}, Y={bw_y}")

            # 收集结果
            surf_label = f"面 {surf}"
            if output_zbf is not None:
                all_results.append({
                    "label": surf_label,
                    "zbf": output_zbf,
                    "header": pop_hdr,
                })

            # 单面增强绘图（辐照度 + 相位 + 截面 + 物理参数标注）
            if output_zbf is not None:
                fig_path = str(Path(OUTPUT_DIR) / f"pop_surf_{surf}.png")
                plot_pop_result(
                    output_zbf, title=surf_label,
                    pop_header=pop_hdr, save_path=fig_path,
                )
                print(f"  图已保存: {fig_path}")

            # 保存 CSV（数据来源: DataFrame）
            if SAVE_CSV:
                save_pop_result_csv(amp, phase, ext, OUTPUT_DIR, f"surf_{surf}")

            # 保存 ZBF（数据来源: Zemax 原生输出）
            if SAVE_ZBF and output_zbf is not None:
                zbf_out = Path(OUTPUT_DIR) / f"surf_{surf}.zbf"
                save_pop_result_zbf(output_zbf, str(zbf_out))

        # 5. 多面汇总绘图 + 报告
        if all_results:
            overview_path = str(Path(OUTPUT_DIR) / "pop_overview.png")
            plot_pop_overview(all_results, save_path=overview_path)
            print(f"\n  汇总图已保存: {overview_path}")

            report_path = str(Path(OUTPUT_DIR) / "pop_report.txt")
            report = generate_pop_report(all_results, output_path=report_path)
            print(f"  报告已保存: {report_path}")
            print(f"\n{report}")

        # 6. 完成
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
