"""
ZMX 文件混合光学仿真主程序

本程序是混合光学仿真系统的唯一官方入口。

使用方法：
    python examples/zmx_simulation_main.py [zmx_file] [options]

示例：
    python examples/zmx_simulation_main.py
    python examples/zmx_simulation_main.py path/to/system.zmx
    python examples/zmx_simulation_main.py --visualize-only
    python examples/zmx_simulation_main.py --wavelength 0.633 --w0 5.0 --grid 512
    python examples/zmx_simulation_main.py --beam-diam-fraction 0.3

作者：混合光学仿真项目
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

# 路径配置
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'optiland-master'))

# 导入 API
from hybrid_simulation import HybridSimulator, SimulationResult
from sequential_system.zmx_visualization import ZmxOpticLoader, view_2d

# 默认配置
DEFAULT_ZMX_DIR = project_root / 'optiland-master' / 'tests' / 'zemax_files'
DEFAULT_ZMX_FILE = 'complicated_fold_mirrors_setup_v2.zmx'
DEFAULT_OUTPUT_DIR = project_root / 'output'


class ZmxSimulationMain:
    """ZMX 文件混合光学仿真主程序"""
    
    def __init__(
        self,
        zmx_path: Optional[str] = None,
        wavelength_um: float = 0.633,
        w0_mm: float = 5.0,
        grid_size: int = 256,
        num_rays: int = 200,
        beam_diam_fraction: Optional[float] = None,
        output_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        # 解析 ZMX 文件路径
        self._zmx_path = self._resolve_zmx_path(zmx_path)
        
        # 光源参数
        self._wavelength_um = wavelength_um
        self._w0_mm = w0_mm
        self._grid_size = grid_size
        self._num_rays = num_rays
        self._beam_diam_fraction = beam_diam_fraction
        self._verbose = verbose
        
        # 输出目录
        if output_dir is None:
            self._output_dir = DEFAULT_OUTPUT_DIR / self._zmx_path.stem
        else:
            self._output_dir = Path(output_dir)
        
        # 状态
        self._optic = None
        self._result: Optional[SimulationResult] = None
        
        if self._verbose:
            self._print_config()
    
    def _resolve_zmx_path(self, zmx_path: Optional[str]) -> Path:
        """解析 ZMX 文件路径"""
        if zmx_path is None:
            return DEFAULT_ZMX_DIR / DEFAULT_ZMX_FILE
        
        path = Path(zmx_path)
        if path.exists():
            return path
        if not path.is_absolute():
            for base in [project_root, DEFAULT_ZMX_DIR]:
                candidate = base / zmx_path
                if candidate.exists():
                    return candidate
        raise FileNotFoundError(f"ZMX 文件不存在: {zmx_path}")
    
    def _print_config(self):
        """打印配置信息"""
        print("=" * 60)
        print("ZMX 混合光学仿真主程序")
        print("=" * 60)
        print(f"ZMX 文件: {self._zmx_path.name}")
        print(f"波长: {self._wavelength_um} μm")
        print(f"束腰半径: {self._w0_mm} mm")
        print(f"网格大小: {self._grid_size}")
        if self._beam_diam_fraction:
            print(f"beam_diam_fraction: {self._beam_diam_fraction}")
        print(f"输出目录: {self._output_dir}")
        print("-" * 60)
    
    @property
    def zmx_path(self) -> Path:
        return self._zmx_path
    
    @property
    def result(self) -> Optional[SimulationResult]:
        return self._result
    
    def visualize(
        self,
        projection: str = 'YZ',
        num_rays: int = 5,
        save: bool = True,
        show: bool = False,
    ) -> None:
        """可视化光路"""
        if self._verbose:
            print("\n[光路可视化]")
        
        loader = ZmxOpticLoader(self._zmx_path)
        self._optic = loader.load()
        
        if self._verbose:
            loader.print_surface_info()
        
        import matplotlib.pyplot as plt
        fig, ax, _ = view_2d(self._optic, projection=projection, num_rays=num_rays)
        ax.set_title(f'{self._zmx_path.name} ({projection} 投影)')
        
        if save:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            output_file = self._output_dir / f'optical_layout_{projection}.png'
            fig.savefig(output_file, dpi=150, bbox_inches='tight')
            if self._verbose:
                print(f"光路图已保存: {output_file}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)


    def simulate(self) -> SimulationResult:
        """执行混合光学仿真"""
        if self._verbose:
            print("\n[混合光学仿真]")
        
        sim = HybridSimulator(verbose=self._verbose)
        sim.load_zmx(str(self._zmx_path))
        sim.set_source(
            wavelength_um=self._wavelength_um,
            w0_mm=self._w0_mm,
            grid_size=self._grid_size,
            beam_diam_fraction=self._beam_diam_fraction,
        )
        
        self._result = sim.run()
        return self._result
    
    def show_results(self, show: bool = True, plot_3d: bool = False) -> None:
        """展示仿真结果"""
        if self._result is None:
            print("请先执行 simulate() 方法")
            return
        
        if self._verbose:
            print("\n[仿真结果]")
        
        self._result.summary()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        
        # 2D 概览图
        overview_path = self._output_dir / 'simulation_overview.png'
        self._result.plot_all(save_path=str(overview_path), show=show)
        
        # 扩展概览图（包含 Pilot Beam 振幅和振幅残差）
        extended_path = self._output_dir / 'simulation_extended.png'
        self._result.plot_all_extended(save_path=str(extended_path), show=False)
        
        if self._verbose:
            print(f"概览图已保存: {overview_path}")
            print(f"扩展概览图已保存: {extended_path}")
        
        # 每个表面的详细图
        for surface in self._result.surfaces:
            if surface.exit is not None or surface.entrance is not None:
                # 2D 详细图
                detail_path = self._output_dir / f'surface_{surface.index}_detail.png'
                self._result.plot_surface(surface.index, save_path=str(detail_path), show=False)
                
                # 3D 详细图（可选）
                if plot_3d:
                    detail_3d_path = self._output_dir / f'surface_{surface.index}_detail_3d.png'
                    self._result.plot_surface_detail_3d(
                        surface.index, save_path=str(detail_3d_path), show=False
                    )
                
                if self._verbose:
                    print(f"表面 {surface.index} 详情图已保存")
    
    def save_results(self) -> None:
        """保存仿真结果"""
        if self._result is None:
            print("请先执行 simulate() 方法")
            return
        
        if self._verbose:
            print("\n[保存结果]")
        
        result_dir = self._output_dir / 'result_data'
        self._result.save(str(result_dir))
        
        if self._verbose:
            print(f"完整结果已保存: {result_dir}")
    
    def run_all(self, show: bool = False, plot_3d: bool = False) -> SimulationResult:
        """执行完整流程"""
        self.visualize(save=True, show=show)
        self.simulate()
        self.show_results(show=show, plot_3d=plot_3d)
        self.save_results()
        
        if self._verbose:
            print("\n" + "=" * 60)
            print("全部完成！")
            print(f"输出目录: {self._output_dir}")
            print("=" * 60)
        
        return self._result


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ZMX 文件混合光学仿真主程序')
    
    parser.add_argument('zmx_file', nargs='?', default=None, help='ZMX 文件路径')
    parser.add_argument('--wavelength', '-w', type=float, default=0.633, help='波长 (μm)')
    parser.add_argument('--w0', type=float, default=5.0, help='束腰半径 (mm)')
    parser.add_argument('--grid', '-g', type=int, default=256, help='网格大小')
    parser.add_argument('--beam-diam-fraction', type=float, default=None,
                        help='PROPER beam_diam_fraction 参数 (0.1-0.9)')
    parser.add_argument('--output', '-o', type=str, default=None, help='输出目录')
    parser.add_argument('--visualize-only', '-v', action='store_true', help='仅可视化光路')
    parser.add_argument('--plot-3d', action='store_true', help='生成 3D 图表')
    parser.add_argument('--show', '-s', action='store_true', help='显示图片窗口')
    parser.add_argument('--quiet', '-q', action='store_true', help='静默模式')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    try:
        program = ZmxSimulationMain(
            zmx_path=args.zmx_file,
            wavelength_um=args.wavelength,
            w0_mm=args.w0,
            grid_size=args.grid,
            beam_diam_fraction=args.beam_diam_fraction,
            output_dir=args.output,
            verbose=not args.quiet,
        )
        
        if args.visualize_only:
            program.visualize(save=True, show=args.show)
        else:
            program.run_all(show=args.show, plot_3d=args.plot_3d)
        
        return 0
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return 1
    except Exception as e:
        print(f"仿真失败: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == '__main__':
    sys.exit(main())
