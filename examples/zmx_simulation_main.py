"""
ZMX 文件混合光学仿真主程序

本程序是混合光学仿真系统的唯一官方入口，提供：
1. ZMX 文件加载与光路可视化
2. 全光路混合仿真（PROPER + optiland）
3. 仿真结果展示与保存

⚠️ 重要说明：
- 本程序是 "the only one and final" 主程序
- 所有测试和验证必须通过本程序完成
- 禁止自行定义或修改主程序接口

使用方法：
    python examples/zmx_simulation_main.py [zmx_file] [options]

示例：
    # 使用默认测试文件
    python examples/zmx_simulation_main.py
    
    # 指定 ZMX 文件
    python examples/zmx_simulation_main.py path/to/system.zmx
    
    # 仅可视化光路
    python examples/zmx_simulation_main.py --visualize-only
    
    # 指定光源参数
    python examples/zmx_simulation_main.py --wavelength 0.633 --w0 5.0 --grid 512

作者：混合光学仿真项目
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

# ============================================================================
# 路径配置（标准方式，不要修改）
# ============================================================================
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'optiland-master'))

# ============================================================================
# 导入模块
# ============================================================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 配置中文字体支持
def _setup_chinese_fonts():
    """设置 matplotlib 中文字体"""
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'Noto Sans SC', 'STHeiti', 'SimSun']
    available_fonts = set(f.name for f in fm.fontManager.ttflist)
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            plt.rcParams['font.family'] = 'sans-serif'
            break
    plt.rcParams['axes.unicode_minus'] = False

_setup_chinese_fonts()

from hybrid_simulation import HybridSimulator, SimulationResult
from sequential_system.zmx_visualization import (
    ZmxOpticLoader,
    visualize_zmx,
    view_2d,
)


# ============================================================================
# 默认配置
# ============================================================================
DEFAULT_ZMX_DIR = project_root / 'optiland-master' / 'tests' / 'zemax_files'
DEFAULT_ZMX_FILE = 'complicated_fold_mirrors_setup_v2.zmx'
DEFAULT_OUTPUT_DIR = project_root / 'output'

# 默认光源参数
DEFAULT_WAVELENGTH_UM = 0.633  # He-Ne 激光波长
DEFAULT_W0_MM = 5.0            # 束腰半径
DEFAULT_GRID_SIZE = 256        # 网格大小
DEFAULT_NUM_RAYS = 200         # 光线采样数



# ============================================================================
# 主程序类
# ============================================================================
class ZmxSimulationMain:
    """ZMX 文件混合光学仿真主程序
    
    这是混合光学仿真系统的唯一官方入口。
    
    功能：
    1. 加载 ZMX 文件
    2. 可视化光路
    3. 执行混合仿真
    4. 展示和保存结果
    
    使用示例：
    
        >>> main = ZmxSimulationMain("system.zmx")
        >>> main.visualize()           # 可视化光路
        >>> result = main.simulate()   # 执行仿真
        >>> main.show_results()        # 展示结果
        >>> main.save_results()        # 保存结果
    """
    
    def __init__(
        self,
        zmx_path: Optional[str] = None,
        wavelength_um: float = DEFAULT_WAVELENGTH_UM,
        w0_mm: float = DEFAULT_W0_MM,
        grid_size: int = DEFAULT_GRID_SIZE,
        num_rays: int = DEFAULT_NUM_RAYS,
        output_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        """初始化主程序
        
        参数:
            zmx_path: ZMX 文件路径，None 则使用默认测试文件
            wavelength_um: 波长 (μm)
            w0_mm: 束腰半径 (mm)
            grid_size: 网格大小
            num_rays: 光线采样数
            output_dir: 输出目录
            verbose: 是否输出详细信息
        """
        # 解析 ZMX 文件路径
        if zmx_path is None:
            self._zmx_path = DEFAULT_ZMX_DIR / DEFAULT_ZMX_FILE
        else:
            self._zmx_path = Path(zmx_path)
            if not self._zmx_path.is_absolute():
                # 尝试相对于项目根目录
                if not self._zmx_path.exists():
                    self._zmx_path = project_root / zmx_path
                # 尝试相对于默认 ZMX 目录
                if not self._zmx_path.exists():
                    self._zmx_path = DEFAULT_ZMX_DIR / zmx_path
        
        if not self._zmx_path.exists():
            raise FileNotFoundError(f"ZMX 文件不存在: {self._zmx_path}")
        
        # 光源参数
        self._wavelength_um = wavelength_um
        self._w0_mm = w0_mm
        self._grid_size = grid_size
        self._num_rays = num_rays
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
            print("=" * 60)
            print("ZMX 混合光学仿真主程序")
            print("=" * 60)
            print(f"ZMX 文件: {self._zmx_path.name}")
            print(f"波长: {self._wavelength_um} μm")
            print(f"束腰半径: {self._w0_mm} mm")
            print(f"网格大小: {self._grid_size}")
            print(f"输出目录: {self._output_dir}")
            print("-" * 60)
    
    @property
    def zmx_path(self) -> Path:
        """ZMX 文件路径"""
        return self._zmx_path
    
    @property
    def result(self) -> Optional[SimulationResult]:
        """仿真结果"""
        return self._result
    
    def visualize(
        self,
        projection: str = 'YZ',
        num_rays: int = 5,
        save: bool = True,
        show: bool = False,
    ) -> None:
        """可视化光路
        
        参数:
            projection: 投影方向 ('XY', 'XZ', 'YZ')
            num_rays: 显示的光线数量
            save: 是否保存图片
            show: 是否显示图片
        """
        if self._verbose:
            print("\n[光路可视化]")
            print("-" * 40)
        
        # 加载 ZMX 文件
        loader = ZmxOpticLoader(self._zmx_path)
        self._optic = loader.load()
        
        # 打印表面信息
        if self._verbose:
            loader.print_surface_info()
        
        # 生成 2D 可视化
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
        """执行混合光学仿真
        
        返回:
            SimulationResult 对象
        """
        if self._verbose:
            print("\n[混合光学仿真]")
            print("-" * 40)
        
        # 创建仿真器并加载 ZMX 文件
        sim = HybridSimulator(verbose=self._verbose)
        sim.load_zmx(str(self._zmx_path))
        
        # 设置光源
        sim.set_source(
            wavelength_um=self._wavelength_um,
            w0_mm=self._w0_mm,
            grid_size=self._grid_size,
        )
        
        # 执行仿真
        self._result = sim.run()
        
        return self._result
    
    def show_results(self, show: bool = True) -> None:
        """展示仿真结果
        
        参数:
            show: 是否显示图片
        """
        if self._result is None:
            print("请先执行 simulate() 方法")
            return
        
        if self._verbose:
            print("\n[仿真结果]")
            print("-" * 40)
        
        # 打印摘要
        self._result.summary()
        
        # 绘制所有表面
        self._output_dir.mkdir(parents=True, exist_ok=True)
        overview_path = self._output_dir / 'simulation_overview.png'
        self._result.plot_all(save_path=str(overview_path), show=show)
        
        if self._verbose:
            print(f"概览图已保存: {overview_path}")
        
        # 绘制每个表面的详细图
        for i, surface in enumerate(self._result.surfaces):
            if surface.exit is not None or surface.entrance is not None:
                detail_path = self._output_dir / f'surface_{surface.index}_detail.png'
                self._result.plot_surface(
                    surface.index,
                    save_path=str(detail_path),
                    show=False,
                )
                if self._verbose:
                    print(f"表面 {surface.index} 详情图已保存: {detail_path}")
    
    def save_results(self) -> None:
        """保存仿真结果"""
        if self._result is None:
            print("请先执行 simulate() 方法")
            return
        
        if self._verbose:
            print("\n[保存结果]")
            print("-" * 40)
        
        # 保存完整结果
        result_dir = self._output_dir / 'result_data'
        self._result.save(str(result_dir))
        
        if self._verbose:
            print(f"完整结果已保存: {result_dir}")
    
    def run_all(self, show: bool = False) -> SimulationResult:
        """执行完整流程：可视化 → 仿真 → 展示 → 保存
        
        参数:
            show: 是否显示图片
        
        返回:
            SimulationResult 对象
        """
        self.visualize(save=True, show=show)
        self.simulate()
        self.show_results(show=show)
        self.save_results()
        
        if self._verbose:
            print("\n" + "=" * 60)
            print("全部完成！")
            print(f"输出目录: {self._output_dir}")
            print("=" * 60)
        
        return self._result



# ============================================================================
# 命令行接口
# ============================================================================
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='ZMX 文件混合光学仿真主程序',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 使用默认测试文件
    python examples/zmx_simulation_main.py
    
    # 指定 ZMX 文件
    python examples/zmx_simulation_main.py path/to/system.zmx
    
    # 仅可视化光路
    python examples/zmx_simulation_main.py --visualize-only
    
    # 指定光源参数
    python examples/zmx_simulation_main.py --wavelength 0.633 --w0 5.0 --grid 512
        """
    )
    
    parser.add_argument(
        'zmx_file',
        nargs='?',
        default=None,
        help='ZMX 文件路径（默认使用测试文件）'
    )
    
    parser.add_argument(
        '--wavelength', '-w',
        type=float,
        default=DEFAULT_WAVELENGTH_UM,
        help=f'波长 (μm)，默认 {DEFAULT_WAVELENGTH_UM}'
    )
    
    parser.add_argument(
        '--w0',
        type=float,
        default=DEFAULT_W0_MM,
        help=f'束腰半径 (mm)，默认 {DEFAULT_W0_MM}'
    )
    
    parser.add_argument(
        '--grid', '-g',
        type=int,
        default=DEFAULT_GRID_SIZE,
        help=f'网格大小，默认 {DEFAULT_GRID_SIZE}'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='输出目录'
    )
    
    parser.add_argument(
        '--visualize-only', '-v',
        action='store_true',
        help='仅可视化光路，不执行仿真'
    )
    
    parser.add_argument(
        '--show', '-s',
        action='store_true',
        help='显示图片窗口'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='静默模式'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    try:
        # 创建主程序实例
        program = ZmxSimulationMain(
            zmx_path=args.zmx_file,
            wavelength_um=args.wavelength,
            w0_mm=args.w0,
            grid_size=args.grid,
            output_dir=args.output,
            verbose=not args.quiet,
        )
        
        if args.visualize_only:
            # 仅可视化
            program.visualize(save=True, show=args.show)
        else:
            # 执行完整流程
            program.run_all(show=args.show)
        
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
