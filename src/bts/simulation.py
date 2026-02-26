"""
simulate 函数模块

提供仿真执行功能，内部调用 HybridSimulator.run()。
"""

from typing import TYPE_CHECKING

from .exceptions import ConfigurationError, SimulationError

if TYPE_CHECKING:
    from .optical_system import OpticalSystem
    from .source import GaussianSource
    from hybrid_simulation import SimulationResult


def simulate(
    system: "OpticalSystem",
    source: "GaussianSource",
    verbose: bool = True,
    num_rays: int = 4096,
    use_global_raytracer: bool = False,
    propagation_method: str = "local_raytracing",
    debug: bool = False,
) -> "SimulationResult":
    """执行混合光学仿真
    
    内部直接调用 HybridSimulator，不重新实现任何逻辑。
    
    参数:
        system: 光学系统定义
        source: 高斯光源定义
        verbose: 是否输出详细信息，默认 True
        num_rays: 光线追迹使用的光线数量，默认 4096
        use_global_raytracer: 是否使用全局坐标系光线追迹器，默认 False
            - False: 使用 HybridElementPropagator（局部坐标系）
            - True: 使用 HybridElementPropagatorGlobal（全局坐标系）
        propagation_method: 元件传播方法，默认 'local_raytracing'
            - 'local_raytracing': 局部光线追迹方法（默认）
            - 'pure_diffraction': 纯衍射方法（使用 tilted_asm 投影传输）
        debug: 是否开启调试模式，默认 False
    
    返回:
        SimulationResult 对象，包含所有表面的波前数据
    
    异常:
        ConfigurationError: 配置不完整（如空系统）
        SimulationError: 仿真执行失败
        ValueError: 参数值无效（如负波长）
    
    示例:
        >>> import bts
        >>> system = bts.load_zmx("system.zmx")
        >>> source = bts.GaussianSource(wavelength_um=0.633, w0_mm=5.0)
        >>> result = bts.simulate(system, source)
        >>> result.summary()
        
        # 使用全局坐标系光线追迹
        >>> result = bts.simulate(system, source, use_global_raytracer=True)
        
        # 使用纯衍射方法
        >>> result = bts.simulate(system, source, propagation_method='pure_diffraction')
    
    **Validates: Requirements 1.4, 5.1, 5.2, 5.3, 5.4, 5.5**
    """
    # 1. 验证系统配置
    print(f"[bts.simulate] Called with debug={debug}")
    if len(system) == 0:
        raise ConfigurationError("光学系统为空，请先添加表面或加载 ZMX 文件")
    
    # 2. 验证光源配置
    if source.wavelength_um <= 0:
        raise ValueError(f"无效参数: wavelength_um = {source.wavelength_um}，必须为正数")
    if source.w0_mm <= 0:
        raise ValueError(f"无效参数: w0_mm = {source.w0_mm}，必须为正数")
    if source.grid_size <= 0:
        raise ValueError(f"无效参数: grid_size = {source.grid_size}，必须为正整数")
    if source.physical_size_mm <= 0:
        raise ValueError(f"无效参数: physical_size_mm = {source.physical_size_mm}，必须为正数")
    
    # 3. 执行仿真（捕获内部异常）
    try:
        from hybrid_simulation import HybridSimulator
        
        # 创建 HybridSimulator 实例
        sim = HybridSimulator(
            verbose=verbose,
            use_global_raytracer=use_global_raytracer,
            propagation_method=propagation_method,
        )
        
        # 设置调试模式
        sim.set_debug_mode(debug)
        
        # 复用现有的表面定义
        # 优先使用 _global_surfaces（从 ZMX 加载或手动创建的全局坐标表面）
        global_surfaces = system.get_global_surfaces()
        if global_surfaces:
            sim._surfaces = global_surfaces
        else:
            # 如果没有全局表面定义，尝试从 _surfaces 创建
            raise ConfigurationError(
                "光学系统缺少全局坐标表面定义。"
                "请使用 add_surface()、add_flat_mirror() 等方法添加表面，"
                "或使用 load_zmx() 从 ZMX 文件加载。"
            )
        
        # 设置光线数量
        sim._num_rays = num_rays
        
        # 复用现有的光源定义
        sim.set_source(
            wavelength_um=source.wavelength_um,
            w0_mm=source.w0_mm,
            grid_size=source.grid_size,
            physical_size_mm=source.physical_size_mm,
            z0_mm=source.z0_mm,
            beam_diam_fraction=source.beam_diam_fraction,
        )
        
        # 调用现有的 run() 方法
        result = sim.run()
        
        return result
        
    except ConfigurationError:
        # 直接重新抛出 ConfigurationError
        raise
    except SimulationError:
        # 直接重新抛出 SimulationError
        raise
    except ValueError:
        # 直接重新抛出 ValueError
        raise
    except Exception as e:
        # 其他异常包装为 SimulationError
        raise SimulationError(f"仿真执行失败: {e}") from e
