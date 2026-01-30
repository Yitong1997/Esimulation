"""
HybridSimulator 主类

提供简洁的步骤化 API，协调光学系统定义、光源配置和仿真执行。

设计原则：
1. 主程序代码极简（< 10 行）
2. 完全复用现有模块
3. 链式调用支持
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING
import numpy as np

from .exceptions import ConfigurationError, SimulationError
from .data_models import SimulationConfig, SourceParams, SurfaceGeometry, OpticalAxisInfo
from .result import SimulationResult, SurfaceRecord, WavefrontData

if TYPE_CHECKING:
    from hybrid_optical_propagation import (
        SourceDefinition,
        HybridOpticalPropagator,
        PropagationResult,
        PropagationState,
        PilotBeamParams,
        GridSampling,
    )
    from sequential_system.coordinate_system import GlobalSurfaceDefinition


class HybridSimulator:
    """混合光学仿真器
    
    提供简洁的步骤化 API，使主程序代码极简。
    
    使用示例：
    
        >>> sim = HybridSimulator()
        >>> sim.load_zmx("system.zmx")
        >>> sim.set_source(wavelength_um=0.55, w0_mm=5.0)
        >>> result = sim.run()
        >>> result.summary()
    
    或使用链式调用：
    
        >>> result = (HybridSimulator()
        ...     .add_flat_mirror(z=50, tilt_x=45)
        ...     .set_source(wavelength_um=0.55, w0_mm=5.0)
        ...     .run())
    """
    
    def __init__(self, verbose: bool = True) -> None:
        """初始化仿真器
        
        参数:
            verbose: 是否输出详细信息
        """
        self._surfaces: List["GlobalSurfaceDefinition"] = []
        self._source: Optional["SourceDefinition"] = None
        self._wavelength_um: Optional[float] = None
        self._grid_size: int = 256
        self._physical_size_mm: Optional[float] = None
        self._num_rays: int = 200
        self._beam_diam_fraction: Optional[float] = None
        self._verbose = verbose
<<<<<<< Updated upstream
=======
        self._use_global_raytracer = use_global_raytracer
        self._propagation_method = propagation_method
        self._debug = False  # Default to False
    
    def set_debug_mode(self, debug: bool = True, debug_dir: Optional[str] = None) -> "HybridSimulator":
        """设置调试模式
        
        参数:
            debug: 是否开启调试模式（开启后会显示详细绘图）
            debug_dir: 调试输出目录（可选）
        """
        self._debug = debug
        self._debug_dir = debug_dir
        return self
>>>>>>> Stashed changes

    def load_zmx(self, path: str) -> "HybridSimulator":
        """从 ZMX 文件加载光学系统
        
        参数:
            path: ZMX 文件路径
        
        返回:
            self（支持链式调用）
        
        异常:
            FileNotFoundError: 文件不存在
            ConfigurationError: 解析错误
        """
        from hybrid_optical_propagation import load_optical_system_from_zmx
        
        try:
            self._surfaces = load_optical_system_from_zmx(path)
            if self._verbose:
                print(f"已加载光学系统: {len(self._surfaces)} 个表面")
        except FileNotFoundError:
            raise FileNotFoundError(f"ZMX 文件不存在: {path}")
        except Exception as e:
            raise ConfigurationError(f"解析 ZMX 文件失败: {e}")
        
        return self
    
    def add_flat_mirror(
        self,
        z: float,
        tilt_x: float = 0.0,
        tilt_y: float = 0.0,
        aperture: float = 25.0,
    ) -> "HybridSimulator":
        """添加平面反射镜
        
        参数:
            z: Z 位置 (mm)
            tilt_x: 绕 X 轴旋转角度（度）
            tilt_y: 绕 Y 轴旋转角度（度）
            aperture: 半口径 (mm)
        
        返回:
            self（支持链式调用）
        """
        surface = self._create_flat_mirror(z, tilt_x, tilt_y, aperture)
        self._surfaces.append(surface)
        
        if self._verbose:
            print(f"已添加平面镜: z={z}mm, tilt_x={tilt_x}°, tilt_y={tilt_y}°")
        
        return self
    
    def add_spherical_mirror(
        self,
        z: float,
        radius: float,
        tilt_x: float = 0.0,
        tilt_y: float = 0.0,
        aperture: float = 25.0,
    ) -> "HybridSimulator":
        """添加球面反射镜
        
        参数:
            z: Z 位置 (mm)
            radius: 曲率半径 (mm)，正值为凹面镜
            tilt_x: 绕 X 轴旋转角度（度）
            tilt_y: 绕 Y 轴旋转角度（度）
            aperture: 半口径 (mm)
        
        返回:
            self（支持链式调用）
        """
        surface = self._create_spherical_mirror(z, radius, tilt_x, tilt_y, aperture)
        self._surfaces.append(surface)
        
        if self._verbose:
            print(f"已添加球面镜: z={z}mm, R={radius}mm, tilt_x={tilt_x}°")
        
        return self
    
    def add_paraxial_lens(
        self,
        z: float,
        focal_length: float,
        aperture: float = 25.0,
    ) -> "HybridSimulator":
        """添加薄透镜（近轴）
        
        参数:
            z: Z 位置 (mm)
            focal_length: 焦距 (mm)
            aperture: 半口径 (mm)
        
        返回:
            self（支持链式调用）
        """
        surface = self._create_paraxial_lens(z, focal_length, aperture)
        self._surfaces.append(surface)
        
        if self._verbose:
            print(f"已添加薄透镜: z={z}mm, f={focal_length}mm")
        
        return self

    def set_source(
        self,
        wavelength_um: float,
        w0_mm: float,
        grid_size: int = 256,
        physical_size_mm: Optional[float] = None,
        z0_mm: float = 0.0,
        beam_diam_fraction: Optional[float] = None,
    ) -> "HybridSimulator":
        """定义高斯光源
        
        参数:
            wavelength_um: 波长 (μm)
            w0_mm: 束腰半径 (mm)
            grid_size: 网格大小，默认 256
            physical_size_mm: 物理尺寸 (mm)，默认 8 倍束腰
            z0_mm: 束腰位置 (mm)，默认 0
            beam_diam_fraction: PROPER beam_diam_fraction 参数（可选）
                
                该参数控制 PROPER 中光束直径与网格宽度的比例。
                
                实际效果：
                - beam_diam_fraction = beam_diameter / grid_width
                - 其中 beam_diameter = 2 × w0（束腰直径）
                - grid_width = physical_size_mm（网格物理尺寸）
                
                影响：
                - 较小的值（如 0.1-0.3）：光束占网格比例小，边缘采样更充分，
                  适合需要观察远场衍射的情况
                - 较大的值（如 0.5-0.8）：光束占网格比例大，中心区域采样更密集，
                  适合近场传播
                
                默认值 None 表示自动计算：beam_diam_fraction = 2*w0 / physical_size_mm
                
                有效范围：0.1 ~ 0.9
        
        返回:
            self（支持链式调用）
        """
        from hybrid_optical_propagation import SourceDefinition
        
        # 网格物理尺寸固定为 4 × w0（PROPER 固定用法）
        if physical_size_mm is None:
            physical_size_mm = 4 * w0_mm
        
        self._wavelength_um = wavelength_um
        self._grid_size = grid_size
        self._physical_size_mm = physical_size_mm
        self._beam_diam_fraction = beam_diam_fraction
        
        self._source = SourceDefinition(
            wavelength_um=wavelength_um,
            w0_mm=w0_mm,
            z0_mm=z0_mm,
            grid_size=grid_size,
            physical_size_mm=physical_size_mm,
            beam_diam_fraction=beam_diam_fraction,
        )
        
        if self._verbose:
            fraction_str = f", beam_diam_fraction={beam_diam_fraction}" if beam_diam_fraction else ""
            print(f"已设置光源: λ={wavelength_um}μm, w0={w0_mm}mm, grid={grid_size}{fraction_str}")
        
        return self
    
    def run(self) -> SimulationResult:
        """执行仿真
        
        返回:
            SimulationResult 对象
        
        异常:
            ConfigurationError: 配置不完整
            SimulationError: 仿真执行失败
        """
        # 验证配置
        self._validate_config()
        
        if self._verbose:
            print("开始仿真...")
        
        try:
            # 创建传播器（复用 HybridOpticalPropagator）
            from hybrid_optical_propagation import HybridOpticalPropagator
            
            propagator = HybridOpticalPropagator(
                optical_system=self._surfaces,
                source=self._source,
                wavelength_um=self._wavelength_um,
                grid_size=self._grid_size,
                num_rays=self._num_rays,
<<<<<<< Updated upstream
=======
                use_global_raytracer=self._use_global_raytracer,
                propagation_method=self._propagation_method,
                debug=self._debug,
                debug_dir=self._debug_dir,
>>>>>>> Stashed changes
            )
            
            # 执行传播
            prop_result = propagator.propagate()
            
            # 转换为 SimulationResult
            result = self._convert_result(prop_result, propagator)
            
            if self._verbose:
                if result.success:
                    print("仿真完成！")
                else:
                    print(f"仿真失败: {result.error_message}")
            
            return result
            
        except Exception as e:
            raise SimulationError(f"仿真执行失败: {e}")
    
    def _validate_config(self) -> None:
        """验证配置完整性"""
        if not self._surfaces:
            raise ConfigurationError("未定义光学系统。请先调用 load_zmx() 或 add_*() 方法。")
        
        if self._source is None:
            raise ConfigurationError("未定义光源。请先调用 set_source() 方法。")

    def _convert_result(
        self,
        prop_result: "PropagationResult",
        propagator: "HybridOpticalPropagator",
    ) -> SimulationResult:
        """将 PropagationResult 转换为 SimulationResult
        
        参数:
            prop_result: HybridOpticalPropagator 的传播结果
            propagator: 传播器实例
        
        返回:
            SimulationResult 对象
        """
        # 创建配置信息
        config = SimulationConfig(
            wavelength_um=self._wavelength_um,
            grid_size=self._grid_size,
            physical_size_mm=self._physical_size_mm,
            num_rays=self._num_rays,
            beam_diam_fraction=self._beam_diam_fraction,
        )
        
        # 创建光源参数
        wavelength_mm = self._wavelength_um * 1e-3
        z_rayleigh = np.pi * self._source.w0_mm**2 / wavelength_mm
        
        source_params = SourceParams(
            wavelength_um=self._wavelength_um,
            w0_mm=self._source.w0_mm,
            z0_mm=self._source.z0_mm,
            z_rayleigh_mm=z_rayleigh,
            grid_size=self._grid_size,
            physical_size_mm=self._physical_size_mm,
        )
        
        # 转换表面记录
        surfaces = self._convert_surface_states(prop_result, propagator)
        
        return SimulationResult(
            success=prop_result.success,
            error_message=prop_result.error_message,
            config=config,
            source_params=source_params,
            surfaces=surfaces,
            total_path_length=prop_result.total_path_length,
        )
    
    def _convert_surface_states(
        self,
        prop_result: "PropagationResult",
        propagator: "HybridOpticalPropagator",
    ) -> List[SurfaceRecord]:
        """转换表面状态列表
        
        参数:
            prop_result: 传播结果
            propagator: 传播器实例
        
        返回:
            SurfaceRecord 列表
        """
        records = []
        
        # 按表面索引分组状态
        states_by_surface = {}
        for state in prop_result.surface_states:
            idx = state.surface_index
            if idx not in states_by_surface:
                states_by_surface[idx] = {}
            states_by_surface[idx][state.position] = state
        
        # 转换每个表面
        for idx in sorted(states_by_surface.keys()):
            states = states_by_surface[idx]
            
            # 获取表面定义
            if idx >= 0 and idx < len(self._surfaces):
                surface_def = self._surfaces[idx]
                name = f"Surface_{idx}"
                surface_type = getattr(surface_def, 'surface_type', 'standard')
                geometry = self._extract_geometry(surface_def)
            else:
                name = "Initial"
                surface_type = "source"
                geometry = None
            
            # 转换波前数据
            entrance = None
            exit_data = None
            
            if 'entrance' in states:
                entrance = self._convert_wavefront_data(states['entrance'])
            if 'source' in states:
                entrance = self._convert_wavefront_data(states['source'])
            if 'exit' in states:
                exit_data = self._convert_wavefront_data(states['exit'])
            
            # 提取光轴信息
            optical_axis = self._extract_optical_axis(states)
            
            records.append(SurfaceRecord(
                index=idx,
                name=name,
                surface_type=surface_type,
                geometry=geometry,
                entrance=entrance,
                exit=exit_data,
                optical_axis=optical_axis,
            ))
        
        return records

    def _convert_wavefront_data(self, state: "PropagationState") -> WavefrontData:
        """将 PropagationState 转换为 WavefrontData
        
        参数:
            state: 传播状态
        
        返回:
            WavefrontData 对象
        """
        return WavefrontData(
            amplitude=state.amplitude.copy(),
            phase=state.phase.copy(),
            pilot_beam=state.pilot_beam_params,
            grid=state.grid_sampling,
            wavelength_um=self._wavelength_um,
        )
    
    def _extract_geometry(self, surface_def: "GlobalSurfaceDefinition") -> SurfaceGeometry:
        """从表面定义提取几何信息
        
        参数:
            surface_def: 全局表面定义
        
        返回:
            SurfaceGeometry 对象
        """
        # surface_normal 是 -Z 轴方向（指向入射侧）
        surface_normal = surface_def.surface_normal
        
        return SurfaceGeometry(
            vertex_position=np.array(surface_def.vertex_position),
            surface_normal=surface_normal,
            radius=getattr(surface_def, 'radius', np.inf),
            conic=getattr(surface_def, 'conic', 0.0),
            semi_aperture=getattr(surface_def, 'semi_aperture', 25.0),
            is_mirror=getattr(surface_def, 'is_mirror', False),
        )
    
    def _extract_optical_axis(self, states: dict) -> Optional[OpticalAxisInfo]:
        """从状态字典提取光轴信息
        
        参数:
            states: 包含 'entrance' 和/或 'exit' 状态的字典
        
        返回:
            OpticalAxisInfo 对象或 None
        """
        entrance_state = states.get('entrance') or states.get('source')
        exit_state = states.get('exit')
        
        if entrance_state is None:
            return None
        
        entrance_axis = entrance_state.optical_axis_state
        exit_axis = exit_state.optical_axis_state if exit_state else entrance_axis
        
        if entrance_axis is None:
            return None
        
        return OpticalAxisInfo(
            entrance_position=entrance_axis.position.to_array(),
            entrance_direction=entrance_axis.direction.to_array(),
            exit_position=exit_axis.position.to_array() if exit_axis else entrance_axis.position.to_array(),
            exit_direction=exit_axis.direction.to_array() if exit_axis else entrance_axis.direction.to_array(),
            path_length=exit_axis.path_length if exit_axis else entrance_axis.path_length,
        )

    def _create_flat_mirror(
        self,
        z: float,
        tilt_x: float,
        tilt_y: float,
        aperture: float,
    ) -> "GlobalSurfaceDefinition":
        """创建平面反射镜定义
        
        参数:
            z: Z 位置 (mm)
            tilt_x: 绕 X 轴旋转角度（度）
            tilt_y: 绕 Y 轴旋转角度（度）
            aperture: 半口径 (mm)
        
        返回:
            GlobalSurfaceDefinition 对象
        """
        from sequential_system.coordinate_system import GlobalSurfaceDefinition
        
        # 转换角度为弧度
        tilt_x_rad = np.radians(tilt_x)
        tilt_y_rad = np.radians(tilt_y)
        
        # 计算姿态矩阵（初始为单位矩阵，然后旋转）
        # 绕 X 轴旋转
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(tilt_x_rad), -np.sin(tilt_x_rad)],
            [0, np.sin(tilt_x_rad), np.cos(tilt_x_rad)],
        ])
        # 绕 Y 轴旋转
        Ry = np.array([
            [np.cos(tilt_y_rad), 0, np.sin(tilt_y_rad)],
            [0, 1, 0],
            [-np.sin(tilt_y_rad), 0, np.cos(tilt_y_rad)],
        ])
        
        # 组合旋转矩阵
        orientation = Ry @ Rx
        
        return GlobalSurfaceDefinition(
            index=len(self._surfaces),
            surface_type='standard',
            vertex_position=np.array([0.0, 0.0, z]),
            orientation=orientation,
            radius=np.inf,
            conic=0.0,
            semi_aperture=aperture,
            is_mirror=True,
            material='MIRROR',
        )
    
    def _create_spherical_mirror(
        self,
        z: float,
        radius: float,
        tilt_x: float,
        tilt_y: float,
        aperture: float,
    ) -> "GlobalSurfaceDefinition":
        """创建球面反射镜定义
        
        参数:
            z: Z 位置 (mm)
            radius: 曲率半径 (mm)
            tilt_x: 绕 X 轴旋转角度（度）
            tilt_y: 绕 Y 轴旋转角度（度）
            aperture: 半口径 (mm)
        
        返回:
            GlobalSurfaceDefinition 对象
        """
        from sequential_system.coordinate_system import GlobalSurfaceDefinition
        
        tilt_x_rad = np.radians(tilt_x)
        tilt_y_rad = np.radians(tilt_y)
        
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(tilt_x_rad), -np.sin(tilt_x_rad)],
            [0, np.sin(tilt_x_rad), np.cos(tilt_x_rad)],
        ])
        Ry = np.array([
            [np.cos(tilt_y_rad), 0, np.sin(tilt_y_rad)],
            [0, 1, 0],
            [-np.sin(tilt_y_rad), 0, np.cos(tilt_y_rad)],
        ])
        
        orientation = Ry @ Rx
        
        return GlobalSurfaceDefinition(
            index=len(self._surfaces),
            surface_type='standard',
            vertex_position=np.array([0.0, 0.0, z]),
            orientation=orientation,
            radius=radius,
            conic=0.0,
            semi_aperture=aperture,
            is_mirror=True,
            material='MIRROR',
        )
    
    def _create_paraxial_lens(
        self,
        z: float,
        focal_length: float,
        aperture: float,
    ) -> "GlobalSurfaceDefinition":
        """创建薄透镜定义
        
        参数:
            z: Z 位置 (mm)
            focal_length: 焦距 (mm)
            aperture: 半口径 (mm)
        
        返回:
            GlobalSurfaceDefinition 对象
        """
        from sequential_system.coordinate_system import GlobalSurfaceDefinition
        
        return GlobalSurfaceDefinition(
            index=len(self._surfaces),
            surface_type='paraxial',
            vertex_position=np.array([0.0, 0.0, z]),
            orientation=np.eye(3),
            radius=np.inf,
            conic=0.0,
            semi_aperture=aperture,
            is_mirror=False,
            material='',
            focal_length=focal_length,
        )
