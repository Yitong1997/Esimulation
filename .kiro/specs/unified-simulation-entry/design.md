# Design Document

## Overview

本设计文档描述"统一仿真入口"功能的技术实现方案。

**设计目标**：
1. 主程序代码极简（< 10 行）
2. 结果存储全面（支持任意后续测试）
3. 完全复用现有模块

## Architecture

### 模块结构

```
src/hybrid_simulation/
├── __init__.py           # 导出 HybridSimulator, SimulationResult
├── simulator.py          # HybridSimulator 主类
├── result.py             # SimulationResult, SurfaceRecord, WavefrontData
├── plotting.py           # 可视化函数
└── exceptions.py         # 异常类
```

### 类图

```
┌─────────────────────────────────────────────────────────────────┐
│                      HybridSimulator                             │
├─────────────────────────────────────────────────────────────────┤
│ - _surfaces: List[SurfaceDefinition]                            │
│ - _source: SourceDefinition                                      │
│ - _wavelength_um: float                                          │
│ - _verbose: bool                                                 │
├─────────────────────────────────────────────────────────────────┤
│ + load_zmx(path) -> self                                        │
│ + add_flat_mirror(z, tilt_x, tilt_y, aperture) -> self          │
│ + add_spherical_mirror(z, radius, tilt_x, tilt_y) -> self       │
│ + add_paraxial_lens(z, focal_length) -> self                    │
│ + set_source(wavelength_um, w0_mm, grid_size, ...) -> self      │
│ + run() -> SimulationResult                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SimulationResult                            │
├─────────────────────────────────────────────────────────────────┤
│ + success: bool                                                  │
│ + error_message: str                                             │
│ + config: SimulationConfig                                       │
│ + source_params: SourceParams                                    │
│ + surfaces: List[SurfaceRecord]                                  │
│ + total_path_length: float                                       │
├─────────────────────────────────────────────────────────────────┤
│ + get_surface(index_or_name) -> SurfaceRecord                   │
│ + summary() -> None                                              │
│ + plot_all(save_path, show) -> None                             │
│ + plot_surface(index, save_path, show) -> None                  │
│ + save(path) -> None                                             │
│ + load(path) -> SimulationResult [classmethod]                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       SurfaceRecord                              │
├─────────────────────────────────────────────────────────────────┤
│ + index: int                                                     │
│ + name: str                                                      │
│ + surface_type: str                                              │
│ + geometry: SurfaceGeometry                                      │
│ + entrance: WavefrontData                                        │
│ + exit: WavefrontData                                            │
│ + optical_axis: OpticalAxisInfo                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       WavefrontData                              │
├─────────────────────────────────────────────────────────────────┤
│ + amplitude: np.ndarray                                          │
│ + phase: np.ndarray                                              │
│ + pilot_beam: PilotBeamInfo                                      │
│ + grid: GridInfo                                                 │
├─────────────────────────────────────────────────────────────────┤
│ + get_intensity() -> np.ndarray                                  │
│ + get_complex_amplitude() -> np.ndarray                          │
│ + get_residual_phase() -> np.ndarray                             │
│ + get_pilot_beam_phase() -> np.ndarray                           │
└─────────────────────────────────────────────────────────────────┘
```

## Data Models

### SimulationConfig

```python
@dataclass
class SimulationConfig:
    """仿真配置信息"""
    wavelength_um: float          # 波长 (μm)
    grid_size: int                # 网格大小
    physical_size_mm: float       # 物理尺寸 (mm)
    num_rays: int                 # 光线采样数量
    propagation_method: str       # 传播方法
```

### SourceParams

```python
@dataclass
class SourceParams:
    """光源参数"""
    wavelength_um: float          # 波长 (μm)
    w0_mm: float                  # 束腰半径 (mm)
    z0_mm: float                  # 束腰位置 (mm)
    z_rayleigh_mm: float          # 瑞利长度 (mm)
    grid_size: int                # 网格大小
    physical_size_mm: float       # 物理尺寸 (mm)
```

### SurfaceGeometry

```python
@dataclass
class SurfaceGeometry:
    """表面几何信息"""
    vertex_position: np.ndarray   # 顶点位置 [x, y, z] (mm)
    surface_normal: np.ndarray    # 表面法向量
    radius: float                 # 曲率半径 (mm)，平面为 inf
    conic: float                  # 圆锥常数
    semi_aperture: float          # 半口径 (mm)
    is_mirror: bool               # 是否为反射镜
```

### OpticalAxisInfo

```python
@dataclass
class OpticalAxisInfo:
    """光轴状态信息"""
    entrance_position: np.ndarray   # 入射点位置 (mm)
    entrance_direction: np.ndarray  # 入射方向
    exit_position: np.ndarray       # 出射点位置 (mm)
    exit_direction: np.ndarray      # 出射方向
    path_length: float              # 累积光程 (mm)
```

### PilotBeamInfo

```python
@dataclass
class PilotBeamInfo:
    """Pilot Beam 参数"""
    w0_mm: float                  # 束腰半径 (mm)
    z_w0_mm: float                # 束腰位置 (mm)
    z_mm: float                   # 当前位置 (mm)
    z_rayleigh_mm: float          # 瑞利长度 (mm)
    curvature_radius_mm: float    # 曲率半径 (mm)
    spot_size_mm: float           # 当前光斑大小 (mm)
```

### GridInfo

```python
@dataclass
class GridInfo:
    """网格采样信息"""
    grid_size: int                # 网格大小
    physical_size_mm: float       # 物理尺寸 (mm)
    pixel_size_mm: float          # 像素大小 (mm)
```

## Component Design

### HybridSimulator

**职责**：提供简洁的步骤化 API，协调光学系统定义、光源配置和仿真执行。

**关键实现**：

```python
class HybridSimulator:
    def __init__(self, verbose: bool = True):
        self._surfaces = []
        self._source = None
        self._wavelength_um = None
        self._verbose = verbose
    
    def load_zmx(self, path: str) -> "HybridSimulator":
        """从 ZMX 文件加载光学系统"""
        # 复用 load_optical_system_from_zmx
        self._surfaces = load_optical_system_from_zmx(path)
        return self
    
    def add_flat_mirror(
        self,
        z: float,
        tilt_x: float = 0.0,
        tilt_y: float = 0.0,
        aperture: float = 25.0,
    ) -> "HybridSimulator":
        """添加平面反射镜（角度单位：度）"""
        # 创建 GlobalSurfaceDefinition
        surface = self._create_flat_mirror(z, tilt_x, tilt_y, aperture)
        self._surfaces.append(surface)
        return self
    
    def set_source(
        self,
        wavelength_um: float,
        w0_mm: float,
        grid_size: int = 256,
        physical_size_mm: float = None,
    ) -> "HybridSimulator":
        """定义高斯光源"""
        if physical_size_mm is None:
            physical_size_mm = 8 * w0_mm  # 默认 8 倍束腰
        
        self._wavelength_um = wavelength_um
        self._source = SourceDefinition(
            wavelength_um=wavelength_um,
            w0_mm=w0_mm,
            z0_mm=0.0,
            grid_size=grid_size,
            physical_size_mm=physical_size_mm,
        )
        return self
    
    def run(self) -> SimulationResult:
        """执行仿真"""
        # 验证配置
        self._validate_config()
        
        # 创建传播器（复用 HybridOpticalPropagator）
        propagator = HybridOpticalPropagator(
            optical_system=self._surfaces,
            source=self._source,
            wavelength_um=self._wavelength_um,
        )
        
        # 执行传播
        prop_result = propagator.propagate()
        
        # 转换为 SimulationResult
        return self._convert_result(prop_result, propagator)
```

### SimulationResult

**职责**：存储所有仿真结果，提供便捷的数据访问和可视化接口。

**关键实现**：

```python
class SimulationResult:
    def __init__(
        self,
        success: bool,
        error_message: str,
        config: SimulationConfig,
        source_params: SourceParams,
        surfaces: List[SurfaceRecord],
        total_path_length: float,
    ):
        self.success = success
        self.error_message = error_message
        self.config = config
        self.source_params = source_params
        self.surfaces = surfaces
        self.total_path_length = total_path_length
    
    def get_surface(self, index_or_name) -> SurfaceRecord:
        """通过索引或名称获取表面记录"""
        if isinstance(index_or_name, int):
            return self.surfaces[index_or_name]
        for surface in self.surfaces:
            if surface.name == index_or_name:
                return surface
        raise KeyError(f"未找到表面: {index_or_name}")
    
    def summary(self) -> None:
        """打印仿真摘要"""
        print("=" * 60)
        print("混合光学仿真结果摘要")
        print("=" * 60)
        print(f"状态: {'成功' if self.success else '失败'}")
        if not self.success:
            print(f"错误: {self.error_message}")
        print(f"波长: {self.config.wavelength_um} μm")
        print(f"网格: {self.config.grid_size} × {self.config.grid_size}")
        print(f"表面数量: {len(self.surfaces)}")
        print(f"总光程: {self.total_path_length:.2f} mm")
        print("-" * 60)
        for surf in self.surfaces:
            print(f"  [{surf.index}] {surf.name}: {surf.surface_type}")
            if surf.exit:
                rms = surf.exit.get_residual_rms_waves()
                print(f"       出射相位残差 RMS: {rms:.6f} waves")
    
    def plot_all(self, save_path: str = None, show: bool = True) -> None:
        """绘制所有表面的振幅和相位"""
        from .plotting import plot_all_surfaces
        plot_all_surfaces(self, save_path, show)
    
    def plot_surface(
        self,
        index: int,
        save_path: str = None,
        show: bool = True,
    ) -> None:
        """绘制指定表面的详细图表"""
        from .plotting import plot_surface_detail
        plot_surface_detail(self.surfaces[index], save_path, show)
    
    def save(self, path: str) -> None:
        """保存结果到目录"""
        from .serialization import save_result
        save_result(self, path)
    
    @classmethod
    def load(cls, path: str) -> "SimulationResult":
        """从目录加载结果"""
        from .serialization import load_result
        return load_result(path)
```

### WavefrontData

**职责**：封装单个位置的波前数据，提供便捷的计算方法。

```python
class WavefrontData:
    def __init__(
        self,
        amplitude: np.ndarray,
        phase: np.ndarray,
        pilot_beam: PilotBeamInfo,
        grid: GridInfo,
    ):
        self.amplitude = amplitude
        self.phase = phase
        self.pilot_beam = pilot_beam
        self.grid = grid
    
    def get_intensity(self) -> np.ndarray:
        """计算光强分布"""
        return self.amplitude ** 2
    
    def get_complex_amplitude(self) -> np.ndarray:
        """获取复振幅"""
        return self.amplitude * np.exp(1j * self.phase)
    
    def get_pilot_beam_phase(self) -> np.ndarray:
        """计算 Pilot Beam 参考相位"""
        half_size = self.grid.physical_size_mm / 2
        coords = np.linspace(-half_size, half_size, self.grid.grid_size)
        X, Y = np.meshgrid(coords, coords)
        r_sq = X**2 + Y**2
        
        R = self.pilot_beam.curvature_radius_mm
        if np.isinf(R):
            return np.zeros_like(r_sq)
        
        k = 2 * np.pi / (self.pilot_beam.wavelength_um * 1e-3)
        return k * r_sq / (2 * R)
    
    def get_residual_phase(self) -> np.ndarray:
        """计算相对于 Pilot Beam 的残差相位"""
        pilot_phase = self.get_pilot_beam_phase()
        return np.angle(np.exp(1j * (self.phase - pilot_phase)))
    
    def get_residual_rms_waves(self) -> float:
        """计算残差相位 RMS（波长数）"""
        residual = self.get_residual_phase()
        # 有效区域掩模
        norm_amp = self.amplitude / np.max(self.amplitude)
        valid_mask = norm_amp > 0.01
        if np.sum(valid_mask) == 0:
            return np.nan
        rms_rad = np.sqrt(np.mean(residual[valid_mask]**2))
        return rms_rad / (2 * np.pi)
```

## Integration with Existing Modules

### 复用关系

| 新组件 | 复用的现有组件 |
|--------|----------------|
| HybridSimulator.load_zmx() | load_optical_system_from_zmx() |
| HybridSimulator.set_source() | SourceDefinition |
| HybridSimulator.run() | HybridOpticalPropagator |
| WavefrontData | PropagationState |
| PilotBeamInfo | PilotBeamParams |
| GridInfo | GridSampling |

### 数据转换

```python
def _convert_propagation_state_to_wavefront_data(
    state: PropagationState,
    wavelength_um: float,
) -> WavefrontData:
    """将 PropagationState 转换为 WavefrontData"""
    pilot_beam = PilotBeamInfo(
        w0_mm=state.pilot_beam_params.w0_mm,
        z_w0_mm=state.pilot_beam_params.z_w0_mm,
        z_mm=state.pilot_beam_params.z_mm,
        z_rayleigh_mm=state.pilot_beam_params.z_rayleigh_mm,
        curvature_radius_mm=state.pilot_beam_params.curvature_radius_mm,
        spot_size_mm=state.pilot_beam_params.spot_size_mm,
        wavelength_um=wavelength_um,
    )
    
    grid = GridInfo(
        grid_size=state.grid_sampling.grid_size,
        physical_size_mm=state.grid_sampling.physical_size_mm,
        pixel_size_mm=state.grid_sampling.pixel_size_mm,
    )
    
    return WavefrontData(
        amplitude=state.amplitude.copy(),
        phase=state.phase.copy(),
        pilot_beam=pilot_beam,
        grid=grid,
    )
```

## Serialization Format

### 目录结构

```
output/
├── config.json           # 仿真配置
├── source.json           # 光源参数
├── summary.json          # 结果摘要
└── surfaces/
    ├── 0_Initial/
    │   ├── entrance_amplitude.npy
    │   ├── entrance_phase.npy
    │   ├── entrance_pilot_beam.json
    │   └── entrance_grid.json
    ├── 1_M1/
    │   ├── entrance_amplitude.npy
    │   ├── entrance_phase.npy
    │   ├── entrance_pilot_beam.json
    │   ├── entrance_grid.json
    │   ├── exit_amplitude.npy
    │   ├── exit_phase.npy
    │   ├── exit_pilot_beam.json
    │   ├── exit_grid.json
    │   ├── geometry.json
    │   └── optical_axis.json
    └── ...
```

## Correctness Properties

### Property 1: 结果完整性

**描述**：SimulationResult 必须包含所有表面的完整数据。

```python
def property_result_completeness(result: SimulationResult):
    """验证结果完整性"""
    for surface in result.surfaces:
        # 每个表面必须有入射面数据
        assert surface.entrance is not None
        assert surface.entrance.amplitude is not None
        assert surface.entrance.phase is not None
        assert surface.entrance.pilot_beam is not None
        
        # 非初始表面必须有出射面数据
        if surface.index >= 0:
            assert surface.exit is not None
```

### Property 2: 序列化可逆性

**描述**：保存和加载的结果必须与原始结果一致。

```python
def property_serialization_reversibility(result: SimulationResult, tmp_path):
    """验证序列化可逆性"""
    result.save(tmp_path)
    loaded = SimulationResult.load(tmp_path)
    
    assert loaded.success == result.success
    assert len(loaded.surfaces) == len(result.surfaces)
    
    for orig, load in zip(result.surfaces, loaded.surfaces):
        np.testing.assert_array_almost_equal(
            orig.entrance.amplitude,
            load.entrance.amplitude,
        )
        np.testing.assert_array_almost_equal(
            orig.entrance.phase,
            load.entrance.phase,
        )
```

### Property 3: Pilot Beam 一致性

**描述**：WavefrontData 的 Pilot Beam 参数必须与 PropagationState 一致。

```python
def property_pilot_beam_consistency(wavefront: WavefrontData, state: PropagationState):
    """验证 Pilot Beam 参数一致性"""
    assert wavefront.pilot_beam.w0_mm == state.pilot_beam_params.w0_mm
    assert wavefront.pilot_beam.curvature_radius_mm == state.pilot_beam_params.curvature_radius_mm
```

## File Structure

```
src/hybrid_simulation/
├── __init__.py           # 导出公共 API
├── simulator.py          # HybridSimulator 类
├── result.py             # SimulationResult, SurfaceRecord, WavefrontData
├── data_models.py        # SimulationConfig, SourceParams, etc.
├── plotting.py           # 可视化函数
├── serialization.py      # 保存/加载函数
└── exceptions.py         # ConfigurationError 等
```
