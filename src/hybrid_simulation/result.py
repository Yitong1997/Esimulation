"""
仿真结果类

定义 WavefrontData、SurfaceRecord 和 SimulationResult 类，
用于存储和访问仿真结果。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from .data_models import (
    SimulationConfig, SourceParams, SurfaceGeometry, OpticalAxisInfo,
    RayData, ChiefRayData, CoordinateSystemData, PilotBeamParamsData,
)

if TYPE_CHECKING:
    from hybrid_optical_propagation import PilotBeamParams, GridSampling


@dataclass
class WavefrontData:
    """波前数据
    
    封装单个位置的波前数据，提供便捷的计算方法。
    
    属性:
        amplitude: 振幅网格（实数，非负）
        phase: 相位网格（实数，非折叠，弧度）
        pilot_beam: Pilot Beam 参数
        grid: 网格采样信息
        wavelength_um: 波长 (μm)
    """
    amplitude: NDArray[np.floating]
    phase: NDArray[np.floating]
    pilot_beam: "PilotBeamParams"
    grid: "GridSampling"
    wavelength_um: float
    
    def get_intensity(self) -> NDArray[np.floating]:
        """计算光强分布
        
        返回:
            光强数组 (amplitude²)
        """
        return self.amplitude ** 2
    
    def get_complex_amplitude(self) -> NDArray[np.complexfloating]:
        """获取复振幅
        
        注意：返回的复振幅会有相位折叠。
        
        返回:
            复振幅数组
        """
        return self.amplitude * np.exp(1j * self.phase)

    def get_pilot_beam_phase(self) -> NDArray[np.floating]:
        """计算 Pilot Beam 参考相位
        
        返回:
            参考相位网格 (弧度)
        """
        return self.pilot_beam.compute_phase_grid(
            self.grid.grid_size,
            self.grid.physical_size_mm,
        )
    
    def get_pilot_beam_amplitude(self) -> NDArray[np.floating]:
        """计算 Pilot Beam 参考振幅（高斯分布）
        
        返回:
            Pilot Beam 振幅网格，归一化到与实际振幅相同的峰值
        """
        n = self.grid.grid_size
        n = self.grid.grid_size
        dx = self.grid.physical_size_mm / n
        coords = (np.arange(n) - n // 2) * dx
        X, Y = np.meshgrid(coords, coords)
        r_sq = X**2 + Y**2
        
        # 使用 Pilot Beam 的光斑大小
        w = self.pilot_beam.spot_size_mm
        
        # 高斯振幅分布
        pilot_amplitude = np.exp(-r_sq / w**2)
        
        # 归一化到与实际振幅相同的峰值
        max_actual = np.max(self.amplitude)
        if max_actual > 0:
            pilot_amplitude = pilot_amplitude * max_actual
        
        return pilot_amplitude
    
    def get_residual_phase(self) -> NDArray[np.floating]:
        """计算相对于 Pilot Beam 的残差相位
        
        返回:
            残差相位网格 (弧度)，范围 [-π, π]
        """
        pilot_phase = self.get_pilot_beam_phase()
        return np.angle(np.exp(1j * (self.phase - pilot_phase)))
    
    def get_residual_amplitude(self) -> NDArray[np.floating]:
        """计算振幅残差（实际振幅 - Pilot Beam 振幅）
        
        返回:
            振幅残差网格
        """
        pilot_amplitude = self.get_pilot_beam_amplitude()
        return self.amplitude - pilot_amplitude
    
    def get_residual_rms_waves(self) -> float:
        """计算残差相位 RMS（波长数）
        
        仅在有效区域（振幅 > 1% 最大值）内计算。
        
        返回:
            残差 RMS (waves)
        """
        residual = self.get_residual_phase()
        
        # 有效区域掩模
        # Debug Amplitude
        if np.any(np.isnan(self.amplitude)):
            print(f"DEBUG: Amplitude contains NaNs! Count: {np.sum(np.isnan(self.amplitude))}")
        print(f"DEBUG: Amplitude Max: {np.nanmax(self.amplitude)}")
            
        norm_amp = self.amplitude / np.nanmax(self.amplitude) if np.nanmax(self.amplitude) > 0 else self.amplitude
        valid_mask = norm_amp > 0.01
        
        if np.sum(valid_mask) == 0:
            return np.nan
        
        # Check for NaNs in residual within valid mask
        valid_residual = residual[valid_mask]
        
        if np.any(np.isnan(valid_residual)):
            # print("Warning: NaNs detected in residual phase within valid readout region. Ignoring NaNs.")
            rms_rad = np.sqrt(np.nanmean(valid_residual ** 2))
        else:
            rms_rad = np.sqrt(np.mean(valid_residual ** 2))
        
        return rms_rad / (2 * np.pi)
    
    def get_residual_pv_waves(self) -> float:
        """计算残差相位 PV（波长数）
        
        仅在有效区域内计算。
        
        返回:
            残差 PV (waves)
        """
        residual = self.get_residual_phase()
        
        # 有效区域掩模
        norm_amp = self.amplitude / np.max(self.amplitude) if np.max(self.amplitude) > 0 else self.amplitude
        valid_mask = norm_amp > 0.01
        
        if np.sum(valid_mask) == 0:
            return np.nan
        
        pv_rad = np.max(residual[valid_mask]) - np.min(residual[valid_mask])
        return pv_rad / (2 * np.pi)
    
    def get_amplitude_residual_rms(self) -> float:
        """计算振幅残差 RMS
        
        仅在有效区域内计算。
        
        返回:
            振幅残差 RMS
        """
        residual = self.get_residual_amplitude()
        
        norm_amp = self.amplitude / np.max(self.amplitude) if np.max(self.amplitude) > 0 else self.amplitude
        valid_mask = norm_amp > 0.01
        
        if np.sum(valid_mask) == 0:
            return np.nan
        
        return np.sqrt(np.mean(residual[valid_mask] ** 2))
    
    def plot(
        self,
        title: str = "Wavefront Analysis",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """绘制波前分析图
        
        参数:
            title: 图表标题
            save_path: 保存路径（可选）
            show: 是否显示图表
        
        返回:
            matplotlib Figure 对象（如果 show=False）
        """
        from .plotting import plot_wavefront
        return plot_wavefront(self, title, save_path, show)


@dataclass
class SurfaceRecord:
    """表面记录
    
    存储单个表面的完整数据。
    
    属性:
        index: 表面索引
        name: 表面名称
        surface_type: 表面类型（如 'standard', 'paraxial', 'coordbrk'）
        geometry: 表面几何信息
        entrance: 入射面波前数据
        exit: 出射面波前数据
        optical_axis: 光轴状态信息
        entrance_rays: 入射面光线数据（可选，用于调试）
        exit_rays: 出射面光线数据（可选，用于调试）
        chief_ray: 主光线数据（可选，用于调试）
        entrance_coord_system: 入射面坐标系（可选，用于调试）
        exit_coord_system: 出射面坐标系（可选，用于调试）
    """
    index: int
    name: str
    surface_type: str
    geometry: Optional[SurfaceGeometry]
    entrance: Optional[WavefrontData]
    exit: Optional[WavefrontData]
    optical_axis: Optional[OpticalAxisInfo]
    # 调试数据字段（可选）
    entrance_rays: Optional["RayData"] = None
    exit_rays: Optional["RayData"] = None
    chief_ray: Optional["ChiefRayData"] = None
    entrance_coord_system: Optional["CoordinateSystemData"] = None
    exit_coord_system: Optional["CoordinateSystemData"] = None


@dataclass
class SimulationResult:
    """仿真结果
    
    存储完整仿真过程的所有结果，提供便捷的数据访问和可视化接口。
    
    属性:
        success: 仿真是否成功
        error_message: 错误信息（如果失败）
        config: 仿真配置
        source_params: 光源参数
        surfaces: 表面记录列表
        total_path_length: 总光程 (mm)
    """
    success: bool
    error_message: str
    config: SimulationConfig
    source_params: SourceParams
    surfaces: List[SurfaceRecord]
    total_path_length: float
    
    def get_surface(self, index_or_name: Union[int, str]) -> SurfaceRecord:
        """通过索引或名称获取表面记录
        
        参数:
            index_or_name: 表面索引（int）或名称（str）
        
        返回:
            SurfaceRecord 对象
        
        异常:
            KeyError: 未找到指定表面
        """
        if isinstance(index_or_name, int):
            for surface in self.surfaces:
                if surface.index == index_or_name:
                    return surface
            raise KeyError(f"未找到索引为 {index_or_name} 的表面")
        
        for surface in self.surfaces:
            if surface.name == index_or_name:
                return surface
        raise KeyError(f"未找到名称为 '{index_or_name}' 的表面")
    
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
            if surf.exit is not None:
                rms = surf.exit.get_residual_rms_waves()
                pv = surf.exit.get_residual_pv_waves()
                print(f"       出射相位残差: RMS={rms:.6f} waves, PV={pv:.4f} waves")
        
        print("=" * 60)
    
    def plot_all(self, save_path: Optional[str] = None, show: bool = True) -> None:
        """绘制所有表面的振幅和相位（2D）
        
        参数:
            save_path: 保存路径（可选）
            show: 是否显示图表
        """
        from .plotting import plot_all_surfaces
        plot_all_surfaces(self, save_path, show)
    
    def plot_all_extended(self, save_path: Optional[str] = None, show: bool = True) -> None:
        """绘制所有表面的扩展概览图（2D）
        
        包含：振幅、残差相位、Pilot Beam 振幅、振幅残差
        
        参数:
            save_path: 保存路径（可选）
            show: 是否显示图表
        """
        from .plotting import plot_all_surfaces_extended
        plot_all_surfaces_extended(self, save_path, show)
    
    def plot_surface(
        self,
        index: int,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """绘制指定表面的详细图表（2D）
        
        参数:
            index: 表面索引
            save_path: 保存路径（可选）
            show: 是否显示图表
        """
        from .plotting import plot_surface_detail
        surface = self.get_surface(index)
        plot_surface_detail(surface, self.config.wavelength_um, save_path, show)
    
    def plot_surface_3d(
        self,
        index: int,
        plot_type: str = "residual_phase",
        save_path: Optional[str] = None,
        show: bool = True,
        elevation: float = 30,
        azimuth: float = -60,
    ) -> None:
        """绘制指定表面的 3D 图表
        
        参数:
            index: 表面索引
            plot_type: 绘图类型，可选：
                - "amplitude": 振幅
                - "phase": 相位
                - "residual_phase": 残差相位
                - "pilot_amplitude": Pilot Beam 振幅
                - "residual_amplitude": 振幅残差
            save_path: 保存路径（可选）
            show: 是否显示图表
            elevation: 3D 视角仰角（度）
            azimuth: 3D 视角方位角（度）
        """
        from .plotting import plot_surface_3d
        surface = self.get_surface(index)
        plot_surface_3d(
            surface, self.config.wavelength_um, plot_type,
            save_path, show, elevation, azimuth
        )
    
    def plot_surface_detail_3d(
        self,
        index: int,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """绘制指定表面的完整 3D 详细图表
        
        包含 6 个子图：振幅、相位、Pilot Beam 相位、残差相位、
                      Pilot Beam 振幅、振幅残差
        
        参数:
            index: 表面索引
            save_path: 保存路径（可选）
            show: 是否显示图表
        """
        from .plotting import plot_surface_detail_3d
        surface = self.get_surface(index)
        plot_surface_detail_3d(surface, self.config.wavelength_um, save_path, show)
    
    def save(self, path: str) -> None:
        """保存结果到目录
        
        参数:
            path: 保存目录路径
        """
        from .serialization import save_result
        save_result(self, path)
    
    @classmethod
    def load(cls, path: str) -> "SimulationResult":
        """从目录加载结果
        
        参数:
            path: 目录路径
        
        返回:
            SimulationResult 对象
        """
        from .serialization import load_result
        return load_result(path)
    
    def get_final_wavefront(self) -> WavefrontData:
        """获取最终表面的出射波前数据
        
        返回:
            最终表面的出射 WavefrontData 对象
        
        异常:
            ValueError: 没有表面或最终表面没有出射波前数据
        """
        if not self.surfaces:
            raise ValueError("仿真结果中没有表面数据")
        
        final_surface = self.surfaces[-1]
        if final_surface.exit is None:
            raise ValueError(f"最终表面 [{final_surface.index}] {final_surface.name} 没有出射波前数据")
        
        return final_surface.exit
    
    def get_entrance_wavefront(self, surface_index: int) -> WavefrontData:
        """获取指定表面的入射波前数据
        
        参数:
            surface_index: 表面索引
        
        返回:
            指定表面的入射 WavefrontData 对象
        
        异常:
            KeyError: 未找到指定索引的表面
            ValueError: 指定表面没有入射波前数据
        """
        surface = self.get_surface(surface_index)
        
        if surface.entrance is None:
            raise ValueError(f"表面 [{surface.index}] {surface.name} 没有入射波前数据")
        
        return surface.entrance
    
    def get_exit_wavefront(self, surface_index: int) -> WavefrontData:
        """获取指定表面的出射波前数据
        
        参数:
            surface_index: 表面索引
        
        返回:
            指定表面的出射 WavefrontData 对象
        
        异常:
            KeyError: 未找到指定索引的表面
            ValueError: 指定表面没有出射波前数据
        """
        surface = self.get_surface(surface_index)
        
        if surface.exit is None:
            raise ValueError(f"表面 [{surface.index}] {surface.name} 没有出射波前数据")
        
        return surface.exit

    # ========================================================================
    # 调试数据读取接口（用于 OAP 混合光线追迹调试）
    # ========================================================================
    
    def get_surface_rays(
        self,
        surface_index: int,
        location: str = "exit",
    ) -> RayData:
        """获取指定表面的光线数据
        
        参数:
            surface_index: 表面索引
            location: "entrance"（入射面）或 "exit"（出射面）
        
        返回:
            RayData 对象，包含光线位置、方向、OPD 等
        
        异常:
            KeyError: 未找到指定表面
            ValueError: 指定位置没有光线数据
        
        **Validates: Requirements 12.1, 12.2**
        """
        surface = self.get_surface(surface_index)
        
        if location == "entrance":
            if surface.entrance_rays is None:
                raise ValueError(
                    f"表面 [{surface.index}] {surface.name} 没有入射面光线数据。"
                    "请确保仿真时启用了光线数据记录。"
                )
            return surface.entrance_rays
        elif location == "exit":
            if surface.exit_rays is None:
                raise ValueError(
                    f"表面 [{surface.index}] {surface.name} 没有出射面光线数据。"
                    "请确保仿真时启用了光线数据记录。"
                )
            return surface.exit_rays
        else:
            raise ValueError(f"无效的 location 参数: {location}，必须是 'entrance' 或 'exit'")
    
    def get_chief_ray(self, surface_index: int) -> ChiefRayData:
        """获取指定表面的主光线数据
        
        参数:
            surface_index: 表面索引
        
        返回:
            ChiefRayData 对象，包含入射/出射方向、交点位置等
        
        异常:
            KeyError: 未找到指定表面
            ValueError: 没有主光线数据
        
        **Validates: Requirements 12.1, 12.2**
        """
        surface = self.get_surface(surface_index)
        
        if surface.chief_ray is None:
            raise ValueError(
                f"表面 [{surface.index}] {surface.name} 没有主光线数据。"
                "请确保仿真时启用了光线数据记录。"
            )
        
        return surface.chief_ray
    
    def get_pilot_beam_params(
        self,
        surface_index: int,
        location: str = "exit",
    ) -> PilotBeamParamsData:
        """获取指定表面的 Pilot Beam 参数
        
        参数:
            surface_index: 表面索引
            location: "entrance" 或 "exit"
        
        返回:
            PilotBeamParamsData 对象
        
        异常:
            KeyError: 未找到指定表面
            ValueError: 指定位置没有波前数据
        
        **Validates: Requirements 12.1, 12.2**
        """
        surface = self.get_surface(surface_index)
        
        if location == "entrance":
            if surface.entrance is None:
                raise ValueError(
                    f"表面 [{surface.index}] {surface.name} 没有入射面波前数据"
                )
            pilot_beam = surface.entrance.pilot_beam
        elif location == "exit":
            if surface.exit is None:
                raise ValueError(
                    f"表面 [{surface.index}] {surface.name} 没有出射面波前数据"
                )
            pilot_beam = surface.exit.pilot_beam
        else:
            raise ValueError(f"无效的 location 参数: {location}，必须是 'entrance' 或 'exit'")
        
        return PilotBeamParamsData.from_pilot_beam_params(pilot_beam)
    
    def get_coordinate_system(
        self,
        surface_index: int,
        location: str = "exit",
    ) -> CoordinateSystemData:
        """获取指定表面的坐标系信息
        
        参数:
            surface_index: 表面索引
            location: "entrance" 或 "exit"
        
        返回:
            CoordinateSystemData 对象，包含原点位置、旋转矩阵等
        
        异常:
            KeyError: 未找到指定表面
            ValueError: 指定位置没有坐标系数据
        
        **Validates: Requirements 12.1, 12.2**
        """
        surface = self.get_surface(surface_index)
        
        if location == "entrance":
            if surface.entrance_coord_system is None:
                raise ValueError(
                    f"表面 [{surface.index}] {surface.name} 没有入射面坐标系数据。"
                    "请确保仿真时启用了光线数据记录。"
                )
            return surface.entrance_coord_system
        elif location == "exit":
            if surface.exit_coord_system is None:
                raise ValueError(
                    f"表面 [{surface.index}] {surface.name} 没有出射面坐标系数据。"
                    "请确保仿真时启用了光线数据记录。"
                )
            return surface.exit_coord_system
        else:
            raise ValueError(f"无效的 location 参数: {location}，必须是 'entrance' 或 'exit'")
