"""
序列化模块

提供仿真结果的保存和加载功能。
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .result import SimulationResult, SurfaceRecord, WavefrontData


def save_result(result: "SimulationResult", path: str) -> None:
    """保存仿真结果到目录
    
    目录结构：
        path/
        ├── config.json
        ├── source.json
        ├── summary.json
        └── surfaces/
            ├── 0_Initial/
            │   ├── entrance_amplitude.npy
            │   ├── entrance_phase.npy
            │   └── entrance_info.json
            └── ...
    
    参数:
        result: 仿真结果
        path: 保存目录路径
    """
    os.makedirs(path, exist_ok=True)
    
    # 保存配置
    config_data = {
        'wavelength_um': result.config.wavelength_um,
        'grid_size': result.config.grid_size,
        'physical_size_mm': result.config.physical_size_mm,
        'num_rays': result.config.num_rays,
        'propagation_method': result.config.propagation_method,
    }
    with open(os.path.join(path, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    # 保存光源参数
    source_data = {
        'wavelength_um': result.source_params.wavelength_um,
        'w0_mm': result.source_params.w0_mm,
        'z0_mm': result.source_params.z0_mm,
        'z_rayleigh_mm': result.source_params.z_rayleigh_mm,
        'grid_size': result.source_params.grid_size,
        'physical_size_mm': result.source_params.physical_size_mm,
    }
    with open(os.path.join(path, 'source.json'), 'w', encoding='utf-8') as f:
        json.dump(source_data, f, indent=2, ensure_ascii=False)
    
    # 保存摘要
    summary_data = {
        'success': result.success,
        'error_message': result.error_message,
        'total_path_length': result.total_path_length,
        'num_surfaces': len(result.surfaces),
    }
    with open(os.path.join(path, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    # 保存表面数据
    surfaces_dir = os.path.join(path, 'surfaces')
    os.makedirs(surfaces_dir, exist_ok=True)
    
    for surface in result.surfaces:
        _save_surface(surface, surfaces_dir)


def _save_surface(surface: "SurfaceRecord", surfaces_dir: str) -> None:
    """保存单个表面数据
    
    参数:
        surface: 表面记录
        surfaces_dir: 表面数据目录
    """
    surface_dir = os.path.join(surfaces_dir, f"{surface.index}_{surface.name}")
    os.makedirs(surface_dir, exist_ok=True)
    
    # 保存表面信息
    info = {
        'index': surface.index,
        'name': surface.name,
        'surface_type': surface.surface_type,
    }
    
    # 几何信息
    if surface.geometry is not None:
        info['geometry'] = {
            'vertex_position': surface.geometry.vertex_position.tolist(),
            'surface_normal': surface.geometry.surface_normal.tolist(),
            'radius': float(surface.geometry.radius) if not np.isinf(surface.geometry.radius) else 'inf',
            'conic': surface.geometry.conic,
            'semi_aperture': surface.geometry.semi_aperture,
            'is_mirror': surface.geometry.is_mirror,
        }
    
    # 光轴信息
    if surface.optical_axis is not None:
        info['optical_axis'] = {
            'entrance_position': surface.optical_axis.entrance_position.tolist(),
            'entrance_direction': surface.optical_axis.entrance_direction.tolist(),
            'exit_position': surface.optical_axis.exit_position.tolist(),
            'exit_direction': surface.optical_axis.exit_direction.tolist(),
            'path_length': surface.optical_axis.path_length,
        }
    
    with open(os.path.join(surface_dir, 'info.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    # 保存入射面数据
    if surface.entrance is not None:
        _save_wavefront(surface.entrance, surface_dir, 'entrance')
    
    # 保存出射面数据
    if surface.exit is not None:
        _save_wavefront(surface.exit, surface_dir, 'exit')


def _save_wavefront(wavefront: "WavefrontData", surface_dir: str, prefix: str) -> None:
    """保存波前数据
    
    参数:
        wavefront: 波前数据
        surface_dir: 表面目录
        prefix: 文件前缀（'entrance' 或 'exit'）
    """
    # 保存振幅和相位
    np.save(os.path.join(surface_dir, f'{prefix}_amplitude.npy'), wavefront.amplitude)
    np.save(os.path.join(surface_dir, f'{prefix}_phase.npy'), wavefront.phase)
    
    # 保存 Pilot Beam 参数
    pilot_data = {
        'wavelength_um': wavefront.pilot_beam.wavelength_um,
        'waist_radius_mm': wavefront.pilot_beam.waist_radius_mm,
        'waist_position_mm': wavefront.pilot_beam.waist_position_mm,
        'curvature_radius_mm': float(wavefront.pilot_beam.curvature_radius_mm) 
            if not np.isinf(wavefront.pilot_beam.curvature_radius_mm) else 'inf',
        'spot_size_mm': wavefront.pilot_beam.spot_size_mm,
    }
    with open(os.path.join(surface_dir, f'{prefix}_pilot_beam.json'), 'w', encoding='utf-8') as f:
        json.dump(pilot_data, f, indent=2, ensure_ascii=False)
    
    # 保存网格信息
    grid_data = {
        'grid_size': wavefront.grid.grid_size,
        'physical_size_mm': wavefront.grid.physical_size_mm,
        'sampling_mm': wavefront.grid.sampling_mm,
    }
    with open(os.path.join(surface_dir, f'{prefix}_grid.json'), 'w', encoding='utf-8') as f:
        json.dump(grid_data, f, indent=2, ensure_ascii=False)


def load_result(path: str) -> "SimulationResult":
    """从目录加载仿真结果
    
    参数:
        path: 目录路径
    
    返回:
        SimulationResult 对象
    """
    from .data_models import SimulationConfig, SourceParams, SurfaceGeometry, OpticalAxisInfo
    from .result import SimulationResult, SurfaceRecord, WavefrontData
    from hybrid_optical_propagation import PilotBeamParams, GridSampling
    
    # 加载配置
    with open(os.path.join(path, 'config.json'), 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    
    config = SimulationConfig(
        wavelength_um=config_data['wavelength_um'],
        grid_size=config_data['grid_size'],
        physical_size_mm=config_data['physical_size_mm'],
        num_rays=config_data.get('num_rays', 200),
        propagation_method=config_data.get('propagation_method', 'local_raytracing'),
    )
    
    # 加载光源参数
    with open(os.path.join(path, 'source.json'), 'r', encoding='utf-8') as f:
        source_data = json.load(f)
    
    source_params = SourceParams(
        wavelength_um=source_data['wavelength_um'],
        w0_mm=source_data['w0_mm'],
        z0_mm=source_data['z0_mm'],
        z_rayleigh_mm=source_data['z_rayleigh_mm'],
        grid_size=source_data['grid_size'],
        physical_size_mm=source_data['physical_size_mm'],
    )
    
    # 加载摘要
    with open(os.path.join(path, 'summary.json'), 'r', encoding='utf-8') as f:
        summary_data = json.load(f)
    
    # 加载表面数据
    surfaces_dir = os.path.join(path, 'surfaces')
    surfaces = []
    
    if os.path.exists(surfaces_dir):
        for surface_name in sorted(os.listdir(surfaces_dir)):
            surface_path = os.path.join(surfaces_dir, surface_name)
            if os.path.isdir(surface_path):
                surface = _load_surface(surface_path, config.wavelength_um)
                surfaces.append(surface)
    
    return SimulationResult(
        success=summary_data['success'],
        error_message=summary_data.get('error_message', ''),
        config=config,
        source_params=source_params,
        surfaces=surfaces,
        total_path_length=summary_data['total_path_length'],
    )


def _load_surface(surface_path: str, wavelength_um: float) -> "SurfaceRecord":
    """加载单个表面数据
    
    参数:
        surface_path: 表面目录路径
        wavelength_um: 波长 (μm)
    
    返回:
        SurfaceRecord 对象
    """
    from .data_models import SurfaceGeometry, OpticalAxisInfo
    from .result import SurfaceRecord, WavefrontData
    
    # 加载表面信息
    with open(os.path.join(surface_path, 'info.json'), 'r', encoding='utf-8') as f:
        info = json.load(f)
    
    # 几何信息
    geometry = None
    if 'geometry' in info:
        g = info['geometry']
        radius = np.inf if g['radius'] == 'inf' else g['radius']
        geometry = SurfaceGeometry(
            vertex_position=np.array(g['vertex_position']),
            surface_normal=np.array(g['surface_normal']),
            radius=radius,
            conic=g['conic'],
            semi_aperture=g['semi_aperture'],
            is_mirror=g['is_mirror'],
        )
    
    # 光轴信息
    optical_axis = None
    if 'optical_axis' in info:
        oa = info['optical_axis']
        optical_axis = OpticalAxisInfo(
            entrance_position=np.array(oa['entrance_position']),
            entrance_direction=np.array(oa['entrance_direction']),
            exit_position=np.array(oa['exit_position']),
            exit_direction=np.array(oa['exit_direction']),
            path_length=oa['path_length'],
        )
    
    # 加载波前数据
    entrance = _load_wavefront(surface_path, 'entrance', wavelength_um)
    exit_data = _load_wavefront(surface_path, 'exit', wavelength_um)
    
    return SurfaceRecord(
        index=info['index'],
        name=info['name'],
        surface_type=info['surface_type'],
        geometry=geometry,
        entrance=entrance,
        exit=exit_data,
        optical_axis=optical_axis,
    )


def _load_wavefront(surface_path: str, prefix: str, wavelength_um: float) -> "WavefrontData":
    """加载波前数据
    
    参数:
        surface_path: 表面目录路径
        prefix: 文件前缀
        wavelength_um: 波长 (μm)
    
    返回:
        WavefrontData 对象或 None
    """
    from .result import WavefrontData
    from hybrid_optical_propagation import PilotBeamParams, GridSampling
    
    amp_path = os.path.join(surface_path, f'{prefix}_amplitude.npy')
    if not os.path.exists(amp_path):
        return None
    
    # 加载振幅和相位
    amplitude = np.load(amp_path)
    phase = np.load(os.path.join(surface_path, f'{prefix}_phase.npy'))
    
    # 加载 Pilot Beam 参数
    with open(os.path.join(surface_path, f'{prefix}_pilot_beam.json'), 'r', encoding='utf-8') as f:
        pilot_data = json.load(f)
    
    curvature = np.inf if pilot_data['curvature_radius_mm'] == 'inf' else pilot_data['curvature_radius_mm']
    
    pilot_beam = PilotBeamParams.from_gaussian_source(
        wavelength_um=pilot_data['wavelength_um'],
        w0_mm=pilot_data['waist_radius_mm'],
        z0_mm=pilot_data['waist_position_mm'],
    )
    # 覆盖曲率半径（可能与计算值不同）
    pilot_beam = PilotBeamParams(
        wavelength_um=pilot_data['wavelength_um'],
        waist_radius_mm=pilot_data['waist_radius_mm'],
        waist_position_mm=pilot_data['waist_position_mm'],
        curvature_radius_mm=curvature,
        spot_size_mm=pilot_data['spot_size_mm'],
        q_parameter=pilot_beam.q_parameter,
    )
    
    # 加载网格信息
    with open(os.path.join(surface_path, f'{prefix}_grid.json'), 'r', encoding='utf-8') as f:
        grid_data = json.load(f)
    
    grid = GridSampling.create(
        grid_size=grid_data['grid_size'],
        physical_size_mm=grid_data['physical_size_mm'],
    )
    
    return WavefrontData(
        amplitude=amplitude,
        phase=phase,
        pilot_beam=pilot_beam,
        grid=grid,
        wavelength_um=wavelength_um,
    )
