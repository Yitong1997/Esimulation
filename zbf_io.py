"""ZBF (Zemax Beam File) 二进制格式读写模块。

ZBF 文件存储光束在某一面上的完整复数电场分布，
包含振幅和相位信息，是 Zemax POP 模块的核心数据交换格式。

二进制格式结构（官方文档）：
  整数部分（4字节 int32 × 9）：
    [0] version（当前为 1）
    [1] nx — X 方向采样点数
    [2] ny — Y 方向采样点数
    [3] is_polarized（0 或 1）
    [4] units（0=mm, 1=cm, 2=in, 3=m）
    [5..8] unused（4 个保留整数）

  双精度部分（8字节 double × 20）：
    [0] dx — X 方向网格间距
    [1] dy — Y 方向网格间距
    [2] zx — 相对于导引光束腰斑的 z 位置，X 方向
    [3] Rx — 导引光束瑞利距离，X 方向
    [4] wx — 导引光束腰斑半径，X 方向
    [5] zy — 相对于导引光束腰斑的 z 位置，Y 方向
    [6] Ry — 导引光束瑞利距离，Y 方向
    [7] wy — 导引光束腰斑半径，Y 方向
    [8] wavelength — 当前介质中的波长（透镜单位）
    [9] index — 当前介质折射率
    [10] receiver_efficiency（未计算光纤耦合时为 0）
    [11] system_efficiency（未计算光纤耦合时为 0）
    [12..19] unused（8 个保留 double）

  电场数据：
    2*nx*ny double: Ex 值（实部虚部交错：re1,im1,re2,im2,...）
    若偏振，再跟 2*nx*ny double: Ey 值

  Ex*Ex + Ey*Ey 的单位为瓦特（W）。
"""

import struct
import numpy as np
from pathlib import Path

INT_SIZE = 4
DBL_SIZE = 8
N_HEADER_INTS = 9    # 5 有效 + 4 保留
N_HEADER_DBLS = 20   # 12 有效 + 8 保留
HEADER_BYTES = N_HEADER_INTS * INT_SIZE + N_HEADER_DBLS * DBL_SIZE  # 196


class ZBFData:
    """ZBF 文件数据容器。

    属性（与官方文档字段一一对应）：
        version: 文件版本号（当前为 1）
        nx, ny: X/Y 方向采样点数
        is_polarized: 是否包含偏振信息
        units: 长度单位 (0=mm, 1=cm, 2=in, 3=m)
        dx, dy: X/Y 方向网格间距（透镜单位）
        zx, zy: 相对于导引光束腰斑的 z 位置
        Rx, Ry: 导引光束瑞利距离
        wx, wy: 导引光束腰斑半径
        wavelength: 当前介质中的波长（透镜单位）
        index: 当前介质折射率
        Ex: 复数电场 X 分量 (ny, nx)
        Ey: 复数电场 Y 分量 (ny, nx)，无偏振时为 None
    """

    def __init__(self):
        self.version = 1
        self.nx = 0
        self.ny = 0
        self.is_polarized = 0
        self.units = 0  # mm
        self.dx = 0.0
        self.dy = 0.0
        self.zx = 0.0
        self.Rx = 0.0
        self.wx = 0.0
        self.zy = 0.0
        self.Ry = 0.0
        self.wy = 0.0
        self.wavelength = 0.0
        self.index = 1.0
        self.receiver_eff = 0.0
        self.system_eff = 0.0
        self.Ex = None   # complex128, shape (ny, nx)
        self.Ey = None   # complex128, shape (ny, nx) 或 None
        self._reserved_ints = (0, 0, 0, 0)
        self._reserved_dbls = (0.0,) * 8

    @property
    def x_coords(self):
        """返回 X 坐标数组（透镜单位，以光束中心为原点）。"""
        return (np.arange(self.nx) - self.nx / 2.0 + 0.5) * self.dx

    @property
    def y_coords(self):
        """返回 Y 坐标数组（透镜单位，以光束中心为原点）。"""
        return (np.arange(self.ny) - self.ny / 2.0 + 0.5) * self.dy

    @property
    def x_width(self):
        """X 方向物理总宽度。"""
        return self.nx * self.dx

    @property
    def y_width(self):
        """Y 方向物理总宽度。"""
        return self.ny * self.dy

    @property
    def amplitude(self):
        """返回总振幅 |E| = sqrt(|Ex|^2 + |Ey|^2)。"""
        amp_sq = np.abs(self.Ex) ** 2
        if self.Ey is not None:
            amp_sq += np.abs(self.Ey) ** 2
        return np.sqrt(amp_sq)

    @property
    def phase(self):
        """返回 Ex 分量的相位（radians）。"""
        return np.angle(self.Ex)

    @property
    def irradiance(self):
        """返回辐照度 |Ex|^2 + |Ey|^2（单位：W）。"""
        irr = np.abs(self.Ex) ** 2
        if self.Ey is not None:
            irr += np.abs(self.Ey) ** 2
        return irr

    def copy(self):
        """深拷贝当前 ZBF 数据。"""
        new = ZBFData()
        for attr in ['version', 'nx', 'ny', 'is_polarized', 'units',
                      'dx', 'dy', 'zx', 'Rx', 'wx', 'zy', 'Ry', 'wy',
                      'wavelength', 'index', 'receiver_eff', 'system_eff',
                      '_reserved_ints', '_reserved_dbls']:
            setattr(new, attr, getattr(self, attr))
        new.Ex = self.Ex.copy() if self.Ex is not None else None
        new.Ey = self.Ey.copy() if self.Ey is not None else None
        return new


def read_zbf(filepath):
    """读取 ZBF 二进制文件。

    参数：
        filepath: ZBF 文件路径（str 或 Path）

    返回：
        ZBFData 对象

    异常：
        FileNotFoundError: 文件不存在
        ValueError: 文件格式不正确
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"ZBF 文件不存在：{filepath}")

    with open(filepath, "rb") as f:
        data = ZBFData()

        # 读取 9 个整数
        raw = f.read(INT_SIZE * N_HEADER_INTS)
        if len(raw) < INT_SIZE * N_HEADER_INTS:
            raise ValueError("ZBF 头部不完整（整数部分）")
        ints = struct.unpack(f"<{N_HEADER_INTS}i", raw)
        data.version = ints[0]
        data.nx = ints[1]
        data.ny = ints[2]
        data.is_polarized = ints[3]
        data.units = ints[4]
        # ints[5..8] 保留
        data._reserved_ints = ints[5:9]

        if data.nx <= 0 or data.ny <= 0:
            raise ValueError(
                f"ZBF 网格尺寸无效：nx={data.nx}, ny={data.ny}")

        # 读取 20 个双精度
        raw = f.read(DBL_SIZE * N_HEADER_DBLS)
        if len(raw) < DBL_SIZE * N_HEADER_DBLS:
            raise ValueError("ZBF 头部不完整（双精度部分）")
        dbls = struct.unpack(f"<{N_HEADER_DBLS}d", raw)
        data.dx = dbls[0]
        data.dy = dbls[1]
        data.zx = dbls[2]
        data.Rx = dbls[3]
        data.wx = dbls[4]
        data.zy = dbls[5]
        data.Ry = dbls[6]
        data.wy = dbls[7]
        data.wavelength = dbls[8]
        data.index = dbls[9]
        data.receiver_eff = dbls[10]
        data.system_eff = dbls[11]
        # dbls[12..19] 保留
        data._reserved_dbls = dbls[12:20]

        # 读取 Ex 电场数据
        # 官方格式：2*nx*ny double，实部虚部交错存储
        # (re1, im1, re2, im2, ...) 共 nx*ny 个复数点
        n_pixels = data.nx * data.ny
        ex_raw = f.read(DBL_SIZE * 2 * n_pixels)
        if len(ex_raw) < DBL_SIZE * 2 * n_pixels:
            raise ValueError("ZBF 文件 Ex 数据不完整")
        ex_pairs = np.frombuffer(ex_raw, dtype="<f8")
        # 交错排列：[re0, im0, re1, im1, ...]
        ex_real = ex_pairs[0::2]  # 偶数索引 = 实部
        ex_imag = ex_pairs[1::2]  # 奇数索引 = 虚部
        data.Ex = (ex_real + 1j * ex_imag).reshape(
            data.ny, data.nx)

        # 若有偏振，读取 Ey
        if data.is_polarized:
            ey_raw = f.read(DBL_SIZE * 2 * n_pixels)
            if len(ey_raw) < DBL_SIZE * 2 * n_pixels:
                raise ValueError("ZBF 文件 Ey 数据不完整")
            ey_pairs = np.frombuffer(ey_raw, dtype="<f8")
            ey_real = ey_pairs[0::2]
            ey_imag = ey_pairs[1::2]
            data.Ey = (ey_real + 1j * ey_imag).reshape(
                data.ny, data.nx)
        else:
            data.Ey = None

    return data


def write_zbf(filepath, data):
    """将 ZBFData 写入 ZBF 二进制文件。"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if data.Ex is None:
        raise ValueError("ZBFData.Ex 不能为 None")
    if data.Ex.shape != (data.ny, data.nx):
        raise ValueError(
            f"Ex 形状 {data.Ex.shape} 与 "
            f"(ny={data.ny}, nx={data.nx}) 不匹配")

    with open(filepath, "wb") as f:
        # 写入 9 个整数
        f.write(struct.pack(
            f"<{N_HEADER_INTS}i",
            data.version, data.nx, data.ny,
            data.is_polarized, data.units,
            *data._reserved_ints,  # 4 个保留整数
        ))

        # 写入 20 个双精度
        dbls = [
            data.dx, data.dy,
            data.zx, data.Rx, data.wx,
            data.zy, data.Ry, data.wy,
            data.wavelength, data.index,
            data.receiver_eff, data.system_eff,
            *data._reserved_dbls,  # 8 个保留 double（保持原值）
        ]
        f.write(struct.pack(f"<{N_HEADER_DBLS}d", *dbls))

        # 写入 Ex 数据（交错格式：re1,im1,re2,im2,...）
        ex_flat = data.Ex.flatten()
        ex_interleaved = np.empty(2 * len(ex_flat), dtype="<f8")
        ex_interleaved[0::2] = np.real(ex_flat)
        ex_interleaved[1::2] = np.imag(ex_flat)
        f.write(ex_interleaved.tobytes())

        # 若有偏振，写入 Ey
        if data.is_polarized and data.Ey is not None:
            if data.Ey.shape != (data.ny, data.nx):
                raise ValueError(
                    f"Ey 形状 {data.Ey.shape} 与 "
                    f"(ny={data.ny}, nx={data.nx}) 不匹配")
            ey_flat = data.Ey.flatten()
            ey_interleaved = np.empty(
                2 * len(ey_flat), dtype="<f8")
            ey_interleaved[0::2] = np.real(ey_flat)
            ey_interleaved[1::2] = np.imag(ey_flat)
            f.write(ey_interleaved.tobytes())


def replace_amplitude(zbf_data, new_amplitude):
    """替换 ZBF 数据中的振幅，保留原始相位。

    GS 算法核心操作：保留传播相位，替换振幅为目标分布。

    参数：
        zbf_data: 原始 ZBFData 对象
        new_amplitude: 新的振幅分布 (ny, nx)，实数数组

    返回：
        修改后的 ZBFData 深拷贝
    """
    result = zbf_data.copy()
    phase_ex = np.angle(zbf_data.Ex)
    result.Ex = new_amplitude * np.exp(1j * phase_ex)

    # 偏振分量按相同比例缩放
    if zbf_data.Ey is not None and zbf_data.is_polarized:
        old_amp = np.abs(zbf_data.Ex)
        safe_old = np.where(old_amp > 0, old_amp, 1.0)
        scale = np.where(old_amp > 0, new_amplitude / safe_old, 0.0)
        result.Ey = zbf_data.Ey * scale

    return result


def zbf_to_extent_info(zbf_data):
    """将 ZBFData 坐标转换为 ao_core 兼容的 extent_info 字典。"""
    return {
        "x": zbf_data.x_coords,
        "y": zbf_data.y_coords,
        "x_width": zbf_data.x_width,
        "y_width": zbf_data.y_width,
    }


def print_zbf_info(zbf_data, label=""):
    """打印 ZBF 数据摘要信息（调试用）。"""
    prefix = f"[{label}] " if label else ""
    units_map = {0: "mm", 1: "cm", 2: "in", 3: "m"}
    u = units_map.get(zbf_data.units, "?")
    print(f"{prefix}ZBF 信息：", flush=True)
    print(f"  版本={zbf_data.version}，"
          f"网格={zbf_data.nx}×{zbf_data.ny}，"
          f"偏振={bool(zbf_data.is_polarized)}", flush=True)
    print(f"  dx={zbf_data.dx:.6e} {u}，"
          f"dy={zbf_data.dy:.6e} {u}", flush=True)
    print(f"  物理范围：{zbf_data.x_width:.6e} × "
          f"{zbf_data.y_width:.6e} {u}", flush=True)
    print(f"  波长={zbf_data.wavelength:.6e} {u}，"
          f"折射率={zbf_data.index:.4f}", flush=True)
    print(f"  导引光束：wx={zbf_data.wx:.6e}，"
          f"wy={zbf_data.wy:.6e}", flush=True)
    if zbf_data.Ex is not None:
        total_power = np.sum(zbf_data.irradiance)
        peak_irr = np.max(zbf_data.irradiance)
        print(f"  总功率={total_power:.6e} W，"
              f"峰值辐照度={peak_irr:.6e} W", flush=True)
