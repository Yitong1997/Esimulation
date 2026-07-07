"""CSV 到 ZBF 转换器的属性测试和单元测试。

使用 pytest + hypothesis 框架验证 csv_to_zbf 模块的正确性属性。
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from csv_to_zbf import read_beam_csv, detect_aperture, match_apertures, convert_csv_to_zbf
from zbf_io import read_zbf

# ---------- 策略定义 ----------

# 二维 float64 数组策略（通用，无 NaN/Inf）
float_2d_strategy = arrays(
    dtype=np.float64,
    shape=st.tuples(
        st.integers(min_value=3, max_value=64),
        st.integers(min_value=3, max_value=64),
    ),
    elements=st.floats(
        min_value=-100, max_value=100,
        allow_nan=False, allow_infinity=False,
    ),
)


# ---------- Property 1：CSV 读写往返 ----------
# Feature: csv-to-zbf-converter, Property 1: CSV 读写往返
# Validates: Requirements 1.1, 1.2


@given(data=float_2d_strategy)
@settings(max_examples=100)
def test_csv_roundtrip_no_header(data):
    """无表头 CSV 的读写往返：写入后读回应与原始数据数值等价。"""
    with tempfile.NamedTemporaryFile(
        suffix=".csv", delete=False, mode="w"
    ) as f:
        tmp_path = f.name
        # 使用 np.savetxt 写入 CSV（无表头）
        np.savetxt(f, data, delimiter=",")

    try:
        # 用 read_beam_csv 读回
        result = read_beam_csv(tmp_path)
        # 验证形状一致
        assert result.shape == data.shape, (
            f"形状不匹配：期望 {data.shape}，实际 {result.shape}"
        )
        # 验证数值近似相等
        assert np.allclose(result, data), (
            f"数值不匹配：最大差异 {np.max(np.abs(result - data))}"
        )
    finally:
        os.unlink(tmp_path)


@given(data=float_2d_strategy)
@settings(max_examples=100)
def test_csv_roundtrip_with_header(data):
    """有表头 CSV 的读写往返：写入后读回应与原始数据数值等价。"""
    with tempfile.NamedTemporaryFile(
        suffix=".csv", delete=False, mode="w"
    ) as f:
        tmp_path = f.name
        # 写入表头行
        ncols = data.shape[1]
        header_line = ",".join(
            [f"col_{i}" for i in range(ncols)]
        )
        f.write(header_line + "\n")
        # 写入数据行
        np.savetxt(f, data, delimiter=",")

    try:
        # 用 read_beam_csv 读回（应自动跳过表头）
        result = read_beam_csv(tmp_path)
        # 验证形状一致
        assert result.shape == data.shape, (
            f"形状不匹配：期望 {data.shape}，"
            f"实际 {result.shape}"
        )
        # 验证数值近似相等
        assert np.allclose(result, data), (
            f"数值不匹配：最大差异 "
            f"{np.max(np.abs(result - data))}"
        )
    finally:
        os.unlink(tmp_path)


# ---------- Property 2：非数值 CSV 内容触发 ValueError ----------
# Feature: csv-to-zbf-converter, Property 2: 非数值 CSV 内容触发 ValueError
# Validates: Requirements 1.5


# 非数值行策略：生成至少包含一个字母字符的字符串
# 确保内容确实无法被解析为纯数值
_non_numeric_text = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz,\n "),
    min_size=1,
    max_size=200,
).filter(lambda s: any(c.isalpha() for c in s))


def _build_non_numeric_csv(lines_text: str) -> str:
    """构建一个至少有两行且每行都包含非数值字符的 CSV 内容。

    这样即使 read_beam_csv 跳过第一行表头，
    剩余行仍然无法被 np.loadtxt 解析为数值。
    """
    # 确保至少有两行非数值内容
    # 第一行可能被当作表头跳过，第二行仍然是非数值
    return lines_text + "\nabc,def\nxyz,uvw\n"


@given(text_content=_non_numeric_text)
@settings(max_examples=100)
def test_non_numeric_csv_raises_valueerror(text_content):
    """非数值 CSV 内容应触发 ValueError。

    对任意包含非数值字符串的文件内容，
    调用 read_beam_csv 应抛出 ValueError。
    """
    # 构建确保多行非数值的 CSV 内容
    csv_content = _build_non_numeric_csv(text_content)

    with tempfile.NamedTemporaryFile(
        suffix=".csv", delete=False, mode="w"
    ) as f:
        tmp_path = f.name
        f.write(csv_content)

    try:
        with pytest.raises(ValueError):
            read_beam_csv(tmp_path)
    finally:
        os.unlink(tmp_path)


# ---------- 单元测试：read_beam_csv ----------
# 验证：需求 1.3, 1.4


def test_read_csv_file_not_found():
    """文件不存在时应抛出 FileNotFoundError。

    验证：需求 1.4
    """
    with pytest.raises(FileNotFoundError):
        read_beam_csv("/nonexistent/path/to/file.csv")


def test_read_csv_with_header_skip():
    """含表头 CSV 应正确跳过表头并返回正确数据。

    验证：需求 1.3 边界案例
    """
    # 构造含表头的 CSV 内容
    expected = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])
    with tempfile.NamedTemporaryFile(
        suffix=".csv", delete=False, mode="w"
    ) as f:
        tmp_path = f.name
        f.write("col_a,col_b,col_c\n")
        np.savetxt(f, expected, delimiter=",")

    try:
        result = read_beam_csv(tmp_path)
        assert result.shape == expected.shape
        assert np.allclose(result, expected)
        assert result.dtype == np.float64
    finally:
        os.unlink(tmp_path)


def test_read_csv_all_zeros():
    """全零数组应能正确读取。"""
    expected = np.zeros((5, 5), dtype=np.float64)
    with tempfile.NamedTemporaryFile(
        suffix=".csv", delete=False, mode="w"
    ) as f:
        tmp_path = f.name
        np.savetxt(f, expected, delimiter=",")

    try:
        result = read_beam_csv(tmp_path)
        assert result.shape == expected.shape
        assert np.array_equal(result, expected)
        assert result.dtype == np.float64
    finally:
        os.unlink(tmp_path)


# ---------- 策略定义（孔径检测） ----------

# 正数阈值策略
threshold_strategy = st.floats(
    min_value=1e-10, max_value=1.0,
    allow_nan=False, allow_infinity=False,
)


# ---------- Property 4：孔径检测掩膜正确性 ----------
# Feature: csv-to-zbf-converter, Property 4: 孔径检测掩膜正确性
# Validates: Requirements 3.1, 3.2


@given(data=float_2d_strategy, threshold=threshold_strategy)
@settings(max_examples=100)
def test_detect_aperture_mask_correctness(data, threshold):
    """孔径检测掩膜正确性：True 像素绝对值 > 阈值，False 像素绝对值 <= 阈值。

    对任意二维浮点数组和正数阈值，detect_aperture 返回的布尔掩膜中，
    所有 True 像素的绝对值应大于阈值，
    所有 False 像素的绝对值应小于或等于阈值。
    """
    mask = detect_aperture(data, threshold)

    # 验证掩膜形状与输入一致
    assert mask.shape == data.shape, (
        f"掩膜形状不匹配：期望 {data.shape}，实际 {mask.shape}"
    )
    assert mask.dtype == bool, (
        f"掩膜类型应为 bool，实际为 {mask.dtype}"
    )

    abs_data = np.abs(data)

    # 所有 True 位置的绝对值应大于阈值
    if np.any(mask):
        assert np.all(abs_data[mask] > threshold), (
            "掩膜中存在 True 像素的绝对值 <= 阈值"
        )

    # 所有 False 位置的绝对值应小于或等于阈值
    if np.any(~mask):
        assert np.all(abs_data[~mask] <= threshold), (
            "掩膜中存在 False 像素的绝对值 > 阈值"
        )


# ---------- 策略定义（孔径匹配） ----------

# 同形状双数组策略
@st.composite
def same_shape_arrays(draw):
    """生成两个同形状的二维 float64 数组。"""
    shape = draw(st.tuples(
        st.integers(min_value=3, max_value=64),
        st.integers(min_value=3, max_value=64),
    ))
    arr1 = draw(arrays(
        dtype=np.float64, shape=shape,
        elements=st.floats(
            min_value=-100, max_value=100,
            allow_nan=False, allow_infinity=False,
        ),
    ))
    arr2 = draw(arrays(
        dtype=np.float64, shape=shape,
        elements=st.floats(
            min_value=-100, max_value=100,
            allow_nan=False, allow_infinity=False,
        ),
    ))
    return arr1, arr2


# 不同形状双数组策略
@st.composite
def different_shape_arrays(draw):
    """生成两个不同形状的二维 float64 数组。"""
    shape1 = draw(st.tuples(
        st.integers(min_value=3, max_value=64),
        st.integers(min_value=3, max_value=64),
    ))
    shape2 = draw(st.tuples(
        st.integers(min_value=3, max_value=64),
        st.integers(min_value=3, max_value=64),
    ).filter(lambda s: s != shape1))
    arr1 = draw(arrays(
        dtype=np.float64, shape=shape1,
        elements=st.floats(
            min_value=-100, max_value=100,
            allow_nan=False, allow_infinity=False,
        ),
    ))
    arr2 = draw(arrays(
        dtype=np.float64, shape=shape2,
        elements=st.floats(
            min_value=-100, max_value=100,
            allow_nan=False, allow_infinity=False,
        ),
    ))
    return arr1, arr2


# ---------- Property 5：同尺寸孔径交集 ----------
# Feature: csv-to-zbf-converter, Property 5: 同尺寸孔径交集
# Validates: Requirements 4.1


@given(pair=same_shape_arrays())
@settings(max_examples=100)
def test_same_size_aperture_intersection(pair):
    """同尺寸孔径交集：统一掩膜应等于两者各自掩膜的逻辑与。

    对任意两个同尺寸二维数组，match_apertures 返回的
    统一掩膜应等于分别对两者调用 detect_aperture 后
    取逻辑与的结果。
    """
    irradiance, phase = pair

    # 调用 match_apertures
    _, _, unified_mask = match_apertures(irradiance, phase)

    # 分别计算各自的孔径掩膜
    irr_mask = detect_aperture(irradiance)
    phase_mask = detect_aperture(phase)
    expected_mask = irr_mask & phase_mask

    # 验证统一掩膜等于逻辑与
    assert np.array_equal(unified_mask, expected_mask), (
        "统一掩膜与两者掩膜逻辑与不一致"
    )


# ---------- Property 6：不同尺寸插值后输出形状 ----------
# Feature: csv-to-zbf-converter, Property 6: 不同尺寸插值后输出形状
# Validates: Requirements 4.2


@given(pair=different_shape_arrays())
@settings(max_examples=100)
def test_different_size_output_shape(pair):
    """不同尺寸插值后输出形状：匹配后数组和掩膜形状应等于较大形状。

    对任意两个不同尺寸的二维数组，match_apertures 返回的
    匹配后光强、相位和统一掩膜的形状应等于两个输入中
    每个维度取最大值的形状。
    """
    irradiance, phase = pair

    matched_irr, matched_ph, unified_mask = match_apertures(
        irradiance, phase,
    )

    # 期望输出形状：每个维度取两个输入中的最大值
    expected_shape = (
        max(irradiance.shape[0], phase.shape[0]),
        max(irradiance.shape[1], phase.shape[1]),
    )

    # 验证三个输出的形状
    assert matched_irr.shape == expected_shape, (
        f"光强形状不匹配：期望 {expected_shape}，"
        f"实际 {matched_irr.shape}"
    )
    assert matched_ph.shape == expected_shape, (
        f"相位形状不匹配：期望 {expected_shape}，"
        f"实际 {matched_ph.shape}"
    )
    assert unified_mask.shape == expected_shape, (
        f"掩膜形状不匹配：期望 {expected_shape}，"
        f"实际 {unified_mask.shape}"
    )


# ---------- Property 7：掩膜外值为零 ----------
# Feature: csv-to-zbf-converter, Property 7: 掩膜外值为零
# Validates: Requirements 4.3, 4.4


@given(pair=same_shape_arrays())
@settings(max_examples=100)
def test_values_outside_mask_are_zero(pair):
    """掩膜外值为零：统一掩膜为 False 的像素值应恰好为 0.0。

    对任意光强和相位数组对，match_apertures 返回的
    匹配后数组中，统一掩膜为 False 的所有像素值
    应恰好为 0.0。
    """
    irradiance, phase = pair

    matched_irr, matched_ph, unified_mask = match_apertures(
        irradiance, phase,
    )

    # 掩膜外的光强值应恰好为零
    if np.any(~unified_mask):
        assert np.all(matched_irr[~unified_mask] == 0.0), (
            "匹配后光强中存在掩膜外非零值"
        )
        assert np.all(matched_ph[~unified_mask] == 0.0), (
            "匹配后相位中存在掩膜外非零值"
        )

# ============================================================
# 单元测试：detect_aperture 和 match_apertures 边界案例
# 验证：需求 3.3
# ============================================================


def test_detect_aperture_all_zeros():
    """全零数组调用 detect_aperture 应返回全 False 掩膜。

    边界案例：当输入数据全为零时，最大绝对值为 0，
    函数应直接返回全 False 掩膜，不会出现除零错误。
    """
    data = np.zeros((5, 5), dtype=np.float64)
    mask = detect_aperture(data)

    # 掩膜形状应与输入一致
    assert mask.shape == data.shape
    # 掩膜类型应为布尔
    assert mask.dtype == bool
    # 全零输入应返回全 False
    assert not np.any(mask), "全零数组应返回全 False 掩膜"


def test_detect_aperture_default_threshold():
    """验证默认阈值为数据最大绝对值的 1e-6 倍。

    构造已知数组 [[0, 0], [0, 100]]：
    - 最大绝对值 = 100
    - 默认阈值 = 100 * 1e-6 = 1e-4
    - 绝对值 > 1e-4 的像素应为 True，其余为 False
    验证：需求 3.3
    """
    data = np.array([[0.0, 0.0], [0.0, 100.0]])
    mask = detect_aperture(data)

    # 默认阈值 = 100 * 1e-6 = 1e-4
    # 只有值为 100 的像素绝对值 > 1e-4
    expected = np.array([
        [False, False],
        [False, True],
    ])
    np.testing.assert_array_equal(
        mask, expected,
        err_msg="默认阈值计算不正确",
    )


# ---------- Property 3：非正网格间距触发 ValueError ----------
# Feature: csv-to-zbf-converter, Property 3: 非正网格间距触发 ValueError
# Validates: Requirements 2.4

# 非正浮点数策略（<= 0）
_non_positive_float = st.floats(
    max_value=0, allow_nan=False, allow_infinity=False,
)
# 正浮点数策略
_positive_float = st.floats(
    min_value=1e-6, max_value=10.0,
    allow_nan=False, allow_infinity=False,
)


# 三种非正网格间距组合策略
_non_positive_dx_dy = st.one_of(
    # dx <= 0, dy > 0
    st.tuples(_non_positive_float, _positive_float),
    # dx > 0, dy <= 0
    st.tuples(_positive_float, _non_positive_float),
    # dx <= 0, dy <= 0
    st.tuples(_non_positive_float, _non_positive_float),
)


@given(dx_dy=_non_positive_dx_dy)
@settings(max_examples=100)
def test_non_positive_grid_spacing_raises(dx_dy):
    """非正网格间距应触发 ValueError。

    对任意 dx <= 0 或 dy <= 0 的值，
    调用 convert_csv_to_zbf 应抛出 ValueError。
    """
    dx, dy = dx_dy

    # 创建临时的有效 CSV 文件（简单 3x3 数组）
    simple_data = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])
    irr_path = None
    ph_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w",
        ) as f:
            irr_path = f.name
            np.savetxt(f, simple_data, delimiter=",")
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w",
        ) as f:
            ph_path = f.name
            np.savetxt(f, simple_data * 0.1, delimiter=",")

        with pytest.raises(ValueError):
            convert_csv_to_zbf(
                irr_path, ph_path, "dummy.zbf",
                dx=dx, dy=dy, wavelength=0.5,
            )
    finally:
        if irr_path and os.path.exists(irr_path):
            os.unlink(irr_path)
        if ph_path and os.path.exists(ph_path):
            os.unlink(ph_path)


# ---------- Property 8：复数电场构造正确性 ----------
# Feature: csv-to-zbf-converter, Property 8: 复数电场构造正确性
# Validates: Requirements 5.1


@st.composite
def irradiance_phase_pair(draw):
    """生成同形状的非负光强数组和相位数组。"""
    shape = draw(st.tuples(
        st.integers(min_value=3, max_value=64),
        st.integers(min_value=3, max_value=64),
    ))
    irr = draw(arrays(
        dtype=np.float64, shape=shape,
        elements=st.floats(
            min_value=0, max_value=100,
            allow_nan=False, allow_infinity=False,
        ),
    ))
    ph = draw(arrays(
        dtype=np.float64, shape=shape,
        elements=st.floats(
            min_value=-np.pi, max_value=np.pi,
            allow_nan=False, allow_infinity=False,
        ),
    ))
    return irr, ph


@given(pair=irradiance_phase_pair())
@settings(max_examples=100)
def test_complex_field_construction(pair):
    """复数电场构造正确性。

    对任意非负光强数组和相位数组，
    Ex = sqrt(irradiance) * exp(j * phase) 应满足：
    - |Ex|^2 ≈ irradiance
    - 对 irradiance > 0 的像素，angle(Ex) ≈ phase
    """
    irradiance, phase = pair

    # 构造复数电场
    Ex = np.sqrt(irradiance) * np.exp(1j * phase)

    # 验证 |Ex|^2 ≈ irradiance
    assert np.allclose(np.abs(Ex) ** 2, irradiance), (
        "复数电场模的平方与原始光强不一致"
    )

    # 对 irradiance > 0 的像素，验证 angle(Ex) ≈ phase
    mask = irradiance > 0
    if np.any(mask):
        assert np.allclose(
            np.angle(Ex)[mask], phase[mask], atol=1e-10,
        ), "复数电场相位与原始相位不一致"


# ---------- Property 9：ZBF 往返 ----------
# Feature: csv-to-zbf-converter, Property 9: ZBF 往返
# Validates: Requirements 6.1, 5.2


@st.composite
def zbf_roundtrip_inputs(draw):
    """生成 ZBF 往返测试所需的输入数据。

    包含同形状的非负光强数组、相位数组和正数物理参数。
    """
    shape = draw(st.tuples(
        st.integers(min_value=3, max_value=32),
        st.integers(min_value=3, max_value=32),
    ))
    irr = draw(arrays(
        dtype=np.float64, shape=shape,
        elements=st.floats(
            min_value=0, max_value=100,
            allow_nan=False, allow_infinity=False,
        ),
    ))
    ph = draw(arrays(
        dtype=np.float64, shape=shape,
        elements=st.floats(
            min_value=-np.pi, max_value=np.pi,
            allow_nan=False, allow_infinity=False,
        ),
    ))
    dx = draw(st.floats(
        min_value=1e-6, max_value=10.0,
        allow_nan=False, allow_infinity=False,
    ))
    dy = draw(st.floats(
        min_value=1e-6, max_value=10.0,
        allow_nan=False, allow_infinity=False,
    ))
    wavelength = draw(st.floats(
        min_value=1e-6, max_value=10.0,
        allow_nan=False, allow_infinity=False,
    ))
    return irr, ph, dx, dy, wavelength


@given(inputs=zbf_roundtrip_inputs())
@settings(max_examples=100)
def test_zbf_roundtrip(inputs):
    """ZBF 往返：写入后读回，振幅和相位应近似相等，物理参数完全一致。

    对任意有效的非负光强数组、相位数组和正数物理参数，
    经完整转换流程后用 read_zbf 读回，
    振幅和相位应与原始数据近似相等，
    物理参数完全一致。
    """
    irr, ph, dx, dy, wavelength = inputs

    irr_path = None
    ph_path = None
    zbf_path = None
    try:
        # 将光强和相位写入临时 CSV 文件
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w",
        ) as f:
            irr_path = f.name
            np.savetxt(f, irr, delimiter=",")
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w",
        ) as f:
            ph_path = f.name
            np.savetxt(f, ph, delimiter=",")
        with tempfile.NamedTemporaryFile(
            suffix=".zbf", delete=False,
        ) as f:
            zbf_path = f.name

        # 调用 convert_csv_to_zbf 生成 ZBF 文件
        zbf_data = convert_csv_to_zbf(
            irr_path, ph_path, zbf_path,
            dx=dx, dy=dy, wavelength=wavelength,
            verify=False,
        )

        # 用 read_zbf 读回
        readback = read_zbf(zbf_path)

        # 执行孔径匹配以获取统一掩膜（与转换流程一致）
        matched_irr, matched_ph, mask = match_apertures(
            irr, ph,
        )

        # 比较振幅：readback 振幅 ≈ sqrt(matched_irr)
        expected_amp = np.sqrt(matched_irr)
        readback_amp = np.abs(readback.Ex)
        assert np.allclose(readback_amp, expected_amp), (
            f"振幅不匹配：最大差异 "
            f"{np.max(np.abs(readback_amp - expected_amp))}"
        )

        # 比较相位（仅在孔径内）
        if np.any(mask):
            expected_phase = matched_ph[mask]
            readback_phase = np.angle(readback.Ex)[mask]
            assert np.allclose(
                readback_phase, expected_phase, atol=1e-10,
            ), "孔径内相位不匹配"

        # 比较物理参数（完全一致）
        assert readback.dx == dx, (
            f"dx 不匹配：期望 {dx}，实际 {readback.dx}"
        )
        assert readback.dy == dy, (
            f"dy 不匹配：期望 {dy}，实际 {readback.dy}"
        )
        assert readback.wavelength == wavelength, (
            f"wavelength 不匹配：期望 {wavelength}，"
            f"实际 {readback.wavelength}"
        )
        assert readback.units == 0, (
            f"units 不匹配：期望 0，实际 {readback.units}"
        )
        assert readback.index == 1.0, (
            f"index 不匹配：期望 1.0，"
            f"实际 {readback.index}"
        )

    finally:
        for p in [irr_path, ph_path, zbf_path]:
            if p and os.path.exists(p):
                os.unlink(p)


# ---------- CLI 单元测试 ----------


def test_cli_dy_defaults_to_dx(tmp_path):
    """--dy 未提供时应默认等于 --dx。验证：需求 8.4"""
    # 创建临时 CSV 文件（简单 3x3 数据）
    irr_data = np.ones((3, 3))
    ph_data = np.zeros((3, 3))
    irr_csv = tmp_path / "irr.csv"
    ph_csv = tmp_path / "ph.csv"
    out_zbf = tmp_path / "out.zbf"
    np.savetxt(str(irr_csv), irr_data, delimiter=",")
    np.savetxt(str(ph_csv), ph_data, delimiter=",")

    # 调用 csv_to_zbf.py，不提供 --dy
    result = subprocess.run(
        [
            sys.executable, "csv_to_zbf.py",
            str(irr_csv), str(ph_csv), str(out_zbf),
            "--dx", "0.1",
            "--wavelength", "0.5",
            "--no-plot",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"命令执行失败：{result.stderr}"
    )

    # 读回 ZBF 文件，验证 dy == dx
    from zbf_io import read_zbf
    zbf = read_zbf(str(out_zbf))
    assert zbf.dy == pytest.approx(0.1), (
        f"dy 应等于 dx=0.1，实际 dy={zbf.dy}"
    )


def test_cli_missing_args_exits_nonzero():
    """参数不完整时应以非零退出码退出。验证：需求 8.5"""
    # 不提供任何参数
    result = subprocess.run(
        [sys.executable, "csv_to_zbf.py"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0, (
        "缺少参数时应以非零退出码退出"
    )
