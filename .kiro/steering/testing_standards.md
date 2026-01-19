<!------------------------------------------------------------------------------------
# 测试规范

本文件定义了混合光学仿真项目的测试标准和规范。
inclusion: fileMatch
fileMatchPattern: '**/tests/**,**/*test*,**/*spec*'
-------------------------------------------------------------------------------------> 

## 测试框架

### 主要工具
- **pytest**：单元测试和集成测试
- **hypothesis**：属性基测试（Property-Based Testing）
- **numpy.testing**：数值比较

### 目录结构
```
tests/
├── unit/                    # 单元测试
│   ├── test_wavefront.py
│   ├── test_propagation.py
│   └── test_raytracing.py
├── integration/             # 集成测试
│   ├── test_hybrid_system.py
│   └── test_end_to_end.py
├── validation/              # 验证测试（与参考软件对比）
│   ├── test_vs_zemax.py
│   └── test_vs_codev.py
├── property/                # 属性基测试
│   ├── test_energy_conservation.py
│   └── test_phase_continuity.py
└── conftest.py              # pytest 配置和 fixtures
```

## 单元测试规范

### 命名约定
```python
# 测试文件：test_<模块名>.py
# 测试类：Test<类名>
# 测试方法：test_<功能描述>

class TestWavefront:
    def test_initialization_with_valid_params(self):
        """测试使用有效参数初始化波前"""
        pass
    
    def test_initialization_raises_on_invalid_grid_size(self):
        """测试无效网格大小时抛出异常"""
        pass
```

### 测试结构（AAA 模式）
```python
def test_opd_calculation():
    # Arrange（准备）
    optic = create_test_optic()
    expected_opd = 0.5  # 波长数
    
    # Act（执行）
    actual_opd = calculate_opd(optic, field_index=0)
    
    # Assert（断言）
    np.testing.assert_allclose(actual_opd, expected_opd, rtol=1e-6)
```

### 数值比较
```python
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

# 浮点数比较（使用相对容差）
assert_allclose(actual, expected, rtol=1e-6, atol=1e-10)

# 数组比较
assert_allclose(actual_array, expected_array, rtol=1e-6)

# 精确比较（整数或布尔）
assert_array_equal(actual, expected)
```

## 属性基测试（PBT）

### 核心属性

#### 1. 能量守恒
```python
from hypothesis import given, strategies as st

@given(
    wavelength=st.floats(min_value=0.4e-6, max_value=0.8e-6),
    grid_size=st.sampled_from([256, 512, 1024])
)
def test_energy_conservation(wavelength, grid_size):
    """
    **Validates: Requirements 1.1 - 能量守恒**
    
    传播前后总能量应保持不变（在数值精度范围内）
    """
    wfo = initialize_wavefront(wavelength, grid_size)
    initial_energy = compute_total_energy(wfo)
    
    propagate(wfo, distance=0.1)
    
    final_energy = compute_total_energy(wfo)
    
    assert_allclose(final_energy, initial_energy, rtol=1e-6)
```

#### 2. 相位连续性
```python
@given(
    opd_rms=st.floats(min_value=0.0, max_value=1.0)
)
def test_phase_continuity(opd_rms):
    """
    **Validates: Requirements 1.2 - 相位连续性**
    
    OPD 变化应该是连续的，不应有突变
    """
    opd = generate_smooth_opd(rms=opd_rms)
    gradient = np.gradient(opd)
    max_gradient = np.max(np.abs(gradient))
    
    # 梯度不应超过合理阈值
    assert max_gradient < 10.0  # 波长/像素
```

#### 3. 几何光学极限
```python
@given(
    focal_length=st.floats(min_value=0.01, max_value=1.0)
)
def test_geometric_optics_limit(focal_length):
    """
    **Validates: Requirements 1.3 - 几何光学极限**
    
    在大孔径、短波长极限下，物理光学结果应趋近几何光学
    """
    # 使用大孔径和短波长
    result_physical = compute_physical_optics(focal_length, wavelength=0.1e-6)
    result_geometric = compute_geometric_optics(focal_length)
    
    # 焦点位置应一致
    assert_allclose(result_physical.focus, result_geometric.focus, rtol=0.01)
```

### 生成器策略
```python
from hypothesis import strategies as st

# 波长策略（可见光范围）
wavelength_strategy = st.floats(min_value=0.38e-6, max_value=0.78e-6)

# 网格大小策略（2的幂次）
grid_size_strategy = st.sampled_from([128, 256, 512, 1024, 2048])

# 光学系统参数策略
optical_params_strategy = st.fixed_dictionaries({
    'focal_length': st.floats(min_value=0.01, max_value=1.0),
    'aperture': st.floats(min_value=0.001, max_value=0.1),
    'field_angle': st.floats(min_value=0.0, max_value=0.1)
})
```

## 验证测试

### 与 Zemax 对比
```python
class TestZemaxValidation:
    """与 Zemax 计算结果的对比验证"""
    
    @pytest.fixture
    def cooke_triplet(self):
        """Cooke Triplet 标准测试系统"""
        return load_test_system('cooke_triplet')
    
    def test_psf_matches_zemax(self, cooke_triplet):
        """PSF 计算结果应与 Zemax 一致"""
        our_psf = compute_psf(cooke_triplet)
        zemax_psf = load_reference_data('cooke_triplet_psf.npy')
        
        # Strehl 比应在 1% 以内
        our_strehl = compute_strehl(our_psf)
        zemax_strehl = compute_strehl(zemax_psf)
        assert_allclose(our_strehl, zemax_strehl, rtol=0.01)
```

### 标准测试系统
- Cooke Triplet
- Double Gauss
- Petzval Lens
- Cassegrain Telescope
- Schmidt Camera

## 测试 Fixtures

### conftest.py
```python
import pytest
import numpy as np

@pytest.fixture
def simple_lens():
    """简单单透镜系统"""
    from optiland import Optic
    lens = Optic()
    # ... 配置
    return lens

@pytest.fixture
def test_wavefront():
    """测试用波前"""
    import proper
    wfo = proper.prop_begin(0.01, 0.55e-6, 512)
    return wfo

@pytest.fixture(params=[256, 512, 1024])
def grid_size(request):
    """参数化网格大小"""
    return request.param
```

## 测试覆盖率

### 目标
- 整体覆盖率 > 80%
- 核心模块覆盖率 > 90%
- 所有公共 API 100% 覆盖

### 运行覆盖率报告
```bash
pytest --cov=src --cov-report=html tests/
```
