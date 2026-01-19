"""
调试 PROPER prop_lens 的行为
"""
import numpy as np
import proper


def test_proper_lens():
    """测试 PROPER 的 prop_lens 功能"""
    
    print("=" * 70)
    print("测试 PROPER prop_lens")
    print("=" * 70)
    
    # 参数
    wavelength_um = 10.64
    wavelength_m = wavelength_um * 1e-6
    w0 = 10.0  # mm
    beam_diameter_m = 4 * w0 * 1e-3  # 40 mm
    grid_size = 256
    beam_ratio = 0.5
    
    # 计算瑞利距离
    z_R = np.pi * w0**2 / wavelength_um  # mm
    print(f"\n光束参数:")
    print(f"  波长: {wavelength_um} μm")
    print(f"  束腰: {w0} mm")
    print(f"  瑞利距离: {z_R:.1f} mm")
    
    # 测试不同焦距的透镜
    focal_lengths = [-50.0, -100.0, 100.0, 50.0]  # mm
    
    for f in focal_lengths:
        print(f"\n" + "-" * 70)
        print(f"焦距 f = {f} mm")
        print("-" * 70)
        
        # 初始化波前
        wfo = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, beam_ratio)
        
        # 应用初始高斯光束
        n = proper.prop_get_gridsize(wfo)
        sampling = proper.prop_get_sampling(wfo) * 1e3  # mm
        half_size = sampling * n / 2
        coords = np.linspace(-half_size, half_size, n)
        X, Y = np.meshgrid(coords, coords)
        R_sq = X**2 + Y**2
        
        amplitude = np.exp(-R_sq / w0**2)
        gaussian_field = proper.prop_shift_center(amplitude)
        wfo.wfarr = wfo.wfarr * gaussian_field
        
        # 记录初始状态
        print(f"  初始状态:")
        print(f"    z = {wfo.z * 1e3:.3f} mm")
        print(f"    z_w0 = {wfo.z_w0 * 1e3:.3f} mm")
        print(f"    w0 = {wfo.w0 * 1e3:.6f} mm")
        print(f"    reference_surface = {wfo.reference_surface}")
        
        # 计算初始光束半径
        amp0 = proper.prop_get_amplitude(wfo)
        intensity0 = amp0**2
        total0 = np.sum(intensity0)
        x_var0 = np.sum(X**2 * intensity0) / total0
        y_var0 = np.sum(Y**2 * intensity0) / total0
        w_init = np.sqrt(2 * (x_var0 + y_var0))
        print(f"    测量光束半径: {w_init:.3f} mm")
        
        # 应用透镜
        proper.prop_lens(wfo, f * 1e-3)
        
        print(f"  透镜后状态:")
        print(f"    z = {wfo.z * 1e3:.3f} mm")
        print(f"    z_w0 = {wfo.z_w0 * 1e3:.3f} mm")
        print(f"    w0 = {wfo.w0 * 1e3:.6f} mm")
        print(f"    reference_surface = {wfo.reference_surface}")
        
        # 传播不同距离
        distances = [10.0, 50.0, 100.0]  # mm
        
        for d in distances:
            # 重新初始化
            wfo2 = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, beam_ratio)
            wfo2.wfarr = wfo2.wfarr * gaussian_field
            proper.prop_lens(wfo2, f * 1e-3)
            proper.prop_propagate(wfo2, d * 1e-3)
            
            # 计算光束半径
            amp = proper.prop_get_amplitude(wfo2)
            intensity = amp**2
            total = np.sum(intensity)
            if total > 1e-15:
                x_var = np.sum(X**2 * intensity) / total
                y_var = np.sum(Y**2 * intensity) / total
                w_proper = np.sqrt(2 * (x_var + y_var))
            else:
                w_proper = 0.0
            
            # ABCD 计算
            # 透镜 + 传播的 ABCD 矩阵
            A = 1 - d/f
            B = d
            C = -1/f
            D = 1
            
            q_in = 1j * z_R
            q_out = (A * q_in + B) / (C * q_in + D)
            inv_q_out = 1 / q_out
            w_abcd = np.sqrt(-wavelength_um / (np.pi * np.imag(inv_q_out)))
            
            error = abs(w_proper - w_abcd) / w_abcd * 100 if w_abcd > 0 else 0
            
            print(f"    传播 {d:5.1f} mm: PROPER={w_proper:8.3f} mm, ABCD={w_abcd:8.3f} mm, 误差={error:6.2f}%")


def test_proper_propagation_only():
    """测试纯传播（无透镜）"""
    
    print("\n" + "=" * 70)
    print("测试纯传播（无透镜）")
    print("=" * 70)
    
    wavelength_um = 10.64
    wavelength_m = wavelength_um * 1e-6
    w0 = 10.0  # mm
    beam_diameter_m = 4 * w0 * 1e-3
    grid_size = 256
    beam_ratio = 0.5
    
    z_R = np.pi * w0**2 / wavelength_um  # mm
    
    distances = [0, 100, 500, 1000, 5000]  # mm
    
    print(f"\n瑞利距离: {z_R:.1f} mm")
    print(f"{'距离 (mm)':<12} {'PROPER (mm)':<12} {'ABCD (mm)':<12} {'误差 (%)':<10}")
    print("-" * 50)
    
    for d in distances:
        wfo = proper.prop_begin(beam_diameter_m, wavelength_m, grid_size, beam_ratio)
        
        n = proper.prop_get_gridsize(wfo)
        sampling = proper.prop_get_sampling(wfo) * 1e3
        half_size = sampling * n / 2
        coords = np.linspace(-half_size, half_size, n)
        X, Y = np.meshgrid(coords, coords)
        R_sq = X**2 + Y**2
        
        amplitude = np.exp(-R_sq / w0**2)
        gaussian_field = proper.prop_shift_center(amplitude)
        wfo.wfarr = wfo.wfarr * gaussian_field
        
        if d > 0:
            proper.prop_propagate(wfo, d * 1e-3)
        
        amp = proper.prop_get_amplitude(wfo)
        intensity = amp**2
        total = np.sum(intensity)
        x_var = np.sum(X**2 * intensity) / total
        y_var = np.sum(Y**2 * intensity) / total
        w_proper = np.sqrt(2 * (x_var + y_var))
        
        # ABCD: 纯传播
        w_abcd = w0 * np.sqrt(1 + (d / z_R)**2)
        
        error = abs(w_proper - w_abcd) / w_abcd * 100
        
        print(f"{d:<12} {w_proper:<12.3f} {w_abcd:<12.3f} {error:<10.2f}")


if __name__ == "__main__":
    test_proper_propagation_only()
    test_proper_lens()
