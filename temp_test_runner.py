
import sys
import os
sys.path.insert(0, r'd:\BTS\tests\integration')
from 离轴抛物面镜传输误差标准测试文件 import run_oap_test

if __name__ == '__main__':
    print("Running OAP Test via wrapper...")
    try:
        try:
            # Set z_mm=1000.0 (focal length) to ensure collimated output
            result = run_oap_test(
                verbose=True, 
                grid_size=512, 
                z_mm=1000.0,
                # propagation_method='local_raytracing' # Default
            )
        except TypeError:
            print("run_oap_test doesn't support grid_size/z_mm arg yet, using default.")
            result = run_oap_test(verbose=False)

        
        print("\n--- TEST RESULTS ---")
        print(f"Success: {result['success']}")
        print(f"Phase RMS: {result['phase_rms_mwaves']:.4f} milli-waves")
        print(f"Phase PV: {result['phase_pv_mwaves']:.4f} milli-waves")
        print(f"Amplitude RMS: {result['amplitude_rms_percent']:.4f} %")
        print(f"Amplitude PV: {result['amplitude_pv_percent']:.4f} %")
        print("--------------------")
        
    except Exception as e:
        print(f"Wrapper failed: {e}")
        import traceback
        traceback.print_exc()
