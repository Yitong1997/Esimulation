'''多元件折叠光路集成测试

本模块测试包含多个折叠镜的光学系统的混合传播模式正确性。

测试内容：
- 双折叠镜系统（Z 形光路）
- 三折叠镜系统
- 折叠镜 + 聚焦镜组合系统
- 各采样面的 WFE RMS < 0.1 波

**Validates: Requirements 6.3**
'''

import sys
import numpy as np
import pytest

sys.path.insert(0, 'src')

from sequential_system import (
    SequentialOpticalSystem,
    GaussianBeamSource,
    FlatMirror,
    ParabolicMirror,
    SphericalMirror,
)


class TestDoubleFoldMirrorSystem:
    '''双折叠镜系统测试（Z 形光路）
    
    **Validates: Requirements 6.3**
    '''
    
    WAVELENGTH_UM = 0.633
    W0_INPUT_MM = 5.0
    D_TO_FOLD1_MM = 50.0
    D_FOLD1_TO_FOLD2_MM = 80.0
    D_FOLD2_TO_OUTPUT_MM = 50.0
    TILT_45_DEG = np.pi / 4
    WFE_RMS_TOLERANCE = 0.1

    def _create_double_fold_system(self, grid_size=256):
        source = GaussianBeamSource(
            wavelength=self.WAVELENGTH_UM,
            w0=self.W0_INPUT_MM,
            z0=0.0,
        )
        system = SequentialOpticalSystem(
            source=source,
            grid_size=grid_size,
            beam_ratio=0.5,
            use_hybrid_propagation=True,
            hybrid_num_rays=100,
        )
        total_path = self.D_TO_FOLD1_MM + self.D_FOLD1_TO_FOLD2_MM + self.D_FOLD2_TO_OUTPUT_MM
        system.add_sampling_plane(distance=0.0, name='Input')
        system.add_surface(FlatMirror(
            thickness=self.D_FOLD1_TO_FOLD2_MM,
            semi_aperture=15.0,
            tilt_x=self.TILT_45_DEG,
            name='Fold1',
        ))
        system.add_sampling_plane(
            distance=self.D_TO_FOLD1_MM + self.D_FOLD1_TO_FOLD2_MM / 2, 
            name='After Fold1'
        )
        system.add_surface(FlatMirror(
            thickness=self.D_FOLD2_TO_OUTPUT_MM,
            semi_aperture=15.0,
            tilt_x=self.TILT_45_DEG,
            name='Fold2',
        ))
        system.add_sampling_plane(distance=total_path, name='Output')
        return system
    
    def test_double_fold_propagation(self):
        '''验证双折叠镜系统传播正常完成'''
        system = self._create_double_fold_system()
        results = system.run()
        assert 'Input' in results.sampling_results
        assert 'After Fold1' in results.sampling_results
        assert 'Output' in results.sampling_results
        for name, result in results.sampling_results.items():
            assert result.beam_radius > 0
            assert np.isfinite(result.beam_radius)
    
    def test_double_fold_wfe_quality(self):
        '''验证双折叠镜系统各采样面的 WFE RMS < 0.1 波'''
        system = self._create_double_fold_system()
        results = system.run()
        for name, result in results.sampling_results.items():
            wfe_rms = result.wavefront_rms
            assert np.isfinite(wfe_rms)
            assert wfe_rms < self.WFE_RMS_TOLERANCE
    
    def test_double_fold_beam_radius_vs_abcd(self):
        '''验证双折叠镜系统与 ABCD 理论的光束半径一致性'''
        system = self._create_double_fold_system()
        results = system.run()
        for name, result in results.sampling_results.items():
            proper_w = result.beam_radius
            abcd_result = system.get_abcd_result(result.distance)
            abcd_w = abcd_result.w
            if abcd_w > 0.001:
                error_pct = abs(proper_w - abcd_w) / abcd_w
            else:
                error_pct = 0.0
            assert error_pct < 0.01


class TestTripleFoldMirrorSystem:
    '''三折叠镜系统测试
    
    **Validates: Requirements 6.3**
    '''
    
    WAVELENGTH_UM = 0.633
    W0_INPUT_MM = 5.0
    D_SEGMENT_MM = 40.0
    TILT_45_DEG = np.pi / 4
    WFE_RMS_TOLERANCE = 0.1
    
    def _create_triple_fold_system(self, grid_size=256):
        source = GaussianBeamSource(
            wavelength=self.WAVELENGTH_UM,
            w0=self.W0_INPUT_MM,
            z0=0.0,
        )
        system = SequentialOpticalSystem(
            source=source,
            grid_size=grid_size,
            beam_ratio=0.5,
            use_hybrid_propagation=True,
            hybrid_num_rays=100,
        )
        system.add_sampling_plane(distance=0.0, name='Input')
        system.add_surface(FlatMirror(
            thickness=self.D_SEGMENT_MM,
            semi_aperture=15.0,
            tilt_x=self.TILT_45_DEG,
            name='Fold1',
        ))
        system.add_sampling_plane(distance=self.D_SEGMENT_MM, name='After Fold1')
        system.add_surface(FlatMirror(
            thickness=self.D_SEGMENT_MM,
            semi_aperture=15.0,
            tilt_x=self.TILT_45_DEG,
            name='Fold2',
        ))
        system.add_sampling_plane(distance=2 * self.D_SEGMENT_MM, name='After Fold2')
        system.add_surface(FlatMirror(
            thickness=self.D_SEGMENT_MM,
            semi_aperture=15.0,
            tilt_x=self.TILT_45_DEG,
            name='Fold3',
        ))
        system.add_sampling_plane(distance=3 * self.D_SEGMENT_MM, name='Output')
        return system
    
    def test_triple_fold_propagation(self):
        '''验证三折叠镜系统传播正常完成'''
        system = self._create_triple_fold_system()
        results = system.run()
        expected_planes = ['Input', 'After Fold1', 'After Fold2', 'Output']
        for name in expected_planes:
            assert name in results.sampling_results
            result = results.sampling_results[name]
            assert result.beam_radius > 0
            assert np.isfinite(result.beam_radius)
    
    def test_triple_fold_wfe_quality(self):
        '''验证三折叠镜系统各采样面的 WFE RMS < 0.1 波'''
        system = self._create_triple_fold_system()
        results = system.run()
        for name, result in results.sampling_results.items():
            wfe_rms = result.wavefront_rms
            assert np.isfinite(wfe_rms)
            assert wfe_rms < self.WFE_RMS_TOLERANCE


class TestFoldWithFocusingMirrorSystem:
    '''折叠镜 + 聚焦镜组合系统测试
    
    **Validates: Requirements 6.3**
    '''
    
    WAVELENGTH_UM = 10.64
    W0_INPUT_MM = 10.0
    FOCAL_LENGTH_MM = 200.0
    OFF_AXIS_DISTANCE_MM = 400.0
    D_TO_FOLD1_MM = 30.0
    D_FOLD1_TO_OAP_MM = 50.0
    D_OAP_TO_FOLD2_MM = 50.0
    D_FOLD2_TO_OUTPUT_MM = 100.0
    TILT_45_DEG = np.pi / 4
    WFE_RMS_TOLERANCE = 0.1
    BEAM_RADIUS_ERROR_TOLERANCE = 0.01
    
    def _create_fold_focus_system(self, grid_size=512, beam_ratio=0.25):
        source = GaussianBeamSource(
            wavelength=self.WAVELENGTH_UM,
            w0=self.W0_INPUT_MM,
            z0=0.0,
        )
        system = SequentialOpticalSystem(
            source=source,
            grid_size=grid_size,
            beam_ratio=beam_ratio,
            use_hybrid_propagation=True,
            hybrid_num_rays=100,
        )
        total_path = (
            self.D_TO_FOLD1_MM +
            self.D_FOLD1_TO_OAP_MM +
            self.D_OAP_TO_FOLD2_MM +
            self.D_FOLD2_TO_OUTPUT_MM
        )
        system.add_sampling_plane(distance=0.0, name='Input')
        system.add_surface(FlatMirror(
            thickness=self.D_FOLD1_TO_OAP_MM,
            semi_aperture=20.0,
            tilt_x=self.TILT_45_DEG,
            name='Fold1',
        ))
        system.add_sampling_plane(
            distance=self.D_TO_FOLD1_MM + self.D_FOLD1_TO_OAP_MM / 2,
            name='After Fold1'
        )
        system.add_surface(ParabolicMirror(
            parent_focal_length=self.FOCAL_LENGTH_MM,
            thickness=self.D_OAP_TO_FOLD2_MM,
            semi_aperture=30.0,
            off_axis_distance=self.OFF_AXIS_DISTANCE_MM,
            tilt_x=self.TILT_45_DEG,
            name='OAP',
        ))
        system.add_sampling_plane(
            distance=self.D_TO_FOLD1_MM + self.D_FOLD1_TO_OAP_MM + self.D_OAP_TO_FOLD2_MM / 2,
            name='After OAP'
        )
        system.add_surface(FlatMirror(
            thickness=self.D_FOLD2_TO_OUTPUT_MM,
            semi_aperture=30.0,
            tilt_x=self.TILT_45_DEG,
            name='Fold2',
        ))
        system.add_sampling_plane(distance=total_path, name='Output')
        return system
    
    def test_fold_focus_propagation(self):
        '''验证折叠镜 + 聚焦镜系统传播正常完成'''
        system = self._create_fold_focus_system()
        results = system.run()
        expected_planes = ['Input', 'After Fold1', 'After OAP', 'Output']
        for name in expected_planes:
            assert name in results.sampling_results
            result = results.sampling_results[name]
            assert result.beam_radius > 0
            assert np.isfinite(result.beam_radius)
    
    def test_fold_focus_wfe_quality(self):
        '''验证折叠镜 + 聚焦镜系统各采样面的 WFE RMS < 0.1 波'''
        system = self._create_fold_focus_system()
        results = system.run()
        for name, result in results.sampling_results.items():
            wfe_rms = result.wavefront_rms
            assert np.isfinite(wfe_rms)
            assert wfe_rms < self.WFE_RMS_TOLERANCE
    
    def test_fold_focus_beam_radius_vs_abcd(self):
        '''验证折叠镜 + 聚焦镜系统与 ABCD 理论的光束半径一致性'''
        system = self._create_fold_focus_system()
        results = system.run()
        for name, result in results.sampling_results.items():
            proper_w = result.beam_radius
            abcd_result = system.get_abcd_result(result.distance)
            abcd_w = abcd_result.w
            if abcd_w > 0.001:
                error_pct = abs(proper_w - abcd_w) / abcd_w
            else:
                error_pct = 0.0
            assert error_pct < self.BEAM_RADIUS_ERROR_TOLERANCE


class TestMultiFoldParametric:
    '''多折叠光路参数化测试'''
    
    @pytest.mark.parametrize('num_folds', [2, 3, 4])
    def test_different_fold_counts(self, num_folds):
        '''测试不同折叠镜数量下的系统行为'''
        wavelength_um = 0.633
        w0_mm = 5.0
        segment_length_mm = 30.0
        source = GaussianBeamSource(
            wavelength=wavelength_um,
            w0=w0_mm,
            z0=0.0,
        )
        system = SequentialOpticalSystem(
            source=source,
            grid_size=256,
            beam_ratio=0.5,
            use_hybrid_propagation=True,
            hybrid_num_rays=100,
        )
        system.add_sampling_plane(distance=0.0, name='Input')
        for i in range(num_folds):
            system.add_surface(FlatMirror(
                thickness=segment_length_mm,
                semi_aperture=15.0,
                tilt_x=np.pi / 4,
                name='Fold{}'.format(i+1),
            ))
        total_path = num_folds * segment_length_mm
        system.add_sampling_plane(distance=total_path, name='Output')
        results = system.run()
        assert 'Output' in results.sampling_results
        output_result = results.sampling_results['Output']
        assert output_result.beam_radius > 0
        assert np.isfinite(output_result.beam_radius)
        assert output_result.wavefront_rms < 0.1
    
    @pytest.mark.parametrize('grid_size', [256, 512])
    def test_different_grid_sizes(self, grid_size):
        '''测试不同网格大小下的结果一致性'''
        source = GaussianBeamSource(
            wavelength=0.633,
            w0=5.0,
            z0=0.0,
        )
        system = SequentialOpticalSystem(
            source=source,
            grid_size=grid_size,
            beam_ratio=0.5,
            use_hybrid_propagation=True,
            hybrid_num_rays=100,
        )
        system.add_sampling_plane(distance=0.0, name='Input')
        system.add_surface(FlatMirror(
            thickness=50.0,
            semi_aperture=15.0,
            tilt_x=np.pi / 4,
        ))
        system.add_surface(FlatMirror(
            thickness=50.0,
            semi_aperture=15.0,
            tilt_x=np.pi / 4,
        ))
        system.add_sampling_plane(distance=100.0, name='Output')
        results = system.run()
        output_result = results.sampling_results['Output']
        assert output_result.beam_radius > 0
        assert np.isfinite(output_result.beam_radius)
        assert output_result.wavefront_rms < 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
