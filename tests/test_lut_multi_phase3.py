"""
Phase 3 test: N-D interpolation and NetCDF persistence.

Tests:
1. forward() interpolation at various query points
2. save() to NetCDF
3. load() from NetCDF
4. Round-trip consistency (save → load → query)
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from mieEODG.lut_multi import LUTAxis, MultiDimLUT


class TestForwardInterpolation:
    """Test forward() method for N-D interpolation."""

    @pytest.fixture
    def h2so4_lut(self):
        """Build and compute a small H₂SO₄ LUT for testing."""
        axes = [
            LUTAxis('r0', np.array([0.1, 0.5, 1.0]), 'µm', log_spacing=True,
                    fixed_for_ri_load=True),
            LUTAxis('composition', np.array([48.0, 58.0]), 'wt%',
                    fixed_for_ri_load=False),
        ]
        wn = np.arange(500, 1500, 100)

        lut = MultiDimLUT.from_aria_database(
            substance='Sulphuric acid',
            source='Myhre 2003',
            axes=axes,
            wavenumber=wn,
            quantity='MEC',
        )
        lut.compute(progress=False)
        return lut

    def test_forward_at_grid_point(self, h2so4_lut):
        """forward() at an exact grid point should match data array."""
        # Query at first grid point
        result = h2so4_lut.forward(r0=0.1, composition=48.0)

        assert 'MEC' in result
        spectrum = result['MEC']
        assert spectrum.shape == (len(h2so4_lut.wavenumber),)

        # Should match the data array exactly
        expected = h2so4_lut.data['MEC'][0, 0, :]
        # Allow small numerical differences
        if not np.any(np.isnan(expected)):
            np.testing.assert_allclose(spectrum, expected, rtol=1e-10)

    def test_forward_at_intermediate_point(self, h2so4_lut):
        """forward() at intermediate point should interpolate."""
        # Query at intermediate r0 (between 0.1 and 0.5)
        result = h2so4_lut.forward(r0=0.25, composition=48.0)

        spectrum = result['MEC']
        assert spectrum.shape == (len(h2so4_lut.wavenumber),)

        # Should be between grid point values
        if not np.any(np.isnan(h2so4_lut.data['MEC'][0, 0, :])):
            val_0p1 = h2so4_lut.data['MEC'][0, 0, 0]  # First wavenumber, r0=0.1
            val_0p5 = h2so4_lut.data['MEC'][1, 0, 0]  # First wavenumber, r0=0.5
            val_0p25 = spectrum[0]

            # Linear interpolation in log space (approximately)
            assert min(val_0p1, val_0p5) <= val_0p25 <= max(val_0p1, val_0p5)

    def test_forward_missing_axis_raises(self, h2so4_lut):
        """forward() raises ValueError if required axis is missing."""
        with pytest.raises(ValueError, match="Missing required axis"):
            h2so4_lut.forward(r0=0.5)  # composition missing

    def test_forward_before_compute_raises(self):
        """forward() raises RuntimeError if compute() not called."""
        axes = [LUTAxis('r0', [0.1, 0.5], 'µm')]
        wn = np.arange(500, 1000, 50)
        ri_constraint = __import__('mieEODG.lut_multi', fromlist=['RIConstraint']).RIConstraint(
            'Sulphuric acid', 'Myhre 2003', []
        )
        lut = MultiDimLUT(axes, ri_constraint, wn)

        with pytest.raises(RuntimeError, match="compute"):
            lut.forward(r0=0.2)


class TestNetCDFPersistence:
    """Test save() and load() for NetCDF persistence."""

    def setup_method(self):
        """Skip tests if netCDF4 is not available."""
        pytest.importorskip('netCDF4')

    @pytest.fixture
    def h2so4_lut(self):
        """Build and compute a small H₂SO₄ LUT."""
        axes = [
            LUTAxis('r0', np.array([0.1, 0.3, 0.5]), 'µm', log_spacing=True,
                    fixed_for_ri_load=True),
            LUTAxis('composition', np.array([48.0, 58.0]), 'wt%',
                    fixed_for_ri_load=False),
        ]
        wn = np.arange(500, 1000, 50)

        lut = MultiDimLUT.from_aria_database(
            substance='Sulphuric acid',
            source='Myhre 2003',
            axes=axes,
            wavenumber=wn,
            quantity='MEC',
        )
        lut.compute(progress=False)
        return lut

    def test_save_creates_file(self, h2so4_lut):
        """save() creates a NetCDF file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test_lut.nc'

            h2so4_lut.save(path)

            assert path.exists()
            assert path.suffix == '.nc'

    def test_save_creates_parent_dir(self, h2so4_lut):
        """save() creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'subdir' / 'nested' / 'test_lut.nc'

            h2so4_lut.save(path)

            assert path.parent.exists()
            assert path.exists()

    def test_save_before_compute_raises(self):
        """save() raises RuntimeError if compute() not called."""
        axes = [LUTAxis('r0', [0.1, 0.5], 'µm')]
        wn = np.arange(500, 1000, 50)
        ri_constraint = __import__('mieEODG.lut_multi', fromlist=['RIConstraint']).RIConstraint(
            'Sulphuric acid', 'Myhre 2003', []
        )
        lut = MultiDimLUT(axes, ri_constraint, wn)

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(RuntimeError, match="compute"):
                lut.save(Path(tmpdir) / 'test.nc')

    def test_load_reconstructs_lut(self, h2so4_lut):
        """load() reconstructs a LUT from saved NetCDF file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test_lut.nc'

            # Save
            h2so4_lut.save(path)

            # Load
            loaded_lut = MultiDimLUT.load(path)

            # Check shape and metadata
            assert loaded_lut.shape == h2so4_lut.shape
            assert loaded_lut.quantity == h2so4_lut.quantity
            assert loaded_lut.ri_constraint.substance == h2so4_lut.ri_constraint.substance
            assert loaded_lut.ri_constraint.source == h2so4_lut.ri_constraint.source

    def test_load_preserves_data(self, h2so4_lut):
        """load() preserves data array exactly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test_lut.nc'

            original_data = h2so4_lut.data['MEC'].copy()
            h2so4_lut.save(path)

            loaded_lut = MultiDimLUT.load(path)

            # Data should match exactly (or within numerical tolerance)
            np.testing.assert_array_equal(loaded_lut.data['MEC'], original_data)

    def test_load_preserves_axes(self, h2so4_lut):
        """load() preserves axis definitions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test_lut.nc'

            h2so4_lut.save(path)
            loaded_lut = MultiDimLUT.load(path)

            # Check axes
            assert len(loaded_lut.axes) == len(h2so4_lut.axes)
            for orig, loaded in zip(h2so4_lut.axes, loaded_lut.axes):
                assert orig.name == loaded.name
                assert orig.units == loaded.units
                np.testing.assert_array_equal(orig.values, loaded.values)
                assert orig.log_spacing == loaded.log_spacing
                assert orig.fixed_for_ri_load == loaded.fixed_for_ri_load

    def test_load_nonexistent_file_raises(self):
        """load() raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            MultiDimLUT.load('/nonexistent/path/to/file.nc')


class TestRoundTrip:
    """Test save → load → query round-trip consistency."""

    def setup_method(self):
        """Skip tests if netCDF4 is not available."""
        pytest.importorskip('netCDF4')

    @pytest.fixture
    def h2so4_lut(self):
        """Build and compute a small LUT."""
        axes = [
            LUTAxis('r0', np.array([0.1, 0.5]), 'µm', log_spacing=True,
                    fixed_for_ri_load=True),
            LUTAxis('composition', np.array([48.0]), 'wt%',
                    fixed_for_ri_load=False),
        ]
        wn = np.arange(500, 1000, 100)

        lut = MultiDimLUT.from_aria_database(
            substance='Sulphuric acid',
            source='Myhre 2003',
            axes=axes,
            wavenumber=wn,
            quantity='MEC',
        )
        lut.compute(progress=False)
        return lut

    def test_query_before_and_after_save_load(self, h2so4_lut):
        """Query results should be identical before and after save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test_lut.nc'

            # Query original
            result_orig = h2so4_lut.forward(r0=0.1, composition=48.0)
            spectrum_orig = result_orig['MEC']

            # Save and load
            h2so4_lut.save(path)
            loaded_lut = MultiDimLUT.load(path)

            # Query loaded
            result_loaded = loaded_lut.forward(r0=0.1, composition=48.0)
            spectrum_loaded = result_loaded['MEC']

            # Should match exactly
            if not np.any(np.isnan(spectrum_orig)):
                np.testing.assert_array_almost_equal(spectrum_orig, spectrum_loaded)


class TestInterpolationEdgeCases:
    """Test edge cases in forward() interpolation."""

    @pytest.fixture
    def lut_with_nans(self):
        """Build a LUT with some NaN values."""
        axes = [
            LUTAxis('r0', np.array([0.1, 0.5]), 'µm', log_spacing=True,
                    fixed_for_ri_load=True),
            LUTAxis('composition', np.array([48.0, 58.0]), 'wt%',
                    fixed_for_ri_load=False),
        ]
        wn = np.arange(500, 1000, 100)

        lut = MultiDimLUT.from_aria_database(
            substance='Sulphuric acid',
            source='Myhre 2003',
            axes=axes,
            wavenumber=wn,
        )
        lut.compute(progress=False)
        return lut

    def test_forward_with_nans_in_data(self, lut_with_nans):
        """forward() handles NaN values gracefully."""
        # Query should still work even if some data points are NaN
        result = lut_with_nans.forward(r0=0.2, composition=50.0)
        spectrum = result['MEC']

        # May contain NaN, but should have correct shape
        assert spectrum.shape == (len(lut_with_nans.wavenumber),)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
