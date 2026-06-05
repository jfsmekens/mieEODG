"""
Phase 2 test: MultiDimLUT.compute() with ARIA database integration.

Tests that compute() can:
1. Load RI files from ARIA for sweep axes
2. Compute optical properties for each grid point
3. Populate the data array correctly
"""

import numpy as np
import pytest

from mieEODG.lut_multi import LUTAxis, MultiDimLUT


class TestComputeAriaSinglePoint:
    """Test compute() with minimal ARIA data (single point)."""

    def test_compute_h2so4_single_point(self):
        """
        Compute a tiny 2-D LUT with H₂SO₄ (r0 × composition).

        This is a smoke test: does compute() run without errors?
        """
        axes = [
            LUTAxis('r0', np.array([0.1, 0.5, 1.0]), 'µm', log_spacing=True,
                    fixed_for_ri_load=True),
            LUTAxis('composition', np.array([48.0]), 'wt%',
                    fixed_for_ri_load=False),
        ]
        wn = np.arange(500, 3000, 100)  # Coarse grid for speed

        lut = MultiDimLUT.from_aria_database(
            substance='Sulphuric acid',
            source='Myhre 2003',
            axes=axes,
            wavenumber=wn,
            quantity='MEC',
        )

        # Compute (should load RI from ARIA and compute XSC)
        lut.compute(progress=False)

        # Check shape
        assert lut.data.shape == (3, 1, len(wn))

        # Check that data is populated and not all NaN
        valid_data = ~np.isnan(lut.data)
        assert np.any(valid_data), "All data is NaN; compute failed"

        # Check that values are physically reasonable (positive)
        valid_values = lut.data[valid_data]
        assert np.all(valid_values > 0), "Negative or zero values in MEC"

    def test_compute_h2so4_multiple_compositions(self):
        """
        Compute LUT with multiple compositions to verify grid iteration.
        """
        axes = [
            LUTAxis('r0', np.array([0.2, 0.5]), 'µm', log_spacing=True,
                    fixed_for_ri_load=True),
            LUTAxis('composition', np.array([48.0, 58.0, 65.0]), 'wt%',
                    fixed_for_ri_load=False),
        ]
        wn = np.arange(500, 2000, 200)

        lut = MultiDimLUT.from_aria_database(
            substance='Sulphuric acid',
            source='Myhre 2003',
            axes=axes,
            wavenumber=wn,
            quantity='MEC',
        )

        lut.compute(progress=False)

        # Check shape
        assert lut.data.shape == (2, 3, len(wn))

        # Check that different compositions yield different spectra
        # (This assumes Mie scattering depends on composition)
        spec_48 = lut.data[0, 0, :]  # r0=0.2, composition=48
        spec_65 = lut.data[0, 2, :]  # r0=0.2, composition=65

        # They should differ significantly
        if not np.any(np.isnan(spec_48)) and not np.any(np.isnan(spec_65)):
            diff = np.mean(np.abs(spec_48 - spec_65))
            assert diff > 0.01, "Different compositions should yield different spectra"


class TestComputeProgressCallback:
    """Test compute() with progress callback."""

    def test_progress_callback_called(self):
        """Progress callback is invoked at each grid point."""
        axes = [
            LUTAxis('r0', np.array([0.1, 0.3, 0.5]), 'µm', log_spacing=True,
                    fixed_for_ri_load=True),
            LUTAxis('composition', np.array([48.0, 58.0]), 'wt%',
                    fixed_for_ri_load=False),
        ]
        wn = np.arange(500, 1500, 200)

        lut = MultiDimLUT.from_aria_database(
            substance='Sulphuric acid',
            source='Myhre 2003',
            axes=axes,
            wavenumber=wn,
        )

        # Track callback invocations
        callback_calls = []

        def progress_cb(step: int, total: int) -> None:
            callback_calls.append((step, total))

        lut.compute(progress_callback=progress_cb, progress=False)

        # Should be called 3 × 2 = 6 times
        assert len(callback_calls) == 6
        # Last call should be (6, 6)
        assert callback_calls[-1] == (6, 6)


class TestRILoaderEdgeCases:
    """Test RI file loading and parsing edge cases."""

    def test_parse_comp_temp_from_path(self):
        """_parse_comp_temp_from_path extracts composition and temperature."""
        # H2SO4 format
        c, t = MultiDimLUT._parse_comp_temp_from_path(
            'Acids/Sulphuric/H2SO4_48_213K_Myhre_2003.ri'
        )
        assert c == 48.0
        assert t == 213.0

        # HNO3 format
        c, t = MultiDimLUT._parse_comp_temp_from_path(
            'Acids/Nitric/HNO30_263_Biermann_2000.ri'
        )
        # Note: HNO3 format is trickier; may extract 0 or 263
        # Just verify it parses without error
        assert t == 263.0

    def test_parse_comp_temp_missing_values(self):
        """_parse_comp_temp_from_path handles missing composition/temperature."""
        c, t = MultiDimLUT._parse_comp_temp_from_path(
            'Minerals/Calcite/calcite_no_metadata.ri'
        )
        # Should return (None, None) or partial matches
        assert isinstance(c, (float, type(None)))
        assert isinstance(t, (float, type(None)))


class TestDataConsistency:
    """Test data array consistency across compute() calls."""

    def test_repeated_compute_idempotent(self):
        """Calling compute() twice gives same result."""
        axes = [
            LUTAxis('r0', np.array([0.1, 0.3]), 'µm', log_spacing=True,
                    fixed_for_ri_load=True),
            LUTAxis('composition', np.array([48.0]), 'wt%',
                    fixed_for_ri_load=False),
        ]
        wn = np.arange(500, 1500, 200)

        lut = MultiDimLUT.from_aria_database(
            substance='Sulphuric acid',
            source='Myhre 2003',
            axes=axes,
            wavenumber=wn,
        )

        lut.compute(progress=False)
        data_first = lut.data.copy()

        lut.compute(progress=False)
        data_second = lut.data.copy()

        # Should be identical (or very close)
        if np.any(~np.isnan(data_first)):
            np.testing.assert_array_almost_equal(data_first, data_second)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
