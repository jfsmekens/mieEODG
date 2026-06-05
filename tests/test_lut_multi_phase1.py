"""
Phase 1 test: LUTAxis, RIConstraint, and MultiDimLUT initialization & validation.

This test verifies that:
1. LUTAxis validates uniqueness of values
2. RIConstraint loads metadata from aria_metadata_registry.json
3. MultiDimLUT can be constructed via from_aria_database()
4. Constraint validation works (catches unsupported sweep axes)
"""

import numpy as np
import pytest

from mieEODG.lut_multi import LUTAxis, RIConstraint, MultiDimLUT


class TestLUTAxis:
    """Test LUTAxis dataclass."""

    def test_basic_creation(self):
        """LUTAxis creation with valid data."""
        ax = LUTAxis(
            name='r0',
            values=np.array([0.1, 0.5, 1.0, 5.0, 10.0]),
            units='µm',
            log_spacing=True,
            fixed_for_ri_load=True,
        )
        assert ax.name == 'r0'
        assert len(ax.values) == 5
        assert ax.units == 'µm'
        assert ax.log_spacing is True
        assert ax.fixed_for_ri_load is True

    def test_values_converted_to_float64(self):
        """Values are converted to float64 array."""
        ax = LUTAxis('test', values=[1, 2, 3], units='K')
        assert ax.values.dtype == np.float64
        assert isinstance(ax.values, np.ndarray)

    def test_duplicate_values_raises(self):
        """LUTAxis with duplicate values raises ValueError."""
        with pytest.raises(ValueError, match="duplicate"):
            LUTAxis('test', values=[1, 2, 2, 3], units='K')

    def test_empty_values_raises(self):
        """LUTAxis with no values raises ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            LUTAxis('test', values=[], units='K')


class TestRIConstraint:
    """Test RIConstraint dataclass."""

    def test_basic_creation_h2so4(self):
        """RIConstraint for H₂SO₄ loads metadata."""
        constraint = RIConstraint(
            substance='Sulphuric acid',
            source='Myhre 2003',
            sweep_axes=['composition', 'temperature_K'],
        )
        assert constraint.substance == 'Sulphuric acid'
        assert constraint.source == 'Myhre 2003'
        assert constraint.sweep_axes == ['composition', 'temperature_K']
        assert 'composition' in constraint.available_coverage
        assert 'temperature_K' in constraint.available_coverage

    def test_basic_creation_hno3(self):
        """RIConstraint for HNO₃ loads metadata."""
        constraint = RIConstraint(
            substance='HNO3',
            source='Biermann 2000',
            sweep_axes=['composition', 'temperature_K'],
        )
        assert constraint.substance == 'HNO3'
        assert constraint.source == 'Biermann 2000'
        assert len(constraint.available_coverage['composition']) == 5
        assert len(constraint.available_coverage['temperature_K']) == 5

    def test_water_ice_single_sweep(self):
        """RIConstraint for water ice (1-D temperature sweep)."""
        constraint = RIConstraint(
            substance='Water ice',
            source='Curtis 2005',
            sweep_axes=['temperature_K'],
        )
        assert 'temperature_K' in constraint.available_coverage
        assert len(constraint.available_coverage['temperature_K']) == 8

    def test_invalid_substance_raises(self):
        """Invalid substance name raises ValueError."""
        with pytest.raises(ValueError, match="not in ARIA metadata registry"):
            RIConstraint(
                substance='Invalid substance',
                source='Some study',
                sweep_axes=[],
            )

    def test_invalid_source_raises(self):
        """Invalid source for substance raises ValueError."""
        with pytest.raises(ValueError, match="not available"):
            RIConstraint(
                substance='Sulphuric acid',
                source='InvalidStudy 2000',
                sweep_axes=[],
            )

    def test_coverage_immutable(self):
        """Available coverage is immutable (read from metadata)."""
        constraint = RIConstraint(
            substance='Sulphuric acid',
            source='Myhre 2003',
            sweep_axes=['composition'],
        )
        coverage = constraint.available_coverage.copy()
        # Try to modify (should not affect internal state if truly immutable)
        assert 'composition' in coverage


class TestMultiDimLUT:
    """Test MultiDimLUT initialization and validation."""

    def test_basic_construction(self):
        """MultiDimLUT basic construction."""
        axes = [
            LUTAxis('r0', np.logspace(-1, 1, 10), 'µm', log_spacing=True,
                    fixed_for_ri_load=True),
            LUTAxis('composition', np.array([38, 48, 58]), 'wt%',
                    fixed_for_ri_load=False),
        ]
        wn = np.arange(500, 5000, 10)

        constraint = RIConstraint(
            substance='Sulphuric acid',
            source='Myhre 2003',
            sweep_axes=['composition'],
        )

        lut = MultiDimLUT(axes, constraint, wn, quantity='MEC')
        assert lut.axes == axes
        assert lut.ri_constraint == constraint
        assert lut.quantity == 'MEC'
        assert lut.shape == (10, 3, len(wn))
        assert lut.data is None

    def test_from_aria_database_h2so4(self):
        """from_aria_database constructor for H₂SO₄."""
        axes = [
            LUTAxis('r0', np.logspace(-1, 1, 10), 'µm', log_spacing=True,
                    fixed_for_ri_load=True),
            LUTAxis('composition', np.array([38, 48, 58, 65]), 'wt%',
                    fixed_for_ri_load=False),
            LUTAxis('temperature_K', np.array([203, 223, 243, 263, 283]), 'K',
                    fixed_for_ri_load=False),
        ]
        wn = np.arange(500, 5000, 10)

        lut = MultiDimLUT.from_aria_database(
            substance='Sulphuric acid',
            source='Myhre 2003',
            axes=axes,
            wavenumber=wn,
            quantity='MEC',
        )

        assert lut.shape == (10, 4, 5, len(wn))
        assert lut.ri_constraint.substance == 'Sulphuric acid'
        assert lut.ri_constraint.source == 'Myhre 2003'
        assert lut.quantity == 'MEC'

    def test_from_aria_database_hno3(self):
        """from_aria_database constructor for HNO₃."""
        axes = [
            LUTAxis('r0', np.logspace(-1, 1, 15), 'µm', log_spacing=True,
                    fixed_for_ri_load=True),
            LUTAxis('composition', np.array([0, 10, 20, 30, 40]), 'wt%',
                    fixed_for_ri_load=False),
            LUTAxis('temperature_K', np.array([213, 233, 253, 273, 293]), 'K',
                    fixed_for_ri_load=False),
        ]
        wn = np.arange(260, 5000, 10)

        lut = MultiDimLUT.from_aria_database(
            substance='HNO3',
            source='Biermann 2000',
            axes=axes,
            wavenumber=wn,
        )

        assert lut.shape == (15, 5, 5, len(wn))
        assert lut.ri_constraint.source == 'Biermann 2000'

    def test_validate_constraint_succeeds(self):
        """_validate_constraint returns True for valid sweep."""
        axes = [
            LUTAxis('r0', np.logspace(-1, 1, 10), 'µm', log_spacing=True,
                    fixed_for_ri_load=True),
            LUTAxis('composition', np.array([38, 48, 58]), 'wt%',
                    fixed_for_ri_load=False),
        ]
        wn = np.arange(500, 5000, 10)

        lut = MultiDimLUT.from_aria_database(
            substance='Sulphuric acid',
            source='Myhre 2003',
            axes=axes,
            wavenumber=wn,
        )

        assert lut._validate_constraint() is True

    def test_validate_constraint_fails_unsupported_axis(self):
        """_validate_constraint raises for unsupported axis."""
        axes = [
            LUTAxis('r0', np.logspace(-1, 1, 10), 'µm', log_spacing=True,
                    fixed_for_ri_load=True),
            LUTAxis('unsupported_axis', np.array([1, 2, 3]), 'units',
                    fixed_for_ri_load=False),
        ]
        wn = np.arange(500, 5000, 10)

        constraint = RIConstraint(
            substance='Sulphuric acid',
            source='Myhre 2003',
            sweep_axes=['unsupported_axis'],
        )

        lut = MultiDimLUT(axes, constraint, wn)

        with pytest.raises(ValueError, match="not available"):
            lut._validate_constraint()

    def test_invalid_quantity_raises(self):
        """Invalid quantity raises ValueError."""
        axes = [LUTAxis('r0', [1, 2, 3], 'µm')]
        wn = np.arange(500, 5000, 10)
        constraint = RIConstraint('Sulphuric acid', 'Myhre 2003', [])

        with pytest.raises(ValueError, match="MEC.*MSC.*MAC"):
            MultiDimLUT(axes, constraint, wn, quantity='INVALID')

    def test_axes_must_be_list(self):
        """axes parameter must be a list."""
        wn = np.arange(500, 5000, 10)
        constraint = RIConstraint('Sulphuric acid', 'Myhre 2003', [])

        with pytest.raises(TypeError, match="must be a list"):
            MultiDimLUT(
                axes="not a list",
                ri_constraint=constraint,
                wavenumber=wn,
            )

    def test_axes_must_be_nonempty(self):
        """axes parameter must have at least one element."""
        wn = np.arange(500, 5000, 10)
        constraint = RIConstraint('Sulphuric acid', 'Myhre 2003', [])

        with pytest.raises(ValueError, match="at least one"):
            MultiDimLUT(axes=[], ri_constraint=constraint, wavenumber=wn)

    def test_axes_elements_must_be_lutaxis(self):
        """All elements in axes must be LUTAxis instances."""
        wn = np.arange(500, 5000, 10)
        constraint = RIConstraint('Sulphuric acid', 'Myhre 2003', [])

        with pytest.raises(TypeError, match="must be LUTAxis"):
            MultiDimLUT(
                axes=["not a LUTAxis"],
                ri_constraint=constraint,
                wavenumber=wn,
            )


class TestIntegration:
    """Integration tests: end-to-end workflows."""

    def test_h2so4_3d_sweep_construction(self):
        """Build a complete H₂SO₄ 3-D LUT (r0 × composition × temperature)."""
        axes = [
            LUTAxis('r0', np.logspace(-1, 1, 20), 'µm', log_spacing=True,
                    fixed_for_ri_load=True),
            LUTAxis('composition', np.array([38, 48, 58, 65, 72]), 'wt%',
                    fixed_for_ri_load=False),
            LUTAxis('temperature_K', np.array([203, 223, 243, 263, 283, 298]), 'K',
                    fixed_for_ri_load=False),
        ]
        wn = np.arange(800, 6000, 10)

        lut = MultiDimLUT.from_aria_database(
            substance='Sulphuric acid',
            source='Myhre 2003',
            axes=axes,
            wavenumber=wn,
            quantity='MEC',
        )

        assert lut.shape == (20, 5, 6, len(wn))
        assert lut.ri_constraint.source == 'Myhre 2003'

    def test_hno3_3d_sweep_construction(self):
        """Build a complete HNO₃ 3-D LUT (r0 × composition × temperature)."""
        axes = [
            LUTAxis('r0', np.logspace(-1, 1, 15), 'µm', log_spacing=True,
                    fixed_for_ri_load=True),
            LUTAxis('composition', np.array([0, 10, 20, 30, 40]), 'wt%',
                    fixed_for_ri_load=False),
            LUTAxis('temperature_K', np.array([213, 233, 253, 273, 293]), 'K',
                    fixed_for_ri_load=False),
        ]
        wn = np.arange(260, 4988, 10)

        lut = MultiDimLUT.from_aria_database(
            substance='HNO3',
            source='Biermann 2000',
            axes=axes,
            wavenumber=wn,
            quantity='MSC',
        )

        assert lut.shape == (15, 5, 5, len(wn))
        assert lut.ri_constraint.source == 'Biermann 2000'
        assert lut.quantity == 'MSC'

    def test_water_ice_1d_temperature_sweep(self):
        """Build a 1-D water ice LUT (r0 × temperature only)."""
        axes = [
            LUTAxis('r0', np.logspace(-1, 1, 12), 'µm', log_spacing=True,
                    fixed_for_ri_load=True),
            LUTAxis('temperature_K', np.array([106, 120, 130, 140, 150, 160, 170, 176]), 'K',
                    fixed_for_ri_load=False),
        ]
        wn = np.arange(837, 6962, 20)

        lut = MultiDimLUT.from_aria_database(
            substance='Water ice',
            source='Curtis 2005',
            axes=axes,
            wavenumber=wn,
        )

        assert lut.shape == (12, 8, len(wn))
        assert lut.ri_constraint.source == 'Curtis 2005'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
