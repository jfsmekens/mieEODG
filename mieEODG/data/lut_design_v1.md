# N-Dimensional LUT Design v1: Source-Consistent Sweeps

## Overview

Multi-dimensional LUTs for aerosol optical properties must maintain **single-source consistency** along each interpolation axis to avoid mixing measurement techniques, wavelength ranges, and processing methods.

**Core Constraint:** For a given sweep (e.g., r0 × composition × temperature), all RI files used must come from a **single authoritative source**.

---

## Key Data Structures

### 1. LUTAxis (dataclass)

```python
@dataclass
class LUTAxis:
    """
    Defines one dimension of a multi-dimensional LUT.
    """
    name: str                          # 'r0', 'composition', 'temperature_K'
    values: np.ndarray                 # Grid points (e.g., [10, 20, 30, ...] for composition)
    units: str                         # e.g., 'µm', 'wt%', 'K'
    log_spacing: bool = False          # True for r0 (log-spaced), False for T/C (linear)
    fixed_for_ri_load: bool = False    # True = use single RI; False = load multiple RIs per point
    
    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=np.float64)
        if len(np.unique(self.values)) != len(self.values):
            raise ValueError(f"Axis {self.name} has duplicate values")
```

### 2. RIConstraint (dataclass)

```python
@dataclass
class RIConstraint:
    """
    Specifies where and how to load RI files from the ARIA database.
    Enforces single-source consistency.
    """
    substance: str                      # e.g., 'Sulphuric acid', 'HNO3', 'Water ice'
    source: str                         # e.g., 'Myhre 2003', 'Biermann 2000', 'Curtis 2005'
    sweep_axes: list[str]               # Which LUT axes vary the RI: ['composition', 'temperature_K']
    fixed_values: dict = None           # e.g., {'concentration': 50.0} if not swept
    
    # Metadata from audit:
    available_coverage: dict = None     # e.g., {'composition': [10, 20, 30, ...], 
                                        #        'temperature_K': [200, 210, ..., 298]}
```

### 3. MultiDimLUT (class)

```python
class MultiDimLUT:
    """
    N-dimensional look-up table with source consistency enforcement.
    """
    
    def __init__(
        self,
        axes: list[LUTAxis],
        ri_constraint: RIConstraint,
        wavenumber: np.ndarray,
        quantity: str = 'MEC',  # 'MEC', 'MSC', 'MAC'
    ):
        """
        Initialize N-D LUT.
        
        Args:
            axes: List of LUTAxis objects defining each dimension
            ri_constraint: RIConstraint specifying source and sweep parameters
            wavenumber: Spectral grid [cm^-1]
            quantity: Which optical property to compute
        """
        self.axes = axes
        self.ri_constraint = ri_constraint
        self.wavenumber = wavenumber
        self.quantity = quantity
        
        # Shape of data array: one dimension per axis, last dimension is wavenumber
        self.shape = tuple(len(ax.values) for ax in axes) + (len(wavenumber),)
        self.data = None  # Populated by compute()
        
        # Interpolators (one per axis except wavenumber)
        self.interpolators = {}
    
    def _validate_constraint(self):
        """
        Verify that the RI constraint is satisfied:
        - All sweep axes correspond to available ARIA data for this source
        - No mixing of sources
        """
        for axis_name in self.ri_constraint.sweep_axes:
            if axis_name not in self.ri_constraint.available_coverage:
                raise ValueError(
                    f"Axis '{axis_name}' requested but not available from "
                    f"{self.ri_constraint.source} for {self.ri_constraint.substance}"
                )
        return True
    
    @classmethod
    def from_aria_database(
        cls,
        substance: str,
        source: str,
        axes: list[LUTAxis],
        wavenumber: np.ndarray,
        quantity: str = 'MEC',
    ):
        """
        Construct LUT by loading RIs from ARIA database.
        
        Args:
            substance: 'Sulphuric acid', 'HNO3', etc.
            source: 'Myhre 2003', 'Biermann 2000', 'Curtis 2005', etc.
            axes: List of LUTAxis defining sweep grid
            wavenumber: Spectral grid
            quantity: 'MEC', 'MSC', 'MAC'
        
        Returns:
            MultiDimLUT instance with data populated
        """
        # Load metadata from aria_source_completeness.json
        constraint = RIConstraint(
            substance=substance,
            source=source,
            sweep_axes=[ax.name for ax in axes if not ax.fixed_for_ri_load],
        )
        
        lut = cls(axes, constraint, wavenumber, quantity)
        lut._validate_constraint()
        lut.compute()  # Populate lut.data by loading RIs
        return lut
```

---

## Audit-Driven Configuration

### H₂SO₄ 2-D LUT: r0 × (composition × temperature)

```python
# From aria_source_completeness.json:
h2so4_config = {
    'substance': 'Sulphuric acid',
    'canonical_source': 'Myhre 2003',
    'available_coverage': {
        'composition': [38, 48, 58, 65, 72],  # wt%
        'temperature_K': [203, 208, 213, ..., 298],  # 14 points
    },
    'file_count': 32,
}

axes = [
    LUTAxis('r0', values=np.logspace(-1, 1, 20), units='µm', log_spacing=True,
            fixed_for_ri_load=True),
    LUTAxis('composition', values=np.array([10, 20, 30, 40, 50, 60, 70, 80]), units='wt%',
            fixed_for_ri_load=False),
    LUTAxis('temperature_K', values=np.arange(200, 301, 10), units='K',
            fixed_for_ri_load=False),
]

lut_h2so4 = MultiDimLUT.from_aria_database(
    substance='Sulphuric acid',
    source='Myhre 2003',  # ← SINGLE SOURCE ENFORCED
    axes=axes,
    wavenumber=wn,
)

spec = lut_h2so4.forward(r0_um=1.5, composition=65.0, temperature_K=250)
```

### HNO₃ 2-D LUT: r0 × (composition × temperature)

```python
hno3_config = {
    'substance': 'HNO3',
    'canonical_source': 'Biermann 2000',
    'available_coverage': {
        'composition': [0, 10, 20, 30, 40],  # wt%, 7 values
        'temperature_K': [213, 233, 253, 273, 293],  # 7 points
    },
    'file_count': 29,
}

lut_hno3 = MultiDimLUT.from_aria_database(
    substance='HNO3',
    source='Biermann 2000',  # ← SINGLE SOURCE ENFORCED
    axes=[...],
    wavenumber=wn,
)
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1)
- [ ] Create `aria_metadata_registry.json` from audit
- [ ] Implement `LUTAxis` dataclass with validation
- [ ] Implement `RIConstraint` dataclass
- [ ] Implement `MultiDimLUT.__init__()` and constraint validation

### Phase 2: Database Integration (Week 2)
- [ ] Implement ARIA RI loader (queries by substance, source, composition, T)
- [ ] Implement `MultiDimLUT.compute()` (grid loop + RI loading + optical property computation)
- [ ] Test on H₂SO₄ with known audit results

### Phase 3: Interpolation & Persistence (Week 3)
- [ ] Implement N-D interpolation using `RegularGridInterpolator`
- [ ] Implement `MultiDimLUT.forward()`
- [ ] NetCDF persistence (extend existing format for N-D arrays)

### Phase 4: Validation & Expansion (Week 4)
- [ ] Test on HNO₃ (Biermann 2000)
- [ ] Add H₂O-ice and H₂O-liquid LUTs
- [ ] Comprehensive testing across all sources

---

## Source Consistency Enforcement

**Immutable constraint:** Once created, a `MultiDimLUT` is locked to a single source. Cannot be changed without recreating the LUT.

```python
# This works:
lut = MultiDimLUT.from_aria_database('Sulphuric acid', 'Myhre 2003', axes, wn)

# This would be prevented at construction time:
# (User specifies source explicitly; mixing is not possible)
```

The `RIConstraint` is **documented and auditable**: any LUT can be traced back to its source study via `lut.ri_constraint.source`.

---

## Key Advantage

By enforcing single-source consistency **at the LUT level**, we guarantee:
✓ Physically consistent interpolations (no measurement artifacts)
✓ Traceable provenance (audit trail from LUT back to ARIA study)
✓ Clear error messages if user requests out-of-range parameters
✓ Freedom to add different sources later (create separate LUT with different source)

---

