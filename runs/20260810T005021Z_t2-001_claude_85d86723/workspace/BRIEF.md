# Low-sidelobe steered beam with 2-bit phase control and element failures

Task id: `t2-001`  (tier 2)

Point a 16x16 half-wavelength planar array at (theta, phi) = (27, 10) degrees at 10 GHz using phase-only control with 2-bit phase shifters (four states, 90 degrees apart). Thirteen elements listed below are dead: the evaluator forces their weights to zero regardless of the submission. Meet all requirements simultaneously. Rounding the ideal steering phases onto the 2-bit grid points the beam correctly but leaves the sidelobes about 3 dB above the limit, so a compliant design has to do better than direct quantization.

## Design context

```yaml
context:
  array:
    type: rectangular
    nx: 16
    ny: 16
    dx_wl: 0.5
    dy_wl: 0.5
    frequency_ghz: 10.0
  element:
    model: cos_q
    q: 1.0
  phase_bits: 2
  target:
    theta_deg: 27.0
    phi_deg: 10.0
  failed_elements: [7, 23, 42, 58, 77, 96, 113, 128, 150, 171, 199, 220, 245]
  exclusion_radius_deg: 8.0
```

## What to submit

Write your design to **`submission.npz`** in this directory, format `npz`.

NumPy .npz file containing a complex array named 'weights' of shape (256,), ordered to match phased_array.create_rectangular_array(16, 16, ...) element ordering. Unit amplitude on all working elements; phases on the 2-bit grid.

## Requirements

Your submission must satisfy every requirement below. Each is scored by
deterministic code that recomputes the physics from your submitted file.

| requirement | metric | limit |
|---|---|---|
| steering | `peak_direction_error_deg` | <= 1.5 |
| sidelobes | `peak_sidelobe_level_db` | <= -14.0 |
| directivity | `directivity_dbi` | >= 25.5 |
| phase-quantization | `phase_grid_error_deg` | <= 0.1 |
| phase-only-control | `amplitude_error` | <= 0.001 |

## How scoring works

- The evaluator applies the hardware constraints and any element failures listed
  in the design context itself. Do not pre-apply failures to your weights;
  submit the weights you would program into working hardware.
- Metrics are computed from your file alone. Nothing you write in prose is scored.
- You may use any method. `numpy` and the `phased_array` package are installed.

Full machine-readable spec: `task.yaml` in this directory.
