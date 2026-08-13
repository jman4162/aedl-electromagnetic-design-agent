# X-band maritime search radar architecture under frame-time, power, and availability floors

Task id: `t3-002`  (tier 3)

Design the architecture of an X-band (9.5 GHz) naval search phased array detecting a low-altitude Swerling-1 target over sea clutter. The design must achieve the required worst-case detection probability over an evaluation envelope withheld from this brief (target range and RCS, scan angle, sea state, and random element-failure draws), sweep its assigned search volume within the frame-time budget, and still meet its pattern spec at end of mission after component failures. Aperture size, per-element power, waveform (PRF, pulses per dwell), taper, and quantization pull against each other: a bigger aperture narrows the beam and multiplies beam positions until the frame budget breaks; fewer pulses lose integration gain and force per-element power up, heating the junction and eroding end-of-mission availability.

## Design context

```yaml
context:
  mission: xband_maritime_search
  radar:
    frequency_ghz: 9.5
    bandwidth_mhz: 5.0
    pfa: 1.0e-6
    pd_required: 0.7
    noise_figure_db: 3.5
    antenna_height_m: 30.0
    cfar_type: CA
    cfar_ref_cells: 16
    cfar_guard_cells: 2
  search:
    az_extent_deg: 60.0
    el_extent_deg: 15.0
    frame_time_ms: 3000.0
    beam_overhead_us: 50.0
  nominal_scenario:
    range_km: 13.0
    rcs_dbsm: 6.0
    scan_deg: 25.0
    sea_state: 3
    target_height_m: 5.0
  evaluation_envelope_bounds:
    # Clutter-limited and noise-limited points have different reach:
    # points with sea clutter (sea_state > 0) occur only within
    # clutter_range_km_max and at rcs >= rcs_dbsm_min_clutter; clear-sea
    # points extend to range_km_max at rcs >= rcs_dbsm_min.
    range_km_max: 26.0
    rcs_dbsm_min: 2.0
    clutter_range_km_max: 14.0
    rcs_dbsm_min_clutter: 6.0
    scan_deg_max: 40.0
    sea_state_max: 3
    element_failure_rate: 0.02
  waveform_bounds:
    prf_hz_min: 500.0
    prf_hz_max: 3000.0
    n_pulses_min: 1
    n_pulses_max: 64
  constraints:
    prime_power_ceiling_w: 5500.0
    unit_cost_ceiling_usd: 650000.0
  reliability:
    component_mtbfs:
      lna: 500000.0
      phase_shifter: 2000000.0
      attenuator: 3000000.0
      switch: 1000000.0
      control_asic: 1000000.0
    thermal_resistance_c_per_w: 40.0
    ambient_temp_c: 40.0
    mttr_hours: 24.0
    mission_hours: 4380.0
  parts:
    element_base_cost_usd: 90.0
    pa_classes:
      A: {pa_efficiency: 0.20, pa_mtbf_hours: 150000.0, cost_adder_usd_per_elem: 40.0}
      B: {pa_efficiency: 0.30, pa_mtbf_hours: 300000.0, cost_adder_usd_per_elem: 120.0}
      C: {pa_efficiency: 0.40, pa_mtbf_hours: 500000.0, cost_adder_usd_per_elem: 260.0}
    phase_shifter:
      3: 5.0
      4: 12.0
      5: 25.0
      6: 50.0
    adc_cost_usd_per_channel: 40.0
    rx_power_w_per_elem: 0.25
    feed_loss_db: 1.5
    tx_power_w_per_elem_max: 12.0
    tau_us: 33.0
  element:
    model: cos_q
    q: 1.3
```

## What to submit

Write your design to **`architecture.yaml`** in this directory, format `yaml`.

A YAML file named architecture.yaml with exactly these keys and no others. array: {nx, ny, dx_lambda, dy_lambda, taper_type, taper_sll_db, phase_bits} with taper_type in {uniform, taylor, chebyshev} and taper_sll_db negative; rf: {tx_power_w_per_elem, pa_class} with pa_class in {A, B, C} and 0 < tx_power_w_per_elem <= 12.0; waveform: {prf_hz, n_pulses} with prf_hz in [500, 3000] and n_pulses in [1, 64] — duty cycle is computed from the pinned 33 us pulse width and your PRF, and submitting it is an error; digital: {digitization_level, adc_enob, subarray_nx, subarray_ny} with digitization_level in {element, subarray, analog}. nx and ny must tile into power-of-two subarrays; phase_bits in {3,4,5,6}. Detection statistics (Swerling model, Pd/Pfa, CFAR), pulse width, efficiency, noise figure, losses, MTBFs, and all costs are set by the task; submitting any of them is an error.

## Requirements

Your submission must satisfy every requirement below. Each is scored by
deterministic code that recomputes the physics from your submitted file.

| requirement | metric | limit |
|---|---|---|
| detection | `worst_case_pd` | >= 0.7 |
| frame-time | `worst_case_timeline_occupancy` | <= 1.0 |
| availability | `array_availability` | >= 0.995 |
| sidelobes | `worst_case_pattern_sll_db` | <= -20.0 |
| pd-crosscheck | `crosscheck_pd_disagreement` | <= 0.12 |
| gain-crosscheck | `crosscheck_gain_disagreement_db` | <= 0.5 |
| prime-power | `prime_power_w` | <= 5500.0 |
| unit-cost | `unit_cost_usd` | <= 650000.0 |
| grating | `grating_margin_lambda` | >= 0.0 |

## How scoring works

- Detection probability is scored worst-case over a held-out evaluation envelope inside the declared bounds (range, RCS, scan angle, sea state), including seeded random element failures at the stated rate. Do not design only for the nominal scenario.
- Pattern claims (gain, sidelobes) are recomputed by full pattern integration with phased-array-modeling, and the detection probability at the binding envelope point is independently recomputed by Monte Carlo simulation of a Swerling-1 target through an actual CA-CFAR. Metrics come from your file alone; nothing you write in prose is scored.
- The search timeline is scored from your beamwidths at the worst scan: frame time must fit the stated budget. End-of-mission availability uses the pinned component MTBFs with junction temperature fed forward from your design's dissipated power.
- numpy, phased_array, and phased_array_systems are installed in your environment.

Full machine-readable spec: `task.yaml` in this directory.
