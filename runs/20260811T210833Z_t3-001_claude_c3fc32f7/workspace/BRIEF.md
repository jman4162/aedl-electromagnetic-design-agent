# 28 GHz LEO SATCOM terminal architecture under power and cost ceilings

Task id: `t3-001`  (tier 3)

Design the architecture of a Ka-band (28 GHz) transmit phased-array ground terminal for a LEO uplink. The terminal must close the link worst-case over an evaluation envelope withheld from this brief (scan angles across the field of regard, rain and correlated sky noise up to the stated climate bounds, and random element-failure draws) while staying under a prime-power ceiling and a unit-cost ceiling priced from the parts table below. Brute-forcing element count violates the ceilings before it closes the worst-case link; a compliant design has to work the aperture-size / power-per-element / taper / quantization trade.

## Design context

```yaml
context:
  mission: leo_uplink_terminal
  link:
    frequency_ghz: 28.0
    bandwidth_mhz: 50.0
    required_snr_db: 6.0
    satellite_rx_gain_db: 38.0
    satellite_noise_figure_db: 2.5
    duty_cycle: 1.0
  nominal_scenario:
    elevation_deg: 45.0
    slant_range_km: 815.0
    scan_angle_deg: 30.0
    rain_rate_mmh: 0.0
    sky_noise_temp_k: 150.0
  evaluation_envelope_bounds:
    scan_angle_deg_max: 60.0
    elevation_deg_min: 20.0
    slant_range_km_max: 1390.0
    rain_rate_mmh_max: 2.5
    sky_noise_temp_k_max: 275.0
    element_failure_rate: 0.02
  constraints:
    prime_power_ceiling_w: 450.0
    unit_cost_ceiling_usd: 45000.0
  parts:
    element_base_cost_usd: 55.0
    pa_classes:
      A: {pa_efficiency: 0.15, cost_adder_usd_per_elem: 15.0}
      B: {pa_efficiency: 0.25, cost_adder_usd_per_elem: 45.0}
      C: {pa_efficiency: 0.35, cost_adder_usd_per_elem: 110.0}
    phase_shifter:
      3: 5.0
      4: 12.0
      5: 25.0
      6: 50.0
    adc_cost_usd_per_channel: 40.0
    rx_power_w_per_elem: 0.15
    feed_loss_db: 1.5
    tx_power_w_per_elem_max: 2.0
  element:
    model: cos_q
    q: 1.3
```

## What to submit

Write your design to **`architecture.yaml`** in this directory, format `yaml`.

A YAML file named architecture.yaml with exactly these keys and no others. array: {nx, ny, dx_lambda, dy_lambda, taper_type, taper_sll_db, phase_bits} with taper_type in {uniform, taylor, chebyshev} and taper_sll_db negative; rf: {tx_power_w_per_elem, pa_class} with pa_class in {A, B, C} and 0 < tx_power_w_per_elem <= 2.0; digital: {digitization_level, adc_enob, subarray_nx, subarray_ny} with digitization_level in {element, subarray, analog}. nx and ny must tile into power-of-two subarrays (subarray_nx/subarray_ny in {2,4,8,16}, nx divisible by subarray_nx, ny by subarray_ny); phase_bits in {3,4,5,6}. Efficiency, noise figure, losses, and all costs are set by the parts table; submitting any of them is an error.

## Requirements

Your submission must satisfy every requirement below. Each is scored by
deterministic code that recomputes the physics from your submitted file.

| requirement | metric | limit |
|---|---|---|
| link-margin | `worst_case_link_margin_db` | >= 0.0 |
| sidelobes | `worst_case_pattern_sll_db` | <= -16.0 |
| independent-link | `opensatcom_worst_case_margin_db` | >= 0.0 |
| clearsky-agreement | `crosscheck_clearsky_margin_disagreement_db` | <= 1.2 |
| gain-crosscheck | `crosscheck_gain_disagreement_db` | <= 0.5 |
| prime-power | `prime_power_w` | <= 450.0 |
| unit-cost | `unit_cost_usd` | <= 45000.0 |
| grating | `grating_margin_lambda` | >= 0.0 |

## How scoring works

- Margins are scored worst-case over a held-out evaluation envelope inside the declared bounds, including seeded random element failures at the stated rate. Do not design only for the nominal scenario.
- Pattern claims (gain, sidelobes) are recomputed by full pattern integration with phased-array-modeling, and the link is independently recomputed with the opensatcom toolkit. Metrics come from your file alone; nothing you write in prose is scored.
- numpy, phased_array, phased_array_systems, and opensatcom are installed in your environment.

Full machine-readable spec: `task.yaml` in this directory.
