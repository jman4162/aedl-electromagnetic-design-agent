import design as d
import itertools
import warnings
warnings.filterwarnings("ignore")

candidates = []

nxny_options = [
    (64, 32), (64, 48), (64, 64), (80, 48), (80, 64), (96, 48), (96, 64),
]
tx_power_options = [4.0, 5.0, 6.0, 8.0, 10.0]
pa_class_options = ["B", "C"]
prf_options = [1000.0, 1500.0, 2000.0, 2500.0]
n_pulses_options = [16, 24, 32]
bits_options = [4]
sll_options = [-30.0]

results = []
for (nx, ny), tx_power, pa_class, prf, n_pulses, bits, sll in itertools.product(
    nxny_options, tx_power_options, pa_class_options, prf_options, n_pulses_options, bits_options, sll_options
):
    try:
        r = d.evaluate_design(
            nx=nx, ny=ny, dx=0.55, dy=0.55, taper_type="taylor", taper_sll_db=sll, phase_bits=bits,
            tx_power_w_per_elem=tx_power, pa_class=pa_class, prf_hz=prf, n_pulses=n_pulses,
            digitization_level="subarray", adc_enob=12.0, subarray_nx=8, subarray_ny=8,
        )
    except Exception as e:
        continue
    r["nx"] = nx; r["ny"] = ny; r["tx_power"] = tx_power; r["pa_class"] = pa_class
    r["prf"] = prf; r["n_pulses"] = n_pulses; r["bits"] = bits; r["sll"] = sll
    results.append(r)

def feasible(r):
    return (not r["nan_hit"] and r["worst_case_pd"] >= 0.72 and r["worst_case_timeline_occupancy"] <= 0.97
            and r["array_availability"] >= 0.996 and r["worst_case_pattern_sll_db"] <= -20.3
            and r["grating_margin_lambda"] >= 0.02 and r["prime_power_w"] <= 5300.0
            and r["unit_cost_usd"] <= 620000.0)

feas = [r for r in results if feasible(r)]
print(f"Total {len(results)}, feasible {len(feas)}")
feas.sort(key=lambda r: r["unit_cost_usd"])
for r in feas[:15]:
    print(r["nx"], r["ny"], r["tx_power"], r["pa_class"], r["prf"], r["n_pulses"], r["bits"], r["sll"],
          "| pd=%.3f occ=%.3f avail=%.4f sll=%.2f grate=%.3f pw=%.0f cost=%.0f" % (
              r["worst_case_pd"], r["worst_case_timeline_occupancy"], r["array_availability"],
              r["worst_case_pattern_sll_db"], r["grating_margin_lambda"], r["prime_power_w"], r["unit_cost_usd"]))

if not feas:
    print("--- no feasible design; showing best by pd+occ+power ---")
    results.sort(key=lambda r: (-(r["worst_case_pd"]), r["worst_case_timeline_occupancy"], r["prime_power_w"]))
    for r in results[:15]:
        print(r["nx"], r["ny"], r["tx_power"], r["pa_class"], r["prf"], r["n_pulses"], r["bits"], r["sll"],
              "| pd=%.3f occ=%.3f avail=%.4f sll=%.2f grate=%.3f pw=%.0f cost=%.0f nan=%s" % (
                  r["worst_case_pd"], r["worst_case_timeline_occupancy"], r["array_availability"],
                  r["worst_case_pattern_sll_db"], r["grating_margin_lambda"], r["prime_power_w"], r["unit_cost_usd"], r["nan_hit"]))
