# Compositional certification for transmission and distribution planning

Python research code for *Compositional Certification of Reduced Models for Transmission and Distribution Reinforcement*. The experiment compares reduced feeder-response models with full-network AC certification in a finite investment and operating-control catalogue. It uses Chebyshev feeder-response approximations, local root-inclusion certificates, and selective refinement of investment comparisons.

The benchmark assembles public 9- and 30-bus transmission cases with 33- and 69-bus distribution feeders into 111- and 132-bus systems. Load profiles, investment options, controls and scenarios are constructed for the numerical study. Feasibility results are conditional on the modeled controls and balanced AC assumptions. The root-inclusion analysis is in real arithmetic and the software uses guarded binary64 arithmetic.

## Start here

Use Python 3.12 on macOS or Linux. Run these commands at the repository root.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python fetch_cases.py
python smoke.py
```

The smoke check constructs one composite case, trains degree-six feeder responses, certifies a reduced AC solution and checks its loss interval against a full AC solve.

## Reproduce the archived experiment

This repository uses the scientific code and outputs from the November 2026 journal submission archive for the original certification and investment-targeted refinement experiments. Run the following in a working copy because the programs write results into the original experiment directories.

```sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python work/research/normalized_response/full_certificate_planning_study.py
python work/research/normalized_response/repeat_full_certificate_planning.py
python work/research/normalized_response/complete_uncertified_runtime_study.py
python work/research/normalized_response/coarse_precision_study.py
python work/research/control_blocks/validate_feeder_tube.py
python work/revision_v3/paper01/adaptive_study.py --coarse 3 --repeats 3
```

| Research output | Location |
| --- | --- |
| Compositional and full-network comparison | `work/research/normalized_response/full_certificate_planning_repeats_summary.json` |
| Complete noncertified baseline | `work/research/normalized_response/complete_uncertified_runtime_results/summary.json` |
| Feeder derivative verification | `work/research/control_blocks/feeder_derivative_validation.json` |
| Selective refinement results | `work/revision_v3/paper01/adaptive3_nested_c1_summary.json` |
| Frozen synthetic benchmark protocol and reference enumeration | `work/benchmarks/frozen_v1/` |

Compare decisions and numerical bounds using the tolerances in the experiment. Record the machine and software environment when comparing wall-clock measurements.

## Benchmark sources

The download script verifies the exact study versions of `case9.m`, `case33bw.m` and `case69.m` from [MATPOWER 8.1](https://github.com/MATPOWER/matpower/tree/8.1/data). The 30-bus arrays are read from the installed [PYPOWER](https://github.com/rwl/PYPOWER/blob/master/pypower/case30.py) package and match the archived case arrays. The source cases are retrieved separately under their upstream terms. The repository contains synthetic study inputs, numerical results and experiment code.
