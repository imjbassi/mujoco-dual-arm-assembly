# Initial validation

Locally verified on Windows, Python 3.14, MuJoCo 3.14.0, and NumPy 2.5.3.
These measurements describe this simplified simulation, not hardware accuracy.

| Check | Outcome |
|---|---|
| Nominal assembly | Success at 9.76 simulated seconds |
| Final nominal insertion depth | 39.992 mm |
| Final nominal lateral error | Less than 0.001 mm with exact state and idealized tooling |
| Randomized workspace seeds 0–19 | 20/20 successful |
| Deliberate 12 mm target bias | Force abort at 7.962 s, peak 36.016 N |
| Deliberate 3.5 mm target bias | Timeout; rejected by success criteria |
| Repeat after reset | Identical result for the same seed |
| Python tests | 16 passed |
| Ruff lint / formatting | Passed |
| Wheel and source distribution | Built successfully |
| Passive desktop viewer | Opened, synchronized, and closed successfully |
| Offscreen recording | Overview and close-up GIF/PNG generated and inspected |

The tests include a regression check that the nominal arm posture stays above the
table and the insertion wrist stays above the peg. This geometric check is not a
general collision detector. Only peg/socket collisions participate in dynamics.

Run these commands to reproduce the experiment logs:

```bash
python -m dual_arm_assembly run --output runs/baseline
python -m dual_arm_assembly benchmark --seed 0 --trials 20
python -m dual_arm_assembly run --offset-mm 12 --output runs/jam
```

The final command intentionally exits with code 1. Each trial writes its config,
metrics, and dependency versions to `result.json`, plus `trace.csv` and `report.html`.
The benchmark's `summary.json` aggregates all trial results.

CI is configured for Python 3.11/3.13 on Linux and Windows. The local checks above
do not constitute evidence that the remote CI matrix has completed.
