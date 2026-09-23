# Architecture and tuning

## Data flow

```text
Config + seeded RNG → generated MJCF → MjModel + live MjData
                                           ↓
Timed state machine → Cartesian targets → scratch-data pose IK
                                           ↓
                            bounded position actuator commands
                                           ↓
                          mj_step → mj_forward → measurements
                                           ↓
                      force / geometry / timeout checks → logs
```

The nominal controller is a deterministic timed state machine, not a motion planner.
Contact-force limits apply throughout the trial. Success is only evaluated during
the hold phase. A trial with insufficient alignment never becomes successful just
because the time sequence has completed.

## Frames and units

World Z points up; distances are meters, joint angles radians, and time seconds.
The socket-center site is in the middle of the bore. Its local +Z is the insertion
axis; the top entrance is at local Z = +0.025 m. The peg-tip site is at the lower
end of the 100 mm peg. Both desired site orientations are world identity.

Metrics transform the peg tip into the socket frame. Depth is `0.025 - local_z`,
lateral error is the XY norm, and axis error is the angle between the sites' +Z
directions. CSV/HTML convert these to millimeters/degrees.

## Kinematics and dynamics

`PoseIK.solve` builds a six-dimensional pose error from Cartesian position and a
quaternion-log orientation error. The rotational rows are scaled by 0.35 so position
and orientation objectives are balanced. The update is
`dq = J.T @ solve(J @ J.T + 1e-4 * I, error)`.
Each update is capped at 0.15 rad and respects joint limits. The solver uses up to
12 iterations per control update; initialization permits 250 iterations and checks
position reachability. It does not solve collision avoidance or global reachability.

Dynamics use a 2 ms timestep and `implicitfast`. Position servos use kp=1200,
kv=90, and a ±180 N·m force range. The original arms are abstract mechanisms with
primitive inertia estimates, not hardware-calibrated models. Ideal per-body gravity
compensation isolates the coordination task from gravity-control tuning.

Peg/socket normal and friction forces are obtained from `mj_contactForce` for each
relevant contact pair. The scalar monitoring signal sums the translational-force
magnitudes. Solver penetration is possible, as in any compliant contact model.

## Extending the task safely

Change robot geometry in `scene.py`, task geometry/timing in `simulation.py`, and IK
parameters in `control.py`. Keep the peg-tip and socket-center sites consistent
with the physical geometry. If dimensions change, update the success envelope too.

For compliant control, replace the right arm's position-command strategy with an
operational-space impedance controller; preserve the force-abort and trace logging.
For grasping, make both parts free bodies and model actual fingers before claiming
grasp stability. Add collision planning before enabling all link collisions.

The `AssemblySimulation` Python API exposes `model`, `data`, `reset()`, `step()`,
`measurements()`, and `result()`. `step()` advances one physics tick. The public
`done` property is true at success, force abort, instability, or timeout.

```python
from dual_arm_assembly import AssemblySimulation, Config

sim = AssemblySimulation(Config(seed=7, randomize=True))
while not sim.done:
    sim.step()
print(sim.result())
```
