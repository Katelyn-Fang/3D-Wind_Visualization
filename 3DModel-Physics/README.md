# Physics-Only Drone Wind Simulator

This application contains two independent physics baselines and never loads or
calls the trained ML model.

## Default: identified state-space baseline

The default option follows the small-perturbation system-identification method
described by González-Rocha et al., *Wind Profiling in the Lower Atmosphere from
Wind-Induced Perturbations to Multirotor UAS* (Sensors 2020, 20, 1341):
https://doi.org/10.3390/s20051341

It uses attitude, velocity, acceleration, and angular-rate disturbances. Its
coefficients were fitted on 167 flights and evaluated independently on 42 other
flights (54,548 samples). The held-out 2D vector MAE is 2.39 m/s. It is intended
for hover and steady ascent near the identified equilibrium condition.

The error card in the simulator reports aggregate held-out measurements—not an
error for the manually positioned on-screen drone, because that interactive
state has no real anemometer measurement to use as ground truth.

This is a calibrated, model-based baseline, not a first-principles aerodynamic
model: measured training flights identify its coefficients, but no trained-ML
prediction is used anywhere in its calculation.

## Optional: inverse-drag demonstration

This Three.js + Vite variant answers a deliberately hypothetical question:

> Given the new position chosen by the user, what wind could have pushed the
> drone through the observed motion?

This option does not use measured flight data, a prescribed ambient wind, or a
user-selected wind direction. It remains separate from the existing `3DModel`
application.

### Inverse-drag model

Each animation frame derives velocity and acceleration from the dragged drone
position. The model then:

1. Computes the net force with `F_net = m a`.
2. Assumes the rotors provide enough thrust to balance gravity at the selected
   pitch and roll.
3. Solves for the unexplained aerodynamic force:

   ```text
   F_aero = m a - F_thrust - F_gravity
   ```

4. Inverts the quadratic drag equation:

   ```text
   |V_relative| = sqrt(2 |F_aero| / (rho Cd A))
   V_wind = V_drone + V_relative
   ```

5. Expands that inferred vector into a display field using a potential-flow
   sphere approximation near the drone and a widening Gaussian wake downstream.

The constants are in `src/windModel.js`. They are plausible demonstration
values, not calibration values for a specific aircraft. The inverse problem is
not unique, so this option must not be reported as the accuracy baseline.

## Run locally

```bash
npm install
npm run dev
```

Open the URL printed by Vite, then drag the drone across the grid.

## Numeric motion mode

Enable **Use numeric motion inputs** to replace mouse dragging with a scripted
relative movement. Enter `x`, `y`, and `z` offsets in inches, attitude offsets
in degrees, and a duration in seconds. The simulator animates a minimum-jerk
trajectory from the current pose to the requested pose.

Because the trajectory supplies analytic velocity and acceleration, the
inverse model responds predictably to the inputs: doubling the distance doubles
the trajectory's velocity and acceleration, while doubling the duration halves
velocity and reduces acceleration to one quarter. The strongest forward wind
estimate is held after arrival so the result can be inspected.

## Validate

```bash
npm test
npm run build
```
