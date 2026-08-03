# Inverse-Physics Drone Wind Simulator

This Three.js + Vite variant answers a deliberately hypothetical question:

> Given the new position chosen by the user, what wind could have pushed the
> drone through the observed motion?

It does not use the repository's trained model, measured flight data, a
prescribed ambient wind, or a user-selected wind direction. The existing
`3DModel` folder remains unchanged.

## Model

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
not unique, so the result should be interpreted as one physically motivated
explanation of the user's motion.

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
