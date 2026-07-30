# Drone Wind-Vector Simulator

A starter Three.js + Vite project for an interactive drone simulation.

## Run it locally

1. Install Node.js.
2. Open this folder in VS Code.
3. Open **Terminal > New Terminal**.
4. Run:

   ```bash
   npm install
   npm run dev
   ```

5. Open the local URL printed in the terminal.

## What the starter does

- Draws a simple 3D quadcopter.
- Lets the user drag it over a horizontal plane.
- Displays a moving lattice of wind vectors around the drone.
- Changes wind speed and direction based on distance from the center.
- Adds a local rotor-disturbance demonstration.
- Reports the local `u`, `v`, and `w` wind components.

## Important coordinate mapping

Three.js uses:

- `x`: east/west
- `y`: up/down
- `z`: north/south

Standard wind notation commonly uses:

- `u`: east/west
- `v`: north/south
- `w`: up/down

Therefore map model outputs as:

```text
Three.js vector = (u, w, v)
```

## Replace the equation with a model

Open `src/main.js` and find:

```js
function predictWind(samplePosition, dronePosition, timeSeconds) {
  // ...
}
```

Keep the rest of the visualization unchanged. Replace only the contents of
that function, or make it read the latest prediction returned by a backend.

## Production build

```bash
npm run build
npm run preview
```

Vite writes the finished static site to the `dist` folder.
