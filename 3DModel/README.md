# Drone Wind-Vector Simulator

An interactive Three.js wind simulator that compares an independent identified
physics baseline with a trained machine-learning wind estimator.

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

## Run with the trained model

The Python model stays outside the browser because `wind_model.joblib` is 5.2 GB.
Start the prediction service in a second terminal:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-ml.txt
WIND_MODEL_PATH="$HOME/Downloads/wind_model.joblib" uvicorn ml_server:app --reload
```

Then run `npm run dev` in the first terminal and enable **Use trained ML wind
model**. The first prediction can take several minutes and substantial memory
while the artifact loads; subsequent predictions reuse the loaded model.

## Independent physics baseline

The physics baseline follows the identified, small-perturbation state-space
approach in González-Rocha et al., *Wind Profiling in the Lower Atmosphere from
Wind-Induced Perturbations to Multirotor UAS* (Sensors 2020, 20, 1341):
https://doi.org/10.3390/s20051341

It uses attitude, ground velocity, linear acceleration, and angular rate. The
checked-in coefficients were fitted on non-test flights only. The physics and ML
models do not consume each other's predictions. `physics_test_predictions.csv`
and `test_predictions.csv` contain results for the same 54,548 samples from the
same 42 held-out flights.

To reproduce the physics fit after changing the raw data:

```bash
python ../fit_physics_baseline.py \
  --flights /path/to/flights_primary.csv
```

Start `ml_server.py` to use measured replay and the comparison endpoint. In the
browser, enable **Replay measured test samples**. The three arrows are measured
(green), physics (orange), and ML (blue). Reported vector MAE includes both
speed and direction. The identified physics model is intended for hover and
steady ascent; aggressive flight is outside its validity envelope.

## Production build

```bash
npm run build
npm run preview
```

Vite writes the finished static site to the `dist` folder.
