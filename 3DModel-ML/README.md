# Extra Trees Drone Wind Simulator

`3DModel-ML` is an isolated copy of the original `3DModel` application with a
numeric motion mode added. The existing drag controls, measured replay, and
physics comparison remain available when numeric mode is unchecked.

## What numeric mode does

Enable **Use numeric motion inputs** and enter:

- x, y, and z position offsets in inches;
- roll, pitch, and yaw offsets in degrees; and
- travel time in seconds.

The drone follows a fifth-order minimum-jerk trajectory that begins and ends at
rest. Choose one wind model before starting:

- **Physics** uses the same inverse-drag equations, strongest-forward-wind
  retention, and wake visualization as `3DModel-Physics`.
- **Extra Trees ML** first sends a time-indexed copy of the complete trajectory
  to the prediction API under a fresh session ID. Animation begins only after
  those predictions are ready, and the cached vectors are interpolated at the
  current animation time. This removes stale or missing vectors caused by
  asynchronous responses arriving after the drone has moved on.

The ML vector field uses the model response directly as `(u, w, v)` in Three.js
coordinates and holds the exact final cached prediction at arrival. Ambient
direction, disturbance, and horizontal-gradient controls remain editable; they
control the demo/reference field and are not substituted for either model.

## Run the frontend

PowerShell users can avoid the `npm.ps1` execution-policy restriction by using
`npm.cmd`:

```powershell
Set-Location .\3DModel-ML
npm.cmd install
npm.cmd run dev
```

Open the local URL printed by Vite.

## Use the repository's Extra Trees pathway

`3DModel-ML/ml_server.py` is a thin entry point to the canonical implementation
in `3DModel/ml_server.py`. Both applications therefore use the same telemetry
schema, `wind_core.py` feature engineering, model artifact, direction
conversion, and prediction code. The only additional endpoint is the batched
trajectory prediction used by numerical motion.

The checked-in repository does **not** contain `wind_model.joblib`; the JSON and
CSV prediction files are recorded outputs and cannot predict a new user-entered
trajectory. Obtain the exact team artifact used by `3DModel` and point
`WIND_MODEL_PATH` to it. Do not point this variable at a separately trained
artifact unless you intentionally want different model behavior.

Start the shared API in a second terminal:

```powershell
Set-Location .\3DModel-ML
python -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r requirements-ml.txt
$env:WIND_MODEL_PATH = "C:\path\to\the-team-extra-trees\wind_model.joblib"
& ".\.venv\Scripts\python.exe" -m uvicorn ml_server:app --host 127.0.0.1 --port 8000
```

Avoid `--reload` for the large artifact because process reloads can consume
additional memory. The first live prediction loads the artifact and can take
several minutes; later predictions reuse it.

After the first prediction, verify which artifact and pathway are active:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

The response reports `inference_server` as `3DModel/ml_server.py`, the artifact
path, training script/data metadata when available, model family, and feature
count. This makes it easier to catch an accidentally selected personal artifact.

Without the artifact, the API can still serve **Replay measured test samples**
from the checked-in validation CSV files.

## Validate

```powershell
npm.cmd test
npm.cmd run build
```

The tests cover inch conversion, exact target pose, minimum-jerk kinematics,
duration scaling, scene-to-model coordinate mapping, complete Extra Trees
trajectory preparation, and frame-time vector interpolation.
