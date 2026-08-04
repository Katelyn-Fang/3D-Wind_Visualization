const baselineResponse = await fetch(
  `${import.meta.env.BASE_URL}data/physics_baseline.json`,
);

if (!baselineResponse.ok) {
  throw new Error(
    `Unable to load physics baseline (${baselineResponse.status}).`,
  );
}

const baseline = await baselineResponse.json();

const radians = (degrees) => (Number(degrees) || 0) * Math.PI / 180;

/**
 * Independent identified linear-dynamics baseline.
 *
 * This is deliberately separate from mlClient.js. It consumes only aircraft
 * state and the coefficients fitted on non-test flights. The output is the
 * horizontal wind vector in east/north coordinates.
 */
export function predictPhysicsWind(telemetry = {}) {
  const roll = radians(telemetry.rollDegrees);
  const pitch = radians(telemetry.pitchDegrees);
  const yaw = radians(telemetry.yawDegrees);
  const vx = Number(telemetry.velocityX) || 0;
  const vz = Number(telemetry.velocityZ) || 0;
  const forwardVelocity = Math.cos(yaw) * vx + Math.sin(yaw) * vz;
  const rightVelocity = -Math.sin(yaw) * vx + Math.cos(yaw) * vz;
  const features = [
    roll,
    pitch,
    forwardVelocity,
    rightVelocity,
    Number(telemetry.velocityY) || 0,
    Number(telemetry.accelerationX) || 0,
    Number(telemetry.accelerationZ) || 0,
    (Number(telemetry.accelerationY) || 0),
    Number(telemetry.angularRateX) || 0,
    Number(telemetry.angularRateY) || 0,
    Number(telemetry.angularRateZ) || 0,
    roll * roll + pitch * pitch,
  ];

  const body = [...baseline.intercept_body_wind_mps];
  for (let row = 0; row < features.length; row += 1) {
    const standardized =
      (features[row] - baseline.feature_mean[row]) /
      baseline.feature_scale[row];
    body[0] += standardized * baseline.coefficients_body_wind[row][0];
    body[1] += standardized * baseline.coefficients_body_wind[row][1];
  }

  const east = Math.cos(yaw) * body[0] - Math.sin(yaw) * body[1];
  const north = Math.sin(yaw) * body[0] + Math.cos(yaw) * body[1];
  const speed = Math.hypot(east, north);
  const nearEquilibrium =
    Math.abs(roll) <= Math.PI / 6 &&
    Math.abs(pitch) <= Math.PI / 6 &&
    Math.hypot(vx, vz) <= 2.5;
  return {
    u: east,
    v: north,
    w: 0,
    speed,
    confidence: nearEquilibrium ? 1 : 0.35,
    validity: nearEquilibrium ? "near equilibrium" : "outside baseline flight envelope",
  };
}

export const physicsBaselineMetadata = baseline;
