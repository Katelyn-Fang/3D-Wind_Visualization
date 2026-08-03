import baseline from "./physics_baseline.json" with { type: "json" };

const toRadians = (degrees) => (Number(degrees) || 0) * Math.PI / 180;

/**
 * Identified small-perturbation physics baseline based on the state-space
 * approach in Gonzalez-Rocha et al., Sensors 2020, 20, 1341.
 *
 * The coefficients were fitted on 167 non-test flights. No ML prediction is
 * used as an input, correction, feature, or initial condition.
 */
export function inferIdentifiedWind(telemetry = {}) {
  const roll = toRadians(telemetry.rollDegrees);
  const pitch = toRadians(telemetry.pitchDegrees);
  const yaw = toRadians(telemetry.yawDegrees);
  const eastVelocity = Number(telemetry.velocityX) || 0;
  const northVelocity = Number(telemetry.velocityZ) || 0;
  const bodyForwardVelocity =
    Math.cos(yaw) * eastVelocity + Math.sin(yaw) * northVelocity;
  const bodyRightVelocity =
    -Math.sin(yaw) * eastVelocity + Math.cos(yaw) * northVelocity;

  // Simulator axes are x=east, y=up, z=north. The source flight-data axes
  // use x/y horizontally and z vertically, so map them explicitly here.
  const features = [
    roll,
    pitch,
    bodyForwardVelocity,
    bodyRightVelocity,
    Number(telemetry.velocityY) || 0,
    Number(telemetry.accelerationX) || 0,
    Number(telemetry.accelerationZ) || 0,
    Number(telemetry.accelerationY) || 0,
    Number(telemetry.angularRateX) || 0,
    Number(telemetry.angularRateY) || 0,
    Number(telemetry.angularRateZ) || 0,
    roll * roll + pitch * pitch,
  ];

  const bodyWind = [...baseline.intercept_body_wind_mps];
  for (let index = 0; index < features.length; index += 1) {
    const standardized =
      (features[index] - baseline.feature_mean[index]) /
      baseline.feature_scale[index];
    bodyWind[0] += standardized * baseline.coefficients_body_wind[index][0];
    bodyWind[1] += standardized * baseline.coefficients_body_wind[index][1];
  }

  const east = Math.cos(yaw) * bodyWind[0] - Math.sin(yaw) * bodyWind[1];
  const north = Math.sin(yaw) * bodyWind[0] + Math.cos(yaw) * bodyWind[1];
  const speed = Math.hypot(east, north);
  const insideEnvelope =
    Math.abs(roll) <= Math.PI / 6 &&
    Math.abs(pitch) <= Math.PI / 6 &&
    Math.hypot(eastVelocity, northVelocity) <= 2.5;

  return {
    x: east,
    y: 0,
    z: north,
    speed,
    confidence: insideEnvelope ? 100 : 35,
    validity: insideEnvelope
      ? "Within hover/steady-ascent envelope"
      : "Outside the identified model's flight envelope",
  };
}

export const IDENTIFIED_BASELINE = baseline;
