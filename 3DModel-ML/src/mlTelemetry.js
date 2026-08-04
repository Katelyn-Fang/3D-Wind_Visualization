const METERS_PER_LATITUDE_DEGREE = 111_320;

export const MODEL_COORDINATE_ORIGIN = Object.freeze({
  longitude: -79.7826006916051,
  latitude: 40.45836389714015,
  altitudeM: 267.1844462604522,
});

function finiteNumber(value, label) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    throw new Error(`${label} must be a finite number.`);
  }
  return number;
}

export function sceneToModelCoordinates(
  position,
  origin = MODEL_COORDINATE_ORIGIN,
) {
  const eastMeters = finiteNumber(position.x, "Scene x");
  const upMeters = finiteNumber(position.y, "Scene y");
  const northMeters = finiteNumber(position.z, "Scene z");
  const latitudeRadians = origin.latitude * Math.PI / 180;

  return {
    x: origin.longitude +
      eastMeters /
        (METERS_PER_LATITUDE_DEGREE * Math.cos(latitudeRadians)),
    y: origin.latitude + northMeters / METERS_PER_LATITUDE_DEGREE,
    z: origin.altitudeM + upMeters,
  };
}

export function createMlTelemetrySample({
  sessionId,
  elapsedSeconds,
  position,
  attitudeDegrees,
  batteryVoltage = 15.2,
  batteryCurrent = 4.0,
}) {
  const modelPosition = sceneToModelCoordinates(position);
  const toRadians = (degrees, label) =>
    finiteNumber(degrees, label) * Math.PI / 180;

  return {
    session_id: String(sessionId),
    elapsed_s: finiteNumber(elapsedSeconds, "Elapsed time"),
    x: modelPosition.x,
    y: modelPosition.y,
    z: modelPosition.z,
    roll: toRadians(attitudeDegrees.roll, "Roll"),
    pitch: toRadians(attitudeDegrees.pitch, "Pitch"),
    yaw: toRadians(attitudeDegrees.yaw, "Yaw"),
    battery_v: finiteNumber(batteryVoltage, "Battery voltage"),
    battery_c: finiteNumber(batteryCurrent, "Battery current"),
  };
}
