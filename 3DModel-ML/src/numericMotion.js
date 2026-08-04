export const INCH_TO_METERS = 0.0254;

function finiteNumber(value, label) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    throw new Error(`${label} must be a finite number.`);
  }
  return number;
}

function copyVector(vector, label) {
  return {
    x: finiteNumber(vector.x, `${label} x`),
    y: finiteNumber(vector.y, `${label} y`),
    z: finiteNumber(vector.z, `${label} z`),
  };
}

function copyAttitude(attitude, label) {
  return {
    roll: finiteNumber(attitude.roll, `${label} roll`),
    pitch: finiteNumber(attitude.pitch, `${label} pitch`),
    yaw: finiteNumber(attitude.yaw, `${label} yaw`),
  };
}

export function createNumericMotionPlan({
  startPosition,
  offsetInches,
  startAttitude,
  attitudeOffsetDegrees,
  durationSeconds,
}) {
  const start = copyVector(startPosition, "Start position");
  const inches = copyVector(offsetInches, "Position offset");
  const attitude = copyAttitude(startAttitude, "Start attitude");
  const attitudeOffset = copyAttitude(
    attitudeOffsetDegrees,
    "Attitude offset",
  );
  const duration = finiteNumber(durationSeconds, "Duration");

  if (duration < 0.1 || duration > 60) {
    throw new Error("Duration must be between 0.1 and 60 seconds.");
  }

  const offsetMeters = {
    x: inches.x * INCH_TO_METERS,
    y: inches.y * INCH_TO_METERS,
    z: inches.z * INCH_TO_METERS,
  };

  return {
    durationSeconds: duration,
    startPosition: start,
    offsetMeters,
    targetPosition: {
      x: start.x + offsetMeters.x,
      y: start.y + offsetMeters.y,
      z: start.z + offsetMeters.z,
    },
    startAttitude: attitude,
    attitudeOffsetDegrees: attitudeOffset,
    targetAttitude: {
      roll: attitude.roll + attitudeOffset.roll,
      pitch: attitude.pitch + attitudeOffset.pitch,
      yaw: attitude.yaw + attitudeOffset.yaw,
    },
    distanceMeters: Math.hypot(
      offsetMeters.x,
      offsetMeters.y,
      offsetMeters.z,
    ),
  };
}

/**
 * Fifth-order minimum-jerk interpolation. Position, velocity, and
 * acceleration are all continuous, and the vehicle starts and ends at rest.
 */
export function sampleNumericMotion(plan, elapsedSeconds) {
  const elapsed = Math.max(0, finiteNumber(elapsedSeconds, "Elapsed time"));
  const duration = plan.durationSeconds;
  const progress = Math.min(elapsed / duration, 1);
  const t2 = progress ** 2;
  const t3 = progress ** 3;
  const t4 = progress ** 4;
  const t5 = progress ** 5;

  const positionScale = 10 * t3 - 15 * t4 + 6 * t5;
  const velocityScale =
    (30 * t2 - 60 * t3 + 30 * t4) / duration;
  const accelerationScale =
    (60 * progress - 180 * t2 + 120 * t3) / (duration ** 2);

  const position = {};
  const velocity = {};
  const acceleration = {};

  for (const axis of ["x", "y", "z"]) {
    const offset = plan.offsetMeters[axis];
    position[axis] = plan.startPosition[axis] + offset * positionScale;
    velocity[axis] = offset * velocityScale;
    acceleration[axis] = offset * accelerationScale;
  }

  const attitude = {};
  const angularVelocity = {};
  const angularAcceleration = {};

  for (const axis of ["roll", "pitch", "yaw"]) {
    const offset = plan.attitudeOffsetDegrees[axis];
    attitude[axis] = plan.startAttitude[axis] + offset * positionScale;
    angularVelocity[axis] = offset * velocityScale;
    angularAcceleration[axis] = offset * accelerationScale;
  }

  return {
    progress,
    elapsedSeconds: Math.min(elapsed, duration),
    done: progress >= 1,
    position,
    velocity,
    acceleration,
    attitude,
    angularVelocity,
    angularAcceleration,
  };
}

/**
 * Keep the strongest wind that has acted in the requested travel direction.
 * The second half of a minimum-jerk trajectory contains braking acceleration;
 * attributing that braking to wind would reverse the displayed field. Instead,
 * the vehicle controller is assumed to brake while the driving wind remains
 * continuous.
 */
export function retainStrongestForwardWind({
  currentWind,
  travelDirection,
  previousWind = { x: 0, y: 0, z: 0 },
  previousScore = -Infinity,
  minimumSpeed = 0.02,
}) {
  const current = copyVector(currentWind, "Current wind");
  const travel = copyVector(travelDirection, "Travel direction");
  const previous = copyVector(previousWind, "Previous wind");
  const travelMagnitude = Math.hypot(travel.x, travel.y, travel.z);
  const currentSpeed = Math.hypot(current.x, current.y, current.z);
  const previousSpeed = Math.hypot(previous.x, previous.y, previous.z);

  const forwardScore = travelMagnitude > 1e-12
    ? (
        current.x * travel.x +
        current.y * travel.y +
        current.z * travel.z
      ) / travelMagnitude
    : currentSpeed;
  const pointsForward = travelMagnitude <= 1e-12 || forwardScore > 0;
  const updated =
    currentSpeed >= minimumSpeed &&
    pointsForward &&
    forwardScore > previousScore;

  if (updated) {
    return { wind: current, score: forwardScore, updated: true };
  }
  if (previousSpeed >= minimumSpeed) {
    return { wind: previous, score: previousScore, updated: false };
  }
  return { wind: current, score: previousScore, updated: false };
}
