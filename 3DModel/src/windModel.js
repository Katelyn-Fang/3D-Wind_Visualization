import * as THREE from "three";

// reusing object avoids creating hundreds of new objects every frame.
const attitudeEuler = new THREE.Euler(
  0,
  0,
  0,
  "YZX"
);

const attitudeQuaternion =
  new THREE.Quaternion();

const downwashDirection =
  new THREE.Vector3(0, -1, 0);

let previousYaw = null;
let previousPitch = null;
let previousRoll = null;

//Placeholder values, can change depending on drone
const dronePhysics = {
  massKg: 1.2,
  gravity: 9.81,
  airDensity: 1.225,
  rotorRadiusMeters: 0.12,
  rotorCount: 4,

  // Placeholder aerodynamic properties.
  dragCoefficient: 1.0,
  referenceAreaM2: 0.08,

}

function calculateTotalThrust(telemetry = {}) {
  const pitchRadians =
    THREE.MathUtils.degToRad(
      telemetry.pitchDegrees ?? 0
    );

  const rollRadians =
    THREE.MathUtils.degToRad(
      telemetry.rollDegrees ?? 0
    );

  // A tilted drone needs additional thrust to maintain altitude.
  const verticalThrustFraction = Math.max(
    0.35,
    Math.cos(pitchRadians) *
      Math.cos(rollRadians)
  );

  const weight =
    dronePhysics.massKg *
    dronePhysics.gravity;

  return weight / verticalThrustFraction;
}

function calculateInducedVelocity(telemetry = {}) {
  const totalThrust = calculateTotalThrust(telemetry);

  const totalRotorArea =
    dronePhysics.rotorCount *
    Math.PI *
    dronePhysics.rotorRadiusMeters ** 2;

  return Math.sqrt(
    totalThrust /
      (
        2 *
        dronePhysics.airDensity *
        totalRotorArea
      )
  );
}

function getDownwashDirection(telemetry) {
  const yaw = telemetry.yawDegrees ?? 0;
  const pitch = telemetry.pitchDegrees ?? 0;
  const roll = telemetry.rollDegrees ?? 0;

  // Only recalculate when the drone's orientation changes.
  if (
    yaw !== previousYaw ||
    pitch !== previousPitch ||
    roll !== previousRoll
  ) {
    attitudeEuler.set(
      THREE.MathUtils.degToRad(roll),

      // This matches the negative yaw rotation used in main.js.
      -THREE.MathUtils.degToRad(yaw),

      THREE.MathUtils.degToRad(pitch),
      "YZX"
    );

    attitudeQuaternion.setFromEuler(
      attitudeEuler
    );

    // Start with straight-down airflow in the drone's
    // local coordinate system, then rotate it into the
    // world coordinate system.
    downwashDirection
      .set(0, -1, 0)
      .applyQuaternion(attitudeQuaternion)
      .normalize();

    previousYaw = yaw;
    previousPitch = pitch;
    previousRoll = roll;
  }

  return downwashDirection;
}

export function predictWind(
  samplePosition,
  dronePosition,
  timeSeconds,
  params,
  telemetry
) {
  //old ambient wind section
  /*const droneOffset = Math.hypot(
    dronePosition.x,
    dronePosition.z
  );

  const ambientSpeed =
    params.baseSpeed +
    params.gradient * droneOffset;

  const ambientAngle = Math.atan2(
    dronePosition.z,
    dronePosition.x
  );
  const ambientWindX = ambientSpeed * Math.cos(ambientAngle);
  const ambientWindZ = ambientSpeed * Math.sin(ambientAngle);*/

  //new ambient wind section
  // Atmospheric wind-shear settings.
  const referenceHeight = 1.0;
  const minimumHeight = 0.25;
  const shearExponent = 0.20;

  // Prevent arrows below or extremely close to the ground
  // from creating invalid or extremely small values.
  const sampleHeight = Math.max(
    samplePosition.y,
    minimumHeight
  );

  // Power-law vertical wind shear.
  //
  // baseSpeed represents the wind speed at referenceHeight.
  const verticalShearSpeed =
    params.baseSpeed *
    Math.pow(
      sampleHeight / referenceHeight,
      shearExponent
    );

  // A controllable horizontal microclimate gradient.
  // This allows dragging the drone away from the center
  // to change the predicted wind speed.
  const horizontalOffset =
    Math.hypot(
      samplePosition.x,
      samplePosition.z
    );

  const horizontalGradient =
    params.gradient * horizontalOffset;

  const ambientSpeed = Math.max(
    0.05,
    verticalShearSpeed +
      horizontalGradient
  );

  // The ambient wind currently blows toward positive x.
  // A wind-direction control will be added later.
  let ambientWindX;
  let ambientWindZ;

  // Read the physics-estimated wind from telemetry.
  const estimatedWindX =
    Number.isFinite(
      telemetry?.estimatedWindX
    )
      ? telemetry.estimatedWindX
      : 0;

  const estimatedWindZ =
    Number.isFinite(
      telemetry?.estimatedWindZ
    )
      ? telemetry.estimatedWindZ
      : 0;

  const estimatedWindSpeed =
    Math.hypot(
      estimatedWindX,
      estimatedWindZ
    );

  // Only use the estimate when it contains a
  // meaningful, finite wind vector.
  const estimateIsUsable =
    params.useEstimatedWind &&
    estimatedWindSpeed > 0.1;

  if (estimateIsUsable) {
    /*
      The wind estimate represents the wind near the
      drone's altitude. Apply wind shear above and
      below the drone.
    */
    const droneReferenceHeight =
      Math.max(
        dronePosition.y,
        minimumHeight
      );

    const estimatedShearScale =
      Math.pow(
        sampleHeight /
          droneReferenceHeight,
        shearExponent
      );

    ambientWindX =
      estimatedWindX *
      estimatedShearScale;

    ambientWindZ =
      estimatedWindZ *
      estimatedShearScale;
  } else {
    // Use the manually selected wind when the
    // physics estimate is disabled or unavailable.
    const windDirectionRadians =
      THREE.MathUtils.degToRad(
        params.windDirectionDegrees ?? 0
      );

    ambientWindX =
      ambientSpeed *
      Math.cos(
        windDirectionRadians
      );

    ambientWindZ =
      ambientSpeed *
      Math.sin(
        windDirectionRadians
      );
  }

  let windX = ambientWindX;
  let windZ = ambientWindZ;
  let windY = 0;

  const dx =
    samplePosition.x - dronePosition.x;

  const dy =
    samplePosition.y - dronePosition.y;

  const dz =
    samplePosition.z - dronePosition.z;

  const distanceSquared =
    dx * dx +
    dy * dy +
    dz * dz;

  const droneInfluence =
    params.turbulence *
    Math.exp(-distanceSquared / 6);


  //create a swirling disturbance around drone  
  windX += -dz * droneInfluence * 0.45;
  windZ += dx * droneInfluence * 0.45;

  //create downward propeller airflow
  //windY -= droneInfluence * 0.35;

  /*const downwash = getDownwashDirection(telemetry);

  //const downwashStrength = droneInfluence * 0.35;
  const inducedVelocity = calculateInducedVelocity(telemetry);

// The momentum-theory result is multiplied by the
// distance falloff already stored in droneInfluence.
  const downwashStrength =
    inducedVelocity *
    droneInfluence *
    0.22;

  windX += downwash.x * downwashStrength;
  windY += downwash.y * downwashStrength;
  windZ += downwash.z * downwashStrength;*/

  //new downwash function
  const downwash = getDownwashDirection(telemetry);

  const inducedVelocity = calculateInducedVelocity(telemetry);

  // Distance from the drone along the downwash direction.
  // Positive means the sample point is downstream of the rotors.
  const axialDistance =
    dx * downwash.x +
    dy * downwash.y +
    dz * downwash.z;

  if (axialDistance > 0) {
    // Find the point on the downwash centerline nearest
    // to this sample position.
    const centerlineX =
      downwash.x * axialDistance;

    const centerlineY =
      downwash.y * axialDistance;

    const centerlineZ =
      downwash.z * axialDistance;

    // Distance sideways from the downwash centerline.
    const radialX = dx - centerlineX;
    const radialY = dy - centerlineY;
    const radialZ = dz - centerlineZ;

    const radialDistanceSquared =
      radialX * radialX +
      radialY * radialY +
      radialZ * radialZ;

    // The rotor wake becomes wider as it travels.
    const wakeRadius =
      0.65 + axialDistance * 0.22;

    // Strongest at the center of the wake and weaker
    // toward its outside edges.
    const radialFalloff = Math.exp(
      -radialDistanceSquared /
        (2 * wakeRadius * wakeRadius)
    );

    // The downwash loses energy farther from the drone.
    const axialFalloff = Math.exp(
      -axialDistance / 6
    );

    // Ideal far-wake speed is approximately twice the
    // induced velocity. The final factor remains a visual
    // calibration value until testing data is available.
    const downwashStrength =
      2 *
      inducedVelocity *
      radialFalloff *
      axialFalloff *
      0.22;

    windX +=
      downwash.x * downwashStrength;

    windY +=
      downwash.y * downwashStrength;

    windZ +=
      downwash.z * downwashStrength;
  }

  //New yaw code, allows wake to follow relative airflow and change gradually
  const droneVelocityX =
    telemetry.velocityX ?? 0;

  const droneVelocityZ =
    telemetry.velocityZ ?? 0;

  // Airflow experienced by the moving drone.
  const relativeFlowX =
    ambientWindX - droneVelocityX;

  const relativeFlowZ =
    ambientWindZ - droneVelocityZ;

  const relativeFlowSpeed =
    Math.hypot(
      relativeFlowX,
      relativeFlowZ
    );

  if (relativeFlowSpeed > 0.05) {
    // Normalize the relative airflow to create a
    // direction vector.
    const flowX =
      relativeFlowX / relativeFlowSpeed;

    const flowZ =
      relativeFlowZ / relativeFlowSpeed;

    // A perpendicular vector used to measure how far
    // the sample is from the wake centerline.
    const rightX = -flowZ;
    const rightZ = flowX;

    // Positive values represent points downstream.
    const distanceDownstream =
      dx * flowX +
      dz * flowZ;

    const sidewaysDistance =
      dx * rightX +
      dz * rightZ;

    if (distanceDownstream > 0) {
      const wakeWidth =
        0.7 +
        distanceDownstream * 0.28;

      const sidewaysFalloff =
        Math.exp(
          -(
            sidewaysDistance *
            sidewaysDistance
          ) /
          (
            2 *
            wakeWidth *
            wakeWidth
          )
        );

      const verticalFalloff =
        Math.exp(
          -(dy * dy) /
          (2 * 1.2 * 1.2)
        );

      const lengthFalloff =
        Math.exp(
          -distanceDownstream / 7
        );

      // Faster relative airflow creates a stronger wake.
      const airflowFactor =
        Math.min(
          0.4 +
            relativeFlowSpeed * 0.18,
          2
        );

      const wakeStrength =
        params.turbulence *
        airflowFactor *
        sidewaysFalloff *
        verticalFalloff *
        lengthFalloff;

      const wakeWobble =
        Math.sin(
          timeSeconds * 4 +
          distanceDownstream * 1.8
        ) * 0.35;

      // Reduce airflow along the wake centerline.
      windX -=
        flowX *
        wakeStrength *
        0.8;

      windZ -=
        flowZ *
        wakeStrength *
        0.8;

      // Add a small turbulent sideways oscillation.
      windX +=
        rightX *
        wakeWobble *
        wakeStrength;

      windZ +=
        rightZ *
        wakeWobble *
        wakeStrength;

      windY -=
        wakeStrength * 0.25;
    }
  }

  // The forward-facing direction of the drone.
  //
  // At 0 degrees:
  // forwardX = 1
  // forwardZ = 0
  //
  // This means the drone faces positive x.
  /*const forwardX = Math.cos(yawRadians);
  const forwardZ = Math.sin(yawRadians);

  // A horizontal direction perpendicular to the drone.
  // This helps calculate how far an arrow is from the
  // centerline of the wake.
  const rightX = -forwardZ;
  const rightZ = forwardX;

  // Determine how far the sample point is behind the drone.
  //
  // A positive value means that the point is behind it.
  const distanceBehind =
    -(dx * forwardX + dz * forwardZ);

  // Determine how far the point is sideways from the wake.
  const sidewaysDistance =
    dx * rightX + dz * rightZ;

  // Only create the wake behind the drone.
  if (distanceBehind > 0) {
    // The wake gradually becomes wider farther behind
    // the aircraft.
    const wakeWidth =
      0.7 + distanceBehind * 0.28;

    // Reduce the effect for arrows far from the wake centerline.
    const sidewaysFalloff = Math.exp(
      -(sidewaysDistance * sidewaysDistance) /
      (2 * wakeWidth * wakeWidth)
    );

    // Reduce the effect for arrows far above or below the drone.
    const wakeHeight = 1.2;

    const verticalFalloff = Math.exp(
      -(dy * dy) /
      (2 * wakeHeight * wakeHeight)
    );

    // The wake becomes weaker farther behind the drone.
    const lengthFalloff = Math.exp(
      -distanceBehind / 7
    );

    const wakeStrength =
      params.turbulence *
      sidewaysFalloff *
      verticalFalloff *
      lengthFalloff;

    // Create a side-to-side oscillation so the wake
    // does not look perfectly straight.
    const wakeWobble =
      Math.sin(
        timeSeconds * 4 +
        distanceBehind * 1.8
      ) * 0.35;

    // push airflow backward along the drone's heading.
    windX -= forwardX * wakeStrength * 0.8;
    windZ -= forwardZ * wakeStrength * 0.8;

    // Add a small sideways turbulent motion.
    windX += rightX * wakeWobble * wakeStrength;
    windZ += rightZ * wakeWobble * wakeStrength;

    // Add some downward airflow behind the drone.
    windY -= wakeStrength * 0.25;
  }*/

  if (params.animateWind) {
    const gust =
      Math.sin(
        timeSeconds * 2 +
        samplePosition.x * 0.8 +
        samplePosition.z * 0.6
      ) * 0.15;

    windX += gust;
    windZ += gust * 0.4;
  }

  return new THREE.Vector3(
    windX,
    windY,
    windZ
  );
}

export function estimateAerodynamicForce(
  telemetry
) {
  const mass = dronePhysics.massKg;
  const gravity = dronePhysics.gravity;

  const accelerationX =
    telemetry.accelerationX ?? 0;

  const accelerationY =
    telemetry.accelerationY ?? 0;

  const accelerationZ =
    telemetry.accelerationZ ?? 0;

  // F_net = m a
  const netForceX =
    mass * accelerationX;

  const netForceY =
    mass * accelerationY;

  const netForceZ =
    mass * accelerationZ;

  // Downwash points opposite the thrust direction.
  const downwash =
    getDownwashDirection(telemetry);

  const totalThrust =
    calculateTotalThrust(telemetry);

  const thrustForceX =
    -downwash.x * totalThrust;

  const thrustForceY =
    -downwash.y * totalThrust;

  const thrustForceZ =
    -downwash.z * totalThrust;

  // Gravity points downward.
  const gravityForceX = 0;
  const gravityForceY =
    -mass * gravity;
  const gravityForceZ = 0;

  // F_aero = ma - F_thrust - F_gravity
  const aerodynamicForceX =
    netForceX -
    thrustForceX -
    gravityForceX;

  const aerodynamicForceY =
    netForceY -
    thrustForceY -
    gravityForceY;

  const aerodynamicForceZ =
    netForceZ -
    thrustForceZ -
    gravityForceZ;

  const magnitude = Math.hypot(
    aerodynamicForceX,
    aerodynamicForceY,
    aerodynamicForceZ
  );

  return {
    x: aerodynamicForceX,
    y: aerodynamicForceY,
    z: aerodynamicForceZ,
    magnitude,
    totalThrust,
  };
}

export function estimateWindFromDrag(
  telemetry,
  aerodynamicForce
) {
  const forceX =
    aerodynamicForce.x ?? 0;

  const forceZ =
    aerodynamicForce.z ?? 0;

  // Start with horizontal wind only.
  const horizontalForceMagnitude =
    Math.hypot(forceX, forceZ);

  const droneVelocityX =
    telemetry.velocityX ?? 0;

  const droneVelocityZ =
    telemetry.velocityZ ?? 0;

  // Avoid dividing by nearly zero.
  if (horizontalForceMagnitude < 0.02) {
    return {
      x: 0,
      y: 0,
      z: 0,
      speed: 0,
      relativeAirSpeed: 0,
    };
  }

  const dragArea =
    dronePhysics.dragCoefficient *
    dronePhysics.referenceAreaM2;

  // Invert the drag equation:
  // Vrel = sqrt(2F / rho Cd A)
  const relativeAirSpeed = Math.sqrt(
    (
      2 *
      horizontalForceMagnitude
    ) /
    (
      dronePhysics.airDensity *
      dragArea
    )
  );

  // The aerodynamic force points in the direction
  // the air pushes the drone.
  const forceDirectionX =
    forceX /
    horizontalForceMagnitude;

  const forceDirectionZ =
    forceZ /
    horizontalForceMagnitude;

  /*
    Drag force is opposite the drone's velocity
    relative to the air.

    Therefore:
    relativeVelocity = -forceDirection * speed
  */
  const relativeVelocityX =
    -forceDirectionX *
    relativeAirSpeed;

  const relativeVelocityZ =
    -forceDirectionZ *
    relativeAirSpeed;

  // Vwind = Vdrone - Vrelative
  let estimatedWindX =
    droneVelocityX -
    relativeVelocityX;

  let estimatedWindZ =
    droneVelocityZ -
    relativeVelocityZ;

  // Prevent unstable simulated mouse motion from
  // producing extremely large results.
  const maximumWindSpeed = 20;

  const estimatedSpeed =
    Math.hypot(
      estimatedWindX,
      estimatedWindZ
    );

  if (estimatedSpeed > maximumWindSpeed) {
    const scale =
      maximumWindSpeed /
      estimatedSpeed;

    estimatedWindX *= scale;
    estimatedWindZ *= scale;
  }

  return {
    x: estimatedWindX,
    y: 0,
    z: estimatedWindZ,

    speed: Math.hypot(
      estimatedWindX,
      estimatedWindZ
    ),

    relativeAirSpeed,
  };
}