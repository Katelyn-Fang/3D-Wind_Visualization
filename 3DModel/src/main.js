import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
import "./style.css";
import { fetchMlWind } from "./mlClient.js";
import {
  predictWind, 
  estimateAerodynamicForce,
  estimateWindFromDrag
} from "./windModel.js";

import {
  loadWindDataset,
  getFlight,
  findNearestWindPoint,
  createWindArrow,
  updateWindArrow,
  addFlightVectorField,
} from "./windData.js";

const canvas = document.querySelector("#scene");
const app = document.querySelector("#app");
const estimatedWindValue = document.querySelector("#estimated-wind-value");
const estimatedWindComponents = document.querySelector("#estimated-wind-components"); 
const referenceWindValue = document.querySelector("#reference-wind-value");
const windErrorValue = document.querySelector("#wind-error-value");

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x07111d, 19, 39);

const camera = new THREE.PerspectiveCamera(
  48,
  window.innerWidth / window.innerHeight,
  0.1,
  100,
);
camera.position.set(12, 11, 15);

const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x07111d, 1);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const orbitControls = new OrbitControls(camera, renderer.domElement);
orbitControls.enableDamping = true;
orbitControls.target.set(0, 1.2, 0);
orbitControls.maxPolarAngle = Math.PI * 0.47;
orbitControls.minDistance = 8;
orbitControls.maxDistance = 32;

scene.add(new THREE.HemisphereLight(0xccecff, 0x18202b, 2.2));

const sun = new THREE.DirectionalLight(0xffffff, 2.7);
sun.position.set(7, 14, 9);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
scene.add(sun);

const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(24, 24),
  new THREE.MeshStandardMaterial({
    color: 0x0a1724,
    roughness: 0.92,
    metalness: 0.05,
  }),
);
floor.rotation.x = -Math.PI / 2;
floor.receiveShadow = true;
scene.add(floor);

const grid = new THREE.GridHelper(24, 24, 0x4bb5e8, 0x213d52);
grid.material.opacity = 0.43;
grid.material.transparent = true;
scene.add(grid);

const centerMarker = new THREE.Mesh(
  new THREE.RingGeometry(0.35, 0.48, 48),
  new THREE.MeshBasicMaterial({
    color: 0x73c7ff,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.9,
  }),
);
centerMarker.rotation.x = -Math.PI / 2;
centerMarker.position.y = 0.012;
scene.add(centerMarker);

const params = {
  baseSpeed: 2.5,
  windDirectionDegrees: 0,
  turbulence: 1.2,
  gradient: 0.18,
  animateWind: true,
  useEstimatedWind: false,
};

const telemetry = {
  yawDegrees: 0,
  pitchDegrees: 0,
  rollDegrees: 0,
  altitude: 1.35,

  velocityX: 0,
  velocityY: 0,
  velocityZ: 0,
  speed: 0,

  accelerationX: 0,
  accelerationY: 0,
  accelerationZ: 0,
  accelerationMagnitude: 0,

  aerodynamicForceX: 0,
  aerodynamicForceY: 0,
  aerodynamicForceZ: 0,
  aerodynamicForceMagnitude: 0,

  estimatedWindX: 0,
  estimatedWindY: 0,
  estimatedWindZ: 0,
  estimatedWindSpeed: 0,

  estimatedWindConfidence: 0,

  referenceWindX: 0,
  referenceWindY: 0,
  referenceWindZ: 0,
  referenceWindSpeed: 0,

  windEstimateError: 0,
  windEstimateErrorPercent: 0,
};
/*
if (estimatedWindValue) {
  estimatedWindValue.textContent =
    `${telemetry.estimatedWindSpeed.toFixed(
      2
    )} m/s`;
}

if (estimatedWindComponents) {
  estimatedWindComponents.textContent =
    `u: ${telemetry.estimatedWindX.toFixed(2)}, ` +
    `v: ${telemetry.estimatedWindZ.toFixed(2)}, ` +
    `w: ${telemetry.estimatedWindY.toFixed(2)} m/s`;
}*/

let droneHeight = 1.35;
const DRAG_LIMIT = 9.5;

function createDrone() {
  const group = new THREE.Group();
  group.name = "drone";

  const bodyMaterial = new THREE.MeshStandardMaterial({
    color: 0xe9f6ff,
    roughness: 0.28,
    metalness: 0.5,
  });
  const darkMaterial = new THREE.MeshStandardMaterial({
    color: 0x182431,
    roughness: 0.45,
    metalness: 0.42,
  });
  const accentMaterial = new THREE.MeshStandardMaterial({
    color: 0x4ec5ff,
    emissive: 0x0b5273,
    emissiveIntensity: 0.55,
    roughness: 0.3,
  });

  const body = new THREE.Mesh(
    new THREE.BoxGeometry(1.55, 0.42, 1.1),
    bodyMaterial,
  );
  body.scale.set(1, 0.78, 1);
  body.castShadow = true;
  group.add(body);

  const nose = new THREE.Mesh(
    new THREE.ConeGeometry(0.35, 0.75, 20),
    accentMaterial,
  );
  nose.rotation.z = -Math.PI / 2;
  //nose.rotation.y = Math.PI / 4;
  nose.position.x = 1.02;
  nose.castShadow = true;
  group.add(nose);

  const armGeometry = new THREE.BoxGeometry(2.9, 0.12, 0.16);
  for (const angle of [Math.PI / 4, -Math.PI / 4]) {
    const arm = new THREE.Mesh(armGeometry, darkMaterial);
    arm.rotation.y = angle;
    arm.castShadow = true;
    group.add(arm);
  }

  const rotorPositions = [
    [1.04, 0.13, 1.04],
    [1.04, 0.13, -1.04],
    [-1.04, 0.13, 1.04],
    [-1.04, 0.13, -1.04],
  ];

  for (const [x, y, z] of rotorPositions) {
    const motor = new THREE.Mesh(
      new THREE.CylinderGeometry(0.16, 0.2, 0.26, 20),
      darkMaterial,
    );
    motor.position.set(x, y, z);
    motor.castShadow = true;
    group.add(motor);

    const rotor = new THREE.Mesh(
      new THREE.CylinderGeometry(0.55, 0.55, 0.025, 34),
      new THREE.MeshStandardMaterial({
        color: 0x7fd7ff,
        transparent: true,
        opacity: 0.38,
        roughness: 0.25,
      }),
    );
    rotor.position.set(x, y + 0.18, z);
    rotor.castShadow = true;
    rotor.userData.isRotor = true;
    group.add(rotor);
  }

  const landingBarGeometry = new THREE.BoxGeometry(1.55, 0.08, 0.08);
  for (const z of [-0.46, 0.46]) {
    const landingBar = new THREE.Mesh(landingBarGeometry, darkMaterial);
    landingBar.position.set(-0.08, -0.42, z);
    group.add(landingBar);
  }

  group.position.set(0, droneHeight, 0);
  return group;
}

const drone = createDrone();
drone.rotation.order = "YZX"; // yaw, pitch, roll
drone.position.set(0, droneHeight, 0);
scene.add(drone);

const droneMeshes = [];
drone.traverse((object) => {
  if (object.isMesh) {
    droneMeshes.push(object);
  }
});

const arrowGroup = new THREE.Group();
scene.add(arrowGroup);

const arrowSamples = [];
//const sampleOffsets = [-3, -1.5, 0, 1.5, 3];
//make wake more visible
const sampleOffsets = [-3, -2, -1, 0, 1, 2, 3];
//const heightOffsets = [-0.85, 0, 0.85];
//creates more arrows
const heightOffsets = [-3, -2, -1, 0, 1];

for (const x of sampleOffsets) {
  for (const y of heightOffsets) {
    for (const z of sampleOffsets) {
      const distanceFromDrone = Math.hypot(x, y, z);
      if (distanceFromDrone < 1.25) {
        continue;
      }

      const origin = new THREE.Vector3();
      const arrow = new THREE.ArrowHelper(
        new THREE.Vector3(1, 0, 0),
        origin,
        1,
        0x52b9f5,
        0.28,
        0.13,
      );
      arrowGroup.add(arrow);
      arrowSamples.push({
        localOffset: new THREE.Vector3(x, y, z),
        arrow,
      });
    }
  }
}

const localWindMarker = new THREE.ArrowHelper(
  new THREE.Vector3(1, 0, 0),
  new THREE.Vector3(),
  2,
  0xffffff,
  0.42,
  0.2,
);
scene.add(localWindMarker);

/**
 * Demonstration wind-field function.
 *
 * Inputs:
 * - samplePosition: world position where the wind is being evaluated
 * - dronePosition: current drone position
 * - timeSeconds: animation time
 *
 * Output:
 * - THREE.Vector3(x, y, z), where y is the vertical component.
 *
 * Replace this function later with:
 * 1. interpolation from measured data,
 * 2. a request to a Python prediction API, or
 * 3. an ONNX model running in the browser.
 */
/*function predictWind(samplePosition, dronePosition, timeSeconds) {
  // Distance of the drone from the center of the grid.
  const droneOffset = Math.hypot(
    dronePosition.x,
    dronePosition.z
  );

  // The overall wind becomes stronger farther from the center.
  const ambientSpeed =
    params.baseSpeed +
    params.gradient * droneOffset;

  // The overall wind points from the center toward the drone.
  const ambientAngle = Math.atan2(
    dronePosition.z,
    dronePosition.x
  );

  let windX =
    ambientSpeed * Math.cos(ambientAngle);

  let windZ =
    ambientSpeed * Math.sin(ambientAngle);

  let windY = 0;

  // Position of this arrow relative to the drone.
  const dx =
    samplePosition.x - dronePosition.x;

  const dy =
    samplePosition.y - dronePosition.y;

  const dz =
    samplePosition.z - dronePosition.z;

  // Distance from this arrow to the drone.
  const distanceSquared =
    dx * dx +
    dy * dy +
    dz * dz;

  // The drone affects nearby arrows more than distant arrows.
  const droneInfluence =
    params.turbulence *
    Math.exp(-distanceSquared / 6);

  // Create a swirling disturbance around the drone.
  windX += -dz * droneInfluence * 0.45;
  windZ += dx * droneInfluence * 0.45;

  // Create simplified downward propeller airflow.
  windY -= droneInfluence * 0.35;

  // Optional changing gusts.
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
}*/

  //Test function to make sure wind vectors act normally


const color = new THREE.Color();
const direction = new THREE.Vector3();
const samplePosition = new THREE.Vector3();

function speedColor(speed) {
  const normalized = THREE.MathUtils.clamp((speed - 0.5) / 7, 0, 1);
  // Blue -> cyan/green -> yellow/red.
  color.setHSL(0.61 - normalized * 0.61, 0.88, 0.56);
  return color;
}

function updateWindField(timeSeconds) {
  const estimatedWindOpacity =
    params.useEstimatedWind
      ? THREE.MathUtils.clamp(
          0.2 + telemetry.estimatedWindConfidence / 125,
          0.2,
          1
        )
      : 1;
  //arrow loop
  for (const sample of arrowSamples) {
    samplePosition.copy(drone.position).add(sample.localOffset);

    const wind = predictWind(
      samplePosition, 
      drone.position, 
      timeSeconds,
      params,
      telemetry
    );
    const speed = wind.length();

    direction.copy(wind).normalize();
    sample.arrow.position.copy(samplePosition);
    sample.arrow.setDirection(direction);
    sample.arrow.setLength(
      THREE.MathUtils.clamp(0.32 + speed * 0.34, 0.55, 3.2),
      0.3,
      0.14,
    );
    sample.arrow.setColor(speedColor(speed));
    
    sample.arrow.line.material.transparent =
      estimatedWindOpacity < 1;

    sample.arrow.cone.material.transparent =
      estimatedWindOpacity < 1;

    sample.arrow.line.material.opacity =
      estimatedWindOpacity;

    sample.arrow.cone.material.opacity =
      estimatedWindOpacity;
  }

  const localWind = predictWind(
    drone.position, 
    drone.position, 
    timeSeconds,
    params,
    telemetry
  );
  const localSpeed = localWind.length();

  localWindMarker.position.copy(drone.position);
  localWindMarker.position.y += 1.05;
  localWindMarker.setDirection(localWind.clone().normalize());
  localWindMarker.setLength(
    THREE.MathUtils.clamp(0.55 + localSpeed * 0.45, 1, 4),
    0.42,
    0.2,
  );
  localWindMarker.setColor(speedColor(localSpeed));

  updateReadout(localWind);
}

//document selectors for telemetry readouts
const xValue = document.querySelector("#x-value");
const zValue = document.querySelector("#z-value");
const offsetValue = document.querySelector("#offset-value");
const speedValue = document.querySelector("#speed-value");
const uValue = document.querySelector("#u-value");
const vValue = document.querySelector("#v-value");
const wValue = document.querySelector("#w-value");
const droneSpeedValue = document.querySelector("#drone-speed-value");
const droneAccelerationValue = document.querySelector("#drone-acceleration-value");
const windConfidenceValue = document.querySelector("#wind-confidence-value");
const windConfidenceBar = document.querySelector("#wind-confidence-bar");


function updateReadout(localWind) {
  const offset = Math.hypot(drone.position.x, drone.position.z);

  xValue.textContent = `${drone.position.x.toFixed(2)} m`;
  zValue.textContent = `${drone.position.z.toFixed(2)} m`;
  offsetValue.textContent = `${offset.toFixed(2)} m`;
  speedValue.textContent = `${localWind.length().toFixed(2)} m/s`;

  // Standard wind notation: u=east, v=north, w=up.
  // Three.js uses y as up, so the mapping is:
  // u -> Three x, v -> Three z, w -> Three y.
  uValue.textContent = `u: ${localWind.x.toFixed(2)} m/s`;
  vValue.textContent = `v: ${localWind.z.toFixed(2)} m/s`;
  wValue.textContent = `w: ${localWind.y.toFixed(2)} m/s`;
}

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const dragPlane = new THREE.Plane(
  new THREE.Vector3(0, 1, 0),
  -droneHeight,
);
const dragPoint = new THREE.Vector3();
const dragOffset = new THREE.Vector3();
let dragging = false;
let hovered = false;

//reusable vectors
const previousDronePosition = drone.position.clone();
const measuredVelocity = new THREE.Vector3();
const smoothedVelocity = new THREE.Vector3();
const previousVelocity = new THREE.Vector3();
const measuredAcceleration = new THREE.Vector3();
const smoothedAcceleration = new THREE.Vector3();
const smoothedEstimatedWind = new THREE.Vector3();
const referenceWindVector = new THREE.Vector3();
let latestMlWind = null;
let mlRequestInFlight = false;
let lastMlRequestSeconds = -Infinity;
const mlSessionId = `vite-${Date.now()}`;

async function updateMlPrediction(timeSeconds) {
  if (!params.useEstimatedWind || mlRequestInFlight || timeSeconds - lastMlRequestSeconds < 0.5) {
    return;
  }
  mlRequestInFlight = true;
  lastMlRequestSeconds = timeSeconds;
  try {
    latestMlWind = await fetchMlWind({
      session_id: mlSessionId,
      elapsed_s: timeSeconds,
      x: drone.position.x,
      y: drone.position.z,
      z: drone.position.y,
      roll: THREE.MathUtils.degToRad(telemetry.rollDegrees),
      pitch: THREE.MathUtils.degToRad(telemetry.pitchDegrees),
      yaw: THREE.MathUtils.degToRad(telemetry.yawDegrees),
      battery_v: 15.2,
      battery_c: 4.0,
    });
  } catch (error) {
    console.warn(error.message);
    latestMlWind = null;
  } finally {
    mlRequestInFlight = false;
  }
}



function calculateWindConfidence(
  rawWindEstimate,
  aerodynamicForce
) {
  const rawX = rawWindEstimate.x ?? 0;
  const rawZ = rawWindEstimate.z ?? 0;

  const rawSpeed =
    Math.hypot(rawX, rawZ);

  const forceMagnitude =
    aerodynamicForce.magnitude ?? 0;

  const valuesAreValid =
    Number.isFinite(rawX) &&
    Number.isFinite(rawZ) &&
    Number.isFinite(forceMagnitude);

  if (!valuesAreValid || rawSpeed < 0.1) {
    return 0;
  }

  /*
    Stronger aerodynamic force gives the estimator
    more information to work with.
  */
  const forceSignal =
    THREE.MathUtils.clamp(
      forceMagnitude / 4,
      0,
      1
    );

  /*
    Compare the raw estimate with the smoothed
    estimate. Large differences indicate instability.
  */
  const estimateDisagreement =
    Math.hypot(
      rawX - smoothedEstimatedWind.x,
      rawZ - smoothedEstimatedWind.z
    );

  const stability =
    1 -
    THREE.MathUtils.clamp(
      estimateDisagreement / 5,
      0,
      1
    );

  /*
    Extremely high acceleration is likely caused by
    sudden mouse movement rather than realistic flight.
  */
  const accelerationNoise =
    THREE.MathUtils.clamp(
      telemetry.accelerationMagnitude / 40,
      0,
      1
    );

  const accelerationQuality =
    1 - accelerationNoise;

  const confidence =
    100 *
    (
      forceSignal * 0.45 +
      stability * 0.45 +
      accelerationQuality * 0.10
    );

  return THREE.MathUtils.clamp(
    confidence,
    0,
    100
  );
}

function updateDroneVelocity(deltaTime) {
  if (deltaTime <= 0) {
    return;
  }

  // Calculate velocity from the drone's change in position.
  if (dragging) {
    measuredVelocity
      .copy(drone.position)
      .sub(previousDronePosition)
      .divideScalar(deltaTime);

    // Prevent extremely fast mouse movement from creating
    // unrealistic velocity values.
    const maximumDragSpeed = 12;

    if (
      measuredVelocity.length() >
      maximumDragSpeed
    ) {
      measuredVelocity.setLength(
        maximumDragSpeed
      );
    }
  } else {
    measuredVelocity.set(0, 0, 0);
  }

  // Smooth the measured velocity.
  const velocitySmoothing =
    1 - Math.exp(-8 * deltaTime);

  smoothedVelocity.lerp(
    measuredVelocity,
    velocitySmoothing
  );

  // Gradually reduce velocity when dragging stops.
  if (!dragging) {
    const velocityDecay =
      Math.exp(-6 * deltaTime);

    smoothedVelocity.multiplyScalar(
      velocityDecay
    );
  }

  // Acceleration is the change in velocity divided
  // by the change in time.
  measuredAcceleration
    .copy(smoothedVelocity)
    .sub(previousVelocity)
    .divideScalar(deltaTime);

  // Prevent mouse movement from creating unrealistic
  // acceleration spikes.
  const maximumAcceleration = 40;

  if (
    measuredAcceleration.length() >
    maximumAcceleration
  ) {
    measuredAcceleration.setLength(
      maximumAcceleration
    );
  }

  // Smooth acceleration because it is naturally noisier
  // than position or velocity.
  const accelerationSmoothing =
    1 - Math.exp(-5 * deltaTime);

  smoothedAcceleration.lerp(
    measuredAcceleration,
    accelerationSmoothing
  );

  telemetry.velocityX =
    smoothedVelocity.x;

  telemetry.velocityY =
    smoothedVelocity.y;

  telemetry.velocityZ =
    smoothedVelocity.z;

  telemetry.speed =
    smoothedVelocity.length();

  telemetry.accelerationX =
    smoothedAcceleration.x;

  telemetry.accelerationY =
    smoothedAcceleration.y;

  telemetry.accelerationZ =
    smoothedAcceleration.z;

  telemetry.accelerationMagnitude =
    smoothedAcceleration.length();

  previousDronePosition.copy(
    drone.position
  );

  previousVelocity.copy(
    smoothedVelocity
  );

  if (droneSpeedValue) {
  droneSpeedValue.textContent =
    `${telemetry.speed.toFixed(2)} m/s`;
  }

  if (droneAccelerationValue) {
    droneAccelerationValue.textContent =
      `${telemetry.accelerationMagnitude.toFixed(2)} m/s²`;
  }
}
/*
const aerodynamicForce =
  estimateAerodynamicForce(telemetry);

const rawWindEstimate =
  estimateWindFromDrag(
    telemetry,
    aerodynamicForce
  );

// Smooth the estimate because acceleration and
// force calculations can be noisy.
const windSmoothing =
  1 - Math.exp(-3 * deltaTime);

smoothedEstimatedWind.lerp(
  new THREE.Vector3(
    rawWindEstimate.x,
    rawWindEstimate.y,
    rawWindEstimate.z
  ),
  windSmoothing
);

telemetry.estimatedWindX =
  smoothedEstimatedWind.x;

telemetry.estimatedWindY =
  smoothedEstimatedWind.y;

telemetry.estimatedWindZ =
  smoothedEstimatedWind.z;

telemetry.estimatedWindSpeed =
  smoothedEstimatedWind.length();

telemetry.aerodynamicForceX =
  aerodynamicForce.x;

telemetry.aerodynamicForceY =
  aerodynamicForce.y;

telemetry.aerodynamicForceZ =
  aerodynamicForce.z;

telemetry.aerodynamicForceMagnitude =
  aerodynamicForce.magnitude;*/

function updatePointer(event) {
  const bounds = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1;
  pointer.y = -((event.clientY - bounds.top) / bounds.height) * 2 + 1;
}

function isDroneHit() {
  raycaster.setFromCamera(pointer, camera);
  return raycaster.intersectObjects(droneMeshes, false).length > 0;
}

renderer.domElement.addEventListener("pointerdown", (event) => {
  updatePointer(event);

  if (!isDroneHit()) {
    return;
  }

  dragging = true;
  orbitControls.enabled = false;
  renderer.domElement.classList.add("dragging");
  renderer.domElement.setPointerCapture(event.pointerId);

  raycaster.setFromCamera(pointer, camera);
  if (raycaster.ray.intersectPlane(dragPlane, dragPoint)) {
    dragOffset.copy(drone.position).sub(dragPoint);
  }
});

renderer.domElement.addEventListener("pointermove", (event) => {
  updatePointer(event);

  if (!dragging) {
    hovered = isDroneHit();
    renderer.domElement.style.cursor = hovered ? "grab" : "default";
    return;
  }

  raycaster.setFromCamera(pointer, camera);
  if (!raycaster.ray.intersectPlane(dragPlane, dragPoint)) {
    return;
  }

  const nextX = THREE.MathUtils.clamp(
    dragPoint.x + dragOffset.x,
    -DRAG_LIMIT,
    DRAG_LIMIT,
  );
  const nextZ = THREE.MathUtils.clamp(
    dragPoint.z + dragOffset.z,
    -DRAG_LIMIT,
    DRAG_LIMIT,
  );

  drone.position.set(nextX, droneHeight, nextZ);
});

function stopDragging(event) {
  if (!dragging) {
    return;
  }

  dragging = false;
  orbitControls.enabled = true;
  renderer.domElement.classList.remove("dragging");

  if (event.pointerId !== undefined) {
    try {
      renderer.domElement.releasePointerCapture(event.pointerId);
    } catch {
      // Pointer capture may already be released by the browser.
    }
  }
}

renderer.domElement.addEventListener("pointerup", stopDragging);
renderer.domElement.addEventListener("pointercancel", stopDragging);

const altitudeInput = document.querySelector("#altitude");
const altitudeOutput = document.querySelector("#altitude-output");

function updateAltitude() {
  droneHeight = Number(altitudeInput.value);

  telemetry.altitude = droneHeight;

  drone.position.y = droneHeight;
  dragPlane.constant = -droneHeight;

  altitudeOutput.textContent = `${droneHeight.toFixed(2)} m`;
}

altitudeInput.addEventListener(
  "input",
  updateAltitude
);

updateAltitude();

const yawInput = document.querySelector("#yaw");
const yawOutput = document.querySelector("#yaw-output");
let droneYawDegrees = 0;
function updateYaw() {
  droneYawDegrees = Number(yawInput.value);
  telemetry.yawDegrees = droneYawDegrees;

  //negative turns slider direc into top-down rot direction
  drone.rotation.y = -THREE.MathUtils.degToRad(droneYawDegrees);

  yawOutput.textContent = `${droneYawDegrees.toFixed(0)}°`;
}

yawInput.addEventListener(
  "input",
  updateYaw
);

updateYaw();

const pitchInput = document.querySelector("#pitch");
const pitchOutput = document.querySelector("#pitch-output");
let dronePitchDegrees = 0;
function updatePitch() {
  dronePitchDegrees = Number(pitchInput.value);
  telemetry.pitchDegrees = dronePitchDegrees;
  // The drone points along positive x, so rotating around
  // z raises or lowers its nose.
  drone.rotation.z = THREE.MathUtils.degToRad(dronePitchDegrees);
  pitchOutput.textContent = `${dronePitchDegrees.toFixed(0)}°`;
}

pitchInput.addEventListener(
  "input",
  updatePitch
);

updatePitch();

const rollInput = document.querySelector("#roll");
const rollOutput = document.querySelector("#roll-output");
let droneRollDegrees = 0;
function updateRoll() {
  droneRollDegrees = Number(rollInput.value);
  telemetry.rollDegrees = droneRollDegrees;
  drone.rotation.x = THREE.MathUtils.degToRad(droneRollDegrees);
  rollOutput.textContent = `${droneRollDegrees.toFixed(0)}°`;
}
rollInput.addEventListener(
  "input",
  updateRoll
);
updateRoll();

const resetButton = document.querySelector("#reset-button");
resetButton.addEventListener("click", () => {
  altitudeInput.value = "1.35";
  updateAltitude();

  yawInput.value = "0";
  updateYaw();

  pitchInput.value = "0";
  updatePitch();

  rollInput.value = "0";
  updateRoll();

  drone.position.set(
    0,
    droneHeight,
    0
  );

  camera.position.set(12, 11, 15);
  orbitControls.target.set(0, 1,2, 0);
  orbitControls.update();

  //old code 
  /*drone.position.set(0, droneHeight, 0);
  camera.position.set(12, 11, 15);
  orbitControls.target.set(0, 1.2, 0);
  orbitControls.update();*/
});

function connectSlider(inputId, outputId, parameterName, suffix = "") {
  const input = document.querySelector(inputId);
  const output = document.querySelector(outputId);

  const update = () => {
    params[parameterName] = Number(input.value);
    output.value = `${input.value}${suffix}`;
    output.textContent = `${input.value}${suffix}`;
  };

  input.addEventListener("input", update);
  update();
}

connectSlider("#base-speed", "#base-output", "baseSpeed", " m/s");
connectSlider("#turbulence", "#turbulence-output", "turbulence");
connectSlider("#gradient", "#gradient-output", "gradient");

const animateWindInput = document.querySelector("#animate-wind");
animateWindInput.addEventListener("change", () => {
  params.animateWind = animateWindInput.checked;
});

//new control selector declarations

const useEstimatedWindInput =
  document.querySelector(
    "#use-estimated-wind"
  );

function updateEstimatedWindMode() {
  params.useEstimatedWind =
    useEstimatedWindInput.checked;
}

useEstimatedWindInput.addEventListener(
  "change",
  updateEstimatedWindMode
);

updateEstimatedWindMode();

const windDirectionOutput =
  document.querySelector(
    "#wind-direction-output"
  );

function updateWindDirection() {
  params.windDirectionDegrees =
    Number(windDirectionInput.value);

  windDirectionOutput.textContent =
    `${params.windDirectionDegrees.toFixed(0)}°`;
}

// wind-direction selectors
const aerodynamicForceValue =
  document.querySelector(
    "#aerodynamic-force-value"
  );

const aerodynamicForceComponents =
  document.querySelector(
    "#aerodynamic-force-components"
  );

const windDirectionInput =
  document.querySelector("#wind-direction");

windDirectionInput.addEventListener(
  "input",
  updateWindDirection
);

updateWindDirection();

window.addEventListener("resize", () => {
  const width = app.clientWidth;
  const height = app.clientHeight;

  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
});

const clock = new THREE.Clock();

function calculateReferenceWind(
  position
) {
  const referenceHeight = 1.0;
  const minimumHeight = 0.25;
  const shearExponent = 0.20;

  const sampleHeight =
    Math.max(
      position.y,
      minimumHeight
    );

  // Vertical wind shear.
  const verticalShearSpeed =
    params.baseSpeed *
    Math.pow(
      sampleHeight /
        referenceHeight,
      shearExponent
    );

  // Horizontal microclimate gradient.
  const horizontalOffset =
    Math.hypot(
      position.x,
      position.z
    );

  const horizontalGradient =
    params.gradient *
    horizontalOffset;

  const referenceSpeed =
    Math.max(
      0.05,
      verticalShearSpeed +
        horizontalGradient
    );

  const directionRadians =
    THREE.MathUtils.degToRad(
      params.windDirectionDegrees ?? 0
    );

  referenceWindVector.set(
    referenceSpeed *
      Math.cos(directionRadians),

    0,

    referenceSpeed *
      Math.sin(directionRadians)
  );

  return referenceWindVector;
}

function animate() {
  requestAnimationFrame(animate);
  // Prevent an unusually large time step after the
  // browser pauses or changes tabs.
  const deltaTime = Math.min(
    clock.getDelta(),
    0.1
  );
  const timeSeconds = clock.elapsedTime;
  updateMlPrediction(timeSeconds);
  updateDroneVelocity(deltaTime);

  const aerodynamicForce =
    estimateAerodynamicForce(telemetry);

  const rawWindEstimate =
    estimateWindFromDrag(
      telemetry,
      aerodynamicForce
    );

  const windSmoothing =
    1 - Math.exp(-3 * deltaTime);

  const rawWindVector =
    new THREE.Vector3(
      rawWindEstimate.x,
      rawWindEstimate.y,
      rawWindEstimate.z
    );

  smoothedEstimatedWind.lerp(
    rawWindVector,
    windSmoothing
  );

  telemetry.estimatedWindX =
    smoothedEstimatedWind.x;

  telemetry.estimatedWindY =
    smoothedEstimatedWind.y;

  telemetry.estimatedWindZ =
    smoothedEstimatedWind.z;

  telemetry.estimatedWindSpeed =
    smoothedEstimatedWind.length();

  if (params.useEstimatedWind && latestMlWind) {
    smoothedEstimatedWind.set(latestMlWind.u, latestMlWind.w, latestMlWind.v);
    telemetry.estimatedWindX = latestMlWind.u;
    telemetry.estimatedWindY = latestMlWind.w;
    telemetry.estimatedWindZ = latestMlWind.v;
    telemetry.estimatedWindSpeed = latestMlWind.speed;
    telemetry.estimatedWindConfidence = 100;
  }

  const referenceWind =
    calculateReferenceWind(
      drone.position
    );

telemetry.referenceWindX =
  referenceWind.x;

telemetry.referenceWindY =
  referenceWind.y;

telemetry.referenceWindZ =
  referenceWind.z;

telemetry.referenceWindSpeed =
  referenceWind.length();

// Vector difference between the estimated
// and reference wind.
telemetry.windEstimateError =
  smoothedEstimatedWind.distanceTo(
    referenceWind
  );

if (
  telemetry.referenceWindSpeed >
  0.1
) {
  telemetry.windEstimateErrorPercent =
    (
      telemetry.windEstimateError /
      telemetry.referenceWindSpeed
    ) * 100;
} else {
  telemetry.windEstimateErrorPercent = 0;
}

  telemetry.estimatedWindConfidence = 
    calculateWindConfidence(
      rawWindEstimate,
      aerodynamicForce
    );

  if (params.useEstimatedWind && latestMlWind) {
    telemetry.estimatedWindConfidence = 100;
  }

  telemetry.aerodynamicForceX =
    aerodynamicForce.x;

  telemetry.aerodynamicForceY =
    aerodynamicForce.y;

  telemetry.aerodynamicForceZ =
    aerodynamicForce.z;

  telemetry.aerodynamicForceMagnitude =
    aerodynamicForce.magnitude;
  
  //display update code

  if (windConfidenceValue) {
    windConfidenceValue.textContent =
      `${telemetry.estimatedWindConfidence.toFixed(0)}%`;
  }

  if (windConfidenceBar) {
    windConfidenceBar.style.width =
      `${telemetry.estimatedWindConfidence}%`;
  }

  if (estimatedWindValue) {
    estimatedWindValue.textContent =
      `${telemetry.estimatedWindSpeed.toFixed(2)} m/s`;
  }

  if (estimatedWindComponents) {
    estimatedWindComponents.textContent =
      `u: ${telemetry.estimatedWindX.toFixed(2)}, ` +
      `v: ${telemetry.estimatedWindZ.toFixed(2)}, ` +
      `w: ${telemetry.estimatedWindY.toFixed(2)} m/s`;
  }

  if (aerodynamicForceValue) {
    aerodynamicForceValue.textContent =
      `${aerodynamicForce.magnitude.toFixed(2)} N`;
  }

  if (aerodynamicForceComponents) {
    aerodynamicForceComponents.textContent =
      `Fx: ${aerodynamicForce.x.toFixed(2)}, ` +
      `Fy: ${aerodynamicForce.y.toFixed(2)}, ` +
      `Fz: ${aerodynamicForce.z.toFixed(2)} N`;
  }

  for (const object of drone.children) {
    if (object.userData.isRotor) {
      object.rotation.y += 0.23;
    }
  }

  if (referenceWindValue) {
    referenceWindValue.textContent =
      `${telemetry.referenceWindSpeed.toFixed(
        2
      )} m/s`;
  }

  if (windErrorValue) {
    const safeErrorPercent =
      Number.isFinite(
        telemetry.windEstimateErrorPercent
      )
        ? telemetry.windEstimateErrorPercent
        : 0;

    windErrorValue.textContent =
      `${telemetry.windEstimateError.toFixed(
        2
      )} m/s ` +
      `(${safeErrorPercent.toFixed(0)}%)`;
  }

  centerMarker.material.opacity = 0.62 + Math.sin(timeSeconds * 2) * 0.22;

  updateWindField(timeSeconds);
  orbitControls.update();
  renderer.render(scene, camera);
}

animate();
