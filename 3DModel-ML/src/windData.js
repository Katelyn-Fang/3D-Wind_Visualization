import * as THREE from "three";

export async function loadWindDataset(
  url = "/data/wind_predictions_preview.json"
) {
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(
      `Could not load wind data: ${response.status} ${response.statusText}`
    );
  }

  return response.json();
}

export function getFlight(dataset, flightId) {
  const flight = dataset.flights[String(flightId)];

  if (!flight) {
    const available = Object.keys(dataset.flights)
      .slice(0, 10)
      .join(", ");

    throw new Error(
      `Flight ${flightId} was not found. Example IDs: ${available}`
    );
  }

  return flight;
}

export function findNearestWindPoint(points, position) {
  if (!points.length) {
    return null;
  }

  let nearest = points[0];
  let nearestDistanceSquared = Infinity;

  for (const point of points) {
    const dx = point.x - position.x;
    const dy = point.y - position.y;
    const dz = point.z - position.z;

    const distanceSquared =
      dx * dx +
      dy * dy +
      dz * dz;

    if (distanceSquared < nearestDistanceSquared) {
      nearest = point;
      nearestDistanceSquared = distanceSquared;
    }
  }

  return nearest;
}

export function createWindArrow(scene) {
  const arrow = new THREE.ArrowHelper(
    new THREE.Vector3(1, 0, 0),
    new THREE.Vector3(0, 0, 0),
    1,
    0x2f80ed,
    0.25,
    0.14
  );

  arrow.line.material.transparent = true;
  arrow.cone.material.transparent = true;

  scene.add(arrow);

  return arrow;
}

export function updateWindArrow(
  arrow,
  point,
  source = "predicted",
  origin = null
) {
  if (!point || !point[source]) {
    return;
  }

  const wind = point[source];

  const direction = new THREE.Vector3(
    wind.vx,
    0,
    wind.vz
  );

  if (direction.lengthSq() < 1e-10) {
    arrow.visible = false;
    return;
  }

  arrow.visible = true;
  direction.normalize();

  const arrowOrigin =
    origin ??
    new THREE.Vector3(
      point.x,
      point.y,
      point.z
    );

  arrow.position.copy(arrowOrigin);
  arrow.setDirection(direction);

  const length = Math.max(
    0.3,
    wind.speed * 0.55
  );

  arrow.setLength(
    length,
    Math.min(0.35, length * 0.3),
    Math.min(0.2, length * 0.18)
  );

  const confidence =
    source === "predicted"
      ? wind.confidence
      : 1;

  const opacity =
    0.25 + 0.75 * confidence;

  arrow.line.material.opacity = opacity;
  arrow.cone.material.opacity = opacity;
}

export function addFlightVectorField(
  scene,
  flight,
  {
    every = 10,
    source = "predicted",
    lengthScale = 0.35,
  } = {}
) {
  const arrows = [];

  for (
    let index = 0;
    index < flight.points.length;
    index += every
  ) {
    const point = flight.points[index];
    const wind = point[source];

    const direction = new THREE.Vector3(
      wind.vx,
      0,
      wind.vz
    );

    if (direction.lengthSq() < 1e-10) {
      continue;
    }

    direction.normalize();

    const length = Math.max(
      0.2,
      wind.speed * lengthScale
    );

    const arrow = new THREE.ArrowHelper(
      direction,
      new THREE.Vector3(
        point.x,
        point.y,
        point.z
      ),
      length,
      source === "predicted"
        ? 0x2f80ed
        : 0xf2994a,
      Math.min(0.28, length * 0.3),
      Math.min(0.16, length * 0.18)
    );

    const confidence =
      source === "predicted"
        ? wind.confidence
        : 1;

    const opacity =
      0.2 + 0.8 * confidence;

    arrow.line.material.transparent = true;
    arrow.cone.material.transparent = true;
    arrow.line.material.opacity = opacity;
    arrow.cone.material.opacity = opacity;

    scene.add(arrow);
    arrows.push(arrow);
  }

  return arrows;
}

export function removeArrows(scene, arrows) {
  for (const arrow of arrows) {
    scene.remove(arrow);

    arrow.line.geometry.dispose();
    arrow.line.material.dispose();

    arrow.cone.geometry.dispose();
    arrow.cone.material.dispose();
  }
}