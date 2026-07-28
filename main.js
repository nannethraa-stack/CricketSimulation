import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ===================== BASIC SETUP =====================
const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a160a);
scene.fog = new THREE.Fog(0x0a160a, 35, 90);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 120);
camera.position.set(14, 9, 19);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(10, 0.6, 0);
controls.enableDamping = true;

// Lights
scene.add(new THREE.AmbientLight(0xffffff, 0.5));
const dirLight = new THREE.DirectionalLight(0xffffff, 1.15);
dirLight.position.set(18, 28, 12);
dirLight.castShadow = true;
scene.add(dirLight);

// ===================== PITCH =====================
function createPitch() {
  const g = new THREE.Group();

  const pitch = new THREE.Mesh(
    new THREE.BoxGeometry(20.12, 0.08, 3.2),
    new THREE.MeshStandardMaterial({ color: 0x2e7d32, roughness: 0.85 })
  );
  pitch.position.set(10.06, 0.04, 0);
  pitch.receiveShadow = true;
  g.add(pitch);

  // Creases
  const creaseMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  [[1.22], [17.68]].forEach(x => {
    const c = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.015, 3.2), creaseMat);
    c.position.set(x[0], 0.09, 0);
    g.add(c);
  });

  // Stumps
  const stumpMat = new THREE.MeshStandardMaterial({ color: 0xffeb3b });
  for (let i = -1; i <= 1; i++) {
    const s = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 0.71, 8), stumpMat);
    s.position.set(20.12, 0.355, i * 0.105);
    s.castShadow = true;
    g.add(s);
  }

  // Length zones
  const zones = [
    [0, 4.5, 0xffeb3b],
    [4.5, 7.5, 0x66bb6a],
    [7.5, 10.5, 0xffa726],
    [10.5, 20.12, 0xef5350]
  ];
  zones.forEach(([start, end, col]) => {
    const w = end - start;
    const z = new THREE.Mesh(
      new THREE.BoxGeometry(w, 0.008, 3.2),
      new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.16 })
    );
    z.position.set(start + w / 2, 0.085, 0);
    g.add(z);
  });

  scene.add(g);
}

createPitch();

// ===================== LOW-POLY PLAYERS =====================
function createLowPolyPerson(color, isBowler = false) {
  const g = new THREE.Group();

  // Body
  const body = new THREE.Mesh(
    new THREE.CylinderGeometry(0.18, 0.22, 0.7, 8),
    new THREE.MeshStandardMaterial({ color })
  );
  body.position.y = 1.05;
  body.castShadow = true;
  g.add(body);

  // Head
  const head = new THREE.Mesh(
    new THREE.SphereGeometry(0.16, 10, 10),
    new THREE.MeshStandardMaterial({ color })
  );
  head.position.y = 1.58;
  g.add(head);

  // Legs
  const legGeo = new THREE.CylinderGeometry(0.07, 0.08, 0.7, 6);
  const legMat = new THREE.MeshStandardMaterial({ color });
  const leftLeg = new THREE.Mesh(legGeo, legMat);
  leftLeg.position.set(-0.11, 0.35, 0);
  g.add(leftLeg);
  const rightLeg = new THREE.Mesh(legGeo, legMat);
  rightLeg.position.set(0.11, 0.35, 0);
  g.add(rightLeg);

  if (isBowler) {
    // Bowling arm raised
    const arm = new THREE.Mesh(
      new THREE.CylinderGeometry(0.05, 0.05, 0.55, 6),
      new THREE.MeshStandardMaterial({ color })
    );
    arm.position.set(0.28, 1.35, 0.1);
    arm.rotation.z = -Math.PI / 2.8;
    g.add(arm);
  } else {
    // Bat
    const bat = new THREE.Mesh(
      new THREE.BoxGeometry(0.08, 0.85, 0.04),
      new THREE.MeshStandardMaterial({ color: 0x5d4037 })
    );
    bat.position.set(0.32, 0.95, 0.15);
    bat.rotation.z = -0.4;
    g.add(bat);
  }

  return g;
}

const bowler = createLowPolyPerson(0xc62828, true);
bowler.position.set(1.3, 0, 0.15);
scene.add(bowler);

const batsman = createLowPolyPerson(0x1565c0, false);
batsman.position.set(18.5, 0, 0);
scene.add(batsman);

// ===================== BALL + TRAIL =====================
const ball = new THREE.Mesh(
  new THREE.SphereGeometry(0.036, 14, 14),
  new THREE.MeshStandardMaterial({ color: 0xff1744, roughness: 0.35 })
);
ball.castShadow = true;
ball.visible = false;
scene.add(ball);

const trailMat = new THREE.LineBasicMaterial({ color: 0xff9100 });
let currentTrail = null;

// Markers
const releaseMarker = new THREE.Mesh(
  new THREE.SphereGeometry(0.055, 10, 10),
  new THREE.MeshBasicMaterial({ color: 0x2979ff })
);
releaseMarker.visible = false;
scene.add(releaseMarker);

const bounceMarker = new THREE.Mesh(
  new THREE.SphereGeometry(0.07, 10, 10),
  new THREE.MeshBasicMaterial({ color: 0xffeb3b })
);
bounceMarker.visible = false;
scene.add(bounceMarker);

// ===================== DATA STORAGE =====================
let deliveries = [];          // for beehive
let wagonPoints = [];        // for wagon wheel
let currentTraj = null;
let animProgress = 0;
let isAnimating = false;
let viewMode = 'beehive';     // or 'wagon'

// ===================== TRAJECTORY WITH SWING + SEAM =====================
function generateTrajectory(speedKmh, lengthM, lineM, swingCm, seamCm) {
  const v0 = speedKmh / 3.6;
  const swing = swingCm / 100;
  const seam = seamCm / 100;
  const releaseHeight = 2.15;
  const tBounce = Math.max(0.37, Math.min(0.95, lengthM / Math.max(v0, 1) * 1.06));

  const points = [];
  const steps = 80;

  // Pre bounce (with swing)
  for (let i = 0; i <= steps; i++) {
    const t = (i / steps) * tBounce;
    const x = v0 * t * 0.975;
    const progress = t / tBounce;
    const y = lineM + swing * Math.pow(progress, 1.45);
    const z = releaseHeight - 0.5 * 9.81 * t * t * 0.91;
    points.push(new THREE.Vector3(x, Math.max(z, 0.04), y));
  }

  const bouncePos = points[points.length - 1];

  // Post bounce (with seam)
  const vPost = v0 * 0.64;
  for (let i = 1; i <= 45; i++) {
    const t = (i / 45) * 0.58;
    const x = bouncePos.x + vPost * t;
    const y = bouncePos.z + seam * (t / 0.58);
    const z = 0.11 + vPost * 0.21 * t - 0.5 * 9.81 * t * t * 0.76;
    points.push(new THREE.Vector3(x, Math.max(z, 0.04), y));
  }

  return {
    points,
    bounceIndex: steps,
    bounceX: bouncePos.x,
    bounceY: bouncePos.z,
    speedKmh,
    lengthM,
    lineM,
    swingCm,
    seamCm
  };
}

// ===================== BEEHIVE + WAGON WHEEL =====================
const beehiveGroup = new THREE.Group();
scene.add(beehiveGroup);

const wagonGroup = new THREE.Group();
wagonGroup.visible = false;
scene.add(wagonGroup);

// Simple wagon wheel base (circle at batsman end)
function createWagonWheelBase() {
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(2.8, 3.0, 64),
    new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide, transparent: true, opacity: 0.3 })
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.set(18.5, 0.1, 0);
  wagonGroup.add(ring);

  // Radial lines
  for (let i = 0; i < 8; i++) {
    const angle = (i / 8) * Math.PI * 2;
    const points = [
      new THREE.Vector3(18.5, 0.11, 0),
      new THREE.Vector3(18.5 + Math.cos(angle) * 2.9, 0.11, Math.sin(angle) * 2.9)
    ];
    const geo = new THREE.BufferGeometry().setFromPoints(points);
    const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.25 }));
    wagonGroup.add(line);
  }
}
createWagonWheelBase();

function addToBeehive(traj) {
  const dot = new THREE.Mesh(
    new THREE.SphereGeometry(0.07, 8, 8),
    new THREE.MeshBasicMaterial({ color: 0xffeb3b })
  );
  dot.position.set(traj.bounceX, 0.13, traj.bounceY);
  beehiveGroup.add(dot);
  deliveries.push(traj);
  document.getElementById('count').textContent = `Deliveries: ${deliveries.length}`;
}

function addToWagon(traj) {
  // Approximate scoring direction based on line + random variation
  const angle = traj.lineM * 1.8 + (Math.random() - 0.5) * 0.9;
  const dist = 1.8 + Math.random() * 1.1;
  const endX = 18.5 + Math.cos(angle) * dist;
  const endZ = Math.sin(angle) * dist;

  const points = [
    new THREE.Vector3(18.5, 0.12, 0),
    new THREE.Vector3(endX, 0.12, endZ)
  ];
  const geo = new THREE.BufferGeometry().setFromPoints(points);
  const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: 0x4fc3f7, linewidth: 2 }));
  wagonGroup.add(line);
  wagonPoints.push(points);
}

// ===================== BOWL =====================
function bowlDelivery() {
  const speed = +document.getElementById('speed').value;
  const length = +document.getElementById('length').value;
  const line = +document.getElementById('line').value;
  const swing = +document.getElementById('swing').value;
  const seam = +document.getElementById('seam').value;

  currentTraj = generateTrajectory(speed, length, line, swing, seam);
  animProgress = 0;
  isAnimating = true;
  ball.visible = true;

  // Clear previous trail
  if (currentTrail) scene.remove(currentTrail);

  releaseMarker.position.copy(currentTraj.points[0]);
  releaseMarker.visible = true;
  bounceMarker.position.set(currentTraj.bounceX, 0.13, currentTraj.bounceY);
  bounceMarker.visible = true;

  document.getElementById('stats').textContent =
    `${speed} km/h | L ${length.toFixed(1)}m | Swing ${swing}cm | Seam ${seam}cm`;
}

function clearAll() {
  isAnimating = false;
  ball.visible = false;
  releaseMarker.visible = false;
  bounceMarker.visible = false;
  if (currentTrail) scene.remove(currentTrail);

  // Clear beehive
  while (beehiveGroup.children.length) beehiveGroup.remove(beehiveGroup.children[0]);
  deliveries = [];

  // Clear wagon
  while (wagonGroup.children.length > 9) { // keep the base
    wagonGroup.remove(wagonGroup.children[wagonGroup.children.length - 1]);
  }
  wagonPoints = [];

  document.getElementById('count').textContent = 'Deliveries: 0';
  document.getElementById('stats').textContent = 'Ready';
}

// ===================== UI =====================
['speed', 'length', 'line', 'swing', 'seam'].forEach(id => {
  document.getElementById(id).oninput = e => {
    document.getElementById(id + 'Val').textContent = e.target.value;
  };
});

document.getElementById('bowlBtn').onclick = bowlDelivery;
document.getElementById('resetBtn').onclick = clearAll;

document.getElementById('btnBeehive').onclick = () => {
  viewMode = 'beehive';
  beehiveGroup.visible = true;
  wagonGroup.visible = false;
  document.getElementById('btnBeehive').classList.add('active');
  document.getElementById('btnWagon').classList.remove('active');
};

document.getElementById('btnWagon').onclick = () => {
  viewMode = 'wagon';
  beehiveGroup.visible = false;
  wagonGroup.visible = true;
  document.getElementById('btnWagon').classList.add('active');
  document.getElementById('btnBeehive').classList.remove('active');
};

// ===================== ANIMATION LOOP =====================
function animate() {
  requestAnimationFrame(animate);
  controls.update();

  if (isAnimating && currentTraj) {
    animProgress += 0.0085;
    if (animProgress >= 1) {
      animProgress = 1;
      isAnimating = false;

      // Add to overlays when finished
      addToBeehive(currentTraj);
      addToWagon(currentTraj);
    }

    const idx = Math.floor(animProgress * (currentTraj.points.length - 1));
    ball.position.copy(currentTraj.points[idx]);

    // Update trail
    const trailPoints = currentTraj.points.slice(0, idx + 1);
    if (currentTrail) scene.remove(currentTrail);
    const geo = new THREE.BufferGeometry().setFromPoints(trailPoints);
    currentTrail = new THREE.Line(geo, trailMat);
    scene.add(currentTrail);
  }

  renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
