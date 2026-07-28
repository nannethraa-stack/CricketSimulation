import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// CHANGE THIS after deploying the backend
const API = https://cricketsimulation-2.onrender.com;


const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x071207);
scene.fog = new THREE.Fog(0x071207, 40, 100);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 150);
camera.position.set(15, 10, 20);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(10, 0.5, 0);
controls.enableDamping = true;

scene.add(new THREE.AmbientLight(0xffffff, 0.5));
const dir = new THREE.DirectionalLight(0xffffff, 1.1);
dir.position.set(20, 30, 15);
dir.castShadow = true;
scene.add(dir);

function createPitch() {
  const g = new THREE.Group();
  const pitch = new THREE.Mesh(
    new THREE.BoxGeometry(20.12, 0.08, 3.2),
    new THREE.MeshStandardMaterial({ color: 0x2e7d32, roughness: 0.85 })
  );
  pitch.position.set(10.06, 0.04, 0);
  pitch.receiveShadow = true;
  g.add(pitch);

  const creaseMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  [1.22, 17.68].forEach(x => {
    const c = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.015, 3.2), creaseMat);
    c.position.set(x, 0.09, 0);
    g.add(c);
  });

  const stumpMat = new THREE.MeshStandardMaterial({ color: 0xffeb3b });
  for (let i = -1; i <= 1; i++) {
    const s = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 0.71, 8), stumpMat);
    s.position.set(20.12, 0.355, i * 0.105);
    g.add(s);
  }

  const zones = [
    [0, 4.5, 0xffeb3b],
    [4.5, 7.5, 0x66bb6a],
    [7.5, 10.5, 0xffa726],
    [10.5, 20.12, 0xef5350]
  ];
  zones.forEach(([a, b, col]) => {
    const w = b - a;
    const z = new THREE.Mesh(
      new THREE.BoxGeometry(w, 0.008, 3.2),
      new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.15 })
    );
    z.position.set(a + w / 2, 0.085, 0);
    g.add(z);
  });

  scene.add(g);
}
createPitch();

function createPerson(color, isBowler = false) {
  const g = new THREE.Group();
  const body = new THREE.Mesh(
    new THREE.CylinderGeometry(0.18, 0.22, 0.7, 8),
    new THREE.MeshStandardMaterial({ color })
  );
  body.position.y = 1.05;
  g.add(body);

  const head = new THREE.Mesh(
    new THREE.SphereGeometry(0.16, 10, 10),
    new THREE.MeshStandardMaterial({ color })
  );
  head.position.y = 1.58;
  g.add(head);

  const legGeo = new THREE.CylinderGeometry(0.07, 0.08, 0.7, 6);
  [-0.11, 0.11].forEach(x => {
    const leg = new THREE.Mesh(legGeo, new THREE.MeshStandardMaterial({ color }));
    leg.position.set(x, 0.35, 0);
    g.add(leg);
  });

  if (isBowler) {
    const arm = new THREE.Mesh(
      new THREE.CylinderGeometry(0.05, 0.05, 0.55, 6),
      new THREE.MeshStandardMaterial({ color })
    );
    arm.position.set(0.28, 1.35, 0.1);
    arm.rotation.z = -Math.PI / 2.8;
    g.add(arm);
  } else {
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

const bowler = createPerson(0xc62828, true);
bowler.position.set(1.3, 0, 0.15);
scene.add(bowler);

const batsman = createPerson(0x1565c0, false);
batsman.position.set(18.5, 0, 0);
scene.add(batsman);

const ball = new THREE.Mesh(
  new THREE.SphereGeometry(0.036, 14, 14),
  new THREE.MeshStandardMaterial({ color: 0xff1744, roughness: 0.3 })
);
ball.visible = false;
scene.add(ball);

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

let currentTrail = null;
let isAnimating = false;
let animProgress = 0;
let currentPoints = [];
let deliveryCount = 0;

async function bowl() {
  const body = {
    speed_kmh: +document.getElementById('speed').value,
    length_m: +document.getElementById('length').value,
    line_m: +document.getElementById('line').value,
    swing_cm: +document.getElementById('swing').value,
    seam_cm: +document.getElementById('seam').value,
    shot_type: document.getElementById('shot').value
  };

  document.getElementById('stats').textContent = "Running Edge-AI Pipeline...";

  try {
    const res = await fetch(`${API}/api/bowl`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    const data = await res.json();

    currentPoints = data.trajectory.map(p => new THREE.Vector3(p.x, p.z, p.y));
    animProgress = 0;
    isAnimating = true;
    ball.visible = true;

    releaseMarker.position.copy(currentPoints[0]);
    releaseMarker.visible = true;
    bounceMarker.position.set(data.bounce_x, 0.13, data.bounce_y);
    bounceMarker.visible = true;

    deliveryCount++;
    document.getElementById('count').textContent = `Deliveries: ${deliveryCount}`;
    document.getElementById('stats').textContent =
      `${data.release_speed_kmh} km/h | ${data.length_category} | ${data.line_category}`;
    document.getElementById('fusion').textContent =
      `Fusion: ${(data.fusion_confidence * 100).toFixed(0)}% | Contact: ${data.contact_quality} | ${data.timing}`;
  } catch (err) {
    document.getElementById('stats').textContent = "Backend not reachable";
    console.error(err);
  }
}

document.getElementById('bowlBtn').onclick = bowl;

document.getElementById('clearBtn').onclick = async () => {
  await fetch(`${API}/api/session/clear`, { method: "POST" });
  deliveryCount = 0;
  document.getElementById('count').textContent = "Deliveries: 0";
  document.getElementById('stats').textContent = "Session cleared";
  document.getElementById('fusion').textContent = "";
};

['speed', 'length', 'line', 'swing', 'seam'].forEach(id => {
  document.getElementById(id).oninput = e => {
    document.getElementById(id + 'Val').textContent = e.target.value;
  };
});

function animate() {
  requestAnimationFrame(animate);
  controls.update();

  if (isAnimating && currentPoints.length) {
    animProgress += 0.009;
    if (animProgress >= 1) {
      animProgress = 1;
      isAnimating = false;
    }
    const idx = Math.floor(animProgress * (currentPoints.length - 1));
    ball.position.copy(currentPoints[idx]);

    if (currentTrail) scene.remove(currentTrail);
    const geo = new THREE.BufferGeometry().setFromPoints(currentPoints.slice(0, idx + 1));
    currentTrail = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: 0xff9100 }));
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
