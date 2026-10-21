import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const container = document.getElementById('moon-container');
const loadingOverlay = document.getElementById('loading-overlay');
const valLat = document.getElementById('val-lat');
const valLon = document.getElementById('val-lon');
const btnConfirm = document.getElementById('btn-confirm');
const coordLandmark = document.getElementById('coord-landmark');
const valLandmark = document.getElementById('val-landmark');
const valElevation = document.getElementById('val-elevation');

let elevationRequest = 0;
async function showElevation(lat, lon) {
    if (!valElevation) return;
    const requestId = ++elevationRequest;
    valElevation.textContent = '…';
    try {
        const res = await apiGet(`/api/moon/elevation?lat=${lat.toFixed(4)}&lon=${lon.toFixed(4)}`);
        if (requestId !== elevationRequest) return;  // a newer pick superseded this one
        const sign = res.elevation >= 0 ? '+' : '−';
        valElevation.textContent = `${sign}${Math.abs(res.elevation / 1000).toFixed(2)} km`;
        valElevation.title = res.source;
    } catch (_) {
        if (requestId === elevationRequest) valElevation.textContent = '—';
    }
}

let selectedLat = null;
let selectedLon = null;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
camera.position.z = 3;

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.setPixelRatio(window.devicePixelRatio);
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.minDistance = 1.2;
controls.maxDistance = 5;

const ambientLight = new THREE.AmbientLight(0x404040, 1.0);
scene.add(ambientLight);
const directionalLight = new THREE.DirectionalLight(0xffffff, 2.5);
scene.add(directionalLight);

const fillLight = new THREE.DirectionalLight(0x8899aa, 0.8);
fillLight.position.set(-5, -2, -5);
scene.add(fillLight);

// ---------------------------------------------------------------------------
// Real sunlight and libration for the current moment
// ---------------------------------------------------------------------------
// /api/moon/illumination runs a Meeus ephemeris on the server: the sub-solar
// point places the light (day/night terminator), and the sub-Earth point is
// the centre of the disc as seen from Earth today (libration), where the
// camera starts.
let illumination = null;

async function loadIllumination() {
    try {
        illumination = await apiGet('/api/moon/illumination');
        const e = illumination.sub_earth;
        camera.position.copy(latLonToLocal(e.latitude, e.longitude).multiplyScalar(camera.position.length()));
        camera.lookAt(0, 0, 0);
        controls.update();
    } catch (err) {
        console.warn('Illumination unavailable, using even lighting', err);
    }
    applyLighting();
}

const sunToggle = document.getElementById('btn-sun-toggle');
const sunInfo = document.getElementById('sun-info');
let realSunlight = true;

function applyLighting() {
    if (realSunlight && illumination) {
        const s = illumination.sub_solar;
        directionalLight.position.copy(latLonToLocal(s.latitude, s.longitude).multiplyScalar(5));
        directionalLight.intensity = 3.0;
        ambientLight.intensity = 0.35;   // faint "earthshine" so the night side stays pickable
        fillLight.visible = false;
        if (sunInfo) {
            const when = new Date(illumination.at).toLocaleString('en-US', { dateStyle: 'medium', timeStyle: 'short' });
            const lib = illumination.sub_earth;
            const fmt = v => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(1)}°`;
            sunInfo.textContent = `${when} · ${illumination.phase_name}, ${Math.round(illumination.illuminated_fraction * 100)}% lit · libration ${fmt(lib.longitude)} lon, ${fmt(lib.latitude)} lat`;
        }
        if (sunToggle) sunToggle.textContent = 'Even lighting';
    } else {
        directionalLight.position.copy(camera.position).add(new THREE.Vector3(2, 1.5, 0)).normalize().multiplyScalar(5);
        directionalLight.intensity = 2.5;
        ambientLight.intensity = 1.0;
        fillLight.visible = true;
        if (sunInfo) sunInfo.textContent = illumination ? 'Even studio lighting (not the real Sun)' : 'Even lighting (ephemeris unavailable)';
        if (sunToggle) sunToggle.textContent = 'Real sunlight';
    }
}

if (sunToggle) {
    sunToggle.addEventListener('click', () => {
        realSunlight = !realSunlight;
        applyLighting();
    });
}

const radius = 1;
const geometry = new THREE.SphereGeometry(radius, 128, 64);
// SphereGeometry puts texture u = 0.5 (longitude 0°) on +X. Rotating by -90°
// moves it to +Z, facing the camera, so the near side faces the viewer as
// seen from Earth, with lunar east (Mare Crisium) on the right. This is the
// orientation localToLatLon / latLonToLocal assume.
geometry.rotateY(-Math.PI / 2);

const material = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    roughness: 0.85,
    metalness: 0.05
});
const moon = new THREE.Mesh(geometry, material);
scene.add(moon);

const markerGeo = new THREE.SphereGeometry(0.02, 16, 16);
const markerMat = new THREE.MeshBasicMaterial({ color: 0xff3333 });
const marker = new THREE.Mesh(markerGeo, markerMat);
marker.visible = false;
scene.add(marker);

// Load real NASA LRO Moon texture map
const textureLoader = new THREE.TextureLoader();
textureLoader.load(
    '/static/textures/moon_color.jpg',
    (texture) => {
        texture.colorSpace = THREE.SRGBColorSpace;
        material.map = texture;
        material.roughness = 0.85;
        material.metalness = 0.05;
        material.needsUpdate = true;
        if (loadingOverlay) loadingOverlay.style.display = 'none';

        // Relief from real LOLA laser altimetry (not from image brightness),
        // so maria read as low plains and highlands as high ground.
        textureLoader.load('/static/textures/moon_lola_height.png', (height) => {
            material.bumpMap = height;
            material.bumpScale = 3.0;
            material.needsUpdate = true;
        });
    },
    undefined,
    (err) => {
        console.warn('NASA texture failed to load, falling back to procedural texture:', err);
        // Fallback to procedural terrain texture endpoint
        fetch('/api/moon/terrain-texture')
            .then(res => res.arrayBuffer())
            .then(buffer => {
                const width = 1024;
                const height = 512;
                const data = new Uint8Array(buffer);
                const procTexture = new THREE.DataTexture(data, width, height, THREE.RedFormat);
                procTexture.needsUpdate = true;
                material.map = procTexture;
                material.displacementMap = procTexture;
                material.displacementScale = 0.05;
                material.needsUpdate = true;
                if (loadingOverlay) loadingOverlay.style.display = 'none';
            })
            .catch(fallbackErr => {
                console.error('All texture loading failed', fallbackErr);
                if (loadingOverlay) loadingOverlay.textContent = '3D Moon Ready (Basic)';
                setTimeout(() => { if (loadingOverlay) loadingOverlay.style.display = 'none'; }, 1000);
            });
    }
);

const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

export function localToLatLon(localPoint) {
    // Math.asin gives values in [-PI/2, PI/2]
    const latRad = Math.asin(localPoint.y / radius);
    
    // In our rotated sphere, localPoint is pre-rotation.
    // Base SphereGeometry:
    // x = r * sin(phi) * sin(theta)
    // z = r * sin(phi) * cos(theta)
    // theta = atan2(x, z). 
    // theta is in [-PI, PI].
    let lonRad = Math.atan2(localPoint.x, localPoint.z);
    
    return {
        lat: latRad * (180 / Math.PI),
        lon: lonRad * (180 / Math.PI)
    };
}

// Dragging rotates the Moon; only a press released without moving selects a
// point. OrbitControls still receives every pointer event, so a drag that ends
// over the Moon no longer counts as a click.
const CLICK_MAX_MOVE_PX = 5;
let pointerDown = null;

renderer.domElement.addEventListener('pointerdown', (event) => {
    if (event.button !== 0) return;
    pointerDown = { x: event.clientX, y: event.clientY, id: event.pointerId };
});

renderer.domElement.addEventListener('pointerup', (event) => {
    if (!pointerDown || event.pointerId !== pointerDown.id) return;
    const moved = Math.hypot(event.clientX - pointerDown.x, event.clientY - pointerDown.y);
    pointerDown = null;
    container.classList.remove('is-dragging');
    if (moved <= CLICK_MAX_MOVE_PX) selectAt(event);
});

renderer.domElement.addEventListener('pointermove', (event) => {
    if (!pointerDown || event.pointerId !== pointerDown.id) return;
    const moved = Math.hypot(event.clientX - pointerDown.x, event.clientY - pointerDown.y);
    if (moved > CLICK_MAX_MOVE_PX) container.classList.add('is-dragging');
});

renderer.domElement.addEventListener('pointercancel', () => {
    pointerDown = null;
    container.classList.remove('is-dragging');
});

function selectAt(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObject(moon);

    if (intersects.length > 0) {
        const p = intersects[0].point;
        const localPoint = moon.worldToLocal(p.clone());
        
        const coords = localToLatLon(localPoint);
        selectedLat = coords.lat;
        selectedLon = coords.lon;
        
        valLat.textContent = selectedLat.toFixed(4) + '°';
        valLon.textContent = selectedLon.toFixed(4) + '°';
        
        marker.position.copy(p);
        marker.position.multiplyScalar(1.06); 
        marker.visible = true;
        
        btnConfirm.disabled = false;
        if (coordLandmark) coordLandmark.style.display = 'none';
        showElevation(selectedLat, selectedLon);
    }
}

window.addEventListener('resize', () => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
});

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}
animate();

export function latLonToLocal(lat, lon) {
    const latRad = lat * (Math.PI / 180);
    const lonRad = lon * (Math.PI / 180);
    const cosLat = Math.cos(latRad);
    return new THREE.Vector3(
        radius * cosLat * Math.sin(lonRad),
        radius * Math.sin(latRad),
        radius * cosLat * Math.cos(lonRad)
    );
}

applyLighting();
loadIllumination();

// Landmark buttons handling

document.querySelectorAll('.landmark-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        e.preventDefault();
        const lat = parseFloat(btn.dataset.lat);
        const lon = parseFloat(btn.dataset.lon);
        selectedLat = lat;
        selectedLon = lon;
        
        valLat.textContent = selectedLat.toFixed(4) + '°';
        valLon.textContent = selectedLon.toFixed(4) + '°';
        if (coordLandmark && valLandmark) {
            valLandmark.textContent = btn.textContent.trim().split('(')[0];
            coordLandmark.style.display = 'flex';
        }
        
        const localPos = latLonToLocal(lat, lon);
        const worldPos = moon.localToWorld(localPos.clone());
        marker.position.copy(worldPos);
        marker.position.multiplyScalar(1.06);
        marker.visible = true;
        btnConfirm.disabled = false;
        showElevation(lat, lon);

        // Smoothly rotate camera towards the selected landmark
        controls.target.set(0, 0, 0);
    });
});

btnConfirm.addEventListener('click', () => {
    if (selectedLat !== null && selectedLon !== null) {
        window.location.href = `/experiment?lat=${selectedLat.toFixed(4)}&lon=${selectedLon.toFixed(4)}`;
    }
});

