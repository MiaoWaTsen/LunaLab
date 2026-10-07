import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const container = document.getElementById('moon-container');
const loadingOverlay = document.getElementById('loading-overlay');
const valLat = document.getElementById('val-lat');
const valLon = document.getElementById('val-lon');
const btnConfirm = document.getElementById('btn-confirm');

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
directionalLight.position.set(5, 3, 5);
scene.add(directionalLight);

const radius = 1;
// SphereGeometry(radius, widthSegments, heightSegments, phiStart, phiLength, thetaStart, thetaLength)
// In Three.js, equirectangular mapping:
// u = 0 is +Z, u = 0.25 is +X, u = 0.5 is -Z, u = 0.75 is -X.
// The texture from our backend goes from -180 to +180.
// So center of texture (lon=0) is at u=0.5, which is -Z.
// If we want prime meridian (lon=0) to be at +Z, we must rotate the geometry by PI around Y.
const geometry = new THREE.SphereGeometry(radius, 128, 64);
geometry.rotateY(Math.PI);

const material = new THREE.MeshStandardMaterial({
    color: 0x999999,
    roughness: 0.9,
    metalness: 0.1
});
const moon = new THREE.Mesh(geometry, material);
scene.add(moon);

const markerGeo = new THREE.SphereGeometry(0.02, 16, 16);
const markerMat = new THREE.MeshBasicMaterial({ color: 0xff3333 });
const marker = new THREE.Mesh(markerGeo, markerMat);
marker.visible = false;
scene.add(marker);

fetch('/api/moon/terrain-texture')
    .then(res => res.arrayBuffer())
    .then(buffer => {
        const width = 1024;
        const height = 512;
        const data = new Uint8Array(buffer);
        const texture = new THREE.DataTexture(data, width, height, THREE.RedFormat);
        texture.needsUpdate = true;
        
        material.map = texture;
        material.displacementMap = texture;
        material.displacementScale = 0.05;
        material.needsUpdate = true;
        
        loadingOverlay.style.display = 'none';
    })
    .catch(err => {
        console.error('Failed to load terrain texture', err);
        loadingOverlay.textContent = 'Failed to load terrain';
    });

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

container.addEventListener('click', (event) => {
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
    }
});

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

btnConfirm.addEventListener('click', () => {
    if (selectedLat !== null && selectedLon !== null) {
        window.location.href = `/experiment?lat=${selectedLat.toFixed(4)}&lon=${selectedLon.toFixed(4)}`;
    }
});
