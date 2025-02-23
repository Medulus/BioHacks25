// script.js

// =======================================================
// 1. THREE.JS 3D BACKGROUND
// =======================================================
let scene, camera, renderer, sphere;

function init3DScene() {
  const container = document.getElementById('scene-container');

  // Create a new scene
  scene = new THREE.Scene();

  // Set up camera (FOV, aspect ratio, near, far)
  camera = new THREE.PerspectiveCamera(
    60, 
    window.innerWidth / window.innerHeight, 
    0.1, 
    1000
  );
  camera.position.z = 50;

  // Create renderer
  renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  container.appendChild(renderer.domElement);

  // Create a glowing sphere
  const geometry = new THREE.SphereGeometry(10, 32, 32);
  const material = new THREE.MeshPhongMaterial({
    color: 0x00d4ff,
    emissive: 0x001a2b,
    emissiveIntensity: 0.5,
    shininess: 100,
  });
  sphere = new THREE.Mesh(geometry, material);
  scene.add(sphere);

  // Add a point light
  const pointLight = new THREE.PointLight(0xffffff, 1);
  pointLight.position.set(25, 50, 25);
  scene.add(pointLight);

  animate();
}

// Animate/Render loop
function animate() {
  requestAnimationFrame(animate);

  // Rotate the sphere gently
  sphere.rotation.y += 0.002;

  renderer.render(scene, camera);
}

// Handle window resize
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Initialize 3D scene
init3DScene();

// =======================================================
// 2. GSAP ANIMATIONS (Glass Card Fade-In, etc.)
// =======================================================
gsap.from(".glass-card", {
  duration: 1.5,
  y: 50,
  opacity: 0,
  ease: "power4.out"
});

// =======================================================
// 3. FORM SUBMISSION LOGIC
// =======================================================
document.getElementById('immunoForm').addEventListener('submit', function(e) {
  e.preventDefault();

  const name = document.getElementById('patientName').value;
  const email = document.getElementById('patientEmail').value;
  const symptoms = document.getElementById('patientSymptoms').value;
  const history = document.getElementById('patientHistory').value;

  // Example: Just alert the input. You would typically do:
  // fetch('/api/your-ai-endpoint', { method: 'POST', body: JSON.stringify(...) })
  alert(`Submitted:\nName: ${name}\nEmail: ${email}\nSymptoms: ${symptoms}\nHistory: ${history}`);

  // Clear fields (optional)
  this.reset();
});
