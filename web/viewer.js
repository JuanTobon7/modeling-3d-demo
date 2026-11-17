// Visor 3D (three.js, MIT). Se carga solo al abrir un modelo.
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

let ctx = null;

export function clearViewer(host) {
  if (!ctx) return;
  cancelAnimationFrame(ctx.raf);
  ctx.ro.disconnect();
  ctx.renderer.dispose();
  ctx.renderer.domElement.remove();
  ctx = null;
}

export async function showModel(host, url) {
  clearViewer(host);
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  host.prepend(renderer.domElement);

  const scene = new THREE.Scene();
  scene.add(new THREE.HemisphereLight(0xffffff, 0x404050, 2.2));
  const sun = new THREE.DirectionalLight(0xffffff, 1.2);
  sun.position.set(2, 3, 2);
  scene.add(sun);
  const grid = new THREE.GridHelper(2, 20, 0x3a4050, 0x262a33);
  scene.add(grid);

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 1.5;
  controls.addEventListener("start", () => (controls.autoRotate = false));

  const gltf = await new GLTFLoader().loadAsync(url);
  const model = gltf.scene;
  model.traverse((o) => {
    if (o.isMesh) {
      if (!o.geometry.attributes.normal) o.geometry.computeVertexNormals();
      const col = o.geometry.attributes.color;
      if (col) {
        // Los colores por vértice vienen en sRGB; three.js espera espacio lineal.
        const lin = new Float32Array(col.count * 3);
        const c = new THREE.Color();
        for (let i = 0; i < col.count; i++) {
          c.setRGB(col.getX(i), col.getY(i), col.getZ(i), THREE.SRGBColorSpace);
          lin[i * 3] = c.r; lin[i * 3 + 1] = c.g; lin[i * 3 + 2] = c.b;
        }
        o.geometry.setAttribute("color", new THREE.BufferAttribute(lin, 3));
      }
      o.material = new THREE.MeshStandardMaterial({
        vertexColors: !!o.geometry.attributes.color, roughness: 0.85, metalness: 0, side: THREE.DoubleSide,
      });
    }
  });
  scene.add(model);
  const box = new THREE.Box3().setFromObject(model);
  const size = box.getSize(new THREE.Vector3()).length();
  const center = box.getCenter(new THREE.Vector3());
  controls.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(0.6, 0.45, 0.9).multiplyScalar(size));

  const resize = () => {
    const w = host.clientWidth, h = host.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / Math.max(h, 1);
    camera.updateProjectionMatrix();
  };
  const ro = new ResizeObserver(resize);
  ro.observe(host);
  resize();
  const loop = () => {
    controls.update();
    renderer.render(scene, camera);
    ctx.raf = requestAnimationFrame(loop);
  };
  ctx = { renderer, ro, raf: 0 };
  loop();
}
