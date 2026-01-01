(function () {
    const container = document.getElementById('shader-canvas');
    if (!container) return;

    const scene = new THREE.Scene();
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
    camera.position.z = 1;

    const renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
        powerPreference: "high-performance"
    });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    container.appendChild(renderer.domElement);

    const vertexShader = `
        varying vec2 vUv;
        void main() {
            vUv = uv;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
    `;

    const fragmentShader = `
        precision highp float;
        
        uniform float uTime;
        uniform vec2 uResolution;
        uniform vec2 uMouse;
        
        varying vec2 vUv;
        
        #define PI 3.14159265359
        
        float hash(vec2 p) {
            return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
        }
        
        float noise(vec2 p) {
            vec2 i = floor(p);
            vec2 f = fract(p);
            f = f * f * (3.0 - 2.0 * f);
            
            float a = hash(i);
            float b = hash(i + vec2(1.0, 0.0));
            float c = hash(i + vec2(0.0, 1.0));
            float d = hash(i + vec2(1.0, 1.0));
            
            return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
        }
        
        float fbm(vec2 p) {
            float value = 0.0;
            float amplitude = 0.5;
            float frequency = 1.0;
            
            for(int i = 0; i < 4; i++) {
                value += amplitude * noise(p * frequency);
                frequency *= 2.0;
                amplitude *= 0.5;
            }
            return value;
        }
        
        float smokeLayer(vec2 uv, float time, float speed, float scale) {
            vec2 movement = vec2(time * speed * 0.3, time * speed * 0.2);
            float n1 = fbm(uv * scale + movement);
            float n2 = fbm(uv * scale * 1.5 - movement * 0.7 + vec2(5.2, 1.3));
            float n3 = fbm(uv * scale * 0.8 + vec2(n1, n2) * 0.5);
            return n3;
        }
        
        void main() {
            vec2 uv = vUv;
            vec2 aspect = vec2(uResolution.x / uResolution.y, 1.0);
            uv = uv * aspect;
            
            float time = uTime * 0.4;
            
            vec2 mouseInfluence = (uMouse - 0.5) * 0.1;
            uv += mouseInfluence * 0.3;
            
            float smoke1 = smokeLayer(uv, time, 0.5, 2.0);
            float smoke2 = smokeLayer(uv + vec2(3.14, 2.71), time * 0.9, 0.4, 2.2);
            float smoke3 = smokeLayer(uv - vec2(1.41, 1.73), time * 1.1, 0.6, 1.8);
            
            float combinedSmoke = smoke1 * 0.4 + smoke2 * 0.35 + smoke3 * 0.25;
            
            combinedSmoke = pow(combinedSmoke, 0.9);
            combinedSmoke = smoothstep(0.2, 0.8, combinedSmoke);
            
            vec3 deepGreen = vec3(0.067, 0.180, 0.133);
            vec3 midGreen = vec3(0.110, 0.290, 0.200);
            vec3 lightGreen = vec3(0.176, 0.412, 0.310);
            vec3 paleGreen = vec3(0.584, 0.835, 0.698);
            vec3 softWhite = vec3(0.940, 0.969, 0.953);
            
            vec3 color;
            if(combinedSmoke < 0.25) {
                color = mix(deepGreen, midGreen, combinedSmoke / 0.25);
            } else if(combinedSmoke < 0.5) {
                color = mix(midGreen, lightGreen, (combinedSmoke - 0.25) / 0.25);
            } else if(combinedSmoke < 0.75) {
                color = mix(lightGreen, paleGreen, (combinedSmoke - 0.5) / 0.25);
            } else {
                color = mix(paleGreen, softWhite, (combinedSmoke - 0.75) / 0.25);
            }
            
            float glow = smoothstep(0.4, 0.9, combinedSmoke) * 0.15;
            color += vec3(0.1, 0.3, 0.2) * glow;
            
            float vignette = 1.0 - length((vUv - 0.5) * 1.2);
            vignette = smoothstep(0.0, 0.7, vignette);
            color *= 0.85 + vignette * 0.15;
            
            float grain = hash(vUv * uResolution + uTime) * 0.015;
            color += grain;
            
            gl_FragColor = vec4(color, 1.0);
        }
    `;

    const uniforms = {
        uTime: { value: 0.0 },
        uResolution: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) },
        uMouse: { value: new THREE.Vector2(0.5, 0.5) }
    };

    const geometry = new THREE.PlaneGeometry(2, 2);
    const material = new THREE.ShaderMaterial({
        vertexShader,
        fragmentShader,
        uniforms
    });

    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);

    let mouseX = 0.5, mouseY = 0.5;
    let targetMouseX = 0.5, targetMouseY = 0.5;

    document.addEventListener('mousemove', (e) => {
        targetMouseX = e.clientX / window.innerWidth;
        targetMouseY = 1.0 - (e.clientY / window.innerHeight);
    });

    window.addEventListener('resize', () => {
        renderer.setSize(window.innerWidth, window.innerHeight);
        uniforms.uResolution.value.set(window.innerWidth, window.innerHeight);
    });

    function animate(timestamp) {
        uniforms.uTime.value = timestamp * 0.001;

        mouseX += (targetMouseX - mouseX) * 0.05;
        mouseY += (targetMouseY - mouseY) * 0.05;
        uniforms.uMouse.value.set(mouseX, mouseY);

        renderer.render(scene, camera);
        requestAnimationFrame(animate);
    }

    requestAnimationFrame(animate);
})();
