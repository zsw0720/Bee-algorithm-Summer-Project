// app.js

// State variables
let state = {
    selectedImage: null,
    selectedGT: null,
    imgWidth: 1024,
    imgHeight: 1024,
    bgImage: null,
    
    // Extracted features
    pixelTargets: [],
    targetNames: [],
    weldBoxes: [],
    pixelObstacles: [],
    physicsTargets: [],
    physicsObstacles: [],
    
    // Path optimization
    pixelPathSegments: [],
    physicsPathSegments: [],
    totalLength: null,
    isValid: false,
    convergenceHistories: [],
    
    // Animation controls
    animating: false,
    animFrameId: null,
    animPoints: [],
    currentFrame: 0,
    animationSpeed: 5 // 1 to 10
};

// UI Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const gtInput = document.getElementById('gt-input');
const fileNameDisplay = document.getElementById('file-name-display');
const fileNameText = document.getElementById('file-name-text');
const vlmModeSelect = document.getElementById('vlm-mode');

const btnInspect = document.getElementById('btn-inspect');
const btnDefaultInspect = document.getElementById('btn-default-inspect');
const btnPlan = document.getElementById('btn-plan');

const canvas = document.getElementById('ndt-canvas');
const ctx = canvas.getContext('2d');

const btnAnimPlay = document.getElementById('btn-anim-play');
const btnAnimPause = document.getElementById('btn-anim-pause');
const btnAnimReset = document.getElementById('btn-anim-reset');
const animSpeedSlider = document.getElementById('anim-speed');

// Chart instance
let convergenceChartInstance = null;

// Console log helper
function logToConsole(message, type = 'system') {
    const consoleBox = document.getElementById('console-logs');
    const line = document.createElement('div');
    line.className = `log-line ${type}`;
    const now = new Date().toLocaleTimeString();
    line.innerHTML = `<span style="color: #69748a">[${now}]</span> <i class="fa-solid fa-chevron-right"></i> ${message}`;
    consoleBox.appendChild(line);
    consoleBox.scrollTop = consoleBox.scrollHeight;
}

// 1. Parameter Sliders Event Listeners
const sliders = ['n', 'm', 'e', 'nep', 'nsp', 'max-it', 'waypoints', 'ngh'];
sliders.forEach(sliderId => {
    const slider = document.getElementById(`param-${sliderId}`);
    const valDisplay = document.getElementById(`val-${sliderId}`);
    if (slider && valDisplay) {
        slider.addEventListener('input', () => {
            valDisplay.textContent = slider.value;
        });
    }
});

// 2. File Drag & Drop Handling
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleImageSelection(e.dataTransfer.files[0]);
    }
});
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleImageSelection(e.target.files[0]);
    }
});
gtInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        state.selectedGT = e.target.files[0];
        logToConsole(`Ground Truth mask selected: ${state.selectedGT.name}`, 'system');
    }
});

function handleImageSelection(file) {
    state.selectedImage = file;
    fileNameText.textContent = file.name;
    fileNameDisplay.style.display = 'flex';
    logToConsole(`NDT image selected: ${file.name}`, 'system');
    
    // Draw raw image preview before inspection
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            state.bgImage = img;
            state.imgWidth = img.naturalWidth;
            state.imgHeight = img.naturalHeight;
            drawCanvas();
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

// 3. API - Inspect Image
btnInspect.addEventListener('click', () => runVisualDetection(false));
btnDefaultInspect.addEventListener('click', () => runVisualDetection(true));

function runVisualDetection(useDefault = false) {
    if (!useDefault && !state.selectedImage && !state.bgImage) {
        alert("Please select, drop, or choose a preloaded dataset image first!");
        return;
    }

    logToConsole(useDefault ? "Loading default plate metal_plate.png..." : (state.selectedImage ? "Uploading image for VLM visual inspection..." : "Requesting VLM visual inspection on loaded dataset image..."), "system");
    
    const formData = new FormData();
    if (!useDefault) {
        if (state.selectedImage) {
            formData.append('image', state.selectedImage);
            if (state.selectedGT) {
                formData.append('gt_mask', state.selectedGT);
            }
        }
    } else {
        formData.append('use_default', 'true');
    }
    formData.append('vlm_mode', vlmModeSelect.value);
    formData.append('user_prompt', document.getElementById('user-prompt').value.trim());

    // Disable buttons during load
    btnInspect.disabled = true;
    btnPlan.disabled = true;

    fetch('/api/inspect', {
        method: 'POST',
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        btnInspect.disabled = false;
        if (data.error) {
            logToConsole(`Inspection failed: ${data.error}`, 'warning');
            alert(data.error);
            return;
        }

        // Save response to state
        state.pixelTargets = data.pixel_targets;
        state.targetNames = data.target_names;
        state.weldBoxes = data.weld_boxes;
        state.pixelObstacles = data.pixel_obstacles;
        state.physicsTargets = data.physics_targets;
        state.physicsObstacles = data.physics_obstacles;
        state.imgWidth = data.width;
        state.imgHeight = data.height;

        logToConsole(`VLM visual recognition succeeded (${data.vlm_mode} mode).`, 'success');
        logToConsole(`Detected ${state.pixelTargets.length} waypoints and ${state.pixelObstacles.length} obstacles in pixel space.`, 'success');

        // Update evaluation metrics card
        updateMetrics(data.metrics);

        // Load background image
        const img = new Image();
        img.onload = () => {
            state.bgImage = img;
            drawCanvas();
            btnPlan.disabled = false; // Enable planning button
        };
        img.src = data.image_url + "?t=" + new Date().getTime(); // Avoid cache
    })
    .catch(err => {
        btnInspect.disabled = false;
        logToConsole(`Network error during inspection: ${err.message}`, 'warning');
    });
}

function updateMetrics(metrics) {
    const noGtWarning = document.getElementById('no-gt-warning');
    const circleIoU = document.getElementById('circle-iou');
    const textIoU = document.getElementById('metric-iou-text');
    const circleBoxIoU = document.getElementById('circle-box-iou');
    const textBoxIoU = document.getElementById('metric-box-iou-text');
    const circleF1 = document.getElementById('circle-f1');
    const textF1 = document.getElementById('metric-f1-text');
    const textPrecision = document.getElementById('metric-precision');
    const textRecall = document.getElementById('metric-recall');

    if (!metrics) {
        noGtWarning.style.display = 'flex';
        // Reset meters
        circleIoU.setAttribute('stroke-dasharray', '0, 100');
        textIoU.textContent = '-';
        circleBoxIoU.setAttribute('stroke-dasharray', '0, 100');
        textBoxIoU.textContent = '-';
        circleF1.setAttribute('stroke-dasharray', '0, 100');
        textF1.textContent = '-';
        textPrecision.textContent = '-';
        textRecall.textContent = '-';
        return;
    }

    noGtWarning.style.display = 'none';
    
    // Set animations and stroke dasharrays (value between 0 and 100)
    const iouPercent = Math.round(metrics.iou * 100);
    const boxIouPercent = Math.round((metrics.box_iou || 0) * 100);
    const f1Percent = Math.round(metrics.f1_score * 100);
    
    circleIoU.setAttribute('stroke-dasharray', `${iouPercent}, 100`);
    textIoU.textContent = `${iouPercent}%`;
    
    circleBoxIoU.setAttribute('stroke-dasharray', `${boxIouPercent}, 100`);
    textBoxIoU.textContent = `${boxIouPercent}%`;
    
    circleF1.setAttribute('stroke-dasharray', `${f1Percent}, 100`);
    textF1.textContent = `${f1Percent}%`;

    textPrecision.textContent = `${Math.round(metrics.precision * 100)}%`;
    textRecall.textContent = `${Math.round(metrics.recall * 100)}%`;

    logToConsole(`VLM defect evaluation accuracy: Pixel IoU = ${iouPercent}%, Box IoU = ${boxIouPercent}%, F1 = ${f1Percent}%`, 'success');
}

// 4. API - Run Path Optimization (Bees Algorithm)
btnPlan.addEventListener('click', () => {
    if (state.pixelTargets.length === 0) {
        alert("Run visual detection first to find weld targets!");
        return;
    }

    logToConsole("Optimizing scanning path segments with Bees Algorithm...", "system");
    btnPlan.disabled = true;

    const requestData = {
        n: parseInt(document.getElementById('param-n').value),
        m: parseInt(document.getElementById('param-m').value),
        e: parseInt(document.getElementById('param-e').value),
        nep: parseInt(document.getElementById('param-nep').value),
        nsp: parseInt(document.getElementById('param-nsp').value),
        max_it: parseInt(document.getElementById('param-max-it').value),
        waypoints_count: parseInt(document.getElementById('param-waypoints').value),
        ngh: parseFloat(document.getElementById('param-ngh').value),
        start_pos: [
            parseFloat(document.getElementById('start-x').value),
            parseFloat(document.getElementById('start-y').value)
        ],
        end_pos: [
            parseFloat(document.getElementById('end-x').value),
            parseFloat(document.getElementById('end-y').value)
        ],
        physics_targets: state.physicsTargets,
        physics_obstacles: state.physicsObstacles
    };

    fetch('/api/plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestData)
    })
    .then(res => res.json())
    .then(data => {
        btnPlan.disabled = false;
        if (data.error) {
            logToConsole(`Optimization failed: ${data.error}`, 'warning');
            alert(data.error);
            return;
        }

        state.pixelPathSegments = data.pixel_path_segments;
        state.physicsPathSegments = data.physics_path_segments;
        state.totalLength = data.total_length;
        state.isValid = data.is_valid;
        state.convergenceHistories = data.convergence_histories;

        logToConsole(`Path planning complete. Total Trajectory Length: ${state.totalLength.toFixed(2)} mm.`, 'success');
        
        // Update summary display
        document.getElementById('summary-length').textContent = `${state.totalLength.toFixed(1)} mm`;
        const badge = document.getElementById('summary-status');
        if (state.isValid) {
            badge.textContent = "SAFE";
            badge.className = "value badge success";
            logToConsole("Trajectory Status: SUCCESS (All segments completely collision-free)", 'success');
        } else {
            badge.textContent = "COLLISION WARNING";
            badge.className = "value badge warning";
            logToConsole("Trajectory Status: WARNING (Path collides with obstacle. Adjust parameters!)", 'warning');
        }

        // Draw path on canvas
        drawCanvas();
        
        // Render Chart.js
        renderConvergenceChart();

        // Start animation automatically
        setupAnimation();
    })
    .catch(err => {
        btnPlan.disabled = false;
        logToConsole(`Network error during path planning: ${err.message}`, 'warning');
    });
});

// 5. Canvas Drawing & Probe Animation
function drawCanvas() {
    if (!state.bgImage) return;

    // Set canvas dimensions to match image natural aspect ratio (but scaled display)
    canvas.width = state.imgWidth;
    canvas.height = state.imgHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 1. Draw Background Image
    ctx.drawImage(state.bgImage, 0, 0, canvas.width, canvas.height);

    // 2. Draw Obstacles (Red boxes)
    ctx.fillStyle = 'rgba(255, 49, 49, 0.25)';
    ctx.strokeStyle = '#ff3131';
    ctx.lineWidth = 3;
    state.pixelObstacles.forEach(obs => {
        // obs: [u_min, v_min, u_max, v_max]
        const w = obs[2] - obs[0];
        const h = obs[3] - obs[1];
        ctx.fillRect(obs[0], obs[1], w, h);
        ctx.strokeRect(obs[0], obs[1], w, h);
    });

    // 3. Draw Weld bounding boxes (Cyan dashed boxes)
    ctx.strokeStyle = 'rgba(0, 255, 255, 0.4)';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 6]);
    state.weldBoxes.forEach(box => {
        const w = box[2] - box[0];
        const h = box[3] - box[1];
        ctx.strokeRect(box[0], box[1], w, h);
    });
    ctx.setLineDash([]); // Reset line dash

    // 4. Draw Waypoints (glowing rings)
    state.pixelTargets.forEach((pt, idx) => {
        ctx.beginPath();
        ctx.arc(pt[0], pt[1], 10, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0, 255, 255, 0.8)';
        ctx.shadowColor = '#00ffff';
        ctx.shadowBlur = 8;
        ctx.fill();
        ctx.shadowBlur = 0; // Reset shadow

        ctx.beginPath();
        ctx.arc(pt[0], pt[1], 12, 0, Math.PI * 2);
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Label waypoints
        ctx.font = "bold 14px Outfit";
        ctx.fillStyle = "white";
        ctx.shadowColor = "black";
        ctx.shadowBlur = 4;
        ctx.fillText(`T${idx+1}`, pt[0] + 16, pt[1] + 5);
        ctx.shadowBlur = 0;
    });

    // 5. Draw Start & End positions
    // Start position
    const startX = parseFloat(document.getElementById('start-x').value);
    const startY = parseFloat(document.getElementById('start-y').value);
    const endX = parseFloat(document.getElementById('end-x').value);
    const endY = parseFloat(document.getElementById('end-y').value);
    
    // Mapped to pixels
    const uS = (startX / 100.0) * canvas.width;
    const vS = (1.0 - (startY / 100.0)) * canvas.height;
    const uE = (endX / 100.0) * canvas.width;
    const vE = (1.0 - (endY / 100.0)) * canvas.height;

    // Draw Start star
    drawStar(uS, vS, 5, 14, 7, '#ffff33', '#ffff33');
    ctx.font = "bold 12px Outfit";
    ctx.fillStyle = "#ffff33";
    ctx.fillText("Start", uS + 12, vS + 4);

    // Draw End cross
    drawCross(uE, vE, 10, '#ff9900');
    ctx.fillStyle = "#ff9900";
    ctx.fillText("End", uE + 12, vE + 4);

    // 6. Draw Optimized Path (Neon green)
    if (state.pixelPathSegments.length > 0) {
        ctx.beginPath();
        ctx.strokeStyle = '#39ff14';
        ctx.lineWidth = 4;
        ctx.shadowColor = '#39ff14';
        ctx.shadowBlur = 4;
        
        let pathStarted = false;
        state.pixelPathSegments.forEach(seg => {
            seg.forEach((pt, idx) => {
                if (!pathStarted) {
                    ctx.moveTo(pt[0], pt[1]);
                    pathStarted = true;
                } else {
                    ctx.lineTo(pt[0], pt[1]);
                }
            });
        });
        ctx.stroke();
        ctx.shadowBlur = 0; // Reset shadow
    }
}

// Draw star helper
function drawStar(cx, cy, spikes, outerRadius, innerRadius, fillStyle, strokeStyle) {
    let rot = Math.PI / 2 * 3;
    let x = cx;
    let y = cy;
    let step = Math.PI / spikes;

    ctx.beginPath();
    ctx.moveTo(cx, cy - outerRadius);
    for (let i = 0; i < spikes; i++) {
        x = cx + Math.cos(rot) * outerRadius;
        y = cy + Math.sin(rot) * outerRadius;
        ctx.lineTo(x, y);
        rot += step;

        x = cx + Math.cos(rot) * innerRadius;
        y = cy + Math.sin(rot) * innerRadius;
        ctx.lineTo(x, y);
        rot += step;
    }
    ctx.lineTo(cx, cy - outerRadius);
    ctx.closePath();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = strokeStyle;
    ctx.stroke();
    ctx.fillStyle = fillStyle;
    ctx.fill();
}

// Draw cross helper
function drawCross(cx, cy, size, strokeStyle) {
    ctx.beginPath();
    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = 3;
    ctx.moveTo(cx - size, cy - size);
    ctx.lineTo(cx + size, cy + size);
    ctx.moveTo(cx + size, cy - size);
    ctx.lineTo(cx - size, cy + size);
    ctx.stroke();
}

// Setup points for linear animation
function setupAnimation() {
    if (state.pixelPathSegments.length === 0) return;

    // Flatten segments into smooth animation points
    state.animPoints = [];
    
    // Flatten segments
    let flatPath = [];
    state.pixelPathSegments.forEach((seg, sIdx) => {
        if (sIdx === 0) {
            flatPath.push(...seg);
        } else {
            flatPath.push(...seg.slice(1)); // Avoid duplicated target points
        }
    });

    // Interpolate points for constant speed
    for (let i = 0; i < flatPath.length - 1; i++) {
        const p1 = flatPath[i];
        const p2 = flatPath[i+1];
        const dist = Math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2);
        
        // Generate steps proportional to distance
        const steps = Math.max(3, Math.floor(dist * 0.1));
        for (let t = 0; t < steps; t++) {
            const ratio = t / steps;
            const x = p1[0] + ratio * (p2[0] - p1[0]);
            const y = p1[1] + ratio * (p2[1] - p1[1]);
            state.animPoints.push([x, y]);
        }
    }
    state.animPoints.push(flatPath[flatPath.length - 1]);

    state.currentFrame = 0;
    startAnimation();
}

function startAnimation() {
    if (state.animating) return;
    state.animating = true;
    animateProbe();
}

function animateProbe() {
    if (!state.animating) return;

    // 1. Redraw static canvas elements (image, path, targets)
    drawCanvas();

    // 2. Draw trailing path line from start to current position
    ctx.beginPath();
    ctx.strokeStyle = '#39ff14';
    ctx.lineWidth = 2.5;
    ctx.setLineDash([4, 4]);
    for (let i = 0; i <= state.currentFrame; i++) {
        const pt = state.animPoints[i];
        if (i === 0) {
            ctx.moveTo(pt[0], pt[1]);
        } else {
            ctx.lineTo(pt[0], pt[1]);
        }
    }
    ctx.stroke();
    ctx.setLineDash([]); // Reset line dash

    // 3. Draw Probe (glowing circular dot)
    const currentPt = state.animPoints[state.currentFrame];
    if (currentPt) {
        ctx.beginPath();
        ctx.arc(currentPt[0], currentPt[1], 10, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff';
        ctx.shadowColor = '#ffffff';
        ctx.shadowBlur = 12;
        ctx.fill();
        ctx.shadowBlur = 0; // Reset shadow

        ctx.beginPath();
        ctx.arc(currentPt[0], currentPt[1], 12, 0, Math.PI * 2);
        ctx.strokeStyle = '#39ff14';
        ctx.lineWidth = 2;
        ctx.stroke();
    }

    // 4. Update frame
    // Advance speed
    state.currentFrame += Math.round(state.animationSpeed * 0.5) || 1;
    if (state.currentFrame >= state.animPoints.length) {
        state.currentFrame = 0; // Loop animation
    }

    state.animFrameId = requestAnimationFrame(animateProbe);
}

// Animation controls events
btnAnimPlay.addEventListener('click', () => {
    if (state.pixelPathSegments.length === 0) return;
    startAnimation();
});
btnAnimPause.addEventListener('click', () => {
    state.animating = false;
    if (state.animFrameId) {
        cancelAnimationFrame(state.animFrameId);
    }
});
btnAnimReset.addEventListener('click', () => {
    state.animating = false;
    if (state.animFrameId) {
        cancelAnimationFrame(state.animFrameId);
    }
    state.currentFrame = 0;
    drawCanvas();
});
animSpeedSlider.addEventListener('input', () => {
    state.animationSpeed = parseInt(animSpeedSlider.value);
});


// 6. Chart.js Convergence curve plotting
function renderConvergenceChart() {
    if (state.convergenceHistories.length === 0) return;

    const chartCanvas = document.getElementById('convergenceChart');
    const ctxChart = chartCanvas.getContext('2d');

    // Destroy old instance if exists
    if (convergenceChartInstance) {
        convergenceChartInstance.destroy();
    }

    const datasets = state.convergenceHistories.map((history, idx) => {
        // Map fitness history to cost history (Cost = 1 / Fitness)
        const costs = history.map(fit => 1.0 / fit);
        
        // Colors
        const colors = [
            'rgba(0, 255, 255, 1)',   // Cyan
            'rgba(57, 255, 20, 1)',   // Green
            'rgba(108, 92, 231, 1)',  // Purple
            'rgba(255, 49, 49, 1)',   // Red
            'rgba(255, 255, 51, 1)',  // Yellow
            'rgba(255, 153, 0, 1)'    // Orange
        ];
        const color = colors[idx % colors.length];

        return {
            label: `Segment ${idx+1}`,
            data: costs,
            borderColor: color,
            backgroundColor: color.replace('1)', '0.05)'),
            borderWidth: 2,
            tension: 0.1,
            pointRadius: 0
        };
    });

    const numIterations = state.convergenceHistories[0].length;
    const labels = Array.from({ length: numIterations }, (_, i) => i + 1);

    convergenceChartInstance = new Chart(ctxChart, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: '#a0aec0',
                        font: { family: 'Outfit', size: 9 }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Iteration',
                        color: '#a0aec0',
                        font: { family: 'Outfit' }
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: { color: '#a0aec0' }
                },
                y: {
                    type: 'logarithmic',
                    title: {
                        display: true,
                        text: 'Path Cost (Length + Penalty)',
                        color: '#a0aec0',
                        font: { family: 'Outfit' }
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: {
                        color: '#a0aec0',
                        callback: function(value, index, values) {
                            return Number(value).toString(); // Format log ticks nicely
                        }
                    }
                }
            }
        }
    });
}

// 7. Initialize Dataset Browser Panel
let datasetStructure = {};
function initDatasetBrowser() {
    const classSelect = document.getElementById('dataset-class');
    const catSelect = document.getElementById('dataset-category');
    const imgSelect = document.getElementById('dataset-image');
    const btnLoad = document.getElementById('btn-load-dataset');
    
    fetch('/api/dataset/list')
        .then(res => res.json())
        .then(data => {
            datasetStructure = data;
            
            // Populate class dropdown
            classSelect.innerHTML = '<option value="">-- Choose Class --</option>';
            Object.keys(data).forEach(cls => {
                const opt = document.createElement('option');
                opt.value = cls;
                opt.textContent = cls.replace('_', ' ').toUpperCase();
                classSelect.appendChild(opt);
            });
        })
        .catch(err => console.error("Failed to load dataset list:", err));
        
    // Change listener for Class
    classSelect.addEventListener('change', () => {
        const cls = classSelect.value;
        catSelect.innerHTML = '<option value="">-- Choose Category --</option>';
        imgSelect.innerHTML = '<option value="">-- Choose Image --</option>';
        catSelect.disabled = true;
        imgSelect.disabled = true;
        btnLoad.disabled = true;
        
        if (cls && datasetStructure[cls]) {
            catSelect.disabled = false;
            Object.keys(datasetStructure[cls]).forEach(cat => {
                const opt = document.createElement('option');
                opt.value = cat;
                opt.textContent = cat.replace('_', ' ').toUpperCase();
                catSelect.appendChild(opt);
            });
        }
    });
    
    // Change listener for Category
    catSelect.addEventListener('change', () => {
        const cls = classSelect.value;
        const cat = catSelect.value;
        imgSelect.innerHTML = '<option value="">-- Choose Image --</option>';
        imgSelect.disabled = true;
        btnLoad.disabled = true;
        
        if (cls && cat && datasetStructure[cls][cat]) {
            imgSelect.disabled = false;
            datasetStructure[cls][cat].forEach(img => {
                const opt = document.createElement('option');
                opt.value = img;
                opt.textContent = img;
                imgSelect.appendChild(opt);
            });
        }
    });
    
    // Change listener for Image
    imgSelect.addEventListener('change', () => {
        btnLoad.disabled = !imgSelect.value;
    });
    
    // Click listener for Load button
    btnLoad.addEventListener('click', () => {
        const cls = classSelect.value;
        const cat = catSelect.value;
        const imgName = imgSelect.value;
        
        logToConsole(`Loading preloaded image: ${cls}/${cat}/${imgName}...`, 'system');
        btnLoad.disabled = true;
        
        fetch('/api/dataset/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ class: cls, category: cat, image: imgName })
        })
        .then(res => res.json())
        .then(data => {
            btnLoad.disabled = false;
            if (data.error) {
                logToConsole(`Failed to load dataset image: ${data.error}`, 'warning');
                alert(data.error);
                return;
            }
            
            // Set state variables
            state.selectedImage = null; // Reset manually uploaded file state
            state.selectedGT = null;
            
            // Hide uploaded file name displays
            document.getElementById('file-name-display').style.display = 'none';
            document.getElementById('file-name-text').textContent = 'None';
            document.getElementById('gt-input').value = '';
            
            state.imgWidth = data.width;
            state.imgHeight = data.height;
            
            // Load background image
            const img = new Image();
            img.onload = () => {
                state.bgImage = img;
                drawCanvas();
                
                // Clear previous features and paths
                state.pixelTargets = [];
                state.weldBoxes = [];
                state.pixelObstacles = [];
                document.getElementById('btn-plan').disabled = true;
                updateMetrics(null); // Reset metrics card
                
                logToConsole(`Preloaded image loaded successfully. Ground truth mask: ${data.gt_url ? 'Auto-loaded' : 'None (Good sample)'}`, 'success');
            };
            img.src = data.image_url + "?t=" + new Date().getTime();
        })
        .catch(err => {
            btnLoad.disabled = false;
            logToConsole(`Network error loading dataset image: ${err.message}`, 'warning');
        });
    });
}

// Call dataset initialization on script run
initDatasetBrowser();
