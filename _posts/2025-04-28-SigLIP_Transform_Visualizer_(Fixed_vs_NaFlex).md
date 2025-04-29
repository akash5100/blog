---
layout: null
permalink: /siglip-transform-visualizer/
title: SigLIP Transform Visualizer (Fixed vs NaFlex)
tags: Representation Learning
---

<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SigLIP Transform Visualizer (Fixed vs. NaFlex)</title>
    <style>
        body {
            font-family: sans-serif;
            line-height: 1.6;
            margin: 0; /* Remove default margin */
            padding: 20px;
            background-color: #f4f4f4;
        }
        h1, h2 {
            text-align: center;
            color: #333;
        }
        .container {
            max-width: 1400px; /* Increased width for more boxes */
            margin: auto;
            background: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .controls, .visualization-area {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-around; /* Distribute space */
            align-items: flex-start; /* Align items top */
            margin-bottom: 20px;
            padding-bottom: 20px;
            border-bottom: 1px solid #eee;
        }
         .controls > div {
            margin: 10px;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #f9f9f9;
         }
         .controls label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
         }
         .controls input[type="number"],
         .controls input[type="text"], /* Changed for comma-separated */
         .controls input[type="file"] {
             /* width: 100px; */ /* Let them size naturally */
             padding: 8px;
             margin-top: 5px;
             box-sizing: border-box;
             border: 1px solid #ccc;
             border-radius: 4px;
         }
          .controls input[type="text"] {
             width: 250px; /* Wider for comma list */
          }
         .controls input[type="file"] {
             width: auto;
         }
        .image-box {
            border: 1px solid #ccc;
            padding: 15px;
            margin: 10px;
            text-align: center;
            background-color: #f8f8f8;
            border-radius: 5px;
            flex: 1; /* Equal flex distribution */
            min-width: 280px; /* Adjusted min-width */
            max-width: 350px; /* Added max-width */
            display: flex;
            flex-direction: column;
            align-items: center;
            position: relative; /* Needed for magnifier positioning context */
        }
        .image-box h3 {
            margin-top: 0;
            color: #555;
            font-size: 1.1em;
        }
        .image-box img, .image-box canvas {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 10px auto;
            background-color: #e0e0e0; /* Canvas background for padding viz */
            border: 1px dashed #aaa;
            cursor: crosshair; /* Indicate zoom possible */
        }
         .image-box canvas {
             max-height: 300px;
             width: auto;
             max-width: 100%;
         }
        .image-box p {
            margin-top: 10px;
            font-size: 0.9em;
            color: #444;
            word-wrap: break-word;
        }
        .image-box .info {
            font-weight: bold;
            color: #0066cc;
        }
         #placeholder {
            text-align: center;
            font-style: italic;
            color: #777;
            width: 100%;
            padding: 50px 0;
         }
        /* --- Magnifier Styles --- */
        #magnifier {
            position: absolute;
            border: 3px solid lightgray;
            border-radius: 50%; /* Circular */
            width: 150px;  /* Magnifier size */
            height: 150px;
            cursor: none; /* Hide cursor over magnifier */
            display: none; /* Hidden by default */
            background-repeat: no-repeat;
            z-index: 10;
            pointer-events: none; /* Prevent magnifier from blocking mouse events on canvas */
            box-shadow: 0 0 10px rgba(0,0,0,0.3);
            background-color: #fff; /* Fallback bg */
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>SigLIP Transform Visualizer</h1>
        <p style="text-align: center;">Compare Fixed Resolution vs. NaFlex image preprocessing for ViTs.</p>
        <div class="controls">
            <div>
                <label for="imageUpload">Upload Image:</label>
                <input type="file" id="imageUpload" accept="image/*">
            </div>
            <div>
                <label for="patchSize">Patch Size (e.g., 14, 16):</label>
                <input type="number" id="patchSize" value="16" min="1">
            </div>
            <div>
                <label for="fixedRes">Fixed Resolution (e.g., 224, 384):</label>
                <input type="number" id="fixedRes" value="384" min="1">
            </div>
             <div>
                <label for="naflexSeqLens">NaFlex Target Seq Lens (comma-sep):</label>
                <input type="text" id="naflexSeqLens" value="256, 576, 1024">
            </div>
        </div>
        <div class="visualization-area" id="visualizationArea">
             <p id="placeholder">Upload an image and adjust parameters to see the transforms.</p>
            <!-- Image boxes will be added here by JS -->
        </div>
        <!-- Magnifier Element (add outside the container or adjust z-index/positioning) -->
        <div id="magnifier"></div>
    </div>
    <script>
        const imageUpload = document.getElementById('imageUpload');
        const patchSizeInput = document.getElementById('patchSize');
        const fixedResInput = document.getElementById('fixedRes');
        const naflexSeqLensInput = document.getElementById('naflexSeqLens'); // Renamed
        const visualizationArea = document.getElementById('visualizationArea');
        const placeholder = document.getElementById('placeholder');
        const magnifier = document.getElementById('magnifier'); // Get magnifier element

        let currentImage = null;
        const zoomLevel = 2.5; // Magnification factor

        imageUpload.addEventListener('change', handleImageUpload);
        patchSizeInput.addEventListener('input', processCurrentImage);
        fixedResInput.addEventListener('input', processCurrentImage);
        naflexSeqLensInput.addEventListener('input', processCurrentImage); // Listener for new input

        function handleImageUpload(event) {
            const file = event.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = function(e) {
                const img = new Image();
                img.onload = () => {
                    currentImage = img;
                    processCurrentImage();
                };
                img.onerror = () => {
                    alert("Could not load image file.");
                    currentImage = null;
                    clearVisualization();
                };
                img.src = e.target.result;
            }
            reader.readAsDataURL(file);
        }

        function clearVisualization() {
            visualizationArea.innerHTML = '';
            visualizationArea.appendChild(placeholder);
            placeholder.style.display = 'block';
        }

        function processCurrentImage() {
            if (!currentImage) {
                if (!imageUpload.files || imageUpload.files.length === 0) {
                    clearVisualization();
                }
                return;
            }

            placeholder.style.display = 'none';
            visualizationArea.innerHTML = ''; // Clear previous results

            const patchSize = parseInt(patchSizeInput.value) || 16;
            const fixedRes = parseInt(fixedResInput.value) || 384;
            const naflexSeqLensStr = naflexSeqLensInput.value || "256"; // Default if empty

            // Parse comma-separated sequence lengths
            const targetSeqLens = naflexSeqLensStr
                .split(',')
                .map(s => parseInt(s.trim()))
                .filter(n => !isNaN(n) && n > 0); // Filter valid positive integers

            if (patchSize <= 0 || fixedRes <= 0 || targetSeqLens.length === 0) {
                alert("Parameters (Patch Size, Fixed Res) must be positive numbers, and at least one valid NaFlex Seq Len is required.");
                clearVisualization();
                return;
            }

            const img = currentImage;
            const originalW = img.naturalWidth;
            const originalH = img.naturalHeight;

            // --- 1. Fixed Resolution Transform ---
            const fixedBox = createImageBox('Fixed Resolution');
            const fixedCanvas = document.createElement('canvas');
            setupFixedCanvas(fixedCanvas, img, fixedRes, patchSize, fixedBox);
            visualizationArea.appendChild(fixedBox);

            // --- 2. Original Image ---
            const originalBox = createImageBox('Original');
            setupOriginalBox(originalBox, img, originalW, originalH);
            visualizationArea.appendChild(originalBox);

            // --- 3. NaFlex Transforms (Loop through specified lengths) ---
            targetSeqLens.forEach(targetSeqLen => {
                const naflexBox = createImageBox(`NaFlex (Seq Len: ${targetSeqLen})`);
                const naflexCanvas = document.createElement('canvas');
                setupNaflexCanvas(naflexCanvas, img, patchSize, targetSeqLen, naflexBox);
                visualizationArea.appendChild(naflexBox);

                // Add hover zoom listeners to this NaFlex canvas
                addMagnifierEvents(naflexCanvas);
            });
        }

        // --- Helper Functions for Setup ---

        function setupFixedCanvas(canvas, img, fixedRes, patchSize, box) {
            canvas.width = fixedRes;
            canvas.height = fixedRes;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, fixedRes, fixedRes); // Resize (and distort)

            const tokensPerRow = fixedRes / patchSize;
            const fixedTokens = tokensPerRow * tokensPerRow;

            box.appendChild(canvas);
            addInfo(box, `Target Res: <span class="info">${fixedRes} x ${fixedRes}</span>`);
            addInfo(box, `Input Tokens: <span class="info">${Math.round(fixedTokens)}</span> (${isValidGrid(tokensPerRow) ? `${tokensPerRow} x ${tokensPerRow}` : 'Invalid Patch'}) patches`);
             // AddMagnifierEvents(canvas); // Optionally add magnifier to fixed res too
        }

         function setupOriginalBox(box, img, originalW, originalH) {
             const originalImgElement = document.createElement('img');
             originalImgElement.src = img.src;
             originalImgElement.style.maxWidth = '100%'; // Use container max width
             originalImgElement.style.maxHeight = '280px'; // Limit display height
             originalImgElement.style.border = 'none';
             originalImgElement.style.background = 'none';
             originalImgElement.style.cursor = 'default'; // No zoom on original

             box.appendChild(originalImgElement);
             addInfo(box, `Dimensions: <span class="info">${originalW} x ${originalH}</span>`);
        }

        function setupNaflexCanvas(canvas, img, patchSize, targetSeqLen, box) {
            const originalW = img.naturalWidth;
            const originalH = img.naturalHeight;
            const naflexCtx = canvas.getContext('2d');

            // Calculate NaFlex dimensions
            const { targetW, targetH, actualPatchesW, actualPatchesH, finalTokens } = calculateNaflexDims(
                originalW, originalH, patchSize, targetSeqLen
            );

            if (targetW <= 0 || targetH <= 0 || finalTokens <= 0) {
                 addInfo(box, `Calculation Error: Check inputs.`)
                 addInfo(box, `(Target Seq Len: ${targetSeqLen})`);
                 return; // Skip drawing if dims are invalid
            }

            // Set canvas size
            canvas.width = targetW;
            canvas.height = targetH;

            // Draw image preserving aspect ratio (letterbox/pillarbox)
            const { drawW, drawH, offsetX, offsetY } = calculateDrawParams(originalW, originalH, targetW, targetH);

            naflexCtx.fillStyle = '#e0e0e0'; // Background for padding
            naflexCtx.fillRect(0, 0, targetW, targetH);
            naflexCtx.drawImage(img, offsetX, offsetY, drawW, drawH);

            box.appendChild(canvas);
            addInfo(box, `Grid Res: <span class="info">${targetW} x ${targetH}</span>`);
            addInfo(box, `Input Tokens: <span class="info">${finalTokens}</span> (${isValidGrid(actualPatchesW) ? `${actualPatchesW} x ${actualPatchesH}` : 'Invalid Grid'}) patches`);
            addInfo(box, `(Target Seq Len: ${targetSeqLen})`);
        }


        // --- Calculation Helper Functions ---

         function calculateNaflexDims(originalW, originalH, patchSize, targetSeqLen) {
            let patchesW = Math.ceil(originalW / patchSize);
            let patchesH = Math.ceil(originalH / patchSize);
            let totalPatches = patchesW * patchesH;

            if (patchesW <= 0 || patchesH <= 0) return { targetW: 0, targetH: 0, actualPatchesW: 0, actualPatchesH: 0, finalTokens: 0 }; // Avoid division by zero etc.

            let actualPatchesW, actualPatchesH;

            if (totalPatches <= targetSeqLen) {
                actualPatchesW = patchesW;
                actualPatchesH = patchesH;
            } else {
                const scale = Math.sqrt(targetSeqLen / totalPatches);
                actualPatchesW = Math.floor(patchesW * scale);
                actualPatchesH = Math.floor(patchesH * scale);

                actualPatchesW = Math.max(1, actualPatchesW);
                actualPatchesH = Math.max(1, actualPatchesH);

                // Iteratively reduce dimensions if product exceeds target, prioritizing aspect ratio
                while (actualPatchesW * actualPatchesH > targetSeqLen && (actualPatchesW > 1 || actualPatchesH > 1)) {
                     const currentTotal = actualPatchesW * actualPatchesH;
                     // Try reducing width
                     const potentialTotalW = (actualPatchesW - 1) * actualPatchesH;
                     // Try reducing height
                     const potentialTotalH = actualPatchesW * (actualPatchesH - 1);

                     // Check which reduction keeps more patches (closer to target) or maintains aspect ratio better
                     // Simple approach: reduce the larger dimension if possible
                     if (actualPatchesW > actualPatchesH && actualPatchesW > 1) {
                         actualPatchesW--;
                     } else if (actualPatchesH > 1) {
                         actualPatchesH--;
                     } else if (actualPatchesW > 1) { // Only width is > 1
                          actualPatchesW--;
                     } else {
                         break; // Cannot reduce further
                     }
                }
            }

             const targetW = actualPatchesW * patchSize;
             const targetH = actualPatchesH * patchSize;
             const finalTokens = actualPatchesW * actualPatchesH;

             return { targetW, targetH, actualPatchesW, actualPatchesH, finalTokens };
         }

         function calculateDrawParams(imgW, imgH, canvasW, canvasH) {
             const imgAspectRatio = imgW / imgH;
             const canvasAspectRatio = canvasW / canvasH;
             let drawW = canvasW, drawH = canvasH;
             let offsetX = 0, offsetY = 0;

             if (imgAspectRatio > canvasAspectRatio) { // Image wider rel to canvas
                 drawH = canvasW / imgAspectRatio;
                 offsetY = (canvasH - drawH) / 2;
             } else { // Image taller or same ratio rel to canvas
                 drawW = canvasH * imgAspectRatio;
                 offsetX = (canvasW - drawW) / 2;
             }
             return { drawW, drawH, offsetX, offsetY };
         }

         function isValidGrid(val) {
             // Check if a value is a reasonably valid grid dimension number
             return Number.isFinite(val) && val > 0 && val < 10000; // Basic sanity check
         }


        // --- UI Helper Functions ---

        function createImageBox(title) {
            const box = document.createElement('div');
            box.classList.add('image-box');
            const h3 = document.createElement('h3');
            h3.textContent = title;
            box.appendChild(h3);
            return box;
        }

        function addInfo(box, text) {
            const p = document.createElement('p');
            p.innerHTML = text; // Use innerHTML for span styles
            box.appendChild(p);
        }


        // --- Magnifier Logic ---

        let currentMagnifiedCanvas = null; // Track which canvas is being magnified

        function addMagnifierEvents(canvas) {
            // Store the image data URL once on the element for efficiency
            canvas.dataset.imageDataUrl = canvas.toDataURL();

            canvas.addEventListener('mouseenter', handleMouseEnter);
            canvas.addEventListener('mousemove', handleMouseMove);
            canvas.addEventListener('mouseleave', handleMouseLeave);
            // Update data URL if canvas content changes (e.g., parameters update)
             canvas.addEventListener('update', () => {
                 canvas.dataset.imageDataUrl = canvas.toDataURL();
             });
        }

         // Trigger update event helper (call after drawing/redrawing a canvas)
         function triggerCanvasUpdate(canvas) {
             canvas.dispatchEvent(new Event('update'));
         }

         // Call this after drawing on canvases that need zoom
         // In setupFixedCanvas: triggerCanvasUpdate(canvas);
         // In setupNaflexCanvas: triggerCanvasUpdate(canvas);


        function handleMouseEnter(e) {
            const canvas = e.target;
            currentMagnifiedCanvas = canvas; // Set current canvas
            const imgDataUrl = canvas.dataset.imageDataUrl;
            if (!imgDataUrl) return; // Safety check

            magnifier.style.backgroundImage = `url(${imgDataUrl})`;
            magnifier.style.backgroundSize = `${canvas.width * zoomLevel}px ${canvas.height * zoomLevel}px`;
            magnifier.style.display = 'block';
        }

        function handleMouseLeave(e) {
            magnifier.style.display = 'none';
            currentMagnifiedCanvas = null; // Clear current canvas
        }

        function handleMouseMove(e) {
            if (!currentMagnifiedCanvas || currentMagnifiedCanvas !== e.target) {
                 magnifier.style.display = 'none'; // Hide if mouse moved off tracked canvas
                 return;
             }
             const canvas = currentMagnifiedCanvas;

            const rect = canvas.getBoundingClientRect(); // Position of canvas on screen
            const scrollX = window.scrollX || window.pageXOffset;
            const scrollY = window.scrollY || window.pageYOffset;

            // Mouse position relative to the document
            const mousePageX = e.pageX;
            const mousePageY = e.pageY;

            // Mouse position relative to the canvas element
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            // --- Calculate background position ---
            // Map mouse coords on the *element* to coords on the *canvas internal resolution*
            const bgX = (x / canvas.clientWidth) * canvas.width;
            const bgY = (y / canvas.clientHeight) * canvas.height;

            // Center the magnified view around the cursor
            // Calculate top-left corner of the background area to show
            let backgroundPosX = -(bgX * zoomLevel - magnifier.offsetWidth / 2);
            let backgroundPosY = -(bgY * zoomLevel - magnifier.offsetHeight / 2);

            magnifier.style.backgroundPosition = `${backgroundPosX}px ${backgroundPosY}px`;

            // --- Position the magnifier element ---
            // Position it centered over the cursor, using document coordinates
            magnifier.style.left = `${mousePageX - magnifier.offsetWidth / 2}px`;
            magnifier.style.top = `${mousePageY - magnifier.offsetHeight / 2}px`;

            magnifier.style.display = 'block'; // Ensure it's visible
        }

        // Initial call to clear placeholder if needed (e.g., if default values should render something)
        // processCurrentImage(); // Optional: run on load if you want default state rendered

    </script>

</body>
</html>
