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
            margin: 20px;
            background-color: #f4f4f4;
        }
        h1, h2 {
            text-align: center;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: auto;
            background: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .controls, .visualization-area {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-around;
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
         .controls input[type="file"] {
             width: 100px;
             padding: 5px;
             margin-top: 5px;
             box-sizing: border-box; /* Include padding in width */
         }
         .controls input[type="file"] {
             width: auto; /* Adjust file input width */
         }

        .image-box {
            border: 1px solid #ccc;
            padding: 15px;
            margin: 10px;
            text-align: center;
            background-color: #f8f8f8;
            border-radius: 5px;
            flex: 1; /* Equal flex distribution */
            min-width: 250px; /* Minimum width for wrapping */
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .image-box h3 {
            margin-top: 0;
            color: #555;
        }
        .image-box img, .image-box canvas {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 10px auto;
            background-color: #e0e0e0; /* Canvas background for padding viz */
            border: 1px dashed #aaa;
        }
         .image-box canvas {
             max-height: 300px; /* Prevent excessively tall canvases */
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
                <label for="fixedRes">Fixed Resolution (e.g., 224, 256, 384):</label>
                <input type="number" id="fixedRes" value="256" min="1">
            </div>
             <div>
                <label for="naflexSeqLen">NaFlex Target Seq Len (e.g., 256, 576):</label>
                <input type="number" id="naflexSeqLen" value="256" min="1">
            </div>
        </div>

        <div class="visualization-area" id="visualizationArea">
             <p id="placeholder">Upload an image and adjust parameters to see the transforms.</p>
            <!-- Image boxes will be added here by JS -->
        </div>
    </div>

    <script>
        const imageUpload = document.getElementById('imageUpload');
        const patchSizeInput = document.getElementById('patchSize');
        const fixedResInput = document.getElementById('fixedRes');
        const naflexSeqLenInput = document.getElementById('naflexSeqLen');
        const visualizationArea = document.getElementById('visualizationArea');
        const placeholder = document.getElementById('placeholder');

        let currentImage = null; // Store the loaded image object

        imageUpload.addEventListener('change', handleImageUpload);
        patchSizeInput.addEventListener('input', processCurrentImage);
        fixedResInput.addEventListener('input', processCurrentImage);
        naflexSeqLenInput.addEventListener('input', processCurrentImage);

        function handleImageUpload(event) {
            const file = event.target.files[0];
            if (!file) {
                return;
            }

            const reader = new FileReader();
            reader.onload = function(e) {
                const img = new Image();
                img.onload = () => {
                    currentImage = img; // Store the loaded image
                    processCurrentImage(); // Process the new image
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
            visualizationArea.innerHTML = ''; // Clear previous results
            visualizationArea.appendChild(placeholder);
            placeholder.style.display = 'block';
        }

        function processCurrentImage() {
            if (!currentImage) {
                 // Don't process if no image is loaded
                 // Or if an input changes before image is selected
                if (!imageUpload.files || imageUpload.files.length === 0) {
                    clearVisualization();
                }
                return;
            }

            placeholder.style.display = 'none'; // Hide placeholder
            visualizationArea.innerHTML = ''; // Clear previous results

            const patchSize = parseInt(patchSizeInput.value) || 16;
            const fixedRes = parseInt(fixedResInput.value) || 256;
            const targetSeqLen = parseInt(naflexSeqLenInput.value) || 256;

            if (patchSize <= 0 || fixedRes <= 0 || targetSeqLen <= 0) {
                alert("Parameters must be positive numbers.");
                clearVisualization();
                return;
            }

            const img = currentImage;
            const originalW = img.naturalWidth;
            const originalH = img.naturalHeight;

            // --- 1. Fixed Resolution Transform ---
            const fixedBox = createImageBox('Fixed Resolution');
            const fixedCanvas = document.createElement('canvas');
            fixedCanvas.width = fixedRes;
            fixedCanvas.height = fixedRes;
            const fixedCtx = fixedCanvas.getContext('2d');
            fixedCtx.drawImage(img, 0, 0, fixedRes, fixedRes); // Resize (and distort)

            const fixedTokens = (fixedRes / patchSize) * (fixedRes / patchSize);

            fixedBox.appendChild(fixedCanvas);
            addInfo(fixedBox, `Target Res: <span class="info">${fixedRes} x ${fixedRes}</span>`);
            addInfo(fixedBox, `Input Tokens: <span class="info">${Math.round(fixedTokens)}</span> (${fixedRes/patchSize} x ${fixedRes/patchSize} patches)`);
            visualizationArea.appendChild(fixedBox);

            // --- 2. Original Image ---
            const originalBox = createImageBox('Original');
            const originalImgElement = document.createElement('img');
            originalImgElement.src = img.src;
            originalImgElement.style.maxWidth = '250px'; // Limit display size
            originalImgElement.style.maxHeight = '250px';
            originalImgElement.style.border = 'none'; // No border for original
            originalImgElement.style.background = 'none';

            originalBox.appendChild(originalImgElement);
            addInfo(originalBox, `Dimensions: <span class="info">${originalW} x ${originalH}</span>`);
            visualizationArea.appendChild(originalBox);

            // --- 3. NaFlex Transform ---
            const naflexBox = createImageBox('NaFlex');
            const naflexCanvas = document.createElement('canvas');
            const naflexCtx = naflexCanvas.getContext('2d');

            // Calculate NaFlex dimensions
            let patchesW = Math.ceil(originalW / patchSize);
            let patchesH = Math.ceil(originalH / patchSize);
            let totalPatches = patchesW * patchesH;

            let targetW, targetH, actualPatchesW, actualPatchesH;

            if (totalPatches <= targetSeqLen) {
                // Image fits without downscaling
                actualPatchesW = patchesW;
                actualPatchesH = patchesH;
                targetW = actualPatchesW * patchSize;
                targetH = actualPatchesH * patchSize;
            } else {
                // Image needs downscaling to fit targetSeqLen
                const scale = Math.sqrt(targetSeqLen / totalPatches);
                actualPatchesW = Math.floor(patchesW * scale);
                actualPatchesH = Math.floor(patchesH * scale);

                // Ensure at least one patch if possible
                actualPatchesW = Math.max(1, actualPatchesW);
                actualPatchesH = Math.max(1, actualPatchesH);

                // Adjust one dimension if product is still too large due to floor/max(1)
                while (actualPatchesW * actualPatchesH > targetSeqLen && (actualPatchesW > 1 || actualPatchesH > 1)) {
                    // Reduce the dimension that causes less aspect ratio change from original needed patches
                    const ratioW = actualPatchesW / patchesW;
                    const ratioH = actualPatchesH / patchesH;
                    if (ratioW > ratioH && actualPatchesW > 1) { // Reducing W moves it closer to target ratio
                         actualPatchesW--;
                    } else if (actualPatchesH > 1) {
                         actualPatchesH--;
                    } else { // Only W > 1 left
                         actualPatchesW--;
                    }
                }

                targetW = actualPatchesW * patchSize;
                targetH = actualPatchesH * patchSize;
                 totalPatches = actualPatchesW * actualPatchesH; // Update total patches count
            }

             const finalTokens = actualPatchesW * actualPatchesH;

            // Set canvas size to the calculated grid size
            naflexCanvas.width = targetW;
            naflexCanvas.height = targetH;

            // Draw the image *preserving aspect ratio* within the canvas (letterbox/pillarbox)
            const imgAspectRatio = originalW / originalH;
            const canvasAspectRatio = targetW / targetH;
            let drawW = targetW, drawH = targetH;
            let offsetX = 0, offsetY = 0;

            if (imgAspectRatio > canvasAspectRatio) { // Image is wider than canvas ratio
                drawH = targetW / imgAspectRatio;
                offsetY = (targetH - drawH) / 2;
            } else { // Image is taller or same ratio
                drawW = targetH * imgAspectRatio;
                offsetX = (targetW - drawW) / 2;
            }

            // Optional: Fill background to visualize padding area
            naflexCtx.fillStyle = '#e0e0e0';
            naflexCtx.fillRect(0, 0, targetW, targetH);

            naflexCtx.drawImage(img, offsetX, offsetY, drawW, drawH);

            naflexBox.appendChild(naflexCanvas);
            addInfo(naflexBox, `Grid Dimensions: <span class="info">${targetW} x ${targetH}</span>`);
             addInfo(naflexBox, `Input Tokens: <span class="info">${finalTokens}</span> (${actualPatchesW} x ${actualPatchesH} patches)`);
            addInfo(naflexBox, `(Target Seq Len: ${targetSeqLen})`);
            visualizationArea.appendChild(naflexBox);
        }

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
            p.innerHTML = text; // Use innerHTML to render span styles
            box.appendChild(p);
        }

    </script>

</body>
</html>
