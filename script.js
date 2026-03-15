document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const clearBtn = document.getElementById('clear-btn');
    const predictBtn = document.getElementById('predict-btn');
    const resultDiv = document.getElementById('result');

    // Setup canvas
    // Needs to have black background and white stroke to match MNIST
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Set stroke style
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 15; // Thick stroke for better visibility when scaled down
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;

    // Drawing functions
    function startDrawing(e) {
        isDrawing = true;
        [lastX, lastY] = getCoordinates(e);
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault(); // Prevent scrolling on touch devices
        
        const [currentX, currentY] = getCoordinates(e);

        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(currentX, currentY);
        ctx.stroke();

        [lastX, lastY] = [currentX, currentY];
    }

    function stopDrawing() {
        isDrawing = false;
    }

    // Get coordinates relative to canvas
    function getCoordinates(e) {
        const rect = canvas.getBoundingClientRect();
        // Handle both mouse and touch events
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        
        // Calculate coordinate scale based on actual rendered size vs internal width
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        
        return [
            (clientX - rect.left) * scaleX,
            (clientY - rect.top) * scaleY
        ];
    }

    // Event listeners for mouse
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    // Event listeners for touch
    canvas.addEventListener('touchstart', startDrawing);
    canvas.addEventListener('touchmove', draw);
    canvas.addEventListener('touchend', stopDrawing);

    // Clear Canvas
    clearBtn.addEventListener('click', () => {
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        resultDiv.innerHTML = 'Prediction will appear here.';
        resultDiv.className = 'result';
    });

    // Predict digit
    predictBtn.addEventListener('click', async () => {
        // Get canvas data as base64
        const dataURL = canvas.toDataURL('image/png');
        
        resultDiv.innerHTML = '<span class="loading">Predicting...</span>';
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image: dataURL })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                resultDiv.innerHTML = `<strong>Prediction: ${data.prediction}</strong><br>
                                       <small>Confidence: ${(data.confidence * 100).toFixed(2)}%</small>`;
            } else {
                resultDiv.innerHTML = `<span style="color: red;">Error: ${data.error}</span>`;
            }
            
        } catch (error) {
            console.error('Error:', error);
            resultDiv.innerHTML = `<span style="color: red;">Failed to connect to server.</span>`;
        }
    });
});