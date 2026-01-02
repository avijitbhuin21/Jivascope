(function () {
    const uploadScreen = document.getElementById('upload-screen');
    const processingScreen = document.getElementById('processing-screen');
    const reportScreen = document.getElementById('report-screen');

    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const filePreview = document.getElementById('file-preview');
    const fileName = document.getElementById('file-name');
    const removeFileBtn = document.getElementById('remove-file');
    const analyzeBtn = document.getElementById('analyze-btn');

    const processingTitle = document.getElementById('processing-title');
    const processingStatus = document.getElementById('processing-status');
    const progressBar = document.getElementById('progress-bar');
    const visualizationsPreview = document.getElementById('visualizations-preview');
    const spectrogramImage = document.getElementById('spectrogram-image');
    const waveformImage = document.getElementById('waveform-image');

    const reportTimestamp = document.getElementById('report-timestamp');
    const audioPlayer = document.getElementById('audio-player');
    const audioFilename = document.getElementById('audio-filename');
    const heartSoundResult = document.getElementById('heart-sound-result');
    const heartSoundValue = document.getElementById('heart-sound-value');
    const heartSoundBadge = document.getElementById('heart-sound-badge');
    const murmurResult = document.getElementById('murmur-result');
    const murmurValue = document.getElementById('murmur-value');
    const murmurBadge = document.getElementById('murmur-badge');
    const bpmResult = document.getElementById('bpm-result');
    const bpmValue = document.getElementById('bpm-value');
    const bpmBadge = document.getElementById('bpm-badge');
    const newAnalysisBtn = document.getElementById('new-analysis-btn');
    const downloadReportBtn = document.getElementById('download-report-btn');

    let selectedFile = null;
    let audioUrl = null;
    let analysisResult = null;

    function switchScreen(screen) {
        [uploadScreen, processingScreen, reportScreen].forEach(s => {
            s.classList.remove('active');
        });
        screen.classList.add('active');
    }

    function handleFile(file) {
        if (!file) return;

        const validTypes = ['audio/wav', 'audio/x-wav'];
        const fileExt = '.' + file.name.split('.').pop().toLowerCase();

        if (!validTypes.includes(file.type) && fileExt !== '.wav') {
            alert('Unsupported format. Please upload a WAV file only.\n\nThis application currently only supports WAV audio format.');
            return;
        }

        selectedFile = file;
        fileName.textContent = file.name;
        filePreview.classList.add('visible');
        uploadZone.style.display = 'none';
        analyzeBtn.disabled = false;

        if (audioUrl) {
            URL.revokeObjectURL(audioUrl);
        }
        audioUrl = URL.createObjectURL(file);
    }

    function clearFile() {
        selectedFile = null;
        fileInput.value = '';
        filePreview.classList.remove('visible');
        uploadZone.style.display = 'block';
        analyzeBtn.disabled = true;
        analysisResult = null;

        if (audioUrl) {
            URL.revokeObjectURL(audioUrl);
            audioUrl = null;
        }
    }

    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    uploadZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });

    uploadZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');

        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    removeFileBtn.addEventListener('click', clearFile);

    analyzeBtn.addEventListener('click', () => {
        if (!selectedFile) return;
        startAnalysis();
    });

    function startAnalysis() {
        switchScreen(processingScreen);
        progressBar.style.width = '0%';
        visualizationsPreview.classList.remove('visible');

        runAnalysis();
    }

    async function runAnalysis() {
        updateProgress(10, 'Uploading Audio...', 'Sending file to server');

        const formData = new FormData();
        formData.append('audio', selectedFile);

        try {
            updateProgress(25, 'Processing...', 'Server is processing your audio');

            const progressInterval = setInterval(() => {
                const currentWidth = parseFloat(progressBar.style.width) || 25;
                if (currentWidth < 85) {
                    const newWidth = currentWidth + Math.random() * 5;
                    progressBar.style.width = newWidth + '%';
                }
            }, 2000);

            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });

            clearInterval(progressInterval);

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Analysis failed');
            }

            const result = await response.json();

            if (!result.success) {
                throw new Error(result.error || 'Analysis failed');
            }

            analysisResult = result;

            updateProgress(90, 'Processing Complete', 'Generating visualizations');

            if (result.spectrogram || result.waveform) {
                visualizationsPreview.classList.add('visible');
                if (result.waveform) {
                    waveformImage.innerHTML = `<img src="${result.waveform}" alt="Waveform" style="width: 100%; height: auto; border-radius: 8px;">`;
                }
                if (result.spectrogram) {
                    spectrogramImage.innerHTML = `<img src="${result.spectrogram}" alt="Spectrogram" style="width: 100%; height: auto; border-radius: 8px;">`;
                }
            }

            await new Promise(resolve => setTimeout(resolve, 500));
            updateProgress(100, 'Analysis Complete', 'Preparing your report');

            await new Promise(resolve => setTimeout(resolve, 500));
            showReport(result);

        } catch (error) {
            console.error('Analysis error:', error);
            showError(error.message || 'An error occurred during analysis');
        }
    }

    function updateProgress(percent, title, status) {
        progressBar.style.width = percent + '%';
        processingTitle.textContent = title;
        processingStatus.textContent = status;
    }

    function showError(message) {
        switchScreen(uploadScreen);
        alert('Analysis Error: ' + message);
    }

    function showReport(result) {
        switchScreen(reportScreen);

        const now = new Date();
        reportTimestamp.textContent = 'Generated on: ' + now.toLocaleString();

        if (audioUrl) {
            audioPlayer.src = audioUrl;
        }
        audioFilename.textContent = selectedFile ? selectedFile.name : 'Unknown file';

        if (result.waveform) {
            document.getElementById('report-waveform').innerHTML =
                `<img src="${result.waveform}" alt="Waveform" style="width: 100%; height: auto; border-radius: 8px;">`;
        }

        if (result.spectrogram) {
            document.getElementById('report-spectrogram').innerHTML =
                `<img src="${result.spectrogram}" alt="Spectrogram" style="width: 100%; height: auto; border-radius: 8px;">`;
        }

        const heartSoundDetected = result.prediction.heart_sound_present;
        const heartSoundConfidence = Math.round(result.confidence.heart_sound * 100);

        heartSoundValue.textContent = heartSoundDetected ? 'Detected' : 'Not Detected';
        heartSoundBadge.textContent = `${heartSoundConfidence}%`;
        heartSoundBadge.className = 'result-badge ' + (heartSoundDetected ? 'detected' : 'not-detected');
        heartSoundResult.className = 'result-card ' + (heartSoundDetected ? 'positive' : '');

        const murmurDetected = result.prediction.murmur_present;
        const murmurConfidence = Math.round(result.confidence.murmur * 100);

        murmurValue.textContent = murmurDetected ? 'Detected' : 'Not Detected';
        murmurBadge.textContent = murmurDetected ? 'Warning' : 'Normal';
        murmurBadge.className = 'result-badge ' + (murmurDetected ? 'warning' : 'detected');
        murmurResult.className = 'result-card ' + (murmurDetected ? 'negative' : 'positive');

        const bpm = result.bpm || 72;
        bpmValue.textContent = `${bpm} BPM`;
        const bpmStatus = bpm < 60 ? 'low' : (bpm > 100 ? 'high' : 'normal');
        bpmBadge.textContent = bpmStatus.charAt(0).toUpperCase() + bpmStatus.slice(1);
        bpmBadge.className = 'result-badge ' + (bpmStatus === 'normal' ? 'detected' : 'warning');
        bpmResult.className = 'result-card ' + (bpmStatus === 'normal' ? 'positive' : '');
    }

    newAnalysisBtn.addEventListener('click', () => {
        clearFile();
        switchScreen(uploadScreen);
    });

    downloadReportBtn.addEventListener('click', () => {
        if (!analysisResult) {
            alert('No analysis result to download.');
            return;
        }

        const reportData = {
            timestamp: new Date().toISOString(),
            filename: selectedFile ? selectedFile.name : 'unknown',
            prediction: analysisResult.prediction,
            confidence: analysisResult.confidence,
            bpm: analysisResult.bpm,
            inference_time_ms: analysisResult.inference_time_ms
        };

        const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `heart-analysis-report-${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });

    const style = document.createElement('style');
    style.textContent = `
        @keyframes spectrogramScroll {
            0% { transform: translateX(0); }
            100% { transform: translateX(4px); }
        }
    `;
    document.head.appendChild(style);
})();
