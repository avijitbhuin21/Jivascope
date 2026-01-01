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
    const spectrogramPreview = document.getElementById('spectrogram-preview');
    const spectrogramImage = document.getElementById('spectrogram-image');

    const reportTimestamp = document.getElementById('report-timestamp');
    const audioPlayer = document.getElementById('audio-player');
    const audioFilename = document.getElementById('audio-filename');
    const spectrogramResult = document.getElementById('spectrogram-result');
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

    function switchScreen(screen) {
        [uploadScreen, processingScreen, reportScreen].forEach(s => {
            s.classList.remove('active');
        });
        screen.classList.add('active');
    }

    function handleFile(file) {
        if (!file) return;

        const validTypes = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/ogg', 'audio/flac', 'audio/x-wav'];
        const validExtensions = ['.wav', '.mp3', '.ogg', '.flac'];
        const fileExt = '.' + file.name.split('.').pop().toLowerCase();

        if (!validTypes.includes(file.type) && !validExtensions.includes(fileExt)) {
            alert('Please upload a valid audio file (WAV, MP3, OGG, or FLAC)');
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
        spectrogramPreview.classList.remove('visible');

        simulateAnalysis();
    }

    function simulateAnalysis() {
        const steps = [
            { progress: 15, title: 'Analyzing Audio...', status: 'Preparing your file for analysis', duration: 800 },
            { progress: 35, title: 'Processing...', status: 'Extracting audio features', duration: 1000 },
            { progress: 55, title: 'Creating Spectrogram...', status: 'Generating visual representation', duration: 1200 },
            { progress: 75, title: 'Spectrogram Ready', status: 'Running AI detection model', duration: 800, showSpectrogram: true },
            { progress: 90, title: 'Almost Done...', status: 'Finalizing detection results', duration: 600 },
            { progress: 100, title: 'Analysis Complete', status: 'Preparing your report', duration: 500 }
        ];

        let stepIndex = 0;

        function runStep() {
            if (stepIndex >= steps.length) {
                setTimeout(() => {
                    showReport();
                }, 500);
                return;
            }

            const step = steps[stepIndex];
            progressBar.style.width = step.progress + '%';
            processingTitle.textContent = step.title;
            processingStatus.textContent = step.status;

            if (step.showSpectrogram) {
                spectrogramPreview.classList.add('visible');
                spectrogramImage.innerHTML = `
                    <div style="width: 100%; height: 150px; background: linear-gradient(180deg, 
                        rgba(45, 106, 79, 0.8) 0%, 
                        rgba(64, 145, 108, 0.6) 25%, 
                        rgba(149, 213, 178, 0.4) 50%, 
                        rgba(45, 106, 79, 0.6) 75%, 
                        rgba(27, 67, 50, 0.8) 100%); 
                        border-radius: 8px;
                        position: relative;
                        overflow: hidden;">
                        <div style="position: absolute; inset: 0; 
                            background: repeating-linear-gradient(90deg, 
                                transparent 0px, transparent 2px, 
                                rgba(255,255,255,0.1) 2px, rgba(255,255,255,0.1) 4px);
                            animation: spectrogramScroll 2s linear infinite;"></div>
                    </div>
                `;
            }

            stepIndex++;
            setTimeout(runStep, step.duration);
        }

        runStep();
    }

    function showReport() {
        switchScreen(reportScreen);

        const now = new Date();
        reportTimestamp.textContent = 'Generated on: ' + now.toLocaleString();

        if (audioUrl) {
            audioPlayer.src = audioUrl;
        }
        audioFilename.textContent = selectedFile ? selectedFile.name : 'Unknown file';

        const spectrogramGradient = `
            <div style="width: 100%; height: 200px; background: linear-gradient(180deg, 
                rgba(45, 106, 79, 0.9) 0%, 
                rgba(64, 145, 108, 0.7) 20%, 
                rgba(149, 213, 178, 0.5) 40%, 
                rgba(64, 145, 108, 0.6) 60%, 
                rgba(45, 106, 79, 0.7) 80%, 
                rgba(27, 67, 50, 0.9) 100%); 
                border-radius: 8px;
                position: relative;">
                <div style="position: absolute; inset: 0; 
                    background: repeating-linear-gradient(90deg, 
                        transparent 0px, transparent 3px, 
                        rgba(255,255,255,0.08) 3px, rgba(255,255,255,0.08) 6px);"></div>
            </div>
        `;
        document.getElementById('report-spectrogram').innerHTML = spectrogramGradient;

        const mockResults = {
            heartSoundDetected: true,
            murmurDetected: Math.random() > 0.5,
            bpm: Math.floor(Math.random() * 40) + 60
        };

        heartSoundValue.textContent = mockResults.heartSoundDetected ? 'Detected' : 'Not Detected';
        heartSoundBadge.textContent = mockResults.heartSoundDetected ? 'Present' : 'Absent';
        heartSoundBadge.className = 'result-badge ' + (mockResults.heartSoundDetected ? 'detected' : 'not-detected');
        heartSoundResult.className = 'result-card ' + (mockResults.heartSoundDetected ? 'positive' : '');

        murmurValue.textContent = mockResults.murmurDetected ? 'Detected' : 'Not Detected';
        murmurBadge.textContent = mockResults.murmurDetected ? 'Warning' : 'Normal';
        murmurBadge.className = 'result-badge ' + (mockResults.murmurDetected ? 'warning' : 'detected');
        murmurResult.className = 'result-card ' + (mockResults.murmurDetected ? 'negative' : 'positive');

        bpmValue.textContent = mockResults.bpm + ' BPM';
        const bpmStatus = mockResults.bpm < 60 ? 'low' : (mockResults.bpm > 100 ? 'high' : 'normal');
        bpmBadge.textContent = bpmStatus.charAt(0).toUpperCase() + bpmStatus.slice(1);
        bpmBadge.className = 'result-badge ' + (bpmStatus === 'normal' ? 'detected' : 'warning');
        bpmResult.className = 'result-card ' + (bpmStatus === 'normal' ? 'positive' : '');
    }

    newAnalysisBtn.addEventListener('click', () => {
        clearFile();
        switchScreen(uploadScreen);
    });

    downloadReportBtn.addEventListener('click', () => {
        alert('Report download will be available once connected to backend.');
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
