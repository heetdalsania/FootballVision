/**
 * NFL Vision - Interactive Web App
 * Handles drag-drop upload, progress tracking, and analysis navigation
 */

document.addEventListener('DOMContentLoaded', () => {
    initDropZone();
    initAnimations();
});

// =========================================
// DROP ZONE UPLOAD
// =========================================

function initDropZone() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');

    if (!dropZone || !fileInput) return;

    // Click to open file picker
    dropZone.addEventListener('click', () => fileInput.click());

    // File selected
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    // Drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });
}

// =========================================
// FILE UPLOAD
// =========================================

async function handleFileUpload(file) {
    // Validate file type
    const validTypes = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/x-matroska', 'video/webm'];
    if (!validTypes.some(type => file.type.startsWith('video/') || file.name.match(/\.(mp4|mov|avi|mkv|webm)$/i))) {
        showError('Please upload a video file (MP4, MOV, AVI, MKV)');
        return;
    }

    // Show progress section
    const dropZone = document.getElementById('drop-zone');
    const progressSection = document.getElementById('progress-section');

    dropZone.style.display = 'none';
    progressSection.classList.remove('hidden');

    // Update file info
    document.getElementById('file-name').textContent = file.name;
    document.getElementById('file-status').textContent = 'Uploading...';
    document.getElementById('progress-stage').textContent = 'Uploading video...';

    try {
        // Create form data
        const formData = new FormData();
        formData.append('video', file);

        // Upload with progress
        const response = await uploadWithProgress('/upload', formData, (progress) => {
            updateProgress(progress, 'Uploading video...');
        });

        if (response.error) {
            throw new Error(response.error);
        }

        // Start polling for processing status
        document.getElementById('file-status').textContent = 'Processing...';
        document.getElementById('progress-stage').textContent = 'Analyzing video with AI...';

        pollJobStatus(response.job_id);

    } catch (error) {
        showError(error.message || 'Upload failed');
        resetUpload();
    }
}

function uploadWithProgress(url, formData, onProgress) {
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();

        xhr.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable) {
                const percent = Math.round((e.loaded / e.total) * 100);
                onProgress(percent);
            }
        });

        xhr.addEventListener('load', () => {
            if (xhr.status === 200) {
                resolve(JSON.parse(xhr.responseText));
            } else {
                reject(new Error('Upload failed'));
            }
        });

        xhr.addEventListener('error', () => reject(new Error('Network error')));

        xhr.open('POST', url);
        xhr.send(formData);
    });
}

// =========================================
// JOB STATUS POLLING
// =========================================

async function pollJobStatus(jobId) {
    const checkStatus = async () => {
        try {
            const response = await fetch(`/status/${jobId}`);
            const data = await response.json();

            if (data.status === 'processing') {
                updateProgress(data.progress, 'Analyzing video with AI...');
                document.getElementById('file-status').textContent = `Processing ${data.progress}%`;
                setTimeout(checkStatus, 1000);
            } else if (data.status === 'complete') {
                updateProgress(100, 'Analysis complete!');
                document.getElementById('file-status').textContent = 'Complete!';

                // Redirect to analysis page after brief delay
                setTimeout(() => {
                    window.location.href = `/analysis/${jobId}`;
                }, 1500);
            } else if (data.status === 'error') {
                showError(data.error_message || 'Analysis failed');
                resetUpload();
            } else {
                setTimeout(checkStatus, 1000);
            }
        } catch (error) {
            showError('Connection lost');
            resetUpload();
        }
    };

    checkStatus();
}

// =========================================
// UI HELPERS
// =========================================

function updateProgress(percent, stage) {
    const fill = document.getElementById('progress-fill');
    const percentText = document.getElementById('progress-percent');
    const stageText = document.getElementById('progress-stage');

    if (fill) fill.style.width = `${percent}%`;
    if (percentText) percentText.textContent = `${percent}%`;
    if (stageText) stageText.textContent = stage;
}

function showError(message) {
    const status = document.getElementById('file-status');
    if (status) {
        status.textContent = message;
        status.style.color = '#ff4757';
    }

    // Also show as alert for visibility
    console.error('NFL Vision Error:', message);
}

function resetUpload() {
    setTimeout(() => {
        const dropZone = document.getElementById('drop-zone');
        const progressSection = document.getElementById('progress-section');

        if (dropZone) dropZone.style.display = 'block';
        if (progressSection) progressSection.classList.add('hidden');

        // Reset progress
        updateProgress(0, '');
        const status = document.getElementById('file-status');
        if (status) {
            status.textContent = '';
            status.style.color = '';
        }
    }, 3000);
}

// =========================================
// ANIMATIONS
// =========================================

function initAnimations() {
    // Intersection observer for scroll animations
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    }, { threshold: 0.1 });

    // Observe feature cards
    document.querySelectorAll('.feature-card').forEach(card => {
        observer.observe(card);
    });

    // Add stagger delay to feature cards
    document.querySelectorAll('.feature-card').forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
    });
}

// =========================================
// ANALYSIS PAGE INTERACTIONS
// =========================================

function seekToPlay(playId, startTime) {
    const video = document.querySelector('video');
    if (video) {
        video.currentTime = startTime / 1000; // Convert ms to seconds
        video.play();
    }

    // Highlight selected play
    document.querySelectorAll('.play-item').forEach(item => {
        item.classList.remove('selected');
    });

    const selectedPlay = document.querySelector(`[data-play-id="${playId}"]`);
    if (selectedPlay) {
        selectedPlay.classList.add('selected');
    }
}

// =========================================
// UTILITY
// =========================================

function formatDuration(ms) {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}
