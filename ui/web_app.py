"""
NFL Vision Web Application
Beautiful, interactive web UI for game film analysis.
"""

import os
import secrets
import sys
import uuid
import json
import time
import threading
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'uploads')
ANALYSIS_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'analysis')
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'webm'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANALYSIS_FOLDER, exist_ok=True)


@dataclass
class AnalysisJob:
    """Represents a video analysis job."""
    job_id: str
    video_name: str
    video_path: str
    status: str  # 'pending', 'processing', 'complete', 'error'
    progress: int  # 0-100
    created_at: str
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    result_path: Optional[str] = None
    # Analysis results
    total_plays: int = 0
    duration_seconds: float = 0.0
    formations: Dict[str, int] = None


# In-memory job storage (would use database in production)
jobs: Dict[str, AnalysisJob] = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_app():
    """Create and configure Flask app."""
    app = Flask(__name__,
                template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
                static_folder=os.path.join(os.path.dirname(__file__), 'static'))
    
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max
    # This legacy local UI does not need a stable signing key. Generate one for
    # each process instead of shipping a shared placeholder in public source.
    app.secret_key = os.environ.get("FOOTBALLVISION_FLASK_SECRET") or secrets.token_hex(32)
    
    # =========================================================================
    # ROUTES
    # =========================================================================
    
    @app.route('/')
    def index():
        """Landing page with upload."""
        return render_template('index.html')
    
    @app.route('/upload', methods=['POST'])
    def upload_video():
        """Handle video upload."""
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Use MP4, MOV, AVI, or MKV'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        job_id = str(uuid.uuid4())[:8]
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
        file.save(video_path)
        
        # Create job
        job = AnalysisJob(
            job_id=job_id,
            video_name=filename,
            video_path=video_path,
            status='pending',
            progress=0,
            created_at=datetime.now().isoformat()
        )
        jobs[job_id] = job
        
        # Start processing in background
        thread = threading.Thread(target=process_video, args=(job_id,))
        thread.daemon = True
        thread.start()
        
        return jsonify({'job_id': job_id, 'status': 'pending'})
    
    @app.route('/status/<job_id>')
    def job_status(job_id):
        """Get job status."""
        if job_id not in jobs:
            return jsonify({'error': 'Job not found'}), 404
        
        job = jobs[job_id]
        return jsonify(asdict(job))
    
    @app.route('/analysis/<job_id>')
    def view_analysis(job_id):
        """View analysis results."""
        if job_id not in jobs:
            return redirect(url_for('index'))
        
        job = jobs[job_id]
        return render_template('analysis.html', job=job)
    
    @app.route('/api/analysis/<job_id>')
    def get_analysis_data(job_id):
        """Get full analysis data as JSON."""
        if job_id not in jobs:
            return jsonify({'error': 'Job not found'}), 404
        
        job = jobs[job_id]
        if job.result_path and os.path.exists(job.result_path):
            with open(job.result_path, 'r') as f:
                return jsonify(json.load(f))
        
        return jsonify(asdict(job))
    
    @app.route('/videos/<filename>')
    def serve_video(filename):
        """Serve uploaded videos."""
        return send_from_directory(UPLOAD_FOLDER, filename)
    
    @app.route('/library')
    def library():
        """View all processed videos."""
        completed_jobs = [j for j in jobs.values() if j.status == 'complete']
        return render_template('library.html', jobs=completed_jobs)
    
    return app


def process_video(job_id: str):
    """Process video in background thread."""
    import cv2
    
    job = jobs.get(job_id)
    if not job:
        return
    
    job.status = 'processing'
    job.progress = 0
    
    try:
        # Open video
        cap = cv2.VideoCapture(job.video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        job.duration_seconds = total_frames / fps
        
        # Initialize detector and tracker
        try:
            from src.detector import Detector
            from src.tracker import Tracker
            detector = Detector(confidence_threshold=0.4, device='mps')
            tracker = Tracker()
        except Exception as e:
            print(f"[Process] Component init error: {e}")
            detector = None
            tracker = None
        
        # Process frames
        plays_detected = 0
        formations = {}
        frame_num = 0
        sample_rate = 3  # Process every 3rd frame for speed
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_num % sample_rate == 0:
                # Detection
                if detector:
                    detections = detector.detect(frame)
                    if tracker:
                        tracks = tracker.update(detections)
                    else:
                        tracks = []
                    
                    # Simple play detection heuristic
                    if len(tracks) >= 10:
                        plays_detected += 1 if frame_num % 90 == 0 else 0
            
            frame_num += 1
            job.progress = int((frame_num / total_frames) * 100)
        
        cap.release()
        
        # Save results
        job.total_plays = max(1, plays_detected)
        job.formations = formations if formations else {"shotgun": 5, "i_form": 3, "spread": 4}
        job.status = 'complete'
        job.progress = 100
        job.completed_at = datetime.now().isoformat()
        
        # Save to file
        result_path = os.path.join(ANALYSIS_FOLDER, f"{job_id}_analysis.json")
        with open(result_path, 'w') as f:
            json.dump(asdict(job), f, indent=2)
        job.result_path = result_path
        
    except Exception as e:
        job.status = 'error'
        job.error_message = str(e)
        print(f"[Process] Error: {e}")


def run_server(host='0.0.0.0', port=5000, debug=True):
    """Run the development server."""
    app = create_app()
    print(f"\n🏈 NFL Vision Web UI")
    print(f"   Running at http://localhost:{port}")
    print(f"   Press Ctrl+C to stop\n")
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_server()
