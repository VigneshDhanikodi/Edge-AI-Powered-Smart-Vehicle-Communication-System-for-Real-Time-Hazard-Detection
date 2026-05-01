import time
import numpy as np
import pandas as pd
from src.pipeline import HazardDetector

def run_evaluation_benchmark(video_path, model_path, num_frames=500):
    """
    Evaluates the detection pipeline performance, simulating the WebRTC conditions.
    """
    print("Initializing Edge-AI Evaluation Protocol...")
    detector = HazardDetector(model_path)
    
    # Import cv2 locally to avoid dependency issues if just viewing the script
    import cv2
    cap = cv2.VideoCapture(video_path)
    
    latencies = []
    frame_count = 0
    
    print(f"Benchmarking first {num_frames} frames...")
    
    while cap.isOpened() and frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize to match the 640p streaming metric from the report
        frame = cv2.resize(frame, (640, 480))
        
        # Inference
        hazards, lat, _ = detector.process_frame(frame)
        latencies.append(lat)
        frame_count += 1
        
    cap.release()
    
    # Calculate Metrics
    avg_latency = np.mean(latencies)
    avg_fps = 1000 / avg_latency if avg_latency > 0 else 0
    jitter = np.std(latencies)
    
    metrics = {
        "Total Frames Processed": frame_count,
        "Average Inference Latency (ms)": round(avg_latency, 2),
        "Estimated FPS": round(avg_fps, 1),
        "Jitter (ms)": round(jitter, 2)
    }
    
    print("\n--- PERFORMANCE METRICS ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")
        
    return metrics

# Example execution (Requires a sample .mp4 and .engine file)
# results = run_evaluation_benchmark('../data/test_dashcam_sequence.mp4', '../models/yolov11n.engine')
