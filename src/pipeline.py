import cv2
import time
import argparse
from ultralytics import YOLO

class HazardDetector:
    def __init__(self, model_path, conf_thresh=0.5, iou_thresh=0.4):
        """
        Initializes the YOLO inference engine.
        Using a .engine file here utilizes TensorRT for max FPS on edge devices.
        """
        print(f"Loading model from {model_path}...")
        self.model = YOLO(model_path, task='detect')
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        
        # Priority mapping for hazard alerts
        self.priority_map = {
            'person': 'CRITICAL',
            'bicycle': 'CRITICAL',
            'car': 'HIGH',
            'truck': 'HIGH',
            'bus': 'HIGH'
        }

    def process_frame(self, frame):
        """Runs inference on a single frame and extracts hazard logic."""
        start_time = time.time()
        
        # Inference (half=True enables FP16 precision if supported)
        results = self.model(frame, conf=self.conf_thresh, iou=self.iou_thresh, half=True, verbose=False)[0]
        
        hazards = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = self.model.names[cls_id]
            
            # Extract coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if class_name in self.priority_map:
                hazards.append({
                    'class': class_name,
                    'confidence': round(conf, 2),
                    'bbox': [x1, y1, x2, y2],
                    'priority': self.priority_map[class_name]
                })
                
        latency = (time.time() - start_time) * 1000 # in ms
        return hazards, latency, results.plot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='../models/yolov11n.engine', help='Path to TRT engine')
    parser.add_argument('--source', type=int, default=0, help='Camera index or video path')
    args = parser.parse_args()

    detector = HazardDetector(args.model)
    cap = cv2.VideoCapture(args.source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        hazards, latency, viz_frame = detector.process_frame(frame)
        
        cv2.putText(viz_frame, f"Latency: {latency:.1f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Hazard Detection Pipeline", viz_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
