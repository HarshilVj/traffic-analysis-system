import cv2
from ultralytics import YOLO
import torch
from ultralytics.nn.tasks import DetectionModel

# Try to add safe globals if the method exists (PyTorch 2.3+)
try:
    torch.serialization.add_safe_globals([DetectionModel])
except AttributeError:
    # For older PyTorch versions, use weights_only=False when loading instead
    pass


class VehicleClassifier:
    """Vehicle Detection and Classification"""
    
    # Vehicle type mapping
    VEHICLE_TYPE_MAP = {
        "bike": "2W",
        "car": "4W",
        "bus": "HMV",
        "truck": "HMV",
        "auto": "3W"
    }
    
    def __init__(self, model_path, conf_threshold=0.3):
        """
        Initialize Vehicle Classifier
        
        Args:
            model_path: Path to YOLO weights for vehicle detection
            conf_threshold: Confidence threshold for detection
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
    
    def detect_vehicles(self, image):
        """
        Detect and classify vehicles in image
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            List of detected vehicles with bounding boxes, class, type, and confidence
        """
        results = self.model(
            image,
            device="cpu",
            conf=self.conf_threshold,
            imgsz=512,
            max_det=50,
            verbose=False
        )[0]
        
        vehicles = []
        
        if results.boxes is not None:
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.model.names[cls]
                vehicle_type = self.VEHICLE_TYPE_MAP.get(class_name, "UNKNOWN")
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                vehicles.append({
                    'bbox': (x1, y1, x2, y2),
                    'class_name': class_name,
                    'vehicle_type': vehicle_type,
                    'confidence': conf
                })
        
        return vehicles
    
    def classify_single_vehicle(self, image_crop):
        """
        Classify a single vehicle crop
        
        Args:
            image_crop: Cropped vehicle image
            
        Returns:
            Dictionary with class_name, vehicle_type, and confidence
        """
        results = self.model(
            image_crop,
            conf=self.conf_threshold,
            verbose=False
        )[0]
        
        if results.boxes is not None and len(results.boxes) > 0:
            box = results.boxes[0]  # Take the first detection
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = self.model.names[cls]
            vehicle_type = self.VEHICLE_TYPE_MAP.get(class_name, "UNKNOWN")
            
            return {
                'class_name': class_name,
                'vehicle_type': vehicle_type,
                'confidence': conf
            }
        
        return None
