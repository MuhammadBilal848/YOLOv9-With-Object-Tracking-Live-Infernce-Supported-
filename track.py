# ML/DL
import os
import numpy as np
import torch

# CV
import cv2
import supervision as sv

# YOLOv9
from models.common import DetectMultiBackend, AutoShape
from utils.torch_utils import select_device
from utils.general import set_logging

# Video Demonstration
from IPython.display import HTML
from base64 import b64encode

from supervision import Detections as BaseDetections
from supervision.config import CLASS_NAME_DATA_FIELD
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class ExtendedDetections(BaseDetections):
    @classmethod
    def from_yolov9(cls, yolov9_results) -> 'ExtendedDetections':
        """
        Creates a Detections instance from YOLOv9 inference results.
        
        Args:
            yolov9_results (yolov9.models.common.Detections):
                The output Detections instance from YOLOv9.

        Returns:
            ExtendedDetections: A new Detections object that includes YOLOv9 detections.

        Example:
            results = model(image)
            detections = ExtendedDetections.from_yolov9(results)
        """
        xyxy, confidences, class_ids = [], [], []

        for det in yolov9_results.pred:
            for *xyxy_coords, conf, cls_id in reversed(det):
                xyxy.append(torch.stack(xyxy_coords).cpu().numpy())
                confidences.append(float(conf))
                class_ids.append(int(cls_id))

        class_names = np.array([yolov9_results.names[i] for i in class_ids])
        
        if not xyxy:
            return cls.empty()  
        
        return cls(
            xyxy=np.vstack(xyxy),
            confidence=np.array(confidences),
            class_id=np.array(class_ids),
            data={CLASS_NAME_DATA_FIELD: class_names},
        )
    
set_logging(verbose=False)
device = select_device('cuda:0')
model = DetectMultiBackend(weights='weights/yolov9-e.pt', device=device, data='data/coco.yaml', fuse=True)
model = AutoShape(model)

def prepare_yolov9(model, conf=0.2, iou=0.7, classes=None, agnostic_nms=False, max_det=1000):
    model.conf = conf
    model.iou = iou
    model.classes = classes
    model.agnostic = agnostic_nms
    model.max_det = max_det
    return model

def play(filename, width=500):
    html = ''
    video = open(filename,'rb').read()
    src = 'data:video/mp4;base64,' + b64encode(video).decode()
    html += fr'<video width=500 controls autoplay loop><source src="%s" type="video/mp4"></video>' % src 
    return HTML(html)

SOURCE_VIDEO_PATH = 0
TARGET_VIDEO_PATH = "output.mp4"

def prepare_model_and_video_info(model, config, source_path):
    # Initialize and configure YOLOv9 model
    model = prepare_yolov9(model, **config)
    # Retrieve video information
    video_info = sv.VideoInfo.from_video_path(source_path)
    return model, video_info

def setup_annotator():
    # Initialize bounding box annotator
    return sv.BoundingBoxAnnotator(thickness=2)

def simple_annotate_frame(frame, model, annotator):
    # Convert BGR to RGB
    frame_rgb = frame[..., ::-1]
    # Model prediction on single frame
    results = model(frame_rgb, size=1440, augment=False)
    # Converting results to Supervision detections
    detections = ExtendedDetections.from_yolov9(results)
    # Annotate frame with bounding boxes
    return annotator.annotate(scene=frame.copy(), detections=detections)

def simple_process_video(model, config=dict(conf=0.1, iou=0.45, classes=None,), source_path=SOURCE_VIDEO_PATH, target_path=TARGET_VIDEO_PATH):
    model, _ = prepare_model_and_video_info(model, config, source_path)
    annotator = setup_annotator()

    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        return simple_annotate_frame(frame, model, annotator)
    
    # Process the whole video
    sv.process_video(source_path=source_path, target_path=target_path, callback=callback)


def setup_model_and_video_info(model, config, source_path):
    # Initialize and configure YOLOv9 model
    model = prepare_yolov9(model, **config)
    # Retrieve video information
    video_info = sv.VideoInfo.from_video_path(source_path)
    return model, video_info

def create_byte_tracker(video_info):
    # Setup BYTETracker with video information
    return sv.ByteTrack(track_thresh=0.25, track_buffer=250, match_thresh=0.95, frame_rate=video_info.fps)

def setup_annotators():
    # Initialize various annotators for bounding boxes, traces, and labels
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.TRACK)
    # round_box_annotator = sv.RoundBoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.TRACK)
    # corner_annotator = sv.BoxCornerAnnotator(thickness=2, color_lookup=sv.ColorLookup.TRACK)
    trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=50, color_lookup=sv.ColorLookup.TRACK)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, color_lookup=sv.ColorLookup.TRACK)
    return [bounding_box_annotator], trace_annotator, label_annotator

def setup_counting_zone(counting_zone, video_info):
    # Configure counting zone based on provided parameters
    if counting_zone == 'whole_frame':
        polygon = np.array([[0, 0], [video_info.width-1, 0], [video_info.width-1, video_info.height-1], [0, video_info.height-1]])
    else:
        polygon = np.array(counting_zone)
    polygon_zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=(video_info.width, video_info.height), triggering_position=sv.Position.CENTER)
    polygon_zone_annotator = sv.PolygonZoneAnnotator(polygon_zone, sv.Color.ROBOFLOW, thickness=2*(2 if counting_zone=='whole_frame' else 1), text_thickness=1, text_scale=0.5)
    return polygon_zone, polygon_zone_annotator

def annotate_frame(frame, index, video_info, detections, byte_tracker, counting_zone, polygon_zone, polygon_zone_annotator, trace_annotator, annotators_list, label_annotator, show_labels, model):
    # Apply tracking to detections
    detections = byte_tracker.update_with_detections(detections)
    annotated_frame = frame.copy()
    
    # Handle counting zone logic
    if counting_zone is not None:
        is_inside_polygon = polygon_zone.trigger(detections)
        detections = detections[is_inside_polygon]
        annotated_frame = polygon_zone_annotator.annotate(annotated_frame)
    
    # Annotate frame with traces
    annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
    
    # Annotate frame with various bounding boxes
    section_index = int(index / (video_info.total_frames / len(annotators_list)))
    annotated_frame = annotators_list[section_index].annotate(scene=annotated_frame, detections=detections)
    
    # Optionally, add labels to the annotations
    if show_labels:
        annotated_frame = add_labels_to_frame(label_annotator, annotated_frame, detections, model)
    
    return annotated_frame

def add_labels_to_frame(annotator, frame, detections, model):
    labels = [f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}" for confidence, class_id, tracker_id in zip(detections.confidence, detections.class_id, detections.tracker_id)]
    return annotator.annotate(scene=frame, detections=detections, labels=labels)

def process_video(model, config=dict(conf=0.1, iou=0.45, classes=None,), counting_zone=None, show_labels=False, source_path=SOURCE_VIDEO_PATH, target_path=TARGET_VIDEO_PATH):
    model, video_info = setup_model_and_video_info(model, config, source_path)
    byte_tracker = create_byte_tracker(video_info)
    annotators_list, trace_annotator, label_annotator = setup_annotators()
    polygon_zone, polygon_zone_annotator = setup_counting_zone(counting_zone, video_info) if counting_zone else (None, None)

    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        frame_rgb = frame[..., ::-1]  # Convert BGR to RGB
        results = model(frame_rgb, size=1440, augment=False)
        detections = ExtendedDetections.from_yolov9(results)
        return annotate_frame(frame, index, video_info, detections, byte_tracker, counting_zone, polygon_zone, polygon_zone_annotator, trace_annotator, annotators_list, label_annotator, show_labels, model)
    
    sv.process_video(source_path=source_path, target_path=target_path, callback=callback)


def process_live_video(model, config=dict(conf=0.1, iou=0.45, classes=None,), counting_zone=None, show_labels=False,source = 0):
    model, video_info = setup_model_and_video_info(model, config, source_path=source)  # Use source_path=0 for the default webcam
    byte_tracker = create_byte_tracker(video_info)
    annotators_list, trace_annotator, label_annotator = setup_annotators()
    polygon_zone, polygon_zone_annotator = setup_counting_zone(counting_zone, video_info) if counting_zone else (None, None)
    cap = cv2.VideoCapture(source)

    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        ret, frame = cap.read()
        if not ret:
            return frame 

        frame_rgb = frame[..., ::-1]  # Convert BGR to RGB
        results = model(frame_rgb, size=1440, augment=False)
        detections = ExtendedDetections.from_yolov9(results)
        return annotate_frame(frame, index, video_info, detections, byte_tracker, counting_zone, polygon_zone, polygon_zone_annotator, trace_annotator, annotators_list, label_annotator, show_labels, model)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  
            annotated_frame = callback(frame, 0)  # Index is not used for live video
            cv2.imshow('Live Inference', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

yolov9_config=dict(conf=0.3, iou=0.45)
process_live_video(model, config=yolov9_config, counting_zone=None, show_labels=True,source=1)