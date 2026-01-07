import cv2
import numpy as np
from ultralytics import YOLO

def classify_motility(tracks, min_progress=5):
    results = {"progressive": 0, "non_progressive": 0, "immotile": 0}

    for tid, points in tracks.items():
        if len(points) < 2:
            results["immotile"] += 1
            continue

        dist = np.linalg.norm(points[-1] - points[0])

        if dist > min_progress:
            results["progressive"] += 1
        elif dist > 1:
            results["non_progressive"] += 1
        else:
            results["immotile"] += 1

    return results


def process_combined(video_path, motility_model_path, morph_model_path):
    motility_model = YOLO(motility_model_path)
    morph_model = YOLO(morph_model_path)

    cap = cv2.VideoCapture(video_path)

    tracks = {}
    morphology_count = {
        "head": 0, "tail": 0, "neck": 0,
        "normal": 0, "abnormal": 0
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ---------- MOTILITY ----------
        mot_res = motility_model.track(frame, persist=True)

        if mot_res[0].boxes.id is not None:
            ids = mot_res[0].boxes.id.cpu().numpy()
            boxes = mot_res[0].boxes.xywh.cpu().numpy()

            for obj_id, box in zip(ids, boxes):
                cx, cy, _, _ = box
                center = np.array([cx, cy])

                if obj_id not in tracks:
                    tracks[obj_id] = []
                tracks[obj_id].append(center)

        # ---------- MORPHOLOGY ----------
        morph_res = morph_model(frame)

        for box in morph_res[0].boxes:
            cls = int(box.cls)

            if cls == 0: morphology_count["head"] += 1
            elif cls == 1: morphology_count["tail"] += 1
            elif cls == 2: morphology_count["neck"] += 1
            elif cls == 3: morphology_count["normal"] += 1
            elif cls == 4: morphology_count["abnormal"] += 1

    cap.release()

    # classify motility at end
    motility_result = classify_motility(tracks)

    return motility_result, morphology_count
