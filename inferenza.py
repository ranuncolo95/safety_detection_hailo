import numpy as np
import cv2
from hailo_platform import HEF, VDevice, InferVStreams, InputVStreamParams, OutputVStreamParams

hef_path   = 'yolov11m.hef'
video_path = './video_2.mp4'
INPUT_SIZE = 640
SCORE_THR  = 0.4

# Parametri della danger zone (frazione della dimensione frame)
DANGER_W_FRAC = 0.60   # larghezza = 30% della larghezza frame
DANGER_H_FRAC = 0.55   # altezza   = 40% dell'altezza frame (dal basso)

def preprocess_yolo(frame, size=INPUT_SIZE):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size)).astype(np.uint8)
    return resized[np.newaxis, ...]

def compute_danger_zone(W, H):
    """Rettangolo ancorato in basso, centrato orizzontalmente.
    Si estende verso l'alto per DANGER_H_FRAC dell'altezza frame."""
    dw = int(W * DANGER_W_FRAC)
    dh = int(H * DANGER_H_FRAC)
    cx = W // 2
    x1 = cx - dw // 2
    x2 = cx + dw // 2
    y2 = H            # bordo inferiore = bordo inferiore immagine
    y1 = H - dh       # estensione verso l'alto
    return (x1, y1, x2, y2)

def boxes_intersect(a, b):
    """Test AABB-AABB. Ogni box è (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return (ax1 < bx2) and (ax2 > bx1) and (ay1 < by2) and (ay2 > by1)

def draw_danger_zone(frame, zone, in_alert):
    x1, y1, x2, y2 = zone
    color = (0, 0, 255) if in_alert else (0, 200, 255)  # rosso se allarme, giallo altrimenti
    # overlay semitrasparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, dst=frame)
    # bordo pieno
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, "DANGER ZONE", (x1 + 5, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def draw_alert_banner(frame, text="!! ALERT: PERSONA IN ZONA DI PERICOLO !!"):
    H, W = frame.shape[:2]
    # banner rosso in alto
    cv2.rectangle(frame, (0, 0), (W, 40), (0, 0, 255), -1)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(frame, text, ((W - tw) // 2, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def run_inference_video():
    hef = HEF(hef_path)
    with VDevice() as target:
        network_group = target.configure(hef)[0]
        in_params  = InputVStreamParams.make_from_network_group(network_group)
        out_params = OutputVStreamParams.make_from_network_group(network_group)
        in_info  = network_group.get_input_vstream_infos()[0]
        out_info = network_group.get_output_vstream_infos()[0]
        print(f"INPUT  {in_info.name}: {in_info.shape}")
        print(f"OUTPUT {out_info.name}: {out_info.shape}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Errore: impossibile aprire {video_path}")
            return
        H_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        W_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        # Zona di pericolo calcolata UNA volta (è fissa)
        danger_zone = compute_danger_zone(W_orig, H_orig)

        with network_group.activate(), \
             InferVStreams(network_group, in_params, out_params) as vstreams:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                img = preprocess_yolo(frame)
                results = vstreams.infer({in_info.name: img})
                output  = results[out_info.name]

                person_dets = output[0][0]

                # raccolgo prima le detection valide, così so se c'è alert
                valid_boxes = []
                if person_dets is not None and len(person_dets) > 0:
                    for det in person_dets:
                        ymin, xmin, ymax, xmax, score = det
                        if score < SCORE_THR:
                            continue
                        x1 = int(xmin * W_orig); y1 = int(ymin * H_orig)
                        x2 = int(xmax * W_orig); y2 = int(ymax * H_orig)
                        bbox = (x1, y1, x2, y2)
                        in_danger = boxes_intersect(bbox, danger_zone)
                        valid_boxes.append((bbox, float(score), in_danger))

                alert_active = any(v[2] for v in valid_boxes)

                # 1) prima la danger zone (così le bbox stanno sopra)
                draw_danger_zone(frame, danger_zone, alert_active)

                # 2) le bbox: rosse se intersecano, verdi altrimenti
                for (x1, y1, x2, y2), score, in_danger in valid_boxes:
                    color = (0, 0, 255) if in_danger else (0, 255, 0)
                    label = f"person {score:.2f}" + (" [DANGER]" if in_danger else "")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, max(20, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # 3) banner di alert
                if alert_active:
                    draw_alert_banner(frame)

                cv2.imshow('YOLOv11 Hailo - Danger Zone', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                frame_idx += 1

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference_video()