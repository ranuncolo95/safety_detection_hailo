import numpy as np
import cv2
from hailo_platform import HEF, VDevice, InferVStreams, InputVStreamParams, OutputVStreamParams

hef_path   = 'yolov11m.hef'
video_path = './video_1.mp4'
INPUT_SIZE = 640
SCORE_THR  = 0.4

def preprocess_yolo(frame, size=INPUT_SIZE):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size)).astype(np.uint8)
    return resized[np.newaxis, ...]   # shape (1, 640, 640, 3)

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

                # CASO NMS ON-CHIP: output è una lista (una entry per classe COCO)
                # ognuna è array di shape (N_det, 5) = [ymin, xmin, ymax, xmax, score]
                # coordinate normalizzate 0-1 rispetto a 640x640
                person_dets = output[0][0]

                if person_dets is not None and len(person_dets) > 0:
                    for det in person_dets:
                        ymin, xmin, ymax, xmax, score = det
                        if score < SCORE_THR:
                            continue
                        # coordinate normalizzate 0-1 -> pixel sul frame originale
                        x1 = int(xmin * W_orig); y1 = int(ymin * H_orig)
                        x2 = int(xmax * W_orig); y2 = int(ymax * H_orig)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"person {score:.2f}",
                                    (x1, max(20, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.imshow('YOLOv10n Hailo', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                frame_idx += 1

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference_video()