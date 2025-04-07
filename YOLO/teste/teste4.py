import cv2
import numpy as np
from ultralytics import YOLO

# Carrega o modelo YOLO
model = YOLO("yolov8n.pt")

# Vídeo de entrada e saída
video_path = "video.mkv"
output_path = "video_processado2.mp4"

cap = cv2.VideoCapture(video_path)

# Informações do vídeo
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Criando o writer para salvar o vídeo
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Linha de contagem vertical (a 1/4 da largura)
line_x = width // 4
line_position = [(line_x, 0), (line_x, height)]
offset = 30  # tolerância horizontal

person_counter = 0
detected_ids = set()

# Função para verificar se o centro cruzou a linha vertical
def crossed_line(center, line_x):
    return abs(center[0] - line_x) < offset

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, classes=[0])  # classe 0 = pessoa
    annotated_frame = results[0].plot()

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        for box, obj_id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if crossed_line((cx, cy), line_x) and obj_id not in detected_ids:
                person_counter += 1
                detected_ids.add(obj_id)

            cv2.circle(annotated_frame, (cx, cy), 5, (0, 255, 0), -1)

    # Desenha a linha vertical de contagem
    cv2.line(annotated_frame, line_position[0], line_position[1], (255, 0, 0), 2)

    # Mostra o contador no frame
    cv2.putText(annotated_frame, f"Pessoas: {person_counter}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    out.write(annotated_frame)

    # Visualização opcional
    # cv2.imshow("Contagem de Pessoas", annotated_frame)
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break

cap.release()
out.release()
cv2.destroyAllWindows()
