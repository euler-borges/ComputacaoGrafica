import cv2
from ultralytics import YOLO
import numpy as np

# Carrega o modelo YOLO pré-treinado
model = YOLO("yolov8n.pt")  # ou yolov8s.pt, etc.

# Define a linha de contagem (x1, y1, x2, y2)
line_position = [(200, 300), (800, 300)]
offset = 30  # margem de tolerância vertical

# Inicializa variáveis
cap = cv2.VideoCapture("video.mkv")  # substitua pelo caminho do seu vídeo
person_counter = 0
detected_ids = set()

# Função para verificar se o centro cruzou a linha
def crossed_line(center, line_y):
    return abs(center[1] - line_y) < offset

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, classes=[0])  # classe 0 = pessoa

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        for box, obj_id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if crossed_line((cx, cy), line_position[0][1]) and obj_id not in detected_ids:
                person_counter += 1
                detected_ids.add(obj_id)

    # Cria uma imagem preta com o contador
    counter_display = 255 * np.ones((150, 400, 3), dtype=np.uint8)
    cv2.putText(counter_display, f"Pessoas: {person_counter}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    cv2.imshow("Contador", counter_display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
