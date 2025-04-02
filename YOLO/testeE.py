from ultralytics import YOLO
import cv2
import os
import json

PERSON_CLASS_ID = 0
IMAGE_FOLDER = "fotos/fotosE"
PROCESSED_FOLDER = "fotos_processadas"
JSON_FOLDER = "JSONs"

# Criar diretórios se não existirem
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(JSON_FOLDER, exist_ok=True)

if __name__ == "__main__":
    images = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith((".jpeg", ".jpg", ".png"))]

    # Carregar o modelo YOLO pré-treinado
    model = YOLO("yolo11n.pt")

    for image in images:
        image_path = os.path.join(IMAGE_FOLDER, image)
        img = cv2.imread(image_path)
        results = model(image_path, verbose=False)

        num_people = 0
        
        for result in results:
            for box in result.boxes:
                if int(box.cls) == PERSON_CLASS_ID:
                    num_people += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        print(f"Imagem: {image} - Número de pessoas detectadas: {num_people}")

        tempo_estimado = num_people / 6

        dados = {
            "num_people": num_people,
            "tempo_estimado": tempo_estimado
        }

        with open(os.path.join(JSON_FOLDER, f"{image}.json"), "w") as f:
            json.dump(dados, f)

        processed_image_path = os.path.join(PROCESSED_FOLDER, image)
        cv2.imwrite(processed_image_path, img)