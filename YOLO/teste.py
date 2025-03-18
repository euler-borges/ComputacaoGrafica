from ultralytics import YOLO
import cv2
import os
import json

PERSON_CLASS_ID = 0
IMAGE_FOLDER = "fotos/"


if __name__ == "__main__":
    
    images = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith((".jpeg", ".jpg", ".png"))]

    # Carregar o modelo YOLO11 pré-treinado
    model = YOLO("yolo11l.pt")


    for image in images:
        image_path = os.path.join(IMAGE_FOLDER, image)
        results = model(image_path, verbose = False)  # Remove `show=True`

        num_people = sum(1 for result in results for box in result.boxes if int(box.cls) == PERSON_CLASS_ID)

        print(f"Imagem: {image} - Número de pessoas detectadas: {num_people}")

        tempo_estimado = num_people / 6

        dados = {
            "num_people": num_people,
            "tempo_estimado": tempo_estimado
        }

        with open(f"JSONs/{image}.json", "w") as f:
            json.dump(dados, f)

# # Exibir a imagem com as detecções
# for result in results:
#     img = result.plot()

# cv2.imshow("Detecção YOLO", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()








