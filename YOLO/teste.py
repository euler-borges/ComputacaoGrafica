from ultralytics import YOLO
import cv2

# Carregar o modelo YOLOv8 pré-treinado
model = YOLO("yolov8n.pt")

# Fazer a detecção em uma imagem
results = model("image.jpg")  # Remove `show=True`

# Obter a imagem com as detecções
for result in results:
    img = result.plot()  # Renderizar as caixas na imagem

# Exibir a imagem com OpenCV
cv2.imshow("Detecção YOLO", img)

# Aguardar uma tecla para fechar a janela
cv2.waitKey(0)
cv2.destroyAllWindows()
