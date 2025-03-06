from ultralytics import YOLO

# Carregar um modelo pré-treinado (YOLOv8)
model = YOLO("yolo11n.pt")

# Fazer a detecção em uma imagem
results = model("video_4096_2160_25fps.mp4", show=True)
