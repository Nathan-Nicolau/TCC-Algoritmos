
from ultralytics import YOLO
from PIL import Image

# Comando para usar GPU para treinamento
comando_gpu = "set CUDA_VISIBLE_DEVICES=0"

#Comando para iniciar o treinamento do yolo
comando_yolo = "yolo task=detect mode=train epochs=100 data=data_custom.yaml model=yolov8n.pt imgsz=640 batch=4"

# ---------------------------------------------------------------------------#

#Usar Modelo pós treinamento para reconhecer novas imagens
# Carregar o modelo treinado
model = YOLO('runs/detect/train15/weights/best.pt')

# Caminho da imagem a ser detectada
source = "C:\\Users\\...\\Downloads\\Data\\24_chromosomes_object\\JEPG\\1053492.jpg"

# Usar modelo para reconhecer a imagem
results = model(source)

for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('results.jpg')  # save image



