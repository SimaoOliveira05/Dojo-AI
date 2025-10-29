# Adicione instruções para ativar o ambiente virtual e instalar dependências
# Certifique-se de que o ambiente virtual esteja ativado antes de executar este script.
# Para criar e ativar o ambiente virtual, execute os seguintes comandos:
# python3 -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt

# Certifique-se de que o OpenCV (cv2) esteja instalado no ambiente virtual
# pip install opencv-python

import cv2
from PIL import Image, ImageOps
import numpy as np
from keras.models import load_model
import time
import pygame

# Inicializar som
pygame.mixer.init()
sounds = {
    "Mola": pygame.mixer.Sound("som1.wav"),
    "Copo": pygame.mixer.Sound("som2.wav"),
    "Carteira": pygame.mixer.Sound("som3.wav")
}

# Carregar modelo
model = load_model("keras_model.h5", compile=False)
# Remove espaços e pega apenas o primeiro token de cada linha
class_names = [line.strip().split()[0] for line in open("labels.txt", "r").readlines()]


# Inicializar câmara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Erro: não foi possível aceder à câmara.")
    exit()

print("A iniciar a captura. Prima Ctrl+C para sair.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Erro ao capturar frame.")
            break

        # Processar frame
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Prever
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]

        confidence = prediction[0][index]
        print(f"Class: {class_name}, Confidence: {confidence:.4f}")

        # Tocar som se for 1, 2 ou 3
        if class_name in sounds:
            sounds[class_name].play()

        time.sleep(0.2)  # 5 FPS

finally:
    cap.release()
