import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Carregar o modelo
classifier = load_model('modelo_libras.keras')

# Lista de classes (letras)
letras = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y']

# Inicializa MediaPipe para detecção de mãos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Função para fazer a predição
def predictor(image):
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = classifier.predict(image)
    result = letras[np.argmax(pred_array)]
    return result

# Inicializa a captura da webcam
cam = cv2.VideoCapture(0)

img_text = ''

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Inverte horizontalmente para visualização
    frame = cv2.flip(frame, 1)

    # Conversão para RGB, pois o MediaPipe usa essa ordem de cores
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Desenhando a região de interesse (ROI) maior e mais à direita
    height, width, _ = frame.shape
    roi_x1, roi_y1, roi_x2, roi_y2 = int(width * 0.55), int(height * 0.2), int(width * 0.95), int(height * 0.8)
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Pegar a região de interesse (ROI) ajustada
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * width)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * width)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * height)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * height)

            hand_img = frame[y_min:y_max, x_min:x_max]
            imggray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
            imggray_resized = cv2.resize(imggray, (64, 64))  # Ajuste para 64x64
            imggray_resized = imggray_resized.reshape(1, 64, 64, 1)  # Mantendo a dimensão correta
            img_text = predictor(imggray_resized)
            print("Shape da imagem antes da predição:", imggray_resized.shape)  # Deve imprimir (1, 64, 64, 1)

    # Exibe a predição
    cv2.putText(frame, img_text, (15, 130), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 0))

    # Exibe o frame com a predição e a ROI
    cv2.imshow("FRAME", frame)

    # Finaliza o programa ao pressionar ESC
    k = cv2.waitKey(1)
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()
