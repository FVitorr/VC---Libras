import cv2
import numpy as np
from tensorflow.keras.models import load_model

image_x, image_y = 64, 64

classifier = load_model('./model/model.h5')

classes = 9
letras = ['C', 'E', 'F', 'I', 'M', 'N', 'O', 'R', 'S']


def predictor(image):  # Faz a predição com a imagem cropada cinza normalizada
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = classifier.predict(image)
    #print(pred_array)
    #print(np.argmax(pred_array))
    result = letras[np.argmax(pred_array)]
    return result


cam = cv2.VideoCapture(0)

img_text = ['', '']
while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)

    img = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), thickness=2, lineType=8, shift=0)

    imcrop = img[102:298, 427:623]

    # Cria uma imagem de fundo branco 150x150 que servirá para printar a predição
    output = np.ones((150, 150, 3)) * 255
    cv2.putText(output, str(img_text), (15, 130), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 0))

    # Abertura de janelas de interesse
    cv2.imshow("ZOOM", imcrop)
    cv2.imshow("FRAME", frame)
    cv2.imshow("PREDICTION", output)
    imggray = cv2.cvtColor(imcrop, cv2.COLOR_BGR2GRAY)
    cv2.imshow("GRAY SCALE ZOOM", imggray)

    k = cv2.waitKey(10)
    if k == 32:  # Aperte barra de espaço para realizar a predição
        frame = np.stack((imggray,) * 1, axis=-1)
        frame = cv2.resize(imggray, (64, 64))
        frame = frame.reshape(1, 64, 64, 1)
        img_text = predictor(frame)

    print(str(img_text))

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
