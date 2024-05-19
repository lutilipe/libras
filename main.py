import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Captura de vídeo
video_cap = cv2.VideoCapture(0)

# Configuração de vídeo
video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, value=600)
video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, value=600)

# Configuração da identificação das mãos pelo mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands(static_image_mode=False)

def format_landmarks(landmarks):
  formatted_landmarks = []
  for landmark in landmarks:
    formatted_landmarks.append([landmark.x, landmark.y, landmark.z])

  return np.concatenate(formatted_landmarks)

sequence = []
sentence = []
predictions = []
threshold = 0.5
actions = ["oi", "bom dia"]

model = tf.keras.models.load_model('libras_model.h5')

# Lendo as imagens de fato
while True:
    success, frame = video_cap.read()
    if success:
        # Converter de BGR para RGB
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hand.process(RGB_frame)
        if result.multi_hand_landmarks:
            keypoints = []
            for hand_landmarks in result.multi_hand_landmarks:
                keypoints.append(format_landmarks(hand_landmarks.landmark))
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            sequence.append(np.concatenate(keypoints))
            sequence = sequence[-30:]

        # Logica de predicao
        if len(sequence) == 30:
            print(sequence)
            print(np.array([sequence]).shape)
            result = model.predict(np.array([sequence]))
            category = actions[np.argmax(result[0])]
            print(category)

        cv2.imshow("imagem capturada em tempo real", frame)
        if cv2.waitKey(10) == ord('q'):
            break

cv2.destroyAllWindows()
