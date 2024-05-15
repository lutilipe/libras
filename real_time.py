import cv2
import mediapipe as mp

# Captura de vídeo
video_cap = cv2.VideoCapture(0)

# Configuração de vídeo
video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, value=600)
video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, value=600)

# Configuração da identificação das mãos pelo mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands()

# Lendo as imagens de fato
while True:
    success, frame = video_cap.read()
    if success:
        # Converter de BGR para RGB
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hand.process(RGB_frame)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                print(hand_landmarks)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow("imagem capturada em tempo real", frame)
        if cv2.waitKey(10) == ord('q'):
            break

cv2.destroyAllWindows()
