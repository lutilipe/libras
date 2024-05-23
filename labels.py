import time
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Captura de vídeo
video_cap = cv2.VideoCapture(0)

# Configuração de vídeo
video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, value=700)
video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, value=700)

# Configuração da identificação das mãos pelo mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands(static_image_mode=False)

colors = [(245,117,16), (117,245,16), (16,117,245), (16,117,245), (16,117,245), (16,117,245), (16,117,245), (16,117,245), (16,117,245), (16,117,245), (16,117,245), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame


def format_landmarks(landmarks):
    formatted_landmarks = []
    for landmark in landmarks:
        formatted_landmarks.append([landmark.x, landmark.y, landmark.z])
    return np.array(formatted_landmarks).flatten()

sequence = []
sentence = []
predictions = []
threshold = 0.8
actions = ['acontecer', 'amarelo', 'aproveitar', 'bala', 'banco', 'banheiro', 'barulho', 'conhecer', 'espelho', 'esquina']

model = tf.keras.models.load_model('libras_model.h5')

frame_rate = 10
prev = 0

# Print model input shape for verification
print(f"Expected model input shape: {model.input_shape}")

# Lendo as imagens de fato
while True:
    time_elapsed = time.time() - prev
    success, frame = video_cap.read()

    if time_elapsed > 1./frame_rate:
        prev = time.time()

    if success:
        # Converter de BGR para RGB
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hand.process(RGB_frame)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                formatted_landmarks = format_landmarks(hand_landmarks.landmark)
            
                if len(formatted_landmarks) > 0:
                    sequence.append(formatted_landmarks)
                    sequence = sequence[-30:]
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    #print(f"Keypoints shape: {formatted_landmarks.shape}")
                    #print(f"Sequence length: {len(sequence)}")
                else:
                    print("Unexpected number of keypoints detected.")
                
        # Logica de predicao
        if len(sequence) == 30:
            #print(sequence)
            #print(np.array([sequence]).shape)
            result = model.predict(np.array([sequence]))[0]
            print("RESULTADO: ", result) # [1.1613590e-09 3.5286481e-03 4.2446494e-02 1.2771429e-05 9.5401204e-01]
            category = actions[np.argmax(result)]
            print(category)

            if result[np.argmax(result)] > threshold: 
                if len(sentence) > 0: 
                    if actions[np.argmax(result)] != sentence[-1]:
                        sentence.append(actions[np.argmax(result)])
                else:
                    sentence.append(actions[np.argmax(result)])

            if len(sentence) > 1: 
                sentence = sentence[-1:]
                sequence = []

            frame = prob_viz(result, actions, frame, colors)
                
        cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(frame, ' '.join(sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("imagem capturada em tempo real", frame)
        if cv2.waitKey(10) == ord('q'):
            break

cv2.destroyAllWindows()
video_cap.release()
