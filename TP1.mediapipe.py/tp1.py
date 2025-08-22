# Control de Spotify con Gestos - VERSION PARA CUENTAS GRATUITAS
# pip install mediapipe opencv-python pynput

import cv2
import time
import mediapipe as mp
import numpy as np
from pynput.keyboard import Key
import pynput.keyboard as keyboard
import platform
import subprocess

# Configuración de MediaPipe
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Colores para visualización
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255, 0, 0)
ORANGE_COLOR = (0, 165, 255)

class MediaKeyController:
    """Controlador usando teclas de medios del sistema"""
    
    def __init__(self):
        self.keyboard = keyboard.Controller()
        self.last_action_time = 0
        self.action_cooldown = 1.5
        self.system = platform.system().lower()
        
    def can_perform_action(self):
        """Verificar si se puede realizar una acción (cooldown)"""
        current_time = time.time()
        return current_time - self.last_action_time > self.action_cooldown
    
    def play_pause_toggle(self):
        """Alternar play/pause usando teclas del sistema"""
        if not self.can_perform_action():
            return False
            
        try:
            print("🎵 Enviando comando Play/Pause...")
            
            if self.system == "windows":
                # Windows: usar tecla de play/pause
                self.keyboard.press(Key.media_play_pause)
                self.keyboard.release(Key.media_play_pause)
            elif self.system == "darwin":  # macOS
                # macOS: usar tecla de play/pause
                self.keyboard.press(Key.media_play_pause)
                self.keyboard.release(Key.media_play_pause)
            else:  # Linux
                # Linux: usar playerctl si está disponible
                try:
                    subprocess.run(['playerctl', 'play-pause'], check=True)
                except:
                    # Fallback: usar tecla de medios
                    self.keyboard.press(Key.media_play_pause)
                    self.keyboard.release(Key.media_play_pause)
            
            self.last_action_time = time.time()
            print("✅ Comando enviado correctamente!")
            return True
            
        except Exception as e:
            print(f"❌ Error al enviar comando: {e}")
            return False
    
    def next_track(self):
        """Siguiente canción"""
        if not self.can_perform_action():
            return False
            
        try:
            print("🎵 Enviando comando Siguiente...")
            
            if self.system == "linux":
                try:
                    subprocess.run(['playerctl', 'next'], check=True)
                except:
                    self.keyboard.press(Key.media_next)
                    self.keyboard.release(Key.media_next)
            else:
                self.keyboard.press(Key.media_next)
                self.keyboard.release(Key.media_next)
            
            self.last_action_time = time.time()
            print("✅ Siguiente canción!")
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False

def calculate_distance(p1, p2):
    """Calcular distancia euclidiana entre dos puntos"""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def is_ok_sign(hand_landmarks):
    """Detectar seña de OK (👌)"""
    if not hand_landmarks:
        return False

    thumb_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP]
    palm_base = hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST]
    
    thumb_index_distance = calculate_distance(thumb_tip, index_tip)
    middle_distance = calculate_distance(middle_tip, palm_base)
    ring_distance = calculate_distance(ring_tip, palm_base)
    pinky_distance = calculate_distance(pinky_tip, palm_base)
    
    circle_threshold = 0.1
    extension_threshold = 0.15
    
    is_circle_formed = thumb_index_distance < circle_threshold
    are_others_extended = all(d > extension_threshold for d in [middle_distance, ring_distance, pinky_distance])
    
    return is_circle_formed and are_others_extended

def is_stop_sign(hand_landmarks):
    """Detectar seña de Stop (✋)"""
    if not hand_landmarks:
        return False
    
    thumb_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP]
    
    thumb_mcp = hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_MCP]
    index_mcp = hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP]
    ring_mcp = hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_MCP]
    
    thumb_extension = calculate_distance(thumb_tip, thumb_mcp)
    index_extension = calculate_distance(index_tip, index_mcp)
    middle_extension = calculate_distance(middle_tip, middle_mcp)
    ring_extension = calculate_distance(ring_tip, ring_mcp)
    pinky_extension = calculate_distance(pinky_tip, pinky_mcp)
    
    extension_threshold = 0.12
    
    all_fingers_extended = all(ext > extension_threshold for ext in 
                             [thumb_extension, index_extension, middle_extension, 
                              ring_extension, pinky_extension])
    
    finger_separation = (
        calculate_distance(thumb_tip, index_tip) > 0.08 and
        calculate_distance(index_tip, middle_tip) > 0.05 and
        calculate_distance(middle_tip, ring_tip) > 0.05 and
        calculate_distance(ring_tip, pinky_tip) > 0.05
    )
    
    return all_fingers_extended and finger_separation

def is_thumbs_up(hand_landmarks):
    """Detectar pulgar arriba (👍) para siguiente canción"""
    if not hand_landmarks:
        return False
    
    thumb_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP]
    thumb_mcp = hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_MCP]
    index_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP]
    
    # Pulgar debe estar arriba (tip más alto que mcp)
    thumb_up = thumb_tip.y < thumb_mcp.y
    
    # Otros dedos deben estar doblados (tip cerca de mcp)
    index_folded = calculate_distance(index_tip, index_mcp) < 0.1
    middle_folded = calculate_distance(middle_tip, middle_mcp) < 0.1
    
    return thumb_up and index_folded and middle_folded

def main():
    """Función principal"""
    
    print("🎵 Control de Spotify para Cuentas GRATUITAS")
    print("=" * 50)
    print("👌 OK = Play/Pause Toggle")
    print("✋ Stop = Play/Pause Toggle") 
    print("👍 Thumbs Up = Siguiente canción")
    print("💡 Funciona con cuentas gratuitas y premium")
    print("Presiona 'q' para salir\n")
    
    # Inicializar controlador
    media_controller = MediaKeyController()
    
    # Inicializar captura de video
    capture = cv2.VideoCapture(0)
    
    if not capture.isOpened():
        print("❌ Error: No se pudo acceder a la cámara")
        return
    
    last_gesture_time = 0
    gesture_cooldown = 1.5
    last_action = ""
    
    while capture.isOpened():
        ret, frame = capture.read()
        
        if not ret:
            print("❌ Error al leer frame de la cámara")
            break
        
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic_model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        current_gesture = ""
        gesture_detected = False
        current_time = time.time()
        
        # Detectar gestos en mano derecha
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            if is_ok_sign(results.right_hand_landmarks):
                current_gesture = "OK - PLAY/PAUSE"
                gesture_detected = True
                
                if current_time - last_gesture_time > gesture_cooldown:
                    print("🎯 Gesto OK detectado!")
                    if media_controller.play_pause_toggle():
                        last_action = "⏯️ PLAY/PAUSE"
                        last_gesture_time = current_time
            
            elif is_stop_sign(results.right_hand_landmarks):
                current_gesture = "STOP - PLAY/PAUSE"
                gesture_detected = True
                
                if current_time - last_gesture_time > gesture_cooldown:
                    print("🛑 Gesto STOP detectado!")
                    if media_controller.play_pause_toggle():
                        last_action = "⏯️ PLAY/PAUSE"
                        last_gesture_time = current_time
            
            elif is_thumbs_up(results.right_hand_landmarks):
                current_gesture = "THUMBS UP - NEXT"
                gesture_detected = True
                
                if current_time - last_gesture_time > gesture_cooldown:
                    print("👍 Gesto THUMBS UP detectado!")
                    if media_controller.next_track():
                        last_action = "⏭️ NEXT"
                        last_gesture_time = current_time
        
        # Detectar en mano izquierda si no hay gesto en la derecha
        if results.left_hand_landmarks and not gesture_detected:
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            if is_ok_sign(results.left_hand_landmarks):
                current_gesture = "OK - PLAY/PAUSE (Left)"
                gesture_detected = True
                
                if current_time - last_gesture_time > gesture_cooldown:
                    print("🎯 Gesto OK detectado (mano izquierda)!")
                    if media_controller.play_pause_toggle():
                        last_action = "⏯️ PLAY/PAUSE"
                        last_gesture_time = current_time
        
        # Mostrar información en pantalla
        y_offset = 30
        
        cv2.putText(image, "Spotify: Cuenta GRATUITA Compatible", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN_COLOR, 2)
        y_offset += 35
        
        if gesture_detected:
            cv2.putText(image, f"Gesto: {current_gesture}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLUE_COLOR, 2)
        else:
            cv2.putText(image, "Gesto: Ninguno detectado", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        y_offset += 35
        
        if last_action:
            cv2.putText(image, f"Accion: {last_action}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, ORANGE_COLOR, 2)
        y_offset += 50
        
        # Instrucciones
        cv2.putText(image, "Instrucciones:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        cv2.putText(image, "OK (👌) o Stop (✋) = Play/Pause", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 20
        cv2.putText(image, "Thumbs Up (👍) = Siguiente cancion", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 20
        cv2.putText(image, "Presiona 'q' para salir", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("Control de Spotify - Version Gratuita", image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print("\nCerrando aplicación...")
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
