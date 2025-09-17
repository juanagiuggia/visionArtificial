# import cv2
# import numpy as np
# import glob

# from extractor import hu_moments_of_file
# from convertidor import int_to_label


# def load_and_test(model):
#     files = glob.glob('../fotos/*')
#     for f in files:
#         hu_moments = hu_moments_of_file(f) # Genera los momentos de hu de los files de testing

#         sample = np.array(hu_moments, dtype=np.float32).reshape(1, -1)
#         test_response = model.predict(sample)[0]

#         #Lee la imagen y la imprime con un texto
#         image = cv2.imread(f)
#         image_with_text = cv2.putText(image, int_to_label(test_response), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#         cv2.imshow("result", image_with_text)
#         cv2.waitKey(0)


import cv2
import numpy as np
import math
import joblib
import os
from extractor import hu_moments_of_file
from convertidor import int_to_label

# Trackbar funciones
def on_trackbar(val):
    pass

def create_trackbar(trackbar_name, window_name, slider_max):
    cv2.createTrackbar(trackbar_name, window_name, 0, slider_max, on_trackbar)

def get_trackbar_value(trackbar_name, window_name):
    return cv2.getTrackbarPos(trackbar_name, window_name)

# Contornos y procesamiento
def get_contours(frame, mode, method):
    contours, _ = cv2.findContours(frame, mode, method)
    return contours

def filter_contours_by_area(contours, min_area, max_area):
    filtered = [cnt for cnt in contours if min_area <= cv2.contourArea(cnt) <= max_area]
    return filtered

def get_bounding_rect(contour):
    return cv2.boundingRect(contour)

def apply_color_conversion(frame, color):
    return cv2.cvtColor(frame, color)

def threshold(frame, slider_max, binary, trackbar_value):
    _, th = cv2.threshold(frame, trackbar_value, slider_max, binary)
    return th

def denoise(frame, method, radius):
    kernel = cv2.getStructuringElement(method, (radius, radius))
    opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return closing

def draw_contours(frame, contours, color, thickness):
    cv2.drawContours(frame, contours, -1, color, thickness)

# Colores
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)

def main():
    window_name = "Weather Elements Classifier"
    cv2.namedWindow(window_name)

    # Cargar modelo
    model_path = os.path.join("models", "modelo_weather_elements.joblib")
    if not os.path.exists(model_path):
        print(f"❌ No se encontró el modelo en {model_path}")
        return

    try:
        model = joblib.load(model_path)
        print("✅ Modelo cargado exitosamente!")
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        return

    # Trackbars para parámetros
    create_trackbar("Threshold", window_name, 255)
    create_trackbar("Kernel denoise", window_name, 10)
    create_trackbar("Min Area", window_name, 10000)
    create_trackbar("Max Area", window_name, 99999)

    # Inicializar cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara")
        return

    print("🌤 Clasificador de Weather Elements iniciado!")
    print("📋 Controles:")
    print(" - Ajusta los parámetros con los trackbars")
    print(" - Presiona 'q' para salir")
    print(" - Muestra elementos frente a la cámara para clasificarlos")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ No se pudo capturar el frame de la cámara")
            break

        gray = apply_color_conversion(frame, cv2.COLOR_BGR2GRAY)

        thresh_val = get_trackbar_value("Threshold", window_name)
        min_area = get_trackbar_value("Min Area", window_name)
        max_area = get_trackbar_value("Max Area", window_name)

        thresh_frame = threshold(gray, 255, cv2.THRESH_BINARY, thresh_val)
        denoised = denoise(thresh_frame, cv2.MORPH_ELLIPSE, 5)

        contours = get_contours(denoised, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        filtered = filter_contours_by_area(contours, min_area, max_area)

        for cnt in filtered:
            try:
                hu_moments = cv2.HuMoments(cv2.moments(cnt))
                for i in range(7):
                    if hu_moments[i] != 0:
                        hu_moments[i] = -1 * math.copysign(1.0, hu_moments[i]) * math.log10(abs(hu_moments[i]))

                sample = np.array(hu_moments, dtype=np.float32).reshape(1, -1)
                pred = model.predict(sample)[0]
                label = int_to_label(pred)

                # Color según label
                color = COLOR_YELLOW
                if label.lower() in ["sol", "sun"]:
                    color = COLOR_YELLOW
                elif label.lower() in ["lluvia", "rain"]:
                    color = COLOR_BLUE
                elif label.lower() in ["nube", "cloud"]:
                    color = COLOR_GREEN
                elif label.lower() in ["trueno", "thunder"]:
                    color = COLOR_RED

                draw_contours(frame, [cnt], color, 2)
                x, y, _, _ = get_bounding_rect(cnt)
                cv2.putText(frame, label, (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            except Exception as e:
                print(f"⚠️ Error procesando contorno: {e}")
                continue

        cv2.imshow(window_name, frame)
        cv2.imshow("Debug - Threshold", denoised)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Clasificador cerrado correctamente")

if __name__ == "__main__":
    main()

