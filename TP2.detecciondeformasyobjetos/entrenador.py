import cv2
import csv
import joblib
import os
import numpy as np
from convertidor import label_to_int
from sklearn.tree import DecisionTreeClassifier

# Agarro las cosas en los archivos, las guardo en variables y las mando a train_data y labels
def load_training_set():
    train_data = []
    train_labels = []
    with open('TP2.detecciondeformasyobjetos/archivosgenerados/weather-elements-hu-moments.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            class_label = row.pop()  # saca el último elemento de la lista
            floats = []
            for n in row:
                floats.append(float(n))  # momentos de Hu a float
            train_data.append(np.array(floats, dtype=np.float32))  # momentos de Hu
            train_labels.append(np.array([label_to_int(class_label)], dtype=np.int32))  # etiquetas
            # Valores y resultados se necesitan por separados
    train_data = np.array(train_data, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.int32)
    return train_data, train_labels

# Entrenamiento del modelo
def train_model():
    train_data, train_labels = load_training_set()
    tree = DecisionTreeClassifier(max_depth=10)
    tree.fit(train_data, train_labels.ravel())

    # Crear carpeta "models" si no existe
    if not os.path.exists('models'):
        os.makedirs('models')
        print("📁 Creada carpeta 'models'")
    else:
        print("📂 La carpeta 'models' ya existía")

    # Guardar el modelo
    model_path = 'models/modelo_weather_elements.joblib'
    joblib.dump(tree, model_path)
    print(f"✅ Modelo guardado en: {model_path}")
    return tree

if __name__ == "__main__":
    print("🚀 Iniciando entrenamiento del modelo...")
    model = train_model()
    print("✅ Entrenamiento completado exitosamente!")
    print(f"📊 Modelo entrenado con {len(model.classes_)} clases: {model.classes_}")

