import cv2
import csv
import glob
import numpy
import math
import os


def hu_moments_of_file(filename):
    """
    Genera los momentos de Hu para una imagen específica.
    Adaptado para procesar gotas de agua, rayos y lunas.
    """
    image = cv2.imread(filename)
    if image is None:
        print(f"Error: No se pudo cargar la imagen {filename}")
        return None
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Usar umbralización adaptativa para diferentes tipos de formas
    # Parámetros ajustados para gotas, rayos y lunas
    bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 67, 2)

    # Invertir la imagen para que el área de las formas esté llena de 1's
    # Esto es necesario para cv2.findContours
    bin = 255 - bin

    # Operaciones morfológicas para limpiar ruido
    kernel = numpy.ones((3, 3), numpy.uint8)
    # Erosión para eliminar pequeños puntos de ruido
    bin = cv2.morphologyEx(bin, cv2.MORPH_ERODE, kernel)
    
    # Opcional: dilatación para recuperar el tamaño original después de la erosión
    # bin = cv2.morphologyEx(bin, cv2.MORPH_DILATE, kernel)

    contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"Error: No se encontraron contornos en {filename}")
        return None
    
    # Obtener el contorno de área máxima (probablemente la forma principal)
    shape_contour = max(contours, key=cv2.contourArea)

    # Descomentar para chequear que estemos agarrando bien el contorno
    # cv2.drawContours(image, [shape_contour], -1, (0, 255, 0), 2)
    # cv2.imshow("test", image)
    # cv2.waitKey(0)

    # Descomentar para visualizar el contorno detectado
    # height, width = image.shape[:2]
    # max_width, max_height = 800, 600
    # 
    # if width > max_width or height > max_height:
    #     # Calcular el factor de escala
    #     scale = min(max_width/width, max_height/height)
    #     new_width = int(width * scale)
    #     new_height = int(height * scale)
    #     
    #     # Redimensionar la imagen
    #     resized_image = cv2.resize(image, (new_width, new_height))
    #     
    #     # Redimensionar el contorno
    #     resized_contour = []
    #     for point in shape_contour:
    #         x, y = point[0]
    #         new_x = int(x * scale)
    #         new_y = int(y * scale)
    #         resized_contour.append([[new_x, new_y]])
    #     resized_contour = numpy.array(resized_contour, dtype=numpy.int32)
    #     
    #     # Dibujar contorno en la imagen redimensionada
    #     cv2.drawContours(resized_image, [resized_contour], -1, (0, 255, 0), 2)
    #     cv2.imshow("Contorno detectado", resized_image)
    # else:
    #     cv2.drawContours(image, [shape_contour], -1, (0, 255, 0), 2)
    #     cv2.imshow("Contorno detectado", image)
    # 
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Calcular momentos de inercia
    moments = cv2.moments(shape_contour)
    
    # Calcular momentos de Hu
    huMoments = cv2.HuMoments(moments)
    
    # Aplicar escala logarítmica a los momentos de Hu
    for i in range(0, 7):
        if abs(huMoments[i]) > 0:  # Evitar log(0)
            huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]))
        else:
            huMoments[i] = 0
    
    return huMoments


def write_hu_moments_for_category(category_folder, writer):
    """
    Procesa todas las imágenes en una categoría específica y escribe sus momentos de Hu
    """
    # Obtener la ruta completa de la carpeta de categoría
    category_path = os.path.join('fotos', category_folder)
    print(f"category_path:{category_path}")
    # Buscar todas las imágenes en la carpeta

    # Lista solo los archivos (sin incluir subcarpetas)
    files = [os.path.join(category_path, f) for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]


    for file in files:
        print(file)

    
    print(f"Procesando {len(files)} imágenes en la categoría '{category_folder}'")
    
    processed_count = 0
    error_count = 0
    
    for file in files:
        # Obtener solo el nombre del archivo (sin la ruta)
        filename = os.path.basename(file)
        
        # Generar los momentos de Hu
        hu_moments = hu_moments_of_file(file)
        
        if hu_moments is not None:
            # Aplanar el array de momentos de Hu
            flattened = hu_moments.ravel()
            
            # Crear la fila con: [hu_moment_1, hu_moment_2, ..., hu_moment_7, category]
            row = numpy.append(flattened, category_folder)
            
            # Escribir la fila en el archivo CSV
            writer.writerow(row)
            processed_count += 1
            print(f"  ✓ Procesado: {filename}")
        else:
            error_count += 1
            print(f"  ✗ Error procesando: {filename}")
    
    print(f"  Resumen: {processed_count} procesadas exitosamente, {error_count} con errores")


def generate_hu_moments_file():
    """
    Función principal que genera el archivo CSV con todos los momentos de Hu
    para gotas de agua, rayos y lunas
    """
    # Crear el directorio de archivos generados si no existe
    os.makedirs('archivosgenerados', exist_ok=True)
    
    # Verificar que existe la carpeta fotos
    fotos_path = 'fotos'
    if not os.path.exists(fotos_path):
        print(f"Error: La carpeta '{fotos_path}' no existe")
        print("Asegúrate de tener una carpeta 'fotos' con subcarpetas para cada categoría:")
        print("  - fotos/gotasH2O/")
        print("  - fotos/rayos/")
        print("  - fotos/lunas/")
        return
    
    # Obtener todas las subcarpetas en la carpeta fotos
    categories = [d for d in os.listdir(fotos_path) 
                  if os.path.isdir(os.path.join(fotos_path, d))]
    
    if not categories:
        print(f"No se encontraron subcarpetas en '{fotos_path}'")
        print("Crea subcarpetas para cada categoría de imágenes")
        return
    
    print(f"Categorías encontradas: {categories}")
    
    # Crear el archivo CSV
    output_file = 'archivosgenerados/weather-elements-hu-moments.csv'
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
            # Escribir el encabezado del CSV (opcional - descomenta si lo necesitas)
            # header = ['hu_moment_1', 'hu_moment_2', 'hu_moment_3', 'hu_moment_4', 
            #           'hu_moment_5', 'hu_moment_6', 'hu_moment_7', 'category']
            # writer.writerow(header)
            
            # Procesar cada categoría
            total_processed = 0
            for category in categories:
                print(f"\n{'='*50}")
                print(f"Procesando categoría: {category}")
                print(f"{'='*50}")
                
                # Contar archivos antes de procesar

                category_path = os.path.join('fotos', category)
                print(f"category_path:{category_path}")

                # Lista solo los archivos (sin incluir subcarpetas)
                files = [os.path.join(category_path, f) for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]


                
                if files:
                    write_hu_moments_for_category(category, writer)
                    total_processed += len(files)
                else:
                    print(f"  ⚠️  No se encontraron imágenes en la categoría '{category}'")
            
            print(f"\n{'='*50}")
            print(f"PROCESO COMPLETADO")
            print(f"{'='*50}")
            print(f"Archivo generado: {output_file}")
            print(f"Total de imágenes procesadas: {total_processed}")
            print(f"Categorías procesadas: {len(categories)}")
            
    except Exception as e:
        print(f"Error al crear o escribir el archivo CSV: {e}")


if __name__ == "__main__":
    print("Iniciando extracción de momentos de Hu para elementos meteorológicos...")
    print("Categorías esperadas: gotasH2O, rayos, lunas")
    print("-" * 60)
    generate_hu_moments_file()
    print("-" * 60)
    print("Proceso completado.")