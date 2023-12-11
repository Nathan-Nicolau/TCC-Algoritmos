import os
import cv2
import shutil
import xml.etree.ElementTree as ET

# Mapeamento das classes para IDs de classe (conforme mencionado anteriormente)
class_mapping = {
    'A1': 0,
    'A2': 1,
    'A3': 2,
    'B4': 3,
    'B5': 4,
    'C6': 5,
    'C7': 6,
    'C8': 7,
    'C9': 8,
    'C10': 9,
    'C11': 10,
    'C12': 11,
    'D13': 12,
    'D14': 13,
    'D15': 14,
    'E16': 15,
    'E17': 16,
    'E18': 17,
    'F19': 18,
    'F20': 19,
    'G21': 20,
    'G22': 21,
    'X': 22,
    'Y': 23
}

# Diretório onde estão localizados os arquivos XML
xml_dir = "C:\\Users\\...\\Downloads\\Data\\24_chromosomes_object\\annotations"
jpg_dir = "C:\\Users\\...\\Downloads\\Data\\24_chromosomes_object\\JEPG"

# Diretório onde as anotações YOLO serão armazenadas
output_dir = "C:\\Users\\...\\Documents\\Projetos\\ReconhecimentoCromossomoYolo\\Labels"
output_img = "C:\\Users\\...\\Documents\\Projetos\\ReconhecimentoCromossomoYolo\\Images"

contador = -1
for xml_file in os.listdir(path=xml_dir):

    contador += 1

    tree = ET.parse(xml_dir + '\\' + xml_file)
    root = tree.getroot()

    # Informações da imagem
    image_filename = root.find("filename").text
    image = cv2.imread(jpg_dir + '\\' + image_filename)

    image_width = float(root.find("size/width").text)
    image_height = float(root.find("size/height").text)

    print(contador, '-', xml_file, '->', image_filename)

    yolo_annotations = []

    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name in class_mapping:
            class_id = class_mapping[class_name]
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            if xmin < 0:
                xmin = 0
            elif xmax < 0:
                xmax = 0
            elif ymax < 0:
                ymax = 0
            elif ymin < 0:
                ymin = 0

            # Calcula as coordenadas normalizadas
            x_center = (xmin + xmax) / (2.0 * image_width)
            y_center = (ymin + ymax) / (2.0 * image_height)
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height

            yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
            cv2.putText(image, class_name, (xmin + 10, ymin + 15), cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image.shape[0], (0, 0, 255), 1)

    # Cria o arquivo de anotação YOLO correspondente
    yolo_annotation_file = os.path.join(output_dir, image_filename.replace(".jpg", ".txt"))

    with open(yolo_annotation_file, "w") as f:
        for annotation in yolo_annotations:
            f.write(annotation + "\n")

    cv2.imwrite(filename=output_img + '\\' + image_filename, img=image)
