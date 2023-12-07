import os
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

# Função para ler informações das anotações XML


def read_xml_annotation(xml_path, target_classes):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_name = root.find('filename').text

    for obj in root.findall("object"):
        class_name = obj.find("name").text
        bbox = obj.find("bndbox")
        x = int(bbox.find("xmin").text)
        y = int(bbox.find("ymin").text)
        width = int(bbox.find("xmax").text) - x
        height = int(bbox.find("ymax").text) - y

        return {'image_name': image_name, 'class_name': class_name,
                'x': x, 'y': y, 'width': width, 'height': height, 'xml_path': xml_path}


# Caminho para a pasta principal com 24 subpastas e arquivos XML de anotações
annotations_path = 'C:\\Users\\Gamer\\Documents\\Datasets\\Cromossomos\\Data\\24_chromosomes_object\\annotations\\'

# Caminho para a pasta com as imagens originais do Dataset
images_path = 'C:\\Users\\Gamer\\Documents\\Datasets\\Cromossomos\\Data\\24_chromosomes_object\\JPEG\\'

# Vetor que armazena o nome das classes existentes
target_classes = ['A1', 'A2', 'A3', 'B4', 'B5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
                  'C12', 'D13', 'D14', 'D15', 'E16', 'E17', 'E18', 'F19', 'F20', 'G21', 'G22', 'X', 'Y']

# Lista para armazenar as informações de anotações
annotations_data = []

# Iteração sobre os arquivos de anotações na pasta
for xml_file in os.listdir(annotations_path):
    annotation_info = read_xml_annotation(
        annotations_path + xml_file, target_classes)
    annotations_data.append(annotation_info)

# Um DataFrame Pandas com as informações de anotações
annotations_df = pd.DataFrame(annotations_data)

# Adiciona o caminho completo das imagens ao DataFrame
annotations_df['image_path'] = images_path + annotations_df['image_name']

# Divisão dos dados em treinamento e validação
train_df, val_df = train_test_split(
    annotations_df, test_size=0.2, random_state=42)


train_datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rescale=1./255
)

# Modifica o gerador de treinamento para usar o DataFrame personalizado
# Seguindo o padrão estabelecido no modelo VGG16, temos:
# imagens redimensionadas num tamanho 224x224
# batch_size de 64 imagens
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='image_path',
    y_col='class_name',
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical',
    classes=target_classes
)

# Crie um gerador de validação sem aumentação
val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='image_path',
    y_col='class_name',
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical',
    classes=target_classes
)

# O modelo VGG16 utilizado comp base
base_model = VGG16(
    weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

# Criação de um modelo de rede básico para ser unido ao VGG16, tendo adições de
# uma camada de pooling, uma camada densa, um Dropout para tentativa de contornar uma possibilidade
# de  'overfitting' e uma última camada de classificação, tendo como número total de neurônios, o mesmo número de classes
model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(len(target_classes), activation='softmax'))

# Taxa de aorendizado padrão
learning_rate = 0.0001

# Um otimizador de taxa de aprendizado
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Compilação do modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy', metrics=['accuracy'])

# Parâmetro para interrompimento do treinamento da rede, de acordo com
# a métrica monitorada e o valor de 'patience' declarado, nesse caso, de 5 epócas
early_stopping = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True)

# Treinamento do modelo com a quantidade de 100 épocas estalebecidas inicialmente
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=val_generator,
    callbacks=[early_stopping]
)

# Comando para salvar o modelo após o treinamento
model.save(f'Rede_VGG16_Treinada_com_anotacoes_multi_classes.h5')


# Variáveis que vão receber as métricas do treinamento
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Gráficos de treinamento e teste
plt.figure(figsize=(14, 5))

# Gráfico de Acurácia
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title(f'Training and Validation Accuracy')

# Gráfico de Perda
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title(f'Training and Validation Loss')

plt.show()
