import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam

# Параметры
img_height = 224
img_width = 224
batch_size = 32
num_classes = 2  # Правая и левая деталь
epochs = 20

# Подготовка данных
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory='path_to_training_images',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    directory='path_to_validation_images',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

# Загрузка предобученной модели ResNet50
base_model = ResNet50(weights='imagenet', include_top=False,
                      input_shape=(img_height, img_width, 3))

# Заморозка слоев базовой модели
for layer in base_model.layers:
    layer.trainable = False

# Добавление собственных слоев
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Сборка новой модели
model = Model(inputs=base_model.input, outputs=predictions)

# Компиляция модели
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0001),
              metrics=['accuracy'])

# Обучение модели
history = model.fit(train_generator,
                    steps_per_epoch=len(train_generator),
                    validation_data=validation_generator,
                    validation_steps=len(validation_generator),
                    epochs=epochs)

# Сохранение модели
model.save('right_left_classifier.h5')

# Загрузка модели
loaded_model = load_model('right_left_classifier.h5')
classes = ['right', 'left']

# Захват видеопотока с камеры
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Предварительная обработка изображения
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame.astype('float32') / 255.0
    expanded_frame = np.expand_dims(normalized_frame, axis=0)

    # Предсказание
    prediction = loaded_model.predict(expanded_frame)[0]
    predicted_class = classes[np.argmax(prediction)]

    # Визуализация результата
    cv2.putText(frame, f"Предсказанный класс: {predicted_class}", (
        10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Отображение кадра
    cv2.imshow('Video', frame)

    # Нажмите 'q' для выхода
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
