import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# تحميل البيانات
train_datagen = ImageDataGenerator(zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15)
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory('ECG_Image_data\\train', target_size=(224, 224), batch_size=32, shuffle=True, class_mode='categorical')
test_generator = test_datagen.flow_from_directory('ECG_Image_data\\test', target_size=(224, 224), batch_size=32, shuffle=False, class_mode='categorical')


def VGG16():
    input_layer = Input(shape=(224, 224, 3))
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='vgg16')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(256, activation='relu', name='fc1')(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    output_layer = Dense(6, activation='softmax', name='output')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# تهيئة النموذج
model = VGG16()
model.summary()

# تعريف وتهيئة معلمات التدريب
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=20)
opt = Adam(learning_rate=1e-4)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# تدريب النموذج مع البيانات
mc = ModelCheckpoint('vgg16_best_model100.keras', monitor='val_accuracy', mode='max', save_best_only=True)
H = model.fit(train_generator, validation_data=test_generator, epochs=30, verbose=1, callbacks=[mc, es])

# عرض نتائج التدريب
plt.plot(H.history['accuracy'])
plt.plot(H.history['val_accuracy'])
plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Accuracy', 'Validation Accuracy', 'loss', 'Validation Loss'])
plt.show()
