import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 128
NUM_CLASSES = 4
EPOCHS = 50

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Modified U-Net Classification Architecture
def unet_classifier(input_size=(224,224,3), num_classes=4):
    inputs = Input(input_size)
    
    # Encoder
    c1 = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3,3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2,2))(c1)
    
    c2 = Conv2D(128, (3,3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3,3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2,2))(c2)
    
    c3 = Conv2D(256, (3,3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3,3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2,2))(c3)
    
    # Bottleneck
    c4 = Conv2D(512, (3,3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3,3), activation='relu', padding='same')(c4)
    
    # Classification Head
    gap = GlobalAveragePooling2D()(c4)
    dense1 = Dense(512, activation='relu')(gap)
    dropout = Dropout(0.5)(dense1)
    outputs = Dense(num_classes, activation='softmax')(dropout)
    
    return Model(inputs, outputs)

# Data Preparation (Assuming directory structure from Kaggle dataset)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    './Training',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    './Testing',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Model Compilation
model = unet_classifier()
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint('best_model.keras', 
                                      save_best_only=True,
                                      monitor='val_accuracy',
                                      mode='max'),
    tf.keras.callbacks.EarlyStopping(patience=10, 
                                   restore_best_weights=True),
    tf.keras.callbacks.TensorBoard(log_dir='./logs')
]

with tf.device('/GPU:0'):  # Explicit GPU device placement
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

# Save final model in Keras format
model.save('brain_tumor_classifier.keras') 

# For TensorFlow Serving compatibility (optional)
model.save('saved_model/brain_tumor_classifier', save_format='tf')
