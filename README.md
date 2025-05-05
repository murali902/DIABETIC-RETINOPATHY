NOTE:"ONCE PREVIEW THE PDF FILE AS EXTRACTED FROM VSCODE TO HTML VIA PDF FORMAT FILE NAME:"(DIABETIC RETINOAPTHIYHTMLEXPORTPDF)" AS TO BE MORE CLARITY AND CORRECT INFO OF THE PROJECT.."df = pd.read_csv(r'train.csv')

diagnosis_dict_binary = {
    0: 'No_DR',
    1: 'DR',
    2: 'DR',
    3: 'DR',
    4: 'DR'
}

diagnosis_dict = {
    0: 'No_DR',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe',
    4: 'Proliferate_DR',
}


df['binary_type'] =  df['diagnosis'].map(diagnosis_dict_binary.get)
df['type'] = df['diagnosis'].map(diagnosis_dict.get)
df.head()from tensorflow import lite
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import random, os
import shutil
import matplotlib.pyplot as plt
from matplotlib.image import imread
# from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import categorical_accuracy
from sklearn.model_selection import train_test_splitprint(test['type'].value_counts(), '\n')
Building the model

model = tf.keras.Sequential([
    # Layer 1: Convolutional Layer
    layers.Conv2D(8, (3,3), padding="valid", input_shape=(224,224,3), activation = 'relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.BatchNormalization(),

      # Layer 2: Convolutional Layer
    layers.Conv2D(16, (3,3), padding="valid", activation = 'relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.BatchNormalization(),
     # Layer 3: Convolutional Layer
    layers.Conv2D(32, (4,4), padding="valid", activation = 'relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.BatchNormalization(),
     # Flatten the data
    layers.Flatten(),
    # Fully Connected Layer 1 (Dense Layer)
    layers.Dense(32, activation = 'relu'),
    layers.Dropout(0.15),
      # Output Layer
    layers.Dense(2, activation = 'softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-5),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['acc'])

history = model.fit(train_batches,
                    epochs=30,
                    validation_data=val_batches)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Evaluate the model using the 'evaluate' method
loss, acc = model.evaluate(test_batches, verbose=1)

# Print results
print("Loss: ", loss)
print("Accuracy: ", acc)

# Generate predictions from the test data
y_pred = model.predict(test_batches)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert predictions to class labels

# Get true labels from test_batches
y_true = test_batches.classes  # Assuming test_batches is a generator with true labels

# Generate the confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_batches.class_indices.keys())
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

import gradio as gr
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model(r'CNN.h5')  # Replace with your model path

# Function to predict image
def predict_image(img):
    # Preprocess the image (resizing and scaling)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  # Rescale the image
    
    # Add batch dimension (as the model expects 4D input)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    predictions = model.predict(img_array)
    
    # Get predicted class
    class_names = ['Diabetic Retinopathy', 'No Diabetic Retinopathy']  # Replace with your class names
    predicted_class = class_names[np.argmax(predictions)]
    
    return predicted_class, predictions[0][np.argmax(predictions)]

# Build Gradio interface
iface = gr.Interface(fn=predict_image, 
                     inputs=gr.Image(type="pil"), 
                     outputs=[gr.Label(), gr.Textbox()],
                     live=True)

# Launch the Gradio app
iface.launch()
