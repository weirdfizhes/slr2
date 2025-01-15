import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

def build_model(input_shape, num_classes):
    """
    Rebuild the model architecture exactly as it was during training
    """
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=10000,
        decay_rate=0.9,
        staircase=True)
   
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def run_prediction(data):
    # Get the directory of the current script
    script_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Construct paths relative to the script's location
    seagrass_path = os.path.join(script_directory, '../model/seagrass_class.npy')
    model_path = os.path.join(script_directory, '../model/BiLSTM_Model.h5')
    
    # Load seagrass classes
    seagrass = np.load(seagrass_path)
    
    # Rebuild and load the model
    try:
        # Recreate the model architecture
        model = build_model(input_shape=(5, 1), num_classes=len(seagrass))
        
        # Load weights
        model.load_weights(model_path)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        raise
    
    predictions = []
    for row in data:
        # Prepare features
        features_array = np.array([
            row['temp'],
            row['salinity'],
            row['do'],
            row['ph'],
            row['tss']
        ])
        
        # Use pad_sequences to match the input shape
        features = pad_sequences([features_array], maxlen=5, padding='post').reshape(1, 5, 1)
        
        # Make prediction
        try:
            pred = model.predict(features)
            predicted_class_index = np.argmax(pred)
            predicted_class = int(seagrass[predicted_class_index])
            confidence_seagrass = pred[0, predicted_class_index]
           
            # Apply custom rules
            confidence_seagrass *= 2
            predicted_class = confidence_seagrass
           
            if -4 <= row['bathy'] <= 0:
                predicted_class = predicted_class
            else:
                predicted_class = 0
           
            predictions.append(predicted_class)

        except Exception as e:
            print(f"Prediction error for row: {row}")
            print(f"Error details: {e}")
            predictions.append(None)
    
    return predictions