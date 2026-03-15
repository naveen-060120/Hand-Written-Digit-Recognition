import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_analyze_data(train_path, test_path):
    logging.info(f"Loading data from {train_path} and {test_path}...")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Testing file not found at {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    logging.info("Dataset Structure Verification:")
    logging.info(f"Train dataset shape: {train_df.shape}")
    logging.info(f"Test dataset shape: {test_df.shape}")
    
    # Check if first column is label (max value 9, min value 0) and remaining are pixels (max 255)
    train_labels_col = train_df.iloc[:, 0]
    num_classes = train_labels_col.nunique()
    logging.info(f"Number of samples in train: {len(train_df)}")
    logging.info(f"Number of classes: {num_classes}")
    
    logging.info("Checking for missing values...")
    if train_df.isnull().values.any() or test_df.isnull().values.any():
        logging.warning("Missing values found! Filling with 0.")
        train_df.fillna(0, inplace=True)
        test_df.fillna(0, inplace=True)
    else:
        logging.info("No missing values found.")
        
    return train_df, test_df

def preprocess_data(train_df, test_df):
    logging.info("Starting Data Preprocessing...")
    
    # Separate labels and pixel values
    y_train = train_df.iloc[:, 0].values
    X_train = train_df.iloc[:, 1:].values
    y_test = test_df.iloc[:, 0].values
    X_test = test_df.iloc[:, 1:].values
    
    # Normalize pixel values from 0-255 to 0-1
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Reshape pixel vectors into 28x28x1 images
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    # Convert labels to one-hot encoding for 10 classes
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    logging.info("Data Preprocessing complete.")
    logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test

def build_model():
    logging.info("Building Model Architecture...")
    model = Sequential([
        # Input Layer implicitly defined by input_shape in the first layer
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Flatten(),
        
        Dense(128, activation='relu'),
        Dropout(0.3),
        
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    logging.info("Model Compiled.")
    model.summary(print_fn=logging.info)
    return model

def main():
    train_path = 'data/mnist_train.csv'
    test_path = 'data/mnist_test.csv'
    
    # Use alternative paths if they are in data/mnist/
    if not os.path.exists(train_path) and os.path.exists('data/mnist/mnist_train.csv'):
        train_path = 'data/mnist/mnist_train.csv'
    if not os.path.exists(test_path) and os.path.exists('data/mnist/mnist_test.csv'):
        test_path = 'data/mnist/mnist_test.csv'
        
    try:
        # 1. Dataset Analysis
        train_df, test_df = load_and_analyze_data(train_path, test_path)
        
        # 2. Data Preprocessing
        X_train, y_train, X_test, y_test = preprocess_data(train_df, test_df)
        
        # 3 & 4. Model Architecture & Compilation
        model = build_model()
        
        # 5. Model Training
        logging.info("Starting Model Training...")
        history = model.fit(
            X_train, y_train,
            batch_size=64,
            epochs=10,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # 6. Model Evaluation
        logging.info("Evaluating Model...")
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        logging.info(f"Final Test Accuracy: {test_accuracy*100:.2f}%")
        logging.info(f"Final Test Loss: {test_loss:.4f}")
        
        # 7. Model Saving
        model_save_path = 'model.h5'
        model.save(model_save_path)
        logging.info(f"Model saved successfully to {model_save_path}")
        
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)

if __name__ == '__main__':
    main()
