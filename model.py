from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError
import joblib

def build_model(input_shape):
    """Build and compile the neural network model"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(
        optimizer='adam',
        loss=MeanSquaredError(),  # Use class-based loss
        metrics=['mae']
    )
    return model

def save_artifacts(encoders, feature_columns):
    """Save preprocessing artifacts"""
    joblib.dump({
        'encoders': encoders,
        'feature_columns': feature_columns
    }, 'preprocessing_artifacts.pkl')