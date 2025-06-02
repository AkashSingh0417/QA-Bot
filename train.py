import pandas as pd
from model import build_model, save_artifacts
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    data = pd.read_csv("tech_support_dataset_enhanced.csv")
    
    # Data cleaning
    data['Resolution_Time'] = data['Resolution_Time'].str.replace(' minutes', '').astype(float)
    data = data.drop('Conversation_ID', axis=1)
    
    # Initialize encoders
    encoders = {}
    categorical_cols = ['Customer_Issue', 'Tech_Response', 
                       'Issue_Category', 'Issue_Status', 'Sentiment']
    
    # Label Encoding
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le
    
    # One-Hot Encoding for Issue_Status
    ohe = OneHotEncoder()
    status_encoded = ohe.fit_transform(data[['Issue_Status']]).toarray()
    status_df = pd.DataFrame(status_encoded, 
                           columns=[f"Issue_Status_{i}" for i in range(status_encoded.shape[1])])
    data = pd.concat([data.drop('Issue_Status', axis=1), status_df], axis=1)
    encoders['Issue_Status_ohe'] = ohe
    
    # Normalization (only on input features, excluding Resolution_Time)
    scaler = MinMaxScaler()
    numerical_cols = ['Response_Time_Min']  # Only include input features
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    encoders['scaler'] = scaler
    
    # Prepare features and target
    X = data.drop('Resolution_Time', axis=1)
    y = data['Resolution_Time']
    
    return X, y, encoders, X.columns.tolist()
def main():
    # Load and preprocess data
    X, y, encoders, feature_columns = load_and_preprocess_data()
    
    # Save artifacts
    save_artifacts(encoders, feature_columns)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build and train model
    model = build_model(X.shape[1])
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Save model
    model.save("tech_support_model.h5")
    print("Model training complete and saved!")

if __name__ == "__main__":
    main()