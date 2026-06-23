import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data():
    # Load dataset
    df = pd.read_csv("churn_dataset.csv")
    
    # Drop customerID as it's not useful for prediction
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        
    # TotalCharges is object because of empty spaces " " for new customers
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill NaN values in TotalCharges with 0 (since they are new customers with tenure 0)
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # Encode categorical variables
    categorical_columns = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])
        
    # Save preprocessed data
    df.to_csv("churn_preprocessing.csv", index=False)
    print("Preprocessing completed. File saved as churn_preprocessing.csv")

if __name__ == "__main__":
    preprocess_data()
