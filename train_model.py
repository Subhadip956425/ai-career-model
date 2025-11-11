# train_model.py

from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import CareerRecommendationModel

def main():
    print("=== Career Recommendation System - Model Training ===\n")
    
    # Step 1: Create/Load Dataset
    print("Step 1: Creating synthetic dataset...")
    preprocessor = DataPreprocessor()
    df = preprocessor.create_synthetic_dataset(n_samples=10000)
    print(f"Dataset created with {len(df)} samples\n")
    
    # Step 2: Feature Engineering
    print("Step 2: Feature engineering...")
    engineer = FeatureEngineer()
    # df = engineer.create_derived_features(df)  # Optional
    
    # Step 3: Encode features
    print("Step 3: Encoding features...")
    df_encoded = preprocessor.encode_features(df, is_training=True)
    preprocessor.save_preprocessors()
    print("Preprocessors saved\n")
    
    # Step 4: Train Model
    print("Step 4: Training model...")
    model = CareerRecommendationModel(model_type='xgboost')  # or 'random_forest'
    
    X_train, X_test, y_train, y_test = model.prepare_data(df_encoded)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}\n")
    
    # Option 1: Quick training
    model.train_model(X_train, y_train)
    
    # Option 2: Hyperparameter tuning (takes longer but better results)
    # model.hyperparameter_tuning(X_train, y_train)
    
    # Step 5: Evaluate
    print("\nStep 5: Evaluating model...")
    results = model.evaluate_model(X_test, y_test)
    
    # Step 6: Feature Importance
    print("\nStep 6: Analyzing feature importance...")
    feature_names = ['age', 'gender', 'location', 'education_level', 'stream', 
                    'academic_percentage', 'career_interest', 'learning_style', 
                    'future_goal', 'tech_domain']
    model.plot_feature_importance(feature_names)
    
    # Step 7: Save Model
    print("\nStep 7: Saving model...")
    model.save_model()
    
    print("\n=== Training Complete! ===")
    print("Model saved to 'models/' directory")
    print("You can now run the Flask API to make predictions")

if __name__ == '__main__':
    main()
