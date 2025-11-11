# train_complete_system.py

from src.data_preprocessing import DataPreprocessor
from src.model_training import CareerRecommendationModel
from src.salary_predictor import SalaryPredictor
import os

def main():
    print("=" * 60)
    print("COMPLETE AI CAREER RECOMMENDATION SYSTEM TRAINING")
    print("=" * 60)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/roadmaps', exist_ok=True)
    
    # ========== PART 1: Career Prediction Model ==========
    print("\n[PART 1] Training Career Prediction Model...\n")
    
    preprocessor = DataPreprocessor()
    df = preprocessor.create_synthetic_dataset(n_samples=10000)
    print(f"✓ Created dataset with {len(df)} samples")
    
    df_encoded = preprocessor.encode_features(df, is_training=True)
    preprocessor.save_preprocessors()
    print("✓ Features encoded and preprocessors saved")
    
    career_model = CareerRecommendationModel(model_type='xgboost')
    X_train, X_test, y_train, y_test = career_model.prepare_data(df_encoded)
    print(f"✓ Data split - Train: {X_train.shape}, Test: {X_test.shape}")
    
    career_model.train_model(X_train, y_train)
    results = career_model.evaluate_model(X_test, y_test)
    print(f"✓ Career Model Accuracy: {results['accuracy']:.4f}")
    
    feature_names = ['age', 'gender', 'location', 'education_level', 'stream', 
                    'academic_percentage', 'career_interest', 'learning_style', 
                    'future_goal', 'tech_domain']
    career_model.plot_feature_importance(feature_names)
    career_model.save_model()
    print("✓ Career prediction model saved")
    
    # ========== PART 2: Salary Prediction Model ==========
    print("\n[PART 2] Training Salary Prediction Model...\n")
    
    salary_predictor = SalaryPredictor()
    salary_df = salary_predictor.create_salary_dataset(n_samples=5000)
    print(f"✓ Created salary dataset with {len(salary_df)} samples")
    
    salary_predictor.train_model(salary_df)
    salary_predictor.save_model()
    print("✓ Salary prediction model saved")
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nModels saved in 'models/' directory:")
    print("  ✓ career_model.pkl")
    print("  ✓ target_encoder.pkl")
    print("  ✓ label_encoders.pkl")
    print("  ✓ scaler.pkl")
    print("  ✓ salary_model.pkl")
    print("  ✓ salary_career_encoder.pkl")
    print("\nNext steps:")
    print("  1. Run Flask API: python api/app.py")
    print("  2. Test endpoints: python test_enhanced_api.py")
    print("=" * 60)

if __name__ == '__main__':
    main()
