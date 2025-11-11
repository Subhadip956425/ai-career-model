# src/model_training.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class CareerRecommendationModel:
    def __init__(self, model_type='xgboost'):
        """
        Initialize model
        model_type: 'random_forest' or 'xgboost'
        """
        self.model_type = model_type
        self.model = None
        self.target_encoder = None
        
    def prepare_data(self, df, test_size=0.2):
        """
        Split data into train and test sets
        """
        # Separate features and target
        feature_cols = ['age', 'gender', 'location', 'education_level', 'stream', 
                       'academic_percentage', 'career_interest', 'learning_style', 
                       'future_goal', 'tech_domain']
        
        X = df[feature_cols]
        y = df['career_path']
        
        # Encode target variable
        from sklearn.preprocessing import LabelEncoder
        self.target_encoder = LabelEncoder()
        y_encoded = self.target_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        return X_train_balanced, X_test, y_train_balanced, y_test
    
    def train_model(self, X_train, y_train):
        """
        Train the selected model
        """
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        else:  # XGBoost
            self.model = XGBClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='mlogloss',
                use_label_encoder=False
            )
        
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        print("Training completed!")
        
        return self.model
    
    def hyperparameter_tuning(self, X_train, y_train):
        """
        Perform hyperparameter tuning using GridSearchCV
        """
        if self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
        else:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [6, 10, 15],
                'learning_rate': [0.01, 0.1, 0.2]
            }
            model = XGBClassifier(random_state=42, eval_metric='mlogloss', use_label_encoder=False)
        
        print("Performing hyperparameter tuning...")
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        self.model = grid_search.best_estimator_
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance
        """
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Accuracy: {accuracy:.4f}")
        
        # Classification report
        target_names = self.target_encoder.classes_
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Cross-validation score
        feature_cols = ['age', 'gender', 'location', 'education_level', 'stream', 
                       'academic_percentage', 'career_interest', 'learning_style', 
                       'future_goal', 'tech_domain']
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred
        }
    
    def plot_feature_importance(self, feature_names):
        """
        Plot feature importance
        """
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title('Feature Importance')
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            plt.savefig('models/feature_importance.png')
            plt.close()
            
            print("\nFeature Importance:")
            for i in indices:
                print(f"{feature_names[i]}: {importances[i]:.4f}")
    
    def save_model(self, path='models/'):
        """Save trained model and target encoder"""
        joblib.dump(self.model, f'{path}career_model.pkl')
        joblib.dump(self.target_encoder, f'{path}target_encoder.pkl')
        print(f"\nModel saved to {path}")
    
    def load_model(self, path='models/'):
        """Load trained model and target encoder"""
        self.model = joblib.load(f'{path}career_model.pkl')
        self.target_encoder = joblib.load(f'{path}target_encoder.pkl')
