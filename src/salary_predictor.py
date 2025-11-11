# src/salary_predictor.py

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

class SalaryPredictor:
    def __init__(self):
        self.model = None
        
    def create_salary_dataset(self, n_samples=5000):
        """
        Create synthetic salary dataset
        """
        np.random.seed(42)
        
        careers = ['Software Engineer', 'Data Scientist', 'Financial Analyst', 
                  'Product Manager', 'Business Analyst', 'Doctor', 'Lawyer']
        
        data = {
            'career': np.random.choice(careers, n_samples),
            'experience_years': np.random.uniform(0, 15, n_samples),
            'skills_count': np.random.randint(3, 15, n_samples),
            'education_level': np.random.choice([1, 2, 3, 4], n_samples),  # 1=UG, 2=PG, 3=PhD, 4=Diploma
            'location_tier': np.random.choice([1, 2, 3], n_samples),  # 1=Metro, 2=Tier2, 3=Tier3
            'certifications': np.random.randint(0, 5, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Generate realistic salary based on features
        df['salary_lpa'] = self._generate_realistic_salary(df)
        
        return df
    
    def _generate_realistic_salary(self, df):
        """Generate realistic salary based on features"""
        base_salary = {
            'Software Engineer': 8,
            'Data Scientist': 10,
            'Financial Analyst': 7,
            'Product Manager': 12,
            'Business Analyst': 6,
            'Doctor': 10,
            'Lawyer': 8
        }
        
        salary = df['career'].map(base_salary)
        
        # Experience multiplier
        salary = salary * (1 + df['experience_years'] * 0.15)
        
        # Skills bonus
        salary = salary * (1 + df['skills_count'] * 0.03)
        
        # Education bonus
        salary = salary * (1 + df['education_level'] * 0.1)
        
        # Location adjustment
        location_multiplier = {1: 1.3, 2: 1.0, 3: 0.8}
        salary = salary * df['location_tier'].map(location_multiplier)
        
        # Certifications bonus
        salary = salary * (1 + df['certifications'] * 0.05)
        
        # Add some noise
        salary = salary * np.random.uniform(0.9, 1.1, len(df))
        
        return salary.round(1)
    
    def train_model(self, df):
        """Train salary prediction model"""
        # Encode career
        from sklearn.preprocessing import LabelEncoder
        self.career_encoder = LabelEncoder()
        df['career_encoded'] = self.career_encoder.fit_transform(df['career'])
        
        # Features
        X = df[['career_encoded', 'experience_years', 'skills_count', 
                'education_level', 'location_tier', 'certifications']]
        y = df['salary_lpa']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        print("Training salary prediction model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"Mean Absolute Error: ₹{mae:.2f} LPA")
        print(f"R² Score: {r2:.4f}")
        
        return self.model
    
    def predict(self, career, experience_years, skills_count, education_level, 
                location_tier, certifications):
        """Predict salary for given parameters"""
        career_encoded = self.career_encoder.transform([career])[0]
        
        features = np.array([[
            career_encoded, experience_years, skills_count,
            education_level, location_tier, certifications
        ]])
        
        predicted_salary = self.model.predict(features)[0]
        
        # Calculate confidence interval (±10%)
        lower_bound = predicted_salary * 0.9
        upper_bound = predicted_salary * 1.1
        
        return {
            'predicted_salary': round(predicted_salary, 1),
            'salary_range': [round(lower_bound, 1), round(upper_bound, 1)],
            'currency': 'INR (LPA)'
        }
    
    def save_model(self, path='models/'):
        """Save salary prediction model"""
        joblib.dump(self.model, f'{path}salary_model.pkl')
        joblib.dump(self.career_encoder, f'{path}salary_career_encoder.pkl')
        print(f"Salary model saved to {path}")
    
    def load_model(self, path='models/'):
        """Load salary prediction model"""
        self.model = joblib.load(f'{path}salary_model.pkl')
        self.career_encoder = joblib.load(f'{path}salary_career_encoder.pkl')
