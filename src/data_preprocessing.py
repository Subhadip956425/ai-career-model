import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def create_synthetic_dataset(self, n_samples=5000):
        """
        Create synthetic training data for career recommendation
        """
        np.random.seed(42)
        
        # Define categorical options
        genders = ['Male', 'Female', 'Other']
        education_levels = ['10th Pass', '12th Pass', 'Undergraduate', 'Graduate', 'Diploma', 'ITI']
        streams = ['Science', 'Commerce', 'Arts', 'Vocational']
        career_interests = ['Engineering', 'Medicine', 'Design', 'Law', 'Business', 
                           'Government Jobs', 'AI/ML', 'Healthcare', 'Arts & Humanities', 
                           'Commerce & Finance', 'Teaching', 'Research']
        learning_styles = ['Online', 'Offline', 'Hybrid']
        future_goals = ['Higher Studies Abroad', 'Competitive Exams', 'Job-Oriented Training', 
                       'Entrepreneurship', 'Research', 'Industry Certification']
        tech_domains = ['AI/ML', 'Web Development', 'Data Science', 'Cybersecurity', 
                       'Cloud Computing', 'Mobile Development', 'DevOps', 'Not Interested']
        
        # Target career paths (33 categories based on your SRS)
        career_paths = [
            'Software Engineer', 'Data Scientist', 'Product Manager', 'Business Analyst',
            'Doctor', 'Nurse', 'Pharmacist', 'Medical Researcher',
            'Graphic Designer', 'UI/UX Designer', 'Fashion Designer', 'Architect',
            'Lawyer', 'Legal Advisor', 'Judge', 'Corporate Lawyer',
            'Entrepreneur', 'Marketing Manager', 'Financial Analyst', 'Accountant',
            'Teacher', 'Professor', 'Education Consultant', 'Training Specialist',
            'Civil Services', 'Bank PO', 'SSC', 'Defense Services',
            'Research Scientist', 'Lab Technician', 'Content Writer', 
            'Digital Marketing Specialist', 'HR Manager'
        ]
        
        data = {
            'age': np.random.randint(16, 35, n_samples),
            'gender': np.random.choice(genders, n_samples),
            'location': np.random.choice(['Urban', 'Semi-Urban', 'Rural'], n_samples),
            'education_level': np.random.choice(education_levels, n_samples, 
                                               p=[0.05, 0.15, 0.40, 0.25, 0.10, 0.05]),
            'stream': np.random.choice(streams, n_samples, p=[0.45, 0.30, 0.20, 0.05]),
            'academic_percentage': np.random.uniform(50, 95, n_samples),
            'career_interest': np.random.choice(career_interests, n_samples),
            'learning_style': np.random.choice(learning_styles, n_samples),
            'future_goal': np.random.choice(future_goals, n_samples),
            'tech_domain': np.random.choice(tech_domains, n_samples),
            'career_path': np.random.choice(career_paths, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Apply realistic constraints
        df = self._apply_business_logic(df, career_paths)
        
        return df
    
    def _apply_business_logic(self, df, career_paths):
        """
        Apply realistic relationships between features and target
        """
        for idx, row in df.iterrows():
            # Engineering path logic
            if row['stream'] == 'Science' and row['career_interest'] == 'Engineering':
                df.at[idx, 'career_path'] = np.random.choice(['Software Engineer', 'Data Scientist'])
            
            # Medicine path logic
            elif row['stream'] == 'Science' and row['career_interest'] == 'Medicine':
                df.at[idx, 'career_path'] = np.random.choice(['Doctor', 'Medical Researcher', 'Pharmacist'])
            
            # Business path logic
            elif row['stream'] == 'Commerce' and row['career_interest'] == 'Business':
                df.at[idx, 'career_path'] = np.random.choice(['Financial Analyst', 'Accountant', 'Entrepreneur'])
            
            # Design path logic
            elif row['career_interest'] == 'Design':
                df.at[idx, 'career_path'] = np.random.choice(['Graphic Designer', 'UI/UX Designer', 'Architect'])
            
            # Government jobs logic
            elif row['career_interest'] == 'Government Jobs':
                df.at[idx, 'career_path'] = np.random.choice(['Civil Services', 'Bank PO', 'SSC', 'Defense Services'])
        
        return df
    
    def encode_features(self, df, is_training=True):
        """
        Encode categorical features and scale numerical features
        """
        df_encoded = df.copy()
        
        categorical_cols = ['gender', 'location', 'education_level', 'stream', 
                           'career_interest', 'learning_style', 'future_goal', 'tech_domain']
        
        for col in categorical_cols:
            if is_training:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                self.label_encoders[col] = le
            else:
                df_encoded[col] = self.label_encoders[col].transform(df_encoded[col])
        
        # Scale numerical features
        numerical_cols = ['age', 'academic_percentage']
        if is_training:
            df_encoded[numerical_cols] = self.scaler.fit_transform(df_encoded[numerical_cols])
        else:
            df_encoded[numerical_cols] = self.scaler.transform(df_encoded[numerical_cols])
        
        return df_encoded
    
    def save_preprocessors(self, path='models/'):
        """Save label encoders and scaler"""
        joblib.dump(self.label_encoders, f'{path}label_encoders.pkl')
        joblib.dump(self.scaler, f'{path}scaler.pkl')
    
    def load_preprocessors(self, path='models/'):
        """Load label encoders and scaler"""
        self.label_encoders = joblib.load(f'{path}label_encoders.pkl')
        self.scaler = joblib.load(f'{path}scaler.pkl')
