# src/feature_engineering.py

import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        pass
    
    def create_derived_features(self, df):
        """
        Create additional features from existing ones
        """
        df_new = df.copy()
        
        # Create age groups
        df_new['age_group'] = pd.cut(df_new['age'], 
                                      bins=[0, 18, 22, 25, 100], 
                                      labels=['Teen', 'Early_20s', 'Mid_20s', 'Late'])
        
        # Create academic performance category
        df_new['academic_category'] = pd.cut(df_new['academic_percentage'], 
                                             bins=[0, 60, 75, 85, 100],
                                             labels=['Average', 'Good', 'Very_Good', 'Excellent'])
        
        # Interaction features
        df_new['stream_interest_match'] = (
            df_new['stream'].astype(str) + '_' + df_new['career_interest'].astype(str)
        )
        
        return df_new
    
    def get_feature_importance_names(self):
        """
        Return feature names for model training
        """
        return ['age', 'gender', 'location', 'education_level', 'stream', 
                'academic_percentage', 'career_interest', 'learning_style', 
                'future_goal', 'tech_domain']
