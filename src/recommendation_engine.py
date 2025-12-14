# src/recommendation_engine.py (Complete Fixed Version)

import pandas as pd
import numpy as np
import joblib
from typing import List, Dict, Any


class RecommendationEngine:
    def __init__(self, model_path='models/'):
        self.model = joblib.load(f'{model_path}career_model.pkl')
        self.target_encoder = joblib.load(f'{model_path}target_encoder.pkl')
        self.label_encoders = joblib.load(f'{model_path}label_encoders.pkl')
        self.scaler = joblib.load(f'{model_path}scaler.pkl')
        
        # Career knowledge base
        self.career_database = self._build_career_database()
        
        # Learning resources database
        self.learning_resources = self._build_learning_resources()
    
    def _build_career_database(self):
        """Enhanced career database with required skills"""
        return {
            'Software Engineer': {
                'description': 'Design, develop, and maintain software applications',
                'required_skills': [
                    'Programming (Java/Python/JavaScript)',
                    'Data Structures & Algorithms',
                    'Object-Oriented Programming',
                    'Database Management',
                    'Version Control (Git)',
                    'Web Development',
                    'API Development',
                    'Testing & Debugging',
                    'Problem Solving',
                    'System Design'
                ],
                'colleges': [
                    'IIT Bombay - B.Tech Computer Science',
                    'IIT Delhi - B.Tech Software Engineering',
                    'BITS Pilani - B.E. Computer Science',
                    'NIT Trichy - B.Tech IT',
                    'VIT Vellore - B.Tech CSE'
                ],
                'courses': [
                    'Data Structures & Algorithms',
                    'Object-Oriented Programming',
                    'Database Management Systems',
                    'Web Development (MERN/MEAN Stack)',
                    'Cloud Computing (AWS/Azure)',
                    'DevOps & CI/CD'
                ],
                'job_profiles': [
                    {'title': 'Full Stack Developer', 'salary_range': [6, 12], 'experience': 0},
                    {'title': 'Backend Engineer', 'salary_range': [8, 15], 'experience': 0},
                    {'title': 'Frontend Developer', 'salary_range': [5, 10], 'experience': 0},
                    {'title': 'DevOps Engineer', 'salary_range': [10, 18], 'experience': 2},
                    {'title': 'Software Architect', 'salary_range': [20, 35], 'experience': 5}
                ],
                'roadmap': {
                    'Year 1': 'Learn programming fundamentals, DSA basics, build small projects',
                    'Year 2': 'Master advanced DSA, contribute to open source, internships',
                    'Year 3': 'Specialize in full-stack/backend, system design, competitive programming',
                    'Year 4': 'Build portfolio projects, prepare for interviews, apply for jobs'
                },
                'certifications': [
                    'AWS Certified Solutions Architect',
                    'Google Cloud Professional',
                    'Microsoft Azure Developer'
                ],
                'avg_salary': {'fresher': [8, 15], '3-5_years': [15, 30], '5+_years': [30, 60]}
            },
            'Data Scientist': {
                'description': 'Analyze complex data to help organizations make better decisions',
                'required_skills': [
                    'Python/R Programming',
                    'Statistics & Probability',
                    'Machine Learning',
                    'Deep Learning',
                    'Data Wrangling (Pandas, NumPy)',
                    'SQL',
                    'Data Visualization',
                    'Feature Engineering',
                    'Model Deployment',
                    'Business Analytics'
                ],
                'colleges': [
                    'IIT Madras - B.Tech Data Science',
                    'ISI Kolkata - B.Stat/M.Stat',
                    'IIIT Hyderabad - B.Tech CSE (AI/ML)',
                    'IIT Kharagpur - M.Tech Data Analytics',
                    'Great Lakes Institute - PGP Data Science'
                ],
                'courses': [
                    'Statistics & Probability',
                    'Machine Learning',
                    'Deep Learning',
                    'Python for Data Science',
                    'SQL & Database Management',
                    'Data Visualization (Tableau, Power BI)'
                ],
                'job_profiles': [
                    {'title': 'Data Analyst', 'salary_range': [5, 10], 'experience': 0},
                    {'title': 'ML Engineer', 'salary_range': [10, 20], 'experience': 1},
                    {'title': 'Data Scientist', 'salary_range': [12, 25], 'experience': 2},
                    {'title': 'AI Research Scientist', 'salary_range': [20, 40], 'experience': 3},
                    {'title': 'Chief Data Officer', 'salary_range': [40, 80], 'experience': 8}
                ],
                'roadmap': {
                    'Year 1': 'Python, Statistics, SQL, Excel, basic ML',
                    'Year 2': 'Advanced ML, Deep Learning, Kaggle competitions',
                    'Year 3': 'Specialize in NLP/Computer Vision, Big Data tools',
                    'Year 4': 'Build end-to-end ML projects, research papers, job applications'
                },
                'certifications': [
                    'TensorFlow Developer Certificate',
                    'AWS Machine Learning Specialty',
                    'Google Data Analytics Professional'
                ],
                'avg_salary': {'fresher': [10, 18], '3-5_years': [20, 40], '5+_years': [40, 80]}
            },
            'Financial Analyst': {
                'description': 'Analyze financial data to help businesses make investment decisions',
                'required_skills': [
                    'Financial Modeling',
                    'Excel & VBA',
                    'Accounting Principles',
                    'Investment Analysis',
                    'Corporate Finance',
                    'Data Analysis',
                    'Financial Reporting',
                    'Risk Management',
                    'Market Research',
                    'Communication Skills'
                ],
                'colleges': [
                    'SRCC Delhi - B.Com (Hons)',
                    'St. Xavier\'s Kolkata - B.Com',
                    'IIM Ahmedabad - MBA Finance',
                    'ISB Hyderabad - MBA',
                    'XLRI Jamshedpur - MBA Finance'
                ],
                'courses': [
                    'Financial Accounting',
                    'Corporate Finance',
                    'Investment Analysis',
                    'Financial Modeling',
                    'Excel & VBA',
                    'CFA Program'
                ],
                'job_profiles': [
                    {'title': 'Junior Analyst', 'salary_range': [4, 8], 'experience': 0},
                    {'title': 'Financial Analyst', 'salary_range': [8, 15], 'experience': 2},
                    {'title': 'Investment Banker', 'salary_range': [15, 30], 'experience': 3},
                    {'title': 'Portfolio Manager', 'salary_range': [20, 40], 'experience': 5},
                    {'title': 'CFO', 'salary_range': [50, 150], 'experience': 10}
                ],
                'roadmap': {
                    'Year 1': 'B.Com/BBA, learn Excel, basic accounting',
                    'Year 2-3': 'Internships, CFA Level 1, financial modeling',
                    'Year 4': 'Campus placements or MBA preparation',
                    'Post-MBA': 'Investment banking, equity research, or corporate finance'
                },
                'certifications': [
                    'CFA (Chartered Financial Analyst)',
                    'FRM (Financial Risk Manager)',
                    'CPA (Certified Public Accountant)'
                ],
                'avg_salary': {'fresher': [6, 12], '3-5_years': [15, 35], '5+_years': [35, 80]}
            },

            'Product Manager': {
            'description': 'Lead product development and strategy',
            'required_skills': ['Product Strategy', 'User Research', 'Agile', 'Communication', 'Market Analysis'],
            'colleges': ['IIM Bangalore - MBA', 'ISB Hyderabad - MBA', 'MDI Gurgaon - MBA'],
            'courses': ['Product Management', 'Agile Methodology', 'UX Design'],
            'job_profiles': [
                {'title': 'Associate PM', 'salary_range': [10, 18], 'experience': 0},
                {'title': 'Product Manager', 'salary_range': [18, 35], 'experience': 3},
                {'title': 'Senior PM', 'salary_range': [35, 60], 'experience': 5}
            ],
            'roadmap': {
                'Year 1': 'Learn product management fundamentals, user research',
                'Year 2': 'Work on real products, learn analytics',
                'Year 3': 'Lead product launches, mentor juniors'
            },
            'certifications': ['Certified Scrum Product Owner', 'Pragmatic Marketing'],
            'avg_salary': {'fresher': [12, 20], '3-5_years': [25, 45], '5+_years': [45, 80]}
        },
        
        'Digital Marketing Manager': {
            'description': 'Plan and execute digital marketing campaigns',
            'required_skills': ['SEO', 'Social Media Marketing', 'Google Ads', 'Analytics', 'Content Marketing'],
            'colleges': ['MICA Ahmedabad', 'IIMS Pune', 'IMT Ghaziabad'],
            'courses': ['Digital Marketing', 'SEO/SEM', 'Social Media Strategy'],
            'job_profiles': [
                {'title': 'Digital Marketing Executive', 'salary_range': [3, 6], 'experience': 0},
                {'title': 'Digital Marketing Manager', 'salary_range': [8, 15], 'experience': 3},
                {'title': 'Head of Digital', 'salary_range': [20, 40], 'experience': 7}
            ],
            'roadmap': {
                'Year 1': 'Learn SEO, social media, Google Ads basics',
                'Year 2': 'Run campaigns, build portfolio',
                'Year 3': 'Specialize in growth hacking or brand strategy'
            },
            'certifications': ['Google Ads Certification', 'HubSpot Content Marketing'],
            'avg_salary': {'fresher': [4, 8], '3-5_years': [10, 20], '5+_years': [20, 45]}
        },
        
        'Mechanical Engineer': {
            'description': 'Design and develop mechanical systems',
            'required_skills': ['CAD', 'Thermodynamics', 'Mechanics', 'Manufacturing', 'Materials Science'],
            'colleges': ['IIT Madras', 'IIT Delhi', 'NIT Trichy', 'BITS Pilani'],
            'courses': ['Machine Design', 'Thermodynamics', 'CAD/CAM'],
            'job_profiles': [
                {'title': 'Design Engineer', 'salary_range': [4, 8], 'experience': 0},
                {'title': 'Senior Engineer', 'salary_range': [8, 15], 'experience': 3},
                {'title': 'Engineering Manager', 'salary_range': [15, 30], 'experience': 7}
            ],
            'roadmap': {
                'Year 1': 'B.Tech Mechanical, learn CAD software',
                'Year 2-3': 'Internships, projects in automotive/manufacturing',
                'Year 4': 'Campus placements or higher studies (M.Tech)'
            },
            'certifications': ['AutoCAD Certified Professional', 'SolidWorks Certification'],
            'avg_salary': {'fresher': [5, 10], '3-5_years': [10, 20], '5+_years': [20, 40]}
        },
        
        'Civil Engineer': {
            'description': 'Design and oversee construction projects',
            'required_skills': ['Structural Design', 'AutoCAD', 'Project Management', 'Surveying'],
            'colleges': ['IIT Bombay', 'IIT Kharagpur', 'NIT Surathkal'],
            'courses': ['Structural Analysis', 'Construction Management', 'Geotechnical Engineering'],
            'job_profiles': [
                {'title': 'Site Engineer', 'salary_range': [3, 6], 'experience': 0},
                {'title': 'Structural Engineer', 'salary_range': [6, 12], 'experience': 3},
                {'title': 'Project Manager', 'salary_range': [15, 30], 'experience': 7}
            ],
            'roadmap': {
                'Year 1': 'B.Tech Civil, AutoCAD training',
                'Year 2-3': 'Site visits, internships',
                'Year 4': 'Prepare for GATE or campus placements'
            },
            'certifications': ['PMP', 'AutoCAD Civil 3D'],
            'avg_salary': {'fresher': [4, 8], '3-5_years': [8, 18], '5+_years': [18, 35]}
        },
        
        'Graphic Designer': {
            'description': 'Create visual content for brands and media',
            'required_skills': ['Adobe Photoshop', 'Illustrator', 'UI/UX Design', 'Typography', 'Branding'],
            'colleges': ['NID Ahmedabad', 'Pearl Academy', 'Srishti School of Design'],
            'courses': ['Graphic Design', 'UI/UX Design', 'Branding'],
            'job_profiles': [
                {'title': 'Junior Designer', 'salary_range': [3, 5], 'experience': 0},
                {'title': 'Graphic Designer', 'salary_range': [5, 10], 'experience': 2},
                {'title': 'Creative Director', 'salary_range': [15, 35], 'experience': 7}
            ],
            'roadmap': {
                'Year 1': 'Learn Adobe Creative Suite, build portfolio',
                'Year 2': 'Freelance projects, internships',
                'Year 3': 'Specialize in branding or UI/UX'
            },
            'certifications': ['Adobe Certified Professional', 'Google UX Design'],
            'avg_salary': {'fresher': [3, 6], '3-5_years': [8, 15], '5+_years': [15, 30]}
        },
        
        'Doctor': {
            'description': 'Diagnose and treat patients',
            'required_skills': ['Medical Knowledge', 'Patient Care', 'Diagnosis', 'Surgery'],
            'colleges': ['AIIMS Delhi', 'CMC Vellore', 'JIPMER Puducherry'],
            'courses': ['MBBS', 'MD/MS Specialization'],
            'job_profiles': [
                {'title': 'Junior Resident', 'salary_range': [8, 12], 'experience': 0},
                {'title': 'Medical Officer', 'salary_range': [12, 20], 'experience': 3},
                {'title': 'Consultant', 'salary_range': [30, 100], 'experience': 10}
            ],
            'roadmap': {
                'Year 1-5': 'MBBS degree',
                'Year 6-8': 'MD/MS specialization',
                'Year 9+': 'Practice and super-specialization'
            },
            'certifications': ['Medical Council Registration', 'Board Certification'],
            'avg_salary': {'fresher': [10, 15], '3-5_years': [20, 40], '5+_years': [50, 150]}
        },
        
        'Lawyer': {
            'description': 'Provide legal counsel and representation',
            'required_skills': ['Legal Research', 'Contract Law', 'Litigation', 'Communication'],
            'colleges': ['NLSIU Bangalore', 'NALSAR Hyderabad', 'NLU Delhi'],
            'courses': ['LLB', 'LLM', 'Corporate Law'],
            'job_profiles': [
                {'title': 'Junior Associate', 'salary_range': [5, 10], 'experience': 0},
                {'title': 'Senior Associate', 'salary_range': [10, 25], 'experience': 4},
                {'title': 'Partner', 'salary_range': [40, 150], 'experience': 10}
            ],
            'roadmap': {
                'Year 1-3/5': 'LLB degree',
                'Year 3-5': 'Internships at law firms',
                'Year 5+': 'Bar exam and practice'
            },
            'certifications': ['Bar Council Enrollment', 'LLM'],
            'avg_salary': {'fresher': [6, 12], '3-5_years': [15, 35], '5+_years': [40, 120]}
        },
        
        'Teacher': {
            'description': 'Educate and mentor students',
            'required_skills': ['Subject Knowledge', 'Communication', 'Classroom Management', 'Curriculum Design'],
            'colleges': ['Delhi University', 'NCERT', 'Jamia Millia Islamia'],
            'courses': ['B.Ed', 'M.Ed', 'Subject-specific training'],
            'job_profiles': [
                {'title': 'Primary Teacher', 'salary_range': [3, 6], 'experience': 0},
                {'title': 'Secondary Teacher', 'salary_range': [5, 10], 'experience': 3},
                {'title': 'Principal', 'salary_range': [15, 30], 'experience': 10}
            ],
            'roadmap': {
                'Year 1-3': 'Graduation in subject',
                'Year 4-5': 'B.Ed degree',
                'Year 5+': 'Teaching practice and CTET'
            },
            'certifications': ['CTET', 'TET', 'B.Ed'],
            'avg_salary': {'fresher': [3, 6], '3-5_years': [6, 12], '5+_years': [12, 25]}
        },
        
        'Business Analyst': {
            'description': 'Analyze business processes and recommend improvements',
            'required_skills': ['Data Analysis', 'SQL', 'Business Intelligence', 'Stakeholder Management'],
            'colleges': ['IIM Ahmedabad', 'ISB Hyderabad', 'SP Jain Mumbai'],
            'courses': ['Business Analysis', 'Data Analytics', 'MBA'],
            'job_profiles': [
                {'title': 'Junior BA', 'salary_range': [5, 10], 'experience': 0},
                {'title': 'Business Analyst', 'salary_range': [10, 20], 'experience': 3},
                {'title': 'Senior BA', 'salary_range': [20, 40], 'experience': 7}
            ],
            'roadmap': {
                'Year 1': 'Learn SQL, Excel, Tableau',
                'Year 2': 'Work on projects, get internships',
                'Year 3+': 'Specialize in domain or get MBA'
            },
            'certifications': ['CBAP', 'PMI-PBA', 'Tableau Certification'],
            'avg_salary': {'fresher': [6, 12], '3-5_years': [12, 25], '5+_years': [25, 50]}
        }
        
        }
    
    def _build_learning_resources(self):
        """Build learning resources database mapped to skills"""
        return {
            'Programming (Java/Python/JavaScript)': [
                {
                    'platform': 'Coursera',
                    'course': 'Python for Everybody Specialization',
                    'provider': 'University of Michigan',
                    'duration': '8 months',
                    'level': 'Beginner',
                    'link': 'https://www.coursera.org/specializations/python'
                },
                {
                    'platform': 'Udemy',
                    'course': 'Complete Python Bootcamp',
                    'provider': 'Jose Portilla',
                    'duration': '22 hours',
                    'level': 'Beginner',
                    'link': 'https://www.udemy.com/course/complete-python-bootcamp/'
                }
            ],
            'Data Structures & Algorithms': [
                {
                    'platform': 'Coursera',
                    'course': 'Algorithms Specialization',
                    'provider': 'Stanford University',
                    'duration': '6 months',
                    'level': 'Intermediate',
                    'link': 'https://www.coursera.org/specializations/algorithms'
                },
                {
                    'platform': 'Udemy',
                    'course': 'Master the Coding Interview: Data Structures + Algorithms',
                    'provider': 'Andrei Neagoie',
                    'duration': '19 hours',
                    'level': 'Intermediate',
                    'link': 'https://www.udemy.com/course/master-the-coding-interview-data-structures-algorithms/'
                }
            ],
            'Machine Learning': [
                {
                    'platform': 'Coursera',
                    'course': 'Machine Learning Specialization',
                    'provider': 'Andrew Ng, Stanford',
                    'duration': '3 months',
                    'level': 'Beginner to Intermediate',
                    'link': 'https://www.coursera.org/specializations/machine-learning-introduction'
                },
                {
                    'platform': 'Udemy',
                    'course': 'Machine Learning A-Z',
                    'provider': 'Kirill Eremenko',
                    'duration': '44 hours',
                    'level': 'Beginner',
                    'link': 'https://www.udemy.com/course/machinelearning/'
                }
            ],
            'Financial Modeling': [
                {
                    'platform': 'Coursera',
                    'course': 'Financial Modeling for Business Analysts and Consultants',
                    'provider': 'Corporate Finance Institute',
                    'duration': '4 weeks',
                    'level': 'Intermediate',
                    'link': 'https://www.coursera.org/learn/financial-modeling'
                },
                {
                    'platform': 'Udemy',
                    'course': 'Financial Modeling & Valuation',
                    'provider': '365 Careers',
                    'duration': '20 hours',
                    'level': 'Intermediate',
                    'link': 'https://www.udemy.com/course/corporate-finance/'
                }
            ],
        }
    
    def analyze_skill_gap(self, user_skills: List[str], required_skills: List[str]) -> Dict[str, Any]:
        """Identify missing skills for target career"""
        user_skills_set = {skill.lower().strip() for skill in user_skills}
        required_skills_set = {skill.lower().strip() for skill in required_skills}
        
        missing_skills = required_skills_set - user_skills_set
        matching_skills = required_skills_set & user_skills_set
        
        if len(required_skills_set) > 0:
            completion_percentage = (len(matching_skills) / len(required_skills_set)) * 100
        else:
            completion_percentage = 0
        
        skill_priority = self._categorize_skill_priority(list(missing_skills), required_skills)
        
        return {
            'missing_skills': list(missing_skills),
            'matching_skills': list(matching_skills),
            'completion_percentage': round(completion_percentage, 2),
            'skill_priority': skill_priority,
            'total_required': len(required_skills_set),
            'total_acquired': len(matching_skills),
            'status': self._get_readiness_status(completion_percentage)
        }
    
    def _categorize_skill_priority(self, missing_skills: List[str], all_required_skills: List[str]) -> Dict[str, List[str]]:
        """Categorize missing skills by priority level"""
        priority = {
            'critical': [],
            'important': [],
            'nice_to_have': []
        }
        
        for skill in missing_skills:
            skill_lower = skill.lower()
            original_skills_lower = [s.lower() for s in all_required_skills]
            if skill_lower in original_skills_lower:
                position = original_skills_lower.index(skill_lower)
                if position < len(all_required_skills) * 0.3:
                    priority['critical'].append(skill)
                elif position < len(all_required_skills) * 0.7:
                    priority['important'].append(skill)
                else:
                    priority['nice_to_have'].append(skill)
            else:
                priority['important'].append(skill)
        
        return priority
    
    def _get_readiness_status(self, completion_percentage: float) -> str:
        """Determine career readiness based on skill completion"""
        if completion_percentage >= 80:
            return "Ready - You have most required skills"
        elif completion_percentage >= 60:
            return "Almost Ready - Focus on key missing skills"
        elif completion_percentage >= 40:
            return "Developing - Continue building foundational skills"
        elif completion_percentage >= 20:
            return "Early Stage - Significant skill development needed"
        else:
            return "Beginner - Start with fundamental skills"
    
    def get_learning_resources(self, missing_skills: List[str]) -> Dict[str, List[Dict]]:
        """Get course recommendations for missing skills"""
        recommendations = {}
        
        for skill in missing_skills:
            if skill in self.learning_resources:
                recommendations[skill] = self.learning_resources[skill]
            else:
                related_resources = self._find_related_resources(skill)
                if related_resources:
                    recommendations[skill] = related_resources
        
        return recommendations
    
    def _find_related_resources(self, skill: str) -> List[Dict]:
        """Find resources for skills with similar keywords"""
        skill_lower = skill.lower()
        related = []
        
        for resource_skill, resources in self.learning_resources.items():
            if any(keyword in resource_skill.lower() for keyword in skill_lower.split()):
                related.extend(resources[:2])
        
        return related[:3] if related else []
    
    def predict_career(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced prediction with skill gap analysis and learning resources"""
        try:
            # Extract optional fields
            current_skills = user_data.get('current_skills', [])
            experience_years = user_data.get('experience_years', 0)
            
            # Create DataFrame with ONLY prediction features
            prediction_features = {
                'age': user_data['age'],
                'gender': user_data['gender'],
                'location': user_data['location'],
                'education_level': user_data['education_level'],
                'stream': user_data['stream'],
                'academic_percentage': user_data['academic_percentage'],
                'career_interest': user_data['career_interest'],
                'learning_style': user_data['learning_style'],
                'future_goal': user_data['future_goal'],
                'tech_domain': user_data['tech_domain']
            }
            
            df_input = pd.DataFrame([prediction_features])
            
            # Encode categorical features
            categorical_cols = ['gender', 'location', 'education_level', 'stream', 
                               'career_interest', 'learning_style', 'future_goal', 'tech_domain']
            
            for col in categorical_cols:
                if col in df_input.columns:
                    df_input[col] = self.label_encoders[col].transform(df_input[col])
            
            # Scale numerical features
            numerical_cols = ['age', 'academic_percentage']
            df_input[numerical_cols] = self.scaler.transform(df_input[numerical_cols])
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(df_input)[0]
            
            # Get top 3 predictions
            top_3_indices = np.argsort(probabilities)[-3:][::-1]
            top_3_careers = self.target_encoder.inverse_transform(top_3_indices)
            top_3_probs = probabilities[top_3_indices]
            
            # Build recommendations
            recommendations = []
            for career, prob in zip(top_3_careers, top_3_probs):
                if career in self.career_database:
                    career_info = self.career_database[career].copy()
                    career_info['career_name'] = str(career)
                    career_info['match_score'] = float(round(prob * 100, 2))
                    
                    if current_skills and len(current_skills) > 0:
                        required_skills = career_info.get('required_skills', [])
                        skill_gap = self.analyze_skill_gap(current_skills, required_skills)
                        career_info['skill_gap_analysis'] = skill_gap
                        
                        learning_resources = self.get_learning_resources(skill_gap['missing_skills'])
                        career_info['recommended_courses'] = learning_resources
                        
                        predicted_salary = self.predict_salary(
                            career, 
                            len(skill_gap['matching_skills']),
                            experience_years
                        )
                        career_info['predicted_salary'] = predicted_salary
                    
                    recommendations.append(career_info)
            
            return {
                'top_recommendation': recommendations[0] if recommendations else None,
                'alternative_careers': recommendations[1:] if len(recommendations) > 1 else [],
                'all_predictions': [
                    {'career': str(c), 'confidence': float(round(p * 100, 2))} 
                    for c, p in zip(top_3_careers, top_3_probs)
                ]
            }
        
        except Exception as e:
            print(f"Error in predict_career: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def predict_salary(self, career_name: str, skills_count: int, experience_years: float) -> Dict[str, Any]:
        """Predict expected salary based on career, skills, and experience"""
        if career_name not in self.career_database:
            return {
                'currency': 'INR (Lakhs per annum)',
                'predicted_range': [0, 0],
                'base_range': [0, 0],
                'skill_adjustment': '0%',
                'factors': {
                    'skills_matched': f"{skills_count}/0",
                    'experience': f"{experience_years} years"
                },
                'note': 'Career not found in database'
            }
        
        try:
            base_salary = self.career_database[career_name]['avg_salary']
            
            if experience_years < 1:
                salary_range = base_salary.get('fresher', [0, 0])
            elif experience_years < 5:
                salary_range = base_salary.get('3-5_years', [0, 0])
            else:
                salary_range = base_salary.get('5+_years', [0, 0])
            
            required_skills_count = len(self.career_database[career_name].get('required_skills', []))
            if required_skills_count > 0:
                skill_percentage = skills_count / required_skills_count
                skill_multiplier = 0.7 + (skill_percentage * 0.3)
            else:
                skill_multiplier = 0.85
            
            predicted_min = salary_range[0] * skill_multiplier
            predicted_max = salary_range[1] * skill_multiplier
            
            return {
                'currency': 'INR (Lakhs per annum)',
                'predicted_range': [round(predicted_min, 1), round(predicted_max, 1)],
                'base_range': salary_range,
                'skill_adjustment': f"{round((skill_multiplier - 1) * 100, 1)}%",
                'factors': {
                    'skills_matched': f"{skills_count}/{required_skills_count}",
                    'experience': f"{experience_years} years"
                },
                'note': 'Salary estimates based on Indian market averages'
            }
        except Exception as e:
            print(f"Error in predict_salary for {career_name}: {str(e)}")
            return {
                'currency': 'INR (Lakhs per annum)',
                'predicted_range': [0, 0],
                'base_range': [0, 0],
                'skill_adjustment': '0%',
                'factors': {
                    'skills_matched': f"{skills_count}/0",
                    'experience': f"{experience_years} years"
                },
                'note': f'Error predicting salary: {str(e)}'
            }
    
    def generate_personalized_roadmap(self, career_name: str, user_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate personalized learning roadmap"""
        if career_name not in self.career_database:
            return None
        
        base_roadmap = self.career_database[career_name]['roadmap']
        education_level = user_data.get('education_level', '')
        
        if education_level in ['10th Pass', '12th Pass']:
            roadmap = {
                'Immediate': 'Focus on completing higher secondary/graduation in relevant stream',
                'Short Term (1-2 years)': base_roadmap.get('Year 1', ''),
                'Medium Term (3-4 years)': base_roadmap.get('Year 2', ''),
                'Long Term (5+ years)': base_roadmap.get('Year 4', '')
            }
        else:
            roadmap = base_roadmap
        
        return roadmap
