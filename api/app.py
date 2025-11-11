# api/app.py (Complete with all endpoints)

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.recommendation_engine import RecommendationEngine
from src.roadmap_generator import RoadmapGenerator

app = Flask(__name__)
CORS(app)

# Load models at startup
models_path = os.path.join(parent_dir, 'models')
try:
    recommendation_engine = RecommendationEngine(model_path=models_path + '/')
    roadmap_generator = RoadmapGenerator()
    print("✓ Models loaded successfully")
except Exception as e:
    print(f"✗ Error loading models: {e}")
    print("Please train the models first by running: python train_complete_system.py")


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Enhanced AI API is running'}), 200


@app.route('/api/careers', methods=['GET'])
def get_all_careers():
    """
    Get list of all available career paths
    """
    try:
        careers = list(recommendation_engine.career_database.keys())
        return jsonify({
            'success': True,
            'careers': careers,
            'total': len(careers)
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/career/<career_name>', methods=['GET'])
def get_career_details(career_name):
    """
    Get detailed information about a specific career
    
    Example: GET /api/career/Software Engineer
    """
    try:
        if career_name in recommendation_engine.career_database:
            career_data = recommendation_engine.career_database[career_name].copy()
            return jsonify({
                'success': True,
                'data': career_data,
                'career_name': career_name
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': f'Career "{career_name}" not found',
                'available_careers': list(recommendation_engine.career_database.keys())
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/predict', methods=['POST'])
def predict_career():
    """
    Enhanced prediction with skill gap analysis
    
    Expected JSON:
    {
        "age": 22,
        "gender": "Male",
        "location": "Urban",
        "education_level": "Undergraduate",
        "stream": "Science",
        "academic_percentage": 85.5,
        "career_interest": "Engineering",
        "learning_style": "Hybrid",
        "future_goal": "Job-Oriented Training",
        "tech_domain": "AI/ML",
        "current_skills": ["Python", "DSA", "Git"],
        "experience_years": 0
    }
    """
    try:
        user_data = request.get_json()
        
        # Get recommendations with skill gap analysis
        recommendations = recommendation_engine.predict_career(user_data)
        
        # Generate roadmap visualization data
        if recommendations['top_recommendation']:
            career_name = recommendations['top_recommendation']['career_name']
            
            # Generate graph structure for visualization
            if 'skill_gap_analysis' in recommendations['top_recommendation']:
                skill_gap = recommendations['top_recommendation']['skill_gap_analysis']
                roadmap_graph = roadmap_generator.generate_roadmap_graph(career_name, skill_gap)
                recommendations['top_recommendation']['roadmap_visualization'] = roadmap_graph
        
        return jsonify({
            'success': True,
            'data': recommendations
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/skill-gap', methods=['POST'])
def analyze_skills():
    """
    Dedicated endpoint for skill gap analysis
    
    Expected JSON:
    {
        "career_name": "Software Engineer",
        "current_skills": ["Python", "Git", "HTML"]
    }
    """
    try:
        data = request.get_json()
        career_name = data.get('career_name')
        current_skills = data.get('current_skills', [])
        
        if career_name not in recommendation_engine.career_database:
            return jsonify({
                'success': False,
                'error': 'Career not found'
            }), 404
        
        required_skills = recommendation_engine.career_database[career_name].get('required_skills', [])
        skill_gap = recommendation_engine.analyze_skill_gap(current_skills, required_skills)
        
        # Get learning resources
        learning_resources = recommendation_engine.get_learning_resources(skill_gap['missing_skills'])
        
        return jsonify({
            'success': True,
            'data': {
                'skill_gap_analysis': skill_gap,
                'learning_resources': learning_resources
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/learning-resources', methods=['POST'])
def get_resources():
    """
    Get learning resources for specific skills
    
    Expected JSON:
    {
        "skills": ["Machine Learning", "Python", "Data Structures"]
    }
    """
    try:
        data = request.get_json()
        skills = data.get('skills', [])
        
        resources = recommendation_engine.get_learning_resources(skills)
        
        return jsonify({
            'success': True,
            'data': resources
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/salary-prediction', methods=['POST'])
def predict_salary():
    """
    Predict salary based on career and skills
    
    Expected JSON:
    {
        "career_name": "Software Engineer",
        "skills_count": 8,
        "experience_years": 2
    }
    """
    try:
        data = request.get_json()
        career_name = data.get('career_name')
        skills_count = data.get('skills_count', 0)
        experience_years = data.get('experience_years', 0)
        
        salary_prediction = recommendation_engine.predict_salary(
            career_name, skills_count, experience_years
        )
        
        return jsonify({
            'success': True,
            'data': salary_prediction
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/roadmap-visualization', methods=['POST'])
def get_roadmap_visualization():
    """
    Generate roadmap visualization data
    
    Expected JSON:
    {
        "career_name": "Data Scientist",
        "current_skills": ["Python", "Statistics"]
    }
    """
    try:
        data = request.get_json()
        career_name = data.get('career_name')
        current_skills = data.get('current_skills', [])
        
        if career_name not in recommendation_engine.career_database:
            return jsonify({'success': False, 'error': 'Career not found'}), 404
        
        required_skills = recommendation_engine.career_database[career_name].get('required_skills', [])
        skill_gap = recommendation_engine.analyze_skill_gap(current_skills, required_skills)
        
        roadmap_graph = roadmap_generator.generate_roadmap_graph(career_name, skill_gap)
        
        return jsonify({
            'success': True,
            'data': roadmap_graph
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("AI Career Recommendation API Server")
    print("="*60)
    print(f"API running on: http://localhost:5000")
    print(f"Models path: {models_path}")
    print("\nAvailable Endpoints:")
    print("  GET  /api/health")
    print("  GET  /api/careers")
    print("  GET  /api/career/<career_name>")
    print("  POST /api/predict")
    print("  POST /api/skill-gap")
    print("  POST /api/learning-resources")
    print("  POST /api/salary-prediction")
    print("  POST /api/roadmap-visualization")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
