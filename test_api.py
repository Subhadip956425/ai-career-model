import requests
import json

BASE_URL = 'http://localhost:5000/api'

def test_enhanced_prediction():
    """Test prediction with skill gap analysis"""
    print("\n" + "="*60)
    print("TEST 1: Enhanced Career Prediction with Skill Gap")
    print("="*60)
    
    url = f'{BASE_URL}/predict'
    test_data = {
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
        "current_skills": ["Python", "Git", "HTML", "DSA"],
        "experience_years": 0
    }
    
    response = requests.post(url, json=test_data)
    if response.status_code == 200:
        result = response.json()
        print("\n✓ SUCCESS")
        print(json.dumps(result['data']['top_recommendation'], indent=2))
    else:
        print(f"✗ FAILED: {response.status_code}")

def test_skill_gap():
    """Test skill gap analysis"""
    print("\n" + "="*60)
    print("TEST 2: Skill Gap Analysis")
    print("="*60)
    
    url = f'{BASE_URL}/skill-gap'
    test_data = {
        "career_name": "Data Scientist",
        "current_skills": ["Python", "Statistics", "SQL"]
    }
    
    response = requests.post(url, json=test_data)
    if response.status_code == 200:
        result = response.json()
        print("\n✓ SUCCESS")
        print(json.dumps(result['data']['skill_gap_analysis'], indent=2))
    else:
        print(f"✗ FAILED: {response.status_code}")

def test_learning_resources():
    """Test learning resource recommendations"""
    print("\n" + "="*60)
    print("TEST 3: Learning Resource Recommendations")
    print("="*60)
    
    url = f'{BASE_URL}/learning-resources'
    test_data = {
        "skills": ["Machine Learning", "Deep Learning"]
    }
    
    response = requests.post(url, json=test_data)
    if response.status_code == 200:
        result = response.json()
        print("\n✓ SUCCESS")
        print(json.dumps(result['data'], indent=2))
    else:
        print(f"✗ FAILED: {response.status_code}")

def test_salary_prediction():
    """Test salary prediction"""
    print("\n" + "="*60)
    print("TEST 4: Salary Prediction")
    print("="*60)
    
    url = f'{BASE_URL}/salary-prediction'
    test_data = {
        "career_name": "Software Engineer",
        "skills_count": 8,
        "experience_years": 2
    }
    
    response = requests.post(url, json=test_data)
    if response.status_code == 200:
        result = response.json()
        print("\n✓ SUCCESS")
        print(json.dumps(result['data'], indent=2))
    else:
        print(f"✗ FAILED: {response.status_code}")

def test_roadmap_visualization():
    """Test roadmap visualization data generation"""
    print("\n" + "="*60)
    print("TEST 5: Roadmap Visualization Data")
    print("="*60)
    
    url = f'{BASE_URL}/roadmap-visualization'
    test_data = {
        "career_name": "Software Engineer",
        "current_skills": ["Python", "HTML"]
    }
    
    response = requests.post(url, json=test_data)
    if response.status_code == 200:
        result = response.json()
        print("\n✓ SUCCESS")
        print(f"Nodes: {len(result['data']['nodes'])}")
        print(f"Edges: {len(result['data']['edges'])}")
        print(json.dumps(result['data']['metadata'], indent=2))
    else:
        print(f"✗ FAILED: {response.status_code}")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("TESTING ENHANCED AI CAREER RECOMMENDATION API")
    print("="*60)
    print("\nMake sure Flask API is running on http://localhost:5000")
    input("Press Enter to start tests...")
    
    test_enhanced_prediction()
    test_skill_gap()
    test_learning_resources()
    test_salary_prediction()
    test_roadmap_visualization()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)
