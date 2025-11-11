# src/roadmap_generator.py

import json
from typing import Dict, List, Any

class RoadmapGenerator:
    """
    Generate structured roadmap data that can be visualized with D3.js/React Flow
    """
    
    def generate_roadmap_graph(self, career_name: str, skill_gap_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate graph structure for career roadmap visualization
        
        Returns JSON structure compatible with D3.js and React Flow
        """
        nodes = []
        edges = []
        node_id = 0
        
        # Start node
        nodes.append({
            'id': str(node_id),
            'type': 'start',
            'data': {'label': 'Current Position', 'status': 'active'},
            'position': {'x': 250, 'y': 0}
        })
        start_id = node_id
        node_id += 1
        
        # Skill acquisition phases
        skill_priority = skill_gap_data.get('skill_priority', {})
        
        # Phase 1: Critical Skills
        if skill_priority.get('critical'):
            phase1_id = node_id
            nodes.append({
                'id': str(node_id),
                'type': 'phase',
                'data': {
                    'label': 'Phase 1: Critical Skills',
                    'skills': skill_priority['critical'],
                    'duration': '3-6 months',
                    'status': 'pending'
                },
                'position': {'x': 250, 'y': 100}
            })
            edges.append({
                'id': f'e{start_id}-{node_id}',
                'source': str(start_id),
                'target': str(node_id),
                'animated': True
            })
            node_id += 1
            prev_id = phase1_id
        else:
            prev_id = start_id
        
        # Phase 2: Important Skills
        if skill_priority.get('important'):
            phase2_id = node_id
            nodes.append({
                'id': str(node_id),
                'type': 'phase',
                'data': {
                    'label': 'Phase 2: Important Skills',
                    'skills': skill_priority['important'],
                    'duration': '6-12 months',
                    'status': 'pending'
                },
                'position': {'x': 250, 'y': 220}
            })
            edges.append({
                'id': f'e{prev_id}-{node_id}',
                'source': str(prev_id),
                'target': str(node_id)
            })
            node_id += 1
            prev_id = phase2_id
        
        # Phase 3: Advanced/Nice-to-have Skills
        if skill_priority.get('nice_to_have'):
            phase3_id = node_id
            nodes.append({
                'id': str(node_id),
                'type': 'phase',
                'data': {
                    'label': 'Phase 3: Advanced Skills',
                    'skills': skill_priority['nice_to_have'],
                    'duration': '12-18 months',
                    'status': 'pending'
                },
                'position': {'x': 250, 'y': 340}
            })
            edges.append({
                'id': f'e{prev_id}-{node_id}',
                'source': str(prev_id),
                'target': str(node_id)
            })
            node_id += 1
            prev_id = phase3_id
        
        # Milestone: Project/Portfolio
        project_id = node_id
        nodes.append({
            'id': str(node_id),
            'type': 'milestone',
            'data': {
                'label': 'Build Portfolio Projects',
                'description': 'Apply learned skills in real projects',
                'duration': '3-6 months'
            },
            'position': {'x': 250, 'y': 460}
        })
        edges.append({
            'id': f'e{prev_id}-{node_id}',
            'source': str(prev_id),
            'target': str(node_id)
        })
        node_id += 1
        
        # Milestone: Certifications
        cert_id = node_id
        nodes.append({
            'id': str(node_id),
            'type': 'milestone',
            'data': {
                'label': 'Earn Certifications',
                'description': 'Industry-recognized certifications',
                'duration': '2-4 months'
            },
            'position': {'x': 450, 'y': 460}
        })
        edges.append({
            'id': f'e{prev_id}-{node_id}',
            'source': str(prev_id),
            'target': str(node_id),
            'type': 'smoothstep'
        })
        node_id += 1
        
        # Goal: Job Ready
        goal_id = node_id
        nodes.append({
            'id': str(node_id),
            'type': 'goal',
            'data': {
                'label': f'{career_name} Ready',
                'description': 'Job applications & interviews',
                'status': 'goal'
            },
            'position': {'x': 250, 'y': 580}
        })
        edges.append({
            'id': f'e{project_id}-{node_id}',
            'source': str(project_id),
            'target': str(goal_id)
        })
        edges.append({
            'id': f'e{cert_id}-{node_id}',
            'source': str(cert_id),
            'target': str(goal_id),
            'type': 'smoothstep'
        })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'career': career_name,
                'total_phases': 3,
                'estimated_duration': '18-24 months',
                'completion_percentage': skill_gap_data.get('completion_percentage', 0)
            }
        }
    
    def generate_timeline_data(self, career_name: str, roadmap: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Generate timeline data for horizontal timeline visualization
        """
        timeline = []
        
        for i, (phase, description) in enumerate(roadmap.items()):
            timeline.append({
                'id': i,
                'phase': phase,
                'description': description,
                'order': i,
                'status': 'completed' if i == 0 else 'upcoming'
            })
        
        return timeline
    
    def save_roadmap_json(self, roadmap_data: Dict[str, Any], filename: str = 'roadmap.json'):
        """Save roadmap as JSON for frontend consumption"""
        with open(f'data/roadmaps/{filename}', 'w') as f:
            json.dump(roadmap_data, f, indent=2)
        print(f"Roadmap saved to data/roadmaps/{filename}")
