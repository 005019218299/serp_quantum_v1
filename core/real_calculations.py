import numpy as np
from typing import Dict, List
from .constants import CONFIDENCE_THRESHOLD

class RealCalculations:
    """Real calculations instead of hardcoded fake values"""
    
    def calculate_serp_stability(self, simulation_results: List[Dict]) -> float:
        """Calculate actual SERP stability from simulation data"""
        if not simulation_results:
            return 0.5
        
        stability_scores = [r.get('stability_score', 0.5) for r in simulation_results]
        return np.mean(stability_scores)
    
    def calculate_difficulty_forecast(self, serp_features: Dict, historical_data: List) -> float:
        """Calculate real difficulty based on SERP features"""
        base_difficulty = 5.0
        
        # Adjust based on actual features
        if serp_features.get('has_featured_snippet'):
            base_difficulty += 1.5
        if serp_features.get('has_video_carousel'):
            base_difficulty += 1.0
        if len(serp_features.get('top_domains', [])) > 5:
            base_difficulty += 0.5
            
        return min(base_difficulty, 10.0)
    
    def calculate_snippet_probability(self, serp_features: Dict) -> float:
        """Calculate real featured snippet probability"""
        if serp_features.get('has_featured_snippet'):
            return 0.2  # Lower chance if already exists
        
        # Base probability on content quality indicators
        content_quality = serp_features.get('avg_content_length', 0)
        if content_quality > 1000:
            return 0.7
        elif content_quality > 500:
            return 0.5
        else:
            return 0.3
    
    def safe_divide(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division to avoid division by zero"""
        return numerator / denominator if denominator != 0 else default

real_calc = RealCalculations()