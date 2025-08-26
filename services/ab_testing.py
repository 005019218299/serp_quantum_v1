from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
import asyncio
from dataclasses import dataclass

@dataclass
class ABTestConfig:
    test_id: str
    name: str
    control_strategy: Dict
    variant_strategy: Dict
    success_metric: str
    duration_days: int
    confidence_level: float = 0.95

class ABTestManager:
    def __init__(self):
        self.active_tests = {}
        self.test_results = {}
    
    async def create_ab_test(self, config: ABTestConfig) -> Dict:
        """Create new A/B test for SERP strategies"""
        test_data = {
            'config': config,
            'start_date': datetime.utcnow(),
            'end_date': datetime.utcnow() + timedelta(days=config.duration_days),
            'control_results': [],
            'variant_results': [],
            'status': 'running'
        }
        
        self.active_tests[config.test_id] = test_data
        
        return {
            'test_id': config.test_id,
            'status': 'created',
            'start_date': test_data['start_date'].isoformat(),
            'estimated_end_date': test_data['end_date'].isoformat()
        }
    
    async def run_test_iteration(self, test_id: str, keyword: str) -> Dict:
        """Run one iteration of A/B test"""
        if test_id not in self.active_tests:
            return {'error': 'Test not found'}
        
        test_data = self.active_tests[test_id]
        config = test_data['config']
        
        # Randomly assign to control or variant (50/50 split)
        is_variant = np.random.random() > 0.5
        
        if is_variant:
            strategy = config.variant_strategy
            group = 'variant'
        else:
            strategy = config.control_strategy
            group = 'control'
        
        # Simulate strategy execution and measure results
        result = await self._execute_strategy_simulation(strategy, keyword)
        
        # Record result
        result_data = {
            'timestamp': datetime.utcnow(),
            'keyword': keyword,
            'strategy': strategy,
            'metric_value': result[config.success_metric],
            'group': group
        }
        
        if is_variant:
            test_data['variant_results'].append(result_data)
        else:
            test_data['control_results'].append(result_data)
        
        return {
            'test_id': test_id,
            'group': group,
            'result': result_data,
            'total_samples': len(test_data['control_results']) + len(test_data['variant_results'])
        }
    
    def analyze_test_results(self, test_id: str) -> Dict:
        """Analyze A/B test results for statistical significance"""
        if test_id not in self.active_tests:
            return {'error': 'Test not found'}
        
        test_data = self.active_tests[test_id]
        control_results = test_data['control_results']
        variant_results = test_data['variant_results']
        
        if len(control_results) < 10 or len(variant_results) < 10:
            return {
                'status': 'insufficient_data',
                'control_samples': len(control_results),
                'variant_samples': len(variant_results),
                'message': 'Need at least 10 samples per group'
            }
        
        # Extract metric values
        control_values = [r['metric_value'] for r in control_results]
        variant_values = [r['metric_value'] for r in variant_results]
        
        # Statistical analysis
        control_mean = np.mean(control_values)
        variant_mean = np.mean(variant_values)
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(variant_values, control_values)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(control_values) - 1) * np.var(control_values) + 
                             (len(variant_values) - 1) * np.var(variant_values)) / 
                            (len(control_values) + len(variant_values) - 2))
        
        cohens_d = (variant_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        
        # Determine significance
        config = test_data['config']
        is_significant = p_value < (1 - config.confidence_level)
        
        # Calculate improvement
        improvement = ((variant_mean - control_mean) / control_mean * 100) if control_mean > 0 else 0
        
        result = {
            'test_id': test_id,
            'status': 'completed' if is_significant else 'running',
            'control_mean': round(control_mean, 4),
            'variant_mean': round(variant_mean, 4),
            'improvement_percent': round(improvement, 2),
            'p_value': round(p_value, 4),
            'is_significant': is_significant,
            'confidence_level': config.confidence_level,
            'effect_size': round(cohens_d, 4),
            'sample_sizes': {
                'control': len(control_results),
                'variant': len(variant_results)
            },
            'recommendation': self._generate_recommendation(improvement, is_significant, cohens_d)
        }
        
        if is_significant:
            test_data['status'] = 'completed'
            self.test_results[test_id] = result
        
        return result
    
    def get_test_summary(self, test_id: str) -> Dict:
        """Get test summary and current status"""
        if test_id not in self.active_tests:
            return {'error': 'Test not found'}
        
        test_data = self.active_tests[test_id]
        config = test_data['config']
        
        return {
            'test_id': test_id,
            'name': config.name,
            'status': test_data['status'],
            'start_date': test_data['start_date'].isoformat(),
            'duration_days': config.duration_days,
            'success_metric': config.success_metric,
            'control_samples': len(test_data['control_results']),
            'variant_samples': len(test_data['variant_results']),
            'control_strategy': config.control_strategy,
            'variant_strategy': config.variant_strategy
        }
    
    async def _execute_strategy_simulation(self, strategy: Dict, keyword: str) -> Dict:
        """Simulate strategy execution and return metrics"""
        # This would integrate with actual SERP simulation
        # For now, simulate with random values based on strategy type
        
        base_traffic = 1000
        base_ranking = 5.0
        base_ctr = 0.03
        
        # Apply strategy effects (simplified simulation)
        traffic_multiplier = 1.0
        ranking_improvement = 0.0
        ctr_improvement = 0.0
        
        if strategy.get('type') == 'featured_snippet':
            traffic_multiplier = 1.4
            ranking_improvement = -2.0  # Better ranking (lower number)
            ctr_improvement = 0.02
        elif strategy.get('type') == 'video_carousel':
            traffic_multiplier = 1.2
            ctr_improvement = 0.015
        elif strategy.get('type') == 'people_also_ask':
            traffic_multiplier = 1.1
            ctr_improvement = 0.01
        
        # Add some randomness to simulate real-world variation
        noise_factor = np.random.normal(1.0, 0.1)
        
        return {
            'traffic_increase': int(base_traffic * traffic_multiplier * noise_factor),
            'ranking_position': max(1, base_ranking + ranking_improvement + np.random.normal(0, 0.5)),
            'click_through_rate': min(1.0, base_ctr + ctr_improvement + np.random.normal(0, 0.005)),
            'conversion_rate': max(0, 0.02 + np.random.normal(0, 0.005))
        }
    
    def _generate_recommendation(self, improvement: float, is_significant: bool, effect_size: float) -> str:
        """Generate recommendation based on test results"""
        if not is_significant:
            return "Continue testing - no significant difference detected yet"
        
        if improvement > 10 and effect_size > 0.5:
            return "Strong recommendation: Implement variant strategy"
        elif improvement > 5 and effect_size > 0.2:
            return "Moderate recommendation: Consider implementing variant strategy"
        elif improvement > 0:
            return "Weak positive effect: Monitor longer before deciding"
        else:
            return "Recommendation: Stick with control strategy"