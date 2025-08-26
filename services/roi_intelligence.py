from typing import Dict, List, Optional
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class ROIMetrics:
    investment: float
    projected_revenue: float
    timeframe_months: int
    confidence_level: float
    risk_factors: List[str]

class ROIIntelligenceEngine:
    def __init__(self):
        self.industry_benchmarks = {
            'ecommerce': {'avg_conversion': 0.025, 'avg_order_value': 85},
            'saas': {'avg_conversion': 0.03, 'avg_order_value': 120},
            'local_business': {'avg_conversion': 0.05, 'avg_order_value': 200},
            'content': {'avg_conversion': 0.015, 'avg_order_value': 25}
        }
        
        self.position_ctr_map = {
            1: 0.284, 2: 0.147, 3: 0.103, 4: 0.073, 5: 0.053,
            6: 0.040, 7: 0.031, 8: 0.025, 9: 0.020, 10: 0.016
        }
    
    async def calculate_predictive_roi(self, seo_actions: List[Dict], business_context: Dict) -> Dict:
        """Calculate predictive ROI for SEO actions"""
        
        # Analyze each action's ROI potential
        action_analyses = []
        total_investment = 0
        
        for action in seo_actions:
            action_roi = await self._analyze_action_roi(action, business_context)
            action_analyses.append(action_roi)
            total_investment += action_roi['investment_required']
        
        # Calculate portfolio ROI
        portfolio_roi = self._calculate_portfolio_roi(action_analyses, business_context)
        
        # Risk assessment
        risk_analysis = self._assess_roi_risks(action_analyses, business_context)
        
        # Generate recommendations
        recommendations = self._generate_roi_recommendations(portfolio_roi, risk_analysis)
        
        return {
            'total_investment': total_investment,
            'portfolio_analysis': portfolio_roi,
            'individual_actions': action_analyses,
            'risk_assessment': risk_analysis,
            'recommendations': recommendations,
            'confidence_score': self._calculate_confidence_score(action_analyses),
            'break_even_timeline': self._calculate_break_even(portfolio_roi)
        }
    
    async def _analyze_action_roi(self, action: Dict, business_context: Dict) -> Dict:
        """Analyze ROI for individual SEO action"""
        
        action_type = action.get('type', 'unknown')
        investment = self._estimate_action_investment(action)
        
        # Estimate traffic impact
        traffic_impact = self._estimate_traffic_impact(action, business_context)
        
        # Convert traffic to revenue
        revenue_impact = self._convert_traffic_to_revenue(traffic_impact, business_context)
        
        # Calculate ROI metrics
        roi_percentage = ((revenue_impact['annual_revenue'] - investment) / investment * 100) if investment > 0 else 0
        
        return {
            'action_id': action.get('id', 'unknown'),
            'action_type': action_type,
            'investment_required': investment,
            'traffic_impact': traffic_impact,
            'revenue_impact': revenue_impact,
            'roi_percentage': roi_percentage,
            'payback_period_months': investment / revenue_impact['monthly_revenue'] if revenue_impact['monthly_revenue'] > 0 else 999,
            'risk_level': self._assess_action_risk(action),
            'implementation_timeline': action.get('timeline_weeks', 4)
        }
    
    def _estimate_action_investment(self, action: Dict) -> float:
        """Estimate investment required for action"""
        
        base_costs = {
            'content_optimization': 2000,
            'technical_seo': 3500,
            'link_building': 5000,
            'schema_markup': 1500,
            'local_seo': 2500,
            'featured_snippet': 1800,
            'video_content': 4000
        }
        
        action_type = action.get('type', 'content_optimization')
        base_cost = base_costs.get(action_type, 2000)
        
        # Adjust for complexity
        complexity_multiplier = {
            'low': 0.7,
            'medium': 1.0,
            'high': 1.5,
            'enterprise': 2.0
        }
        
        complexity = action.get('complexity', 'medium')
        final_cost = base_cost * complexity_multiplier.get(complexity, 1.0)
        
        return final_cost
    
    def _estimate_traffic_impact(self, action: Dict, business_context: Dict) -> Dict:
        """Estimate traffic impact of SEO action"""
        
        current_position = business_context.get('current_ranking', 10)
        target_position = action.get('target_position', max(1, current_position - 3))
        
        current_monthly_searches = business_context.get('monthly_search_volume', 5000)
        
        # Calculate CTR improvement
        current_ctr = self.position_ctr_map.get(current_position, 0.01)
        target_ctr = self.position_ctr_map.get(target_position, 0.01)
        
        ctr_improvement = target_ctr - current_ctr
        additional_monthly_traffic = current_monthly_searches * ctr_improvement
        
        # Apply action-specific multipliers
        action_multipliers = {
            'featured_snippet': 1.8,  # Featured snippets get extra traffic
            'video_content': 1.3,
            'local_seo': 1.4,
            'content_optimization': 1.1
        }
        
        action_type = action.get('type', 'content_optimization')
        multiplier = action_multipliers.get(action_type, 1.0)
        
        final_additional_traffic = additional_monthly_traffic * multiplier
        
        return {
            'current_monthly_traffic': int(current_monthly_searches * current_ctr),
            'projected_monthly_traffic': int(current_monthly_searches * target_ctr * multiplier),
            'additional_monthly_traffic': int(final_additional_traffic),
            'traffic_increase_percentage': (final_additional_traffic / max(current_monthly_searches * current_ctr, 1)) * 100,
            'position_improvement': current_position - target_position
        }
    
    def _convert_traffic_to_revenue(self, traffic_impact: Dict, business_context: Dict) -> Dict:
        """Convert traffic impact to revenue projections"""
        
        industry = business_context.get('industry', 'ecommerce')
        benchmarks = self.industry_benchmarks.get(industry, self.industry_benchmarks['ecommerce'])
        
        # Use custom conversion data if available, otherwise use benchmarks
        conversion_rate = business_context.get('conversion_rate', benchmarks['avg_conversion'])
        avg_order_value = business_context.get('avg_order_value', benchmarks['avg_order_value'])
        
        additional_monthly_traffic = traffic_impact['additional_monthly_traffic']
        
        # Calculate revenue
        monthly_conversions = additional_monthly_traffic * conversion_rate
        monthly_revenue = monthly_conversions * avg_order_value
        annual_revenue = monthly_revenue * 12
        
        return {
            'monthly_conversions': monthly_conversions,
            'monthly_revenue': monthly_revenue,
            'annual_revenue': annual_revenue,
            'conversion_rate_used': conversion_rate,
            'avg_order_value_used': avg_order_value
        }
    
    def _calculate_portfolio_roi(self, action_analyses: List[Dict], business_context: Dict) -> Dict:
        """Calculate portfolio-level ROI"""
        
        total_investment = sum(action['investment_required'] for action in action_analyses)
        total_annual_revenue = sum(action['revenue_impact']['annual_revenue'] for action in action_analyses)
        
        # Account for diminishing returns when multiple actions target same keywords
        overlap_factor = self._calculate_overlap_factor(action_analyses)
        adjusted_annual_revenue = total_annual_revenue * overlap_factor
        
        portfolio_roi = ((adjusted_annual_revenue - total_investment) / total_investment * 100) if total_investment > 0 else 0
        
        return {
            'total_investment': total_investment,
            'gross_annual_revenue': total_annual_revenue,
            'adjusted_annual_revenue': adjusted_annual_revenue,
            'overlap_factor': overlap_factor,
            'portfolio_roi_percentage': portfolio_roi,
            'net_annual_profit': adjusted_annual_revenue - total_investment,
            'average_monthly_profit': (adjusted_annual_revenue - total_investment) / 12
        }
    
    def _assess_roi_risks(self, action_analyses: List[Dict], business_context: Dict) -> Dict:
        """Assess risks that could impact ROI"""
        
        risk_factors = []
        risk_score = 0.0
        
        # Market competition risk
        competition_level = business_context.get('competition_level', 'medium')
        if competition_level == 'high':
            risk_factors.append("High market competition may reduce effectiveness")
            risk_score += 0.3
        
        # Algorithm change risk
        if any(action['action_type'] in ['featured_snippet', 'schema_markup'] for action in action_analyses):
            risk_factors.append("SERP feature changes could impact results")
            risk_score += 0.2
        
        # Investment concentration risk
        max_investment = max(action['investment_required'] for action in action_analyses)
        total_investment = sum(action['investment_required'] for action in action_analyses)
        
        if max_investment / total_investment > 0.6:
            risk_factors.append("High investment concentration in single action")
            risk_score += 0.25
        
        # Timeline risk
        long_timeline_actions = [a for a in action_analyses if a['implementation_timeline'] > 12]
        if len(long_timeline_actions) > len(action_analyses) * 0.5:
            risk_factors.append("Extended implementation timeline increases uncertainty")
            risk_score += 0.15
        
        risk_level = 'low' if risk_score < 0.3 else 'medium' if risk_score < 0.6 else 'high'
        
        return {
            'overall_risk_score': min(risk_score, 1.0),
            'risk_level': risk_level,
            'identified_risks': risk_factors,
            'mitigation_strategies': self._generate_risk_mitigation_strategies(risk_factors)
        }
    
    def _generate_roi_recommendations(self, portfolio_roi: Dict, risk_analysis: Dict) -> List[Dict]:
        """Generate ROI optimization recommendations"""
        
        recommendations = []
        
        roi_percentage = portfolio_roi['portfolio_roi_percentage']
        
        if roi_percentage > 200:
            recommendations.append({
                'type': 'aggressive_expansion',
                'priority': 'high',
                'description': 'Excellent ROI potential - consider increasing investment',
                'action': 'Scale successful actions and add complementary strategies'
            })
        
        elif roi_percentage > 100:
            recommendations.append({
                'type': 'steady_growth',
                'priority': 'medium',
                'description': 'Good ROI potential - proceed with current plan',
                'action': 'Execute planned actions with regular monitoring'
            })
        
        elif roi_percentage > 50:
            recommendations.append({
                'type': 'optimization_needed',
                'priority': 'medium',
                'description': 'Moderate ROI - optimize for better returns',
                'action': 'Focus on highest-impact, lowest-cost actions first'
            })
        
        else:
            recommendations.append({
                'type': 'strategy_revision',
                'priority': 'high',
                'description': 'Low ROI potential - revise strategy',
                'action': 'Reconsider action selection and target different opportunities'
            })
        
        # Risk-based recommendations
        if risk_analysis['risk_level'] == 'high':
            recommendations.append({
                'type': 'risk_mitigation',
                'priority': 'high',
                'description': 'High risk detected - implement safeguards',
                'action': 'Diversify actions and implement phased approach'
            })
        
        return recommendations
    
    def _calculate_overlap_factor(self, action_analyses: List[Dict]) -> float:
        """Calculate overlap factor for actions targeting similar outcomes"""
        
        # Simplified overlap calculation
        # In reality, this would analyze keyword overlap, SERP feature competition, etc.
        
        num_actions = len(action_analyses)
        
        if num_actions <= 1:
            return 1.0
        elif num_actions <= 3:
            return 0.9  # Minimal overlap
        elif num_actions <= 5:
            return 0.8  # Some overlap
        else:
            return 0.7  # Significant overlap
    
    def _assess_action_risk(self, action: Dict) -> str:
        """Assess risk level for individual action"""
        
        risk_factors = 0
        
        # High-cost actions are riskier
        if action.get('investment_required', 0) > 5000:
            risk_factors += 1
        
        # Long timeline actions are riskier
        if action.get('timeline_weeks', 4) > 16:
            risk_factors += 1
        
        # Certain action types are inherently riskier
        high_risk_types = ['link_building', 'technical_seo']
        if action.get('type') in high_risk_types:
            risk_factors += 1
        
        if risk_factors >= 2:
            return 'high'
        elif risk_factors == 1:
            return 'medium'
        else:
            return 'low'
    
    def _generate_risk_mitigation_strategies(self, risk_factors: List[str]) -> List[str]:
        """Generate risk mitigation strategies"""
        
        strategies = []
        
        for risk in risk_factors:
            if 'competition' in risk.lower():
                strategies.append("Monitor competitor activities and adjust strategy accordingly")
            
            elif 'algorithm' in risk.lower():
                strategies.append("Diversify across multiple SERP features and ranking factors")
            
            elif 'concentration' in risk.lower():
                strategies.append("Distribute investment across multiple lower-risk actions")
            
            elif 'timeline' in risk.lower():
                strategies.append("Implement phased approach with early wins to validate strategy")
        
        return strategies
    
    def _calculate_confidence_score(self, action_analyses: List[Dict]) -> float:
        """Calculate confidence score for ROI projections"""
        
        # Factors that increase confidence
        confidence_factors = []
        
        # Historical data availability
        confidence_factors.append(0.7)  # Assume moderate historical data
        
        # Action diversity
        action_types = set(action['action_type'] for action in action_analyses)
        diversity_score = min(len(action_types) / 5.0, 1.0)
        confidence_factors.append(diversity_score)
        
        # Risk level (lower risk = higher confidence)
        avg_risk_score = np.mean([
            {'low': 0.9, 'medium': 0.7, 'high': 0.4}[action['risk_level']]
            for action in action_analyses
        ])
        confidence_factors.append(avg_risk_score)
        
        return np.mean(confidence_factors)
    
    def _calculate_break_even(self, portfolio_roi: Dict) -> Dict:
        """Calculate break-even timeline"""
        
        monthly_profit = portfolio_roi['average_monthly_profit']
        total_investment = portfolio_roi['total_investment']
        
        if monthly_profit <= 0:
            return {
                'break_even_months': 999,
                'break_even_achievable': False,
                'message': 'Break-even not achievable with current projections'
            }
        
        break_even_months = total_investment / monthly_profit
        
        return {
            'break_even_months': round(break_even_months, 1),
            'break_even_achievable': True,
            'break_even_date': (datetime.now() + timedelta(days=break_even_months * 30)).strftime('%Y-%m-%d'),
            'message': f'Break-even expected in {round(break_even_months, 1)} months'
        }