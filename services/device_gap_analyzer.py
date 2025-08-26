from typing import Dict, List
from services.device_harvester import DeviceType, DeviceSpecificHarvester

class DeviceGapAnalyzer:
    def __init__(self):
        self.harvester = DeviceSpecificHarvester()
    
    async def analyze_device_opportunities(self, keyword: str, location: str) -> Dict:
        """Analyze opportunities across different devices"""
        multi_device_data = await self.harvester.fetch_serp_multi_device(keyword, location)
        
        opportunities = {
            "mobile_specific": self._find_mobile_opportunities(multi_device_data),
            "desktop_specific": self._find_desktop_opportunities(multi_device_data),
            "cross_device": self._find_cross_device_opportunities(multi_device_data)
        }
        
        return {
            "keyword": keyword,
            "opportunities": opportunities,
            "priority_score": self._calculate_priority_score(opportunities),
            "recommended_actions": self._generate_device_specific_actions(opportunities)
        }
    
    def _find_mobile_opportunities(self, data: Dict) -> List[Dict]:
        """Find mobile-specific opportunities"""
        opportunities = []
        device_results = data.get('device_results', {})
        
        mobile_data = device_results.get('mobile', {})
        desktop_data = device_results.get('desktop', {})
        
        if isinstance(mobile_data, dict) and isinstance(desktop_data, dict):
            # AMP optimization opportunity
            if mobile_data.get('amp_carousel') and not desktop_data.get('amp_carousel'):
                opportunities.append({
                    "type": "amp_optimization",
                    "priority": "high",
                    "description": "Mobile AMP carousel present - optimize for AMP"
                })
            
            # Local pack differences
            mobile_local = mobile_data.get('local_pack', False)
            desktop_local = desktop_data.get('local_pack', False)
            
            if mobile_local and not desktop_local:
                opportunities.append({
                    "type": "local_seo",
                    "priority": "medium",
                    "description": "Mobile shows local results - optimize Google Business Profile"
                })
        
        return opportunities
    
    def _find_desktop_opportunities(self, data: Dict) -> List[Dict]:
        """Find desktop-specific opportunities"""
        opportunities = []
        device_results = data.get('device_results', {})
        
        desktop_data = device_results.get('desktop', {})
        mobile_data = device_results.get('mobile', {})
        
        if isinstance(desktop_data, dict) and isinstance(mobile_data, dict):
            # More organic results on desktop
            desktop_organic = len(desktop_data.get('organic_results', []))
            mobile_organic = len(mobile_data.get('organic_results', []))
            
            if desktop_organic > mobile_organic:
                opportunities.append({
                    "type": "desktop_content",
                    "priority": "medium",
                    "description": "Desktop shows more organic results - optimize for comprehensive content"
                })
        
        return opportunities
    
    def _find_cross_device_opportunities(self, data: Dict) -> List[Dict]:
        """Find cross-device opportunities"""
        opportunities = []
        device_differences = data.get('device_differences', {})
        
        feature_diffs = device_differences.get('feature_differences', {})
        
        for feature, diff_data in feature_diffs.items():
            if isinstance(diff_data, dict):
                mobile_has = diff_data.get('mobile', False)
                desktop_has = diff_data.get('desktop', False)
                
                if mobile_has != desktop_has:
                    opportunities.append({
                        "type": "cross_device_optimization",
                        "priority": "high",
                        "description": f"{feature} differs between devices - optimize for consistency",
                        "feature": feature,
                        "mobile": mobile_has,
                        "desktop": desktop_has
                    })
        
        return opportunities
    
    def _calculate_priority_score(self, opportunities: Dict) -> float:
        """Calculate overall priority score"""
        total_score = 0
        total_opportunities = 0
        
        priority_weights = {"high": 3, "medium": 2, "low": 1}
        
        for category, opps in opportunities.items():
            for opp in opps:
                priority = opp.get('priority', 'low')
                total_score += priority_weights.get(priority, 1)
                total_opportunities += 1
        
        return total_score / max(total_opportunities, 1)
    
    def _generate_device_specific_actions(self, opportunities: Dict) -> List[Dict]:
        """Generate specific actions for device opportunities"""
        actions = []
        
        for category, opps in opportunities.items():
            for opp in opps:
                opp_type = opp.get('type')
                
                if opp_type == 'amp_optimization':
                    actions.append({
                        "action": "Implement AMP pages for mobile optimization",
                        "priority": opp.get('priority'),
                        "estimated_effort": "medium",
                        "expected_impact": "high"
                    })
                
                elif opp_type == 'local_seo':
                    actions.append({
                        "action": "Optimize Google Business Profile and local citations",
                        "priority": opp.get('priority'),
                        "estimated_effort": "low",
                        "expected_impact": "medium"
                    })
                
                elif opp_type == 'cross_device_optimization':
                    actions.append({
                        "action": f"Ensure {opp.get('feature')} consistency across devices",
                        "priority": opp.get('priority'),
                        "estimated_effort": "medium",
                        "expected_impact": "high"
                    })
        
        return actions