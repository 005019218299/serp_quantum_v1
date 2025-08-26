from typing import List, Dict, Optional
import numpy as np
from datetime import datetime, timedelta

class StrategyEngine:
    def __init__(self):
        self.strategy_templates = {
            'featured_snippet': {
                'name': 'Featured Snippet Capture',
                'description': 'Tạo nội dung toàn diện để chiếm vị trí Featured Snippet',
                'base_cost': 'Medium',
                'base_time': '2-4 weeks',
                'base_success_prob': 0.65,
                'impact_multiplier': 4.2
            },
            'video_carousel': {
                'name': 'Video Carousel Domination',
                'description': 'Sản xuất video chất lượng cao để xuất hiện trong Video Carousel',
                'base_cost': 'High',
                'base_time': '3-5 weeks',
                'base_success_prob': 0.55,
                'impact_multiplier': 3.8
            },
            'people_also_ask': {
                'name': 'People Also Ask Optimization',
                'description': 'Tạo nội dung FAQ chi tiết để chiếm People Also Ask',
                'base_cost': 'Low',
                'base_time': '1-2 weeks',
                'base_success_prob': 0.75,
                'impact_multiplier': 2.5
            },
            'local_pack': {
                'name': 'Local Pack Optimization',
                'description': 'Tối ưu Google Business Profile và local SEO',
                'base_cost': 'Low',
                'base_time': '1 week',
                'base_success_prob': 0.85,
                'impact_multiplier': 3.2
            }
        }
    
    def generate_multi_move_strategy(self, keyword: str, current_serp: Dict, 
                                   budget_constraint: str = 'medium',
                                   content_assets: Dict = None) -> List[Dict]:
        """Tạo chiến lược nước đi kép"""
        
        # Phân tích cơ hội từ SERP hiện tại
        opportunities = self._analyze_opportunities(current_serp)
        
        # Tạo các strategic moves dựa trên cơ hội và ngân sách
        possible_moves = self._generate_possible_moves(opportunities, budget_constraint)
        
        # Tối ưu hóa tổ hợp moves
        optimal_strategies = self._optimize_move_combination(possible_moves, budget_constraint)
        
        return optimal_strategies
    
    def _analyze_opportunities(self, serp_state: Dict) -> Dict:
        """Phân tích cơ hội từ trạng thái SERP hiện tại"""
        opportunities = {}
        
        # Featured Snippet opportunity
        if not serp_state.get('has_featured_snippet', 0):
            opportunities['featured_snippet'] = {
                'priority': 'high',
                'difficulty': 'medium',
                'potential_impact': 4.5
            }
        
        # Video Carousel opportunity
        if not serp_state.get('has_video_carousel', 0):
            opportunities['video_carousel'] = {
                'priority': 'medium',
                'difficulty': 'high',
                'potential_impact': 3.8
            }
        
        # People Also Ask opportunity
        paa_count = serp_state.get('paa_questions_count', 0)
        if paa_count < 3:
            opportunities['people_also_ask'] = {
                'priority': 'medium',
                'difficulty': 'low',
                'potential_impact': 2.8
            }
        
        # Local Pack opportunity (nếu có ý định local)
        if not serp_state.get('local_pack_presence', 0):
            opportunities['local_pack'] = {
                'priority': 'low',
                'difficulty': 'low',
                'potential_impact': 3.0
            }
        
        return opportunities
    
    def _generate_possible_moves(self, opportunities: Dict, budget_constraint: str) -> List[Dict]:
        """Tạo các nước đi có thể thực hiện"""
        moves = []
        
        budget_limits = {
            'low': 2,
            'medium': 4,
            'high': 6
        }
        
        max_moves = budget_limits.get(budget_constraint, 3)
        
        # Sắp xếp opportunities theo priority và impact
        sorted_opportunities = sorted(
            opportunities.items(),
            key=lambda x: (
                {'high': 3, 'medium': 2, 'low': 1}[x[1]['priority']] + 
                x[1]['potential_impact']
            ),
            reverse=True
        )
        
        for opp_type, opp_data in sorted_opportunities[:max_moves]:
            if opp_type in self.strategy_templates:
                template = self.strategy_templates[opp_type]
                
                move = {
                    'type': opp_type,
                    'name': template['name'],
                    'description': template['description'],
                    'target': self._get_target_name(opp_type),
                    'cost': template['base_cost'],
                    'time': template['base_time'],
                    'success_probability': self._adjust_success_probability(
                        template['base_success_prob'], 
                        opp_data['difficulty']
                    ),
                    'expected_impact': template['impact_multiplier'] * opp_data['potential_impact'],
                    'implementation_day': self._calculate_implementation_day(opp_type),
                    'detailed_actions': self._generate_detailed_actions(opp_type)
                }
                moves.append(move)
        
        return moves
    
    def _optimize_move_combination(self, possible_moves: List[Dict], budget_constraint: str) -> List[Dict]:
        """Tối ưu hóa tổ hợp các moves"""
        if not possible_moves:
            return []
        
        # Tạo các strategies kết hợp
        strategies = []
        
        # Strategy 1: High-impact combination
        high_impact_moves = [m for m in possible_moves if m['expected_impact'] > 10]
        if high_impact_moves:
            strategies.append({
                'strategy_id': 'HIGH-IMPACT-001',
                'name': 'Chiến lược Tác động Cao',
                'description': 'Tập trung vào các nước đi có tác động lớn nhất',
                'moves': high_impact_moves[:2],
                'composite_success_probability': self._calculate_composite_probability(high_impact_moves[:2]),
                'predicted_traffic_increase': f"+{sum(m['expected_impact'] for m in high_impact_moves[:2]) * 5:.0f}%",
                'total_cost': self._calculate_total_cost(high_impact_moves[:2]),
                'estimated_timeline': self._calculate_timeline(high_impact_moves[:2])
            })
        
        # Strategy 2: Quick wins
        quick_moves = [m for m in possible_moves if 'week' in m['time'] and int(m['time'].split()[0]) <= 2]
        if quick_moves:
            strategies.append({
                'strategy_id': 'QUICK-WIN-002',
                'name': 'Chiến lược Thắng Nhanh',
                'description': 'Tập trung vào các cơ hội có thể thực hiện nhanh chóng',
                'moves': quick_moves,
                'composite_success_probability': self._calculate_composite_probability(quick_moves),
                'predicted_traffic_increase': f"+{sum(m['expected_impact'] for m in quick_moves) * 3:.0f}%",
                'total_cost': self._calculate_total_cost(quick_moves),
                'estimated_timeline': self._calculate_timeline(quick_moves)
            })
        
        # Strategy 3: Balanced approach
        if len(possible_moves) >= 2:
            balanced_moves = possible_moves[:3]  # Top 3 moves
            strategies.append({
                'strategy_id': 'BALANCED-003',
                'name': 'Chiến lược Cân bằng',
                'description': 'Kết hợp nhiều loại nước đi để tối đa hóa coverage',
                'moves': balanced_moves,
                'composite_success_probability': self._calculate_composite_probability(balanced_moves),
                'predicted_traffic_increase': f"+{sum(m['expected_impact'] for m in balanced_moves) * 4:.0f}%",
                'total_cost': self._calculate_total_cost(balanced_moves),
                'estimated_timeline': self._calculate_timeline(balanced_moves)
            })
        
        return strategies
    
    def _adjust_success_probability(self, base_prob: float, difficulty: str) -> float:
        """Điều chỉnh xác suất thành công dựa trên độ khó"""
        difficulty_modifiers = {
            'low': 1.2,
            'medium': 1.0,
            'high': 0.8
        }
        
        adjusted = base_prob * difficulty_modifiers.get(difficulty, 1.0)
        return min(adjusted, 0.95)  # Cap at 95%
    
    def _calculate_composite_probability(self, moves: List[Dict]) -> float:
        """Tính xác suất thành công tổng hợp"""
        if not moves:
            return 0.0
        
        # Sử dụng công thức xác suất độc lập
        prob_failure = 1.0
        for move in moves:
            prob_failure *= (1 - move['success_probability'])
        
        composite_success = 1 - prob_failure
        return min(composite_success, 0.98)  # Cap at 98%
    
    def _calculate_total_cost(self, moves: List[Dict]) -> str:
        """Tính tổng chi phí"""
        cost_values = {'Low': 1, 'Medium': 2, 'High': 3}
        total_cost_value = sum(cost_values.get(move['cost'], 2) for move in moves)
        
        if total_cost_value <= 2:
            return 'Low'
        elif total_cost_value <= 4:
            return 'Medium'
        else:
            return 'High'
    
    def _calculate_timeline(self, moves: List[Dict]) -> str:
        """Tính timeline tổng thể"""
        max_weeks = 0
        for move in moves:
            time_str = move['time']
            if 'week' in time_str:
                weeks = int(time_str.split('-')[-1].split()[0]) if '-' in time_str else int(time_str.split()[0])
                max_weeks = max(max_weeks, weeks)
        
        return f"{max_weeks} weeks"
    
    def _get_target_name(self, move_type: str) -> str:
        """Lấy tên target cho move type"""
        target_names = {
            'featured_snippet': 'Featured Snippet',
            'video_carousel': 'Video Carousel',
            'people_also_ask': 'People Also Ask',
            'local_pack': 'Local Pack'
        }
        return target_names.get(move_type, 'Unknown')
    
    def _calculate_implementation_day(self, move_type: str) -> int:
        """Tính ngày bắt đầu thực hiện move"""
        implementation_days = {
            'local_pack': 3,
            'people_also_ask': 7,
            'featured_snippet': 14,
            'video_carousel': 21
        }
        return implementation_days.get(move_type, 14)
    
    def _generate_detailed_actions(self, move_type: str) -> List[str]:
        """Tạo danh sách hành động chi tiết cho từng move type"""
        actions = {
            'featured_snippet': [
                'Nghiên cứu featured snippet hiện tại và format',
                'Viết nội dung 2500+ từ với cấu trúc rõ ràng',
                'Tối ưu hóa heading tags và schema markup',
                'Thêm bảng, danh sách và infographic'
            ],
            'video_carousel': [
                'Sản xuất video 2-3 phút chất lượng cao',
                'Tối ưu hóa title, description và tags',
                'Tạo thumbnail hấp dẫn',
                'Upload lên YouTube và embed vào website'
            ],
            'people_also_ask': [
                'Thu thập 15-20 câu hỏi phổ biến nhất',
                'Tạo trang FAQ chi tiết',
                'Sử dụng schema markup cho FAQ',
                'Tối ưu hóa nội dung cho voice search'
            ],
            'local_pack': [
                'Cập nhật Google Business Profile',
                'Thu thập reviews tích cực',
                'Tối ưu hóa NAP consistency',
                'Tạo local landing pages'
            ]
        }
        return actions.get(move_type, [])