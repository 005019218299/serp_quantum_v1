import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
from datetime import datetime, timedelta

class SERPComponentType(Enum):
    ORGANIC = "organic"
    FEATURED_SNIPPET = "featured_snippet"
    VIDEO = "video"
    PEOPLE_ALSO_ASK = "people_also_ask"
    LOCAL_PACK = "local_pack"
    KNOWLEDGE_GRAPH = "knowledge_graph"

@dataclass
class SERPComponent:
    type: SERPComponentType
    position: int
    stability: float  # 0.0-1.0 xác suất thay đổi
    content: Dict
    owner: str  # Domain sở hữu vị trí này

class SERPSimulator:
    def __init__(self, temporal_model, competitive_model):
        self.temporal_model = temporal_model
        self.competitive_model = competitive_model
        self.current_serp = []
        self.simulation_history = []
        
    def initialize_serp(self, current_data: Dict):
        """Khởi tạo simulation với trạng thái SERP hiện tại"""
        self.current_serp = self._create_components_from_data(current_data)
        self.simulation_history = [self._serialize_serp_state()]
    
    def simulate_time_evolution(self, days: int = 30, strategic_moves: Optional[List[Dict]] = None) -> List[Dict]:
        """Mô phỏng sự tiến hóa của SERP theo thời gian"""
        predictions = []
        
        # Lấy dự đoán từ temporal model
        current_features = self._extract_current_features()
        temporal_predictions = self.temporal_model.predict_future_serp(current_features, days)
        
        # Mô phỏng từng ngày
        for day in range(days):
            day_prediction = self._simulate_single_day(
                temporal_predictions[day] if day < len(temporal_predictions) else current_features,
                strategic_moves,
                day
            )
            predictions.append(day_prediction)
            
        return predictions
    
    def _simulate_single_day(self, temporal_pred: Dict, strategic_moves: Optional[List[Dict]], day: int) -> Dict:
        """Mô phỏng SERP cho một ngày cụ thể"""
        # Dự đoán phản ứng đối thủ
        competitive_pred = self.competitive_model.predict_competitor_moves(temporal_pred)
        
        # Áp dụng strategic moves nếu có
        if strategic_moves:
            competitive_pred = self.competitive_model.simulate_competitor_response(
                temporal_pred, strategic_moves
            )
        
        # Tính toán trạng thái SERP mới
        new_serp_state = self._calculate_new_serp_state(temporal_pred, competitive_pred, strategic_moves, day)
        
        return {
            'day': day + 1,
            'date': (datetime.now() + timedelta(days=day + 1)).isoformat(),
            'serp_state': new_serp_state,
            'temporal_prediction': temporal_pred,
            'competitive_prediction': competitive_pred,
            'stability_score': self._calculate_stability_score(new_serp_state),
            'opportunity_score': self._calculate_opportunity_score(new_serp_state, strategic_moves)
        }
    
    def _calculate_new_serp_state(self, temporal_pred: Dict, competitive_pred: Dict, 
                                 strategic_moves: Optional[List[Dict]], day: int) -> Dict:
        """Tính toán trạng thái SERP mới dựa trên các dự đoán"""
        new_state = temporal_pred.copy()
        
        # Áp dụng thay đổi từ competitive predictions
        if competitive_pred.get('featured_snippet_change_prob', 0) > 0.7:
            new_state['has_featured_snippet'] = 1 - new_state.get('has_featured_snippet', 0)
        
        if competitive_pred.get('video_carousel_change_prob', 0) > 0.6:
            new_state['has_video_carousel'] = 1 - new_state.get('has_video_carousel', 0)
        
        if competitive_pred.get('paa_change_prob', 0) > 0.5:
            new_state['has_people_also_ask'] = 1 - new_state.get('has_people_also_ask', 0)
        
        # Áp dụng strategic moves
        if strategic_moves:
            new_state = self._apply_strategic_moves(new_state, strategic_moves, day)
        
        return new_state
    
    def _apply_strategic_moves(self, serp_state: Dict, strategic_moves: List[Dict], day: int) -> Dict:
        """Áp dụng các nước đi chiến lược vào SERP state"""
        modified_state = serp_state.copy()
        
        for move in strategic_moves:
            # Kiểm tra xem move có được thực hiện vào ngày này không
            move_day = move.get('implementation_day', 14)  # Default sau 14 ngày
            
            if day >= move_day:
                success_prob = move.get('success_probability', 0.5)
                
                # Mô phỏng thành công của move
                if np.random.random() < success_prob:
                    if move['target'] == 'Featured Snippet':
                        modified_state['has_featured_snippet'] = 1
                    elif move['target'] == 'Video Carousel':
                        modified_state['has_video_carousel'] = 1
                    elif move['target'] == 'People Also Ask':
                        modified_state['has_people_also_ask'] = 1
                        modified_state['paa_questions_count'] = max(
                            modified_state.get('paa_questions_count', 0), 3
                        )
        
        return modified_state
    
    def _calculate_stability_score(self, serp_state: Dict) -> float:
        """Tính điểm ổn định của SERP"""
        # SERP càng phức tạp thì càng ít ổn định
        complexity = serp_state.get('serp_complexity_score', 10)
        features_count = sum([
            serp_state.get('has_featured_snippet', 0),
            serp_state.get('has_video_carousel', 0),
            serp_state.get('has_people_also_ask', 0),
            serp_state.get('local_pack_presence', 0)
        ])
        
        # Điểm ổn định từ 0-10
        stability = max(0, 10 - (complexity / 5) - features_count)
        return min(stability, 10) / 10
    
    def _calculate_opportunity_score(self, serp_state: Dict, strategic_moves: Optional[List[Dict]]) -> float:
        """Tính điểm cơ hội dựa trên SERP state"""
        opportunity = 0
        
        # Cơ hội từ việc thiếu featured snippet
        if not serp_state.get('has_featured_snippet', 0):
            opportunity += 3
        
        # Cơ hội từ việc thiếu video
        if not serp_state.get('has_video_carousel', 0):
            opportunity += 2
        
        # Cơ hội từ diversity thấp
        diversity = serp_state.get('domain_diversity_score', 0.5)
        if diversity < 0.7:
            opportunity += 2
        
        # Bonus nếu có strategic moves phù hợp
        if strategic_moves:
            opportunity += len(strategic_moves) * 0.5
        
        return min(opportunity, 10) / 10
    
    def _extract_current_features(self) -> Dict:
        """Trích xuất features từ trạng thái SERP hiện tại"""
        # Placeholder - sẽ được implement dựa trên current_serp
        return {
            'serp_complexity_score': 10,
            'has_featured_snippet': 1,
            'has_video_carousel': 0,
            'has_people_also_ask': 1,
            'local_pack_presence': 0,
            'total_ads_count': 3,
            'knowledge_graph_presence': 1,
            'paa_questions_count': 4,
            'organic_results_count': 10,
            'top_domain_concentration': 2,
            'top_3_domains_ratio': 0.6,
            'unique_domains_count': 8,
            'domain_diversity_score': 0.8,
            'hour_of_day': datetime.now().hour,
            'day_of_week': datetime.now().weekday()
        }
    
    def _create_components_from_data(self, data: Dict) -> List[SERPComponent]:
        """Tạo SERP components từ dữ liệu thô"""
        components = []
        
        # Tạo organic results
        organic_results = data.get('organic_results', [])
        for i, result in enumerate(organic_results[:10]):
            components.append(SERPComponent(
                type=SERPComponentType.ORGANIC,
                position=i + 1,
                stability=0.7,  # Organic results tương đối ổn định
                content=result,
                owner=self._extract_domain(result.get('url', ''))
            ))
        
        return components
    
    def _extract_domain(self, url: str) -> str:
        """Trích xuất domain từ URL"""
        from urllib.parse import urlparse
        try:
            return urlparse(url).netloc.lower().replace('www.', '')
        except:
            return 'unknown'
    
    def _serialize_serp_state(self) -> Dict:
        """Serialize SERP state để lưu trữ"""
        return {
            'timestamp': datetime.now().isoformat(),
            'components': [
                {
                    'type': comp.type.value,
                    'position': comp.position,
                    'stability': comp.stability,
                    'owner': comp.owner
                }
                for comp in self.current_serp
            ]
        }