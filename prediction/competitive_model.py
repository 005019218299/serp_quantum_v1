from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import numpy as np
import joblib
import os
from typing import List, Dict, Tuple

class CompetitiveResponsePredictor:
    def __init__(self, model_path: str = "./models"):
        self.model_path = model_path
        self.model = MultiOutputClassifier(
            RandomForestClassifier(n_estimators=100, random_state=42)
        )
        self.feature_names = [
            'serp_complexity_score', 'has_featured_snippet', 'has_video_carousel',
            'top_domain_concentration', 'unique_domains_count', 'domain_diversity_score'
        ]
        self.is_trained = False
        self._load_model()
    
    def prepare_competitive_data(self, historical_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Chuẩn bị dữ liệu cho dự đoán phản ứng đối thủ"""
        if not historical_data or len(historical_data) < 2:
            return np.array([]), np.array([])
        
        try:
            # Vectorized data preparation for better performance
            data_pairs = [(historical_data[i], historical_data[i + 1]) 
                         for i in range(len(historical_data) - 1)]
            
            X = [self._extract_features(current['serp_features']) 
                 for current, _ in data_pairs]
            y = [self._calculate_changes(current['serp_features'], next_data['serp_features']) 
                 for current, next_data in data_pairs]
            
            return np.array(X), np.array(y)
        except Exception as e:
            print(f"Error preparing competitive data: {e}")
            return np.array([]), np.array([])
    
    def _extract_features(self, state: Dict) -> List[float]:
        """Trích xuất features liên quan cho dự đoán"""
        return [state.get(name, 0) for name in self.feature_names]
    
    def _calculate_changes(self, current_state: Dict, next_state: Dict) -> List[int]:
        """Tính toán thay đổi giữa hai trạng thái SERP"""
        changes = []
        
        # Thay đổi trong featured snippet
        changes.append(int(current_state.get('has_featured_snippet', 0) != 
                          next_state.get('has_featured_snippet', 0)))
        
        # Thay đổi trong video carousel
        changes.append(int(current_state.get('has_video_carousel', 0) != 
                          next_state.get('has_video_carousel', 0)))
        
        # Thay đổi trong people also ask
        changes.append(int(current_state.get('has_people_also_ask', 0) != 
                          next_state.get('has_people_also_ask', 0)))
        
        # Thay đổi trong domain concentration (đối thủ mới xuất hiện)
        current_conc = current_state.get('top_domain_concentration', 0)
        next_conc = next_state.get('top_domain_concentration', 0)
        changes.append(int(abs(current_conc - next_conc) > 0.1))
        
        return changes
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Huấn luyện mô hình dự đoán phản ứng đối thủ"""
        if len(X) == 0:
            print("Không đủ dữ liệu để huấn luyện competitive model")
            return
        
        self.model.fit(X, y)
        self.is_trained = True
        self._save_model()
        print(f"Đã huấn luyện competitive model với {len(X)} samples")
    
    def predict_competitor_moves(self, current_state: Dict) -> Dict:
        """Dự đoán phản ứng của đối thủ"""
        if not self.is_trained:
            return {
                'featured_snippet_change_prob': 0.5,
                'video_carousel_change_prob': 0.3,
                'paa_change_prob': 0.4,
                'new_competitor_prob': 0.2
            }
        
        features = self._extract_features(current_state)
        predictions = self.model.predict_proba([features])
        
        # Trích xuất xác suất thay đổi cho từng thành phần
        result = {}
        change_types = ['featured_snippet', 'video_carousel', 'paa', 'new_competitor']
        
        for i, change_type in enumerate(change_types):
            if len(predictions[i]) > 1:
                result[f'{change_type}_change_prob'] = predictions[i][0][1]  # Prob of change
            else:
                result[f'{change_type}_change_prob'] = 0.5
        
        return result
    
    def simulate_competitor_response(self, current_state: Dict, strategic_moves: List[Dict]) -> Dict:
        """Mô phỏng phản ứng của đối thủ với chiến lược của người dùng"""
        base_predictions = self.predict_competitor_moves(current_state)
        
        if not strategic_moves:
            return base_predictions
        
        # Define multipliers as constants for better maintainability
        MULTIPLIERS = {
            'Featured Snippet': 1.5,
            'Video Carousel': 1.3, 
            'People Also Ask': 1.2
        }
        
        # Apply multipliers efficiently
        adjusted_predictions = base_predictions.copy()
        
        for move in strategic_moves:
            target = move.get('target')
            if target in MULTIPLIERS:
                key_map = {
                    'Featured Snippet': 'featured_snippet_change_prob',
                    'Video Carousel': 'video_carousel_change_prob',
                    'People Also Ask': 'paa_change_prob'
                }
                
                key = key_map.get(target)
                if key in adjusted_predictions:
                    adjusted_predictions[key] = min(adjusted_predictions[key] * MULTIPLIERS[target], 1.0)
        
        return adjusted_predictions
    
    def _save_model(self):
        """Lưu mô hình đã huấn luyện"""
        os.makedirs(self.model_path, exist_ok=True)
        joblib.dump(self.model, f"{self.model_path}/competitive_model.pkl")
    
    def _load_model(self):
        """Load mô hình đã huấn luyện"""
        model_file = f"{self.model_path}/competitive_model.pkl"
        
        try:
            if os.path.exists(model_file):
                self.model = joblib.load(model_file)
                self.is_trained = True
                print("Đã load mô hình competitive prediction")
            else:
                print("Chưa có mô hình competitive được huấn luyện")
        except Exception as e:
            print(f"Lỗi khi load competitive model: {e}")
            self.is_trained = False