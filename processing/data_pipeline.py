import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
from urllib.parse import urlparse

class DataProcessor:
    def __init__(self):
        self.feature_set = []
        
    def process_serp_data(self, raw_data: Dict) -> Dict:
        """Xử lý dữ liệu SERP thô thành features có cấu trúc"""
        processed = {
            'keyword': raw_data['keyword'],
            'timestamp': datetime.now(),
            'serp_features': self._extract_serp_features(raw_data['data']),
            'competitor_features': self._extract_competitor_features(raw_data['data']),
            'temporal_features': self._extract_temporal_features(raw_data)
        }
        return processed
    
    def _extract_serp_features(self, data: Dict) -> Dict:
        """Trích xuất đặc trưng từ cấu trúc SERP"""
        organic_results = data.get('organic_results', [])
        
        features = {
            'serp_complexity_score': len(organic_results),
            'has_featured_snippet': int('featured_snippet' in data),
            'has_video_carousel': int(data.get('video_results', False)),
            'has_people_also_ask': int(len(data.get('people_also_ask', [])) > 0),
            'local_pack_presence': int('local_pack' in data),
            'total_ads_count': len(data.get('ads', [])),
            'knowledge_graph_presence': int('knowledge_graph' in data),
            'paa_questions_count': len(data.get('people_also_ask', [])),
            'organic_results_count': len(organic_results)
        }
        return features
    
    def _extract_competitor_features(self, data: Dict) -> Dict:
        """Phân tích sự thống trị của đối thủ trong SERP"""
        organic_results = data.get('organic_results', [])
        domains = []
        
        for result in organic_results:
            if 'url' in result and result['url']:
                domain = self._extract_domain(result['url'])
                if domain:
                    domains.append(domain)
        
        if not domains:
            return {
                'top_domain_concentration': 0,
                'top_3_domains_ratio': 0,
                'unique_domains_count': 0,
                'domain_diversity_score': 0
            }
        
        domain_counts = pd.Series(domains).value_counts()
        
        return {
            'top_domain_concentration': domain_counts.iloc[0] if len(domain_counts) > 0 else 0,
            'top_3_domains_ratio': domain_counts.head(3).sum() / len(domains),
            'unique_domains_count': len(domain_counts),
            'domain_diversity_score': len(domain_counts) / len(domains) if domains else 0
        }
    
    def _extract_temporal_features(self, raw_data: Dict) -> Dict:
        """Trích xuất đặc trưng thời gian"""
        return {
            'hour_of_day': datetime.now().hour,
            'day_of_week': datetime.now().weekday(),
            'sources_reliability': raw_data.get('sources_count', 0)
        }
    
    def _extract_domain(self, url: str) -> str:
        """Trích xuất domain từ URL"""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower().replace('www.', '')
        except:
            return None
    
    def calculate_serp_volatility(self, historical_data: List[Dict]) -> float:
        """Tính độ biến động của SERP"""
        if len(historical_data) < 2:
            return 0.0
        
        changes = 0
        total_comparisons = len(historical_data) - 1
        
        for i in range(1, len(historical_data)):
            prev_features = historical_data[i-1]['serp_features']
            curr_features = historical_data[i]['serp_features']
            
            # So sánh các đặc trưng quan trọng
            key_features = ['has_featured_snippet', 'has_video_carousel', 'has_people_also_ask']
            for feature in key_features:
                if prev_features.get(feature) != curr_features.get(feature):
                    changes += 1
        
        return changes / (total_comparisons * len(key_features)) if total_comparisons > 0 else 0.0