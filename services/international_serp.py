from enum import Enum
from typing import Dict, List
import asyncio
import aiohttp
from datetime import datetime

class SupportedLanguage(Enum):
    VIETNAMESE = {"code": "vi", "gl": "vn", "name": "Vietnamese"}
    ENGLISH = {"code": "en", "gl": "us", "name": "English"}
    JAPANESE = {"code": "ja", "gl": "jp", "name": "Japanese"}
    KOREAN = {"code": "ko", "gl": "kr", "name": "Korean"}
    CHINESE = {"code": "zh", "gl": "cn", "name": "Chinese"}
    THAI = {"code": "th", "gl": "th", "name": "Thai"}
    SPANISH = {"code": "es", "gl": "es", "name": "Spanish"}
    FRENCH = {"code": "fr", "gl": "fr", "name": "French"}

class InternationalSERPAnalyzer:
    def __init__(self):
        self.regional_algorithms = {
            'vn': {'mobile_first': True, 'local_pack_priority': 'high'},
            'us': {'mobile_first': True, 'featured_snippet_priority': 'high'},
            'jp': {'mobile_first': True, 'amp_priority': 'high'},
            'kr': {'mobile_first': True, 'video_priority': 'high'},
            'cn': {'mobile_first': False, 'baidu_factors': True},
            'th': {'mobile_first': True, 'local_pack_priority': 'medium'}
        }
    
    async def analyze_multi_language_serp(self, keyword: str, languages: List[str]) -> Dict:
        """Analyze SERP across multiple languages/regions"""
        results = {}
        
        tasks = []
        for lang_code in languages:
            if self._is_supported_language(lang_code):
                task = self._fetch_regional_serp(keyword, lang_code)
                tasks.append((lang_code, task))
        
        for lang_code, task in tasks:
            try:
                regional_data = await task
                results[lang_code] = {
                    'serp_data': regional_data,
                    'regional_insights': self._analyze_regional_patterns(regional_data, lang_code),
                    'localization_opportunities': self._identify_localization_gaps(regional_data, lang_code)
                }
            except Exception as e:
                results[lang_code] = {'error': str(e)}
        
        return {
            'keyword': keyword,
            'timestamp': datetime.now().isoformat(),
            'regional_results': results,
            'cross_regional_analysis': self._compare_regional_differences(results)
        }
    
    async def _fetch_regional_serp(self, keyword: str, lang_code: str) -> Dict:
        """Fetch SERP for specific language/region"""
        lang_config = self._get_language_config(lang_code)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept-Language': f"{lang_config['code']},{lang_config['code']};q=0.9"
        }
        
        params = {
            'q': keyword,
            'hl': lang_config['code'],
            'gl': lang_config['gl'],
            'lr': f"lang_{lang_config['code']}"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                'https://www.google.com/search',
                headers=headers,
                params=params
            ) as response:
                html = await response.text()
                return self._parse_regional_serp(html, lang_code)
    
    def _parse_regional_serp(self, html: str, lang_code: str) -> Dict:
        """Parse region-specific SERP features"""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        results = {
            'language': lang_code,
            'organic_results': [],
            'featured_snippet': None,
            'local_pack': [],
            'regional_features': {}
        }
        
        # Parse organic results
        for g in soup.select('.g')[:10]:
            title_elem = g.select_one('h3')
            url_elem = g.select_one('a')
            if title_elem and url_elem:
                results['organic_results'].append({
                    'title': title_elem.text,
                    'url': url_elem.get('href'),
                    'language_detected': self._detect_content_language(title_elem.text)
                })
        
        # Regional-specific features
        if lang_code in ['ja', 'kr']:
            results['regional_features']['mobile_amp'] = len(soup.select('[data-amp]')) > 0
        
        if lang_code in ['vn', 'th']:
            results['regional_features']['local_emphasis'] = len(soup.select('.rllt__details')) > 0
        
        return results
    
    def _analyze_regional_patterns(self, serp_data: Dict, lang_code: str) -> Dict:
        """Analyze region-specific SERP patterns"""
        algorithm_config = self.regional_algorithms.get(lang_code, {})
        
        patterns = {
            'mobile_optimization_importance': algorithm_config.get('mobile_first', True),
            'local_seo_priority': algorithm_config.get('local_pack_priority', 'medium'),
            'content_length_preference': self._analyze_content_length_patterns(serp_data),
            'domain_authority_impact': self._analyze_domain_patterns(serp_data)
        }
        
        return patterns
    
    def _identify_localization_gaps(self, serp_data: Dict, lang_code: str) -> List[Dict]:
        """Identify localization opportunities"""
        gaps = []
        
        organic_results = serp_data.get('organic_results', [])
        local_content_ratio = sum(1 for result in organic_results 
                                if result.get('language_detected') == lang_code) / max(len(organic_results), 1)
        
        if local_content_ratio < 0.7:
            gaps.append({
                'type': 'language_localization',
                'priority': 'high',
                'description': f'Only {local_content_ratio:.1%} of results are in local language'
            })
        
        if not serp_data.get('regional_features', {}).get('local_emphasis'):
            gaps.append({
                'type': 'local_presence',
                'priority': 'medium',
                'description': 'Limited local business presence in SERP'
            })
        
        return gaps
    
    def _compare_regional_differences(self, results: Dict) -> Dict:
        """Compare differences across regions"""
        if len(results) < 2:
            return {}
        
        differences = {
            'serp_structure_variations': {},
            'content_language_distribution': {},
            'regional_feature_differences': {}
        }
        
        for lang_code, data in results.items():
            if 'serp_data' in data:
                serp_data = data['serp_data']
                differences['content_language_distribution'][lang_code] = len(serp_data.get('organic_results', []))
        
        return differences
    
    def _is_supported_language(self, lang_code: str) -> bool:
        return any(lang.value['code'] == lang_code for lang in SupportedLanguage)
    
    def _get_language_config(self, lang_code: str) -> Dict:
        for lang in SupportedLanguage:
            if lang.value['code'] == lang_code:
                return lang.value
        return SupportedLanguage.ENGLISH.value
    
    def _detect_content_language(self, text: str) -> str:
        # Simplified language detection
        if any(ord(char) > 127 for char in text):
            return 'non-latin'
        return 'latin'
    
    def _analyze_content_length_patterns(self, serp_data: Dict) -> str:
        organic_results = serp_data.get('organic_results', [])
        avg_title_length = sum(len(r.get('title', '')) for r in organic_results) / max(len(organic_results), 1)
        return 'long' if avg_title_length > 60 else 'short'
    
    def _analyze_domain_patterns(self, serp_data: Dict) -> str:
        # Simplified domain authority analysis
        return 'high_authority_preferred'