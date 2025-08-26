#!/usr/bin/env python3
"""
Auto Keyword Discovery - Tự động phát hiện keyword trending worldwide
Multi-language, Multi-region Keyword Intelligence System
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Set
import json
import re
from collections import defaultdict
import random

class AutoKeywordDiscovery:
    def __init__(self):
        self.trending_sources = {
            'google_trends': 'https://trends.google.com/trends/api/dailytrends',
            'twitter_trends': 'https://api.twitter.com/1.1/trends/place.json',
            'reddit_trending': 'https://www.reddit.com/r/all/hot.json',
            'news_apis': ['newsapi.org', 'currentsapi.services']
        }
        
        self.discovered_keywords = defaultdict(list)
        self.keyword_scores = {}
        
        # Multi-language support
        self.languages = {
            'en': {'regions': ['US', 'GB', 'AU', 'CA'], 'weight': 1.0},
            'es': {'regions': ['ES', 'MX', 'AR', 'CO'], 'weight': 0.9},
            'fr': {'regions': ['FR', 'CA', 'BE', 'CH'], 'weight': 0.8},
            'de': {'regions': ['DE', 'AT', 'CH'], 'weight': 0.8},
            'zh': {'regions': ['CN', 'TW', 'HK', 'SG'], 'weight': 1.2},
            'ja': {'regions': ['JP'], 'weight': 1.1},
            'ko': {'regions': ['KR'], 'weight': 1.0},
            'vi': {'regions': ['VN'], 'weight': 0.7},
            'th': {'regions': ['TH'], 'weight': 0.6},
            'id': {'regions': ['ID'], 'weight': 0.6}
        }
    
    async def discover_trending_keywords(self, max_keywords: int = 1000) -> Dict[str, List[str]]:
        """Discover trending keywords across multiple sources and languages"""
        
        print("🔍 Starting Global Keyword Discovery...")
        
        all_keywords = defaultdict(list)
        
        # Discover from multiple sources
        tasks = [
            self._discover_from_google_trends(),
            self._discover_from_social_media(),
            self._discover_from_news_sources(),
            self._discover_from_search_suggestions(),
            self._discover_from_competitor_analysis()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Consolidate results
        for result in results:
            if isinstance(result, dict):
                for lang, keywords in result.items():
                    all_keywords[lang].extend(keywords)
        
        # Score and rank keywords
        ranked_keywords = await self._score_and_rank_keywords(all_keywords, max_keywords)
        
        print(f"✅ Discovered {sum(len(kws) for kws in ranked_keywords.values())} trending keywords")
        
        return ranked_keywords
    
    async def _discover_from_google_trends(self) -> Dict[str, List[str]]:
        """Discover keywords from Google Trends (simulated)"""
        
        print("📈 Discovering from Google Trends...")
        
        # Simulate Google Trends data
        trends_data = {
            'en': [
                'AI chatbot 2024', 'electric vehicles', 'remote work tools',
                'cryptocurrency news', 'climate change solutions', 'space exploration',
                'quantum computing', 'virtual reality games', 'sustainable fashion',
                'mental health apps', 'smart home devices', 'renewable energy'
            ],
            'es': [
                'inteligencia artificial', 'coches eléctricos', 'trabajo remoto',
                'criptomonedas', 'cambio climático', 'exploración espacial',
                'computación cuántica', 'realidad virtual', 'moda sostenible'
            ],
            'fr': [
                'intelligence artificielle', 'voitures électriques', 'travail à distance',
                'cryptomonnaies', 'changement climatique', 'exploration spatiale'
            ],
            'de': [
                'künstliche intelligenz', 'elektroautos', 'homeoffice',
                'kryptowährungen', 'klimawandel', 'weltraumforschung'
            ],
            'zh': [
                '人工智能', '电动汽车', '远程工作', '加密货币', '气候变化', '太空探索'
            ],
            'ja': [
                '人工知能', '電気自動車', 'リモートワーク', '暗号通貨', '気候変動', '宇宙探査'
            ],
            'vi': [
                'trí tuệ nhân tạo', 'xe điện', 'làm việc từ xa',
                'tiền mã hóa', 'biến đổi khí hậu', 'khám phá vũ trụ'
            ]
        }
        
        await asyncio.sleep(1)  # Simulate API delay
        return trends_data
    
    async def _discover_from_social_media(self) -> Dict[str, List[str]]:
        """Discover keywords from social media trends"""
        
        print("📱 Discovering from Social Media...")
        
        # Simulate social media trending topics
        social_trends = {
            'en': [
                'viral TikTok challenge', 'Instagram reels tips', 'Twitter spaces',
                'LinkedIn learning', 'YouTube shorts', 'Facebook marketplace',
                'Discord communities', 'Clubhouse rooms', 'Snapchat filters'
            ],
            'es': [
                'desafío viral TikTok', 'consejos Instagram reels', 'espacios Twitter',
                'aprendizaje LinkedIn', 'YouTube shorts', 'marketplace Facebook'
            ],
            'fr': [
                'défi viral TikTok', 'conseils Instagram reels', 'espaces Twitter',
                'apprentissage LinkedIn', 'YouTube shorts', 'marketplace Facebook'
            ],
            'vi': [
                'thử thách viral TikTok', 'mẹo Instagram reels', 'Twitter spaces',
                'học LinkedIn', 'YouTube shorts', 'chợ Facebook'
            ]
        }
        
        await asyncio.sleep(0.8)
        return social_trends
    
    async def _discover_from_news_sources(self) -> Dict[str, List[str]]:
        """Discover keywords from news sources"""
        
        print("📰 Discovering from News Sources...")
        
        # Simulate news trending topics
        news_trends = {
            'en': [
                'breaking news today', 'world cup 2024', 'election results',
                'stock market crash', 'new iPhone release', 'COVID variants',
                'climate summit', 'tech earnings', 'space mission'
            ],
            'es': [
                'noticias de última hora', 'copa mundial 2024', 'resultados electorales',
                'caída bolsa valores', 'nuevo iPhone', 'variantes COVID'
            ],
            'fr': [
                'dernières nouvelles', 'coupe du monde 2024', 'résultats élections',
                'krach boursier', 'nouvel iPhone', 'variants COVID'
            ],
            'vi': [
                'tin tức mới nhất', 'world cup 2024', 'kết quả bầu cử',
                'sụp đổ chứng khoán', 'iPhone mới', 'biến thể COVID'
            ]
        }
        
        await asyncio.sleep(1.2)
        return news_trends
    
    async def _discover_from_search_suggestions(self) -> Dict[str, List[str]]:
        """Discover keywords from search suggestions"""
        
        print("🔎 Discovering from Search Suggestions...")
        
        # Simulate search suggestion data
        suggestion_keywords = {
            'en': [
                'how to make money online', 'best laptop 2024', 'healthy recipes',
                'workout at home', 'learn programming', 'travel destinations',
                'investment tips', 'online courses', 'job interview tips'
            ],
            'es': [
                'cómo ganar dinero online', 'mejor laptop 2024', 'recetas saludables',
                'ejercicio en casa', 'aprender programación', 'destinos viaje'
            ],
            'vi': [
                'cách kiếm tiền online', 'laptop tốt nhất 2024', 'công thức nấu ăn',
                'tập thể dục tại nhà', 'học lập trình', 'điểm du lịch'
            ]
        }
        
        await asyncio.sleep(0.5)
        return suggestion_keywords
    
    async def _discover_from_competitor_analysis(self) -> Dict[str, List[str]]:
        """Discover keywords from competitor analysis"""
        
        print("🎯 Discovering from Competitor Analysis...")
        
        # Simulate competitor keyword data
        competitor_keywords = {
            'en': [
                'SEO tools comparison', 'digital marketing strategy', 'content creation',
                'social media management', 'email marketing', 'web analytics',
                'conversion optimization', 'brand awareness', 'customer retention'
            ],
            'es': [
                'comparación herramientas SEO', 'estrategia marketing digital', 'creación contenido',
                'gestión redes sociales', 'email marketing', 'analítica web'
            ],
            'vi': [
                'so sánh công cụ SEO', 'chiến lược marketing số', 'tạo nội dung',
                'quản lý mạng xã hội', 'email marketing', 'phân tích web'
            ]
        }
        
        await asyncio.sleep(1.5)
        return competitor_keywords
    
    async def _score_and_rank_keywords(self, all_keywords: Dict[str, List[str]], 
                                     max_keywords: int) -> Dict[str, List[str]]:
        """Score and rank keywords by relevance and potential"""
        
        print("📊 Scoring and ranking keywords...")
        
        ranked_keywords = {}
        
        for language, keywords in all_keywords.items():
            # Remove duplicates
            unique_keywords = list(set(keywords))
            
            # Score keywords
            scored_keywords = []
            for keyword in unique_keywords:
                score = self._calculate_keyword_score(keyword, language)
                scored_keywords.append((keyword, score))
            
            # Sort by score and limit
            scored_keywords.sort(key=lambda x: x[1], reverse=True)
            
            # Get language weight
            lang_weight = self.languages.get(language, {}).get('weight', 1.0)
            max_for_lang = int(max_keywords * lang_weight / 10)  # Distribute based on weight
            
            ranked_keywords[language] = [kw[0] for kw in scored_keywords[:max_for_lang]]
        
        return ranked_keywords
    
    def _calculate_keyword_score(self, keyword: str, language: str) -> float:
        """Calculate keyword score based on multiple factors"""
        
        score = 0.0
        
        # Length factor (prefer 2-4 words)
        word_count = len(keyword.split())
        if 2 <= word_count <= 4:
            score += 1.0
        elif word_count == 1 or word_count == 5:
            score += 0.5
        
        # Commercial intent keywords
        commercial_terms = ['buy', 'best', 'review', 'price', 'cheap', 'discount', 
                          'comprar', 'mejor', 'precio', 'barato', 'descuento',
                          'acheter', 'meilleur', 'prix', 'pas cher', 'remise',
                          'mua', 'tốt nhất', 'giá', 'rẻ', 'giảm giá']
        
        if any(term in keyword.lower() for term in commercial_terms):
            score += 1.5
        
        # Trending indicators
        trending_terms = ['2024', '2025', 'new', 'latest', 'trending', 'viral',
                         'nuevo', 'último', 'tendencia', 'viral',
                         'nouveau', 'dernier', 'tendance', 'viral',
                         'mới', 'mới nhất', 'xu hướng', 'viral']
        
        if any(term in keyword.lower() for term in trending_terms):
            score += 1.2
        
        # Technology/AI keywords (high value)
        tech_terms = ['AI', 'artificial intelligence', 'machine learning', 'blockchain',
                     'IA', 'inteligencia artificial', 'aprendizaje automático',
                     'trí tuệ nhân tạo', 'học máy', 'blockchain']
        
        if any(term in keyword.lower() for term in tech_terms):
            score += 2.0
        
        # Language weight
        lang_weight = self.languages.get(language, {}).get('weight', 1.0)
        score *= lang_weight
        
        return score
    
    async def get_personalized_keywords(self, user_profile: Dict) -> List[str]:
        """Get personalized keywords based on user profile"""
        
        user_language = user_profile.get('language', 'en')
        user_interests = user_profile.get('interests', [])
        user_industry = user_profile.get('industry', 'general')
        
        # Get base trending keywords
        trending = await self.discover_trending_keywords(max_keywords=500)
        base_keywords = trending.get(user_language, [])
        
        # Filter by interests
        personalized = []
        for keyword in base_keywords:
            keyword_lower = keyword.lower()
            
            # Match user interests
            if any(interest.lower() in keyword_lower for interest in user_interests):
                personalized.append(keyword)
            
            # Match industry
            if user_industry.lower() in keyword_lower:
                personalized.append(keyword)
        
        # Add some general trending keywords
        personalized.extend(base_keywords[:20])
        
        # Remove duplicates and limit
        return list(set(personalized))[:100]
    
    async def continuous_keyword_monitoring(self, interval_hours: int = 6):
        """Continuously monitor and update trending keywords"""
        
        print(f"🔄 Starting continuous keyword monitoring (every {interval_hours}h)...")
        
        while True:
            try:
                # Discover new keywords
                new_keywords = await self.discover_trending_keywords()
                
                # Update keyword database
                timestamp = datetime.now().isoformat()
                
                for language, keywords in new_keywords.items():
                    self.discovered_keywords[language] = keywords
                    
                    # Store in database (simulated)
                    print(f"💾 Updated {len(keywords)} keywords for {language}")
                
                print(f"✅ Keyword monitoring cycle completed at {timestamp}")
                
                # Wait for next cycle
                await asyncio.sleep(interval_hours * 3600)
                
            except Exception as e:
                print(f"❌ Keyword monitoring error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

# Demo usage
async def demo_keyword_discovery():
    """Demo auto keyword discovery"""
    
    discovery = AutoKeywordDiscovery()
    
    # Discover trending keywords
    trending_keywords = await discovery.discover_trending_keywords(max_keywords=200)
    
    print("\n🌍 Global Trending Keywords:")
    for language, keywords in trending_keywords.items():
        print(f"\n📍 {language.upper()} ({len(keywords)} keywords):")
        for i, keyword in enumerate(keywords[:5], 1):
            print(f"   {i}. {keyword}")
    
    # Get personalized keywords
    user_profile = {
        'language': 'vi',
        'interests': ['technology', 'AI', 'programming'],
        'industry': 'software'
    }
    
    personalized = await discovery.get_personalized_keywords(user_profile)
    print(f"\n👤 Personalized Keywords for Vietnamese Tech User:")
    for i, keyword in enumerate(personalized[:10], 1):
        print(f"   {i}. {keyword}")

if __name__ == "__main__":
    asyncio.run(demo_keyword_discovery())