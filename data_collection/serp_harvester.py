import asyncio
import aiohttp
from bs4 import BeautifulSoup
import json
from typing import Dict, List, Optional
from datetime import datetime
from services.api_key_manager import api_key_manager

class SERPHarvester:
    def __init__(self):
        self.serp_components = [
            'organic_results', 'featured_snippet', 'people_also_ask',
            'related_searches', 'video_results', 'image_results',
            'local_pack', 'knowledge_graph'
        ]
        
    async def fetch_serp(self, keyword: str, location: str = 'vn', num_results: int = 50) -> Dict:
        """Thu thập dữ liệu SERP từ nhiều nguồn với API key rotation"""
        params = {
            'q': keyword,
            'location': location,
            'hl': 'vi',
            'gl': 'vn',
            'num': num_results
        }
        
        # Thu thập từ nhiều nguồn với API key rotation
        sources = [
            self._fetch_google_direct(params),
            self._fetch_with_api_rotation('serpapi', params),
            self._fetch_with_api_rotation('serpstack', params)
        ]
        
        results = await asyncio.gather(*[s for s in sources if s], return_exceptions=True)
        return self._consolidate_results(results, keyword)

    async def _fetch_google_direct(self, params: Dict) -> Dict:
        """Real Google SERP data collection"""
        try:
            headers = {
                'User-Agent': self._get_random_user_agent(),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Cache-Control': 'max-age=0'
            }
            
            # Add delay to avoid rate limiting
            await asyncio.sleep(2)
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get('https://www.google.com/search', 
                                    params=params, headers=headers) as response:
                    if response.status != 200:
                        return {'error': f'HTTP {response.status}'}
                    
                    html = await response.text()
                    return self._parse_google_html(html)
        except Exception as e:
            return {'error': str(e)}
    
    def _get_random_user_agent(self) -> str:
        """Get random user agent to avoid detection"""
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0'
        ]
        import random
        return random.choice(user_agents)

    def _parse_google_html(self, html: str) -> Dict:
        """Real Google HTML parsing with comprehensive SERP features"""
        soup = BeautifulSoup(html, 'html.parser')
        results = {
            'organic_results': [],
            'featured_snippet': None,
            'people_also_ask': [],
            'video_results': [],
            'image_results': [],
            'local_pack': [],
            'ads': [],
            'related_searches': [],
            'knowledge_graph': None
        }
        
        # Real organic results parsing
        for i, g in enumerate(soup.select('.g, .tF2Cxc')):
            if i >= 10:  # Limit to top 10
                break
                
            title_elem = g.select_one('h3')
            url_elem = g.select_one('a')
            desc_elem = g.select_one('.VwiC3b, .s')
            
            if title_elem and url_elem:
                url = url_elem.get('href', '')
                if url.startswith('/url?q='):
                    url = url.split('/url?q=')[1].split('&')[0]
                
                results['organic_results'].append({
                    'position': i + 1,
                    'title': title_elem.get_text(strip=True),
                    'url': url,
                    'description': desc_elem.get_text(strip=True) if desc_elem else '',
                    'domain': self._extract_domain(url)
                })
        
        # Real featured snippet detection
        featured_selectors = ['.xpdopen', '.kp-blk', '.IZ6rdc']
        for selector in featured_selectors:
            featured = soup.select_one(selector)
            if featured:
                source_link = featured.select_one('a')
                results['featured_snippet'] = {
                    'text': featured.get_text(strip=True)[:500],  # Limit text
                    'source': source_link.get('href') if source_link else None,
                    'type': 'paragraph'
                }
                break
        
        # Real People Also Ask
        paa_elements = soup.select('.related-question-pair, .JlqpRe')
        for elem in paa_elements[:8]:  # Limit to 8
            question_text = elem.get_text(strip=True)
            if question_text and len(question_text) > 10:
                results['people_also_ask'].append(question_text)
        
        # Real video results
        video_elements = soup.select('.RzdJxc, .P94G9b')
        for elem in video_elements[:5]:  # Limit to 5
            title_elem = elem.select_one('h3, .fc9yUc')
            if title_elem:
                results['video_results'].append({
                    'title': title_elem.get_text(strip=True),
                    'source': 'YouTube'  # Most video results are from YouTube
                })
        
        # Real local pack
        local_elements = soup.select('.rllt__details')
        for elem in local_elements[:3]:  # Limit to 3
            name_elem = elem.select_one('.dbg0pd')
            if name_elem:
                results['local_pack'].append({
                    'name': name_elem.get_text(strip=True)
                })
        
        # Real ads detection
        ad_elements = soup.select('.uEierd, .v5yQqb')
        results['ads'] = [{'position': i+1} for i in range(min(len(ad_elements), 4))]
        
        # Real related searches
        related_elements = soup.select('.k8XOCe, .AuVD')
        for elem in related_elements[:8]:  # Limit to 8
            text = elem.get_text(strip=True)
            if text and len(text) > 3:
                results['related_searches'].append(text)
        
        return results
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc.lower().replace('www.', '')
        except:
            return 'unknown'
    
    def _consolidate_results(self, results: List, keyword: str) -> Dict:
        """Consolidate results from multiple sources with data validation"""
        consolidated = {
            'keyword': keyword,
            'timestamp': datetime.now().isoformat(),
            'sources_count': 0,
            'data': {
                'organic_results': [],
                'featured_snippet': None,
                'people_also_ask': [],
                'video_results': [],
                'ads': [],
                'local_pack': [],
                'related_searches': []
            }
        }
        
        valid_results = [r for r in results if not isinstance(r, Exception) and r and 'error' not in r]
        consolidated['sources_count'] = len(valid_results)
        
        if not valid_results:
            return consolidated
        
        # Merge organic results (prioritize first source)
        for result in valid_results:
            if 'organic_results' in result and result['organic_results']:
                if not consolidated['data']['organic_results']:
                    consolidated['data']['organic_results'] = result['organic_results'][:10]
                break
        
        # Merge other SERP features
        for result in valid_results:
            for feature in ['featured_snippet', 'people_also_ask', 'video_results', 'ads', 'local_pack']:
                if feature in result and result[feature] and not consolidated['data'][feature]:
                    consolidated['data'][feature] = result[feature]
        
        # Calculate SERP complexity score
        complexity = len(consolidated['data']['organic_results'])
        if consolidated['data']['featured_snippet']: complexity += 2
        if consolidated['data']['people_also_ask']: complexity += 1
        if consolidated['data']['video_results']: complexity += 1
        if consolidated['data']['local_pack']: complexity += 1
        
        consolidated['data']['serp_complexity_score'] = complexity
        
        return consolidated
    
    async def _fetch_with_api_rotation(self, api_type: str, params: Dict) -> Dict:
        """Fetch data with API key rotation"""
        max_retries = 3
        
        for attempt in range(max_retries):
            api_key = api_key_manager.get_active_key(api_type)
            
            if not api_key:
                return {'error': f'No active {api_type} keys available'}
            
            try:
                if api_type == 'serpapi':
                    result = await self._fetch_serpapi(params, api_key)
                elif api_type == 'serpstack':
                    result = await self._fetch_serpstack(params, api_key)
                else:
                    return {'error': f'Unknown API type: {api_type}'}
                
                # Success - increment usage
                api_key_manager.increment_usage(api_type, api_key)
                return result
                
            except Exception as e:
                # Mark key as failed and try next one
                api_key_manager.mark_key_failed(api_type, api_key)
                if attempt == max_retries - 1:
                    return {'error': f'{api_type} failed after {max_retries} attempts: {str(e)}'}
        
        return {'error': f'All {api_type} keys exhausted'}
    
    async def _fetch_serpapi(self, params: Dict, api_key: str) -> Dict:
        """Fetch from SerpAPI with specific key"""
        try:
            async with aiohttp.ClientSession() as session:
                serpapi_params = {**params, 'api_key': api_key, 'engine': 'google'}
                async with session.get('https://serpapi.com/search', params=serpapi_params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        raise Exception(f'SerpAPI error: {response.status}')
        except Exception as e:
            raise Exception(f'SerpAPI request failed: {str(e)}')
    
    async def _fetch_serpstack(self, params: Dict, api_key: str) -> Dict:
        """Fetch from SerpStack with specific key"""
        try:
            async with aiohttp.ClientSession() as session:
                serpstack_params = {**params, 'access_key': api_key}
                async with session.get('http://api.serpstack.com/search', params=serpstack_params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        raise Exception(f'SerpStack error: {response.status}')
        except Exception as e:
            raise Exception(f'SerpStack request failed: {str(e)}')