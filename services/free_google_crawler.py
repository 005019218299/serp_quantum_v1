import asyncio
import aiohttp
from bs4 import BeautifulSoup
import random
import time
from typing import Dict, List
from fake_useragent import UserAgent
import json
from datetime import datetime

class FreeGoogleCrawler:
    def __init__(self):
        self.ua = UserAgent()
        self.session_pool = []
        self.proxy_list = []
        self.request_count = 0
        self.success_rate = 0.0
        
    async def setup_stealth_mode(self):
        """Setup stealth crawling với multiple techniques"""
        print("🔧 Setting up stealth mode...")
        
        # 1. Load free proxies
        self.proxy_list = await self._get_free_proxies()
        print(f"📡 Loaded {len(self.proxy_list)} free proxies")
        
        # 2. Create session pool
        for _ in range(3):
            session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                connector=aiohttp.TCPConnector(limit=10)
            )
            self.session_pool.append(session)
        
        print("✅ Stealth mode ready")
    
    async def crawl_google_free(self, keyword: str, location: str = 'vn') -> Dict:
        """Crawl Google hoàn toàn miễn phí với fallback strategies"""
        
        strategies = [
            ('rotation', self._crawl_with_rotation),
            ('mobile', self._crawl_mobile_version),
            ('proxy', self._crawl_with_proxy)
        ]
        
        for strategy_name, strategy_func in strategies:
            try:
                print(f"🔄 Trying {strategy_name} strategy...")
                result = await strategy_func(keyword, location)
                
                if result and 'error' not in result and result.get('organic_results'):
                    print(f"✅ {strategy_name} strategy successful")
                    self.success_rate += 1
                    return self._process_results(result, keyword)
                    
            except Exception as e:
                print(f"❌ {strategy_name} failed: {str(e)}")
                continue
        
        return {'keyword': keyword, 'data': {}, 'error': 'All strategies failed'}
    
    async def _crawl_with_rotation(self, keyword: str, location: str) -> Dict:
        """Strategy 1: User-Agent + Headers rotation"""
        
        headers = {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'vi-VN,vi;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }
        
        # Smart delay
        delay = random.uniform(3, 8)
        await asyncio.sleep(delay)
        
        params = {
            'q': keyword,
            'hl': 'vi',
            'gl': location,
            'num': 20,
            'start': 0
        }
        
        session = random.choice(self.session_pool)
        
        async with session.get('https://www.google.com/search', 
                              params=params, headers=headers) as response:
            if response.status == 200:
                html = await response.text()
                return self._parse_google_html(html)
            else:
                raise Exception(f'HTTP {response.status}')
    
    async def _crawl_mobile_version(self, keyword: str, location: str) -> Dict:
        """Strategy 2: Mobile Google version"""
        
        mobile_headers = {
            'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'vi-VN,vi;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive'
        }
        
        params = {
            'q': keyword,
            'hl': 'vi',
            'gl': location,
            'num': 10
        }
        
        session = random.choice(self.session_pool)
        
        await asyncio.sleep(random.uniform(2, 5))
        
        async with session.get('https://www.google.com/search',
                              params=params, 
                              headers=mobile_headers) as response:
            
            if response.status == 200:
                html = await response.text()
                return self._parse_google_html(html)
            else:
                raise Exception(f'Mobile crawl failed: {response.status}')
    
    async def _crawl_with_proxy(self, keyword: str, location: str) -> Dict:
        """Strategy 3: Free proxy rotation"""
        
        if not self.proxy_list:
            raise Exception('No proxies available')
        
        proxy = random.choice(self.proxy_list)
        
        headers = {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml',
            'Accept-Language': 'vi,en;q=0.9'
        }
        
        connector = aiohttp.TCPConnector()
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=20)
        ) as session:
            
            proxy_url = f"http://{proxy['ip']}:{proxy['port']}"
            params = {'q': keyword, 'hl': 'vi', 'gl': location}
            
            async with session.get('https://www.google.com/search',
                                  params=params, 
                                  headers=headers,
                                  proxy=proxy_url) as response:
                
                if response.status == 200:
                    html = await response.text()
                    return self._parse_google_html(html)
                else:
                    raise Exception(f'Proxy failed: {response.status}')
    
    async def _get_free_proxies(self) -> List[Dict]:
        """Get free proxy list"""
        try:
            proxies = []
            
            # Free proxy API
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get('https://www.proxy-list.download/api/v1/get?type=http') as response:
                        if response.status == 200:
                            text = await response.text()
                            for line in text.strip().split('\n')[:5]:
                                if ':' in line:
                                    ip, port = line.strip().split(':')
                                    proxies.append({'ip': ip, 'port': port})
            except:
                pass
            
            return proxies
            
        except:
            return []
    
    def _parse_google_html(self, html: str) -> Dict:
        """Parse Google HTML với comprehensive SERP features"""
        soup = BeautifulSoup(html, 'html.parser')
        
        results = {
            'organic_results': [],
            'featured_snippet': None,
            'people_also_ask': [],
            'video_results': [],
            'local_pack': [],
            'ads': [],
            'related_searches': []
        }
        
        # Parse organic results
        for i, g in enumerate(soup.select('.g, .tF2Cxc, .hlcw0c')[:10]):
            title_elem = g.select_one('h3')
            url_elem = g.select_one('a')
            desc_elem = g.select_one('.VwiC3b, .s, .st')
            
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
        
        # Parse featured snippet
        for selector in ['.xpdopen', '.kp-blk', '.IZ6rdc', '.g-blk']:
            featured = soup.select_one(selector)
            if featured:
                results['featured_snippet'] = {
                    'text': featured.get_text(strip=True)[:300],
                    'type': 'paragraph'
                }
                break
        
        # Parse People Also Ask
        for elem in soup.select('.related-question-pair, .JlqpRe, .cbphWd')[:5]:
            question = elem.get_text(strip=True)
            if question and len(question) > 10:
                results['people_also_ask'].append(question)
        
        # Parse video results
        for elem in soup.select('.RzdJxc, .P94G9b')[:3]:
            title_elem = elem.select_one('h3, .fc9yUc')
            if title_elem:
                results['video_results'].append({
                    'title': title_elem.get_text(strip=True),
                    'source': 'YouTube'
                })
        
        # Parse local pack
        for elem in soup.select('.rllt__details')[:3]:
            name_elem = elem.select_one('.dbg0pd')
            if name_elem:
                results['local_pack'].append({
                    'name': name_elem.get_text(strip=True)
                })
        
        # Parse ads
        ad_elements = soup.select('.uEierd, .v5yQqb')
        results['ads'] = [{'position': i+1} for i in range(min(len(ad_elements), 4))]
        
        return results
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc.lower().replace('www.', '')
        except:
            return 'unknown'
    
    def _process_results(self, results: Dict, keyword: str) -> Dict:
        """Process and consolidate results"""
        return {
            'keyword': keyword,
            'timestamp': datetime.now().isoformat(),
            'data': results,
            'sources_count': 1,
            'method': 'free_crawl'
        }
    
    async def cleanup(self):
        """Cleanup sessions"""
        for session in self.session_pool:
            await session.close()