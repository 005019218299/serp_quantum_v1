from enum import Enum
from typing import Dict, List
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from datetime import datetime

class DeviceType(Enum):
    DESKTOP = "desktop"
    MOBILE = "mobile"
    TABLET = "tablet"

class DeviceSpecificHarvester:
    def __init__(self):
        self.device_configs = {
            DeviceType.DESKTOP: {
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "viewport": {"width": 1920, "height": 1080}
            },
            DeviceType.MOBILE: {
                "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15",
                "viewport": {"width": 390, "height": 844}
            },
            DeviceType.TABLET: {
                "user_agent": "Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X) AppleWebKit/605.1.15",
                "viewport": {"width": 768, "height": 1024}
            }
        }
    
    async def fetch_serp_multi_device(self, keyword: str, location: str) -> Dict:
        """Fetch SERP results for all device types"""
        results = {}
        
        tasks = []
        for device_type in DeviceType:
            task = self._fetch_for_device(keyword, location, device_type)
            tasks.append((device_type, task))
        
        for device_type, task in tasks:
            try:
                device_data = await task
                results[device_type.value] = device_data
            except Exception as e:
                results[device_type.value] = {"error": str(e)}
            
        return {
            "keyword": keyword,
            "location": location,
            "timestamp": datetime.now().isoformat(),
            "device_results": results,
            "device_differences": self._analyze_device_differences(results)
        }
    
    async def _fetch_for_device(self, keyword: str, location: str, device_type: DeviceType) -> Dict:
        """Fetch SERP for specific device type"""
        config = self.device_configs[device_type]
        
        headers = {
            "User-Agent": config["user_agent"],
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }
        
        params = {
            "q": keyword,
            "location": location,
            "hl": "vi",
            "gl": "vn"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://www.google.com/search",
                headers=headers,
                params=params
            ) as response:
                html = await response.text()
                return self._parse_device_specific_serp(html, device_type)
    
    def _parse_device_specific_serp(self, html: str, device_type: DeviceType) -> Dict:
        """Parse device-specific SERP features"""
        soup = BeautifulSoup(html, 'html.parser')
        results = {"device_type": device_type.value}
        
        # Organic results
        results['organic_results'] = []
        for g in soup.select('.g')[:10]:
            title_elem = g.select_one('h3')
            url_elem = g.select_one('a')
            if title_elem and url_elem:
                results['organic_results'].append({
                    'title': title_elem.text,
                    'url': url_elem.get('href')
                })
        
        # Device-specific features
        if device_type == DeviceType.MOBILE:
            # AMP carousel
            results['amp_carousel'] = len(soup.select('[data-amp]')) > 0
            # Mobile-specific local pack
            results['local_pack'] = len(soup.select('.rllt__details')) > 0
        
        # Featured snippet
        results['featured_snippet'] = len(soup.select('.xpdopen')) > 0
        
        # Video results
        results['video_results'] = len(soup.select('.RzdJxc')) > 0
        
        return results
    
    def _analyze_device_differences(self, results: Dict) -> Dict:
        """Analyze differences between device results"""
        differences = {}
        
        if 'mobile' in results and 'desktop' in results:
            mobile_data = results['mobile']
            desktop_data = results['desktop']
            
            # Compare organic results count
            mobile_organic = len(mobile_data.get('organic_results', []))
            desktop_organic = len(desktop_data.get('organic_results', []))
            
            differences['organic_count_diff'] = mobile_organic - desktop_organic
            
            # Compare features
            features = ['featured_snippet', 'video_results', 'local_pack']
            differences['feature_differences'] = {}
            
            for feature in features:
                mobile_has = mobile_data.get(feature, False)
                desktop_has = desktop_data.get(feature, False)
                
                if mobile_has != desktop_has:
                    differences['feature_differences'][feature] = {
                        'mobile': mobile_has,
                        'desktop': desktop_has
                    }
        
        return differences