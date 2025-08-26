import json
import os
from typing import Optional, Dict, List
from datetime import datetime

class APIKeyManager:
    def __init__(self, keys_file: str = "config/api_keys.json"):
        self.keys_file = keys_file
        self.keys_data = self._load_keys()
    
    def _load_keys(self) -> Dict:
        """Load API keys from JSON file"""
        try:
            with open(self.keys_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"serpapi_keys": [], "serpstack_keys": []}
    
    def _save_keys(self):
        """Save updated keys back to JSON file"""
        with open(self.keys_file, 'w') as f:
            json.dump(self.keys_data, f, indent=2)
    
    def get_active_key(self, key_type: str) -> Optional[str]:
        """Get next available API key"""
        key_list = f"{key_type}_keys"
        
        if key_list not in self.keys_data:
            return None
        
        # Find first active key with available requests
        for key_info in self.keys_data[key_list]:
            if (key_info.get("active", True) and 
                not key_info.get("expired", False) and
                key_info.get("request_current", 0) < key_info.get("request_limit", 100)):
                return key_info["value_key"]
        
        return None
    
    def increment_usage(self, key_type: str, api_key: str) -> bool:
        """Increment usage counter for API key"""
        key_list = f"{key_type}_keys"
        
        if key_list not in self.keys_data:
            return False
        
        for key_info in self.keys_data[key_list]:
            if key_info["value_key"] == api_key:
                key_info["request_current"] = key_info.get("request_current", 0) + 1
                
                # Mark as inactive if limit reached
                if key_info["request_current"] >= key_info.get("request_limit", 100):
                    key_info["active"] = False
                
                self._save_keys()
                return True
        
        return False
    
    def mark_key_failed(self, key_type: str, api_key: str):
        """Mark API key as failed/inactive"""
        key_list = f"{key_type}_keys"
        
        if key_list not in self.keys_data:
            return
        
        for key_info in self.keys_data[key_list]:
            if key_info["value_key"] == api_key:
                key_info["active"] = False
                key_info["expired"] = True
                self._save_keys()
                break
    
    def get_key_stats(self, key_type: str) -> Dict:
        """Get statistics for key type"""
        key_list = f"{key_type}_keys"
        
        if key_list not in self.keys_data:
            return {"total": 0, "active": 0, "expired": 0}
        
        keys = self.keys_data[key_list]
        total = len(keys)
        active = len([k for k in keys if k.get("active", True) and not k.get("expired", False)])
        expired = len([k for k in keys if k.get("expired", False)])
        
        return {"total": total, "active": active, "expired": expired}

# Global instance
api_key_manager = APIKeyManager()