import asyncio
import aiohttp
from typing import Dict, List
import re
from transformers import pipeline
import torch

class VoiceSearchAnalyzer:
    def __init__(self):
        self.nlp = pipeline("text-classification", model="microsoft/DialoGPT-medium")
        self.voice_patterns = {
            'question_words': ['what', 'how', 'where', 'when', 'why', 'who'],
            'conversational': ['please', 'can you', 'help me', 'i need'],
            'local_intent': ['near me', 'nearby', 'around here', 'close to']
        }
    
    async def analyze_voice_intent(self, query: str) -> Dict:
        """Analyze voice search intent and patterns"""
        query_lower = query.lower()
        
        intent_scores = {
            'informational': self._calculate_informational_score(query_lower),
            'navigational': self._calculate_navigational_score(query_lower),
            'transactional': self._calculate_transactional_score(query_lower),
            'local': self._calculate_local_score(query_lower)
        }
        
        primary_intent = max(intent_scores, key=intent_scores.get)
        
        return {
            'query': query,
            'primary_intent': primary_intent,
            'intent_scores': intent_scores,
            'voice_characteristics': self._analyze_voice_characteristics(query_lower),
            'featured_snippet_potential': self._assess_snippet_potential(query_lower),
            'optimization_suggestions': self._generate_voice_optimizations(query_lower, primary_intent)
        }
    
    def _calculate_informational_score(self, query: str) -> float:
        score = 0.0
        for word in self.voice_patterns['question_words']:
            if word in query:
                score += 0.3
        if '?' in query:
            score += 0.2
        return min(score, 1.0)
    
    def _calculate_navigational_score(self, query: str) -> float:
        nav_keywords = ['website', 'site', 'homepage', 'official']
        score = sum(0.25 for word in nav_keywords if word in query)
        return min(score, 1.0)
    
    def _calculate_transactional_score(self, query: str) -> float:
        trans_keywords = ['buy', 'purchase', 'order', 'price', 'cost', 'cheap']
        score = sum(0.2 for word in trans_keywords if word in query)
        return min(score, 1.0)
    
    def _calculate_local_score(self, query: str) -> float:
        score = 0.0
        for phrase in self.voice_patterns['local_intent']:
            if phrase in query:
                score += 0.4
        return min(score, 1.0)
    
    def _analyze_voice_characteristics(self, query: str) -> Dict:
        return {
            'is_conversational': any(phrase in query for phrase in self.voice_patterns['conversational']),
            'is_question': query.endswith('?') or any(word in query for word in self.voice_patterns['question_words']),
            'word_count': len(query.split()),
            'natural_language': len(query.split()) > 3
        }
    
    def _assess_snippet_potential(self, query: str) -> float:
        """Assess potential for featured snippet optimization"""
        snippet_indicators = ['how to', 'what is', 'best way', 'steps to']
        score = sum(0.25 for phrase in snippet_indicators if phrase in query)
        return min(score, 1.0)
    
    def _generate_voice_optimizations(self, query: str, intent: str) -> List[str]:
        suggestions = []
        
        if intent == 'informational':
            suggestions.extend([
                "Create FAQ-style content",
                "Use natural language in headings",
                "Optimize for question-based queries"
            ])
        
        if intent == 'local':
            suggestions.extend([
                "Optimize Google Business Profile",
                "Include location-specific content",
                "Use local schema markup"
            ])
        
        return suggestions