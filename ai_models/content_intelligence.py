import asyncio
from typing import Dict, List
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
from datetime import datetime

class ContentIntelligenceAI:
    def __init__(self):
        from .lazy_model_loader import lazy_loader
        self.lazy_loader = lazy_loader
        self.model_name = "t5-small"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
    
    def _ensure_model_loaded(self):
        """Load model chỉ khi cần"""
        if self.model is None:
            self.model, self.tokenizer = self.lazy_loader.get_model_and_tokenizer(self.model_name, 't5')
            if self.model:
                self.model.to(self.device)
    
    async def generate_serp_optimized_content(self, keyword: str, competitor_analysis: Dict) -> Dict:
        """Generate content optimized for SERP features"""
        content_types = {
            'featured_snippet': await self._generate_snippet_content(keyword, competitor_analysis),
            'faq_content': await self._generate_faq_content(keyword),
            'meta_description': await self._generate_meta_description(keyword),
            'title_tags': await self._generate_title_variations(keyword)
        }
        
        return {
            'keyword': keyword,
            'generated_content': content_types,
            'optimization_score': self._calculate_optimization_score(content_types),
            'serp_mapping': self._map_content_to_serp_features(content_types),
            'implementation_priority': self._prioritize_content_implementation(content_types)
        }
    
    async def _generate_snippet_content(self, keyword: str, competitor_analysis: Dict) -> Dict:
        """Generate featured snippet optimized content"""
        self._ensure_model_loaded()
        if not self.model or not self.tokenizer:
            return {'error': 'Model not available'}
            
        prompt = f"Create a concise answer for: {keyword}. Include definition, benefits, and key points."
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = inputs.to(self.device)
        
        with torch.inference_mode():
            outputs = self.model.generate(
                inputs,
                max_length=150,
                num_return_sequences=3,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        snippets = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
        return {
            'type': 'featured_snippet',
            'variations': snippets,
            'word_count': [len(s.split()) for s in snippets],
            'optimization_tips': [
                "Use numbered lists or bullet points",
                "Keep answer under 50 words",
                "Include the question in H2 tag"
            ]
        }
    
    async def _generate_faq_content(self, keyword: str) -> Dict:
        """Generate FAQ content for People Also Ask"""
        common_questions = [
            f"What is {keyword}?",
            f"How does {keyword} work?",
            f"Why choose {keyword}?",
            f"Where to buy {keyword}?",
            f"When to use {keyword}?"
        ]
        
        faq_pairs = []
        # Batch process for better performance
        prompts = [f"Answer concisely: {q}" for q in common_questions]
        
        for i, prompt in enumerate(prompts):
            try:
                inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=256, truncation=True)
                inputs = inputs.to(self.device)
                
                with torch.inference_mode():
                    outputs = self.model.generate(
                        inputs,
                        max_length=100,
                        temperature=0.6,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer = answer[len(prompt):].strip()  # Remove prompt from answer
                faq_pairs.append({'question': common_questions[i], 'answer': answer})
            except Exception as e:
                # Fallback answer
                faq_pairs.append({'question': common_questions[i], 'answer': f'Information about {keyword}'})
        
        return {
            'type': 'faq_content',
            'faq_pairs': faq_pairs,
            'schema_markup': self._generate_faq_schema(faq_pairs),
            'optimization_tips': [
                "Use FAQ schema markup",
                "Keep answers under 300 characters",
                "Include related questions"
            ]
        }
    
    async def _generate_meta_description(self, keyword: str) -> Dict:
        """Generate SEO meta descriptions"""
        prompt = f"Write a compelling meta description for {keyword} (max 160 characters)"
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=256, truncation=True)
        inputs = inputs.to(self.device)
        
        with torch.inference_mode():
            outputs = self.model.generate(
                inputs,
                max_length=80,  # Increased for proper meta descriptions
                num_return_sequences=3,  # Reduced for performance
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        descriptions = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
        return {
            'type': 'meta_description',
            'variations': descriptions,
            'character_counts': [len(d) for d in descriptions],
            'ctr_optimization': self._analyze_ctr_potential(descriptions)
        }
    
    async def _generate_title_variations(self, keyword: str) -> Dict:
        """Generate title tag variations"""
        title_formats = [
            f"Best {keyword} - Complete Guide 2024",
            f"{keyword}: Everything You Need to Know",
            f"Top {keyword} Reviews & Buying Guide",
            f"How to Choose the Right {keyword}",
            f"{keyword} - Benefits, Features & Pricing"
        ]
        
        return {
            'type': 'title_tags',
            'variations': title_formats,
            'character_counts': [len(t) for t in title_formats],
            'keyword_placement': ['beginning', 'beginning', 'middle', 'middle', 'beginning']
        }
    
    def _calculate_optimization_score(self, content_types: Dict) -> float:
        """Calculate overall content optimization score"""
        scores = []
        
        # Featured snippet score
        if 'featured_snippet' in content_types:
            snippet_data = content_types['featured_snippet']
            avg_length = np.mean(snippet_data['word_count'])
            snippet_score = 1.0 if 30 <= avg_length <= 50 else 0.7
            scores.append(snippet_score)
        
        # FAQ score
        if 'faq_content' in content_types:
            faq_data = content_types['faq_content']
            faq_score = min(len(faq_data['faq_pairs']) / 5.0, 1.0)
            scores.append(faq_score)
        
        # Meta description score
        if 'meta_description' in content_types:
            meta_data = content_types['meta_description']
            valid_lengths = [1 for count in meta_data['character_counts'] if 120 <= count <= 160]
            meta_score = len(valid_lengths) / len(meta_data['character_counts'])
            scores.append(meta_score)
        
        return np.mean(scores) if scores else 0.5
    
    def _map_content_to_serp_features(self, content_types: Dict) -> Dict:
        """Map generated content to SERP features"""
        return {
            'featured_snippet': content_types.get('featured_snippet', {}).get('variations', []),
            'people_also_ask': content_types.get('faq_content', {}).get('faq_pairs', []),
            'title_optimization': content_types.get('title_tags', {}).get('variations', []),
            'meta_optimization': content_types.get('meta_description', {}).get('variations', [])
        }
    
    def _prioritize_content_implementation(self, content_types: Dict) -> List[Dict]:
        """Prioritize content implementation based on impact"""
        priorities = []
        
        if 'featured_snippet' in content_types:
            priorities.append({
                'content_type': 'featured_snippet',
                'priority': 'high',
                'estimated_impact': 4.2,
                'implementation_effort': 'medium'
            })
        
        if 'faq_content' in content_types:
            priorities.append({
                'content_type': 'faq_content',
                'priority': 'medium',
                'estimated_impact': 2.8,
                'implementation_effort': 'low'
            })
        
        return sorted(priorities, key=lambda x: x['estimated_impact'], reverse=True)
    
    def _generate_faq_schema(self, faq_pairs: List[Dict]) -> Dict:
        """Generate FAQ schema markup"""
        schema = {
            "@context": "https://schema.org",
            "@type": "FAQPage",
            "mainEntity": []
        }
        
        for faq in faq_pairs:
            schema["mainEntity"].append({
                "@type": "Question",
                "name": faq['question'],
                "acceptedAnswer": {
                    "@type": "Answer",
                    "text": faq['answer']
                }
            })
        
        return schema
    
    def _analyze_ctr_potential(self, descriptions: List[str]) -> Dict:
        """Analyze CTR potential of meta descriptions"""
        ctr_keywords = ['best', 'top', 'guide', 'review', '2024', 'free', 'easy']
        
        scores = []
        for desc in descriptions:
            desc_lower = desc.lower()
            keyword_count = sum(1 for keyword in ctr_keywords if keyword in desc_lower)
            ctr_score = min(keyword_count / 3.0, 1.0)
            scores.append(ctr_score)
        
        return {
            'average_ctr_potential': np.mean(scores),
            'best_performing_index': np.argmax(scores),
            'optimization_suggestions': [
                "Include power words like 'best', 'top', 'guide'",
                "Add current year for freshness",
                "Include emotional triggers"
            ]
        }