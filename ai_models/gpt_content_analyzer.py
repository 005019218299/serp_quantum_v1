import asyncio
import aiohttp
import json
from typing import Dict, List, Optional
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class GPTContentAnalyzer:
    def __init__(self):
        # Use smaller GPT model for 16GB RAM constraint
        self.model_name = "gpt2-medium"  # 355M parameters, ~1.4GB
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Add padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    async def analyze_content_quality(self, content: str, keyword: str) -> Dict:
        """Analyze content quality using GPT"""
        try:
            quality_metrics = {
                'semantic_relevance': await self._calculate_semantic_relevance(content, keyword),
                'readability_score': self._calculate_readability(content),
                'content_depth': self._assess_content_depth(content),
                'seo_optimization': self._analyze_seo_factors(content, keyword),
                'improvement_suggestions': await self._generate_improvements(content, keyword)
            }
            
            overall_score = (
                quality_metrics['semantic_relevance'] * 0.3 +
                quality_metrics['readability_score'] * 0.2 +
                quality_metrics['content_depth'] * 0.3 +
                quality_metrics['seo_optimization'] * 0.2
            )
            
            return {
                'overall_quality_score': round(overall_score, 2),
                'metrics': quality_metrics,
                'content_length': len(content.split()),
                'keyword_density': self._calculate_keyword_density(content, keyword)
            }
        except Exception as e:
            return {'error': str(e), 'overall_quality_score': 0.5}
    
    async def generate_optimized_content(self, keyword: str, content_type: str, target_length: int = 500) -> Dict:
        """Generate SEO-optimized content"""
        try:
            prompts = {
                'blog_post': f"Write a comprehensive blog post about {keyword}. Include practical tips and examples.",
                'faq': f"Create frequently asked questions and answers about {keyword}.",
                'product_description': f"Write a detailed product description for {keyword}.",
                'meta_description': f"Write an SEO meta description for {keyword} (max 160 characters)."
            }
            
            prompt = prompts.get(content_type, prompts['blog_post'])
            generated_content = await self._generate_text(prompt, target_length)
            
            return {
                'content_type': content_type,
                'generated_content': generated_content,
                'word_count': len(generated_content.split()),
                'seo_score': self._analyze_seo_factors(generated_content, keyword),
                'optimization_tips': self._get_optimization_tips(content_type)
            }
        except Exception as e:
            return {'error': str(e)}
    
    async def _calculate_semantic_relevance(self, content: str, keyword: str) -> float:
        """Calculate semantic relevance using GPT embeddings"""
        try:
            # Simplified semantic analysis
            keyword_lower = keyword.lower()
            content_lower = content.lower()
            
            # Basic keyword presence
            keyword_presence = keyword_lower in content_lower
            
            # Related terms (simplified)
            related_terms = self._get_related_terms(keyword_lower)
            related_presence = sum(1 for term in related_terms if term in content_lower)
            
            base_score = 0.3 if keyword_presence else 0.0
            related_score = min(related_presence * 0.1, 0.7)
            
            return min(base_score + related_score, 1.0)
        except:
            return 0.5
    
    def _calculate_readability(self, content: str) -> float:
        """Calculate readability score"""
        sentences = content.split('.')
        words = content.split()
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        
        # Simplified readability (ideal: 15-20 words per sentence)
        if 15 <= avg_sentence_length <= 20:
            return 1.0
        elif 10 <= avg_sentence_length <= 25:
            return 0.8
        else:
            return 0.6
    
    def _assess_content_depth(self, content: str) -> float:
        """Assess content depth and comprehensiveness"""
        word_count = len(content.split())
        
        # Depth indicators
        depth_indicators = ['example', 'because', 'however', 'therefore', 'additionally', 'furthermore']
        depth_score = sum(1 for indicator in depth_indicators if indicator in content.lower())
        
        # Length factor
        length_factor = min(word_count / 1000, 1.0)  # Normalize to 1000 words
        
        # Combine factors
        return min((depth_score * 0.1) + (length_factor * 0.5), 1.0)
    
    def _analyze_seo_factors(self, content: str, keyword: str) -> float:
        """Analyze SEO optimization factors"""
        score = 0.0
        content_lower = content.lower()
        keyword_lower = keyword.lower()
        
        # Keyword in content
        if keyword_lower in content_lower:
            score += 0.3
        
        # Keyword density (2-4% is ideal)
        density = self._calculate_keyword_density(content, keyword)
        if 0.02 <= density <= 0.04:
            score += 0.3
        elif 0.01 <= density <= 0.06:
            score += 0.2
        
        # Content length (300+ words)
        word_count = len(content.split())
        if word_count >= 300:
            score += 0.2
        elif word_count >= 150:
            score += 0.1
        
        # Headers (simplified check)
        if any(indicator in content for indicator in ['#', 'H1', 'H2', 'H3']):
            score += 0.2
        
        return min(score, 1.0)
    
    async def _generate_improvements(self, content: str, keyword: str) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []
        
        word_count = len(content.split())
        keyword_density = self._calculate_keyword_density(content, keyword)
        
        if word_count < 300:
            suggestions.append("Increase content length to at least 300 words")
        
        if keyword_density < 0.01:
            suggestions.append(f"Include the keyword '{keyword}' more frequently")
        elif keyword_density > 0.05:
            suggestions.append(f"Reduce keyword density for '{keyword}' to avoid over-optimization")
        
        if keyword.lower() not in content.lower():
            suggestions.append(f"Include the main keyword '{keyword}' in the content")
        
        return suggestions
    
    async def _generate_text(self, prompt: str, max_length: int) -> str:
        """Generate text using GPT model"""
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=min(max_length + len(inputs[0]), 1024),
                    num_return_sequences=1,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text[len(prompt):].strip()
        except Exception as e:
            return f"Content generation failed: {str(e)}"
    
    def _calculate_keyword_density(self, content: str, keyword: str) -> float:
        """Calculate keyword density"""
        words = content.lower().split()
        keyword_lower = keyword.lower()
        keyword_count = words.count(keyword_lower)
        return keyword_count / max(len(words), 1)
    
    def _get_related_terms(self, keyword: str) -> List[str]:
        """Get related terms (simplified)"""
        # This would ideally use a more sophisticated semantic similarity model
        related_map = {
            'máy lọc nước': ['water filter', 'purifier', 'filtration', 'clean water'],
            'seo': ['search engine', 'optimization', 'ranking', 'google'],
            'marketing': ['advertising', 'promotion', 'brand', 'customer']
        }
        return related_map.get(keyword, [])
    
    def _get_optimization_tips(self, content_type: str) -> List[str]:
        """Get content type specific optimization tips"""
        tips = {
            'blog_post': [
                "Use H2 and H3 headings",
                "Include internal links",
                "Add relevant images with alt text",
                "Write compelling meta description"
            ],
            'faq': [
                "Use question format in headings",
                "Implement FAQ schema markup",
                "Keep answers concise but complete",
                "Include related questions"
            ],
            'product_description': [
                "Highlight key features and benefits",
                "Include technical specifications",
                "Add customer reviews section",
                "Use bullet points for readability"
            ]
        }
        return tips.get(content_type, tips['blog_post'])