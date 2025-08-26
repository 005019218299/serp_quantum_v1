"""Hugging Face model configurations"""

HF_MODELS = {
    'multilingual_bert': {
        'model_name': 'bert-base-multilingual-cased',
        'cache_dir': '/app/hf_cache',
        'max_length': 512
    },
    'content_generator': {
        'model_name': 't5-small',
        'cache_dir': '/app/hf_cache',
        'max_length': 1024
    },
    'gpt_analyzer': {
        'model_name': 'gpt2-medium',
        'cache_dir': '/app/hf_cache',
        'max_length': 1024
    }
}

def get_model_config(model_key: str):
    return HF_MODELS.get(model_key, HF_MODELS['multilingual_bert'])