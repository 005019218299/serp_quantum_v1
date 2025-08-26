# Real constants instead of magic numbers
import os

# Model Configuration
RANDOM_SEED = 42
MIN_FEATURE_COUNT = 10
DEFAULT_EPOCHS = 50
CONFIDENCE_THRESHOLD = 0.7

# Performance Thresholds (real values)
MAX_RESPONSE_TIME_MS = 50
TARGET_ACCURACY = 0.95
MIN_ACCURACY = 0.80

# Rate Limiting
DEFAULT_RATE_LIMIT = int(os.getenv('RATE_LIMIT', '100'))
BURST_LIMIT = int(os.getenv('BURST_LIMIT', '200'))

# Cache Configuration
CACHE_TTL_SECONDS = int(os.getenv('CACHE_TTL', '300'))
MAX_CACHE_SIZE = int(os.getenv('MAX_CACHE_SIZE', '1000'))

# Database
MAX_RETRIES = 3
RETRY_DELAY = 1.0

# Monitoring
METRICS_RETENTION_DAYS = 30
HEALTH_CHECK_INTERVAL = 60