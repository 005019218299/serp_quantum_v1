# 🌍 Distributed Training System

Hệ thống training phân tán cho SERP data với khả năng auto-scaling và fault-tolerant.

## 🚀 Tính năng chính

### ✨ Auto Keyword Discovery
- **Tự động phát hiện** trending keywords từ multiple sources
- **Multi-language support**: EN, ES, FR, DE, VI, ZH, JA, KO, TH, ID
- **Real-time trending**: Google Trends, Social Media, News
- **Personalized keywords** dựa trên user profile

### 🕷️ Distributed Crawling
- **Multi-region workers**: US, EU, Asia-Southeast
- **Auto load balancing** và fault tolerance
- **IP rotation** để tránh blocking
- **Parallel processing** với async/await

### 🧠 Central Training
- **Data aggregation** từ tất cả workers
- **Model training** tập trung với distributed data
- **Real-time monitoring** và progress tracking

### 📊 Web Dashboard
- **Real-time monitoring** system stats
- **Worker management** và health checks
- **Training cycle tracking**
- **Interactive controls**

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Master Coordinator                       │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐│
│  │   Auto Keyword  │ │   Distributed   │ │    Global       ││
│  │   Discovery     │ │   Worker Mgr    │ │   Training      ││
│  │                 │ │                 │ │  Orchestrator   ││
│  └─────────────────┘ └─────────────────┘ └─────────────────┘│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Worker A  │    │   Worker B  │    │   Worker C  │
│  (US-East)  │    │  (EU-West)  │    │ (Asia-SE)   │
│             │    │             │    │             │
│ • Crawling  │    │ • Crawling  │    │ • Crawling  │
│ • Keywords  │    │ • Keywords  │    │ • Keywords  │
│ • Upload    │    │ • Upload    │    │ • Upload    │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           ▼
              ┌─────────────────────────┐
              │    Shared Storage       │
              │  (Google Drive/S3/FTP)  │
              └─────────────────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │    Central Training     │
              │      Server             │
              └─────────────────────────┘
```

## 🛠️ Installation

### 1. Install Dependencies
```bash
cd quantum-serp-resonance/backend
pip install -r requirements_distributed.txt
```

### 2. Setup Redis (for coordination)
```bash
# Windows
# Download and install Redis from https://redis.io/download

# Linux/Mac
sudo apt-get install redis-server
# or
brew install redis
```

### 3. Setup PostgreSQL (optional, for persistent storage)
```bash
# Create database
createdb serp_global
```

## 🚀 Quick Start

### Method 1: Demo Mode
```bash
python run_distributed_system.py --mode demo
```

### Method 2: Web Server Mode
```bash
python run_distributed_system.py --mode server
```

Sau đó mở: http://localhost:8000

## 📡 API Endpoints

### Training Management
- `POST /api/training/start` - Start training cycle
- `GET /api/training/status/{cycle_id}` - Get cycle status
- `GET /api/training/cycles` - List all cycles

### Worker Management  
- `POST /api/workers/register` - Register new worker
- `GET /api/workers/list` - List all workers
- `GET /api/workers/stats` - Get worker statistics

### Keyword Discovery
- `POST /api/keywords/discover` - Discover trending keywords
- `GET /api/keywords/personalized/{user_id}` - Get personalized keywords

### System Monitoring
- `GET /api/system/overview` - System overview
- `GET /api/system/health` - Health check

## 💡 Usage Examples

### Start Training Cycle
```python
import requests

response = requests.post('http://localhost:8000/api/training/start', json={
    'user_id': 'user123',
    'seed_keywords': ['AI tools', 'machine learning', 'web development'],
    'target_regions': ['us-east', 'eu-west'],
    'max_keywords_per_worker': 50
})

cycle_id = response.json()['cycle_id']
print(f"Started cycle: {cycle_id}")
```

### Monitor Progress
```python
import requests
import time

while True:
    response = requests.get(f'http://localhost:8000/api/training/status/{cycle_id}')
    status = response.json()['status']
    
    print(f"Status: {status['status']}")
    
    if status['status'] in ['completed', 'failed']:
        break
    
    time.sleep(30)
```

### Register Worker
```python
import requests

response = requests.post('http://localhost:8000/api/workers/register', json={
    'region': 'us-east',
    'ip_address': '192.168.1.100',
    'capabilities': ['crawling', 'keyword_discovery']
})

worker_id = response.json()['worker_id']
print(f"Registered worker: {worker_id}")
```

## 🔧 Configuration

### Storage Options
```python
# Google Drive
storage_config = {
    'type': 'google_drive',
    'credentials': {
        'service_account_key': 'path/to/service-account.json',
        'folder_id': 'your_shared_folder_id'
    }
}

# Amazon S3
storage_config = {
    'type': 's3',
    'credentials': {
        'aws_access_key_id': 'your_access_key',
        'aws_secret_access_key': 'your_secret_key',
        'bucket_name': 'your_bucket'
    }
}

# FTP Server
storage_config = {
    'type': 'ftp',
    'credentials': {
        'host': 'ftp.example.com',
        'username': 'your_username',
        'password': 'your_password',
        'path': '/shared_data/'
    }
}
```

### Worker Regions
```python
regions = {
    'us-east': {
        'location': 'US East Coast',
        'proxy_pool': ['proxy1.us.com', 'proxy2.us.com'],
        'search_engines': ['google.com', 'bing.com']
    },
    'eu-west': {
        'location': 'Europe West',
        'proxy_pool': ['proxy1.eu.com', 'proxy2.eu.com'],
        'search_engines': ['google.de', 'google.fr']
    },
    'asia-southeast': {
        'location': 'Asia Southeast',
        'proxy_pool': ['proxy1.sg.com', 'proxy2.sg.com'],
        'search_engines': ['google.com.sg', 'google.co.th']
    }
}
```

## 📊 Performance

### Benchmarks
- **Single Server**: 50 keywords → 2-3 hours
- **3 Workers**: 50 keywords → 30-45 minutes  
- **9 Workers**: 150 keywords → 45-60 minutes
- **Scalability**: Linear scaling up to 50+ workers

### Cost Optimization
- **Worker Instances**: $5-10/month each (t3.micro)
- **Master Server**: $20-30/month (t3.medium)
- **Storage**: $5/month (100GB)
- **Total**: ~$50-100/month for 10 workers

## 🔍 Monitoring

### Web Dashboard
- Real-time worker status
- Training progress tracking
- System health monitoring
- Interactive controls

### Metrics
- Worker utilization
- Task completion rates
- Error rates by region
- Keyword discovery trends

### Alerts
- Worker failures
- Training job failures
- Storage issues
- Performance degradation

## 🛡️ Security

### API Security
- Rate limiting
- API key authentication
- CORS protection
- Input validation

### Worker Security
- Encrypted communication
- Worker authentication
- Secure credential storage
- Network isolation

## 🚨 Troubleshooting

### Common Issues

**Workers not connecting:**
```bash
# Check Redis connection
redis-cli ping

# Check worker logs
tail -f worker.log
```

**Training jobs failing:**
```bash
# Check system health
curl http://localhost:8000/api/system/health

# Check worker stats
curl http://localhost:8000/api/workers/stats
```

**Storage upload issues:**
```bash
# Test storage connection
python -c "from services.distributed_worker_manager import DistributedWorkerManager; print('Storage OK')"
```

## 🔄 Scaling

### Horizontal Scaling
```python
# Add more workers
for i in range(5):  # Add 5 more workers
    worker_config = {
        'region': 'us-east',
        'ip_address': f'192.168.1.{100+i}',
        'capabilities': ['crawling', 'keyword_discovery']
    }
    worker_id = await manager.register_worker(worker_config)
```

### Vertical Scaling
```python
# Increase worker capacity
worker_config = {
    'max_keywords_per_worker': 100,  # Increase from 50
    'concurrent_requests': 10,       # Increase parallelism
    'memory_limit': '4GB'           # Increase memory
}
```

## 📈 Roadmap

### v1.1 (Next Month)
- [ ] Docker containerization
- [ ] Kubernetes deployment
- [ ] Advanced monitoring with Prometheus
- [ ] Auto-scaling based on queue size

### v1.2 (Q2 2024)
- [ ] Machine learning for keyword scoring
- [ ] Advanced proxy rotation
- [ ] Multi-cloud deployment
- [ ] Cost optimization algorithms

### v1.3 (Q3 2024)
- [ ] Real-time streaming data
- [ ] Advanced analytics dashboard
- [ ] Custom model architectures
- [ ] Enterprise features

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

## 📄 License

MIT License - see LICENSE file for details.

## 📞 Support

- **Documentation**: [README files](.)
- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Email**: support@yourcompany.com

---

**🎉 Happy Distributed Training!** 🚀