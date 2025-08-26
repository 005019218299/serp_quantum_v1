REGIONS = {
    'US': {
        'nodes': 20,
        'capacity_per_node': 50,
        'languages': ['en', 'es'],
        'redis_cluster': 'redis-us.cluster.local:6379'
    },
    'EU': {
        'nodes': 15, 
        'capacity_per_node': 50,
        'languages': ['en', 'fr', 'de', 'es'],
        'redis_cluster': 'redis-eu.cluster.local:6379'
    },
    'ASIA': {
        'nodes': 25,
        'capacity_per_node': 40, 
        'languages': ['en', 'zh', 'ja', 'ko', 'vi', 'th', 'id'],
        'redis_cluster': 'redis-asia.cluster.local:6379'
    }
}