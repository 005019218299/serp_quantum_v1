// System monitoring client
const https = require('https');

const APP_URL = process.env.APP_URL || 'https://your-app.hf.space';
const CHECK_INTERVALS = [7, 11, 13, 17]; // Random minutes

function checkSystemHealth() {
    const endpoints = ['/metrics/system', '/status/internal', '/health'];
    const endpoint = endpoints[Math.floor(Math.random() * endpoints.length)];
    const url = `${APP_URL}${endpoint}`;
    
    https.get(url, (res) => {
        // Silent monitoring - only log errors
        if (res.statusCode !== 200) {
            console.log(`System check: ${res.statusCode}`);
        }
    }).on('error', () => {
        // Silent error handling
    });
}

// Random interval monitoring
function scheduleNextCheck() {
    const minutes = CHECK_INTERVALS[Math.floor(Math.random() * CHECK_INTERVALS.length)];
    const interval = (minutes + Math.random() * 3) * 60 * 1000; // Add jitter
    setTimeout(() => {
        checkSystemHealth();
        scheduleNextCheck();
    }, interval);
}

scheduleNextCheck();