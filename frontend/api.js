// ===========================
// API Client for Backend Communication
// ===========================

class GolfAPI {
    constructor() {
        this.baseUrl = '';  // Same origin
        this.isTraining = false;
        this.pollingInterval = null;

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.updateStatusIndicator('ready');
    }

    setupEventListeners() {
        document.getElementById('trainBtn').addEventListener('click', () => this.startTraining());
        document.getElementById('stopBtn').addEventListener('click', () => this.stopTraining());
    }

    async startTraining() {
        const learningRate = parseFloat(document.getElementById('learningRate').value);
        const totalTimesteps = parseInt(document.getElementById('totalTimesteps').value);

        try {
            this.setButtonState(true);
            this.updateStatusIndicator('training');
            document.getElementById('trainingStatus').textContent = 'Starting training...';

            const response = await fetch('/api/train/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    learning_rate: learningRate,
                    total_timesteps: totalTimesteps
                })
            });

            if (!response.ok) {
                throw new Error('Failed to start training');
            }

            const data = await response.json();
            console.log('Training started:', data);

            this.isTraining = true;
            this.startPolling();

        } catch (error) {
            console.error('Error starting training:', error);
            this.updateStatusIndicator('error');
            document.getElementById('trainingStatus').textContent = 'Error: ' + error.message;
            this.setButtonState(false);
        }
    }

    async stopTraining() {
        try {
            const response = await fetch('/api/train/stop', {
                method: 'POST'
            });

            if (!response.ok) {
                throw new Error('Failed to stop training');
            }

            const data = await response.json();
            console.log('Training stopped:', data);

            this.isTraining = false;
            this.stopPolling();
            this.setButtonState(false);
            this.updateStatusIndicator('ready');
            document.getElementById('trainingStatus').textContent = 'Training stopped';

        } catch (error) {
            console.error('Error stopping training:', error);
        }
    }

    startPolling() {
        this.pollingInterval = setInterval(() => this.checkStatus(), 1000);
    }

    stopPolling() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
        }
    }

    async checkStatus() {
        try {
            const response = await fetch('/api/train/status');

            if (!response.ok) {
                throw new Error('Failed to get status');
            }

            const data = await response.json();
            this.updateTrainingUI(data);

            if (!data.is_training && this.isTraining) {
                this.isTraining = false;
                this.stopPolling();
                this.setButtonState(false);
                this.updateStatusIndicator('ready');
                document.getElementById('trainingStatus').textContent = 'Training complete!';
            }

        } catch (error) {
            console.error('Error checking status:', error);
        }
    }

    updateTrainingUI(data) {
        document.getElementById('episodeCount').textContent = data.episodes || 0;
        document.getElementById('avgReward').textContent = (data.mean_reward || 0).toFixed(2);

        const progress = data.progress || 0;
        document.getElementById('progressBar').style.width = progress + '%';

        if (data.is_training) {
            document.getElementById('trainingStatus').textContent = `Training... ${progress.toFixed(1)}%`;
        }
    }

    async predict() {
        if (!game || !game.isAIMode || game.ball.isMoving || !game.isPlaying) {
            return null;
        }

        const state = game.getState();

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(state)
            });

            if (!response.ok) {
                throw new Error('Failed to get prediction');
            }

            const data = await response.json();
            return data;

        } catch (error) {
            console.error('Error getting prediction:', error);
            return null;
        }
    }

    async runAIMode() {
        if (!game.isAIMode || game.ball.isMoving || !game.isPlaying) {
            return;
        }

        const prediction = await this.predict();

        if (prediction && prediction.action) {
            const angle = prediction.action.angle;
            const power = prediction.action.power;
            game.aiShoot(angle, power);
        }
    }

    setButtonState(isTraining) {
        document.getElementById('trainBtn').disabled = isTraining;
        document.getElementById('stopBtn').disabled = !isTraining;
    }

    updateStatusIndicator(status) {
        const indicator = document.getElementById('statusIndicator');
        indicator.className = 'status-indicator ' + status;
    }
}

// Initialize API client
const api = new GolfAPI();

// AI auto-play loop
setInterval(async () => {
    if (game && game.isAIMode && !game.ball.isMoving && game.isPlaying) {
        await api.runAIMode();
    }
}, 1500);
