// ===========================
// Golf Game Engine
// ===========================

class GolfGame {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        
        // Game dimensions
        this.width = this.canvas.width;
        this.height = this.canvas.height;
        
        // Game state
        this.isAIMode = false;
        this.isPlaying = true;
        this.shots = 0;
        this.bestScore = null;
        
        // Ball properties
        this.ball = {
            x: this.width / 2,
            y: this.height - 80,
            radius: 8,
            vx: 0,
            vy: 0,
            isMoving: false
        };
        
        // Hole properties
        this.hole = {
            x: this.width / 2,
            y: 80,
            radius: 14
        };
        
        // Aiming
        this.isDragging = false;
        this.dragStart = { x: 0, y: 0 };
        this.dragEnd = { x: 0, y: 0 };
        this.maxPower = 25;
        
        // Physics
        this.friction = 0.985;
        this.minVelocity = 0.1;
        
        // Visual elements
        this.flagPole = { height: 50 };
        
        // Initialize
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.gameLoop();
        this.updateUI();
    }
    
    setupEventListeners() {
        // Mouse events
        this.canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.handleMouseUp(e));
        this.canvas.addEventListener('mouseleave', (e) => this.handleMouseUp(e));
        
        // Touch events for mobile
        this.canvas.addEventListener('touchstart', (e) => this.handleTouchStart(e));
        this.canvas.addEventListener('touchmove', (e) => this.handleTouchMove(e));
        this.canvas.addEventListener('touchend', (e) => this.handleTouchEnd(e));
        
        // Mode buttons
        document.getElementById('humanModeBtn').addEventListener('click', () => this.setMode(false));
        document.getElementById('aiModeBtn').addEventListener('click', () => this.setMode(true));
        
        // Restart button
        document.getElementById('restartBtn').addEventListener('click', () => this.restart());
    }
    
    getMousePos(e) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    }
    
    handleMouseDown(e) {
        if (this.isAIMode || this.ball.isMoving || !this.isPlaying) return;
        
        const pos = this.getMousePos(e);
        const dist = Math.hypot(pos.x - this.ball.x, pos.y - this.ball.y);
        
        if (dist < 50) {
            this.isDragging = true;
            this.dragStart = { x: this.ball.x, y: this.ball.y };
            this.dragEnd = pos;
        }
    }
    
    handleMouseMove(e) {
        if (!this.isDragging) return;
        this.dragEnd = this.getMousePos(e);
    }
    
    handleMouseUp(e) {
        if (!this.isDragging) return;
        this.isDragging = false;
        this.shoot();
    }
    
    handleTouchStart(e) {
        e.preventDefault();
        const touch = e.touches[0];
        this.handleMouseDown({ clientX: touch.clientX, clientY: touch.clientY });
    }
    
    handleTouchMove(e) {
        e.preventDefault();
        const touch = e.touches[0];
        this.handleMouseMove({ clientX: touch.clientX, clientY: touch.clientY });
    }
    
    handleTouchEnd(e) {
        e.preventDefault();
        this.handleMouseUp(e);
    }
    
    shoot() {
        const dx = this.dragStart.x - this.dragEnd.x;
        const dy = this.dragStart.y - this.dragEnd.y;
        
        const power = Math.min(Math.hypot(dx, dy) / 10, this.maxPower);
        
        if (power < 0.5) return;
        
        const angle = Math.atan2(dy, dx);
        
        this.ball.vx = Math.cos(angle) * power;
        this.ball.vy = Math.sin(angle) * power;
        this.ball.isMoving = true;
        this.shots++;
        
        this.updateUI();
    }
    
    // AI 제어 메서드
    aiShoot(angle, power) {
        if (this.ball.isMoving || !this.isPlaying) return false;
        
        const normalizedPower = Math.min(Math.max(power, 0), 1) * this.maxPower;
        const radians = angle * Math.PI / 180;
        
        this.ball.vx = Math.cos(radians) * normalizedPower;
        this.ball.vy = Math.sin(radians) * normalizedPower;
        this.ball.isMoving = true;
        this.shots++;
        
        this.updateUI();
        return true;
    }
    
    update() {
        if (!this.ball.isMoving) return;
        
        // Apply velocity
        this.ball.x += this.ball.vx;
        this.ball.y += this.ball.vy;
        
        // Apply friction
        this.ball.vx *= this.friction;
        this.ball.vy *= this.friction;
        
        // Boundary collision
        if (this.ball.x - this.ball.radius < 0) {
            this.ball.x = this.ball.radius;
            this.ball.vx *= -0.8;
        }
        if (this.ball.x + this.ball.radius > this.width) {
            this.ball.x = this.width - this.ball.radius;
            this.ball.vx *= -0.8;
        }
        if (this.ball.y - this.ball.radius < 0) {
            this.ball.y = this.ball.radius;
            this.ball.vy *= -0.8;
        }
        if (this.ball.y + this.ball.radius > this.height) {
            this.ball.y = this.height - this.ball.radius;
            this.ball.vy *= -0.8;
        }
        
        // Stop if velocity is very low
        const speed = Math.hypot(this.ball.vx, this.ball.vy);
        if (speed < this.minVelocity) {
            this.ball.vx = 0;
            this.ball.vy = 0;
            this.ball.isMoving = false;
            
            // Check for hole
            this.checkHole();
        }
    }
    
    checkHole() {
        const dist = Math.hypot(this.ball.x - this.hole.x, this.ball.y - this.hole.y);
        
        if (dist < this.hole.radius) {
            this.isPlaying = false;
            this.showOverlay(true);
            
            if (this.bestScore === null || this.shots < this.bestScore) {
                this.bestScore = this.shots;
            }
            this.updateUI();
        }
    }
    
    showOverlay(success) {
        const overlay = document.getElementById('gameOverlay');
        const title = document.getElementById('overlayTitle');
        const shotCount = document.getElementById('shotCount');
        
        if (success) {
            if (this.shots === 1) {
                title.textContent = '🎉 Hole in One!';
            } else if (this.shots <= 3) {
                title.textContent = '⭐ Excellent!';
            } else {
                title.textContent = '✅ Complete!';
            }
        }
        
        shotCount.textContent = this.shots;
        overlay.classList.add('active');
    }
    
    hideOverlay() {
        document.getElementById('gameOverlay').classList.remove('active');
    }
    
    restart() {
        this.ball.x = this.width / 2;
        this.ball.y = this.height - 80;
        this.ball.vx = 0;
        this.ball.vy = 0;
        this.ball.isMoving = false;
        
        // Randomize hole position slightly
        this.hole.x = 100 + Math.random() * (this.width - 200);
        this.hole.y = 60 + Math.random() * 100;
        
        this.shots = 0;
        this.isPlaying = true;
        this.hideOverlay();
        this.updateUI();
    }
    
    setMode(isAI) {
        this.isAIMode = isAI;
        
        document.getElementById('humanModeBtn').classList.toggle('active', !isAI);
        document.getElementById('aiModeBtn').classList.toggle('active', isAI);
        
        const helpText = document.getElementById('helpText');
        helpText.textContent = isAI 
            ? 'AI will automatically control the ball after training!'
            : 'Drag from the ball to aim and release to shoot!';
        
        this.restart();
    }
    
    updateUI() {
        document.getElementById('currentShots').textContent = this.shots;
        
        const dist = Math.hypot(this.ball.x - this.hole.x, this.ball.y - this.hole.y);
        document.getElementById('distanceToHole').textContent = Math.round(dist / 10) + 'm';
        
        document.getElementById('bestScore').textContent = this.bestScore !== null ? this.bestScore : '-';
    }
    
    // Get current state for RL
    getState() {
        return {
            ball_x: this.ball.x / this.width,
            ball_y: this.ball.y / this.height,
            hole_x: this.hole.x / this.width,
            hole_y: this.hole.y / this.height,
            distance: Math.hypot(this.ball.x - this.hole.x, this.ball.y - this.hole.y) / Math.hypot(this.width, this.height)
        };
    }
    
    render() {
        const ctx = this.ctx;
        
        // Clear and draw background (golf course)
        this.drawCourse();
        
        // Draw hole
        this.drawHole();
        
        // Draw aim line
        if (this.isDragging && !this.ball.isMoving) {
            this.drawAimLine();
        }
        
        // Draw ball
        this.drawBall();
    }
    
    drawCourse() {
        const ctx = this.ctx;
        
        // Fairway gradient
        const gradient = ctx.createLinearGradient(0, 0, 0, this.height);
        gradient.addColorStop(0, '#15803d');
        gradient.addColorStop(0.5, '#16a34a');
        gradient.addColorStop(1, '#15803d');
        
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, this.width, this.height);
        
        // Rough edges
        ctx.fillStyle = '#14532d';
        ctx.fillRect(0, 0, 30, this.height);
        ctx.fillRect(this.width - 30, 0, 30, this.height);
        
        // Putting green area
        ctx.beginPath();
        ctx.ellipse(this.hole.x, this.hole.y, 80, 60, 0, 0, Math.PI * 2);
        ctx.fillStyle = '#22c55e';
        ctx.fill();
        
        // Green border
        ctx.beginPath();
        ctx.ellipse(this.hole.x, this.hole.y, 82, 62, 0, 0, Math.PI * 2);
        ctx.strokeStyle = '#16a34a';
        ctx.lineWidth = 3;
        ctx.stroke();
        
        // Tee area
        ctx.beginPath();
        ctx.ellipse(this.width / 2, this.height - 80, 40, 30, 0, 0, Math.PI * 2);
        ctx.fillStyle = '#22c55e';
        ctx.fill();
    }
    
    drawHole() {
        const ctx = this.ctx;
        
        // Hole shadow
        ctx.beginPath();
        ctx.arc(this.hole.x + 2, this.hole.y + 2, this.hole.radius, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
        ctx.fill();
        
        // Hole
        ctx.beginPath();
        ctx.arc(this.hole.x, this.hole.y, this.hole.radius, 0, Math.PI * 2);
        ctx.fillStyle = '#1a1a1a';
        ctx.fill();
        
        // Hole rim
        ctx.beginPath();
        ctx.arc(this.hole.x, this.hole.y, this.hole.radius + 2, 0, Math.PI * 2);
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Flag pole
        ctx.beginPath();
        ctx.moveTo(this.hole.x, this.hole.y);
        ctx.lineTo(this.hole.x, this.hole.y - this.flagPole.height);
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Flag
        ctx.beginPath();
        ctx.moveTo(this.hole.x, this.hole.y - this.flagPole.height);
        ctx.lineTo(this.hole.x + 25, this.hole.y - this.flagPole.height + 10);
        ctx.lineTo(this.hole.x, this.hole.y - this.flagPole.height + 20);
        ctx.closePath();
        ctx.fillStyle = '#ef4444';
        ctx.fill();
    }
    
    drawBall() {
        const ctx = this.ctx;
        
        // Ball shadow
        ctx.beginPath();
        ctx.ellipse(this.ball.x + 3, this.ball.y + 3, this.ball.radius, this.ball.radius * 0.6, 0, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
        ctx.fill();
        
        // Ball
        const ballGradient = ctx.createRadialGradient(
            this.ball.x - 2, this.ball.y - 2, 0,
            this.ball.x, this.ball.y, this.ball.radius
        );
        ballGradient.addColorStop(0, '#ffffff');
        ballGradient.addColorStop(0.5, '#f0f0f0');
        ballGradient.addColorStop(1, '#d0d0d0');
        
        ctx.beginPath();
        ctx.arc(this.ball.x, this.ball.y, this.ball.radius, 0, Math.PI * 2);
        ctx.fillStyle = ballGradient;
        ctx.fill();
        
        // Ball outline
        ctx.strokeStyle = '#bbb';
        ctx.lineWidth = 1;
        ctx.stroke();
    }
    
    drawAimLine() {
        const ctx = this.ctx;
        
        const dx = this.dragStart.x - this.dragEnd.x;
        const dy = this.dragStart.y - this.dragEnd.y;
        const power = Math.min(Math.hypot(dx, dy) / 10, this.maxPower);
        const powerRatio = power / this.maxPower;
        
        // Direction line (dotted)
        ctx.beginPath();
        ctx.setLineDash([5, 5]);
        ctx.moveTo(this.ball.x, this.ball.y);
        ctx.lineTo(this.ball.x + dx * 2, this.ball.y + dy * 2);
        
        // Color based on power
        const hue = 120 - powerRatio * 120; // Green to Red
        ctx.strokeStyle = `hsla(${hue}, 80%, 50%, 0.8)`;
        ctx.lineWidth = 3;
        ctx.stroke();
        ctx.setLineDash([]);
        
        // Power indicator
        ctx.beginPath();
        ctx.arc(this.dragEnd.x, this.dragEnd.y, 10, 0, Math.PI * 2);
        ctx.fillStyle = `hsla(${hue}, 80%, 50%, 0.6)`;
        ctx.fill();
        
        // Power text
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 14px Inter';
        ctx.textAlign = 'center';
        ctx.fillText(Math.round(powerRatio * 100) + '%', this.dragEnd.x, this.dragEnd.y - 20);
    }
    
    gameLoop() {
        this.update();
        this.render();
        requestAnimationFrame(() => this.gameLoop());
    }
}

// Initialize game
const game = new GolfGame('gameCanvas');
