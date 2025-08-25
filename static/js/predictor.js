class F1Predictor {
    constructor() {
        this.initializeElements();
        this.setupEventListeners();
        this.initializeSliders();
    }

    initializeElements() {
        // Form elements
        this.form = document.getElementById('predictionForm');
        this.resetBtn = document.getElementById('resetBtn');
        this.predictBtn = document.getElementById('predictBtn');
        this.resultsSection = document.getElementById('resultsSection');
        this.predictedTimeElement = document.getElementById('predictedTime');

        // Button states
        this.btnText = this.predictBtn?.querySelector('.btn-text');
        this.btnLoader = this.predictBtn?.querySelector('.btn-loader');

        // Slider configuration with defaults
        this.sliderConfig = {
            qualifying_time: { min: 69.0, max: 73.0, default: 70.5, unit: 's', decimals: 1 },
            rain_probability: { min: 0, max: 100, default: 20, unit: '%', decimals: 0 },
            temperature: { min: 15, max: 40, default: 25, unit: '°C', decimals: 0 },
            team_performance: { min: 0.0, max: 1.0, default: 0.70, unit: '', decimals: 2 },
            clean_air_pace: { min: 90.0, max: 100.0, default: 94.0, unit: 's', decimals: 1 },
            position_change: { min: -3.0, max: 3.0, default: 0.0, unit: '', decimals: 1 },
            sector_time: { min: 90.0, max: 100.0, default: 95.0, unit: 's', decimals: 1 }
        };

        // Get all sliders
        this.sliders = {};
        Object.keys(this.sliderConfig).forEach(name => {
            const input = document.getElementById(name);
            const valueDisplay = document.getElementById(`${name}_value`);
            const parameterGroup = input?.closest('.parameter-group');
            const trackFill = parameterGroup?.querySelector('.slider-track-fill');

            if (input && valueDisplay && trackFill) {
                this.sliders[name] = {
                    input: input,
                    valueDisplay: valueDisplay,
                    trackFill: trackFill
                };
            } else {
                console.error(`Missing elements for slider: ${name}`);
            }
        });
    }

    setupEventListeners() {
        // Form submission
        if (this.form) {
            this.form.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handlePredict(e);
            });
        }

        // Reset button
        if (this.resetBtn) {
            this.resetBtn.addEventListener('click', (e) => {
                e.preventDefault();
                this.resetValues();
            });
        }

        // Predict button
        if (this.predictBtn) {
            this.predictBtn.addEventListener('click', (e) => {
                e.preventDefault();
                this.handlePredict(e);
            });
        }

        // Slider events
        Object.keys(this.sliders).forEach(name => {
            const slider = this.sliders[name];
            if (slider && slider.input) {
                slider.input.addEventListener('input', () => this.updateSlider(name));
                slider.input.addEventListener('change', () => this.updateSlider(name));
                slider.input.addEventListener('mousemove', () => this.updateSlider(name));
                slider.input.addEventListener('touchmove', () => this.updateSlider(name), {passive: true});
            }
        });

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                e.preventDefault();
                this.handlePredict(e);
            }
            if (e.key === 'Escape') {
                e.preventDefault();
                this.resetValues();
            }
        });
    }

    initializeSliders() {
        Object.keys(this.sliderConfig).forEach(name => {
            const config = this.sliderConfig[name];
            const slider = this.sliders[name];
            if (slider && slider.input) {
                slider.input.value = config.default;
                this.updateSlider(name);
            }
        });
    }

    updateSlider(name) {
        const config = this.sliderConfig[name];
        const slider = this.sliders[name];
        
        if (!config || !slider || !slider.input || !slider.valueDisplay || !slider.trackFill) {
            return;
        }

        const value = parseFloat(slider.input.value);
        const formattedValue = value.toFixed(config.decimals);
        slider.valueDisplay.textContent = `${formattedValue}${config.unit}`;

        const percentage = ((value - config.min) / (config.max - config.min)) * 100;
        slider.trackFill.style.width = `${Math.max(0, Math.min(100, percentage))}%`;
    }

    resetValues() {
        Object.keys(this.sliders).forEach(name => {
            const config = this.sliderConfig[name];
            const slider = this.sliders[name];
            if (slider && slider.input && config) {
                slider.input.value = config.default;
                this.updateSlider(name);
            }
        });
        
        // Hide results section
        if (this.resultsSection) {
            this.resultsSection.classList.add('hidden');
        }
        
        // Remove podium section if exists
        const podiumSection = document.getElementById('podiumResultsSection');
        if (podiumSection) {
            podiumSection.remove();
        }
    }

    setLoadingState(loading) {
        if (this.predictBtn) {
            this.predictBtn.disabled = loading;
            this.predictBtn.classList.toggle('loading', loading);
        }
        
        if (this.btnText && this.btnLoader) {
            this.btnText.style.display = loading ? 'none' : 'block';
            this.btnLoader.style.display = loading ? 'block' : 'none';
        }
    }

    async delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    async handlePredict(e) {
        if (e) {
            e.preventDefault();
        }

        this.setLoadingState(true);

        try {
            const formData = this.getFormData();
            console.log('🚀 Sending prediction data:', formData);
            
            await this.delay(800); // Realistic loading time
            
            const prediction = await this.callMLPrediction(formData);
            console.log('✅ Received prediction:', prediction);
            
            this.displayResults(prediction);
        } catch (error) {
            console.error('❌ Prediction error:', error);
            this.showError('Failed to predict lap time. Please try again.');
        } finally {
            this.setLoadingState(false);
        }
    }

    getFormData() {
        const data = {};
        Object.keys(this.sliders).forEach(name => {
            if (this.sliders[name] && this.sliders[name].input) {
                data[name] = parseFloat(this.sliders[name].input.value);
            }
        });
        return data;
    }

    async callMLPrediction(data) {
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const responseData = await response.json();
            
            if (!responseData.success) {
                throw new Error(responseData.error || 'Prediction failed');
            }

            return {
                lapTime: responseData.predicted_lap_time,
                confidence: responseData.confidence,
                podium: responseData.podium || [],
                allPredictions: responseData.all_predictions || []
            };
        } catch (error) {
            console.error('🔥 API call failed:', error);
            throw error;
        }
    }

    async displayResults(prediction) {
        if (!this.predictedTimeElement || !this.resultsSection) {
            console.error('❌ Results elements not found');
            return;
        }

        // Format and display lap time
        const formattedTime = parseFloat(prediction.lapTime).toFixed(3);
        this.predictedTimeElement.textContent = `${formattedTime}s`;

        // Display podium if available
        if (prediction.podium && prediction.podium.length >= 3) {
            console.log('🏆 Displaying podium:', prediction.podium);
            this.displayPodiumResults(prediction.podium);
        }

        // Show results section
        this.resultsSection.classList.remove('hidden');

        // Scroll to results
        setTimeout(() => {
            this.resultsSection.scrollIntoView({
                behavior: 'smooth',
                block: 'center'
            });
        }, 300);
    }

    displayPodiumResults(podium) {
        console.log('🏁 Creating podium display for:', podium);
        
        // Remove existing podium section if it exists
        const existingPodium = document.getElementById('podiumResultsSection');
        if (existingPodium) {
            existingPodium.remove();
        }

        // Create new podium section
        const podiumSection = this.createPodiumSection(podium);
        
        // Insert podium section before results
        const resultsSection = document.getElementById('resultsSection');
        if (resultsSection && resultsSection.parentNode) {
            resultsSection.parentNode.insertBefore(podiumSection, resultsSection);
        } else {
            // Fallback: append to main container
            const mainContainer = document.querySelector('.prediction-container') || 
                                 document.querySelector('main') || 
                                 document.body;
            mainContainer.appendChild(podiumSection);
        }

        // Animate podium appearance
        setTimeout(() => {
            podiumSection.classList.add('visible');
        }, 100);
    }

    createPodiumSection(podium) {
        const section = document.createElement('div');
        section.id = 'podiumResultsSection';
        section.className = 'podium-results-container';
        
        const top3 = podium.slice(0, 3);
        const positionClasses = ['first', 'second', 'third'];
        const positionNumbers = ['1', '2', '3'];
        const medals = ['🥇', '🥈', '🥉'];

        section.innerHTML = `
            <div class="podium-results-title">
                🏆 RACE PODIUM PREDICTION
            </div>
            <div class="podium-list">
                ${top3.map((driver, index) => `
                    <div class="podium-item ${positionClasses[index]}" data-position="${index + 1}">
                        <div class="podium-position">${positionNumbers[index]}</div>
                        <div class="podium-info">
                            <div class="podium-driver">${driver.driver}</div>
                            <div class="podium-team">${driver.team}</div>
                            <div class="podium-time">${this.formatLapTime(driver.predicted_time)}</div>
                        </div>
                        <div class="podium-stats">
                            <div class="podium-confidence">${driver.confidence}%</div>
                            <div class="confidence-badge">${medals[index]}</div>
                        </div>
                    </div>
                `).join('')}
            </div>
            <div class="podium-footer">
                <p>🎯 ML-powered predictions based on current race conditions</p>
            </div>
        `;

        return section;
    }

    formatLapTime(seconds) {
        return `${parseFloat(seconds).toFixed(3)}s`;
    }

    showError(message) {
        // Create or update error display
        let errorDiv = document.getElementById('errorMessage');
        if (!errorDiv) {
            errorDiv = document.createElement('div');
            errorDiv.id = 'errorMessage';
            errorDiv.className = 'error-message';
            
            const resultsSection = document.getElementById('resultsSection');
            if (resultsSection && resultsSection.parentNode) {
                resultsSection.parentNode.insertBefore(errorDiv, resultsSection);
            }
        }
        
        errorDiv.innerHTML = `
            <div class="error-content">
                <h3>⚠️ Prediction Error</h3>
                <p>${message}</p>
                <button onclick="this.parentElement.parentElement.remove()" class="error-close">Close</button>
            </div>
        `;
        
        errorDiv.style.display = 'block';
        
        // Auto-hide error after 5 seconds
        setTimeout(() => {
            if (errorDiv && errorDiv.parentNode) {
                errorDiv.remove();
            }
        }, 5000);
    }

    animatePodiumAppearance(podiumSection) {
        const items = podiumSection.querySelectorAll('.podium-item');
        items.forEach((item, index) => {
            setTimeout(() => {
                item.style.opacity = '0';
                item.style.transform = 'translateY(30px)';
                item.style.transition = 'all 0.6s ease';
                
                setTimeout(() => {
                    item.style.opacity = '1';
                    item.style.transform = 'translateY(0)';
                }, 50);
            }, index * 150);
        });
    }
}

// Visual Effects Class (for fancy animations)
class VisualEffects {
    constructor() {
        this.initializeParticles();
        this.setupBackgroundAnimation();
    }

    initializeParticles() {
        console.log('✨ Visual effects initialized');
    }

    setupBackgroundAnimation() {
        // Add subtle F1 racing animations
        const body = document.body;
        if (body) {
            body.classList.add('f1-theme-loaded');
        }
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
        try {
            window.f1Predictor = new F1Predictor();
            console.log('✅ F1 Predictor initialized successfully');
        } catch (error) {
            console.error('❌ Error initializing F1 Predictor:', error);
        }

        try {
            window.visualEffects = new VisualEffects();
            console.log('✅ Visual effects initialized successfully');
        } catch (error) {
            console.error('❌ Error initializing visual effects:', error);
        }

        document.body.classList.add('loaded');
        console.log('🏎️ F1 Monaco GP Predictor Ready!');
    }, 100);
});

window.F1Predictor = F1Predictor;