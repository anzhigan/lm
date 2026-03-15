class NLPWebClient {
    constructor() {
        this.baseUrl = window.location.origin;
        this.currentMethod = 'tf-idf';
        this.sampleTexts = [
            "FastAPI is a modern web framework for Python.",
            "Machine learning enables solutions to complex data analysis problems",
            "Natural language processing is an important area of artificial intelligence.",
            "Python is a popular language for scientific computing and data analysis.",
            "Neural networks show excellent results in NLP tasks"
        ];

        console.log('API Base URL:', this.baseUrl);
        this.initEventListeners();
        this.checkServerStatus();
    }

    initEventListeners() {
        // Методы обработки
        document.querySelectorAll('.method-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.selectMethod(e));
        });

        // Основные кнопки
        document.getElementById('processBtn').addEventListener('click', () => this.processText());
        document.getElementById('clearBtn').addEventListener('click', () => this.clearOutput());
        document.getElementById('clearInputBtn').addEventListener('click', () => {
            document.getElementById('inputText').value = '';
        });

        // Примеры текстов
        document.querySelectorAll('.sample-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.insertSampleText(e));
        });
    }

    selectMethod(event) {
        document.querySelectorAll('.method-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        event.currentTarget.classList.add('active');
        this.currentMethod = event.currentTarget.dataset.method;
    }

    insertSampleText(event) {
        const index = event.currentTarget.dataset.index;
        document.getElementById('inputText').value = this.sampleTexts[index];
    }

    async checkServerStatus() {
        try {
            const response = await fetch(`${this.baseUrl}/api/v1/health`);
            if (response.ok) {
                this.updateStatus('Server connected', 'success');
            } else {
                this.updateStatus('Server unavailable', 'error');
            }
        } catch (error) {
            this.updateStatus('Connection error', 'error');
        }
    }

    updateStatus(message, type) {
        document.getElementById('statusMessage').textContent = message;
        const statusDot = document.getElementById('statusDot');
        statusDot.className = type === 'success' ? 'status-dot' : 'status-dot offline';
    }

    async processText() {
        const text = document.getElementById('inputText').value.trim();

        if (!text) {
            this.showError('Please enter some text');
            return;
        }

        this.showLoading(true);
        this.hideMessages();

        try {
            let response;

            if (['tf-idf', 'bag-of-words', 'word2vec', 'lsa'].includes(this.currentMethod)) {
                const url = `${this.baseUrl}/api/v1/${this.currentMethod}`;
                const body = this.currentMethod === 'lsa'
                    ? JSON.stringify({ texts: [text], n_components: 3 })
                    : JSON.stringify({ texts: [text] });

                response = await fetch(url, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: body
                });
            } else {
                const url = `${this.baseUrl}/api/v1/nltk/${this.currentMethod}/${encodeURIComponent(text)}`;
                response = await fetch(url);
            }

            if (!response.ok) {
                throw new Error(`Error: ${response.status}`);
            }

            const data = await response.json();
            this.displayResult(data);
            this.showSuccess('Processing complete');

        } catch (error) {
            this.showError(error.message);
        } finally {
            this.showLoading(false);
        }
    }

    displayResult(data) {
        document.getElementById('output').textContent = JSON.stringify(data, null, 2);
    }

    clearOutput() {
        document.getElementById('output').textContent = '// Result will appear here';
        this.hideMessages();
    }

    showLoading(show) {
        document.getElementById('loading').style.display = show ? 'block' : 'none';
    }

    showError(message) {
        const errorElement = document.getElementById('errorMessage');
        errorElement.textContent = message;
        errorElement.style.display = 'block';
        setTimeout(() => errorElement.style.display = 'none', 5000);
    }

    showSuccess(message) {
        const successElement = document.getElementById('successMessage');
        successElement.textContent = message;
        successElement.style.display = 'block';
        setTimeout(() => successElement.style.display = 'none', 3000);
    }

    hideMessages() {
        document.getElementById('errorMessage').style.display = 'none';
        document.getElementById('successMessage').style.display = 'none';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new NLPWebClient();
});