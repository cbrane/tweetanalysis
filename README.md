# Social Media/ Tweet Analysis Chatbot 🤖

## Vision & Purpose
This Social Media Analysis Chatbot is an educational tool designed to bridge the gap between complex social media analytics and practical understanding. Born from the need to make social media analysis more accessible and educational, this project combines state-of-the-art NLP techniques with interactive learning features to help users not just analyze their social media content, but truly understand what the metrics mean and how to improve their social media presence.

### Why This Project?
- **Educational Focus**: Traditional social media analytics tools provide metrics but often lack explanations. Our tool focuses on teaching users what these metrics mean and how to use them effectively.
- **Interactive Learning**: Instead of just displaying numbers, we provide interactive explanations, practical examples, and actionable insights.
- **Comprehensive Analysis**: By combining multiple NLP models with educational AI, we offer both depth in analysis and clarity in explanation.

## 🌟 Features

### Core Analysis Capabilities
- **Multi-Model Sentiment Analysis**: Combined analysis using VADER (specialized for social media), Transformers, and NLTK for comprehensive sentiment understanding
- **Linguistic Analysis**: Advanced POS tagging, named entity recognition, and key phrase extraction
- **Basic Metrics**: Essential metrics like word count, character count, sentence structure, hashtags, mentions, and URLs
- **Content Analysis**: Sophisticated topic modeling with BERTopic, writing style assessment, and engagement metrics
- **Educational Insights**: LLM-powered explanations that break down complex metrics into understandable concepts

### Interactive Learning Features
- Real-time analysis with instant educational feedback
- Detailed explanations of each metric in plain language
- Best practices and actionable improvement suggestions
- Interactive chat interface for deeper understanding
- Thread detection and analysis for context awareness
- Emoji and emoticon analysis for social media relevance

## 🛠️ Technical Implementation

### Core Technologies
- Python 3.9+
- Streamlit (frontend)
- OpenAI API
- spaCy (core NLP)
- Hugging Face Transformers
- VADER Sentiment
- NLTK
- TextBlob
- KeyBERT & BERTopic
- Sentence Transformers

### Key Dependencies
- **Core**: streamlit, openai, python-dotenv
- **NLP/ML**: spacy, nltk, textblob, transformers, vaderSentiment, torch, scikit-learn
- **Advanced NLP**: keybert, bertopic, sentence-transformers
- **Text Analysis**: textstat, emot
- **Additional**: umap-learn, hdbscan (required by BERTopic)

## 🚀 Development Journey

### Phase 1: Foundation (MVP)
- Established core project structure
- Implemented basic tweet analysis using TextBlob
- Created initial Streamlit interface
- Integrated OpenAI API for explanations
- Developed basic chat functionality

### Phase 2: Enhanced Features
- Added advanced metrics and analyses
- Implemented educational features
- Enhanced UI/UX for better learning experience
- Added visualization components
- Implemented comprehensive error handling

### Phase 3: Advanced Capabilities
- Integrated multiple sentiment models
- Implemented user session management
- Enhanced educational content generation
- Added emoji analysis

## 📚 Educational Components

### Learning Approach
1. **Interactive Analysis**: Users can analyze their tweets and get instant feedback
2. **Guided Learning**: Step-by-step explanations of each metric and its importance
3. **Practical Application**: Real-world examples and improvement suggestions
4. **Comprehensive Understanding**: Deep dives into technical aspects when requested

### Key Learning Objectives
1. Understanding social media metrics and their impact
2. Interpreting sentiment analysis results
3. Identifying characteristics of effective content
4. Applying social media best practices
5. Understanding technical aspects of content analysis

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- pip package manager
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd socialmediaproject
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required models:
```bash
python -m spacy download en_core_web_sm
```

5. Configure environment variables:
Create a `.env` file in the root directory:
```plaintext
OPENAI_API_KEY=your_api_key_here
```

### Running the Application
```bash
streamlit run app/main.py
```

## 📁 Project Structure

```
social_media_analyzer/
├── app/
│   ├── main.py              # Streamlit app entry point and configuration
│   ├── analyzer.py          # Tweet analysis with multiple NLP models
│   ├── llm_manager.py       # OpenAI API integration for explanations
│   └── educational.py       # Educational content generation
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## 💡 Usage Guide

### Basic Analysis
1. Launch the application
2. Enter a social media post in the text area
3. Click "Analyze Tweet" to get comprehensive analysis

### Interactive Features
- **Basic Metrics**: View word count, character count, sentence structure, and metadata
- **Sentiment Analysis**: Multi-model sentiment scores with detailed breakdowns
- **Linguistic Analysis**: POS distribution, named entities, key phrases
- **Content Analysis**: Topic modeling, writing style, engagement potential
- **Chat Interface**: Ask questions about the analysis for detailed explanations

### Educational Features
- Detailed metric explanations with practical context
- Specific improvement suggestions based on analysis
- Best practices for social media content
- Real-world examples and applications
- Technical insights into analysis methods

## 🔍 Analysis Components

### Sentiment Analysis
- VADER sentiment scores (specialized for social media)
- Transformer-based sentiment classification
- NLTK sentiment analysis
- Emotion detection and subjectivity analysis

### Linguistic Analysis
- Parts of speech distribution
- Named entity recognition
- Key phrase extraction using KeyBERT
- Topic modeling with BERTopic
- Readability scoring

### Content Analysis
- Thread detection and analysis
- Writing style assessment
- Engagement potential evaluation
- Emoji and emoticon analysis
- Best practices recommendations

## 🔮 Future Roadmap
1. Additional social media platform support
2. Custom sentiment models for specific domains
3. Trend analysis and historical data tracking
4. Enhanced user feedback integration
5. Advanced visualization options
6. Social media API integrations
7. Batch processing capabilities
8. Export and reporting features