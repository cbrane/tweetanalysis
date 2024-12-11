# Social Media Analysis Chatbot - Project Plan

## Project Overview
An educational chatbot that analyzes social media posts (tweets) and provides detailed analysis with interactive explanations. The system combines traditional NLP metrics with LLM-powered explanations to create an engaging learning tool for social media analysis.

## Technical Stack

### Core Technologies
- Python 3.9+
- Streamlit (frontend)
- OpenAI GPT-4o API
- NLTK/TextBlob (basic sentiment analysis)
- spaCy (NLP features)
- Hugging Face Transformers (advanced sentiment)

### Dependencies
```plaintext
streamlit>=1.28.0
openai>=1.3.0
nltk>=3.8.1
textblob>=0.17.1
spacy>=3.7.2
transformers>=4.35.0
python-dotenv>=1.0.0
pandas>=2.1.0
numpy>=1.24.0
```

## Project Structure
```plaintext
social_media_analyzer/
├── app/
│   ├── main.py              # Streamlit app entry point
│   ├── analyzer.py          # Tweet analysis logic
│   ├── llm_manager.py       # GPT-4o integration
│   ├── educational.py       # Educational features
│   └── utils.py            # Helper functions
├── config/
│   ├── __init__.py
│   └── settings.py         # Configuration settings
├── models/
│   └── sentiment_model/    # Local models (if any)
├── tests/
│   ├── __init__.py
│   ├── test_analyzer.py
│   └── test_llm.py
├── .env                    # Environment variables
├── requirements.txt
└── README.md
```

## Component Specifications

### 1. Tweet Analyzer (analyzer.py)
```python
class TweetAnalyzer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.sentiment_analyzer = TextBlob
        self.initialize_models()

    def analyze_tweet(self, tweet_text: str) -> dict:
        """
        Comprehensive tweet analysis
        """
        return {
            'basic_metrics': self.get_basic_metrics(tweet_text),
            'linguistic_analysis': self.get_linguistic_analysis(tweet_text),
            'sentiment_analysis': self.get_sentiment_analysis(tweet_text),
            'content_analysis': self.get_content_analysis(tweet_text)
        }

    def get_basic_metrics(self, text: str) -> dict:
        """
        Basic text statistics
        """
        return {
            'word_count': len(text.split()),
            'char_count': len(text),
            'avg_word_length': self.calculate_avg_word_length(text),
            'sentence_count': len(self.nlp(text).sents),
            'unique_words': len(set(text.lower().split())),
            'hashtags': self.extract_hashtags(text),
            'mentions': self.extract_mentions(text),
            'urls': self.extract_urls(text)
        }

    def get_linguistic_analysis(self, text: str) -> dict:
        """
        Detailed linguistic analysis
        """
        doc = self.nlp(text)
        return {
            'pos_distribution': self.get_pos_distribution(doc),
            'named_entities': self.get_named_entities(doc),
            'key_phrases': self.extract_key_phrases(doc),
            'readability_score': self.calculate_readability(text),
            'formality_score': self.calculate_formality(doc)
        }

    def get_sentiment_analysis(self, text: str) -> dict:
        """
        Multi-faceted sentiment analysis
        """
        return {
            'basic_sentiment': self.get_basic_sentiment(text),
            'sentiment_score': self.get_sentiment_score(text),
            'emotion_detection': self.detect_emotions(text),
            'subjectivity': self.get_subjectivity(text),
            'sarcasm_probability': self.detect_sarcasm(text)
        }

    def get_content_analysis(self, text: str) -> dict:
        """
        Content categorization and analysis
        """
        return {
            'topic_category': self.categorize_topic(text),
            'writing_style': self.analyze_writing_style(text),
            'clickbait_score': self.detect_clickbait(text),
            'toxicity_check': self.check_toxicity(text),
            'bias_indicators': self.detect_bias(text),
            'call_to_action': self.detect_cta(text)
        }
```

### 2. LLM Manager (llm_manager.py)
```python
class LLMManager:
    def __init__(self):
        self.client = OpenAI()
        self.load_system_prompts()

    def load_system_prompts(self):
        self.system_prompts = {
            'analysis': """You are an educational social media analysis assistant.
            Explain metrics clearly and connect them to practical social media usage.
            When asked about technical details, provide step-by-step explanations.
            Focus on being educational while maintaining engagement.""",
            
            'educational': """You are a social media education expert.
            Break down complex concepts into understandable parts.
            Provide real-world examples and best practices.
            Connect technical metrics to practical social media strategies."""
        }

    def get_analysis_explanation(self, metrics: dict, question: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompts['analysis']},
            {"role": "user", "content": f"Analysis results: {json.dumps(metrics, indent=2)}"},
            {"role": "user", "content": f"Question: {question}"}
        ]
        
        return self._get_completion(messages)

    def get_educational_content(self, 
                              metric_name: str, 
                              score: float, 
                              context: str) -> str:
        prompt = f"""
        Regarding the {metric_name} score of {score} for this tweet:
        1. Explain what this metric means
        2. Break down how it was calculated
        3. Provide context for this score
        4. Suggest improvements if relevant
        5. Connect to social media best practices
        6. Give an example of optimal use
        Context: {context}
        """
        
        messages = [
            {"role": "system", "content": self.system_prompts['educational']},
            {"role": "user", "content": prompt}
        ]
        
        return self._get_completion(messages)

    def _get_completion(self, messages: list) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content
```

### 3. Educational Features (educational.py)
```python
class EducationalFeatures:
    def __init__(self, llm_manager: LLMManager, analyzer: TweetAnalyzer):
        self.llm_manager = llm_manager
        self.analyzer = analyzer

    def explain_metric(self, 
                      metric_name: str, 
                      score: float, 
                      tweet_context: str) -> str:
        """
        Get detailed explanation of a specific metric
        """
        return self.llm_manager.get_educational_content(
            metric_name=metric_name,
            score=score,
            context=tweet_context
        )

    def generate_learning_module(self, 
                               analysis_results: dict, 
                               focus_area: str) -> dict:
        """
        Create a focused learning module based on analysis results
        """
        return {
            'explanation': self.get_focused_explanation(analysis_results, focus_area),
            'best_practices': self.get_best_practices(focus_area),
            'examples': self.get_examples(focus_area),
            'practice_exercises': self.generate_exercises(focus_area)
        }

    def suggest_improvements(self, analysis_results: dict) -> list:
        """
        Generate specific improvement suggestions
        """
        return self.llm_manager.get_analysis_explanation(
            metrics=analysis_results,
            question="What specific improvements could make this tweet more effective?"
        )
```

### 4. Streamlit Interface (main.py)
```python
import streamlit as st
from app.analyzer import TweetAnalyzer
from app.llm_manager import LLMManager
from app.educational import EducationalFeatures

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_analysis" not in st.session_state:
        st.session_state.current_analysis = None
    if "explanation_history" not in st.session_state:
        st.session_state.explanation_history = []

def display_analysis_results(analysis_results):
    """Display analysis results with interactive elements"""
    st.subheader("Analysis Results")
    
    # Basic Metrics
    with st.expander("Basic Metrics", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Word Count", analysis_results['basic_metrics']['word_count'])
            st.metric("Character Count", analysis_results['basic_metrics']['char_count'])
        with col2:
            st.metric("Unique Words", analysis_results['basic_metrics']['unique_words'])
            st.metric("Sentences", analysis_results['basic_metrics']['sentence_count'])

    # Sentiment Analysis
    with st.expander("Sentiment Analysis", expanded=True):
        sentiment_score = analysis_results['sentiment_analysis']['sentiment_score']
        st.metric("Sentiment Score", f"{sentiment_score:.2f}")
        st.progress((sentiment_score + 1) / 2)  # Normalize to 0-1

    # Additional Metrics
    with st.expander("Detailed Analysis"):
        st.json(analysis_results)

def main():
    st.title("Social Media Analysis Chatbot")
    
    # Initialize components
    initialize_session_state()
    analyzer = TweetAnalyzer()
    llm_manager = LLMManager()
    educational = EducationalFeatures(llm_manager, analyzer)

    # Tweet input
    tweet_text = st.text_area("Enter a tweet to analyze:", height=100)
    if st.button("Analyze Tweet"):
        with st.spinner("Analyzing..."):
            analysis_results = analyzer.analyze_tweet(tweet_text)
            st.session_state.current_analysis = analysis_results
            display_analysis_results(analysis_results)

    # Chat interface
    st.subheader("Ask about the analysis")
    if prompt := st.chat_input("Ask a question about the analysis"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response
        with st.spinner("Thinking..."):
            response = llm_manager.get_analysis_explanation(
                st.session_state.current_analysis,
                prompt
            )
            st.session_state.messages.append({"role": "assistant", "content": response})

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if __name__ == "__main__":
    main()
```

## Implementation Phases

### Phase 1: Basic MVP
1. Set up project structure and dependencies
2. Implement basic tweet analysis (TextBlob sentiment)
3. Create simple Streamlit interface
4. Add basic GPT-4o integration
5. Implement basic chat functionality

### Phase 2: Enhanced Features
1. Add advanced metrics and analyses
2. Implement educational features
3. Enhance UI/UX
4. Add visualization components
5. Implement error handling

### Phase 3: Advanced Features
1. Add advanced sentiment models
2. Implement comparative analysis
3. Add user session management
4. Create export functionality
5. Add batch analysis capabilities

## Setup Instructions

1. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

2. Configuration
```plaintext
# .env file
OPENAI_API_KEY=your_api_key_here
```

3. Running the Application
```bash
streamlit run app/main.py
```

## Additional Resources

### Example Prompts
1. Metric Explanation: "Can you explain how the sentiment score was calculated?"
2. Improvement Suggestions: "How could this tweet be more engaging?"
3. Best Practices: "What are the characteristics of high-performing tweets?"
4. Technical Details: "How does the emotion detection work?"

### Learning Objectives
1. Understanding social media metrics
2. Interpreting sentiment analysis
3. Identifying effective content characteristics
4. Applying best practices
5. Understanding technical aspects of social media analysis

## Future Enhancements
1. Additional social media platform support
2. Custom sentiment models
3. Trend analysis
4. User feedback integration
5. Advanced visualization options
6. Integration with social media APIs
7. Batch processing capabilities
8. Export and reporting features

## Contributing Guidelines
1. Follow PEP 8 style guide
2. Write unit tests for new features
3. Document code changes
4. Create detailed pull requests
5. Update requirements.txt as needed