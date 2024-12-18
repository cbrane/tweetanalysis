"""
Tweet Analysis Module

This module provides comprehensive analysis of social media posts (tweets) using
various NLP techniques and models. It combines multiple analysis approaches to
provide insights into sentiment, linguistics, and content quality.

The TweetAnalyzer class serves as the main interface for all analysis operations.
"""

from textblob import TextBlob
import spacy
import re
from collections import Counter
from typing import List, Dict, Any, Tuple
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer as NLTK_SIA
from textstat import textstat
from emot.emo_unicode import EMOTICONS_EMO, UNICODE_EMOJI
from keybert import KeyBERT
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class TweetAnalyzer:
    """
    A comprehensive tweet analysis tool that combines multiple NLP approaches.

    This class provides methods for analyzing social media posts across multiple
    dimensions including sentiment, linguistics, and engagement potential.

    Attributes:
        nlp: SpaCy language model for core NLP tasks
        vader_analyzer: VADER sentiment analyzer
        _transformer_sentiment: Transformer-based sentiment analysis (lazy loaded)
        _nltk_analyzer: NLTK sentiment analyzer (lazy loaded)
        _keyword_model: KeyBERT model for keyword extraction (lazy loaded)
        _topic_model: BERTopic model for topic modeling (lazy loaded)
        _sentence_model: Sentence transformer model (lazy loaded)
        _style_classifier: Style classification model (lazy loaded)
        _zero_shot_classifier: Zero-shot classification model (lazy loaded)
    """

    def __init__(self):
        """Initialize the TweetAnalyzer with core NLP models."""
        try:
            # Load only essential models initially
            self.nlp = spacy.load("en_core_web_sm")
            self.vader_analyzer = SentimentIntensityAnalyzer()
            
            # Initialize placeholders for lazy loading
            self._transformer_sentiment = None
            self._nltk_analyzer = None
            self._keyword_model = None
            self._topic_model = None
            self._sentence_model = None
            self._style_classifier = None
            self._zero_shot_classifier = None
            
        except Exception as e:
            print(f"Error initializing models: {str(e)}")
            raise

    @property
    def transformer_sentiment(self):
        """Lazy load transformer-based sentiment analysis model."""
        if self._transformer_sentiment is None:
            self._transformer_sentiment = pipeline("sentiment-analysis")
        return self._transformer_sentiment

    @property
    def nltk_analyzer(self):
        """Lazy load NLTK sentiment analyzer."""
        if self._nltk_analyzer is None:
            self._nltk_analyzer = NLTK_SIA()
        return self._nltk_analyzer

    @property
    def keyword_model(self):
        """Lazy load KeyBERT model for keyword extraction."""
        if self._keyword_model is None:
            self._keyword_model = KeyBERT()
        return self._keyword_model

    @property
    def topic_model(self):
        """Lazy load BERTopic model for topic modeling."""
        if self._topic_model is None:
            self._topic_model = BERTopic()
        return self._topic_model

    @property
    def sentence_model(self):
        """Lazy load sentence transformer model."""
        if self._sentence_model is None:
            self._sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._sentence_model

    @property
    def style_classifier(self):
        """Lazy load style classification model."""
        if self._style_classifier is None:
            self._style_classifier = pipeline(
                "text-classification", 
                model="SamLowe/roberta-base-go_emotions",
                return_all_scores=True
            )
        return self._style_classifier

    @property
    def zero_shot_classifier(self):
        """Lazy load zero-shot classification model."""
        if self._zero_shot_classifier is None:
            self._zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
        return self._zero_shot_classifier

    def _preprocess_tweet(self, raw_text: str) -> Tuple[str, dict]:
        """
        Preprocess and clean tweet text, extracting metadata and main content.

        Args:
            raw_text: The original tweet text to be processed.

        Returns:
            tuple: (cleaned_text, metadata_dict) where metadata_dict contains
                  extracted information about the tweet.
        """
        # Initialize metadata with None values and thread indicators
        metadata = {
            'author_name': None,
            'author_handle': None,
            'timestamp': None,
            'views': None,
            'main_content': None,
            'is_thread': False,
            'thread_position': None,
            'thread_indicators': []
        }
        
        # Thread indicators to look for
        thread_patterns = [
            (r'🧵', 'Thread emoji'),
            (r'\d+/', 'Numbered thread marker'),
            (r'(?i)thread:', 'Thread label'),
            (r'(?i)\b(thread)\b', 'Thread mention'),
            (r'(?i)(continued|cont\.)', 'Continuation marker'),
            (r'👇|⬇️', 'Directional emoji'),
            (r'(?i)(more in replies|thread below)', 'Thread reference')
        ]
        
        # Check for thread indicators
        text_lower = raw_text.lower()
        for pattern, indicator_type in thread_patterns:
            if re.search(pattern, raw_text):
                metadata['is_thread'] = True
                metadata['thread_indicators'].append(indicator_type)
        
        # Try to extract thread position if it exists
        thread_position_match = re.search(r'(\d+)/(\d+)|(\d+)/', raw_text)
        if thread_position_match:
            position = thread_position_match.group(1) or thread_position_match.group(3)
            metadata['thread_position'] = position
        
        # Split text into lines and remove empty lines
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        
        if not lines:
            return '', metadata
        
        # Check if the text contains any metadata indicators
        has_metadata = any(
            line.startswith('@') or 
            any(time_indicator in line.lower() for time_indicator in ['am', 'pm', ':']) or
            'views' in line.lower() or
            'view' in line.lower()
            for line in lines
        )
        
        if not has_metadata:
            # If no metadata indicators found, treat entire text as content
            main_content = ' '.join(lines)
            metadata['main_content'] = self._clean_tweet_content(main_content)
            return metadata['main_content'], metadata
        
        # If metadata is present, process as before...
        content_lines = []
        for i, line in enumerate(lines):
            # Check for Twitter handle (starts with @)
            if line.startswith("@"):
                metadata["author_handle"] = line.strip()
                continue

            # Check for timestamp (contains time indicators)
            if any(
                time_indicator in line.lower() for time_indicator in ["am", "pm", ":"]
            ) and any(
                date_indicator in line.lower()
                for date_indicator in [
                    "jan",
                    "feb",
                    "mar",
                    "apr",
                    "may",
                    "jun",
                    "jul",
                    "aug",
                    "sep",
                    "oct",
                    "nov",
                    "dec",
                ]
            ):
                metadata["timestamp"] = line.strip()
                continue

            # Check for views
            if "views" in line.lower() or "view" in line.lower():
                metadata["views"] = line.strip()
                continue

            # If first line and not caught by above rules, likely author name
            if i == 0 and not metadata["author_name"] and not line.startswith("@"):
                metadata["author_name"] = line.strip()
                continue

            # If we reach here, it's likely part of the main content
            content_lines.append(line)

        # Join the remaining lines as the main content
        main_content = " ".join(content_lines)

        # Clean the main content
        main_content = self._clean_tweet_content(main_content)
        metadata["main_content"] = main_content

        return main_content, metadata

    def _clean_tweet_content(self, text: str) -> str:
        """Clean the tweet content of unnecessary elements"""
        # Remove URLs
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            text,
        )

        # Remove multiple spaces
        text = re.sub(r"\s+", " ", text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def analyze_tweet(self, raw_tweet_text: str) -> dict:
        """
        Perform comprehensive analysis of a tweet.

        Args:
            raw_tweet_text: The original tweet text to analyze.

        Returns:
            dict: Complete analysis results including:
                - metadata: Tweet metadata and context
                - basic_metrics: Word count, character count, etc.
                - linguistic_analysis: POS tags, entities, etc.
                - sentiment_analysis: Multiple sentiment scores
                - content_analysis: Topic analysis, writing style, etc.
        """
        # Preprocess the tweet
        clean_text, metadata = self._preprocess_tweet(raw_tweet_text)
        doc = self.nlp(clean_text)

        # Get basic metrics
        basic_metrics = self.get_basic_metrics(clean_text)
        
        # Get sentiment analysis (using VADER and transformers)
        sentiment_analysis = self.get_sentiment_analysis(clean_text)
        
        # Get linguistic analysis (using spaCy and textstat)
        linguistic_analysis = self.get_linguistic_analysis(clean_text)
        
        # Get content analysis (using BERT and transformers)
        content_analysis = self.get_content_analysis(clean_text)
        
        # Combine all analyses
        analysis_results = {
            "metadata": metadata,
            "basic_metrics": basic_metrics,
            "linguistic_analysis": linguistic_analysis,
            "sentiment_analysis": sentiment_analysis,
            "content_analysis": content_analysis
        }

        return analysis_results

    def get_basic_metrics(self, text: str) -> dict:
        """
        Calculate basic text metrics for a tweet.

        Args:
            text: Preprocessed tweet text.

        Returns:
            dict: Basic metrics including:
                - word_count: Number of words
                - char_count: Number of characters
                - avg_word_length: Average word length
                - sentence_count: Number of sentences
                - unique_words: Number of unique words
                - hashtags: List of hashtags
                - mentions: List of mentions
                - urls: List of URLs
        """
        doc = self.nlp(text)
        words = text.split()

        return {
            "word_count": len(words),
            "char_count": len(text),
            "avg_word_length": (
                sum(len(word) for word in words) / len(words) if words else 0
            ),
            "sentence_count": len(list(doc.sents)),
            "unique_words": len(set(word.lower() for word in words)),
            "hashtags": re.findall(r"#\w+", text),
            "mentions": re.findall(r"@\w+", text),
            "urls": re.findall(
                r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                text,
            ),
        }

    def get_sentiment_analysis(self, text: str) -> dict:
        """
        Perform multi-model sentiment analysis using multiple packages.

        This method combines VADER (optimized for social media), transformer-based,
        and NLTK sentiment analysis to provide comprehensive sentiment scoring.

        Args:
            text: Preprocessed tweet text to analyze.

        Returns:
            dict: Sentiment analysis results including:
                - sentiment_score: Combined weighted sentiment (-1 to 1)
                - subjectivity: Measure of opinion vs fact (0 to 1)
                - detailed_scores: Individual scores from each model
        """
        if not text or not text.strip():
            return {"sentiment_score": 0.0, "subjectivity": 0.0}

        # Get VADER sentiment (specialized for social media)
        vader_scores = self.vader_analyzer.polarity_scores(text)
        
        # Get Transformer-based sentiment
        transformer_result = self.transformer_sentiment(text)[0]
        
        # Get NLTK sentiment
        nltk_score = self.nltk_analyzer.polarity_scores(text)

        # Calculate weighted compound score
        compound_score = (
            vader_scores['compound'] * 0.5 +  # VADER specialized for social media
            (1 if transformer_result['label'] == 'POSITIVE' else -1) * transformer_result['score'] * 0.3 +  # Transformer
            nltk_score['compound'] * 0.2  # NLTK
        )

        return {
            "sentiment_score": compound_score,
            "subjectivity": nltk_score['neu'],
            "detailed_scores": {
                "vader": vader_scores,
                "transformer": {
                    "label": transformer_result['label'],
                    "score": transformer_result['score']
                },
                "nltk": nltk_score
            }
        }

    def get_linguistic_analysis(self, text: str) -> dict:
        """
        Perform detailed linguistic analysis of tweet text.

        Uses spaCy for part-of-speech tagging, named entity recognition,
        and other linguistic features.

        Args:
            text: Preprocessed tweet text to analyze.

        Returns:
            dict: Linguistic analysis results including:
                - pos_distribution: Distribution of parts of speech
                - named_entities: Identified named entities
                - key_phrases: Important phrases extracted
                - readability_score: Text readability metric
                - formality_score: Text formality metric
        """
        doc = self.nlp(text)

        return {
            "pos_distribution": self._get_pos_distribution(doc),
            "named_entities": self._get_named_entities(doc),
            "key_phrases": self._extract_key_phrases(doc),
            "readability_score": self._calculate_readability(text),
            "formality_score": self._calculate_formality(doc),
        }

    def get_content_analysis(self, text: str) -> dict:
        """
        Advanced content analysis using NLP models.
        
        This method uses a combination of traditional NLP techniques and transformer
        models (if available) to analyze content. If transformer models are not
        loaded, falls back to basic NLP analysis.
        """
        if not text or not text.strip():
            return self._get_empty_content_analysis()

        doc = self.nlp(text)

        # Get content metrics using our advanced models
        content_metrics = self._calculate_content_metrics(text)
        
        # Topic analysis using zero-shot classification
        topic_categories = self._categorize_topic(doc)
        
        # Get writing style analysis
        writing_style = {
            "tone": self._determine_tone(doc),
            "complexity": content_metrics['readability']['automated_readability'] / 100,  # Normalize to 0-1
            "personality": self._detect_personality(doc)
        }
        
        # Calculate engagement metrics
        engagement_metrics = self._calculate_engagement_potential(doc)

        return {
            "topic_category": topic_categories,
            "writing_style": writing_style,
            "engagement_potential": engagement_metrics,
            "content_density": self._calculate_content_density(doc),
            "advanced_metrics": {
                "readability": content_metrics['readability'],
                "keywords": content_metrics['keywords'],
                "semantic_coherence": content_metrics['semantic_coherence']
            }
        }

    def _get_pos_distribution(self, doc) -> Dict[str, float]:
        """
        Calculate distribution of parts of speech in the text.

        Args:
            doc: spaCy Doc object of processed text.

        Returns:
            dict: Mapping of POS tags to their frequency (normalized to 0-1).
        """
        pos_counts = Counter([token.pos_ for token in doc])
        total = len(doc)
        return {pos: count / total for pos, count in pos_counts.items()}

    def _get_named_entities(self, doc) -> List[Dict[str, str]]:
        """
        Extract and classify named entities from the text.

        Args:
            doc: spaCy Doc object of processed text.

        Returns:
            list: List of dictionaries containing entity text and label.
        """
        return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

    def _extract_key_phrases(self, doc) -> List[str]:
        """
        Extract important phrases based on dependency parsing.

        Focuses on noun phrases that are subjects, objects, or prepositional objects.

        Args:
            doc: spaCy Doc object of processed text.

        Returns:
            list: List of key phrases extracted from the text.
        """
        key_phrases = []
        for chunk in doc.noun_chunks:
            if chunk.root.dep_ in ["nsubj", "dobj", "pobj"]:
                key_phrases.append(chunk.text)
        return key_phrases

    def _calculate_readability(self, text: str) -> float:
        """
        Calculate simplified readability score for the text.

        Uses a basic words-per-sentence metric for quick assessment.

        Args:
            text: Preprocessed tweet text.

        Returns:
            float: Readability score (higher scores indicate more complex text).
        """
        words = len(text.split())
        sentences = len([sent for sent in self.nlp(text).sents])
        if sentences == 0:
            return 0
        return round(words / sentences, 2)

    def _calculate_formality(self, doc) -> float:
        """
        Calculate text formality score based on POS patterns.

        Formal indicators include nouns, adjectives, and prepositions.
        Informal indicators include interjections and particles.

        Args:
            doc: spaCy Doc object of processed text.

        Returns:
            float: Formality score (-1 to 1, where 1 is most formal).
        """
        formal_indicators = len(
            [token for token in doc if token.pos_ in ["NOUN", "ADJ", "PREP"]]
        )
        informal_indicators = len(
            [token for token in doc if token.pos_ in ["INTJ", "PART"]]
        )
        total_tokens = len(doc)
        if total_tokens == 0:
            return 0
        return round((formal_indicators - informal_indicators) / total_tokens, 2)

    def _categorize_topic(self, doc) -> List[str]:
        """
        Categorize text topics using zero-shot classification.

        Uses a predefined set of categories and returns those exceeding
        a confidence threshold.

        Args:
            doc: spaCy Doc object of processed text.

        Returns:
            list: List of identified topic categories.
        """
        categories = [
            "Technology", "Business", "Social", "Politics", 
            "Entertainment", "Health", "Education", "General"
        ]
        
        result = self.zero_shot_classifier(
            doc.text,
            categories,
            multi_label=True
        )
        
        # Return categories that exceed confidence threshold
        return [
            label for label, score in zip(result['labels'], result['scores'])
            if score > 0.3  # Confidence threshold
        ] or ["General"]

    def _analyze_writing_style(self, doc) -> Dict[str, Any]:
        """Analyze writing style characteristics"""
        return {
            "tone": self._determine_tone(doc),
            "complexity": self._calculate_complexity(doc),
            "personality": self._detect_personality(doc),
        }

    def _determine_tone(self, doc) -> str:
        """
        Determine the overall tone of the text using emotion classification.

        Maps detailed emotions to broader tone categories (Positive, Negative,
        Neutral, Formal).

        Args:
            doc: spaCy Doc object of processed text.

        Returns:
            str: Identified tone category.
        """
        emotions = self.style_classifier(doc.text)[0]
        # Group emotions into broader categories
        tone_mapping = {
            'joy': 'Positive', 'gratitude': 'Positive', 'admiration': 'Positive',
            'anger': 'Negative', 'disgust': 'Negative', 'sadness': 'Negative',
            'neutral': 'Neutral', 'approval': 'Formal', 'realization': 'Formal'
        }
        
        # Get the dominant emotion
        dominant_emotion = max(emotions, key=lambda x: x['score'])
        return tone_mapping.get(dominant_emotion['label'], 'Neutral')

    def _calculate_complexity(self, doc) -> float:
        """
        Calculate text complexity score based on multiple factors.

        Considers average word length, proportion of long words, and
        sentence length in the calculation.

        Args:
            doc: spaCy Doc object of processed text.

        Returns:
            float: Complexity score (0 to 1, where 1 is most complex).
        """
        # Check if document is empty
        if len(doc) == 0:
            return 0.0
        
        # Calculate average word length
        avg_word_length = sum(len(token.text) for token in doc) / len(doc)
        
        # Count long words (safely)
        long_words = sum(1 for token in doc if len(token.text) > 6)
        
        # Calculate sentence length (with safety check)
        sentences = list(doc.sents)
        sentence_length = len(doc) / len(sentences) if sentences else 0
        
        # Calculate complexity score with safety checks
        complexity = (
            avg_word_length / 10 + 
            (long_words / len(doc) if len(doc) > 0 else 0) + 
            (sentence_length / 20)
        ) / 3
        
        return round(min(complexity, 1.0), 2)

    def _detect_personality(self, doc) -> str:
        """
        Detect writing personality using RoBERTa model.

        Uses zero-shot classification to identify the dominant writing style
        from predefined personality types.

        Args:
            doc: spaCy Doc object of processed text.

        Returns:
            str: Identified personality type (Enthusiastic, Inquisitive, 
                Personal, or Professional).
        """
        candidate_labels = ["Enthusiastic", "Inquisitive", "Personal", "Professional"]
        
        result = self.zero_shot_classifier(
            doc.text,
            candidate_labels,
            multi_label=False
        )
        
        return result['labels'][0]  # Return highest scoring personality type

    def _calculate_engagement_potential(self, doc) -> Dict[str, float]:
        """
        Calculate engagement metrics using multiple analysis components.

        Combines various engagement indicators including hashtags, mentions,
        emoji usage, emotional content, readability, and length optimization.

        Args:
            doc: spaCy Doc object of processed text.

        Returns:
            dict: Engagement metrics including:
                - virality_score: Overall virality potential (0-1)
                - call_to_action_strength: CTA effectiveness (0-1)
                - emotional_appeal: Emotional impact score (0-1)
                - components: Detailed component scores
        """
        text = doc.text

        # Use emoji package for better emoji detection
        emoji_count = len([c for c in text if c in UNICODE_EMOJI])
        emoticon_count = len([e for e in text.split() if e in EMOTICONS_EMO])

        # Calculate virality metrics
        virality_components = {
            'hashtags': len([t for t in doc if t.text.startswith('#')]) * 0.15,
            'mentions': len([t for t in doc if t.text.startswith('@')]) * 0.1,
            'emoji_density': (emoji_count + emoticon_count) * 0.15,
            'emotional_words': self._calculate_emotional_words_score(doc) * 0.2,
            'readability': (textstat.flesch_reading_ease(text) / 100) * 0.2,
            'length_optimization': self._calculate_length_optimization(doc) * 0.2
        }

        virality_score = sum(virality_components.values())
        
        return {
            "virality_score": min(virality_score, 1.0),
            "call_to_action_strength": self._detect_cta_strength(doc),
            "emotional_appeal": self._calculate_emotional_appeal(doc),
            "components": virality_components
        }

    def _calculate_emotional_words_score(self, doc) -> float:
        """
        Calculate emotional words score using VADER sentiment analysis.

        Uses the absolute compound score from VADER to measure emotional
        intensity regardless of polarity.

        Args:
            doc: spaCy Doc object of processed text.

        Returns:
            float: Emotional words score (0-1).
        """
        text = doc.text
        vader_scores = self.vader_analyzer.polarity_scores(text)
        return abs(vader_scores['compound'])  # Use absolute compound score

    def _calculate_length_optimization(self, doc) -> float:
        """
        Calculate length optimization score for social media impact.

        Evaluates text length against optimal social media post length
        (60-280 characters, with 170 being ideal).

        Args:
            doc: spaCy Doc object of processed text.

        Returns:
            float: Length optimization score (0-1).
        """
        text_length = len(doc.text)
        if 60 <= text_length <= 280:
            return 1.0 - (abs(170 - text_length) / 170)  # Optimal around 170 chars
        return 0.5

    def _calculate_content_density(self, doc) -> float:
        """
        Calculate content density as ratio of meaningful words to total words.

        Meaningful words are defined as non-stop words and alphabetic tokens.

        Args:
            doc: spaCy Doc object of processed text.

        Returns:
            float: Content density score (0-1).
        """
        meaningful_words = len(
            [token for token in doc if not token.is_stop and token.is_alpha]
        )
        total_words = len([token for token in doc if token.is_alpha])
        return round(meaningful_words / total_words if total_words > 0 else 0, 2)

    def _calculate_semantic_coherence(self, embeddings):
        """
        Calculate semantic coherence using sentence embeddings.

        Measures how well the content flows and connects by calculating
        average cosine similarity between sentence embeddings.

        Args:
            embeddings: Sentence transformer embeddings.

        Returns:
            float: Semantic coherence score (0-1).
        """
        # Calculate pairwise cosine similarity
        similarity = embeddings @ embeddings.T
        # Calculate average similarity
        avg_similarity = similarity.mean()
        return avg_similarity

    def _calculate_content_metrics(self, text: str) -> dict:
        """
        Calculate comprehensive content metrics using advanced NLP models.

        Combines keyword extraction, semantic analysis, and readability
        metrics for detailed content evaluation.

        Args:
            text: Preprocessed tweet text.

        Returns:
            dict: Content metrics including:
                - keywords: Key terms and phrases
                - semantic_coherence: Text flow measure
                - readability: Multiple readability scores
        """
        # Extract keywords using BERT
        keywords = self.keyword_model.extract_keywords(text, 
                                                     keyphrase_ngram_range=(1, 2),
                                                     stop_words='english',
                                                     use_maxsum=True,
                                                     nr_candidates=20,
                                                     top_n=5)

        # Get sentence embeddings for semantic analysis
        embeddings = self.sentence_model.encode([text])
        
        # Calculate semantic coherence
        semantic_coherence = float(self._calculate_semantic_coherence(embeddings))
        
        # Get readability metrics
        readability = {
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'dale_chall': textstat.dale_chall_readability_score(text),
            'automated_readability': textstat.automated_readability_index(text)
        }

        return {
            "keywords": [k[0] for k in keywords],  # Extract just the keywords
            "semantic_coherence": semantic_coherence,
            "readability": readability
        }

    def _get_empty_content_analysis(self) -> dict:
        """
        Return empty content analysis structure for empty text input.

        Provides a consistent return structure with default/empty values
        when analyzing empty or invalid text.

        Returns:
            dict: Empty content analysis structure with default values.
        """
        return {
            "topic_analysis": {
                "main_topics": [],
                "keywords": []
            },
            "semantic_features": {
                "embeddings": [],
                "semantic_similarity": 0.0
            },
            "content_metrics": {
                "topic_category": ["Unknown"],
                "writing_style": {
                    "tone": "Neutral",
                    "complexity": 0.0,
                    "personality": "Unknown"
                },
                "engagement_potential": {
                    "virality_score": 0.0,
                    "call_to_action_strength": 0.0,
                    "emotional_appeal": 0.0
                },
                "content_density": 0.0
            }
        }

    def _detect_cta_strength(self, doc) -> float:
        """
        Detect call-to-action strength using zero-shot classification.

        Evaluates text for presence and strength of calls to action,
        instructions, requests, and suggestions.

        Args:
            doc: spaCy Doc object of processed text.

        Returns:
            float: CTA strength score (0-1).
        """
        cta_categories = [
            "call to action",
            "instruction",
            "request",
            "suggestion",
            "information only"
        ]
        
        result = self.zero_shot_classifier(
            doc.text,
            cta_categories,
            multi_label=True
        )
        
        # Calculate CTA strength based on confidence scores
        cta_scores = [
            score for label, score in zip(result['labels'], result['scores'])
            if label != "information only"
        ]
        
        return max(cta_scores) if cta_scores else 0.0

    def _calculate_emotional_appeal(self, doc) -> float:
        """
        Calculate emotional appeal using multiple emotion detection models.

        Combines VADER sentiment intensity, emotion classifications,
        and emoji density for comprehensive emotional impact scoring.

        Args:
            doc: spaCy Doc object of processed text.

        Returns:
            float: Emotional appeal score (0-1).
        """
        text = doc.text
        
        # Get VADER sentiment intensity
        vader_scores = self.vader_analyzer.polarity_scores(text)
        sentiment_intensity = abs(vader_scores['compound'])
        
        # Get emotion classifications
        emotions = self.style_classifier(text)[0]
        emotion_intensity = max(e['score'] for e in emotions)
        
        # Get emoji density
        emoji_count = len([c for c in text if c in UNICODE_EMOJI])
        emoji_density = emoji_count / len(text) if len(text) > 0 else 0
        
        # Combine scores with weights
        weighted_score = (
            sentiment_intensity * 0.4 +
            emotion_intensity * 0.4 +
            emoji_density * 0.2
        )
        
        return round(min(weighted_score, 1.0), 2)
