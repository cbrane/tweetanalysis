"""
Social Media Analysis Chatbot - Main Application

This module serves as the main entry point for the Streamlit-based social media
analysis application. It provides an interactive interface for analyzing social
media posts, displaying results, and providing educational content.

The application combines multiple analysis components including sentiment analysis,
linguistic analysis, and engagement metrics, presented through an intuitive
user interface.
"""

import warnings
import os
import logging
import streamlit as st
from analyzer import TweetAnalyzer
from llm_manager import LLMManager
from educational import EducationalFeatures

# Configure logging to suppress specific messages
logging.getLogger("torch.classes").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", message=".*huggingface/tokenizers.*")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Disable PyTorch MPS warning messages
if hasattr(logging, "getLogger"):
    logging.getLogger("torch.distributed.nn.jit.instantiator").setLevel(logging.ERROR)


def initialize_session_state():
    """
    Initialize Streamlit session state variables.

    Sets up persistent state variables for storing chat history,
    analysis results, and other session-specific data.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_analysis" not in st.session_state:
        st.session_state.current_analysis = None


def display_analysis_results(analysis_results: dict):
    """
    Display analysis results with interactive elements.

    Creates an organized display of analysis results using Streamlit
    components, with expandable sections for different types of analysis.

    Args:
        analysis_results: Dictionary containing complete analysis results
                        including basic metrics, sentiment analysis, and
                        linguistic analysis.
    """
    # Create a container for all analysis content
    with st.container():
        st.subheader("Analysis Results")

        # Create containers for each section
        tweet_info = st.container()
        basic_metrics = st.container()
        sentiment = st.container()

        # Display Tweet Metadata
        with tweet_info:
            if "metadata" in analysis_results:
                with st.expander("Tweet Information", expanded=True):
                    metadata = analysis_results["metadata"]
                    if metadata["author_name"]:
                        st.write(f"**Author:** {metadata['author_name']}")
                    if metadata["author_handle"]:
                        st.write(f"**Handle:** {metadata['author_handle']}")
                    if metadata["timestamp"]:
                        st.write(f"**Posted:** {metadata['timestamp']}")
                    if metadata["views"]:
                        st.write(f"**Views:** {metadata['views']}")
                    st.write("**Tweet Content:**")
                    st.write(f">{metadata['main_content']}")

        # Basic Metrics
        with basic_metrics:
            with st.expander("Basic Metrics", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Word Count", analysis_results["basic_metrics"]["word_count"]
                    )
                    st.metric(
                        "Character Count",
                        analysis_results["basic_metrics"]["char_count"],
                    )
                with col2:
                    st.metric(
                        "Unique Words",
                        analysis_results["basic_metrics"]["unique_words"],
                    )
                    st.metric(
                        "Sentences", analysis_results["basic_metrics"]["sentence_count"]
                    )

        # Sentiment Analysis
        with sentiment:
            with st.expander("Sentiment Analysis", expanded=True):
                sentiment_score = analysis_results["sentiment_analysis"][
                    "sentiment_score"
                ]
                st.metric("Sentiment Score", f"{sentiment_score:.2f}")

                # Add a progress bar for sentiment visualization
                st.write("Sentiment Range (-1 to +1):")
                progress_val = (sentiment_score + 1) / 2  # Normalize to 0-1
                st.progress(progress_val)

                # Add sentiment interpretation
                if sentiment_score > 0.5:
                    st.success("This tweet has a very positive tone! 😊")
                elif sentiment_score > 0:
                    st.info("This tweet has a somewhat positive tone 🙂")
                elif sentiment_score > -0.5:
                    st.warning("This tweet has a somewhat negative tone 😐")
                else:
                    st.error("This tweet has a very negative tone 😟")


def display_advanced_analysis(analysis_results: dict):
    """
    Display advanced analysis results with detailed explanations.

    Creates expandable sections for detailed sentiment analysis,
    linguistic analysis, and content analysis with interactive
    visualizations and explanations.

    Args:
        analysis_results: Dictionary containing complete analysis results
                        including advanced metrics and detailed scores.
    """
    # Sentiment Analysis Details
    with st.expander("Detailed Sentiment Analysis", expanded=True):
        st.subheader("Multi-Model Sentiment Analysis")
        detailed_scores = analysis_results["sentiment_analysis"]["detailed_scores"]

        # VADER Scores
        st.write("**VADER Analysis:**")
        vader_cols = st.columns(4)
        for i, (key, value) in enumerate(detailed_scores["vader"].items()):
            vader_cols[i].metric(key.title(), f"{value:.2f}")

        # Transformer Results
        st.write("**Transformer Analysis:**")
        trans_cols = st.columns(2)
        trans_cols[0].metric("Label", detailed_scores["transformer"]["label"])
        trans_cols[1].metric(
            "Confidence", f"{detailed_scores['transformer']['score']:.2f}"
        )

        # NLTK Scores
        st.write("**NLTK Analysis:**")
        nltk_cols = st.columns(4)
        for i, (key, value) in enumerate(detailed_scores["nltk"].items()):
            nltk_cols[i].metric(key.title(), f"{value:.2f}")

        # Add explanation
        st.markdown(
            """
        ---
        ### Understanding the Sentiment Analysis
        
        This analysis combines three different sentiment analysis models to provide a comprehensive view:
        
        **VADER Analysis** (Specialized for Social Media):
        - Neg: Negative sentiment (0 to 1)
        - Neu: Neutral sentiment (0 to 1)
        - Pos: Positive sentiment (0 to 1)
        - Compound: Overall sentiment (-1 to 1, where -1 is most negative, +1 is most positive)
        
        **Transformer Analysis** (State-of-the-art ML Model):
        - Label: Overall sentiment classification
        - Confidence: How confident the model is in its prediction (0 to 1)
        
        **NLTK Analysis** (Traditional NLP Approach):
        - Similar to VADER, provides a different perspective using classical NLP techniques
        - Compound score ranges from -1 (very negative) to +1 (very positive)
        """
        )

    # Linguistic Analysis
    with st.expander("Linguistic Analysis", expanded=True):
        # Parts of Speech Distribution
        st.subheader("Parts of Speech Distribution")
        pos_dist = analysis_results["linguistic_analysis"]["pos_distribution"]
        # Convert to percentage for better visualization
        total = sum(pos_dist.values())
        pos_dist_pct = {k: (v / total) * 100 for k, v in pos_dist.items()}
        st.bar_chart(pos_dist_pct)

        # Display actual counts in a more readable format
        st.write("**Distribution Breakdown:**")
        # Create a markdown table for better formatting
        pos_table = "| Part of Speech | Count | Percentage |\n|---------------|--------|------------|\n"
        for pos, count in pos_dist.items():
            percentage = (count / total) * 100
            pos_table += f"| {pos} | {count:.2f} | {percentage:.1f}% |\n"
        st.markdown(pos_table)

        # Add POS explanation
        st.markdown(
            """
        ---
        ### Understanding Parts of Speech
        - **NOUN**: Names of things (e.g., 'cat', 'table')
        - **VERB**: Action words (e.g., 'run', 'write')
        - **ADJ**: Descriptive words (e.g., 'blue', 'tall')
        - **ADV**: Modifies verbs (e.g., 'quickly', 'very')
        - **PRON**: Pronouns (e.g., 'he', 'they')
        - **DET**: Determiners (e.g., 'the', 'a')
        - **ADP**: Prepositions (e.g., 'in', 'on')
        - **CONJ**: Conjunctions (e.g., 'and', 'but')
        - **PUNCT**: Punctuation marks
        """
        )

        # Named Entities
        st.subheader("Named Entities")
        entities = analysis_results["linguistic_analysis"]["named_entities"]
        if entities:
            for ent in entities:
                st.write(f"• {ent['text']} ({ent['label']})")
        else:
            st.write("No named entities found")

        st.markdown(
            """
        ---
        ### Understanding Named Entities
        Named entities are real-world objects that can be denoted with a proper name:
        - **PERSON**: Names of people
        - **ORG**: Organizations, companies
        - **LOC**: Locations, cities, countries
        - **DATE**: Dates and times
        - **GPE**: Geopolitical entities
        """
        )

        # Key Phrases
        st.subheader("Key Phrases")
        key_phrases = analysis_results["linguistic_analysis"]["key_phrases"]
        if key_phrases:
            for phrase in key_phrases:
                st.write(f"• {phrase}")
        else:
            st.write("No key phrases found")

        # Readability and Formality
        st.subheader("Text Characteristics")
        text_cols = st.columns(2)
        with text_cols[0]:
            st.metric(
                "Readability Score",
                f"{analysis_results['linguistic_analysis']['readability_score']:.2f}",
            )
        with text_cols[1]:
            st.metric(
                "Formality Score",
                f"{analysis_results['linguistic_analysis']['formality_score']:.2f}",
            )

        st.markdown(
            """
        ---
        ### Understanding Text Characteristics
        
        **Readability Score:**
        - Measures how easy the text is to read
        - Higher scores (>10) indicate more complex text
        - Lower scores (<8) indicate more accessible text
        
        **Formality Score:**
        - Ranges from -1 (very informal) to 1 (very formal)
        - Based on word choice and sentence structure
        - Around 0 indicates neutral formality
        """
        )

    # Content Analysis
    with st.expander("Content Analysis", expanded=True):
        # Topic Categories
        st.subheader("Topic Categories")
        st.write(", ".join(analysis_results["content_analysis"]["topic_category"]))

        st.markdown(
            """
        ---
        ### Understanding Topic Categories
        Topics are determined using advanced AI classification:
        - Multiple topics may be detected if the content spans different subjects
        - Confidence threshold of 30% is used to ensure reliability
        - "General" is assigned when no specific topic exceeds the threshold
        """
        )

        # Writing Style
        st.subheader("Writing Style Analysis")
        style = analysis_results["content_analysis"]["writing_style"]
        style_cols = st.columns(3)
        with style_cols[0]:
            st.metric("Tone", style["tone"])
        with style_cols[1]:
            st.metric("Complexity", f"{style['complexity']:.2f}")
        with style_cols[2]:
            st.metric("Personality", style["personality"])

        st.markdown(
            """
        ---
        ### Understanding Writing Style
        
        **Tone:**
        - Positive: Optimistic, upbeat content
        - Negative: Critical or concerning content
        - Neutral: Factual, unbiased content
        - Formal: Professional, structured content
        
        **Complexity (0-1):**
        - 0-0.3: Simple, easy to understand
        - 0.3-0.7: Moderate complexity
        - 0.7-1.0: Complex, technical content
        
        **Personality:**
        - Enthusiastic: Energetic, passionate
        - Inquisitive: Questioning, curious
        - Personal: Intimate, conversational
        - Professional: Formal, business-like
        """
        )

        st.metric(
            "Content Density",
            f"{analysis_results['content_analysis']['content_density']:.2f}",
        )

        st.markdown(
            """
        **Content Density (0-1):**
        - Measures the ratio of meaningful words to total words
        - Higher values indicate more informative content
        - Lower values suggest more casual or conversational content
        """
        )

        # Engagement Metrics
        st.subheader("Engagement Analysis")
        engagement = analysis_results["content_analysis"]["engagement_potential"]

        # Main engagement metrics
        eng_cols = st.columns(3)
        with eng_cols[0]:
            st.metric("Virality Score", f"{engagement['virality_score']:.2f}")
        with eng_cols[1]:
            st.metric(
                "Call to Action Strength",
                f"{engagement['call_to_action_strength']:.2f}",
            )
        with eng_cols[2]:
            st.metric("Emotional Appeal", f"{engagement['emotional_appeal']:.2f}")

        st.markdown(
            """
        ---
        ### Understanding Engagement Metrics
        
        **Virality Score (0-1):**
        - Predicts potential for social media engagement
        - Based on multiple factors including hashtags, mentions, and emotional content
        - Higher scores suggest better viral potential
        
        **Call to Action Strength (0-1):**
        - Measures how effectively the content prompts user action
        - Based on imperative language and engagement triggers
        - Higher scores indicate stronger calls to action
        
        **Emotional Appeal (0-1):**
        - Measures emotional impact on readers
        - Combines sentiment intensity and emotional language
        - Higher scores suggest more emotionally engaging content
        """
        )

        # Virality Components
        st.write("**Virality Components:**")
        if "components" in engagement:
            comp_cols = st.columns(len(engagement["components"]))
            for i, (component, score) in enumerate(engagement["components"].items()):
                comp_cols[i].metric(component.replace("_", " ").title(), f"{score:.2f}")

        st.markdown(
            """
        **Virality Components Explained:**
        - Hashtags: Effective use of relevant hashtags
        - Mentions: Strategic user mentions
        - Emoji Density: Appropriate use of emojis
        - Emotional Words: Use of emotionally engaging language
        - Readability: How easy the content is to understand
        - Length Optimization: Optimal content length for platform
        """
        )

        # Advanced Metrics
        st.subheader("Advanced Content Metrics")
        advanced = analysis_results["content_analysis"]["advanced_metrics"]

        # Readability Metrics
        st.write("**Readability Scores:**")
        read_cols = st.columns(3)
        with read_cols[0]:
            st.metric(
                "Flesch Reading Ease",
                f"{advanced['readability']['flesch_reading_ease']:.1f}",
            )
        with read_cols[1]:
            st.metric(
                "Dale-Chall Score", f"{advanced['readability']['dale_chall']:.1f}"
            )
        with read_cols[2]:
            st.metric(
                "Automated Readability",
                f"{advanced['readability']['automated_readability']:.1f}",
            )

        st.markdown(
            """
        ---
        ### Understanding Readability Metrics
        
        **Flesch Reading Ease (0-100):**
        - 90-100: Very easy to read
        - 60-70: Standard/average
        - 0-30: Very difficult to read
        
        **Dale-Chall Score:**
        - <4.9: Easily understood by 4th grader
        - 5.0-6.9: 5th-6th grade
        - 7.0-8.9: 7th-8th grade
        - 9.0+: College level
        
        **Automated Readability Index:**
        - Approximates the US grade level needed to understand the text
        - Lower scores indicate more accessible content
        """
        )

        # Keywords and Semantic Coherence
        st.write("**Content Analysis:**")
        key_cols = st.columns(2)
        with key_cols[0]:
            st.write("**Keywords:**")
            if advanced.get("keywords"):
                st.write(", ".join(advanced["keywords"]))
        with key_cols[1]:
            st.metric(
                "Semantic Coherence", f"{advanced.get('semantic_coherence', 0):.2f}"
            )

        st.markdown(
            """
        ---
        ### Understanding Content Analysis
        
        **Keywords:**
        - Most important terms extracted from the content
        - Based on relevance and frequency
        - Useful for understanding main topics
        
        **Semantic Coherence (0-1):**
        - Measures how well the content flows and connects
        - Higher scores indicate more coherent, well-structured content
        - Lower scores might suggest scattered or disconnected ideas
        """
        )


def display_metric_with_llm_option(
    title: str,
    value: any,
    description: str,
    metric_name: str,
    analysis_results: dict,
    educational: EducationalFeatures,
    tab: str,
):
    """
    Display a metric with an option to get LLM-powered explanation.

    Args:
        title: Display title for the metric
        value: Current value of the metric (can be numeric or string)
        description: Brief description of the metric
        metric_name: Internal name of the metric for analysis
        analysis_results: Complete analysis results for context
        educational: EducationalFeatures instance for generating explanations
        tab: Current tab name for context
    """
    # Initialize storage for LLM explanations if needed
    if "llm_explanations" not in st.session_state:
        st.session_state.llm_explanations = {}

    metric_container = st.container()

    with metric_container:
        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            st.metric(title, value)
        with col2:
            st.write(description)
        with col3:
            button_key = f"explain_{metric_name}_{tab}_{title.lower().replace(' ', '_')}"

            if st.button(
                "Get More Info 🔍" if metric_name not in st.session_state.llm_explanations
                else "Update Info 🔄",
                key=button_key,
                use_container_width=True,
            ):
                with st.spinner("Getting detailed analysis..."):
                    try:
                        # Convert value to float only for numeric metrics
                        metric_value = (
                            float(value) 
                            if isinstance(value, (int, float)) or 
                               (isinstance(value, str) and value.replace('.', '').isdigit())
                            else value
                        )
                        
                        explanation = educational.explain_metric(
                            metric_name, metric_value, analysis_results
                        )
                        st.session_state.llm_explanations[metric_name] = explanation
                    except Exception as e:
                        st.error(f"Unable to analyze {title}: {str(e)}")
                        return

    # Display explanation if available
    if metric_name in st.session_state.llm_explanations:
        explanation = st.session_state.llm_explanations[metric_name]
        
        st.markdown("---")
        
        with st.expander(f"Analysis of {explanation['name']}", expanded=True):
            st.write(explanation["analysis"])

            if explanation.get("interpretation"):
                st.markdown("#### Key Points")
                for key, value in explanation["interpretation"].items():
                    if isinstance(value, dict):
                        st.markdown(f"**{key.title()}**")
                        for subkey, subvalue in value.items():
                            st.markdown(f"- {subkey.title()}: {subvalue}")
                    else:
                        st.markdown(f"- **{key.title()}:** {value}")

            if explanation.get("components"):
                st.markdown("#### Components Analyzed")
                for component in explanation["components"]:
                    st.markdown(f"- {component}")

        st.markdown("---")


def display_educational_content(educational: EducationalFeatures, analysis_results: dict):
    """Display educational content and suggestions for tweet improvement."""
    st.subheader("Learning Resources")

    # Add "Get All Explanations" button at the top
    if st.button(
        "Get Detailed Analysis for All Metrics ✨",
        key="get_all_analysis",
        use_container_width=True,
    ):
        with st.spinner("Analyzing all metrics..."):
            all_explanations = {}
            for category in educational.metric_categories.keys():
                # Skip Basic Metrics category
                if category == "Basic Metrics":
                    continue
                for metric_name in educational.metric_categories[category].keys():
                    try:
                        value = educational.llm_manager._get_metric_value(
                            analysis_results, metric_name
                        )
                        # Handle non-numeric values appropriately
                        if isinstance(value, (list, tuple)):
                            value = ", ".join(map(str, value))
                        explanation = educational.explain_metric(
                            metric_name, value, analysis_results
                        )
                        all_explanations[metric_name] = explanation
                    except Exception as e:
                        st.error(f"Error analyzing {metric_name}: {str(e)}")

            for metric_name, explanation in all_explanations.items():
                with st.expander(
                    f"{explanation['name']} Analysis",
                    expanded=True,
                    key=f"all_{metric_name}",
                ):
                    st.write(explanation["analysis"])

    # Create tabs for different aspects of the analysis (excluding Basic Metrics)
    metric_tabs = st.tabs(
        [
            "Sentiment Analysis",
            "Linguistic Analysis",
            "Content Analysis",
            "Engagement Metrics",
            "Improvement Suggestions",
        ]
    )

    # Sentiment Analysis Tab
    with metric_tabs[0]:
        st.write("### Understanding Emotional Impact")

        # Overall Sentiment Score
        sentiment_score = analysis_results["sentiment_analysis"]["sentiment_score"]
        display_metric_with_llm_option(
            "Overall Sentiment",
            sentiment_score,
            "Combined sentiment score from multiple models",
            "sentiment_score",
            analysis_results,
            educational,
            tab="sentiment",
        )

        # VADER Sentiment
        vader_scores = analysis_results["sentiment_analysis"]["detailed_scores"][
            "vader"
        ]
        for score_type, value in vader_scores.items():
            display_metric_with_llm_option(
                f"VADER {score_type.title()}",
                value,
                f"VADER's {score_type} sentiment score",
                f"vader_{score_type}",
                analysis_results,
                educational,
                tab="sentiment",
            )

        # Subjectivity
        subjectivity = analysis_results["sentiment_analysis"]["subjectivity"]
        display_metric_with_llm_option(
            "Subjectivity",
            subjectivity,
            "Measure of opinion vs fact-based content",
            "subjectivity",
            analysis_results,
            educational,
            tab="sentiment",
        )

    # Linguistic Analysis Tab
    with metric_tabs[1]:
        st.write("### Understanding Language Structure")

        # Parts of Speech Distribution
        pos_dist = analysis_results["linguistic_analysis"]["pos_distribution"]
        for pos, value in pos_dist.items():
            # Use a standardized metric name format for POS tags
            metric_name = f"linguistic_pos_{pos.lower()}"
            display_metric_with_llm_option(
                f"{pos} Usage",
                f"{value:.2f}",
                f"Frequency of {pos} in the text",
                metric_name,
                analysis_results,
                educational,
                tab="linguistic",
            )

        # Readability Score
        readability = analysis_results["linguistic_analysis"]["readability_score"]
        display_metric_with_llm_option(
            "Readability Score",
            readability,
            "How easy the text is to read",
            "linguistic_readability",
            analysis_results,
            educational,
            tab="linguistic",
        )

        # Formality Score
        formality = analysis_results["linguistic_analysis"]["formality_score"]
        display_metric_with_llm_option(
            "Formality Score",
            formality,
            "Level of formal language used",
            "linguistic_formality",
            analysis_results,
            educational,
            tab="linguistic",
        )

    # Content Analysis Tab
    with metric_tabs[2]:
        st.write("### Understanding Content Quality")

        # Topic Categories - handle as string
        topics = ", ".join(analysis_results["content_analysis"]["topic_category"])
        display_metric_with_llm_option(
            "Topic Categories",
            topics,  # Pass as string directly
            "Main topics detected in the content",
            "topic_categories",
            analysis_results,
            educational,
            tab="content",
        )

        # Writing Style
        style = analysis_results["content_analysis"]["writing_style"]
        for style_aspect, value in style.items():
            display_metric_with_llm_option(
                f"Writing Style - {style_aspect.title()}",
                value,
                f"Analysis of content {style_aspect}",
                f"writing_style_{style_aspect}",
                analysis_results,
                educational,
                tab="content",
            )

        # Content Density
        density = analysis_results["content_analysis"]["content_density"]
        display_metric_with_llm_option(
            "Content Density",
            density,
            "Ratio of meaningful content to total content",
            "content_density",
            analysis_results,
            educational,
            tab="content",
        )

    # Engagement Metrics Tab
    with metric_tabs[3]:
        st.write("### Understanding Engagement Potential")

        # Main engagement metrics - only display these
        engagement = analysis_results["content_analysis"]["engagement_potential"]
        
        # Display only the main metrics, not the components
        display_metric_with_llm_option(
            "Virality Score",
            engagement["virality_score"],
            "Overall potential for content to be shared",
            "virality_score",
            analysis_results,
            educational,
            tab="engagement"
        )

        display_metric_with_llm_option(
            "Call to Action Strength",
            engagement["call_to_action_strength"],
            "Effectiveness of prompting user action",
            "call_to_action_strength",
            analysis_results,
            educational,
            tab="engagement"
        )

        display_metric_with_llm_option(
            "Emotional Appeal",
            engagement["emotional_appeal"],
            "Strength of emotional connection",
            "emotional_appeal",
            analysis_results,
            educational,
            tab="engagement"
        )

        # Content Density
        density = analysis_results["content_analysis"]["content_density"]
        display_metric_with_llm_option(
            "Content Density",
            density,
            "Ratio of meaningful content to total content",
            "content_density",
            analysis_results,
            educational,
            tab="engagement"
        )

    # Improvement Suggestions Tab
    with metric_tabs[4]:
        st.write("### 📈 Areas for Improvement")

        # Identify weak points
        weak_points = educational._identify_weak_points(analysis_results)

        if weak_points:
            for metric, score in weak_points.items():
                metric_display_name = metric.replace("_", " ").title()
                display_metric_with_llm_option(
                    metric_display_name,
                    score,
                    "Could be improved for better engagement",
                    metric,
                    analysis_results,
                    educational,
                    tab="improvement",
                )
        else:
            st.success("👏 Great job! No significant areas for improvement identified.")

            # Optional: General improvement tips
            display_metric_with_llm_option(
                "General Tips",
                "✨",
                "Get tips to make your already good tweet even better",
                "general_improvement",
                analysis_results,
                educational,
                tab="improvement",
            )


def main():
    """
    Main application entry point.

    Sets up the Streamlit interface, initializes components, and
    manages the main application flow including:
    - Tweet input and analysis
    - Results display
    - Educational content
    - Chat interface
    """
    st.title("Social Media Analysis Chatbot")

    # Initialize components
    initialize_session_state()
    analyzer = TweetAnalyzer()
    llm_manager = LLMManager()
    educational = EducationalFeatures(llm_manager)

    # Tweet input section
    tweet_text = st.text_area("Enter a tweet to analyze:", height=100)

    # Create a container for buttons that's always visible
    button_container = st.container()
    analysis_container = st.container()

    with button_container:
        # Center the buttons using columns
        col1, col2, col3 = st.columns([1, 1, 1])

        with col2:
            # Main analyze button - always visible
            analyze_button = st.button(
                "Analyze Tweet", type="primary", use_container_width=True
            )

    # Store analysis results in session state when analyze button is clicked
    if analyze_button and tweet_text.strip():
        with st.spinner("Analyzing..."):
            analysis_results = analyzer.analyze_tweet(tweet_text)
            st.session_state.current_analysis = analysis_results

    # Display analysis if we have results
    if (
        hasattr(st.session_state, "current_analysis")
        and st.session_state.current_analysis is not None
    ):
        with analysis_container:
            # Show tabs for different types of analysis
            tab1, tab2, tab3 = st.tabs(
                ["Basic Analysis", "Advanced Analysis", "Learning"]
            )

            with tab1:
                display_analysis_results(st.session_state.current_analysis)

            with tab2:
                display_advanced_analysis(st.session_state.current_analysis)

            with tab3:
                display_educational_content(
                    educational, st.session_state.current_analysis
                )

    # Always show chat interface at the bottom
    st.markdown("---")
    st.subheader("Ask Questions")

    # Chat input and history
    if prompt := st.chat_input(
        "Ask anything about the analysis or social media writing in general"
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            if (
                hasattr(st.session_state, "current_analysis")
                and st.session_state.current_analysis is not None
            ):
                response = llm_manager.get_analysis_explanation(
                    st.session_state.current_analysis, prompt
                )
            else:
                response = llm_manager.get_educational_content(
                    prompt, "general_question"
                )

            st.session_state.messages.append({"role": "assistant", "content": response})

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


if __name__ == "__main__":
    main()
