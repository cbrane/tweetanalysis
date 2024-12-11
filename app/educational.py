"""
Educational Features Module

This module provides educational content and explanations for social media analysis
metrics. It uses LLM-powered explanations to help users understand analysis results
and improve their social media content.
"""

from typing import Dict, List
from llm_manager import LLMManager
import streamlit as st


class EducationalFeatures:
    """
    A class that provides educational features and metric explanations.

    This class manages the educational aspects of the social media analyzer,
    providing detailed explanations of metrics, improvement suggestions,
    and learning resources.

    Attributes:
        llm_manager: LLM manager instance for generating explanations
        metric_categories: Dictionary of metric explanations by category
    """

    def __init__(self, llm_manager: LLMManager):
        """
        Initialize educational features with LLM manager.

        Args:
            llm_manager: Instance of LLMManager for generating explanations
        """
        self.llm_manager = llm_manager
        self.load_metric_explanations()

    def load_metric_explanations(self):
        """
        Load explanations for all metrics used in analysis.

        Initializes a structured dictionary of metric explanations organized by
        category, including descriptions, interpretations, and optimal ranges.
        """
        self.metric_categories = {
            "Basic Metrics": {
                "word_count": {
                    "title": "Word Count",
                    "description": "Number of words in the tweet",
                    "interpretation": {
                        "optimal_range": "15-25 words for highest engagement",
                        "impact": "Affects readability and engagement potential",
                    },
                },
                "unique_words": {
                    "title": "Vocabulary Diversity",
                    "description": "Number of unique words used",
                    "interpretation": {
                        "optimal_range": "60-80% of total words",
                        "impact": "Indicates content richness and creativity",
                    },
                },
            },
            "Sentiment Analysis": {
                "sentiment_score": {
                    "title": "Emotional Tone",
                    "description": "Measures the emotional direction of the content",
                    "interpretation": {
                        "range": {"min": -1, "max": 1},
                        "meaning": {
                            "positive": "Conveys optimism, excitement, or approval",
                            "negative": "Expresses criticism, disappointment, or concern",
                            "neutral": "Presents information without emotional bias",
                        },
                    },
                },
                "subjectivity": {
                    "title": "Opinion vs Fact Balance",
                    "description": "Measures how subjective or objective the content is",
                    "interpretation": {
                        "range": {"min": 0, "max": 1},
                        "impact": "Affects credibility and audience perception",
                    },
                },
            },
            "Linguistic Analysis": {
                "linguistic_pos_noun": {
                    "title": "Noun Usage",
                    "description": "Frequency of nouns in the text",
                    "interpretation": {
                        "optimal_range": "20-30% of total words",
                        "impact": "Indicates information density and topic focus",
                    },
                },
                "linguistic_pos_verb": {
                    "title": "Verb Usage",
                    "description": "Frequency of verbs in the text",
                    "interpretation": {
                        "optimal_range": "15-25% of total words",
                        "impact": "Shows action and engagement level",
                    },
                },
                "linguistic_pos_adj": {
                    "title": "Adjective Usage",
                    "description": "Frequency of adjectives in the text",
                    "interpretation": {
                        "optimal_range": "10-20% of total words",
                        "impact": "Indicates descriptive richness",
                    },
                },
                "linguistic_pos_adv": {
                    "title": "Adverb Usage",
                    "description": "Frequency of adverbs in the text",
                    "interpretation": {
                        "optimal_range": "5-15% of total words",
                        "impact": "Shows modification of actions and descriptions",
                    },
                },
                "linguistic_pos_propn": {
                    "title": "Proper Noun Usage",
                    "description": "Frequency of proper nouns in the text",
                    "interpretation": {
                        "optimal_range": "5-15% of total words",
                        "impact": "Indicates reference to specific entities",
                    },
                },
                "linguistic_pos_det": {
                    "title": "Determiner Usage",
                    "description": "Frequency of determiners in the text",
                    "interpretation": {
                        "optimal_range": "5-15% of total words",
                        "impact": "Shows specificity in language",
                    },
                },
                "linguistic_pos_pron": {
                    "title": "Pronoun Usage",
                    "description": "Frequency of pronouns in the text",
                    "interpretation": {
                        "optimal_range": "5-15% of total words",
                        "impact": "Indicates reference and personal tone",
                    },
                },
                "linguistic_readability": {
                    "title": "Readability Score",
                    "description": "Measure of how easy the text is to read",
                    "interpretation": {
                        "optimal_range": "60-80",
                        "impact": "Affects content accessibility",
                    },
                },
                "linguistic_formality": {
                    "title": "Formality Score",
                    "description": "Measure of formal vs informal language",
                    "interpretation": {
                        "optimal_range": "-0.5 to 0.5",
                        "impact": "Influences tone and audience perception",
                    },
                },
                "linguistic_pos_adp": {
                    "title": "Adposition Usage",
                    "description": "Frequency of prepositions and postpositions in the text",
                    "interpretation": {
                        "optimal_range": "10-20% of total words",
                        "impact": "Shows relationships between words and complexity of sentence structure",
                    },
                },
                "linguistic_pos_punct": {
                    "title": "Punctuation Usage",
                    "description": "Frequency of punctuation marks in the text",
                    "interpretation": {
                        "optimal_range": "5-15% of total tokens",
                        "impact": "Indicates sentence structure and writing style clarity",
                    },
                },
                "linguistic_pos_sym": {
                    "title": "Symbol Usage",
                    "description": "Frequency of symbols in the text",
                    "interpretation": {
                        "optimal_range": "0-5% of total tokens",
                        "impact": "Shows use of special characters and mathematical/technical content",
                    },
                },
                "linguistic_pos_num": {
                    "title": "Number Usage",
                    "description": "Frequency of numerical tokens in the text",
                    "interpretation": {
                        "optimal_range": "0-10% of total tokens",
                        "impact": "Indicates presence of quantitative information",
                    },
                },
                "linguistic_pos_aux": {
                    "title": "Auxiliary Verb Usage",
                    "description": "Frequency of auxiliary verbs in the text",
                    "interpretation": {
                        "optimal_range": "5-15% of total words",
                        "impact": "Shows complexity of verb phrases and tense structures",
                    },
                },
            },
            "Engagement Metrics": {
                "virality_score": {
                    "title": "Viral Potential",
                    "description": "Likelihood of content being shared widely",
                    "components": [
                        "hashtag usage",
                        "emotional appeal",
                        "call-to-action strength",
                    ],
                    "interpretation": {
                        "ranges": {
                            "low": "0-0.3: Limited viral potential",
                            "medium": "0.3-0.7: Moderate viral potential",
                            "high": "0.7-1.0: Strong viral potential",
                        },
                        "impact": "Indicates overall shareability of content",
                    },
                },
                "call_to_action_strength": {
                    "title": "Call to Action Strength",
                    "description": "Effectiveness of prompting user action",
                    "interpretation": {
                        "ranges": {
                            "low": "0-0.3: Weak or missing CTA",
                            "medium": "0.3-0.7: Moderate CTA strength",
                            "high": "0.7-1.0: Strong, clear CTA",
                        },
                        "impact": "Influences audience response and conversion rates",
                    },
                },
                "emotional_appeal": {
                    "title": "Emotional Appeal",
                    "description": "Strength of emotional connection with audience",
                    "interpretation": {
                        "ranges": {
                            "low": "0-0.3: Limited emotional impact",
                            "medium": "0.3-0.7: Moderate emotional resonance",
                            "high": "0.7-1.0: Strong emotional impact",
                        },
                        "impact": "Affects audience connection and memorability",
                    },
                },
                "virality_components_hashtags": {
                    "title": "Hashtag Usage",
                    "description": "Effectiveness of hashtag implementation",
                    "interpretation": {
                        "optimal_range": "1-3 hashtags per tweet",
                        "impact": "Increases content discoverability and reach",
                    },
                },
                "virality_components_mentions": {
                    "title": "Mentions Usage",
                    "description": "Strategic use of @mentions",
                    "interpretation": {
                        "optimal_range": "1-2 mentions per tweet",
                        "impact": "Enhances network engagement and visibility",
                    },
                },
                "virality_components_emoji_density": {
                    "title": "Emoji Density",
                    "description": "Use of emojis in content",
                    "interpretation": {
                        "optimal_range": "1-3 emojis per tweet",
                        "impact": "Adds visual appeal and emotional context",
                    },
                },
                "virality_components_emotional_words": {
                    "title": "Emotional Language",
                    "description": "Use of emotionally charged words",
                    "interpretation": {
                        "optimal_range": "15-25% of content",
                        "impact": "Strengthens emotional connection and engagement",
                    },
                },
                "virality_components_readability": {
                    "title": "Content Readability",
                    "description": "Ease of reading and comprehension",
                    "interpretation": {
                        "optimal_range": "60-80 on Flesch scale",
                        "impact": "Affects content accessibility and reach",
                    },
                },
                "virality_components_length_optimization": {
                    "title": "Length Optimization",
                    "description": "Optimal content length for platform",
                    "interpretation": {
                        "optimal_range": "70-100 characters",
                        "impact": "Influences engagement and readability",
                    },
                },
                "content_density": {
                    "title": "Information Density",
                    "description": "How much meaningful content is packed into the tweet",
                    "interpretation": {
                        "ranges": {
                            "low": "0-0.3: Sparse content",
                            "medium": "0.3-0.7: Balanced density",
                            "high": "0.7-1.0: Dense content",
                        },
                        "impact": "Affects information value and engagement",
                    },
                },
            },
            "Content Analysis": {
                "topic_categories": {
                    "title": "Topic Categories",
                    "description": "Main topics detected in the content",
                    "interpretation": {
                        "categories": {
                            "Technology": "Technical or digital subject matter",
                            "Business": "Professional or commercial content",
                            "Social": "Community or relationship focused",
                            "Politics": "Political or policy related",
                            "Entertainment": "Fun or leisure focused",
                            "Health": "Wellness or medical content",
                            "Education": "Learning or instructional content",
                            "General": "Broad or multi-topic content",
                        },
                        "impact": "Helps target content to relevant audiences",
                    },
                },
                "writing_style_tone": {
                    "title": "Writing Style - Tone",
                    "description": "Overall emotional tone of the writing",
                    "interpretation": {
                        "categories": {
                            "Positive": "Optimistic and upbeat content",
                            "Negative": "Critical or concerning content",
                            "Neutral": "Balanced and objective content",
                            "Formal": "Professional and structured content",
                        },
                        "impact": "Influences emotional response and engagement",
                    },
                },
                "writing_style_complexity": {
                    "title": "Writing Style - Complexity",
                    "description": "Level of sophistication in language use",
                    "interpretation": {
                        "ranges": {
                            "low": "0-0.3: Simple, accessible content",
                            "medium": "0.3-0.7: Balanced complexity",
                            "high": "0.7-1.0: Advanced or technical content",
                        },
                        "optimal_range": "0.3-0.7",
                        "impact": "Affects readability and target audience reach",
                    },
                },
                "writing_style_personality": {
                    "title": "Writing Style - Personality",
                    "description": "Distinctive character of the writing",
                    "interpretation": {
                        "types": {
                            "Enthusiastic": "Energetic and passionate tone",
                            "Inquisitive": "Questioning and curious approach",
                            "Personal": "Intimate and conversational style",
                            "Professional": "Formal and business-like tone",
                        },
                        "impact": "Shapes audience connection and brand voice",
                    },
                },
            },
        }

    def explain_metric(
        self, metric_name: str, score: float, analysis_results: dict
    ) -> dict:
        """
        Provide detailed explanation of a specific metric and its current value.

        Args:
            metric_name: Name of the metric to explain
            score: Current value of the metric
            analysis_results: Complete analysis results for context

        Returns:
            dict: Detailed explanation including:
                - name: Metric name
                - description: What the metric measures
                - analysis: LLM-generated explanation
                - interpretation: Guidelines for interpretation
                - components: List of components (if applicable)
        """
        # Special handling for improvement suggestions and general tips
        if metric_name == "general_improvement":
            return {
                "name": "General Improvements",
                "description": "Overall suggestions for enhancing tweet effectiveness",
                "analysis": self.llm_manager.get_educational_content(
                    "Provide general tips for making an already good tweet even better.",
                    "general_improvement",
                ),
            }

        if metric_name in [
            "emotional_impact",
            "engagement_potential",
            "content_density",
        ]:
            # Handle improvement-specific metrics
            improvement_info = {
                "emotional_impact": {
                    "name": "Emotional Impact",
                    "description": "How to enhance emotional resonance",
                },
                "engagement_potential": {
                    "name": "Engagement Potential",
                    "description": "Ways to increase audience engagement",
                },
                "content_density": {
                    "name": "Content Density",
                    "description": "How to optimize information delivery",
                },
            }

            metric_info = improvement_info.get(
                metric_name,
                {
                    "name": metric_name.replace("_", " ").title(),
                    "description": "How to improve this aspect",
                },
            )

            # Get contextual analysis from LLM
            context_prompt = f"""
            Analyze this {metric_name} score of {score} and provide:
            1. What this score indicates about the tweet
            2. Specific areas that need improvement
            3. Actionable suggestions for enhancement
            4. Examples of better approaches
            
            Current analysis context: {analysis_results}
            """

            llm_analysis = self.llm_manager.get_educational_content(
                context_prompt, "improvement_analysis"
            )

            return {
                "name": metric_info["name"],
                "description": metric_info["description"],
                "current_score": score,
                "analysis": llm_analysis,
            }

        # Find the metric in our categories (original behavior for standard metrics)
        metric_info = None
        category_name = None
        for cat_name, metrics in self.metric_categories.items():
            if metric_name in metrics:
                metric_info = metrics[metric_name]
                category_name = cat_name
                break

        if not metric_info:
            return {
                "name": metric_name.replace("_", " ").title(),
                "description": "Metric analysis",
                "analysis": "Metric information not found",
            }

        # Get contextual analysis from LLM
        context_prompt = f"""
        Analyze this {metric_name} score of {score} in the context of social media writing.
        Explain:
        1. What this specific score indicates
        2. Whether this is a strong or weak point
        3. Specific suggestions for improvement if needed
        4. Examples of successful approaches
        
        Current analysis context: {analysis_results}
        """

        llm_analysis = self.llm_manager.get_educational_content(
            context_prompt, "metric_analysis"
        )

        return {
            "name": metric_info["title"],
            "category": category_name,
            "description": metric_info["description"],
            "current_score": score,
            "interpretation": metric_info.get("interpretation", {}),
            "analysis": llm_analysis,
            "components": metric_info.get("components", []),
        }

    def generate_improvement_suggestions(
        self, analysis_results: dict
    ) -> List[Dict[str, str]]:
        """
        Generate specific improvement suggestions based on analysis results.

        Analyzes weak points in the content and provides actionable
        suggestions for improvement.

        Args:
            analysis_results: Complete analysis results

        Returns:
            list: List of improvement suggestions, each containing:
                - metric: Name of the metric to improve
                - current_score: Current metric value
                - suggestions: Specific improvement recommendations
        """
        weak_points = self._identify_weak_points(analysis_results)
        suggestions = []

        for metric, score in weak_points.items():
            prompt = f"""
            Provide specific, actionable improvement suggestions for the {metric} 
            which currently scores {score}. Focus on practical tips that can be 
            implemented immediately.
            """
            suggestion = self.llm_manager.get_educational_content(prompt, "improvement")
            suggestions.append(
                {"metric": metric, "current_score": score, "suggestions": suggestion}
            )

        return suggestions

    def _identify_weak_points(self, analysis_results: dict) -> Dict[str, float]:
        """
        Identify metrics that could be improved in the content.

        Analyzes various metrics against their optimal ranges and identifies
        those that fall below acceptable thresholds.

        Args:
            analysis_results: Complete analysis results

        Returns:
            dict: Mapping of metric names to their current scores for metrics
                 that need improvement
        """
        weak_points = {}

        # Check sentiment balance
        sentiment = analysis_results["sentiment_analysis"]["sentiment_score"]
        if abs(sentiment) < 0.2:
            weak_points["emotional_impact"] = sentiment

        # Check engagement potential
        if (
            analysis_results["content_analysis"]["engagement_potential"][
                "virality_score"
            ]
            < 0.4
        ):
            weak_points["engagement_potential"] = analysis_results["content_analysis"][
                "engagement_potential"
            ]["virality_score"]

        # Check content density
        if analysis_results["content_analysis"]["content_density"] < 0.5:
            weak_points["content_density"] = analysis_results["content_analysis"][
                "content_density"
            ]

        return weak_points

    def highlight_strengths(self, analysis_results: dict) -> List[Dict[str, str]]:
        """
        Identify and explain what the tweet does well.

        Analyzes various metrics to identify particularly strong aspects
        of the content and provides detailed explanations of why they're
        effective.

        Args:
            analysis_results: Complete analysis results

        Returns:
            dict: Analysis of content strengths including:
                - identified_strengths: List of strong aspects
                - detailed_analysis: LLM-generated analysis of strengths
        """
        strengths = []

        # Analyze various metrics to identify strong points
        if abs(analysis_results["sentiment_analysis"]["sentiment_score"]) > 0.6:
            strengths.append(
                {
                    "aspect": "Emotional Impact",
                    "reason": "Strong emotional tone that can drive engagement",
                }
            )

        if analysis_results["content_analysis"]["content_density"] > 0.7:
            strengths.append(
                {
                    "aspect": "Content Quality",
                    "reason": "High information density while maintaining clarity",
                }
            )

        # Get detailed analysis of strengths from LLM
        strengths_prompt = f"""
        Analyze the following strong points of this tweet and explain why they're effective:
        {strengths}
        
        Analysis context: {analysis_results}
        """

        detailed_analysis = self.llm_manager.get_educational_content(
            strengths_prompt, "strengths_analysis"
        )

        return {
            "identified_strengths": strengths,
            "detailed_analysis": detailed_analysis,
        }
