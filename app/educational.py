from typing import Dict, List
from llm_manager import LLMManager
import streamlit as st

class EducationalFeatures:
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.load_metric_explanations()

    def load_metric_explanations(self):
        """Load explanations for all metrics we analyze"""
        self.metric_categories = {
            'Basic Metrics': {
                'word_count': {
                    'title': 'Word Count',
                    'description': 'Number of words in the tweet',
                    'interpretation': {
                        'optimal_range': '15-25 words for highest engagement',
                        'impact': 'Affects readability and engagement potential'
                    }
                },
                'unique_words': {
                    'title': 'Vocabulary Diversity',
                    'description': 'Number of unique words used',
                    'interpretation': {
                        'optimal_range': '60-80% of total words',
                        'impact': 'Indicates content richness and creativity'
                    }
                }
            },
            'Sentiment Analysis': {
                'sentiment_score': {
                    'title': 'Emotional Tone',
                    'description': 'Measures the emotional direction of the content',
                    'interpretation': {
                        'range': {'min': -1, 'max': 1},
                        'meaning': {
                            'positive': 'Conveys optimism, excitement, or approval',
                            'negative': 'Expresses criticism, disappointment, or concern',
                            'neutral': 'Presents information without emotional bias'
                        }
                    }
                },
                'subjectivity': {
                    'title': 'Opinion vs Fact Balance',
                    'description': 'Measures how subjective or objective the content is',
                    'interpretation': {
                        'range': {'min': 0, 'max': 1},
                        'impact': 'Affects credibility and audience perception'
                    }
                }
            },
            'Engagement Metrics': {
                'virality_score': {
                    'title': 'Viral Potential',
                    'description': 'Likelihood of content being shared widely',
                    'components': ['hashtag usage', 'emotional appeal', 'call-to-action strength']
                },
                'content_density': {
                    'title': 'Information Density',
                    'description': 'How much meaningful content is packed into the tweet',
                    'interpretation': {
                        'high': 'Information-rich but might be harder to digest',
                        'low': 'Easy to read but might lack substance',
                        'optimal': 'Balance between information and readability'
                    }
                }
            }
        }

    def explain_metric(self, metric_name: str, score: float, analysis_results: dict) -> dict:
        """
        Provide detailed explanation of a specific metric and its current value
        """
        # Special handling for improvement suggestions and general tips
        if metric_name == 'general_improvement':
            return {
                "name": "General Improvements",
                "description": "Overall suggestions for enhancing tweet effectiveness",
                "analysis": self.llm_manager.get_educational_content(
                    "Provide general tips for making an already good tweet even better.",
                    "general_improvement"
                )
            }
        
        if metric_name in ['emotional_impact', 'engagement_potential', 'content_density']:
            # Handle improvement-specific metrics
            improvement_info = {
                'emotional_impact': {
                    'name': 'Emotional Impact',
                    'description': 'How to enhance emotional resonance'
                },
                'engagement_potential': {
                    'name': 'Engagement Potential',
                    'description': 'Ways to increase audience engagement'
                },
                'content_density': {
                    'name': 'Content Density',
                    'description': 'How to optimize information delivery'
                }
            }
            
            metric_info = improvement_info.get(metric_name, {
                'name': metric_name.replace('_', ' ').title(),
                'description': 'How to improve this aspect'
            })
            
            # Get contextual analysis from LLM
            context_prompt = f"""
            Analyze this {metric_name} score of {score} and provide:
            1. What this score indicates about the tweet
            2. Specific areas that need improvement
            3. Actionable suggestions for enhancement
            4. Examples of better approaches
            
            Current analysis context: {analysis_results}
            """

            llm_analysis = self.llm_manager.get_educational_content(context_prompt, "improvement_analysis")

            return {
                "name": metric_info["name"],
                "description": metric_info["description"],
                "current_score": score,
                "analysis": llm_analysis
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
                "name": metric_name.replace('_', ' ').title(),
                "description": "Metric analysis",
                "analysis": "Metric information not found"
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

        llm_analysis = self.llm_manager.get_educational_content(context_prompt, "metric_analysis")

        return {
            "name": metric_info["title"],
            "category": category_name,
            "description": metric_info["description"],
            "current_score": score,
            "interpretation": metric_info.get("interpretation", {}),
            "analysis": llm_analysis,
            "components": metric_info.get("components", [])
        }

    def generate_improvement_suggestions(self, analysis_results: dict) -> List[Dict[str, str]]:
        """
        Generate specific improvement suggestions based on the analysis results
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
            suggestions.append({
                "metric": metric,
                "current_score": score,
                "suggestions": suggestion
            })

        return suggestions

    def _identify_weak_points(self, analysis_results: dict) -> Dict[str, float]:
        """
        Identify metrics that could be improved
        """
        weak_points = {}
        
        # Check sentiment balance
        sentiment = analysis_results['sentiment_analysis']['sentiment_score']
        if abs(sentiment) < 0.2:
            weak_points['emotional_impact'] = sentiment

        # Check engagement potential
        if analysis_results['content_analysis']['engagement_potential']['virality_score'] < 0.4:
            weak_points['engagement_potential'] = analysis_results['content_analysis']['engagement_potential']['virality_score']

        # Check content density
        if analysis_results['content_analysis']['content_density'] < 0.5:
            weak_points['content_density'] = analysis_results['content_analysis']['content_density']

        return weak_points

    def highlight_strengths(self, analysis_results: dict) -> List[Dict[str, str]]:
        """
        Identify and explain what the tweet does well
        """
        strengths = []
        
        # Analyze various metrics to identify strong points
        if abs(analysis_results['sentiment_analysis']['sentiment_score']) > 0.6:
            strengths.append({
                "aspect": "Emotional Impact",
                "reason": "Strong emotional tone that can drive engagement"
            })

        if analysis_results['content_analysis']['content_density'] > 0.7:
            strengths.append({
                "aspect": "Content Quality",
                "reason": "High information density while maintaining clarity"
            })

        # Get detailed analysis of strengths from LLM
        strengths_prompt = f"""
        Analyze the following strong points of this tweet and explain why they're effective:
        {strengths}
        
        Analysis context: {analysis_results}
        """
        
        detailed_analysis = self.llm_manager.get_educational_content(
            strengths_prompt, 
            "strengths_analysis"
        )
        
        return {
            "identified_strengths": strengths,
            "detailed_analysis": detailed_analysis
        }