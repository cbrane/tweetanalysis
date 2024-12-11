"""
LLM Manager Module

This module manages interactions with the OpenAI GPT models for generating
explanations and educational content. It handles prompt management and
response generation for the social media analysis system.
"""

from openai import OpenAI
import json
import os
from dotenv import load_dotenv


class LLMManager:
    """
    Manager for LLM-based content generation and analysis.

    This class handles all interactions with OpenAI's models, managing prompts
    and generating responses for various types of analysis and educational
    content.

    Attributes:
        client: OpenAI client instance
        system_prompts: Dictionary of predefined system prompts for different
                       content types
    """

    def __init__(self):
        """
        Initialize the LLM manager with OpenAI client and system prompts.

        Loads environment variables and sets up the OpenAI client and
        system prompts for different types of content generation.
        """
        load_dotenv()
        self.client = OpenAI()
        self.load_system_prompts()

    def load_system_prompts(self):
        """
        Load predefined system prompts for different content types.

        Initializes a dictionary of carefully crafted prompts that guide
        the LLM's response style and content focus.
        """
        self.system_prompts = {
            "analysis": """You are an educational social media analysis assistant.
            Explain metrics clearly and connect them to practical social media usage.
            When asked about technical details, provide step-by-step explanations.
            Focus on being educational while maintaining engagement.""",
            
            "educational": """You are a social media education expert.
            Break down complex concepts into understandable parts.
            Provide real-world examples and best practices.
            Connect technical metrics to practical social media strategies."""
        }

    def get_analysis_explanation(self, metrics: dict, question: str) -> str:
        """
        Generate explanation for specific analysis metrics.

        Args:
            metrics: Dictionary of analysis metrics and their values
            question: User's specific question about the analysis

        Returns:
            str: Detailed explanation addressing the user's question in the
                context of the provided metrics
        """
        messages = [
            {"role": "system", "content": self.system_prompts["analysis"]},
            {
                "role": "user",
                "content": f"Analysis results: {json.dumps(metrics, indent=2)}",
            },
            {"role": "user", "content": f"Question: {question}"},
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content

    def get_educational_content(self, prompt: str, content_type: str) -> str:
        """
        Generate educational content based on prompt and content type.

        Args:
            prompt: Specific prompt for content generation
            content_type: Type of educational content needed (e.g., "improvement",
                        "explanation", "best_practices")

        Returns:
            str: Generated educational content tailored to the specified type
                and prompt
        """
        messages = [
            {"role": "system", "content": self.system_prompts["educational"]},
            {"role": "user", "content": f"Content type: {content_type}\nPrompt: {prompt}"}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content

    def _get_relevant_metrics(self, focus_area: str) -> list:
        """
        Get list of metrics relevant to a specific focus area.

        Args:
            focus_area: Area of analysis (e.g., "Writing Style", "Engagement")

        Returns:
            list: List of metric names relevant to the specified focus area
        """
        metric_mapping = {
            "Writing Style": ["sentiment_score", "content_density", "formality_score"],
            "Engagement": ["virality_score", "emotional_appeal", "call_to_action_strength"],
            "Content Strategy": ["readability_score", "complexity", "topic_relevance"]
        }
        return metric_mapping.get(focus_area, [])

    def _get_metric_value(self, analysis_results: dict, metric: str) -> float:
        """
        Safely extract metric value from analysis results.

        Args:
            analysis_results: Dictionary containing all analysis results
            metric: Name of the metric to extract

        Returns:
            float: Value of the requested metric, or 0.0 if not found
        """
        try:
            if metric in analysis_results.get('sentiment_analysis', {}):
                return analysis_results['sentiment_analysis'][metric]
            elif metric in analysis_results.get('content_analysis', {}):
                return analysis_results['content_analysis'][metric]
            elif metric in analysis_results.get('linguistic_analysis', {}):
                return analysis_results['linguistic_analysis'][metric]
            return 0.0
        except (KeyError, TypeError):
            return 0.0
