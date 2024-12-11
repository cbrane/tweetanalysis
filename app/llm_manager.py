from openai import OpenAI
import json
import os
from dotenv import load_dotenv


class LLMManager:
    def __init__(self):
        load_dotenv()
        self.client = OpenAI()
        self.load_system_prompts()

    def load_system_prompts(self):
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
        messages = [
            {"role": "system", "content": self.system_prompts["analysis"]},
            {
                "role": "user",
                "content": f"Analysis results: {json.dumps(metrics, indent=2)}",
            },
            {"role": "user", "content": f"Question: {question}"},
        ]

        response = self.client.chat.completions.create(
            model="gpt-4",  # Using GPT-4 instead of GPT-4o
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content

    def get_educational_content(self, prompt: str, content_type: str) -> str:
        """
        Get educational content based on prompt and content type
        """
        messages = [
            {"role": "system", "content": self.system_prompts["educational"]},
            {"role": "user", "content": f"Content type: {content_type}\nPrompt: {prompt}"}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content

    def _get_relevant_metrics(self, focus_area: str) -> list:
        """
        Get relevant metrics for a specific focus area
        """
        metric_mapping = {
            "Writing Style": ["sentiment_score", "content_density", "formality_score"],
            "Engagement": ["virality_score", "emotional_appeal", "call_to_action_strength"],
            "Content Strategy": ["readability_score", "complexity", "topic_relevance"]
        }
        return metric_mapping.get(focus_area, [])

    def _get_metric_value(self, analysis_results: dict, metric: str) -> float:
        """
        Safely extract metric value from analysis results
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
