"""
Sentiment Analysis Service for AI Engine.
Provides sentiment analysis for compliance reports and text content.
"""

import json
import re
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import structlog
from azure.ai.textanalytics.aio import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from azure.identity.aio import DefaultAzureCredential

from ....shared.config import get_settings
from ..models import SentimentType

settings = get_settings()
logger = structlog.get_logger(__name__)


class SentimentAnalyzer:
    """Sentiment analysis service for text processing."""

    def __init__(self):
        self.settings = settings
        self.text_analytics_client = None
        self.azure_credential = None
        self.sentiment_lexicon = self._load_sentiment_lexicon()
        self.emotion_patterns = self._load_emotion_patterns()
        self.domain_specific_terms = self._load_domain_terms()

    def _load_sentiment_lexicon(self) -> Dict[str, Dict[str, float]]:
        """Load sentiment lexicon with word scores."""
        return {
            "positive": {
                "excellent": 0.9,
                "outstanding": 0.9,
                "exceptional": 0.9,
                "great": 0.8,
                "good": 0.7,
                "satisfactory": 0.6,
                "adequate": 0.5,
                "acceptable": 0.5,
                "compliant": 0.7,
                "secure": 0.8,
                "reliable": 0.7,
                "stable": 0.6,
                "efficient": 0.8,
                "optimized": 0.8,
                "improved": 0.7,
                "success": 0.8,
                "achieved": 0.7,
                "resolved": 0.7,
                "approved": 0.7,
                "authorized": 0.7,
                "validated": 0.7,
                "verified": 0.7,
                "certified": 0.8,
                "complies": 0.8,
            },
            "negative": {
                "terrible": -0.9,
                "awful": -0.9,
                "horrible": -0.9,
                "bad": -0.7,
                "poor": -0.7,
                "inadequate": -0.6,
                "unsatisfactory": -0.8,
                "unacceptable": -0.8,
                "non-compliant": -0.9,
                "violation": -0.9,
                "breach": -0.9,
                "insecure": -0.8,
                "vulnerable": -0.8,
                "exposed": -0.8,
                "failed": -0.8,
                "error": -0.6,
                "issue": -0.5,
                "problem": -0.6,
                "risk": -0.5,
                "threat": -0.7,
                "unauthorized": -0.8,
                "denied": -0.7,
                "rejected": -0.7,
                "outdated": -0.6,
                "deprecated": -0.6,
                "critical": -0.8,
            },
            "neutral": {
                "policy": 0.0,
                "procedure": 0.0,
                "standard": 0.0,
                "guideline": 0.0,
                "requirement": 0.0,
                "control": 0.0,
                "framework": 0.0,
                "process": 0.0,
                "system": 0.0,
                "resource": 0.0,
                "configuration": 0.0,
                "setting": 0.0,
                "monitor": 0.0,
                "review": 0.0,
                "assessment": 0.0,
                "analysis": 0.0,
                "evaluation": 0.0,
                "report": 0.0,
            },
        }

    def _load_emotion_patterns(self) -> Dict[str, List[str]]:
        """Load emotion detection patterns."""
        return {
            "concern": [
                r"concerned about",
                r"worried about",
                r"anxious about",
                r"concerned that",
                r"concern is",
                r"major concern",
            ],
            "confidence": [
                r"confident that",
                r"confident in",
                r"assurance that",
                r"certain that",
                r"trust that",
                r"believe that",
            ],
            "frustration": [
                r"frustrated with",
                r"annoyed by",
                r"irritated by",
                r"disappointed with",
                r"upset about",
                r"dissatisfied with",
            ],
            "satisfaction": [
                r"satisfied with",
                r"pleased with",
                r"happy with",
                r"content with",
                r"delighted with",
                r"impressed with",
            ],
            "urgency": [
                r"urgent",
                r"immediate",
                r"critical",
                r"emergency",
                r"asap",
                r"priority",
                r"time-sensitive",
            ],
            "uncertainty": [
                r"uncertain",
                r"unclear",
                r"ambiguous",
                r"confused",
                r"not sure",
                r"unsure",
                r"questionable",
            ],
        }

    def _load_domain_terms(self) -> Dict[str, Dict[str, float]]:
        """Load domain-specific terms and their sentiment weights."""
        return {
            "compliance": {
                "audit_passed": 0.8,
                "audit_failed": -0.8,
                "compliant": 0.7,
                "non_compliant": -0.9,
                "violation": -0.9,
                "adherence": 0.7,
                "breach": -0.9,
                "conformance": 0.7,
            },
            "security": {
                "secure": 0.8,
                "insecure": -0.8,
                "encrypted": 0.7,
                "unencrypted": -0.7,
                "protected": 0.7,
                "exposed": -0.8,
                "authorized": 0.6,
                "unauthorized": -0.8,
                "safe": 0.7,
                "unsafe": -0.7,
            },
            "performance": {
                "fast": 0.7,
                "slow": -0.6,
                "efficient": 0.8,
                "inefficient": -0.7,
                "responsive": 0.7,
                "unresponsive": -0.8,
                "optimized": 0.8,
                "degraded": -0.7,
                "stable": 0.6,
                "unstable": -0.7,
            },
            "cost": {
                "cost_effective": 0.8,
                "expensive": -0.6,
                "savings": 0.7,
                "waste": -0.7,
                "budget": 0.0,
                "over_budget": -0.8,
                "affordable": 0.6,
                "costly": -0.6,
            },
        }

    async def initialize(self) -> None:
        """Initialize the sentiment analyzer."""
        try:
            logger.info("Initializing sentiment analyzer")

            # Initialize Azure Text Analytics client if available
            if self.settings.azure.client_id and self.settings.is_production():
                await self._initialize_azure_text_analytics()

            logger.info("Sentiment analyzer initialized successfully")

        except Exception as e:
            logger.error("Sentiment analyzer initialization failed", error=str(e))
            raise

    async def _initialize_azure_text_analytics(self) -> None:
        """Initialize Azure Text Analytics client."""
        try:
            # For production, use Azure Text Analytics
            endpoint = "https://your-text-analytics-resource.cognitiveservices.azure.com/"

            if hasattr(self.settings, "azure_text_analytics_key"):
                credential = AzureKeyCredential(self.settings.azure_text_analytics_key)
            else:
                self.azure_credential = DefaultAzureCredential()
                credential = self.azure_credential

            self.text_analytics_client = TextAnalyticsClient(
                endpoint=endpoint, credential=credential
            )

            logger.info("Azure Text Analytics client initialized")

        except Exception as e:
            logger.warning("Failed to initialize Azure Text Analytics", error=str(e))

    async def analyze_sentiment(
        self,
        text: str,
        analysis_type: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Analyze sentiment of text."""
        try:
            logger.info(
                "Starting sentiment analysis", text_length=len(text), analysis_type=analysis_type
            )

            # Initialize results
            results = {
                "text_length": len(text),
                "analysis_type": analysis_type,
                "processed_at": datetime.utcnow().isoformat(),
                "sentiment": SentimentType.NEUTRAL,
                "confidence": 0.0,
                "emotions": {},
                "key_phrases": [],
                "sentiment_scores": {},
                "domain_analysis": {},
            }

            # Use Azure Text Analytics if available
            if self.text_analytics_client:
                azure_results = await self._analyze_with_azure(text)
                results.update(azure_results)
            else:
                # Use local sentiment analysis
                local_results = await self._analyze_with_local_methods(text)
                results.update(local_results)

            # Perform emotion analysis
            emotions = await self._analyze_emotions(text)
            results["emotions"] = emotions

            # Extract key phrases
            key_phrases = await self._extract_key_phrases(text)
            results["key_phrases"] = key_phrases

            # Domain-specific analysis
            domain_analysis = await self._analyze_domain_sentiment(text, analysis_type)
            results["domain_analysis"] = domain_analysis

            # Adjust sentiment based on domain analysis
            results = await self._adjust_sentiment_for_domain(results, domain_analysis)

            logger.info(
                "Sentiment analysis completed",
                sentiment=results["sentiment"],
                confidence=results["confidence"],
            )

            return results

        except Exception as e:
            logger.error("Sentiment analysis failed", error=str(e))
            raise

    async def _analyze_with_azure(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using Azure Text Analytics."""
        try:
            # Split text into chunks if too long
            chunks = self._split_text_into_chunks(text, 5000)

            all_results = []
            for chunk in chunks:
                response = await self.text_analytics_client.analyze_sentiment(
                    documents=[chunk], show_opinion_mining=True
                )
                all_results.extend(response)

            # Aggregate results
            if all_results:
                # Get overall sentiment
                sentiments = [doc.sentiment for doc in all_results]
                confidence_scores = [doc.confidence_scores for doc in all_results]

                # Determine overall sentiment
                sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
                total_confidence = {"positive": 0, "negative": 0, "neutral": 0}

                for i, sentiment in enumerate(sentiments):
                    sentiment_counts[sentiment] += 1
                    scores = confidence_scores[i]
                    total_confidence["positive"] += scores.positive
                    total_confidence["negative"] += scores.negative
                    total_confidence["neutral"] += scores.neutral

                # Determine overall sentiment
                overall_sentiment = max(sentiment_counts, key=sentiment_counts.get)
                avg_confidence = total_confidence[overall_sentiment] / len(all_results)

                return {
                    "sentiment": overall_sentiment,
                    "confidence": avg_confidence,
                    "sentiment_scores": {
                        "positive": total_confidence["positive"] / len(all_results),
                        "negative": total_confidence["negative"] / len(all_results),
                        "neutral": total_confidence["neutral"] / len(all_results),
                    },
                }

            return {"sentiment": "neutral", "confidence": 0.0}

        except Exception as e:
            logger.error("Azure sentiment analysis failed", error=str(e))
            return {"sentiment": "neutral", "confidence": 0.0}

    async def _analyze_with_local_methods(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using local methods."""
        try:
            # Tokenize and clean text
            words = self._tokenize_text(text)

            # Calculate sentiment scores
            positive_score = 0.0
            negative_score = 0.0
            neutral_score = 0.0
            total_words = 0

            for word in words:
                word_lower = word.lower()

                # Check positive words
                if word_lower in self.sentiment_lexicon["positive"]:
                    positive_score += self.sentiment_lexicon["positive"][word_lower]
                    total_words += 1

                # Check negative words
                elif word_lower in self.sentiment_lexicon["negative"]:
                    negative_score += abs(self.sentiment_lexicon["negative"][word_lower])
                    total_words += 1

                # Check neutral words
                elif word_lower in self.sentiment_lexicon["neutral"]:
                    neutral_score += 1
                    total_words += 1

            # Normalize scores
            if total_words > 0:
                positive_score /= total_words
                negative_score /= total_words
                neutral_score /= total_words

            # Determine overall sentiment
            if positive_score > negative_score and positive_score > neutral_score:
                sentiment = SentimentType.POSITIVE
                confidence = positive_score
            elif negative_score > positive_score and negative_score > neutral_score:
                sentiment = SentimentType.NEGATIVE
                confidence = negative_score
            else:
                sentiment = SentimentType.NEUTRAL
                confidence = neutral_score

            return {
                "sentiment": sentiment,
                "confidence": min(confidence, 1.0),
                "sentiment_scores": {
                    "positive": positive_score,
                    "negative": negative_score,
                    "neutral": neutral_score,
                },
            }

        except Exception as e:
            logger.error("Local sentiment analysis failed", error=str(e))
            return {"sentiment": SentimentType.NEUTRAL, "confidence": 0.0}

    async def _analyze_emotions(self, text: str) -> Dict[str, float]:
        """Analyze emotions in text."""
        try:
            emotions = {
                "concern": 0.0,
                "confidence": 0.0,
                "frustration": 0.0,
                "satisfaction": 0.0,
                "urgency": 0.0,
                "uncertainty": 0.0,
            }

            text_lower = text.lower()

            for emotion, patterns in self.emotion_patterns.items():
                emotion_score = 0.0
                for pattern in patterns:
                    matches = len(re.findall(pattern, text_lower))
                    emotion_score += matches

                # Normalize by text length
                emotions[emotion] = min(emotion_score / (len(text.split()) / 100), 1.0)

            return emotions

        except Exception as e:
            logger.error("Emotion analysis failed", error=str(e))
            return {}

    async def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text."""
        try:
            key_phrases = []

            # Simple key phrase extraction
            # Look for noun phrases and important terms
            words = self._tokenize_text(text)

            # Extract important words (simplified)
            important_words = []
            for word in words:
                word_lower = word.lower()
                if len(word) > 3 and word_lower not in [
                    "the",
                    "and",
                    "or",
                    "but",
                    "with",
                    "for",
                    "from",
                    "that",
                    "this",
                ]:
                    important_words.append(word)

            # Create phrases from consecutive important words
            current_phrase = []
            for word in important_words:
                if len(current_phrase) == 0:
                    current_phrase.append(word)
                elif len(current_phrase) < 3:
                    current_phrase.append(word)
                else:
                    if current_phrase:
                        key_phrases.append(" ".join(current_phrase))
                    current_phrase = [word]

            if current_phrase:
                key_phrases.append(" ".join(current_phrase))

            # Return top phrases
            return key_phrases[:10]

        except Exception as e:
            logger.error("Key phrase extraction failed", error=str(e))
            return []

    async def _analyze_domain_sentiment(
        self, text: str, analysis_type: Optional[str]
    ) -> Dict[str, Any]:
        """Analyze sentiment for specific domains."""
        try:
            domain_analysis = {}

            # Check each domain
            for domain, terms in self.domain_specific_terms.items():
                domain_score = 0.0
                domain_mentions = 0

                text_lower = text.lower()

                for term, weight in terms.items():
                    term_clean = term.replace("_", " ")
                    if term_clean in text_lower:
                        domain_score += weight
                        domain_mentions += 1

                if domain_mentions > 0:
                    domain_analysis[domain] = {
                        "score": domain_score / domain_mentions,
                        "mentions": domain_mentions,
                        "sentiment": (
                            "positive"
                            if domain_score > 0
                            else "negative" if domain_score < 0 else "neutral"
                        ),
                    }

            return domain_analysis

        except Exception as e:
            logger.error("Domain sentiment analysis failed", error=str(e))
            return {}

    async def _adjust_sentiment_for_domain(
        self, results: Dict[str, Any], domain_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adjust sentiment based on domain-specific analysis."""
        try:
            if not domain_analysis:
                return results

            # Calculate domain-weighted sentiment
            domain_weights = {"compliance": 0.3, "security": 0.25, "performance": 0.2, "cost": 0.15}

            weighted_sentiment = 0.0
            total_weight = 0.0

            for domain, analysis in domain_analysis.items():
                weight = domain_weights.get(domain, 0.1)
                domain_score = analysis.get("score", 0.0)

                weighted_sentiment += domain_score * weight
                total_weight += weight

            if total_weight > 0:
                weighted_sentiment /= total_weight

                # Adjust confidence based on domain analysis
                domain_confidence = min(abs(weighted_sentiment), 1.0)

                # Combine with original sentiment
                original_confidence = results.get("confidence", 0.0)
                combined_confidence = (original_confidence + domain_confidence) / 2

                # Adjust sentiment if domain analysis is strong
                if abs(weighted_sentiment) > 0.3:
                    if weighted_sentiment > 0:
                        results["sentiment"] = SentimentType.POSITIVE
                    else:
                        results["sentiment"] = SentimentType.NEGATIVE

                    results["confidence"] = combined_confidence

            return results

        except Exception as e:
            logger.error("Sentiment adjustment failed", error=str(e))
            return results

    def _split_text_into_chunks(self, text: str, max_chars: int) -> List[str]:
        """Split text into chunks for processing."""
        if len(text) <= max_chars:
            return [text]

        chunks = []
        current_chunk = ""

        sentences = re.split(r"[.!?]+", text)

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chars:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple tokenization
        text = re.sub(r"[^\w\s]", " ", text)
        words = text.split()
        return [word for word in words if len(word) > 1]

    def is_ready(self) -> bool:
        """Check if sentiment analyzer is ready."""
        return len(self.sentiment_lexicon) > 0

    async def cleanup(self) -> None:
        """Cleanup resources on shutdown."""
        try:
            if self.text_analytics_client:
                await self.text_analytics_client.close()

            logger.info("Sentiment analyzer cleanup completed")

        except Exception as e:
            logger.error("Sentiment analyzer cleanup failed", error=str(e))
