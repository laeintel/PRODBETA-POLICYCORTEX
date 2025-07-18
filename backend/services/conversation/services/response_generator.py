"""
Response Generator Service.
Generates intelligent responses using Azure OpenAI based on intent, context, and service data.
"""

import json
from typing import Dict, Any, List, Optional
import structlog
from openai import AsyncAzureOpenAI
import asyncio

from ....shared.config import get_settings
from ..models import (
    ConversationIntent,
    Entity,
    ResponseGenerationContext,
    ResponseGenerationResult
)

settings = get_settings()
logger = structlog.get_logger(__name__)


class ResponseGenerator:
    """Generates intelligent responses using Azure OpenAI."""
    
    def __init__(self):
        self.settings = settings
        self.client = None
        self.response_templates = self._load_response_templates()
        self.max_response_length = 2000
        self.temperature = 0.7
    
    def _load_response_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load response templates for different intents."""
        return {
            "greeting": {
                "system_prompt": "You are a friendly Azure governance assistant. Respond to greetings warmly and offer help.",
                "max_tokens": 200,
                "temperature": 0.8
            },
            "help": {
                "system_prompt": "You are an Azure governance expert. Provide helpful guidance and explain capabilities clearly.",
                "max_tokens": 500,
                "temperature": 0.6
            },
            "cost_analysis": {
                "system_prompt": "You are an Azure cost management expert. Help users understand and optimize their Azure costs.",
                "max_tokens": 800,
                "temperature": 0.5
            },
            "policy_query": {
                "system_prompt": "You are an Azure policy and governance expert. Help users understand and implement Azure policies.",
                "max_tokens": 800,
                "temperature": 0.4
            },
            "resource_management": {
                "system_prompt": "You are an Azure resource management expert. Help users manage their Azure resources effectively.",
                "max_tokens": 800,
                "temperature": 0.5
            },
            "security_analysis": {
                "system_prompt": "You are an Azure security expert. Help users understand and improve their Azure security posture.",
                "max_tokens": 800,
                "temperature": 0.4
            },
            "rbac_query": {
                "system_prompt": "You are an Azure RBAC expert. Help users understand and configure role-based access control.",
                "max_tokens": 800,
                "temperature": 0.4
            },
            "network_analysis": {
                "system_prompt": "You are an Azure networking expert. Help users understand and configure Azure networking.",
                "max_tokens": 800,
                "temperature": 0.5
            },
            "optimization_suggestion": {
                "system_prompt": "You are an Azure optimization expert. Provide actionable recommendations for improving Azure resources.",
                "max_tokens": 800,
                "temperature": 0.6
            },
            "general_query": {
                "system_prompt": "You are an Azure governance assistant. Provide helpful information about Azure services and best practices.",
                "max_tokens": 600,
                "temperature": 0.6
            }
        }
    
    async def _get_openai_client(self) -> AsyncAzureOpenAI:
        """Get Azure OpenAI client."""
        if not self.client:
            self.client = AsyncAzureOpenAI(
                api_key=self.settings.ai.azure_openai_key,
                api_version=self.settings.ai.azure_openai_api_version,
                azure_endpoint=self.settings.ai.azure_openai_endpoint
            )
        return self.client
    
    async def generate_response(
        self,
        message: str,
        intent: ConversationIntent,
        entities: List[Entity],
        context: ResponseGenerationContext,
        service_data: Optional[Dict[str, Any]] = None
    ) -> ResponseGenerationResult:
        """Generate intelligent response based on input and context."""
        try:
            # Get template for intent
            template = self.response_templates.get(
                intent.value, 
                self.response_templates["general_query"]
            )
            
            # Build response prompt
            response_prompt = await self._build_response_prompt(
                message, intent, entities, context, service_data
            )
            
            # Generate response using OpenAI
            response_text = await self._generate_ai_response(
                response_prompt, template
            )
            
            # Generate suggestions and follow-up questions
            suggestions = await self._generate_suggestions(
                intent, entities, context, service_data
            )
            
            follow_up_questions = await self._generate_follow_up_questions(
                intent, entities, context
            )
            
            # Calculate confidence based on context and service data
            confidence = self._calculate_response_confidence(
                intent, entities, context, service_data
            )
            
            return ResponseGenerationResult(
                message=response_text,
                suggestions=suggestions,
                follow_up_questions=follow_up_questions,
                confidence=confidence,
                data=service_data
            )
            
        except Exception as e:
            logger.error(
                "response_generation_failed",
                error=str(e),
                intent=intent.value,
                message=message
            )
            
            # Fallback response
            return ResponseGenerationResult(
                message=self._generate_fallback_response(intent, message),
                suggestions=["Can you provide more details?", "Would you like me to explain this differently?"],
                follow_up_questions=["What specific aspect would you like to know more about?"],
                confidence=0.3,
                data={}
            )
    
    async def _build_response_prompt(
        self,
        message: str,
        intent: ConversationIntent,
        entities: List[Entity],
        context: ResponseGenerationContext,
        service_data: Optional[Dict[str, Any]]
    ) -> str:
        """Build comprehensive prompt for response generation."""
        
        # Extract relevant information
        entity_text = self._format_entities(entities)
        context_text = self._format_context(context)
        service_data_text = self._format_service_data(service_data)
        
        # Build prompt based on intent
        if intent == ConversationIntent.COST_ANALYSIS:
            return self._build_cost_analysis_prompt(
                message, entity_text, context_text, service_data_text
            )
        elif intent == ConversationIntent.POLICY_QUERY:
            return self._build_policy_query_prompt(
                message, entity_text, context_text, service_data_text
            )
        elif intent == ConversationIntent.RESOURCE_MANAGEMENT:
            return self._build_resource_management_prompt(
                message, entity_text, context_text, service_data_text
            )
        elif intent == ConversationIntent.SECURITY_ANALYSIS:
            return self._build_security_analysis_prompt(
                message, entity_text, context_text, service_data_text
            )
        elif intent == ConversationIntent.RBAC_QUERY:
            return self._build_rbac_query_prompt(
                message, entity_text, context_text, service_data_text
            )
        elif intent == ConversationIntent.GREETING:
            return self._build_greeting_prompt(message, context_text)
        elif intent == ConversationIntent.HELP:
            return self._build_help_prompt(message, context_text)
        else:
            return self._build_general_prompt(
                message, entity_text, context_text, service_data_text
            )
    
    def _format_entities(self, entities: List[Entity]) -> str:
        """Format entities for prompt inclusion."""
        if not entities:
            return "No specific entities mentioned."
        
        entity_groups = {}
        for entity in entities:
            entity_type = entity.type.value
            if entity_type not in entity_groups:
                entity_groups[entity_type] = []
            entity_groups[entity_type].append(entity.value)
        
        formatted = []
        for entity_type, values in entity_groups.items():
            formatted.append(f"{entity_type}: {', '.join(values)}")
        
        return "Entities: " + "; ".join(formatted)
    
    def _format_context(self, context: ResponseGenerationContext) -> str:
        """Format context for prompt inclusion."""
        context_parts = []
        
        # Recent conversation
        if context.conversation_history:
            recent_messages = context.conversation_history[-3:]
            context_parts.append(
                "Recent conversation: " + 
                "; ".join([f"{msg['intent']}: {msg['content'][:100]}" for msg in recent_messages])
            )
        
        # Azure context
        if context.azure_context:
            azure_parts = []
            for key, value in context.azure_context.items():
                if value:
                    azure_parts.append(f"{key}: {value}")
            
            if azure_parts:
                context_parts.append("Azure context: " + "; ".join(azure_parts))
        
        # Intent context
        if context.intent_context:
            intent_parts = []
            current_topic = context.intent_context.get("current_topic")
            if current_topic:
                intent_parts.append(f"Current topic: {current_topic}")
            
            recent_intents = context.intent_context.get("recent_intents", [])
            if recent_intents:
                intent_parts.append(f"Recent intents: {', '.join(recent_intents[-3:])}")
            
            if intent_parts:
                context_parts.append("Intent context: " + "; ".join(intent_parts))
        
        return " | ".join(context_parts) if context_parts else "No additional context."
    
    def _format_service_data(self, service_data: Optional[Dict[str, Any]]) -> str:
        """Format service data for prompt inclusion."""
        if not service_data:
            return "No service data available."
        
        formatted_data = []
        for key, value in service_data.items():
            if isinstance(value, (dict, list)):
                formatted_data.append(f"{key}: {json.dumps(value)[:200]}...")
            else:
                formatted_data.append(f"{key}: {value}")
        
        return "Service data: " + "; ".join(formatted_data)
    
    def _build_cost_analysis_prompt(
        self,
        message: str,
        entity_text: str,
        context_text: str,
        service_data_text: str
    ) -> str:
        """Build cost analysis specific prompt."""
        return f"""
        User is asking about Azure costs and spending. Provide a helpful, accurate response.
        
        User question: "{message}"
        {entity_text}
        {context_text}
        {service_data_text}
        
        Guidelines:
        - Be specific about cost figures if available in service data
        - Explain cost drivers and factors
        - Provide actionable cost optimization suggestions
        - Include relevant timeframes and comparisons
        - Mention cost alerts and budgets when appropriate
        - Use clear, non-technical language for cost explanations
        
        Respond in a helpful, professional manner with specific recommendations.
        """
    
    def _build_policy_query_prompt(
        self,
        message: str,
        entity_text: str,
        context_text: str,
        service_data_text: str
    ) -> str:
        """Build policy query specific prompt."""
        return f"""
        User is asking about Azure policies and governance. Provide accurate policy information.
        
        User question: "{message}"
        {entity_text}
        {context_text}
        {service_data_text}
        
        Guidelines:
        - Explain policy concepts clearly
        - Provide specific policy recommendations
        - Mention compliance status if available
        - Include remediation steps for policy violations
        - Reference built-in Azure policies when relevant
        - Explain the impact of policy changes
        
        Respond with clear, actionable policy guidance.
        """
    
    def _build_resource_management_prompt(
        self,
        message: str,
        entity_text: str,
        context_text: str,
        service_data_text: str
    ) -> str:
        """Build resource management specific prompt."""
        return f"""
        User is asking about Azure resource management. Provide helpful resource guidance.
        
        User question: "{message}"
        {entity_text}
        {context_text}
        {service_data_text}
        
        Guidelines:
        - Provide specific resource information if available
        - Explain resource relationships and dependencies
        - Suggest resource optimization opportunities
        - Include resource tagging and organization tips
        - Mention resource lifecycle management
        - Provide step-by-step instructions when appropriate
        
        Respond with practical resource management advice.
        """
    
    def _build_security_analysis_prompt(
        self,
        message: str,
        entity_text: str,
        context_text: str,
        service_data_text: str
    ) -> str:
        """Build security analysis specific prompt."""
        return f"""
        User is asking about Azure security. Provide comprehensive security guidance.
        
        User question: "{message}"
        {entity_text}
        {context_text}
        {service_data_text}
        
        Guidelines:
        - Highlight security vulnerabilities if found in service data
        - Provide specific remediation steps
        - Explain security best practices
        - Mention relevant security services and tools
        - Include compliance and regulatory considerations
        - Prioritize security recommendations by risk level
        
        Respond with actionable security recommendations.
        """
    
    def _build_rbac_query_prompt(
        self,
        message: str,
        entity_text: str,
        context_text: str,
        service_data_text: str
    ) -> str:
        """Build RBAC query specific prompt."""
        return f"""
        User is asking about Azure RBAC and access control. Provide clear access guidance.
        
        User question: "{message}"
        {entity_text}
        {context_text}
        {service_data_text}
        
        Guidelines:
        - Explain role assignments and permissions clearly
        - Provide specific role recommendations
        - Mention principle of least privilege
        - Include custom role considerations
        - Explain scope and inheritance
        - Provide step-by-step access management instructions
        
        Respond with clear RBAC guidance and best practices.
        """
    
    def _build_greeting_prompt(self, message: str, context_text: str) -> str:
        """Build greeting specific prompt."""
        return f"""
        User is greeting you. Respond warmly and offer assistance.
        
        User message: "{message}"
        {context_text}
        
        Guidelines:
        - Respond warmly and professionally
        - Briefly introduce your capabilities
        - Offer to help with Azure governance questions
        - Keep response concise and friendly
        - Mention key areas you can assist with
        
        Respond in a welcoming, helpful manner.
        """
    
    def _build_help_prompt(self, message: str, context_text: str) -> str:
        """Build help specific prompt."""
        return f"""
        User is asking for help. Provide comprehensive assistance information.
        
        User message: "{message}"
        {context_text}
        
        Guidelines:
        - Explain your capabilities clearly
        - Provide examples of questions you can answer
        - Mention key Azure governance areas you cover
        - Include tips for getting the best responses
        - Offer to help with specific topics
        
        Respond with helpful guidance about using the assistant.
        """
    
    def _build_general_prompt(
        self,
        message: str,
        entity_text: str,
        context_text: str,
        service_data_text: str
    ) -> str:
        """Build general query prompt."""
        return f"""
        User has a general Azure governance question. Provide helpful information.
        
        User question: "{message}"
        {entity_text}
        {context_text}
        {service_data_text}
        
        Guidelines:
        - Provide accurate Azure information
        - Include relevant examples and best practices
        - Offer additional resources when helpful
        - Keep response focused and actionable
        - Suggest related topics the user might find useful
        
        Respond with helpful, accurate information.
        """
    
    async def _generate_ai_response(
        self,
        prompt: str,
        template: Dict[str, Any]
    ) -> str:
        """Generate AI response using Azure OpenAI."""
        try:
            client = await self._get_openai_client()
            
            response = await client.chat.completions.create(
                model=self.settings.ai.azure_openai_deployment,
                messages=[
                    {"role": "system", "content": template["system_prompt"]},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=template.get("max_tokens", 800),
                temperature=template.get("temperature", 0.6),
                top_p=0.95,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error("ai_response_generation_failed", error=str(e))
            raise
    
    async def _generate_suggestions(
        self,
        intent: ConversationIntent,
        entities: List[Entity],
        context: ResponseGenerationContext,
        service_data: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate follow-up suggestions based on intent and context."""
        suggestions = []
        
        # Intent-specific suggestions
        if intent == ConversationIntent.COST_ANALYSIS:
            suggestions.extend([
                "Show me cost optimization recommendations",
                "What are my highest cost resources?",
                "Set up cost alerts for my subscription"
            ])
        elif intent == ConversationIntent.POLICY_QUERY:
            suggestions.extend([
                "Show me policy violations",
                "Recommend policies for my resources",
                "Help me create a custom policy"
            ])
        elif intent == ConversationIntent.RESOURCE_MANAGEMENT:
            suggestions.extend([
                "Show me unused resources",
                "Help me organize my resources",
                "Recommend resource tagging strategy"
            ])
        elif intent == ConversationIntent.SECURITY_ANALYSIS:
            suggestions.extend([
                "Show me security vulnerabilities",
                "Recommend security improvements",
                "Help me configure security policies"
            ])
        elif intent == ConversationIntent.RBAC_QUERY:
            suggestions.extend([
                "Show me user permissions",
                "Recommend role assignments",
                "Help me create custom roles"
            ])
        else:
            suggestions.extend([
                "Tell me about Azure best practices",
                "Help me optimize my Azure resources",
                "Show me compliance status"
            ])
        
        # Context-based suggestions
        if context.azure_context:
            if context.azure_context.get("current_subscription"):
                suggestions.append("Analyze this subscription in detail")
            if context.azure_context.get("current_resource_group"):
                suggestions.append("Show resources in this resource group")
        
        return suggestions[:3]  # Return top 3 suggestions
    
    async def _generate_follow_up_questions(
        self,
        intent: ConversationIntent,
        entities: List[Entity],
        context: ResponseGenerationContext
    ) -> List[str]:
        """Generate follow-up questions to gather more context."""
        questions = []
        
        # Intent-specific follow-ups
        if intent == ConversationIntent.COST_ANALYSIS:
            questions.extend([
                "Which specific time period are you interested in?",
                "Are you looking at a particular service or resource?",
                "What's your target cost reduction goal?"
            ])
        elif intent == ConversationIntent.POLICY_QUERY:
            questions.extend([
                "Which resources or resource groups should I check?",
                "Are you looking for a specific type of policy?",
                "Do you need help with policy remediation?"
            ])
        elif intent == ConversationIntent.RESOURCE_MANAGEMENT:
            questions.extend([
                "Which type of resources are you focusing on?",
                "Are you looking to optimize or reorganize?",
                "Do you need help with resource lifecycle management?"
            ])
        
        # Entity-based follow-ups
        if not any(e.type.value == "subscription" for e in entities):
            questions.append("Which subscription should I analyze?")
        
        if not any(e.type.value == "resource_group" for e in entities):
            questions.append("Are you interested in a specific resource group?")
        
        return questions[:2]  # Return top 2 questions
    
    def _calculate_response_confidence(
        self,
        intent: ConversationIntent,
        entities: List[Entity],
        context: ResponseGenerationContext,
        service_data: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for the response."""
        confidence = 0.5  # Base confidence
        
        # Intent-based confidence
        if intent in [ConversationIntent.GREETING, ConversationIntent.HELP]:
            confidence += 0.3
        elif intent != ConversationIntent.UNKNOWN:
            confidence += 0.2
        
        # Entity-based confidence
        if entities:
            confidence += min(0.2, len(entities) * 0.05)
        
        # Context-based confidence
        if context.conversation_history:
            confidence += min(0.2, len(context.conversation_history) * 0.05)
        
        # Service data confidence
        if service_data:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _generate_fallback_response(
        self,
        intent: ConversationIntent,
        message: str
    ) -> str:
        """Generate fallback response when AI generation fails."""
        
        fallback_responses = {
            ConversationIntent.COST_ANALYSIS: "I can help you analyze your Azure costs. Could you provide more specific details about what you'd like to know?",
            ConversationIntent.POLICY_QUERY: "I can help you with Azure policies and governance. What specific policy information are you looking for?",
            ConversationIntent.RESOURCE_MANAGEMENT: "I can help you manage your Azure resources. What would you like to do with your resources?",
            ConversationIntent.SECURITY_ANALYSIS: "I can help you with Azure security analysis. What security aspect would you like me to examine?",
            ConversationIntent.RBAC_QUERY: "I can help you with Azure role-based access control. What permissions or roles would you like me to check?",
            ConversationIntent.GREETING: "Hello! I'm your Azure governance assistant. How can I help you today?",
            ConversationIntent.HELP: "I can help you with Azure governance, costs, policies, resources, and security. What would you like to know?",
        }
        
        return fallback_responses.get(
            intent,
            "I understand you're asking about Azure governance. Could you provide more details so I can better assist you?"
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the response generator."""
        try:
            client = await self._get_openai_client()
            
            # Simple test request
            response = await client.chat.completions.create(
                model=self.settings.ai.azure_openai_deployment,
                messages=[
                    {"role": "system", "content": "You are a test assistant."},
                    {"role": "user", "content": "Hello"}
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            return {
                "status": "healthy",
                "model": self.settings.ai.azure_openai_deployment,
                "test_response": response.choices[0].message.content
            }
            
        except Exception as e:
            logger.error("response_generator_health_check_failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e)
            }