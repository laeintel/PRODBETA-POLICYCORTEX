"""
NLP Policy Extractor using Azure OpenAI
Extracts policies, requirements, and compliance rules from documents using advanced NLP
"""

import re
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import asyncio

import structlog
from pydantic import BaseModel, Field, validator
from openai import AsyncAzureOpenAI
from azure.ai.textanalytics.aio import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import spacy
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

logger = structlog.get_logger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

class PolicyType(str, Enum):
    """Types of policies that can be extracted"""
    SECURITY = "security"
    COMPLIANCE = "compliance"
    DATA_GOVERNANCE = "data_governance"
    ACCESS_CONTROL = "access_control"
    COST_MANAGEMENT = "cost_management"
    OPERATIONAL = "operational"
    REGULATORY = "regulatory"

class PolicySeverity(str, Enum):
    """Severity levels for extracted policies"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"

class ExtractedPolicy(BaseModel):
    """Represents an extracted policy from document"""
    policy_id: str
    title: str
    description: str
    policy_type: PolicyType
    severity: PolicySeverity
    requirements: List[str] = Field(default_factory=list)
    conditions: List[Dict[str, Any]] = Field(default_factory=list)
    actions: List[str] = Field(default_factory=list)
    exceptions: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    confidence_score: float = Field(ge=0.0, le=1.0)
    source_text: str
    extracted_at: datetime = Field(default_factory=datetime.utcnow)

class ComplianceRule(BaseModel):
    """Represents a compliance rule extracted from policies"""
    rule_id: str
    name: str
    description: str
    policy_references: List[str] = Field(default_factory=list)
    evaluation_criteria: Dict[str, Any]
    remediation_steps: List[str] = Field(default_factory=list)
    automation_possible: bool = False
    risk_score: int = Field(ge=1, le=10)
    tags: List[str] = Field(default_factory=list)

class NLPPolicyExtractor:
    """
    Extracts policies and compliance rules from documents using Azure OpenAI and NLP
    """
    
    def __init__(self,
                 azure_openai_endpoint: str,
                 azure_openai_key: str,
                 azure_openai_deployment: str,
                 text_analytics_endpoint: Optional[str] = None,
                 text_analytics_key: Optional[str] = None):
        
        # Azure OpenAI client
        self.openai_client = AsyncAzureOpenAI(
            azure_endpoint=azure_openai_endpoint,
            api_key=azure_openai_key,
            api_version="2024-02-15-preview"
        )
        self.deployment_name = azure_openai_deployment
        
        # Azure Text Analytics client (optional)
        if text_analytics_endpoint and text_analytics_key:
            self.text_analytics_client = TextAnalyticsClient(
                endpoint=text_analytics_endpoint,
                credential=AzureKeyCredential(text_analytics_key)
            )
        else:
            self.text_analytics_client = None
            
        # Load spaCy model for additional NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("spaCy model not available, using basic NLP")
            self.nlp = None
            
        # Policy extraction patterns
        self.policy_patterns = {
            'requirement': [
                r'(?:shall|must|required to|need to|should)\s+(.+?)(?:\.|;|$)',
                r'(?:it is required that|it is mandatory that)\s+(.+?)(?:\.|;|$)',
                r'(?:ensure that|verify that|confirm that)\s+(.+?)(?:\.|;|$)'
            ],
            'condition': [
                r'(?:if|when|where|in case)\s+(.+?)(?:then|,|\.|$)',
                r'(?:provided that|given that|assuming that)\s+(.+?)(?:\.|;|$)'
            ],
            'action': [
                r'(?:perform|execute|implement|apply|enforce)\s+(.+?)(?:\.|;|$)',
                r'(?:take action to|proceed to|initiate)\s+(.+?)(?:\.|;|$)'
            ],
            'exception': [
                r'(?:except|unless|excluding|not applicable)\s+(.+?)(?:\.|;|$)',
                r'(?:with the exception of|does not apply to)\s+(.+?)(?:\.|;|$)'
            ]
        }
        
    async def extract_policies(self, 
                              text: str,
                              document_context: Optional[Dict[str, Any]] = None) -> List[ExtractedPolicy]:
        """
        Extract policies from document text using Azure OpenAI and NLP
        
        Args:
            text: Document text to analyze
            document_context: Additional context about the document
            
        Returns:
            List of extracted policies
        """
        # Preprocess text
        preprocessed_text = self._preprocess_text(text)
        
        # Split into chunks for processing
        chunks = self._split_into_chunks(preprocessed_text, max_tokens=3000)
        
        # Extract policies from each chunk
        all_policies = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            # Use Azure OpenAI for policy extraction
            openai_policies = await self._extract_with_openai(chunk, document_context)
            
            # Use pattern matching for additional extraction
            pattern_policies = self._extract_with_patterns(chunk)
            
            # Use NLP for entity and relationship extraction
            if self.nlp:
                nlp_policies = self._extract_with_spacy(chunk)
                pattern_policies.extend(nlp_policies)
            
            # Merge and deduplicate policies
            merged_policies = self._merge_policies(openai_policies, pattern_policies)
            all_policies.extend(merged_policies)
            
        # Post-process and validate policies
        validated_policies = self._validate_policies(all_policies)
        
        # Enhance with text analytics if available
        if self.text_analytics_client:
            validated_policies = await self._enhance_with_text_analytics(
                validated_policies,
                text
            )
            
        return validated_policies
        
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better extraction"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR errors
        text = text.replace('l)', '1)')
        text = text.replace('O)', '0)')
        
        # Normalize bullet points
        text = re.sub(r'[•●▪▫◦‣⁃]', '-', text)
        
        # Add periods to lines that look like headers
        lines = text.split('\n')
        processed_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.endswith(('.', '?', '!', ':', ';')):
                if line[0].isupper() and len(line) < 100:
                    line += '.'
            processed_lines.append(line)
            
        return '\n'.join(processed_lines)
        
    def _split_into_chunks(self, text: str, max_tokens: int = 3000) -> List[str]:
        """Split text into processable chunks"""
        # Estimate tokens (roughly 4 characters per token)
        max_chars = max_tokens * 4
        
        if len(text) <= max_chars:
            return [text]
            
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            paragraph_size = len(paragraph)
            
            if current_size + paragraph_size > max_chars:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_size = paragraph_size
            else:
                current_chunk.append(paragraph)
                current_size += paragraph_size
                
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            
        return chunks
        
    async def _extract_with_openai(self, 
                                  text: str,
                                  context: Optional[Dict[str, Any]] = None) -> List[ExtractedPolicy]:
        """Extract policies using Azure OpenAI"""
        
        system_prompt = """You are an expert policy analyst specializing in extracting compliance and governance policies from documents.
        Extract all policies, requirements, and rules from the provided text.
        
        For each policy, identify:
        1. Title: A clear, concise title for the policy
        2. Description: Detailed description of what the policy entails
        3. Type: security, compliance, data_governance, access_control, cost_management, operational, or regulatory
        4. Severity: critical, high, medium, low, or informational
        5. Requirements: Specific requirements or obligations
        6. Conditions: Any conditions or prerequisites
        7. Actions: Required actions or steps
        8. Exceptions: Any exceptions or exclusions
        
        Return the results as a JSON array of policy objects."""
        
        user_prompt = f"""Extract all policies from the following text:
        
        {text}
        
        Context: {json.dumps(context) if context else 'General policy document'}
        
        Return as JSON array with the structure described."""
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            policies_data = json.loads(content)
            
            # Convert to ExtractedPolicy objects
            policies = []
            for policy_data in policies_data.get('policies', []):
                try:
                    policy = ExtractedPolicy(
                        policy_id=f"pol_{datetime.utcnow().timestamp()}_{len(policies)}",
                        title=policy_data.get('title', 'Untitled Policy'),
                        description=policy_data.get('description', ''),
                        policy_type=PolicyType(policy_data.get('type', 'operational')),
                        severity=PolicySeverity(policy_data.get('severity', 'medium')),
                        requirements=policy_data.get('requirements', []),
                        conditions=policy_data.get('conditions', []),
                        actions=policy_data.get('actions', []),
                        exceptions=policy_data.get('exceptions', []),
                        confidence_score=0.85,  # High confidence for OpenAI extraction
                        source_text=text[:500]  # Store partial source
                    )
                    policies.append(policy)
                except Exception as e:
                    logger.error(f"Error creating policy object: {e}")
                    
            return policies
            
        except Exception as e:
            logger.error(f"OpenAI extraction error: {e}")
            return []
            
    def _extract_with_patterns(self, text: str) -> List[ExtractedPolicy]:
        """Extract policies using regex patterns"""
        policies = []
        sentences = sent_tokenize(text) if nltk else text.split('.')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check for requirement patterns
            requirements = []
            for pattern in self.policy_patterns['requirement']:
                matches = re.findall(pattern, sentence, re.IGNORECASE)
                requirements.extend(matches)
                
            # Check for condition patterns
            conditions = []
            for pattern in self.policy_patterns['condition']:
                matches = re.findall(pattern, sentence, re.IGNORECASE)
                conditions.extend([{'condition': m} for m in matches])
                
            # Check for action patterns
            actions = []
            for pattern in self.policy_patterns['action']:
                matches = re.findall(pattern, sentence, re.IGNORECASE)
                actions.extend(matches)
                
            # Check for exception patterns
            exceptions = []
            for pattern in self.policy_patterns['exception']:
                matches = re.findall(pattern, sentence, re.IGNORECASE)
                exceptions.extend(matches)
                
            # Create policy if patterns found
            if requirements or conditions or actions:
                policy = ExtractedPolicy(
                    policy_id=f"pat_{datetime.utcnow().timestamp()}_{len(policies)}",
                    title=self._generate_title_from_text(sentence),
                    description=sentence,
                    policy_type=self._infer_policy_type(sentence),
                    severity=self._infer_severity(sentence),
                    requirements=requirements,
                    conditions=conditions,
                    actions=actions,
                    exceptions=exceptions,
                    confidence_score=0.65,  # Medium confidence for pattern extraction
                    source_text=sentence
                )
                policies.append(policy)
                
        return policies
        
    def _extract_with_spacy(self, text: str) -> List[ExtractedPolicy]:
        """Extract policies using spaCy NLP"""
        if not self.nlp:
            return []
            
        policies = []
        doc = self.nlp(text)
        
        # Extract entities and their relationships
        for sent in doc.sents:
            # Look for deontic modality (must, shall, should)
            has_requirement = any(token.text.lower() in ['must', 'shall', 'should', 'required'] 
                                for token in sent)
            
            if has_requirement:
                # Extract subject-verb-object patterns
                subjects = [token.text for token in sent if token.dep_ == "nsubj"]
                verbs = [token.text for token in sent if token.pos_ == "VERB"]
                objects = [token.text for token in sent if token.dep_ in ["dobj", "pobj"]]
                
                if subjects and verbs:
                    policy = ExtractedPolicy(
                        policy_id=f"nlp_{datetime.utcnow().timestamp()}_{len(policies)}",
                        title=f"{' '.join(subjects)} {' '.join(verbs[:1])}",
                        description=sent.text,
                        policy_type=self._infer_policy_type(sent.text),
                        severity=self._infer_severity(sent.text),
                        requirements=[sent.text],
                        confidence_score=0.70,  # Medium-high confidence for NLP extraction
                        source_text=sent.text
                    )
                    policies.append(policy)
                    
        return policies
        
    def _generate_title_from_text(self, text: str) -> str:
        """Generate a concise title from text"""
        # Remove common words
        stop_words = set(stopwords.words('english')) if nltk else set()
        words = word_tokenize(text.lower()) if nltk else text.lower().split()
        
        important_words = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Take first few important words
        title_words = important_words[:5]
        
        return ' '.join(title_words).title() if title_words else "Policy Requirement"
        
    def _infer_policy_type(self, text: str) -> PolicyType:
        """Infer policy type from text content"""
        text_lower = text.lower()
        
        type_keywords = {
            PolicyType.SECURITY: ['security', 'encryption', 'authentication', 'authorization', 'firewall', 'threat'],
            PolicyType.COMPLIANCE: ['compliance', 'regulation', 'audit', 'standard', 'requirement'],
            PolicyType.DATA_GOVERNANCE: ['data', 'privacy', 'retention', 'classification', 'handling'],
            PolicyType.ACCESS_CONTROL: ['access', 'permission', 'role', 'privilege', 'identity'],
            PolicyType.COST_MANAGEMENT: ['cost', 'budget', 'expense', 'optimization', 'savings'],
            PolicyType.OPERATIONAL: ['operational', 'procedure', 'process', 'workflow', 'maintenance'],
            PolicyType.REGULATORY: ['regulatory', 'legal', 'law', 'statute', 'ordinance']
        }
        
        scores = {}
        for policy_type, keywords in type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[policy_type] = score
            
        # Return type with highest score
        if scores:
            return max(scores, key=scores.get)
        return PolicyType.OPERATIONAL
        
    def _infer_severity(self, text: str) -> PolicySeverity:
        """Infer policy severity from text content"""
        text_lower = text.lower()
        
        severity_keywords = {
            PolicySeverity.CRITICAL: ['critical', 'emergency', 'immediate', 'severe', 'catastrophic'],
            PolicySeverity.HIGH: ['high', 'important', 'significant', 'major', 'urgent'],
            PolicySeverity.MEDIUM: ['medium', 'moderate', 'standard', 'normal'],
            PolicySeverity.LOW: ['low', 'minor', 'minimal', 'optional'],
            PolicySeverity.INFORMATIONAL: ['informational', 'advisory', 'guideline', 'recommendation']
        }
        
        for severity, keywords in severity_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return severity
                
        # Default based on requirement strength
        if any(word in text_lower for word in ['must', 'shall', 'required']):
            return PolicySeverity.HIGH
        elif any(word in text_lower for word in ['should', 'recommended']):
            return PolicySeverity.MEDIUM
        else:
            return PolicySeverity.LOW
            
    def _merge_policies(self, 
                       openai_policies: List[ExtractedPolicy],
                       pattern_policies: List[ExtractedPolicy]) -> List[ExtractedPolicy]:
        """Merge and deduplicate policies from different sources"""
        all_policies = openai_policies + pattern_policies
        
        # Deduplicate based on similarity
        unique_policies = []
        seen_descriptions = set()
        
        for policy in all_policies:
            # Simple deduplication based on description similarity
            description_key = policy.description[:100].lower()
            
            if description_key not in seen_descriptions:
                seen_descriptions.add(description_key)
                unique_policies.append(policy)
            else:
                # Merge with existing policy if similar
                for existing in unique_policies:
                    if existing.description[:100].lower() == description_key:
                        # Merge requirements, conditions, etc.
                        existing.requirements.extend(policy.requirements)
                        existing.conditions.extend(policy.conditions)
                        existing.actions.extend(policy.actions)
                        existing.exceptions.extend(policy.exceptions)
                        
                        # Update confidence score (average)
                        existing.confidence_score = (existing.confidence_score + policy.confidence_score) / 2
                        break
                        
        return unique_policies
        
    def _validate_policies(self, policies: List[ExtractedPolicy]) -> List[ExtractedPolicy]:
        """Validate and clean extracted policies"""
        validated = []
        
        for policy in policies:
            # Remove duplicates in lists
            policy.requirements = list(set(policy.requirements))
            policy.actions = list(set(policy.actions))
            policy.exceptions = list(set(policy.exceptions))
            
            # Remove empty strings
            policy.requirements = [r for r in policy.requirements if r.strip()]
            policy.actions = [a for a in policy.actions if a.strip()]
            policy.exceptions = [e for e in policy.exceptions if e.strip()]
            
            # Validate minimum content
            if policy.title and (policy.description or policy.requirements):
                validated.append(policy)
                
        return validated
        
    async def _enhance_with_text_analytics(self,
                                          policies: List[ExtractedPolicy],
                                          full_text: str) -> List[ExtractedPolicy]:
        """Enhance policies with Azure Text Analytics insights"""
        if not self.text_analytics_client:
            return policies
            
        try:
            # Analyze sentiment
            sentiment_result = await self.text_analytics_client.analyze_sentiment(
                documents=[{"id": "1", "text": full_text[:5000]}]
            )
            
            # Extract key phrases
            key_phrases_result = await self.text_analytics_client.extract_key_phrases(
                documents=[{"id": "1", "text": full_text[:5000]}]
            )
            
            # Entity recognition
            entities_result = await self.text_analytics_client.recognize_entities(
                documents=[{"id": "1", "text": full_text[:5000]}]
            )
            
            # Enhance policies with insights
            for policy in policies:
                policy.metadata['sentiment'] = sentiment_result[0].sentiment if sentiment_result else None
                policy.metadata['key_phrases'] = key_phrases_result[0].key_phrases[:10] if key_phrases_result else []
                
                # Add relevant entities as tags
                if entities_result:
                    relevant_entities = [e.text for e in entities_result[0].entities 
                                       if e.category in ['Organization', 'Product', 'Event']]
                    policy.metadata['entities'] = relevant_entities[:5]
                    
        except Exception as e:
            logger.error(f"Text Analytics enhancement error: {e}")
            
        return policies
        
    async def extract_compliance_rules(self,
                                      policies: List[ExtractedPolicy]) -> List[ComplianceRule]:
        """Convert extracted policies into actionable compliance rules"""
        rules = []
        
        for policy in policies:
            # Generate evaluation criteria
            evaluation_criteria = self._generate_evaluation_criteria(policy)
            
            # Determine if automation is possible
            automation_possible = self._check_automation_feasibility(policy)
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(policy)
            
            # Generate remediation steps
            remediation_steps = self._generate_remediation_steps(policy)
            
            rule = ComplianceRule(
                rule_id=f"rule_{policy.policy_id}",
                name=policy.title,
                description=policy.description,
                policy_references=[policy.policy_id],
                evaluation_criteria=evaluation_criteria,
                remediation_steps=remediation_steps,
                automation_possible=automation_possible,
                risk_score=risk_score,
                tags=[policy.policy_type.value, policy.severity.value]
            )
            
            rules.append(rule)
            
        return rules
        
    def _generate_evaluation_criteria(self, policy: ExtractedPolicy) -> Dict[str, Any]:
        """Generate evaluation criteria for compliance checking"""
        criteria = {
            'type': 'composite',
            'operator': 'AND',
            'conditions': []
        }
        
        # Add requirement-based conditions
        for req in policy.requirements:
            criteria['conditions'].append({
                'type': 'requirement',
                'description': req,
                'evaluation_method': 'manual'  # or 'automated' based on content
            })
            
        # Add policy-specific conditions
        for condition in policy.conditions:
            criteria['conditions'].append({
                'type': 'condition',
                'details': condition
            })
            
        return criteria
        
    def _check_automation_feasibility(self, policy: ExtractedPolicy) -> bool:
        """Check if policy can be automated"""
        # Keywords indicating automation possibility
        automation_keywords = [
            'configuration', 'setting', 'parameter', 'value',
            'enable', 'disable', 'deploy', 'install', 'update'
        ]
        
        text = (policy.description + ' '.join(policy.requirements)).lower()
        
        return any(keyword in text for keyword in automation_keywords)
        
    def _calculate_risk_score(self, policy: ExtractedPolicy) -> int:
        """Calculate risk score based on severity and type"""
        severity_scores = {
            PolicySeverity.CRITICAL: 10,
            PolicySeverity.HIGH: 8,
            PolicySeverity.MEDIUM: 5,
            PolicySeverity.LOW: 3,
            PolicySeverity.INFORMATIONAL: 1
        }
        
        base_score = severity_scores.get(policy.severity, 5)
        
        # Adjust based on policy type
        if policy.policy_type in [PolicyType.SECURITY, PolicyType.COMPLIANCE]:
            base_score = min(10, base_score + 2)
            
        return base_score
        
    def _generate_remediation_steps(self, policy: ExtractedPolicy) -> List[str]:
        """Generate remediation steps based on policy actions"""
        steps = []
        
        # Add policy-defined actions as steps
        for i, action in enumerate(policy.actions, 1):
            steps.append(f"Step {i}: {action}")
            
        # Add generic steps if no specific actions
        if not steps:
            steps = [
                f"Review current configuration against policy: {policy.title}",
                "Identify non-compliant resources or settings",
                "Apply required changes to achieve compliance",
                "Document changes and verify compliance",
                "Schedule regular compliance reviews"
            ]
            
        return steps