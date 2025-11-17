"""
AI Engine for Organization Assistant
Handles role-based response generation using retrieved context
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime
import os

# Third-party imports
from sentence_transformers import SentenceTransformer

# AWS imports
try:
    import boto3
    import botocore
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    logger.warning("boto3 not available, AWS Bedrock won't work")

from document_processor import VectorStore, DocumentProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserRole(Enum):
    DEVELOPER = "developer"
    SUPPORT = "support"
    MANAGER = "manager"
    GENERAL = "general"

@dataclass
class QueryContext:
    """Context information for a user query"""
    user_role: UserRole
    query: str
    additional_context: str = ""
    filters: Optional[Dict] = None
    max_context_length: int = 4000

@dataclass
class AIResponse:
    """Response from the AI assistant"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence_score: float
    processing_time: float
    role_specific_notes: List[str]
    suggested_actions: List[str]

class RoleBasedPromptBuilder:
    """Builds role-specific prompts for different user types"""
    
    def __init__(self):
        self.role_prompts = {
            UserRole.DEVELOPER: {
                "system_prompt": """You are an AI assistant helping software developers. Your responses should be:

TECHNICAL FOCUS:
- Provide detailed technical implementation details
- Include code examples, configuration snippets, and API usage
- Explain architecture decisions and design patterns
- Focus on HOW to implement, deploy, and configure systems
- Include best practices, standards, and conventions
- Mention performance implications and optimization opportunities

RESPONSE STYLE:
- Use technical terminology appropriately
- Provide step-by-step implementation guides
- Include relevant file paths, configuration keys, and command examples
- Reference specific functions, classes, and modules when applicable
- Suggest debugging techniques and troubleshooting approaches

AVOID:
- High-level business explanations
- User-facing feature descriptions
- Marketing or sales content""",

                "context_instruction": """Based on the technical documentation and code examples provided below, give a comprehensive technical answer that a developer can immediately act upon.""",
                
                "response_format": """Structure your response as:
1. **Technical Overview** - Brief technical summary
2. **Implementation Details** - Step-by-step technical instructions
3. **Code Examples** - Relevant code snippets or configurations
4. **Best Practices** - Technical recommendations and gotchas
5. **Related Resources** - Links to relevant documentation or code files"""
            },

            UserRole.SUPPORT: {
                "system_prompt": """You are an AI assistant helping support engineers resolve customer issues. Your responses should be:

SUPPORT FOCUS:
- Provide clear troubleshooting steps and diagnostic procedures
- Explain common issues and their root causes
- Focus on WHAT to check, HOW to diagnose, and HOW to resolve problems
- Include monitoring, logging, and diagnostic information
- Provide customer-friendly explanations for complex technical issues
- Suggest escalation paths when needed

RESPONSE STYLE:
- Use clear, actionable language
- Provide step-by-step troubleshooting procedures
- Include specific error messages and their meanings
- Suggest multiple approaches (quick fixes vs. permanent solutions)
- Explain impact on users and business operations

AVOID:
- Deep technical implementation details
- Code development instructions
- Architecture discussions unless relevant to troubleshooting""",

                "context_instruction": """Based on the support documentation, troubleshooting guides, and issue reports provided below, give a practical support response that helps resolve customer issues.""",
                
                "response_format": """Structure your response as:
1. **Issue Summary** - Brief description of the problem
2. **Immediate Steps** - Quick diagnostic or temporary fixes
3. **Detailed Troubleshooting** - Systematic investigation steps
4. **Resolution** - Permanent solution if available
5. **Prevention** - How to prevent this issue in the future
6. **Escalation** - When and how to escalate if needed"""
            },

            UserRole.MANAGER: {
                "system_prompt": """You are an AI assistant helping engineering managers and team leads. Your responses should be:

MANAGEMENT FOCUS:
- Provide strategic and operational insights
- Explain business impact and technical risks
- Focus on team processes, planning, and decision-making
- Include timeline estimates and resource requirements
- Balance technical details with business implications
- Suggest organizational and process improvements

RESPONSE STYLE:
- Use business and management terminology
- Provide executive summaries and key takeaways
- Include risk assessments and mitigation strategies
- Suggest team coordination and communication approaches
- Balance technical accuracy with business clarity

AVOID:
- Detailed implementation code
- Step-by-step technical procedures
- Low-level troubleshooting steps""",

                "context_instruction": """Based on the project documentation, process guides, and team information provided below, give a strategic response that helps with management decisions.""",
                
                "response_format": """Structure your response as:
1. **Executive Summary** - Key points for quick understanding
2. **Impact Assessment** - Business and technical implications
3. **Recommendations** - Strategic actions and decisions
4. **Resource Requirements** - Team, time, and technical needs
5. **Risk Mitigation** - Potential issues and prevention strategies
6. **Next Steps** - Immediate and long-term actions"""
            },

            UserRole.GENERAL: {
                "system_prompt": """You are a helpful AI assistant providing comprehensive information about organizational systems and processes. Adapt your response style based on the context and question complexity.""",
                
                "context_instruction": """Based on the documentation provided below, give a comprehensive and balanced answer appropriate for a general audience.""",
                
                "response_format": """Structure your response clearly with appropriate sections based on the question type."""
            }
        }

    def build_prompt(self, query_context: QueryContext, retrieved_docs: List[Dict]) -> str:
        """Build role-specific prompt with context"""
        
        role = query_context.user_role
        role_config = self.role_prompts[role]
        
        # Build context from retrieved documents
        context_text = self._build_context_text(retrieved_docs, query_context.max_context_length)
        
        # Construct the full prompt
        prompt = f"""{role_config['system_prompt']}

{role_config['context_instruction']}

CONTEXT INFORMATION:
{context_text}

ADDITIONAL CONTEXT: {query_context.additional_context}

USER QUESTION: {query_context.query}

{role_config['response_format']}

Please provide a comprehensive answer based on the context information above. If the information is insufficient or unclear, state what additional information would be needed."""

        return prompt

    def _build_context_text(self, retrieved_docs: List[Dict], max_length: int) -> str:
        """Build context text from retrieved documents with length limits"""
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(retrieved_docs):
            # Format document information
            source_info = f"Source: {doc['metadata'].get('source', 'unknown')}"
            if doc['metadata'].get('repository'):
                source_info += f" | Repository: {doc['metadata']['repository']}"
            if doc['metadata'].get('file_path'):
                source_info += f" | File: {doc['metadata']['file_path']}"
            if doc['metadata'].get('title'):
                source_info += f" | Title: {doc['metadata']['title']}"
            
            doc_text = f"--- Document {i+1} ---\n{source_info}\nContent: {doc['content']}\n"
            
            # Check if adding this document would exceed limit
            if current_length + len(doc_text) > max_length:
                if context_parts:  # If we have at least one document, break
                    break
                else:  # If this is the first document and it's too long, truncate it
                    available_length = max_length - len(f"--- Document 1 ---\n{source_info}\nContent: ")
                    truncated_content = doc['content'][:available_length] + "...[truncated]"
                    doc_text = f"--- Document 1 ---\n{source_info}\nContent: {truncated_content}\n"
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "\n".join(context_parts)

class AIEngine:
    """Main AI engine for processing queries and generating responses"""
    
    def __init__(self, 
                 vector_store: VectorStore,
                 document_processor: DocumentProcessor,
                 aws_region: str = "us-east-1",
                 model: str = "amazon.titan-text-express-v1"):
        
        # Use AWS Bedrock instead of OpenAI
        self.bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=aws_region
        )
        self.vector_store = vector_store
        self.document_processor = document_processor
        self.model = model
        self.aws_region = aws_region
        self.prompt_builder = RoleBasedPromptBuilder()
        self.embedder = SentenceTransformer('BAAI/bge-large-en-v1.5')
        
        logger.info(f"AI Engine initialized with AWS Bedrock model: {model} in region {aws_region}")

    async def process_query(self, query_context: QueryContext) -> AIResponse:
        """Process a user query and generate role-based response"""
        
        start_time = datetime.now()
        retrieved_docs = []  # Initialize to preserve in exception handler
        
        try:
            # Step 1: Retrieve relevant documents
            retrieved_docs = await self._retrieve_relevant_docs(query_context)
            
            if not retrieved_docs:
                return AIResponse(
                    answer="I don't have enough information to answer your question. Please provide more specific details or check if the relevant documentation is available in the system.",
                    sources=[],
                    confidence_score=0.0,
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    role_specific_notes=["No relevant documents found"],
                    suggested_actions=["Try rephrasing your question", "Check if documentation exists for this topic"]
                )
            
            # Step 2: Build role-specific prompt
            prompt = self.prompt_builder.build_prompt(query_context, retrieved_docs)
            
            # Step 3: Generate AI response
            ai_response_text = await self._generate_response(prompt)
            
            # Step 4: Calculate confidence and extract additional info
            confidence_score = self._calculate_confidence(retrieved_docs, query_context.query)
            role_notes, suggested_actions = self._extract_role_specific_info(
                query_context.user_role, 
                retrieved_docs, 
                ai_response_text
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AIResponse(
                answer=ai_response_text,
                sources=self._format_sources(retrieved_docs),
                confidence_score=confidence_score,
                processing_time=processing_time,
                role_specific_notes=role_notes,
                suggested_actions=suggested_actions
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            
            # If we have retrieved docs, include them even on error
            formatted_sources = self._format_sources(retrieved_docs) if retrieved_docs else []
            confidence = self._calculate_confidence(retrieved_docs, query_context.query) if retrieved_docs else 0.0
            
            # Create a helpful fallback response with sources
            if retrieved_docs:
                # Build a simple context-based answer from sources
                context_summary = self._build_simple_summary(retrieved_docs)
                answer = f"I found {len(retrieved_docs)} relevant documents but encountered an error generating a detailed response: {str(e)}\n\nHere's what I found:\n\n{context_summary}\n\nPlease check the sources below for more details."
                notes = [f"Found {len(retrieved_docs)} relevant documents", "AI generation failed - showing raw sources"]
            else:
                answer = f"I encountered an error while processing your question: {str(e)}. Please try again or rephrase your question."
                notes = ["Error occurred during processing"]
            
            return AIResponse(
                answer=answer,
                sources=formatted_sources,
                confidence_score=confidence,
                processing_time=(datetime.now() - start_time).total_seconds(),
                role_specific_notes=notes,
                suggested_actions=["Review the source documents below", "Try rephrasing your question", "Contact system administrator if error persists"]
            )

    async def _retrieve_relevant_docs(self, query_context: QueryContext) -> List[Dict]:
        """Retrieve relevant documents based on query and role"""
        
        logger.info(f"\n{'='*80}")
        logger.info(f"RETRIEVING DOCUMENTS FOR QUERY")
        logger.info(f"{'='*80}")
        logger.info(f"Query: {query_context.query}")
        logger.info(f"User Role: {query_context.user_role.value}")
        
        # Search in role-specific collection first, then general
        results = await self.vector_store.search_similar(
            query=query_context.query,
            user_role=query_context.user_role.value,
            n_results=15,  # Get more results for better selection
            filters=query_context.filters,
            processor=self.document_processor
        )
        
        logger.info(f"Initial retrieval: {len(results)} documents found")
        
        # Detect query type and apply source weighting
        query_type = self._detect_query_type(query_context.query)
        results = self._apply_source_weighting(results, query_type)
        
        # Filter and re-rank results based on role relevance
        filtered_results = self._filter_by_role_relevance(results, query_context.user_role)
        
        # Log final selection
        logger.info(f"\nFinal selection: {len(filtered_results[:8])} documents")
        logger.info(f"Source breakdown:")
        source_counts = {}
        for doc in filtered_results[:8]:
            source = doc.get('metadata', {}).get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        for source, count in sorted(source_counts.items()):
            logger.info(f"  {source}: {count} documents")
        logger.info(f"{'='*80}\n")
        
        # Return top 8 results
        return filtered_results[:8]

    def _detect_query_type(self, query: str) -> str:
        """
        Detect whether query is code-related or documentation-related
        Returns: 'code', 'documentation', or 'general'
        """
        query_lower = query.lower()
        
        # Code-related keywords
        code_keywords = [
            'code', 'function', 'class', 'method', 'api', 'endpoint', 'implementation',
            'implement', 'deploy', 'configure', 'setup', 'install', 'repository',
            'commit', 'branch', 'pull request', 'merge', 'build', 'compile',
            'debug', 'test', 'unit test', 'integration', 'library', 'package',
            'module', 'import', 'export', 'variable', 'parameter', 'return',
            'syntax', 'algorithm', 'data structure', 'performance', 'optimize',
            'refactor', 'codebase', 'source code', 'programming', 'script'
        ]
        
        # Documentation/architecture keywords
        doc_keywords = [
            'architecture', 'diagram', 'design', 'workflow', 'process',
            'documentation', 'document', 'specification', 'requirement',
            'plan', 'planning', 'roadmap', 'strategy', 'overview',
            'concept', 'approach', 'philosophy', 'guideline', 'standard',
            'policy', 'procedure', 'protocol', 'framework', 'structure',
            'system design', 'high-level', 'component', 'integration',
            'project', 'epic', 'story', 'task', 'ticket', 'issue'
        ]
        
        # Count keyword matches
        code_score = sum(1 for keyword in code_keywords if keyword in query_lower)
        doc_score = sum(1 for keyword in doc_keywords if keyword in query_lower)
        
        # Determine query type with threshold
        if code_score > doc_score and code_score >= 1:
            logger.info(f"Query classified as CODE-related (score: {code_score} vs {doc_score})")
            return 'code'
        elif doc_score > code_score and doc_score >= 1:
            logger.info(f"Query classified as DOCUMENTATION-related (score: {doc_score} vs {code_score})")
            return 'documentation'
        else:
            logger.info(f"Query classified as GENERAL (code: {code_score}, doc: {doc_score})")
            return 'general'
    
    def _apply_source_weighting(self, results: List[Dict], query_type: str) -> List[Dict]:
        """
        Apply source weighting based on query type
        Boost GitHub sources for code queries, Confluence/Jira for documentation queries
        """
        if query_type == 'general':
            return results  # No weighting for general queries
        
        for result in results:
            metadata = result.get('metadata', {})
            source = metadata.get('source', '').lower()
            
            # Initialize source weight
            source_weight = 0.0
            
            if query_type == 'code':
                # Boost GitHub and repository sources for code queries
                if 'github' in source or 'repository' in source or 'repo' in source:
                    source_weight = 0.3  # Significant boost for GitHub
                    logger.debug(f"Boosting GitHub source for code query: {metadata.get('file_path', 'unknown')}")
                elif 'confluence' in source or 'jira' in source:
                    source_weight = -0.1  # Slight penalty for doc sources
                    
            elif query_type == 'documentation':
                # Boost Confluence and Jira sources for documentation queries
                if 'confluence' in source or 'jira' in source:
                    source_weight = 0.3  # Significant boost for Confluence/Jira
                    logger.debug(f"Boosting Confluence/Jira source for doc query: {metadata.get('title', 'unknown')}")
                elif 'github' in source or 'repository' in source:
                    source_weight = -0.1  # Slight penalty for code sources
            
            # Store the source weight for later use in ranking
            result['source_weight'] = source_weight
        
        return results

    def _filter_by_role_relevance(self, results: List[Dict], user_role: UserRole) -> List[Dict]:
        """Filter and re-rank results based on role relevance"""
        
        role_keywords = {
            UserRole.DEVELOPER: ['code', 'api', 'implementation', 'technical', 'architecture', 'deployment', 'configuration'],
            UserRole.SUPPORT: ['troubleshooting', 'error', 'issue', 'problem', 'solution', 'support', 'diagnostic'],
            UserRole.MANAGER: ['process', 'team', 'planning', 'strategy', 'decision', 'management', 'roadmap']
        }
        
        relevant_keywords = role_keywords.get(user_role, [])
        
        # Score results based on role relevance
        for result in results:
            content = result['content'].lower()
            metadata = result['metadata']
            
            role_score = 0
            
            # Check for role-specific keywords
            for keyword in relevant_keywords:
                if keyword in content:
                    role_score += 1
            
            # Check metadata role tags
            if 'role_tags' in metadata and user_role.value in metadata['role_tags']:
                role_score += 3
            
            # Check content type relevance
            content_type = metadata.get('content_type', '')
            if user_role == UserRole.DEVELOPER and content_type in ['code_snippet', 'api_documentation', 'configuration']:
                role_score += 2
            elif user_role == UserRole.SUPPORT and content_type in ['troubleshooting', 'setup_instructions']:
                role_score += 2
            
            # Add role score to result (normalize by content length to avoid bias)
            result['role_relevance_score'] = role_score / max(len(content.split()), 1)
        
        # Sort by combined score (similarity + role relevance + source weight)
        results.sort(
            key=lambda x: (1 - x['distance']) + x.get('role_relevance_score', 0) + x.get('source_weight', 0), 
            reverse=True
        )
        
        return results

    async def _generate_response(self, prompt: str) -> str:
        """Generate response using AWS Bedrock API"""
        
        try:
            # Add comprehensive system instruction
            system_instruction = """You are an expert AI assistant for organizational knowledge management. Your role is to provide accurate, helpful, and actionable answers based on the provided documentation.

CRITICAL INSTRUCTIONS:
1. **Only use information from the provided context** - Do not make up or hallucinate information
2. **Be specific and actionable** - Provide concrete steps, examples, and references
3. **Cite sources** - Reference specific documents, files, or sections when providing information
4. **Acknowledge limitations** - If information is missing or unclear, explicitly state it
5. **Maintain accuracy** - Technical accuracy is paramount; if unsure, say so
6. **Respect the user's role** - Tailor complexity and focus to their perspective
7. **Be comprehensive but concise** - Cover all relevant aspects without unnecessary verbosity
8. **Structure your response** - Use clear formatting, bullet points, and numbered steps
9. **Highlight critical information** - Use bold for key points and warnings
10. **Provide context** - Explain WHY along with HOW when relevant
11. Dont print code examples, just the text.

SOURCE PRIORITIZATION BASED ON QUERY TYPE:
- **For code-related queries** (implementation, APIs, functions, code structure, technical setup):
  * PRIORITIZE GitHub sources, repositories, and code files
  * Reference specific code files, functions, classes, and implementations
  * GitHub sources should be your PRIMARY reference for technical implementation details
  
- **For documentation, architecture diagrams, process, and planning queries** (system design, workflows, project plans, requirements):
  * PRIORITIZE Confluence pages and Jira tickets
  * Reference architecture diagrams, design documents, and project documentation
  * Confluence/Jira should be your PRIMARY reference for high-level design and process information

- **When both source types are available**: Explain which source is more authoritative for the specific question and reference accordingly

QUALITY STANDARDS:
- Accuracy > Completeness > Speed
- Clarity > Brevity
- Actionable > Theoretical
- Recent information > Outdated information
- Official documentation > Informal notes
- Right source for right question > All sources equally"""

            # Prepare request body based on model type
            if "anthropic.claude" in self.model:
                # Claude format with system message
                request_body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 2000,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "system": system_instruction,
                    "messages": [{"role": "user", "content": prompt}]
                }
            elif "amazon.titan" in self.model:
                # Titan format - prepend system instruction to prompt
                full_prompt = f"{system_instruction}\n\n{prompt}"
                request_body = {
                    "inputText": full_prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": 2000,
                        "temperature": 0.1,
                        "topP": 0.9,
                        "stopSequences": []
                    }
                }
            else:
                raise ValueError(f"Unsupported model: {self.model}")
            
            # Call Bedrock API (synchronous, so wrap in async)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.bedrock_runtime.invoke_model(
                    modelId=self.model,
                    body=json.dumps(request_body)
                )
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            
            # Extract text based on model type
            if "anthropic.claude" in self.model:
                if 'content' in response_body and len(response_body['content']) > 0:
                    return response_body['content'][0]['text'].strip()
            elif "amazon.titan" in self.model:
                if 'results' in response_body and len(response_body['results']) > 0:
                    return response_body['results'][0]['outputText'].strip()
            
            raise ValueError("Unexpected response format from Bedrock")
            
        except Exception as e:
            logger.error(f"Error generating AI response from Bedrock: {e}")
            raise

    def _calculate_confidence(self, retrieved_docs: List[Dict], query: str) -> float:
        """Calculate confidence score based on retrieval quality"""
        
        if not retrieved_docs:
            return 0.0
        
        # Factors affecting confidence:
        # 1. Similarity scores of retrieved documents
        # 2. Number of relevant documents
        # 3. Content quality indicators
        
        similarity_scores = [1 - doc['distance'] for doc in retrieved_docs]
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        
        # Penalty for few documents
        document_factor = min(len(retrieved_docs) / 5, 1.0)
        
        # Bonus for recent documents (skip if date parsing fails)
        recent_docs = 0
        try:
            for doc in retrieved_docs:
                updated_at = doc.get('metadata', {}).get('updated_at')
                if updated_at:
                    try:
                        doc_date = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                        if doc_date.tzinfo:
                            # Make datetime.now() timezone-aware for comparison
                            from datetime import timezone
                            now = datetime.now(timezone.utc)
                        else:
                            now = datetime.now()
                        if (now - doc_date).days < 30:
                            recent_docs += 1
                    except:
                        pass  # Skip problematic dates
        except:
            pass
        recency_factor = 1.0 + (recent_docs / len(retrieved_docs)) * 0.1 if len(retrieved_docs) > 0 else 1.0
        
        confidence = avg_similarity * document_factor * recency_factor
        return min(confidence, 1.0)

    def _extract_role_specific_info(self, user_role: UserRole, retrieved_docs: List[Dict], response: str) -> Tuple[List[str], List[str]]:
        """Extract role-specific notes and suggested actions"""
        
        notes = []
        actions = []
        
        # Analyze the sources for role-specific insights
        doc_types = set()
        sources = set()
        
        for doc in retrieved_docs:
            doc_types.add(doc.get('metadata', {}).get('content_type', 'unknown'))
            sources.add(doc.get('metadata', {}).get('source', 'unknown'))
        
        # Role-specific notes
        if user_role == UserRole.DEVELOPER:
            if 'code_snippet' in doc_types:
                notes.append("Code examples available in sources")
            if 'api_documentation' in doc_types:
                notes.append("API documentation referenced")
            
            actions.extend([
                "Review code examples in referenced files",
                "Check for related test files or documentation",
                "Consider implementation best practices"
            ])
            
        elif user_role == UserRole.SUPPORT:
            if 'troubleshooting' in doc_types:
                notes.append("Troubleshooting guides available")
            if any('error' in doc.get('content', '').lower() for doc in retrieved_docs):
                notes.append("Error cases and solutions documented")
            
            actions.extend([
                "Follow diagnostic steps systematically", 
                "Document issue details for tracking",
                "Escalate if resolution steps don't work"
            ])
            
        elif user_role == UserRole.MANAGER:
            notes.append(f"Information gathered from {len(sources)} different sources")
            actions.extend([
                "Review team processes and documentation",
                "Consider resource allocation for improvements",
                "Plan knowledge sharing sessions"
            ])
        
        return notes, actions

    def _format_sources(self, retrieved_docs: List[Dict]) -> List[Dict[str, Any]]:
        """Format source information for response"""
        
        logger.info(f"Formatting {len(retrieved_docs)} sources")
        
        sources = []
        for idx, doc in enumerate(retrieved_docs):
            metadata = doc.get('metadata', {})
            distance = doc.get('distance', 1)
            
            # Calculate similarity score (lower distance = higher similarity)
            # Distance typically ranges from 0 (identical) to 2 (completely different)
            similarity_score = max(0, 1 - distance)
            
            source = {
                'type': metadata.get('source', 'unknown'),
                'content_type': metadata.get('content_type', 'general'),
                'similarity_score': similarity_score,
                'title': metadata.get('title', metadata.get('file_path', 'Unknown')),
                'distance': distance  # Keep for debugging
            }
            
            # Add source-specific information
            if metadata.get('repository'):
                source['repository'] = metadata['repository']
            if metadata.get('file_path'):
                source['file_path'] = metadata['file_path']
            if metadata.get('url'):
                source['url'] = metadata['url']
            if metadata.get('updated_at'):
                source['last_updated'] = metadata['updated_at']
            
            sources.append(source)
            
            # Log each source before sorting
            logger.debug(f"  Source {idx}: distance={distance:.4f}, similarity={similarity_score:.4f}, title={source['title'][:50]}")
        
        # Sort sources by similarity_score in descending order (highest confidence first)
        logger.info(f"BEFORE SORT - First 3 similarity scores: {[s['similarity_score'] for s in sources[:3]]}")
        sources.sort(key=lambda x: x['similarity_score'], reverse=True)
        logger.info(f"AFTER SORT - First 3 similarity scores: {[s['similarity_score'] for s in sources[:3]]}")
        
        # Log the sorted order
        for idx, s in enumerate(sources[:5]):
            logger.info(f"  Sorted #{idx+1}: {s['title'][:50]} - similarity={s['similarity_score']:.4f}")
        
        return sources
    
    def _build_simple_summary(self, retrieved_docs: List[Dict]) -> str:
        """Build a simple summary from retrieved documents when AI generation fails"""
        
        summary_parts = []
        
        for i, doc in enumerate(retrieved_docs[:3], 1):  # Show top 3
            metadata = doc.get('metadata', {})
            content = doc.get('content', '')[:200]  # First 200 chars
            
            title = metadata.get('file_path') or metadata.get('title', 'Unknown')
            doc_type = metadata.get('doc_type', 'document')
            similarity = 1 - doc.get('distance', 1)
            
            summary_parts.append(
                f"{i}. **{title}** ({doc_type}, {similarity:.1%} match)\n"
                f"   {content}..."
            )
        
        return "\n\n".join(summary_parts)

# Example usage and testing
async def main():
    """Example usage of the AI Engine"""
    
    # This would require actual OpenAI API key and vector store with data
    # vector_store = VectorStore()
    # ai_engine = AIEngine("your-openai-key", vector_store)
    
    # Example query context
    query_context = QueryContext(
        user_role=UserRole.DEVELOPER,
        query="How do I deploy the authentication service?",
        additional_context="Production environment deployment"
    )
    
    # This would generate a response
    # response = await ai_engine.process_query(query_context)
    # print(f"Answer: {response.answer}")
    # print(f"Confidence: {response.confidence_score}")
    # print(f"Sources: {len(response.sources)}")
    
    print("AI Engine example - requires OpenAI API key and populated vector store to run")

if __name__ == "__main__":
    asyncio.run(main())

