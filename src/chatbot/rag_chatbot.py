"""
RAG-based conversational chatbot with chat history support.
Based on LangChain's qa_chat_history tutorial architecture.
"""
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import Document

from src.config import config
from src.logger import app_logger
from src.rag.data_manager import RAGDataManager


@dataclass
class ChatResponse:
    """Represents a chat response from the bot."""
    response: str
    sources: List[Dict[str, Any]]
    confidence: float
    processing_time: float
    retrieved_docs_count: int
    chat_history_length: int


class RAGChatbot:
    """RAG-based conversational chatbot with memory."""
    
    def __init__(self, rag_manager: RAGDataManager = None, api_key: str = None, 
                 memory_window: int = 10):
        """Initialize the RAG chatbot."""
        self.config = config.get_llm_config()
        
        # Initialize LLM
        if not api_key:
            api_key = config.google_api_key
        
        if not api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable.")
        
        self.llm = ChatGoogleGenerativeAI(
            model=self.config['gemini']['model'],
            temperature=self.config['gemini']['temperature'],
            google_api_key=api_key
        )
        
        # Initialize RAG manager
        self.rag_manager = rag_manager or RAGDataManager()
        
        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            k=memory_window,
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create the conversation chain
        self.chain = self._create_conversation_chain()
        
        app_logger.info(f"RAGChatbot initialized with memory window of {memory_window}")
    
    def _create_conversation_chain(self):
        """Create the conversational RAG chain."""
        # Create the contextualize question prompt
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        # Create contextualize chain
        contextualize_q_chain = contextualize_q_prompt | self.llm | StrOutputParser()
        
        # Create the QA system prompt
        qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer based on the context, just say that you don't know. \
Use three sentences maximum and keep the answer concise.

Context: {context}

Chat History: {chat_history}

Guidelines:
1. Base your answer primarily on the provided context
2. Reference the chat history for continuity when relevant
3. If the context doesn't contain relevant information, clearly state this
4. Be helpful and conversational while staying factual
5. If referencing sources, mention them naturally in your response"""
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        def contextualized_question(input_dict):
            """Get contextualized question based on chat history."""
            if input_dict.get("chat_history"):
                return contextualize_q_chain
            else:
                return input_dict["input"]
        
        def retrieve_docs(input_dict):
            """Retrieve relevant documents."""
            question = input_dict["question"] if isinstance(input_dict, dict) else input_dict
            return self.rag_manager.query_similar(question, n_results=5)
        
        def format_docs(docs):
            """Format retrieved documents for the prompt."""
            if not docs:
                return "No relevant context found in the knowledge base."
            
            formatted = []
            for i, doc in enumerate(docs, 1):
                source_info = ""
                metadata = doc.get('metadata', {})
                if metadata.get('source_url'):
                    source_info = f" (Source: {metadata.get('title', 'Unknown')})"
                elif metadata.get('source_type'):
                    source_info = f" (Type: {metadata.get('source_type', 'Unknown')})"
                
                formatted.append(f"{i}. {doc['document'][:500]}...{source_info}")
            
            return "\n\n".join(formatted)
        
        # Create the final chain
        rag_chain = (
            RunnablePassthrough.assign(
                question=contextualized_question,
            )
            | RunnablePassthrough.assign(
                context=lambda x: format_docs(retrieve_docs(x["question"]))
            )
            | qa_prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def _get_chat_history_messages(self) -> List[BaseMessage]:
        """Get chat history as list of messages."""
        return self.memory.chat_memory.messages
    
    def chat(self, user_input: str) -> ChatResponse:
        """Process a user input and return a response."""
        start_time = time.time()
        
        app_logger.info(f"Processing user input: {user_input[:100]}...")
        
        # Get chat history
        chat_history = self._get_chat_history_messages()
        
        try:
            # Retrieve relevant documents first (for response metadata)
            retrieved_docs = self.rag_manager.query_similar(user_input, n_results=5)
            
            # Invoke the chain
            response = self.chain.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            
            # Save to memory
            self.memory.chat_memory.add_user_message(user_input)
            self.memory.chat_memory.add_ai_message(response)
            
            # Calculate confidence based on retrieved documents
            confidence = self._calculate_confidence(retrieved_docs, response)
            
            processing_time = time.time() - start_time
            
            # Format sources
            sources = []
            for doc in retrieved_docs:
                metadata = doc.get('metadata', {})
                sources.append({
                    'title': metadata.get('title', 'Unknown'),
                    'source_url': metadata.get('source_url', ''),
                    'source_type': metadata.get('source_type', 'unknown'),
                    'similarity': doc.get('similarity', 0.0),
                    'snippet': doc['document'][:200] + "..."
                })
            
            chat_response = ChatResponse(
                response=response,
                sources=sources,
                confidence=confidence,
                processing_time=processing_time,
                retrieved_docs_count=len(retrieved_docs),
                chat_history_length=len(chat_history)
            )
            
            app_logger.info(f"Generated response in {processing_time:.2f}s with confidence {confidence:.2f}")
            return chat_response
            
        except Exception as e:
            app_logger.error(f"Error generating response: {e}")
            
            # Return fallback response
            fallback_response = "I apologize, but I encountered an error while processing your question. Please try rephrasing your question or ask something else."
            
            self.memory.chat_memory.add_user_message(user_input)
            self.memory.chat_memory.add_ai_message(fallback_response)
            
            return ChatResponse(
                response=fallback_response,
                sources=[],
                confidence=0.0,
                processing_time=time.time() - start_time,
                retrieved_docs_count=0,
                chat_history_length=len(chat_history)
            )
    
    def _calculate_confidence(self, retrieved_docs: List[Dict], response: str) -> float:
        """Calculate confidence score based on retrieved documents and response."""
        if not retrieved_docs:
            return 0.1  # Very low confidence if no docs retrieved
        
        # Base confidence on document similarities
        avg_similarity = sum(doc.get('similarity', 0) for doc in retrieved_docs) / len(retrieved_docs)
        
        # Adjust based on response length and content
        response_length_factor = min(len(response) / 200, 1.0)  # Favor longer, detailed responses
        
        # Check if response indicates uncertainty
        uncertainty_phrases = ["i don't know", "not sure", "unclear", "cannot determine", "no information"]
        uncertainty_penalty = 0.3 if any(phrase in response.lower() for phrase in uncertainty_phrases) else 0
        
        confidence = (avg_similarity * 0.7 + response_length_factor * 0.3) - uncertainty_penalty
        return max(0.0, min(1.0, confidence))
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get formatted chat history."""
        history = []
        messages = self._get_chat_history_messages()
        
        for message in messages:
            if isinstance(message, HumanMessage):
                history.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"role": "assistant", "content": message.content})
        
        return history
    
    def clear_history(self):
        """Clear chat history."""
        self.memory.clear()
        app_logger.info("Chat history cleared")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of the current conversation."""
        history = self.get_chat_history()
        
        if not history:
            return {
                "total_exchanges": 0,
                "conversation_length": 0,
                "topics_discussed": [],
                "recent_topics": []
            }
        
        user_messages = [msg["content"] for msg in history if msg["role"] == "user"]
        assistant_messages = [msg["content"] for msg in history if msg["role"] == "assistant"]
        
        return {
            "total_exchanges": len(user_messages),
            "conversation_length": sum(len(msg) for msg in user_messages + assistant_messages),
            "avg_user_message_length": sum(len(msg) for msg in user_messages) / len(user_messages) if user_messages else 0,
            "avg_assistant_message_length": sum(len(msg) for msg in assistant_messages) / len(assistant_messages) if assistant_messages else 0,
            "recent_topics": user_messages[-3:] if len(user_messages) >= 3 else user_messages  # Last 3 questions
        }
    
    def export_conversation(self, filename: str = None) -> str:
        """Export conversation history to file."""
        import json
        from pathlib import Path
        
        if not filename:
            timestamp = int(time.time())
            filename = f"conversation_{timestamp}.json"
        
        history = self.get_chat_history()
        summary = self.get_conversation_summary()
        
        export_data = {
            "export_timestamp": time.time(),
            "conversation_summary": summary,
            "chat_history": history,
            "rag_stats": self.rag_manager.get_collection_stats()
        }
        
        # Create directory if needed
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        app_logger.info(f"Exported conversation to {filename}")
        return filename


def create_interactive_chat_session(rag_manager: RAGDataManager = None) -> None:
    """Create an interactive chat session."""
    try:
        chatbot = RAGChatbot(rag_manager)
        
        print("🤖 RAG Chatbot initialized!")
        print("Type 'quit', 'exit', or 'bye' to end the conversation.")
        print("Type 'clear' to clear chat history.")
        print("Type 'summary' to see conversation summary.")
        print("Type 'export' to export conversation history.")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye! Thanks for chatting!")
                    break
                elif user_input.lower() == 'clear':
                    chatbot.clear_history()
                    print("🧹 Chat history cleared!")
                    continue
                elif user_input.lower() == 'summary':
                    summary = chatbot.get_conversation_summary()
                    print("\n📊 Conversation Summary:")
                    for key, value in summary.items():
                        print(f"  {key}: {value}")
                    continue
                elif user_input.lower() == 'export':
                    filename = chatbot.export_conversation()
                    print(f"💾 Conversation exported to: {filename}")
                    continue
                
                # Get response
                print("🤔 Thinking...")
                response = chatbot.chat(user_input)
                
                print(f"\n🤖 Assistant: {response.response}")
                
                # Show additional info if verbose
                if response.sources:
                    print(f"\n📚 Sources used ({len(response.sources)}):")
                    for i, source in enumerate(response.sources[:3], 1):
                        title = source.get('title', 'Unknown')
                        similarity = source.get('similarity', 0)
                        print(f"  {i}. {title} (similarity: {similarity:.2f})")
                
                print(f"\n⚡ Processing time: {response.processing_time:.2f}s | Confidence: {response.confidence:.2f}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thanks for chatting!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again.")
    
    except Exception as e:
        print(f"Failed to initialize chatbot: {e}")
        print("Make sure your API key is set and the RAG database is accessible.")


if __name__ == "__main__":
    # Example usage
    print("🚀 Starting RAG Chatbot...")
    
    # Initialize RAG manager
    rag_manager = RAGDataManager()
    stats = rag_manager.get_collection_stats()
    
    if stats.get('total_documents', 0) == 0:
        print("⚠️  No documents found in RAG database.")
        print("💡 Run the scraper and data processing pipeline first to populate the knowledge base.")
        print("📝 For testing, the chatbot will still work but responses will be based on general knowledge only.")
    else:
        print(f"✅ Found {stats['total_documents']} documents in knowledge base.")
    
    # Start interactive session
    create_interactive_chat_session(rag_manager)