"""
RAG System: Retrieval-Augmented Generation for energy data documentation
"""

import os
from typing import List, Dict
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from src.utils import get_api_key


class RAGSystem:
    """RAG system for querying energy data quality documentation."""
    
    def __init__(self, knowledge_base_path: str = "knowledge_base/energy_docs.txt"):
        """
        Initialize RAG system with knowledge base.
        
        Args:
            knowledge_base_path: Path to documentation file
        """
        self.kb_path = knowledge_base_path
        self.api_key = get_api_key()
        
        # Initialize embeddings model
        print("📚 Loading embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize LLM
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=self.api_key,
            temperature=0.3,
        )
        
        # Load and create vector store
        self.vectorstore = None
        self._load_knowledge_base()
        
        print("✅ RAG system initialized")
    
    def _load_knowledge_base(self):
        """Load documentation and create vector store."""
        print("📚 Loading knowledge base...")
        
        # Read documentation
        with open(self.kb_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_text(text)
        print(f"  📄 Split documentation into {len(chunks)} chunks")
        
        # Create vector store
        self.vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        print("  ✅ Vector store created")
    
    def query_documentation(self, query: str, k: int = 3) -> str:
        """
        Query the documentation for relevant information.
        
        Args:
            query: Question or search query
            k: Number of relevant chunks to retrieve
            
        Returns:
            Relevant documentation text
        """
        # Retrieve relevant documents
        docs = self.vectorstore.similarity_search(query, k=k)
        
        # Combine retrieved text
        context = "\n\n".join([doc.page_content for doc in docs])
        
        return context
    
    def get_cleaning_recommendation(self, issue_type: str, context: str = "") -> str:
        """
        Get AI-powered recommendation for fixing a specific issue.
        
        Args:
            issue_type: Type of data quality issue
            context: Additional context about the issue
            
        Returns:
            Recommended solution
        """
        # Query documentation
        doc_context = self.query_documentation(f"{issue_type} data quality issue")
        
        # Create prompt
        prompt = f"""Based on the following energy data quality guidelines:

{doc_context}

Issue Type: {issue_type}
Additional Context: {context}

Provide a concise recommendation on how to fix this issue. Include:
1. Why this issue occurs
2. Recommended solution
3. Any important considerations

Keep the response practical and specific to energy consumption data."""

        # Get LLM response
        response = self.llm.invoke(prompt)
        
        return response.content
    
    def explain_issue(self, issue_type: str, sample_data: str) -> str:
        """
        Get detailed explanation of a data quality issue.
        
        Args:
            issue_type: Type of issue
            sample_data: Sample of problematic data
            
        Returns:
            Explanation from LLM
        """
        doc_context = self.query_documentation(issue_type)
        
        prompt = f"""Based on these energy data quality guidelines:

{doc_context}

I found the following {issue_type} issue in the data:

{sample_data}

Please explain:
1. What this issue means
2. Why it's a problem for energy data analysis
3. How it should be addressed

Be specific and concise."""

        response = self.llm.invoke(prompt)
        
        return response.content