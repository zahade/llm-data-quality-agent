"""
LLM Agent: Uses LLM to reason about data quality issues and generate insights
"""

import json
from typing import Dict, List, Any
from langchain_groq import ChatGroq
from src.utils import get_api_key, format_stats_for_llm


class LLMAgent:
    """Agent that uses LLM to analyze and explain data quality issues."""
    
    def __init__(self):
        """Initialize the LLM agent."""
        self.api_key = get_api_key()
        
        # Initialize LLM with specific settings
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=self.api_key,
            temperature=0.7,  # Slightly higher for more creative reasoning
            max_tokens=2048,
        )
        
        print("✅ LLM Agent initialized (Groq - Llama 3.3 70B)")
    
    def analyze_data_quality(self, stats: Dict, issues: Dict, sample_data: str = "") -> str:
        """
        Comprehensive analysis of data quality using LLM reasoning.
        
        Args:
            stats: Statistical summary of data
            issues: Detected issues dictionary
            sample_data: Sample of problematic data
            
        Returns:
            LLM-generated analysis report
        """
        print("🤖 LLM Agent analyzing data quality...")
        
        # Format statistics
        stats_text = format_stats_for_llm(stats)
        
        # Format issues
        issues_text = self._format_issues(issues)
        
        # Create comprehensive prompt
        prompt = f"""You are an expert data quality analyst specializing in energy consumption data.

Analyze the following energy dataset and provide a comprehensive assessment:

{stats_text}

{issues_text}

{sample_data}

Please provide:

1. **Overall Data Quality Assessment** (1-10 score with justification)

2. **Critical Issues** (ranked by severity):
   - List the most severe issues first
   - Explain why each is problematic
   - Estimate impact on analysis

3. **Root Cause Analysis**:
   - What likely caused these issues?
   - Are there patterns in the problems?

4. **Recommended Actions** (prioritized):
   - Step-by-step cleaning strategy
   - What to fix first and why
   - What might need manual review

5. **Risk Assessment**:
   - Can this data be salvaged?
   - What percentage is reliable?
   - Any showstoppers for analysis?

Be specific, technical, and actionable. Think step-by-step through the problems."""

        # Get LLM response
        response = self.llm.invoke(prompt)
        
        return response.content
    
    def explain_anomaly(self, anomaly_type: str, value: float, context: Dict) -> str:
        """
        Get LLM explanation for a specific anomaly.
        
        Args:
            anomaly_type: Type of anomaly (e.g., "outlier", "negative")
            value: The anomalous value
            context: Surrounding context (mean, std, etc.)
            
        Returns:
            Natural language explanation
        """
        prompt = f"""You are analyzing energy consumption data.

Anomaly Detected:
- Type: {anomaly_type}
- Value: {value:.2f} kWh
- Dataset Mean: {context.get('mean', 0):.2f} kWh
- Dataset Std Dev: {context.get('std', 0):.2f} kWh
- Expected Range: {context.get('expected_range', 'unknown')}

Question: Why is this value anomalous and what might have caused it?

Provide a concise explanation in 2-3 sentences focusing on:
1. Why this value is unusual for energy consumption
2. Most likely cause (equipment error, actual event, or data issue)
3. Whether it should be corrected or investigated further

Be specific to energy consumption patterns."""

        response = self.llm.invoke(prompt)
        
        return response.content
    
    def generate_cleaning_code(self, issues: Dict, strategy: str = "conservative") -> str:
        """
        Generate Python code to clean the data based on detected issues.
        
        Args:
            issues: Detected issues dictionary
            strategy: Cleaning strategy ("conservative" or "aggressive")
            
        Returns:
            Python code as string
        """
        print(f"🤖 Generating cleaning code ({strategy} strategy)...")
        
        issues_summary = self._format_issues_summary(issues)
        
        prompt = f"""You are a data engineering expert. Generate Python code to clean energy consumption data.

Issues Detected:
{issues_summary}

Strategy: {strategy}
- Conservative: Only fix clear errors, preserve questionable data
- Aggressive: Remove all suspicious values

Generate complete, executable Python code that:
1. Handles each issue type systematically
2. Logs all changes made
3. Preserves original data before modifications
4. Uses pandas DataFrame operations
5. Includes comments explaining each step

Assume the DataFrame is named 'df' and consumption column is 'consumption_kwh'.

Return ONLY the Python code, no explanations outside comments."""

        response = self.llm.invoke(prompt)
        
        # Extract code (remove markdown if present)
        code = response.content
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        
        return code.strip()
    
    def chain_of_thought_analysis(self, question: str, data_context: str) -> str:
        """
        Use chain-of-thought reasoning to answer complex questions about the data.
        
        Args:
            question: Question about the data
            data_context: Relevant data context
            
        Returns:
            Step-by-step reasoning and answer
        """
        prompt = f"""You are analyzing energy consumption data. Use step-by-step reasoning to answer this question.

Data Context:
{data_context}

Question: {question}

Think through this step-by-step:
1. What information do I have?
2. What information is missing?
3. What assumptions can I make?
4. What's the logical conclusion?

Provide your reasoning process, then your final answer."""

        response = self.llm.invoke(prompt)
        
        return response.content
    
    def prioritize_issues(self, issues: Dict) -> List[Dict[str, Any]]:
        """
        Use LLM to prioritize which issues to fix first.
        
        Args:
            issues: Detected issues dictionary
            
        Returns:
            Prioritized list of issues with rationale
        """
        issues_text = self._format_issues(issues)
        
        prompt = f"""You are a data quality expert prioritizing data cleaning tasks.

Issues Found:
{issues_text}

Rank these issues from highest to lowest priority for cleaning. For each, provide:
1. Priority rank (1 = highest)
2. Severity (Critical/High/Medium/Low)
3. Reason for priority
4. Estimated effort to fix (Easy/Medium/Hard)

Format your response as a numbered list, most critical first.
Consider: impact on analysis, ease of fixing, risk of data loss."""

        response = self.llm.invoke(prompt)
        
        return response.content
    
    def _format_issues(self, issues: Dict) -> str:
        """Format issues dictionary for LLM prompt."""
        formatted = "Data Quality Issues Detected:\n"
        
        for issue_type, count in issues.items():
            if isinstance(count, int) and count > 0:
                formatted += f"- {issue_type.replace('_', ' ').title()}: {count:,}\n"
        
        return formatted
    
    def _format_issues_summary(self, issues: Dict) -> str:
        """Format issues for code generation prompt."""
        summary = ""
        
        if issues.get('missing_values', 0) > 0:
            summary += f"- {issues['missing_values']} missing values\n"
        if issues.get('negative_values', 0) > 0:
            summary += f"- {issues['negative_values']} negative values (INVALID)\n"
        if issues.get('zero_values', 0) > 0:
            summary += f"- {issues['zero_values']} zero values\n"
        if issues.get('outliers', 0) > 0:
            summary += f"- {issues['outliers']} outliers (>3σ)\n"
        if issues.get('duplicates', 0) > 0:
            summary += f"- {issues['duplicates']} duplicate timestamps\n"
        if issues.get('unit_errors', 0) > 0:
            summary += f"- {issues['unit_errors']} suspected unit errors (>500 kWh)\n"
        
        return summary
    
    def interactive_qa(self, question: str, data_summary: str, conversation_history: List = None) -> str:
        """
        Interactive Q&A about the data with conversation memory.
        
        Args:
            question: User's question
            data_summary: Summary of the dataset
            conversation_history: Previous Q&A pairs
            
        Returns:
            Answer from LLM
        """
        # Build conversation context
        context = f"Dataset Summary:\n{data_summary}\n\n"
        
        if conversation_history:
            context += "Previous conversation:\n"
            for qa in conversation_history[-3:]:  # Last 3 exchanges
                context += f"Q: {qa['question']}\nA: {qa['answer']}\n\n"
        
        context += f"Current Question: {question}\n\nProvide a clear, concise answer based on the data."
        
        response = self.llm.invoke(context)
        
        return response.content