\# 🤖 LLM Data Quality Agent



An autonomous AI-powered system for detecting, analyzing, and fixing data quality issues in energy consumption datasets using Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG).

## 📸 Screenshots

### Dashboard Overview
![Dashboard](screenshots/01_dashboard_overview.png)
*Clean, professional interface with 5 main tabs for different analysis stages*

### Data Quality Analysis
![Quality Analysis](screenshots/03_quality_analysis_summary.png)
*Automated detection of 6 types of data quality issues*

![Visualizations](screenshots/04_quality_analysis_charts.png)
*Interactive charts showing issue distribution and consumption patterns*

### AI-Powered Insights
![LLM Analysis](screenshots/05_ai_insights_report.png)
*LLM generates comprehensive quality assessment with scoring and root cause analysis*

![Action Plan](screenshots/06_ai_prioritized_actions.png)
*Prioritized recommendations ranked by severity and effort*

### Automated Data Cleaning
![Cleaning Summary](screenshots/07_data_cleaning_summary.png)
*Detailed log of all cleaning operations performed*

![Before/After](screenshots/08_data_cleaning_before_after.png)
*Statistical comparison showing data quality improvement*

### Interactive Q&A
![Q&A](screenshots/09_interactive_qa.png)
*Natural language conversation with LLM about your data*

\## 🎯 Project Overview



This project demonstrates advanced AI integration by combining statistical analysis with LLM reasoning to create an intelligent data quality agent. The system autonomously:



\- Detects data quality issues (missing values, outliers, anomalies, duplicates)

\- Uses LLMs to explain problems and recommend solutions

\- Implements RAG to query domain-specific documentation

\- Automatically cleans data with configurable strategies

\- Provides interactive Q\&A about the dataset



\## 🚀 Key Features



\### 1. \*\*Autonomous Data Analysis\*\*

\- Statistical anomaly detection using z-scores and distribution analysis

\- Identifies 6 types of issues: missing values, negative values, outliers, duplicates, unit errors, zero values

\- Real-time visualization of data quality metrics



\### 2. \*\*LLM-Powered Insights\*\*

\- Uses Groq API (Llama 3.3 70B) for intelligent reasoning

\- Generates comprehensive quality assessment reports

\- Prioritizes issues by severity and impact

\- Provides step-by-step cleaning recommendations



\### 3. \*\*RAG System\*\*

\- Vector database (ChromaDB) storing energy data quality documentation

\- Semantic search using HuggingFace embeddings

\- Context-aware recommendations based on domain knowledge



\### 4. \*\*Automated Data Cleaning\*\*

\- Two strategies: Conservative (preserve data) vs Aggressive (remove suspicious values)

\- Handles missing values, outliers, duplicates, unit conversion errors

\- Detailed logging of all cleaning operations

\- Before/after comparison metrics



\### 5. \*\*Interactive Dashboard\*\*

\- Built with Streamlit for user-friendly interface

\- Real-time data upload and analysis

\- Interactive Q\&A with LLM about your data

\- Download cleaned datasets



\## 🛠️ Technology Stack



| Component | Technology | Purpose |

|-----------|-----------|---------|

| \*\*LLM\*\* | Groq API (Llama 3.3 70B) | AI reasoning and analysis |

| \*\*Vector DB\*\* | ChromaDB | RAG documentation storage |

| \*\*Embeddings\*\* | HuggingFace (all-MiniLM-L6-v2) | Semantic search |

| \*\*Agent Framework\*\* | LangChain | LLM orchestration |

| \*\*Data Processing\*\* | Pandas, NumPy | Data manipulation |

| \*\*Statistics\*\* | SciPy, Scikit-learn | Anomaly detection |

| \*\*Visualization\*\* | Matplotlib, Seaborn | Charts and plots |

| \*\*Web Interface\*\* | Streamlit | Interactive dashboard |



\## 📦 Installation



\### Prerequisites

\- Python 3.9+

\- Groq API key (free tier available)



\### Setup



1\. \*\*Clone the repository\*\*

```bash

git clone https://github.com/yourusername/llm-data-quality-agent.git

cd llm-data-quality-agent

```



2\. \*\*Create virtual environment\*\*

```bash

python -m venv venv

venv\\Scripts\\activate  # Windows

\# source venv/bin/activate  # Mac/Linux

```



3\. \*\*Install dependencies\*\*

```bash

pip install -r requirements.txt

```



4\. \*\*Configure API key\*\*



Create a `.env` file in the project root:

```

GROQ\_API\_KEY=your\_groq\_api\_key\_here

```



Get your free Groq API key at: https://console.groq.com/



5\. \*\*Generate sample data\*\* (optional)

```bash

cd data/sample

python generate\_sample\_data.py

cd ../..

```



\## 🚀 Usage



\### Run the Application

```bash

streamlit run app.py

```



The dashboard will open at `http://localhost:8501`



\### Using the Dashboard



1\. \*\*Load Data\*\*: Upload CSV or use sample data

2\. \*\*Analyze\*\*: Click "Run Analysis" to detect issues

3\. \*\*AI Insights\*\*: Generate LLM-powered quality report

4\. \*\*Clean\*\*: Choose strategy and clean data automatically

5\. \*\*Q\&A\*\*: Ask questions about your data



\### Sample Questions to Ask



\- "What percentage of my data has quality issues?"

\- "Should I trust this data for forecasting?"

\- "What's causing the outliers in my dataset?"

\- "How should I handle the missing values?"



\## 📊 Sample Output



\### Data Quality Analysis

```

🔍 Data Quality Issues Detected:



❌ Missing Values: 438

❌ Negative Values: 20

⚠️  Zero Values: 15

❌ Outliers (>3σ): 30

❌ Duplicate Timestamps: 5

❌ Suspected Unit Errors: 50



📊 Total Issues: 558

```



\### LLM-Generated Report

The agent provides:

\- Overall quality score (1-10)

\- Critical issues ranked by severity

\- Root cause analysis

\- Prioritized action plan

\- Risk assessment



\## 🧪 Project Structure

```

llm-data-quality-agent/

│

├── data/

│   ├── raw/              # Original datasets

│   ├── processed/        # Cleaned datasets

│   └── sample/           # Sample energy data

│

├── src/

│   ├── data\_analyzer.py      # Statistical anomaly detection

│   ├── llm\_agent.py          # LLM reasoning agent

│   ├── rag\_system.py         # RAG implementation

│   ├── cleaning\_engine.py    # Automated cleaning

│   └── utils.py              # Helper functions

│

├── knowledge\_base/

│   └── energy\_docs.txt       # Domain documentation

│

├── app.py                    # Streamlit interface

├── requirements.txt          # Dependencies

└── README.md                 # This file

```



\## 🎓 Key Learnings



This project demonstrates:



\- \*\*LLM Integration\*\*: Programmatic API usage (not just chat interfaces)

\- \*\*Agentic AI\*\*: Building autonomous reasoning workflows

\- \*\*RAG Implementation\*\*: Combining retrieval with generation

\- \*\*Production MLOps\*\*: Modular code, error handling, logging

\- \*\*Energy Domain\*\*: Industry-specific data quality rules



\## 🔮 Future Enhancements



\- \[ ] Add more LLM providers (OpenAI, Anthropic)

\- \[ ] Implement automated testing suite

\- \[ ] Add database connectivity (PostgreSQL, ClickHouse)

\- \[ ] Create API endpoint for programmatic access

\- \[ ] Expand to other data domains (finance, healthcare)

\- \[ ] Add data profiling and schema validation



\## 📝 License



MIT License - Feel free to use for learning and portfolio purposes



\## 👤 Author



\*\*Alzahad Nowshad\*\*

\- GitHub: \[@yourusername](https://github.com/yourusername)

\- LinkedIn: \[alzahad-nowshad](https://linkedin.com/in/alzahad-nowshad)

\- Email: alzahad200@gmail.com



\## 🙏 Acknowledgments



\- Built with \[Groq](https://groq.com/) for fast LLM inference

\- Uses \[LangChain](https://langchain.com/) for AI orchestration

\- Inspired by real-world energy data challenges



---



\*\*⭐ If you find this project helpful, please star it on GitHub!\*\*

