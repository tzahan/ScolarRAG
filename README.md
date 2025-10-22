# 🎓 ScholarRAG

**Retrieval-Augmented Generation (RAG) System for Academic Research**

A complete RAG pipeline built on my Master's thesis: *"Personalized Summarization of Global News: Managing Bias with Large Language Models"*

This project demonstrates end-to-end RAG implementation, from document processing to interactive Q&A, showcasing fundamental LLM engineering skills.

---

## 🎯 Project Overview

### What This Demonstrates

- ✅ **Document Processing**: Intelligent PDF parsing and chunking strategies
- ✅ **Vector Search**: Semantic similarity using embeddings
- ✅ **RAG Pipeline**: Complete retrieval-augmented generation implementation
- ✅ **Evaluation**: Systematic testing and metrics
- ✅ **Deployment**: Interactive Gradio interface

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/tzahan/ScolarRAG.git
cd ScolarRA

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Notebook (First Time)

```bash
# Start Jupyter
jupyter notebook scholar_rag_notebook.ipynb

# Follow the notebook to:
# - Process your thesis PDF
# - Create vector database
# - Run evaluation
# - Generate visualizations
```

**Important**: Edit the notebook configuration:
- Set `Config.PDF_PATH` to your thesis PDF
- Set `OPENAI_API_KEY` environment variables in a .env file:

```.env
OPENAI_API_KEY=your-openai-api-key
```

### 3. Launch the Web Interface

```bash
# After running the notebook, start the Gradio app
python app.py

# Open browser to http://localhost:7860
# Or use the live link (public URL)
```

---

## 📊 Key Results

### System Performance

| Metric | Value |
|--------|-------|
| Total Pages | 77 |
| Total Chunks | 186 |
| Avg Chunk Size | ~200 words |
| Retrieval Accuracy | 62.5% |
| Avg Answer Length | 155 words |

### Chunking Strategy

- **Chunk Size**: 1000 characters
- **Overlap**: 200 characters
- **Rationale**: Optimal balance between context preservation and specificity

### Evaluation Results

Tested on 8 questions across different categories:
- ✅ Overview questions: 100% accuracy
- ✅ Technical questions: 75% accuracy  
- ✅ Results questions: 0% accuracy
- ✅ Comparison questions: 100% accuracy

*See `results/evaluation_results.csv` for detailed breakdown*

---

## 🖥️ Demo Screenshot

---

## 🔬 Technical Deep Dive

### 1. Document Processing

**Challenge**: Academic PDFs have complex structure (chapters, sections, figures, tables)

**Solution**:
- PyMuPDF for robust PDF parsing
- Automatic section detection using regex patterns
- Metadata enrichment (section type, page number, word count)

### 2. Chunking Strategy

**Challenge**: Finding optimal chunk size for context vs. specificity

**Approach**:
- Used RecursiveCharacterTextSplitter (preserves semantic boundaries)
- Selected 1000 chars with 200 overlap

### 3. Retrieval Optimization

**Method**: Semantic similarity search
- Embedding model: `text-embedding-3-small` (1536 dims)
- Retrieval: Top-5 most similar chunks
- Metadata filtering by section type (when needed)

### 4. Prompt Engineering

Custom prompt template for academic Q&A:
- Emphasizes grounding in context
- Requests section references
- Maintains academic tone
- Acknowledges uncertainty

### 5. Evaluation Framework

**Test Set**: 8 carefully crafted questions
- Covers all major thesis sections
- Multiple question types (overview, technical, results)

**Metrics**:
- Retrieval precision (correct section retrieved?)
- Answer quality (manual review)
- Source attribution quality

---

## 🎨 Example Queries

Try these questions with the system:

```
1. "What is the main research question of this thesis?"
2. "What methodology was used for bias detection?"
3. "What datasets were used in the evaluation?"
4. "What are the key findings and contributions?"
5. "How does personalization work in this system?"
6. "What are the limitations mentioned?"
7. "How does this approach differ from existing work?"
8. "What evaluation metrics were used?"
```

---

## 🔧 Configuration

### Adjustable Parameters

**In Notebook** (`Config` class):
```python
CHUNK_SIZE = 1000          # Characters per chunk
CHUNK_OVERLAP = 200        # Overlap between chunks
TOP_K = 5                  # Number of chunks to retrieve
SEARCH_TYPE = "similarity" # or "mmr" for diversity
```

**In App** (top of `app.py`):
```python
VECTOR_DB_PATH = "./thesis_vectordb"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
TOP_K = 5
```

---

## 👤 Author

**Tasnim Zahan**

**Email:** zahan.tasnim@gmail.com  
**GitHub:** [tzahan](https://github.com/tzahan)

---

## 📚 Resources

**Learn More About RAG**:
- [LangChain Documentation](https://docs.langchain.com)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [ChromaDB Documentation](https://docs.trychroma.com)

**My Thesis**:
- Title: "Personalized Summarization of Global News: Managing Bias with Large Language Models"
- [Thesis Link](https://urn.fi/URN:NBN:fi:jyu-202506024759)


