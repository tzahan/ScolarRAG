"""
Simple Gradio UI for Thesis RAG System
Uses the pipeline developed in the Jupyter notebook

Usage:
    python app.py
"""

import gradio as gr
import os
from dotenv import load_dotenv
from typing import Dict

# Import from notebook (or copy the core classes)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from langchain_community.embeddings import HuggingFaceEmbeddings

# Configuration
VECTOR_DB_PATH = "./scholar_vectordb"
EMBEDDING_MODEL = "text-embedding-3-small" #"sentence-transformers/all-MiniLM-L6-v2" 
LLM_MODEL = "gpt-4o-mini"
TOP_K = 4

CUSTOM_PROMPT = """You are an AI assistant helping to answer questions about a Master's thesis titled "Personalized Summarization of Global News: Managing Bias with Large Language Models".

Use the following pieces of context from the thesis to answer the question. If you don't know the answer based on the context, say so - don't make up information.

When answering:
1. Be specific and reference the relevant sections when possible
2. If the answer relates to methodology, results, or specific findings, cite them clearly
3. Maintain academic tone but be clear and concise
4. If the context doesn't fully answer the question, acknowledge what you can and cannot answer

Context from thesis:
{context}

Question: {question}

Answer:"""


class SimpleThesisRAG:
    """Simplified RAG system for Gradio"""
    
    def __init__(self):
        self.qa_chain = None
        self.is_initialized = False
    
    def initialize(self) -> str:
        """Initialize the RAG system with API key"""
        try:
            #if not api_key:
            #    return "❌ Please provide an OpenAI API key"
            
            # Set API key
            load_dotenv(override=True)
            os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
            #os.environ["OPENAI_API_KEY"] = api_key
            
            # Check if vector store exists
            if not os.path.exists(VECTOR_DB_PATH):
                return "❌ Vector store not found. Please run the Jupyter notebook first to create it."
            
            # Initialize embeddings
            #embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

            # Initialize components
            embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
            llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
            
            # Load vector store
            vectorstore = Chroma(
                persist_directory=VECTOR_DB_PATH,
                embedding_function=embeddings
                #collection_name="thesis_collection"
            )
            
            # Create retriever
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": TOP_K}
            )
            
            # Create prompt
            prompt = PromptTemplate(
                template=CUSTOM_PROMPT,
                input_variables=["context", "question"]
            )
            
            # Build QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
            )
            
            self.is_initialized = True
            return "✅ System initialized successfully! You can now ask questions."
            
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def query(self, question: str) -> tuple[str, str]:
        """Query the thesis"""
        
        if not self.is_initialized:
            print(self.initialize())
            #return "❌ Please initialize the system first (enter API key and click Initialize)", ""
        
        if not question.strip():
            return "❌ Please enter a question", ""
        
        try:
            # Get response
            response = self.qa_chain.invoke({"query": question})
     
            # Format answer
            answer = f"**Answer:**\n\n{response['result']}"
            
            # Format sources
            sources_text = "\n\n---\n\n**📚 Sources:**\n\n"
            
            if 'source_documents' in response:
                for i, doc in enumerate(response['source_documents']):
                    page = doc.metadata.get('page', 'N/A')
                    section = doc.metadata.get('section_type', 'general')
                    content = doc.page_content[:250]
                    
                    sources_text += f"**Source {i+1}** (Page {page}, Section: {section})\n"
                    sources_text += f"```\n{content}...\n```\n\n"
            
            return answer, sources_text
            
        except Exception as e:
            return f"❌ Error: {str(e)}", ""


# Initialize global RAG system
rag_system = SimpleThesisRAG()


# Gradio Interface Functions
def initialize_system() -> str:
    """Wrapper for initialization"""
    return rag_system.initialize()


def ask_question(question: str) -> tuple[str, str]:
    """Wrapper for querying"""
    return rag_system.query(question)


def create_interface():
    """Create the Gradio interface"""
    
    # Example questions
    examples = [
        "What is the main research question of this thesis?",
        "What methodology was used for bias detection?",
        "What datasets were used in the evaluation?",
        "What are the key findings and contributions?",
        "How does personalization work in this system?",
        "What are the limitations mentioned?",
        "How does this approach differ from existing work?",
        "What evaluation metrics were used?"
    ]
    
    with gr.Blocks(title="ScholarRAG", theme=gr.themes.Soft()) as app:
        
        gr.Markdown("""
        # 🎓 ScholarRAG
        ### Interactive Q&A System for "Personalized Summarization of Global News: Managing Bias with Large Language Models"
        
        Ask questions about the thesis and get AI-powered answers with source citations.
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## 💬 Ask Questions")
                
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="What methodology was used for bias detection?",
                    lines=2
                )
                
                ask_button = gr.Button("🔍 Get Answer", variant="primary", size="lg")
                
                with gr.Blocks(css="""
                    .scrollable-box {
                        height: 200px;
                        overflow-y: auto;
                        border: 1px solid #ccc;
                        padding: 10px;
                        border-radius: 5px;
                    }
                """) as demo:
                    answer_output = gr.Markdown(label="Answer", elem_classes="scrollable-box")
                    sources_output = gr.Markdown(label="Sources", elem_classes="scrollable-box")
                    #answer_output = gr.Markdown(label="Answer")
                    #sources_output = gr.Markdown(label="Sources")
                
                gr.Markdown("---")
                gr.Markdown("### 📌 Example Questions (click to use)")
                
                with gr.Row():
                    ex1 = gr.Button(examples[0], size="sm")
                    ex2 = gr.Button(examples[1], size="sm")
                
                with gr.Row():
                    ex3 = gr.Button(examples[3], size="sm")
                    ex4 = gr.Button(examples[4], size="sm")
                
                with gr.Row():
                    ex5 = gr.Button(examples[6], size="sm")
                    ex6 = gr.Button(examples[7], size="sm")
            
          
            with gr.Column(scale=1):
                # gr.Markdown("## 🔧 Setup")
                
                # api_key_input = gr.Textbox(
                #     label="OpenAI API Key",
                #     type="password",
                #     placeholder="sk-...",
                #     info="Enter your API key to get started"
                # )
                
                # init_button = gr.Button("🚀 Initialize System", variant="primary", size="lg")
                # status_output = gr.Textbox(label="Status", lines=3, interactive=False)
                
                 gr.Markdown("""
                ---
                ### 📊 About This Project
                
                This RAG system demonstrates:
                - **Document Processing**: Intelligent chunking and metadata extraction
                - **Vector Search**: Semantic similarity using embeddings
                - **LLM Generation**: Context-aware answer generation
                - **Source Attribution**: Transparent citation tracking
                
                Built with: LangChain, OpenAI, ChromaDB, Gradio
                """)
                

                # gr.Markdown("---")
                # gr.Markdown("## 💡 Tips")
                # gr.Markdown("""
                # - First time? Run the Jupyter notebook to create the vector database
                # - Click "Initialize System"
                # - Then ask your questions!
                # """)
        
        # Event handlers
        '''init_button.click(
            fn=initialize_system,
            inputs=[api_key_input],
            outputs=[status_output]
        )'''
        
        ask_button.click(
            fn=ask_question,
            inputs=[question_input],
            outputs=[answer_output, sources_output]
        )
        
        # Example button clicks
        ex1.click(lambda: examples[0], outputs=question_input)
        ex2.click(lambda: examples[1], outputs=question_input)
        ex3.click(lambda: examples[2], outputs=question_input)
        ex4.click(lambda: examples[3], outputs=question_input)
        ex5.click(lambda: examples[4], outputs=question_input)
        ex6.click(lambda: examples[5], outputs=question_input)
        
    
    return app


if __name__ == "__main__":
    print("Starting Thesis RAG System...")
    print(f"Vector DB Path: {VECTOR_DB_PATH}")
    print(f"LLM Model: {LLM_MODEL}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print("\nMake sure you've run the Jupyter notebook first to create the vector database!")
    print("\nLaunching Gradio interface...")
    
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )