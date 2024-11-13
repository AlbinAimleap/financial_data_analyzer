from tika import parser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models.openai import ChatOpenAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
import os
from pathlib import Path
from typing import Tuple, List
from dataclasses import dataclass

def load_prompt() -> str:
    cur_dir = Path(__file__).parent
    with open(cur_dir / "prompt.txt", "r", encoding='utf-8') as f:
        return f.read().strip()

@dataclass
class PDFConfig:
    chunk_size: int = 8000
    chunk_overlap: int = 100 
    embedding_ctx_length: int = 10000
    temperature: float = 0.7
    score_threshold: float = 0.3
    verbose: bool = False
    kval: int = 20
    model_name: str = "gpt-4o-mini-2024-07-18"
    embedding_model: str = "text-embedding-3-large"

class PDFProcessor:
    def __init__(self, api_key: str = None) -> None:
        self.config = PDFConfig()
        self.set_api_key(api_key)
        self.key_present = False
        self.qa_chains = {}
        
    def set_api_key(self, api_key: str) -> None:
        if api_key:
            self.key_present = True
            os.environ["OPENAI_API_KEY"] = api_key
            
    def _parse_pdf(self, pdf_file) -> str:
        temp_file_path = Path("temp.pdf")
        with open(temp_file_path, "wb") as f:
            f.write(pdf_file.getvalue())
        text = parser.from_file("temp.pdf")['content']
        return text
    
    def _create_text_chunks(self, text: str) -> List:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len
        )
        return splitter.create_documents([text])
    
    def _create_vector_store(self, chunks: List) -> FAISS:
        embeddings = OpenAIEmbeddings(
            model=self.config.embedding_model, 
            chunk_size=self.config.chunk_size,
            embedding_ctx_length=self.config.embedding_ctx_length
        )
        return FAISS.from_documents(chunks, embeddings)
    
    def _create_qa_chain(self, vector_store: FAISS) -> ConversationalRetrievalChain:
        llm = ChatOpenAI(
            temperature=self.config.temperature, 
            model_name=self.config.model_name
        )
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(
                search_type="mmr", 
                search_kwargs={
                    "k": self.config.kval, 
                    "fetch_k": self.config.kval * 2, 
                    "score_threshold": self.config.score_threshold
                    }),
            verbose=self.config.verbose
        )
        
    def process_pdf(self, pdf_file) -> ConversationalRetrievalChain:
        text = self._parse_pdf(pdf_file)
        chunks = self._create_text_chunks(text)
        vector_store = self._create_vector_store(chunks)
        qa_chain = self._create_qa_chain(vector_store)
        return qa_chain

    def extract_transaction_details(self, pdf_file, chat_history=None) -> str:
        qa_chain = self.process_pdf(pdf_file)
        query = load_prompt()
        chat_history = chat_history or []
        result = qa_chain.invoke({"question": query, "chat_history": chat_history})
        return result["answer"]