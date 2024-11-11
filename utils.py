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

@dataclass
class PDFConfig:
    chunk_size: int = 4000
    chunk_overlap: int = 50
    temperature: float = 0.5
    model_name: str = "gpt-4o-mini-2024-07-18"
    embedding_model: str = "text-embedding-3-large"

class PDFProcessor:
    def __init__(self, api_key: str = None):
        self.config = PDFConfig()
        self.set_api_key(api_key)
        self.key_present = False
        
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
            chunk_size=self.config.chunk_size
        )
        return FAISS.from_documents(chunks, embeddings)
    
    def _create_qa_chain(self, vector_store: FAISS) -> ConversationalRetrievalChain:
        llm = ChatOpenAI(
            temperature=self.config.temperature, 
            model_name=self.config.model_name
        )
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

    def process_pdf(self, pdf_file) -> Tuple[ConversationalRetrievalChain, List]:
        text = self._parse_pdf(pdf_file)
        chunks = self._create_text_chunks(text)
        vector_store = self._create_vector_store(chunks)
        qa_chain = self._create_qa_chain(vector_store)
        return qa_chain, chunks

    def extract_transaction_details(self, qa_chain, chat_history=None) -> str:
        query = """
        Extract all transaction records from the provided document. For each transaction, return a list of RFC8259-compliant JSON objects containing the following fields:

        Name_of_the_custodian: The financial institution holding the account
        Name_of_account: The specific account owner name
        Account_number: The full account identifier
        Date_of_statement: The statement period in YYYY/MM/DD to YYYY/MM/DD format
        Unrealized_Gain_Loss_Total: The total unrealized gain/loss amount
        Equity_Name: The full legal name of the security
        Ticker: The trading symbol
        ISIN: The International Securities Identification Number
        Price: The current market price per share/unit
        Quantity: The number of shares/units held
        Value: The total market value
        Original_Cost: The initial purchase cost
        Unrealized_Gain_Loss: The individual unrealized gain/loss

        Return as JSON array with no additional text.
        """
        chat_history = chat_history or []
        result = qa_chain.invoke({"question": query, "chat_history": chat_history})
        return result["answer"]