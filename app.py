import streamlit as st
import pandas as pd
from utils import PDFProcessor
import json
from typing import List
from dataclasses import dataclass
from datetime import datetime
import asyncio
import concurrent.futures

@dataclass
class TransactionSummary:
    custodian: str
    account_name: str
    account_number: str
    statement_period: str
    total_unrealized_gain_loss: float

class TransactionDetailsExtractor:
    def __init__(self):
        self.qa_chain = None
        self.chunks = None
        self.processor = None
        self._setup_page_config()
        self._initialize_session_state()
    
    def _initialize_session_state(self) -> None:
        if 'transaction_data' not in st.session_state:
            st.session_state.transaction_data = {}
        if 'theme' not in st.session_state:
            st.session_state.theme = 'light'
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = []
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
    
    def _setup_page_config(self) -> None:
        st.set_page_config(
            page_title="RAG-Based Financial Document Analyzer",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/AlbinAimleap/financial_data_analyzer',
                'Report a bug': "https://www.example.com/bug",
                'About': "# Financial Document Analyzer\nVersion 1.0"
            }
        )
        self._setup_sidebar()
    
    def _setup_sidebar(self) -> None:
        with st.sidebar:
            st.image("https://media.designrush.com/agencies/323733/conversions/AIMLEAP-logo-profile.jpg", caption="Financial Analyzer")
            st.title("Document Analysis Tools")
            st.markdown("""
            <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
            <h2>Getting Started</h2>
            <ol>
                <li>🔑 Enter your OpenAI API key</li>
                <li>📁 Upload PDF documents</li>
                <li>⚙️ Process and analyze</li>
                <li>📊 View detailed results</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
    
    def setup_ui(self) -> None:
        st.markdown("""
        <h1 style='text-align: center; color: #1f77b4;'>Financial Document Analyzer</h1>
        <p style='text-align: center; font-size: 1.2em;'>Advanced Transaction Analysis Platform</p>
        """, unsafe_allow_html=True)
        
        if not self._initialize_processor():
            return
        self._handle_file_upload()
    
    def _initialize_processor(self) -> bool:
        api_key = self._get_api_key()
        if not api_key:
            st.warning("⚠️ Please provide your OpenAI API key to continue.", icon="⚠️")
            return False
        self.processor = PDFProcessor(api_key)
        return True
    
    def _get_api_key(self) -> str:
        return st.sidebar.text_input("🔑 OpenAI API Key", type="password", help="Enter your OpenAI API key")
    
    def _handle_file_upload(self) -> None:
        st.markdown("---")
        uploaded_pdfs = st.file_uploader(
            "📁 Upload Financial Documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Select one or more PDF files to analyze"
        )
        st.session_state.uploaded_files = uploaded_pdfs
        if st.session_state.uploaded_files:
            col1, col2 = st.columns([3, 1])
            processed_data = []
            with col1:
                st.info(f"📎 {len(st.session_state.uploaded_files)} file{'(s)' if len(st.session_state.uploaded_files)  > 1 else ''} selected")
                
            btn = None
            with col2:
                if st.button("🔍 Analyze Documents", use_container_width=True):
                    btn = True
            if btn:
                processed_data.extend(asyncio.run(self._process_uploaded_files_async(st.session_state.uploaded_files)))
                st.session_state.processed_data = processed_data
            self.display_results()
    
    def rerun_page(self) -> None:
        st.session_state.processed_data = []
        st.session_state.uploaded_files = []
        st.rerun(scope="app")
        
    def display_results(self) -> None:
        if st.session_state.processed_data:
            st.markdown("### 📊 Transaction Analysis")
            for summary, transactional_df, filename in st.session_state.processed_data:
                st.markdown(f"#### 📄 {filename}")
                self._display_summary(summary)
            
                if not transactional_df.empty:
                    self._display_dataframe_with_download(transactional_df, filename)
                else:
                    st.warning("No transaction details found in the document.")
                
                # Add a button to rerun the page
            if st.button("🔄 Clear"):
                self.rerun_page()
    
    async def _process_uploaded_files_async(self, uploaded_pdfs: List) -> List[tuple]:
        cols = st.columns(len(uploaded_pdfs))
        tasks = []
        
        with st.spinner("🔍 Analyzing documents..."):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                loop = asyncio.get_event_loop()
                for idx, pdf in enumerate(uploaded_pdfs):
                    with cols[idx]:
                        task = loop.run_in_executor(
                                executor,
                                self._process_document,
                                pdf
                            )
                        tasks.append((task, pdf.name))
                
                results = []
                for task, filename in tasks:
                    summary, transactional_df = await task
                    results.append((summary, transactional_df, filename))
                
                st.success("✅ All documents processed successfully!")
                return results
    
    def _process_document(self, pdf_file) -> tuple:
        with st.spinner("⚙️ Processing document..."):
            self.qa_chain, self.chunks = self.processor.process_pdf(pdf_file)
        summary, transactional_df = self._extract_and_display_details(pdf_file)
        st.success(f"✅ Processed {pdf_file.name} successfully!")
        return summary, transactional_df
    
    def _extract_and_display_details(self, pdf_file) -> tuple:
        with st.spinner("🔍 Extracting transaction details..."):            
            # Extract transaction details
            transactions = self.processor.extract_transaction_details(self.qa_chain)
            
            transactions = transactions.replace("```json", "").replace("```", "")
            transactions_data = json.loads(transactions)
      
            transactions_df = pd.DataFrame(transactions_data if len(transactions_data) > 1 else [])
            
            # Extract summary from first element
            summary_data = transactions_data[0] if transactions_data else {}
            
            # Create transaction summary object
            summary = TransactionSummary(
                custodian=summary_data.get("Name_of_the_custodian"),
                account_name=summary_data.get("Name_of_account"),
                account_number=summary_data.get("Account_number"),
                statement_period=summary_data.get("Date_of_statement"),
                total_unrealized_gain_loss=summary_data.get("Unrealized_Gain_Loss_Total"),
            )
            return summary, transactions_df
                
    
    def _display_summary(self, summary: TransactionSummary) -> None:
        st.markdown("""
        <style>
        .summary-box {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            height: 100px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        .summary-box:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .summary-label {
            font-weight: bold;
            color: #666;
        }
        .summary-value {
            font-size: 1.1em;
            color: #333;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }
        </style>
        """, unsafe_allow_html=True)
        
        from_, to_ = summary.statement_period.split("to")
        
        with st.container():
            cols = st.columns(6)
            summary_items = [
                ("Custodian", summary.custodian),
                ("Name", summary.account_name),
                ("Account #", summary.account_number),
                ("Period From", from_),
                ("Period To", to_),
                ("Gain/Loss", summary.total_unrealized_gain_loss)
            ]
            
            for col, (label, value) in zip(cols, summary_items):
                with col:
                    st.markdown(f"""
                    <div class='summary-box'>
                        <div class='summary-label'>{label}</div>
                        <div class='summary-value'>{value}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
    def _display_dataframe_with_download(self, df: pd.DataFrame, file_name: str) -> None:
        
        tab1, tab2 = st.tabs(["📈 Data View", "📊 Statistics"])
        with tab1:
            st.dataframe(
                df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Amount": st.column_config.NumberColumn(
                        "Amount",
                        format="$%.2f"
                    )
                }
            )
        
        with tab2:
            st.markdown("#### Summary Statistics")
            st.dataframe(df.describe(), use_container_width=True)
        
        self._add_download_button(df, file_name)
    
    def _add_download_button(self, df: pd.DataFrame, file_name: str) -> None:
        csv = df.to_csv(index=False)
        unique_key = f"{file_name}_{datetime.now().timestamp()}"
        st.download_button(
            "📥 Download Analysis",
            csv,
            file_name=f"{file_name}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            key=unique_key,
            help="Download the analysis results as a CSV file",
            use_container_width=True
        )

if __name__ == "__main__":
    app = TransactionDetailsExtractor()
    app.setup_ui()
