from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata


class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
    # <s>: Este é um token especial que representa o início de uma sequência (ou sentença)
    # [INST] e [/INST]: [INST] marca o início das instruções e [/INST] marca o fim.
    #    """
    #        <s> [INST] Given a query {question}, which of the following two documents is more relevant to the query? 
    #                   As an output you must answer whether it is Document A or Document B [/INST] </s> 
    #    """
    # models phi | deepseek-r1:1.5b
        self.model = ChatOllama(model="phi") 
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for document ranking. Given a query {question}, 
                       which of the following two documents is more relevant to the query? 
                       As an Answer you must answer whether it is Documento1 or Documento2.[/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: Documento1 or Documento2 [/INST]
            """
        )

    def ingest(self, pdf_file_path: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )

        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())

    def ask(self, query: str):
        if not self.chain:
            return "Please, add a PDF document first."

        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None