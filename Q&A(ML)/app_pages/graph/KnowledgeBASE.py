# from langchain_openai import OpenAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFDirectoryLoader

# # Load the PDF file and split documents
# loader = PyPDFDirectoryLoader("C:\\Users\\aravs\\Desktop\\Q&A(ML)\\1-Unit1-Notes.pdf")
# docs = loader.load_and_split()

# # Text splitting
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# texts = text_splitter.split_documents(docs)

# # Initialize FAISS vector store with OpenAI embeddings
# db = FAISS.from_documents(texts, OpenAIEmbeddings())
# retriever = db.as_retriever()
