import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from pypdf import PdfReader
from langchain_groq import ChatGroq
from io import BytesIO
from dotenv import load_dotenv
import os
from langchain.docstore.document import Document  # Import Document class

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Set up embeddings and chat model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Replace with your desired embedding model
chat_model = ChatGroq(
    model="mixtral-8x7b-32768",  # Change the model as needed
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)  # Assuming chat_model is your Groq model instance

# Function to extract text from PDF files
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into smaller chunks
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return splitter.split_text(text)

# Function to index documents with FAISS
def create_faiss_index(docs, embedding_model):
    # Convert raw text documents into Document objects
    documents = [Document(page_content=doc) for doc in docs]
    # Create a FAISS index from the documents and the embedding model
    faiss_index = FAISS.from_documents(documents, embedding_model)
    return faiss_index

# Function to interact with the chat model using `invoke`
def chat_with_pdf(query, faiss_index):
    # Perform a similarity search in the FAISS index
    search_results = faiss_index.similarity_search(query, k=5)
    # Use the `page_content` attribute from the Document objects
    context = "\n".join([result.page_content for result in search_results])

    # Now, use the 'invoke' method to query the model
    try:
        # Prepare the messages as a conversation
        messages = [
            ("system", "You are a helpful assistant that provides answers based on uploaded documents. If the query is irrelevant to the documents, respond with 'Your question is not relevant to the uploaded documents.'"),
            ("human", query),  # Pass the user query here
            ("assistant", context)  # Context from the FAISS search results
        ]
        ai_msg = chat_model.invoke(messages)  # Correct usage of 'invoke'

        # Check the response type and extract text accordingly
        if hasattr(ai_msg, 'text'):
            # Generate medium-sized response (500 words limit) by removing newlines and limiting the word count
            response_text = ai_msg.text
            medium_response = generate_medium_response(response_text, max_word_count=500)
            return medium_response  # Return shortened response
        else:
            return str(ai_msg)  # In case the response is of an unexpected format
    except Exception as e:
        st.error(f"Error calling the model: {e}")
        return "There was an error processing your request."

# Function to generate a medium-sized response with a word count limit
def generate_medium_response(content, max_word_count=500):
    # Clean the content by removing unwanted newline characters
    cleaned_content = content.replace('\n', ' ')
    
    # Split the content into words
    words = cleaned_content.split()

    # Ensure the response doesn't exceed the specified word count
    if len(words) > max_word_count:
        words = words[:max_word_count]

    # Join the words back into a string
    medium_response = ' '.join(words)

    return medium_response

# Streamlit UI
st.title("🦜 LangChain: Chat with PDF")
st.sidebar.header("Upload PDFs")

# Allow users to upload multiple PDFs
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

# Initialize FAISS index to store document embeddings
faiss_index = None

if uploaded_files:
    # Extract and process text from each uploaded PDF
    all_docs = []
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        st.sidebar.write(f"Processing {file_name}...")
        text = extract_text_from_pdf(uploaded_file)
        docs = split_text(text)
        all_docs.extend(docs)

    # Create a FAISS index from the processed documents
    faiss_index = create_faiss_index(all_docs, embedding_model)
    st.sidebar.write(f"Processed {len(all_docs)} chunks of text from {len(uploaded_files)} PDFs.")

    # Allow the user to chat with the PDF content
    user_input = st.text_input("Ask something about the PDFs:")

    if user_input:
        if faiss_index:
            # Get the model's response based on user input and FAISS index
            answer = chat_with_pdf(user_input, faiss_index)
            
            # Display the response in green if it's valid
            if answer:
                st.markdown(f"<span style='color: green;'>{answer}</span>", unsafe_allow_html=True)
        else:
            st.write("No documents to process.")
else:
    st.write("Please upload some PDFs to get started.")
