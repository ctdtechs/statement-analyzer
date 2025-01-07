import streamlit as st
import PyPDF2
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama

# Initialize the ChatOllama model
ollama = ChatOllama(model="llama3.1:latest", base_url="http://37.27.125.244:11434")

# Define the template for the AI's response
template = """
Based on the knowledge base below, write an answer like a bank manager for the User's question:
{knowledgebase}
 
User's question: hey analysis this bank statement based on this suggest we can proceed with this customer for loan process how much confident on this.
"""

# Create the prompt using the template
prompt = ChatPromptTemplate.from_template(template)

# Function to read PDF text and replace new lines with spaces
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    # Replace new lines with spaces
    text = text.replace("\n", " ")
    return text

# Streamlit UI
st.title("AI Bank Statement Analyzer")

# PDF file uploader
uploaded_file = st.file_uploader("Select a Bank Statement PDF", type=["pdf"])

if uploaded_file is not None:
    # Extract text from the selected PDF
    text = extract_text_from_pdf(uploaded_file)
    
    # Display the extracted text inside an expander
    with st.expander("Bank Statement Content"):
        st.write(text)

    # Prepare the knowledge base for the prompt (i.e., the extracted text)
    knowledgebase = text
    prompt_input = prompt.format(knowledgebase=knowledgebase)

    # Get AI's response
    response = ollama.invoke(prompt_input)

    # Display the AI's response
    st.write("AI Model Response:")
    st.write(response.content)