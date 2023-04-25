import streamlit as st
from langchain import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI

# Using vector store
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

import requests, json, io, tempfile, os


template = """
    Below is a API response that contains the profile content of a prospect's linked in profile.
    Your goal is to:
    - Read through their profile
    - Write a message based on the [INPUT OBJECTIVE]

    Use a personable tone and please include information of the prospect. PLease only include information relevant to my input objective.
    
    Keep it under 300 characters
    
    Below is the INPUT OBJECTIVE and Relevant Prospect Data:

    INPUT OBJECTIVE: {objective}

    Relevant Prospect Data: {prospect_related_documents}

"""

prompt = PromptTemplate(
    input_variables=["objective", "prospect_related_documents"],
    template=template,
)

# Initialize Functions
def load_LLM(openai_api_key, temperature=.7):
    """Logic for loading the chain you want to use should go here."""
    # Make sure your openai_api_key is set as an environment variable
    llm = OpenAI(temperature=temperature, openai_api_key=openai_api_key)
    return llm
def get_split_tests(profile, chunk_size=1000, chunk_overlap=40):
    loader = TextLoader(profile) 
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)
def get_related_documents(texts, embeddings, related_string):
    db = FAISS.from_documents(texts,embeddings)
    retriever = db.as_retriever()
    return retriever.get_relevant_documents(related_string)

st.set_page_config(page_title="Project Demo", page_icon=":robot:")
st.header("Project Demo")

col1, col2 = st.columns(2)

with col1:
    st.markdown("This tool will help you improve your prospecting by doing the research on a linked in profile for you. \
                Powered by [LangChain](https://langchain.com/), [OpenAI](https://openai.com), and [PeopleDataLabs](https://docs.peopledatalabs.com/docs/enrichment-api). \
                \n Programmed by [roncahlon](https://twitter.com/roncahlon). \n\n")

with col2:
    image = Image.open('prospecting.png')
    st.image(image=image, width=250)

st.markdown("## Enter Your LinkedinURL To Convert")

def get_api_key():
    input_text = st.text_input(label="OpenAI API Key ",  placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input")
    return input_text

openai_api_key = get_api_key()

def get_people_lab_key():
    input_text = st.text_input(label="People Data Labs API Key ",  placeholder="Ex: 1eb...", key="people_lab_api_key_input")
    return input_text

people_lab_api_key = get_people_lab_key()

objective = st.text_area(label="What is your objective?", value = "Sell my services that will automate sales prospecting work")

st.write("LinkedIn URL:")
def get_url():
    input_text = st.text_area(label="URL Input ", label_visibility='collapsed', placeholder="https://www.linkedin.com/in/ron-cahlon-1b2a46170/", key="url")
    return input_text

url_input = get_url()


st.markdown("### Your Custom Message:")

if st.button('Generate Custom Message'):
    if not (openai_api_key and people_lab_api_key):
        st.warning('Please insert both API Key. Instructions for OpenAI [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key) \
                   and instructions for People Data Labs [here]("https://docs.peopledatalabs.com/docs/quickstart#try-looking-up-another-company-profile")', icon="⚠️")
        st.stop()

    # Set the Person Enrichment API URL
    PDL_URL = "https://api.peopledatalabs.com/v5/person/enrich"

    # Create a parameters JSON object
    PARAMS = {
        "api_key": people_lab_api_key,
        "profile": [url_input],
        "min_likelihood": 6
    }
    
    # Pass the parameters object to the Person Enrichment API
    json_response = requests.get(PDL_URL, params=PARAMS).json()

    if json_response["status"] == 200:
        content = json_response['data']
        
        file_obj = io.StringIO(json.dumps(content))
        # Write the contents of the file-like object to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write(file_obj.read())
        # Pass the path to the temporary file to split text
        split_texts = get_split_tests(temp_file.name)
        # Remove the temporary file
        os.remove(temp_file.name)

        # define embeddings engine
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        prospect_info = get_related_documents(split_texts, embeddings, objective)
        
        prompt_with_input = prompt.format(objective=objective, prospect_related_documents=prospect_info)
        llm = load_LLM(openai_api_key=openai_api_key, temperature=.8)
        custom_message = llm(prompt_with_input)

        st.write(custom_message)
    else:
        st.error(f"Error: {json_response}")
            
else:
    st.warning('Please Enter URL', icon="⚠️")
