from sqlalchemy.sql.expression import label
import streamlit as st
import pandas as pd
import openai
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.prompts import PromptTemplate
import matplotlib.pyplot as plt
import re
from streamlit_chat import message as streamMSG
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.document_loaders.csv_loader import CSVLoader



# Set up OpenAI API key
openai.api_key = "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"
OPENAI_API_KEY = "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"


def extract_code_from_response(response):
    """Extracts Python code from a string response."""
    # Use a regex pattern to match content between triple backticks
    code_pattern = r"```python(.*?)```"
    match = re.search(code_pattern, response, re.DOTALL)
    
    if match:
        # Extract the matched code and strip any leading/trailing whitespaces
        return match.group(1).strip()
    return None

def storage():
  try:
    loader = CSVLoader('/content/file.csv')
    CSVdata = loader.load()
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model = "text-embedding-ada-002")
    vectorstore = FAISS.from_documents(CSVdata, embeddings)
    return vectorstore
  except:
    pass

def conversationModel(storage, Modeloption, modelTemp):
  Memory =  ConversationBufferMemory(memory_key = 'chat_history', return_messages = True)

  OpenAIllm = ChatOpenAI(model= Modeloption, temperature=modelTemp, openai_api_key=OPENAI_API_KEY, max_tokens = 2000)
  prompt_template = """Use the following pieces of context to answer and the question related to food suggestion and dietary plans at the end.
  If the question is hello or hi or hey then just answer Hello!  how may i help you.

  {context}

  Question: {question}
  Answer in english:"""

  PROMPT = PromptTemplate(
      template=prompt_template, 
      input_variables=["context", "question"]
  )

  chain = RetrievalQA.from_chain_type( 
      llm = OpenAIllm,
      memory = Memory, 
      chain_type = "stuff",
      retriever = storage.as_retriever(), 
      chain_type_kwargs = {"prompt": PROMPT}, 
      verbose = True
  )

  return chain

def main():

    st.set_page_config(page_title = 'FoodGPT', page_icon='🍖', layout='wide')
    # Set up the Streamlit app
    st.title("Restaurant Food Waste Reduction")

    Manualdf = pd.DataFrame()
    CSVdf    = pd.DataFrame()
    global storage

    if "model"  not in st.session_state:
        st.session_state.model = None

    if 'History' not in st.session_state:
        st.session_state.History = None

    with st.sidebar:
      Modeloption = st.selectbox('Select the model',
                                ('gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0613')
                                )
      st.write('------')

      modelTemp = st.slider("select the model temperature",0.0, 1.0)
  

      st.write('------')

      Dataopetion = st.selectbox('Select the data', ('CSV', 'Manual'))

    tab1, tab2 = st.tabs(['CSV', 'Manual Entry'])

    with tab1:
      # Option to upload CSV file
      uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

      if st.button('Upload file'):
        try:
          CSVdf = pd.read_csv(uploaded_file)
          CSVdf.to_csv('/content/file.csv')
          st.success('File uploaded')
        except ValueError:
          st.error('Upload the CSV File')

    with tab2:
      # Input fields for manual data entry
      st.subheader("Manual Data Entry")
      item_name = st.text_input("Item Name",)
      category = st.text_input("Category")
      quantity = st.number_input("Quantity", value=0)
      unit_price = st.number_input("Unit Price (£)", value=0.0)
      supplier = st.text_input("Supplier")
      purchase_date = st.date_input("Purchase Date")
      expiration_date = st.date_input("Expiration Date")

      new_row = {
              "Item Name": item_name,
              "Category": category,
              "Quantity": quantity,
              "Unit Price": unit_price,
              "Supplier": supplier,
              "Purchase Date": purchase_date,
              "Expiration Date": expiration_date
          }
 
        # Add manually entered data to DataFrame
      if st.button("Add to Inventory"):
          Manualdf = Manualdf.append(new_row, ignore_index=True)
          st.success("Item added to inventory!")
          try:
            # Display the current inventory
            st.subheader("Current Inventory")
            st.write(Manualdf)
          except UnboundLocalError:
            pass
    
    st.write('-----------')
    st.header('View uploaded CSV')
    
    if st.button('View Data'):
      try:
        uploadedCSV = pd.read_csv(uploaded_file)
        st.write(uploadedCSV)
      except:
        st.error('Upload the CSV File')
      
    st.write('-----------')
    # know details abot the csv
    st.header('Know details abot the csv')
    try:
      try:
        uploadedCSV = pd.read_csv(uploaded_file)
      except:
        pass
      agent = create_csv_agent(
                  ChatOpenAI(temperature=modelTemp, model = Modeloption, openai_api_key = OPENAI_API_KEY),
                  '/content/file.csv',
                  verbose=True,
                  agent_type=AgentType.OPENAI_FUNCTIONS
              )
      with st.form('CSV_Detail_form', clear_on_submit= True):
        query = st.text_input(label='Enter your prompt')
        # Generate food waste reduction recommendations
        submit = st.form_submit_button('submit')
        if submit:
          # Properly format the user's input and wrap it with the required "input" key
          tool_input = {
              "input": {
                  "name": "python",
                  "arguments": query
              }
          }
          response = agent.run(tool_input)
          res = extract_code_from_response(response)

          if res:
                try:
                    # Making df available for execution in the context
                    exec(res, globals(), {"df": uploadedCSV, "plt": plt})
                    fig = plt.gcf()  # Get current figure
                    st.pyplot(fig)  # Display using Streamlit
                except Exception as e:
                    st.write(f"Error executing code: {e}")
          else:
                st.write(response)
      
            
    except FileNotFoundError:
      pass

    st.write('-----------')
    
    st.header('Conversation Model')
    with st.form('Conversation model', clear_on_submit= True):
        user_input = st.text_input(label = 'Enter text')
        # Generate food waste reduction recommendations
        submit = st.form_submit_button('submit')
        if submit:    
          storage = storage()
          st.session_state.model   = conversationModel(storage, Modeloption, modelTemp)
          AgentResponse = st.session_state.model({'query': user_input})
          st.session_state.History = AgentResponse['chat_history']

          for i, msg in enumerate(st.session_state.History):
            if i % 2 == 0:
                streamMSG(msg.content,  is_user=True)
            else:
                streamMSG(msg.content,  is_user=False)


if __name__ == '__main__':
    main()
    