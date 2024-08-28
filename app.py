
from crewai import Crew, Process, Agent, Task
from crewai_tools import WebsiteSearchTool
from langchain_community.vectorstores import FAISS
#from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os, json, bs4
load_dotenv()

os.environ['HF_TOKEN']="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"

#GUCCI -> "purchase-column purchase-column--has-stock", "detail-accordion", "accordion-product-details"

def create_tool(web_url):
    if web_url:
        loader=WebBaseLoader(web_paths=(web_url,),
                     bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                         class_=lambda x: x in ["css-8atqhb", "css-hfoyj8", "css-14ktbsh"]
                     ))
                     )
        docs=loader.load()
        #st.write(docs)
        text_splitter_docs=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
        #st.write(text_splitter_docs)
        #embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        embeddings=OllamaEmbeddings(model="llama3.1")
        vector_store=FAISS.from_documents(text_splitter_docs, embeddings)
        retriever=vector_store.as_retriever()
        retriever_tool=create_retriever_tool(retriever, "Analyzer", "Get related document information in the form of table")
        return retriever_tool

def create_agent(tool, llm):
    agent_analyzer=Agent(
    role='Information Analyzer',
    goal='The Provided output contains product details of the product and it is in the form of json.Product details are product_name, image src or image url, price, color, description, style_number, features, measurements and material.',
    backstory=(
       "Expert in understanding the information and provide the related information in the form of json" 
    ),
    verbose=True,
    memory=True,
    allow_delegation=True,
    tools=[tool],
    llm=llm
    )
    return agent_analyzer

def create_task(agent_analyzer):
    task_analyzer=Task(
    description='Analyze and return the information in the form of json',
    expected_output='The Provided output contains product details of the product and it is in the form of json. Product details are product_name, image src or image url, price, color, description, style_number, features, measurements and material.',
    agent=agent_analyzer
    )
    return task_analyzer

if __name__=="__main__":
    try:
        st.set_page_config(page_title="Dynamic Product Page", page_icon="🦜")
        st.title("🦜 Dynamic Page")
        st.subheader('Create Dynamic Page')

        df = pd.DataFrame()

        with st.sidebar:
            groq_api_key=st.text_input("Groq Api key", value="", type="password")
        
        if groq_api_key:
            llm=ChatGroq(groq_api_key=groq_api_key, model="Llama3-8b-8192",streaming=True, temperature=0)
        else:
            llm=ChatGroq(groq_api_key=groq_api_key, model="Llama3-8b-8192",streaming=True)
            st.sidebar.error("Please Provide Groq Api Key")
        
        web_url=st.text_input("Web Url", placeholder="Enter Website URL")

        if web_url:
            tool=create_tool(web_url)
            agent_analyzer=create_agent(tool, llm)
            task_analyzer=create_task(agent_analyzer)
            crew=Crew(
                agents=[agent_analyzer],
                tasks=[task_analyzer],
                process=Process.sequential
            )

            resp=crew.kickoff().raw
            #st.write(resp)
            start_index=resp.index("{")
            end_index=resp.rindex("}")
            json_array=resp[start_index:end_index+1]
            json_array='['+json_array+']'
            #st.write(json_array)
            json_load=json.loads(json_array)
            #st.write(json_load)
            data=pd.DataFrame(json_load)
            st.dataframe(data)
        else:
            st.error("Please Provide Web URL")
        result=data.empty
        if result!=True:
            if 'product_name' in data.columns:
                st.header(data['product_name'].to_string(index=False))
            if "image_src" in data.columns:
                st.write("Image Url : ", data['image_src'].to_string(index=False))
            if 'price' in data.columns:
                st.write("Price : ", data['price'].to_string(index=False))
            if "color" in data.columns:
                st.write("Color : ", data['color'].to_string(index=False))
            if "style_number" in data.columns:
                st.write("Style Number : ", data['style_number'].to_string(index=False))
            if "features" in data.columns:
                st.write("Features : ", data['features'].to_string(index=False))
            if 'material' in data.columns:
                st.write("Material : ", data['material'].to_string(index=False))
            if "description" in data.columns:
                st.write("Description : ", data['description'].to_string(index=False))

    except Exception as e:
        st.write("The Error is : ", e)
