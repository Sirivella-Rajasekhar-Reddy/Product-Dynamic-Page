
from crewai import Crew, Process, Agent, Task
from crewai_tools import WebsiteSearchTool
from langchain_community.vectorstores import FAISS
#from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from flask import Flask, render_template, request
import pandas as pd
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os, json, bs4
load_dotenv()

os.environ['HF_TOKEN']="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"

app = Flask(__name__)

#GUCCI -> "purchase-column purchase-column--has-stock", "detail-accordion", "accordion-product-details"

def create_tool(web_url):
    if web_url:
        loader=WebBaseLoader(web_paths=(web_url,),
                     bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                         class_=lambda x: x in ["css-8atqhb", "css-hfoyj8", "css-14ktbsh"]
                     ))
                     )
        docs=loader.load()
        text_splitter_docs=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
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

def create_crew(product_url):
    try:
        groq_api_key=os.getenv("GROQ_API_KEY")
        llm=ChatGroq(groq_api_key=groq_api_key, model="Llama3-8b-8192",streaming=True, temperature=0)
        tool=create_tool(product_url)
        agent_analyzer=create_agent(tool, llm)
        task_analyzer=create_task(agent_analyzer)
        crew=Crew(
            agents=[agent_analyzer],
            tasks=[task_analyzer],
            process=Process.sequential
        )
        resp=crew.kickoff().raw
        start_index=resp.index("{")
        end_index=resp.rindex("}")
        json_array=resp[start_index:end_index+1]
        json_array='['+json_array+']'
        json_load=json.loads(json_array)
        data=pd.DataFrame(json_load)
        return data
    except Exception as e:
        print("The Error is : ", e)

@app.route("/", methods=["GET", "POST"])
def index():
    df=pd.DataFrame()
    try:
        if request.method == "POST":
            product_url = request.form.get("product_url")
            if product_url:
                response=create_crew(product_url)
                df=df._append(response, ignore_index = True)
            else:
                print("error")
        
        return render_template("index.html", product_details=df)
    except Exception as e:
        print("The Error is : ", e)

if __name__ == "__main__":
    app.run(debug=True)
