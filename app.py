from twilio.twiml.messaging_response import MessagingResponse
import os
from flask import Flask, request
from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
from langchain.chains import RetrievalQAWithSourcesChain


load_dotenv()
WEAVIATE_URL= os.environ.get('WEAVIATE_URL')

def load_file(fname):
    with open(fname) as f:
        starwarssummary = f.read()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_text(starwarssummary)

def chain_qs(fname, prompt):
    texts = load_file(fname)
    embeddings = OpenAIEmbeddings()
    docsearch = Weaviate.from_texts(
        texts,
        embeddings,
        weaviate_url=WEAVIATE_URL,
        by_text=False,
        metadatas=[{"source": f"{i}-pl"} for i in range(len(texts))])
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        OpenAI(temperature=0), chain_type="stuff", retriever=docsearch.as_retriever()
    )
    c = chain(
        {"question": prompt},
    return_only_outputs=True,
    )
    print(c['answer'])
    return c['answer']


app = Flask(__name__)
@app.route('/sms', methods=['POST'])
def sms():
    resp = MessagingResponse()
    query = request.form['Body'].lower().strip() #get inbound text body
    resp_output = chain_qs("starwars.txt", query)
    resp.message(resp_output)
    return str(resp)

if __name__ == "__main__":
    app.run(debug=True)