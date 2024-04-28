
import streamlit as st
import streamlit.components.v1 as components
import os
from transformers import pipeline
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from transformers import pipeline, AutoTokenizer,AutoModelForSeq2SeqLM
import fitz  # PyMuPDF
import io
from docx import Document
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from pyvis.network import Network

def extract_text_from_pdf(bytes_data):
    # Create a file-like object from bytes
    file_stream = io.BytesIO(bytes_data)
    document = fitz.open("pdf", file_stream)
    text = ''
    for page in document:
        text += page.get_text()
    document.close()
    return text

def extract_text_from_docx(bytes_data):
    # Create a file-like object from bytes
    file_stream = io.BytesIO(bytes_data)
    document = Document(file_stream)
    text = ''.join([paragraph.text for paragraph in document.paragraphs])
    return text


def summarize_text(text, model_name='sshleifer/distilbart-cnn-12-6'):
    if not text.strip():
        return "No content to summarize."
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Initialize the summarizer pipeline with the model and tokenizer
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

    # Tokenize text into sentences and manage larger chunks
    sentences = sent_tokenize(text)
    max_tokens = 1024
    current_chunk = []
    chunks = []

    for sentence in sentences:
        test_chunk = current_chunk + [sentence]
        test_chunk_tokens = tokenizer.encode(' '.join(test_chunk), add_special_tokens=True)

        if len(test_chunk_tokens) > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
        else:
            current_chunk = test_chunk

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    summaries = []

    for chunk in chunks:
        try:
            # Apply truncation directly in the summarizer call
            summary = summarizer(chunk, truncation=True, max_length=150, min_length=50, length_penalty=2.0)
            summaries.append(summary[0]['summary_text'])
        except Exception as e:
            summaries.append(f"Error summarizing text part: {str(e)}")

    # Combine summaries and remove redundancy
    seen = set()
    final_summary = []
    for summary in summaries:
        if summary not in seen:
            final_summary.append(summary)
            seen.add(summary)

    return ' '.join(final_summary)


nlp = spacy.load("en_core_web_sm")

def build_knowledge_graph(text):
    doc = nlp(text)
    graph = nx.Graph()

    for ent in doc.ents:
        graph.add_node(ent.text, label=ent.label_)

    for ent1 in doc.ents:
        for ent2 in doc.ents:
            if ent1 != ent2 and ent1.sent.start == ent2.sent.start:  # Checking if they are in the same sentence
                graph.add_edge(ent1.text, ent2.text, label='co-occurrence')

    return graph

def draw_knowledge_graph(graph):
    # Create a network graph
    nt = Network("500px", "1000px", notebook=False, heading='')
    for node, attrs in graph.nodes(data=True):
        nt.add_node(node, title=node, label=node, color='skyblue')
    
    for source, target, attrs in graph.edges(data=True):
        nt.add_edge(source, target, title=attrs['label'])
    
    # Set physics layout for better visualization
    nt.toggle_physics(True)
    nt.set_options("""
    var options = {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -80000,
          "centralGravity": 0.3,
          "springLength": 200,
          "springConstant": 0.05,
          "damping": 0.09
        },
        "maxVelocity": 50,
        "minVelocity": 0.1
      }
    }
    """)

    # Save and display the graph within the Streamlit app
    path = "graph.html"
    nt.save_graph(path)
    return path

def main():
    st.title("Document Summarizer and Knowledge Graph Visualizer")
    uploaded_file = st.file_uploader("Choose a PDF or DOCX file", type=["pdf", "docx"])
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(bytes_data)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = extract_text_from_docx(bytes_data)
        if st.button('Summarize'):
            summary = summarize_text(text)
            st.write("Summary:", summary)
        if st.button('Visualize Knowledge Graph'):
            kg = build_knowledge_graph(text)
            graph_path = draw_knowledge_graph(kg)
            HtmlFile = open(graph_path, 'r', encoding='utf-8')
            source_code = HtmlFile.read() 
            st.components.v1.html(source_code, height = 800)



if __name__ == "__main__":
    main()
