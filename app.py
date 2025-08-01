# Category Mapping RAG Evaluator (Prototype with Streamlit)
# Requirements: streamlit, pandas, sentence-transformers, sklearn, clip, torch, PIL

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import requests
from io import BytesIO
import torch
import clip
import base64

st.set_page_config(page_title="Category Mapping RAG Tool", layout="wide")
st.title("🧠 Product-to-Category Mapping RAG Evaluator")

@st.cache_resource
def load_models():
    text_model = SentenceTransformer('all-MiniLM-L6-v2')
    clip_model, clip_preprocess = clip.load("ViT-B/32", device="cpu")
    return text_model, clip_model, clip_preprocess

text_model, clip_model, clip_preprocess = load_models()

uploaded_file = st.file_uploader("Upload CSV with Product Data", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    required_cols = ['Product ID', 'Category Name', 'Product Title', 'Specifications', 'Image URL', 'Price', 'Lead Feedback']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Missing required columns. Expected: {required_cols}")
    else:
        with st.spinner("Processing..."):
            df['title_score'] = df.apply(lambda row: util.cos_sim(
                text_model.encode(row['Product Title'], convert_to_tensor=True),
                text_model.encode(row['Category Name'], convert_to_tensor=True)
            ).item(), axis=1)

            df['spec_score'] = df.apply(lambda row: 
                sum([1 for word in str(row['Category Name']).lower().split() if word in str(row['Specifications']).lower()]) 
                / len(row['Category Name'].split()), axis=1)

            scaler = MinMaxScaler()
            df['price_score'] = scaler.fit_transform(df[['Price']])

            df['feedback_score'] = df['Lead Feedback'].apply(lambda x: 1 if str(x).lower() in ['relevant', 'yes', '1'] else 0)

            def get_image_score(image_url, category_name):
                try:
                    response = requests.get(image_url, timeout=5)
                    img = clip_preprocess(Image.open(BytesIO(response.content))).unsqueeze(0)
                    text = clip.tokenize([category_name])
                    with torch.no_grad():
                        image_features = clip_model.encode_image(img)
                        text_features = clip_model.encode_text(text)
                        score = torch.cosine_similarity(image_features, text_features).item()
                    return score
                except:
                    return 0.0

            df['image_score'] = df.apply(lambda row: get_image_score(row['Image URL'], row['Category Name']), axis=1)

            df['overall_score'] = df[['title_score', 'spec_score', 'image_score', 'price_score', 'feedback_score']].mean(axis=1)
            df['RAG'] = df['overall_score'].apply(lambda x: 'Green' if x >= 0.7 else ('Amber' if x >= 0.4 else 'Red'))

            st.success("RAG Scoring Completed")
            st.dataframe(df[['Product ID', 'Category Name', 'Product Title', 'RAG', 'overall_score', 'title_score', 'spec_score', 'image_score', 'price_score', 'feedback_score']])

            def get_table_download_link(df):
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="rag_output.csv">Download CSV</a>'
                return href

            st.markdown(get_table_download_link(df), unsafe_allow_html=True)
            st.bar_chart(df['RAG'].value_counts())
