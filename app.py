
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import requests
from PIL import Image
from io import BytesIO

st.title("🧠 Category Mapping RAG Tool with Reasoning")

uploaded_file = st.file_uploader("📤 Upload cleaned product CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Load sentence transformer model
    text_model = SentenceTransformer('all-MiniLM-L6-v2')

    def get_title_score_and_reason(title, category):
        if pd.isna(title) or pd.isna(category):
            return 'Red', "Missing product title or category name"
        sim = util.cos_sim(
            text_model.encode(str(title), convert_to_tensor=True),
            text_model.encode(str(category), convert_to_tensor=True)
        ).item()
        if sim > 0.7:
            return 'Green', f"High similarity (score = {sim:.2f})"
        elif sim > 0.4:
            return 'Amber', f"Moderate similarity (score = {sim:.2f})"
        else:
            return 'Red', f"Low similarity (score = {sim:.2f})"

    def get_spec_score_and_reason(specs):
        if pd.isna(specs) or len(str(specs).strip()) == 0:
            return 'Red', "No specifications provided"
        spec_count = len(str(specs).split(","))
        if spec_count >= 3:
            return 'Green', f"{spec_count} specs found"
        elif spec_count == 2:
            return 'Amber', "Only 2 specs provided"
        else:
            return 'Red', "Less than 2 specs"

    def get_image_score_and_reason(url):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                if img.size[0] < 100 or img.size[1] < 100:
                    return 'Red', f"Image too small: {img.size}"
                return 'Green', "Image valid and clear"
            return 'Red', "Image not reachable"
        except:
            return 'Red', "Invalid or broken image URL"

    def get_price_score_and_reason(price):
        if pd.isna(price) or str(price).lower() in ["not mentioned", "na", "-", ""]:
            return 'Amber', "Price missing or generic"
        return 'Green', "Price provided"

    def get_feedback_score_and_reason(feedback):
        if isinstance(feedback, str) and "wrong" in feedback.lower():
            return 'Red', "Seller reported wrong product mapping"
        return 'Green', "No major issue reported"

    df[['title_rag', 'title_reason']] = df.apply(
        lambda row: pd.Series(get_title_score_and_reason(row['Product Title'], row['Category Name'])),
        axis=1
    )
    df[['spec_rag', 'spec_reason']] = df['Specifications'].apply(lambda s: pd.Series(get_spec_score_and_reason(s)))
    df[['image_rag', 'image_reason']] = df['Image URL'].apply(lambda url: pd.Series(get_image_score_and_reason(url)))
    df[['price_rag', 'price_reason']] = df['Price'].apply(lambda p: pd.Series(get_price_score_and_reason(p)))
    df[['feedback_rag', 'feedback_reason']] = df['Lead Feedback'].apply(lambda fb: pd.Series(get_feedback_score_and_reason(fb)))

    def calculate_final_rag(row):
        rag_values = [row['title_rag'], row['spec_rag'], row['image_rag'], row['price_rag'], row['feedback_rag']]
        red_count = rag_values.count('Red')
        amber_count = rag_values.count('Amber')
        if red_count >= 2:
            return 'Red'
        elif amber_count >= 2:
            return 'Amber'
        else:
            return 'Green'

    df['Final RAG'] = df.apply(calculate_final_rag, axis=1)

    # Display product-wise breakdown
    for i, row in df.iterrows():
        st.markdown("----")
        st.subheader(f"🧾 Product ID: {row['Product ID']}")
        st.markdown(f"**Category:** {row['Category Name']}")
        st.markdown(f"**Product Title:** {row['Product Title']}")

        param_data = {
            'Parameter': ['Title', 'Specifications', 'Image', 'Price', 'Feedback'],
            'Score': [row['title_rag'], row['spec_rag'], row['image_rag'], row['price_rag'], row['feedback_rag']],
            'Reason': [row['title_reason'], row['spec_reason'], row['image_reason'], row['price_reason'], row['feedback_reason']]
        }
        st.table(pd.DataFrame(param_data))
        st.markdown(f"✅ **Final RAG Score**: `{row['Final RAG']}`")

    st.download_button("📥 Download Full Output as CSV", df.to_csv(index=False), file_name="RAG_output.csv", mime="text/csv")
