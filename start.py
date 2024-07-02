import streamlit as st
from pinecone import Pinecone
import re
import pandas as pd
from openai import OpenAI
from pinecone_text.sparse import BM25Encoder
bm25 = BM25Encoder()

data = pd.read_csv('another_one.csv')
bm25.fit(data['text'])
pc = Pinecone(api_key="3661dc2a-3710-4669-a187-51faaa0cc557")
index = pc.Index("semantic-rag-chunking3")
client = None

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def hybrid_scale(dense, sparse, alpha: float):
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    hsparse = {'indices': sparse['indices'], 'values': [v * (1 - alpha) for v in sparse['values']]}
    hdense = [v * alpha for v in dense]
    return hdense, hsparse

def summarize_text(text: str):
    response = client.chat.completions.create(
    model = 'gpt-3.5-turbo',
    messages = [
        {"role": "system", "content": "You are a very very helpful assistant"},
        {"role": "system", "content": "I have a piece of text. I want you to generate a concise summary of {text}. Do not include any text of your own, I need just the results."},
        {"role": "user", "content": f"provided text: '{text}'"},  # Text to be summarized
    ])
    return response.choices[0].message.content

def main():
    st.title("Pinecone Search Application")
    
    search_text = st.text_input("Enter search text:")
    top_k = st.number_input("Enter top_k:", min_value=1, value=5)
    
    openai_api_key = st.text_input("Enter OpenAI API Key:")
    global client
    client = OpenAI(api_key=openai_api_key)

    if st.button("Search"):
        if search_text:
            try:
                dense = get_embedding(search_text)
                sparse = bm25.encode_queries(search_text)
                hdense, hsparse = hybrid_scale(dense, sparse, alpha=0)

                matches = index.query(
                    top_k=top_k,
                    vector=hdense,
                    sparse_vector=hsparse,
                    include_metadata=True
                )
                match_ids =[m['metadata']['id'] for m in matches['matches']]
                filtered_df = data[data['id'].isin(match_ids)]
                attributes_to_extract = ['title', 'text', 'url']
                extracted_df = filtered_df[attributes_to_extract]
                extracted_df['desription'] = extracted_df['text'].apply(summarize_text)
                st.table(extracted_df.drop(columns=['text']))
            
            except Exception as e:
                st.write('An error occurred')

        else:
            st.write('Please enter search text')

if __name__ == '__main__':
    main()

            