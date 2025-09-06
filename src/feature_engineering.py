import pandas as pd
import numpy as np
from src import config
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch  

def create_segment_vectors(users_df, segments_df):
    """
    Creates a representative vector for each of the 500 user segments
    by averaging the vectors of all users within each segment.
    """
    print("--- Creating Segment Representative Vectors ---")
    
    # Merge the two dataframes to have user vectors and their segment in one place
    merged_df = pd.merge(users_df, segments_df, on='user_id')

    # Define the columns that represent the user vector dimensions
    vector_columns = [f'f{i}' for i in range(1, config.USER_VECTOR_DIM + 1)]

    # Group by segment and calculate the mean for each vector dimension
    segment_vectors = merged_df.groupby('segment')[vector_columns].mean()

    print(f"✅ Representative vectors created for {len(segment_vectors)} segments.")
    
    return segment_vectors

def create_banner_embeddings(banners_df, max_features=500):
    """
    Converts banner captions (Persian text) into numerical vectors (embeddings)
    using the TF-IDF method.
    """
    print("--- Creating Banner Embeddings using TF-IDF ---")
    
    # Initialize the TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    
    # Fit the vectorizer to the captions and transform them into a matrix
    banner_embeddings = tfidf_vectorizer.fit_transform(banners_df['caption']).toarray()
    
    print(f"✅ Banner embeddings created. Shape: {banner_embeddings.shape}")
    
    return banner_embeddings, tfidf_vectorizer
def create_banner_embeddings_bert(banners_df):
    """
    Converts banner captions into contextualized numerical vectors (embeddings)
    using a pre-trained ParsBERT model.

    Args:
        banners_df (pd.DataFrame): DataFrame containing banner captions.

    Returns:
        np.ndarray: A dense matrix of BERT embeddings for the banners.
    """
    print("\n--- Creating Banner Embeddings using ParsBERT ---")
    print("Loading ParsBERT model... (This may take a moment)")
    
    # Load the pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased")
    model = AutoModel.from_pretrained("HooshvareLab/bert-fa-base-uncased")
    
    print("✅ Model loaded. Starting embedding process...")
    
    # Process captions in batches for efficiency (optional but good practice)
    captions = banners_df['caption'].tolist()
    
    # Tokenize the captions
    inputs = tokenizer(captions, padding=True, truncation=True, return_tensors="pt", max_length=128)
    
    # Get embeddings from the model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # We use mean pooling on the last hidden state to get a single vector per caption
    banner_embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    
    print(f"✅ ParsBERT embeddings created. Shape: {banner_embeddings.shape}")
    
    return banner_embeddings