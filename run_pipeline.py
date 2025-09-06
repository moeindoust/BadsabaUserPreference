from src import config
from src.data_processing import load_raw_data, create_final_datasets
from src.feature_engineering import create_segment_vectors, create_banner_embeddings, create_banner_embeddings_bert
from src.train_evaluate import train_and_evaluate

def main():
    """
    Executes the entire data science pipeline for both TF-IDF and ParsBERT
    and compares the results.
    """
    # Step 1: Load all the raw data (common for both pipelines)
    raw_data_dict = load_raw_data()
    if raw_data_dict is None:
        print("Pipeline stopped due to data loading errors.")
        return
    
    # Create segment vectors (common for both pipelines)
    segment_vectors = create_segment_vectors(
        users_df=raw_data_dict['users'],
        segments_df=raw_data_dict['segments']
    )

    # --- PIPELINE 1: TF-IDF ---
    print("\n" + "="*50)
    print("  RUNNING PIPELINE WITH TF-IDF EMBEDDINGS")
    print("="*50)
    
    banner_embeddings_tfidf, _ = create_banner_embeddings(
        banners_df=raw_data_dict['banners']
    )
    final_train_tfidf, _ = create_final_datasets(
        p_train_df=raw_data_dict['p_train'],
        p_test_df=raw_data_dict['p_test'],
        segment_vectors=segment_vectors,
        banner_embeddings=banner_embeddings_tfidf
    )
    print("\n--- Training model on TF-IDF features ---")
    train_and_evaluate(final_train_tfidf)

    # --- PIPELINE 2: ParsBERT ---
    print("\n" * 3 + "="*50)
    print("  RUNNING PIPELINE WITH PARSBERT EMBEDDINGS")
    print("="*50)

    banner_embeddings_bert = create_banner_embeddings_bert(
        banners_df=raw_data_dict['banners']
    )
    final_train_bert, _ = create_final_datasets(
        p_train_df=raw_data_dict['p_train'],
        p_test_df=raw_data_dict['p_test'],
        segment_vectors=segment_vectors,
        banner_embeddings=banner_embeddings_bert
    )
    print("\n--- Training model on ParsBERT features ---")
    train_and_evaluate(final_train_bert)

    print("\n✅ --- Both pipelines executed successfully! --- ✅")


if __name__ == '__main__':
    main()