import pandas as pd
import numpy as np
from src import config

def load_raw_data():
    """
    Loads all raw data files (.npy, .txt, .csv) and converts them to pandas DataFrames.
    """
    print("--- Loading Raw Data ---")
    
    try:
        users_npy = np.load(config.USER_MATRIX_PATH)
        users_df = pd.DataFrame(users_npy, columns=[f'f{i}' for i in range(1, config.USER_VECTOR_DIM + 1)])
        users_df['user_id'] = users_df.index
        print(f"✅ Users data (.npy) loaded. Shape: {users_df.shape}")

        segments_npy = np.load(config.SEGMENTS_ARRAY_PATH)
        segments_df = pd.DataFrame({'user_id': range(len(segments_npy)), 'segment': segments_npy})
        print(f"✅ Segments data (.npy) loaded. Shape: {segments_df.shape}")

        banners_df = pd.read_csv(config.BANNER_CAPTIONS_PATH, header=None, names=['caption'])
        banners_df['banner_id'] = banners_df.index
        print(f"✅ Banners data (.txt) loaded. Shape: {banners_df.shape}")

        p_train_df = pd.read_csv(config.TRAIN_PREFERENCES_PATH)
        print(f"✅ Training preferences data loaded. Shape: {p_train_df.shape}")

        p_test_df = pd.read_csv(config.TEST_PREFERENCES_PATH)
        print(f"✅ Test preferences data loaded. Shape: {p_test_df.shape}")

        data_dict = {
            'users': users_df, 'banners': banners_df, 'segments': segments_df,
            'p_train': p_train_df, 'p_test': p_test_df
        }
        
        print("\nAll raw data loaded.")
        return data_dict

    except FileNotFoundError as e:
        print(f"🚨 ERROR: File not found. Make sure the path in config.py is correct.")
        print(e)
        return None


def create_final_datasets(p_train_df, p_test_df, segment_vectors, banner_embeddings):
    """
    Merges preference data with segment and banner vectors to create final,
    model-ready datasets. Uses correct column names 'i' and 'j'.
    """
    print("--- Creating Final Model-Ready Datasets ---")

    # --- Clean up p_train and p_test ---
    # Drop the extra index column and rename 'i' and 'j' for clarity
    p_train_df = p_train_df.rename(columns={'i': 'segment', 'j': 'banner_id'}).drop(columns=['Unnamed: 0'])
    p_test_df = p_test_df.rename(columns={'i': 'segment', 'j': 'banner_id'}).drop(columns=['Unnamed: 0'])

    # --- Prepare segment vectors for merging ---
    seg_vec_df = segment_vectors.add_prefix('sv_')

    # --- Prepare banner embeddings for merging ---
    banner_emb_df = pd.DataFrame(banner_embeddings).add_prefix('bv_')

    # --- Merge data using the correct columns and indexes ---
    # We merge on the column in the left df and the index in the right df
    train_df = pd.merge(p_train_df, seg_vec_df, left_on='segment', right_index=True)
    final_train_df = pd.merge(train_df, banner_emb_df, left_on='banner_id', right_index=True)
    print(f"✅ Final training dataset created. Shape: {final_train_df.shape}")

    test_df = pd.merge(p_test_df, seg_vec_df, left_on='segment', right_index=True)
    final_test_df = pd.merge(test_df, banner_emb_df, left_on='banner_id', right_index=True)
    print(f"✅ Final test dataset created. Shape: {final_test_df.shape}")

    # --- Save processed data ---
    final_train_df.to_csv(config.PROCESSED_DATA_DIR / "final_train.csv", index=False)
    final_test_df.to_csv(config.PROCESSED_DATA_DIR / "final_test.csv", index=False)
    print(f"💾 Processed datasets saved to {config.PROCESSED_DATA_DIR}")
    
    return final_train_df, final_test_df