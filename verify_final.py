import os
# Optimize OpenBLAS for Implicit ALS
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import polars as pl
import numpy as np
import scipy.sparse as sparse
import implicit
import time
from datetime import datetime, timedelta

def verify_full_pipeline():
    print("Starting Final Verification...")
    start_total = time.time()

    # 1. Load Data
    print("\n[1/6] Loading & Filtering Data...")
    try:
        df = pl.read_csv("dataset_purchases.csv", try_parse_dates=False)
        df = df.with_columns(
            pl.col("timestamp").str.replace(" UTC", "").str.to_datetime(format="%Y-%m-%d %H:%M:%S")
        )
        
        # Filter date range
        df = df.filter(
            (pl.col("timestamp") >= datetime(2024, 1, 15)) & 
            (pl.col("timestamp") <= datetime(2024, 2, 15))
        )
        print(f"Data Loaded: {len(df)} rows (2024-01-15 to 2024-02-15)")
        
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. Temporal Split
    print("\n[2/6] Temporal Split (Last 7 Days)...")
    max_date = df["timestamp"].max()
    test_start_date = max_date - timedelta(days=7)
    
    train_raw = df.filter(pl.col("timestamp") < test_start_date)
    test_raw = df.filter(pl.col("timestamp") >= test_start_date)
    
    # Calculate percentages
    total_rows = len(df)
    train_pct = (len(train_raw) / total_rows) * 100
    test_pct = (len(test_raw) / total_rows) * 100
    
    print(f"Train: {len(train_raw)} ({train_pct:.2f}%), Test: {len(test_raw)} ({test_pct:.2f}%)")

    # 3. Item Definition & Mapping
    print("\n[3/6] Defining Items (USD, Coins)...")
    full_df = pl.concat([train_raw, test_raw])
    unique_items = full_df.select(['usd', 'coins']).unique().sort(['usd', 'coins']).with_row_index("item_id")
    unique_users = full_df.select('user_id').unique().sort('user_id').with_row_index("user_idx")
    
    train_data = train_raw.join(unique_items, on=['usd', 'coins'], how='left') \
                          .join(unique_users, on='user_id', how='left')
    
    test_data = test_raw.join(unique_items, on=['usd', 'coins'], how='left') \
                        .join(unique_users, on='user_id', how='left')
    
    print(f"Unique Items: {len(unique_items)}, Unique Users: {len(unique_users)}")

    # 4. Matrix Construction
    print("\n[4/6] Building Matrix...")
    train_agg = train_data.group_by(['user_idx', 'item_id']).len().rename({'len': 'count'})
    rows = train_agg['user_idx'].to_numpy()
    cols = train_agg['item_id'].to_numpy()
    data = train_agg['count'].to_numpy()
    
    sparse_train = sparse.csr_matrix(
        (data, (rows, cols)), 
        shape=(len(unique_users), len(unique_items))
    )

    # 5. Training
    print("\n[5/6] Training ALS...")
    use_gpu = False
    model = implicit.als.AlternatingLeastSquares(
        factors=64, regularization=0.05, alpha=40, iterations=15, 
        random_state=42, use_gpu=use_gpu
    )
    model.fit(sparse_train)

    # 6. Final Function Test
    print("\n[6/6] Testing Recommendations Function...")
    item_lookup_df = unique_items.to_pandas().set_index("item_id")
    user_to_idx = dict(zip(unique_users["user_id"], unique_users["user_idx"]))

    def get_recommendations(user_id):
        if user_id not in user_to_idx:
            return []
        u_idx = user_to_idx[user_id]
        ids, scores = model.recommend(u_idx, sparse_train[u_idx], N=6, filter_already_liked_items=False)
        recs = item_lookup_df.loc[ids]
        recs['vfm'] = recs['coins'] / recs['usd']
        results = []
        for _, row in recs.iterrows():
            results.append({'price': row['usd'], 'vfm': row['vfm']})
        return results

    sample_user = full_df['user_id'][0]
    recs = get_recommendations(sample_user)
    print(f"Recs for {sample_user}: {recs}")
    
    print(f"\nTotal Time: {time.time() - start_total:.2f}s")
    print("VERIFICATION SUCCESSFUL")

if __name__ == "__main__":
    verify_full_pipeline()
