import polars as pl
import argparse
import time

def compute_graph_features(edges_path: str, output_path: str):
    print(f"Loading data from {edges_path}...")
    t0 = time.time()
    
    df = pl.read_csv(edges_path)
    
    if df.schema["timestamp"] == pl.String:
        df = df.with_columns(pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f"))
        
    df = df.sort(["source_id", "timestamp"])
    
    print("Computing rolling outgoing features (out_degree, neighbor_entropy)...")
    out_features = df.rolling(
        index_column="timestamp",
        group_by="source_id",
        period="1d",
        closed="left"
    ).agg([
        pl.col("target_id").n_unique().alias("src_out_degree_1d"),
        pl.col("amount").entropy(normalize=True).fill_nan(0.0).alias("src_neighbor_entropy")
    ])
    out_features = out_features.with_columns(df["txn_id"])
    
    print("Computing incoming events for in_degree and delta_time...")
    # To compute strictly point-in-time in_degree for ANY timestamp, we can use group_by_dynamic
    # on the incoming events, BUT it's easier to use a window function or `join_asof`.
    # Let's create an event log of INCOMING transactions for each node.
    in_events = df.select([
        pl.col("target_id").alias("node_id"),
        pl.col("source_id").alias("sender_id"),
        pl.col("timestamp")
    ]).sort(["node_id", "timestamp"])
    
    # We can pre-aggregate these by day or just use `rolling` but `rolling` needs to evaluate AT the outgoing timestamp!
    # A powerful trick in Polars: append query points to the event log!
    # Query points are the OUTGOING transactions where we need to know the in_degree.
    query_points = df.select([
        pl.col("source_id").alias("node_id"),
        pl.col("timestamp"),
        pl.col("txn_id")
    ]).with_columns(pl.lit("query").alias("event_type"))
    
    data_points = in_events.with_columns([
        pl.lit("data").alias("event_type"),
        pl.lit(None).cast(pl.Int64).alias("txn_id")
    ])
    
    combined = pl.concat([
        data_points, 
        query_points.with_columns(pl.lit(None).cast(pl.Int64).alias("sender_id"))
    ], how="diagonal").sort(["node_id", "timestamp"])
    
    # Now use rolling on the combined dataframe
    combined_features = combined.rolling(
        index_column="timestamp",
        group_by="node_id",
        period="1d",
        closed="left"
    ).agg([
        pl.col("sender_id").drop_nulls().n_unique().alias("src_in_degree_1d"),
        pl.col("timestamp").drop_nulls().max().alias("last_in_timestamp")
    ])
    
    # Add txn_id back
    combined_features = combined_features.with_columns([
        combined["txn_id"],
        combined["event_type"],
        combined["timestamp"].alias("current_timestamp")
    ])
    
    # Filter only query points
    in_features = combined_features.filter(pl.col("event_type") == "query")
    
    # Compute ego_in_out_delta_time (seconds)
    in_features = in_features.with_columns(
        (pl.col("current_timestamp") - pl.col("last_in_timestamp")).dt.total_seconds().alias("src_ego_in_out_delta_time")
    ).drop(["event_type", "current_timestamp", "last_in_timestamp", "node_id", "timestamp"])
    
    print("Joining all features together...")
    # Join everything back to main df
    df = df.join(out_features.drop(["source_id", "timestamp"]), on="txn_id", how="left")
    df = df.join(in_features, on="txn_id", how="left")
    
    # Calculate degree ratio
    df = df.with_columns([
        (pl.col("src_out_degree_1d") / (pl.col("src_in_degree_1d") + 1)).alias("src_degree_ratio"),
        pl.col("src_ego_in_out_delta_time").fill_null(86400 * 30) # Fill missing with 30 days
    ])
    
    # Sort back by timestamp
    df = df.sort("timestamp")
    
    print(f"Features computed in {time.time() - t0:.1f}s")
    print(f"Saving to {output_path}...")
    df.write_csv(output_path)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="datasets/synthfin-small_edges.csv")
    parser.add_argument("--output", default="datasets/synthfin-small_edges_features.csv")
    args = parser.parse_args()
    compute_graph_features(args.input, args.output)
