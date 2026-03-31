import pandas as pd
import json
import zlib

def get_tweet_type(row):
    if pd.notna(row.get('retweeted_status')):
        return 'Retweet'
    elif pd.notna(row.get('in_reply_to_status_id')) or pd.notna(row.get('in_reply_to_user_id')):
        return 'Reply'
    else:
        return 'Original'

def extract_user_behavior_sequences(df, user_id_col='user_id'):
    user_sequences = {}
    for _, row in df.iterrows():
        user_id = row[user_id_col]
        tweet_type = get_tweet_type(row)
        if user_id not in user_sequences:
            user_sequences[user_id] = []
        user_sequences[user_id].append(tweet_type)
    return user_sequences

def generate_behavior_sequence(tweet_types):
    type_mapping = {
        'Original': 'O',
        'Retweet': 'R',
        'Reply': 'R'
    }
    return ''.join([type_mapping.get(tweet_type, '?') for tweet_type in tweet_types])

def calculate_compression_ratio(sequence):
    original_size = len(sequence)
    compressed = zlib.compress(sequence.encode('utf-8'))
    compressed_size = len(compressed)
    if compressed_size == 0:
        return 0
    return original_size / compressed_size

def process_behavior_sequences(input_file, output_file):
    df = pd.read_csv(input_file, low_memory=False)
    user_sequences = extract_user_behavior_sequences(df)
    
    results = []
    for user_id, tweet_types in user_sequences.items():
        behavior_sequence = generate_behavior_sequence(tweet_types)
        original_size = len(behavior_sequence)
        compression_ratio = calculate_compression_ratio(behavior_sequence)
        
        results.append({
            'user_id': user_id,
            'behavior_sequence': behavior_sequence,
            'original_sequence_size': original_size,
            'compression_ratio': compression_ratio
        })
    
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"✅ 行为序列处理完成，结果保存到 {output_file}")

if __name__ == "__main__":
    input_file = "expanded_tweet_data_cleaned.csv"
    output_file = "user_behavior_sequences.csv"
    process_behavior_sequences(input_file, output_file)
