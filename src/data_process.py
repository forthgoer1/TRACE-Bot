import pandas as pd
import json
import os

def ndjson_to_csv(ndjson_file, csv_file):
    try:
        with open(ndjson_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f if line.strip()]
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"✅ 成功将 {ndjson_file} 转换为 {csv_file}")
    except Exception as e:
        print(f"❌ 转换失败: {e}")

def process_json_column(df, column_name):
    try:
        if column_name in df.columns:
            json_data = df[column_name].dropna().apply(json.loads)
            expanded_df = pd.json_normalize(json_data)
            expanded_df.columns = [f"{column_name}_{col}" for col in expanded_df.columns]
            df = df.drop(column_name, axis=1)
            df = pd.concat([df, expanded_df], axis=1)
            print(f"✅ 成功处理 {column_name} 列")
        else:
            print(f"⚠️  列 {column_name} 不存在")
    except Exception as e:
        print(f"❌ 处理 {column_name} 列失败: {e}")
    return df

def expand_nested_columns(df, columns_to_expand):
    for col in columns_to_expand:
        if col in df.columns:
            df = process_json_column(df, col)
    return df

if __name__ == "__main__":
    ndjson_file = "tweets.ndjson"
    csv_file = "tweet_data.csv"
    
    ndjson_to_csv(ndjson_file, csv_file)
    
    df = pd.read_csv(csv_file, low_memory=False)
    print(f"初始数据形状: {df.shape}")
    
    columns_to_expand = ['user', 'entities', 'retweeted_status', 'quoted_status']
    df = expand_nested_columns(df, columns_to_expand)
    
    output_file = "expanded_tweet_data_cleaned.csv"
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"✅ 处理完成，结果保存到 {output_file}")
    print(f"处理后数据形状: {df.shape}")
