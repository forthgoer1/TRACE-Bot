import pandas as pd
import json
import os

def extract_user_info(user_json_str):
    try:
        user_data = json.loads(user_json_str)
        return {
            'followers_count': user_data.get('followers_count', 0),
            'friends_count': user_data.get('friends_count', 0),
            'statuses_count': user_data.get('statuses_count', 0),
            'favourites_count': user_data.get('favourites_count', 0),
            'listed_count': user_data.get('listed_count', 0),
            'verified': int(user_data.get('verified', False)),
            'default_profile': int(user_data.get('default_profile', False)),
            'protected': int(user_data.get('protected', False)),
            'geo_enabled': int(user_data.get('geo_enabled', False)),
            'profile_use_background_image': int(user_data.get('profile_use_background_image', False)),
            'default_profile_image': int(user_data.get('default_profile_image', False)),
            'has_extended_profile': int(user_data.get('has_extended_profile', False)),
            'follow_request_sent': int(user_data.get('follow_request_sent', False)),
            'is_translation_enabled': int(user_data.get('is_translation_enabled', False)),
            'contributors_enabled': int(user_data.get('contributors_enabled', False)),
            'is_translator': int(user_data.get('is_translator', False)),
            'profile_background_tile': int(user_data.get('profile_background_tile', False))
        }
    except Exception as e:
        return {
            'followers_count': 0,
            'friends_count': 0,
            'statuses_count': 0,
            'favourites_count': 0,
            'listed_count': 0,
            'verified': 0,
            'default_profile': 0,
            'protected': 0,
            'geo_enabled': 0,
            'profile_use_background_image': 0,
            'default_profile_image': 0,
            'has_extended_profile': 0,
            'follow_request_sent': 0,
            'is_translation_enabled': 0,
            'contributors_enabled': 0,
            'is_translator': 0,
            'profile_background_tile': 0
        }

def integrate_features(input_files, output_file):
    df_main = pd.read_csv(input_files['main'], low_memory=False)
    df_behavior = pd.read_csv(input_files['behavior'], low_memory=False)
    df_gltr = pd.read_csv(input_files['gltr'], low_memory=False)
    df_fdgpt = pd.read_csv(input_files['fdgpt'], low_memory=False)
    
    merged_df = df_main.merge(df_behavior, on='user_id', how='left')
    merged_df = merged_df.merge(df_gltr, on='user_id', how='left')
    merged_df = merged_df.merge(df_fdgpt, on='user_id', how='left')
    
    user_info_list = []
    for _, row in merged_df.iterrows():
        user_json_str = row.get('user', '{}')
        user_info = extract_user_info(user_json_str)
        user_info['user_id'] = row.get('user_id')
        user_info_list.append(user_info)
    
    user_info_df = pd.DataFrame(user_info_list)
    merged_df = merged_df.merge(user_info_df, on='user_id', how='left')
    
    merged_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"✅ 特征整合完成，结果保存到 {output_file}")

if __name__ == "__main__":
    input_files = {
        'main': "expanded_tweet_data_cleaned.csv",
        'behavior': "user_behavior_sequences.csv",
        'gltr': "GLTR_results.csv",
        'fdgpt': "FDGPT_results.csv"
    }
    output_file = "feature_user_behavior_GLTR_FDGPT_label_results.csv"
    integrate_features(input_files, output_file)
