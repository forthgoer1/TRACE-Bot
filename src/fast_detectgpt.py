import random
import numpy as np
import torch
import os
import glob
import argparse
import json
import pandas as pd
from model import load_tokenizer, load_model
from fast_detect_gpt import get_sampling_discrepancy_analytic
from scipy.stats import norm

def compute_prob_norm(x, mu0, sigma0, mu1, sigma1):
    pdf_value0 = norm.pdf(x, loc=mu0, scale=sigma0)
    pdf_value1 = norm.pdf(x, loc=mu1, scale=sigma1)
    prob = pdf_value1 / (pdf_value0 + pdf_value1)
    return prob

class FastDetectGPT:
    def __init__(self, args):
        self.args = args
        self.criterion_fn = get_sampling_discrepancy_analytic
        self.scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.cache_dir)
        self.scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
        self.scoring_model.eval()
        if args.sampling_model_name != args.scoring_model_name:
            self.sampling_tokenizer = load_tokenizer(args.sampling_model_name, args.cache_dir)
            self.sampling_model = load_model(args.sampling_model_name, args.device, args.cache_dir)
            self.sampling_model.eval()

        distrib_params = {
            'gpt-j-6B_gpt-neo-2.7B': {'mu0': 0.2713, 'sigma0': 0.9366, 'mu1': 2.2334, 'sigma1': 1.8731},
            'gpt-neo-2.7B_gpt-neo-2.7B': {'mu0': -0.2489, 'sigma0': 0.9968, 'mu1': 1.8983, 'sigma1': 1.9935},
            'falcon-7b_falcon-7b-instruct': {'mu0': -0.0707, 'sigma0': 0.9520, 'mu1': 2.9306, 'sigma1': 1.9039},
        }
        key = f'{args.sampling_model_name}_{args.scoring_model_name}'
        self.classifier = distrib_params[key]

    def compute_crit(self, text):
        tokenized = self.scoring_tokenizer(text, truncation=True, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = self.scoring_model(**tokenized).logits[:, :-1]
            if self.args.sampling_model_name == self.args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = self.sampling_tokenizer(text, truncation=True, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.args.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = self.sampling_model(**tokenized).logits[:, :-1]
            crit = self.criterion_fn(logits_ref, logits_score, labels)
        return crit, labels.size(1)

    def compute_prob(self, text):
        crit, ntoken = self.compute_crit(text)
        mu0 = self.classifier['mu0']
        sigma0 = self.classifier['sigma0']
        mu1 = self.classifier['mu1']
        sigma1 = self.classifier['sigma1']
        prob = compute_prob_norm(crit, mu0, sigma0, mu1, sigma1)
        return prob, crit, ntoken

def run(args):
    detector = FastDetectGPT(args)

    input_file = "expanded_tweet_data_cleaned.csv"
    output_file = "FDGPT_results.csv"

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    df = pd.read_csv(input_file, low_memory=False)

    total_rows = len(df)
    print(f"--Total rows to process: {total_rows}--")

    if 'text' not in df.columns:
        print("Error: The input CSV must contain a 'text' column.")
        return

    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file, low_memory=False)
        last_index = existing_df.index[-1] if not existing_df.empty else -1
        df = df.iloc[last_index + 1:]
        print(f"Resuming from row {last_index + 1}...")
    else:
        last_index = -1
        existing_df = pd.DataFrame(columns=df.columns)

    batch_size = 10000
    results = []
    for idx, row in df.iterrows():
        text = row['text']
        if isinstance(text, str):
            prob, crit, ntokens = detector.compute_prob(text)
            results.append({
                "FDGPT_probability": prob,
                "FDGPT_criterion": crit,
                "FDGPT_tokens": ntokens,
            })
        else:
            results.append({
                "FDGPT_probability": None,
                "FDGPT_criterion": None,
                "FDGPT_tokens": None,
            })

        if (idx + 1) % 1000 == 0:
            print(f"--Processed {idx + 1} rows...--")

        if (idx + 1) % batch_size == 0:
            temp_df = pd.concat([existing_df, df.iloc[:idx + 1].assign(**pd.DataFrame(results))], ignore_index=True)
            temp_df.to_csv(output_file, index=False)
            print(f"==Saved progress up to row {idx + 1}.==")

    final_df = pd.concat([existing_df, df.assign(**pd.DataFrame(results))], ignore_index=True)
    final_df.to_csv(output_file, index=False)
    print(f"***All processing complete. Final results saved to '{output_file}'.***")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling_model_name', type=str, default="gpt-neo-2.7B")
    parser.add_argument('--scoring_model_name', type=str, default="gpt-neo-2.7B")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    run(args)
