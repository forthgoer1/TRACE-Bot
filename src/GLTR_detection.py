import torch
import numpy as np
import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertTokenizer, BertForMaskedLM
import pandas as pd
import os

class AbstractLanguageChecker:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def check_probabilities(self, in_text, topk=40):
        raise NotImplementedError

    def postprocess(self, token):
        raise NotImplementedError

class LM(AbstractLanguageChecker):
    def __init__(self, model_name_or_path="./gpt2"):
        super(LM, self).__init__()
        self.enc = GPT2Tokenizer.from_pretrained(model_name_or_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()
        self.start_token = self.enc(self.enc.bos_token, return_tensors='pt').data['input_ids'][0]
        print(f"Loaded GPT-2 model! on {self.device}")

    def check_probabilities(self, in_text, topk=40):
        if not isinstance(in_text, str):
            print(f"Warning: Input 'in_text' is not a string. Converting to string. Original type: {type(in_text)}")
            in_text = str(in_text)

        if not in_text.strip():
            raise ValueError("Input text is empty or contains only whitespace.")

        token_ids = self.enc(in_text, return_tensors='pt')['input_ids'][0]
        token_ids = torch.cat([self.start_token, token_ids])
        output = self.model(token_ids.to(self.device))
        
        all_logits = output.logits[:-1].detach()
        if len(all_logits.shape) == 1:
            all_logits = all_logits.unsqueeze(0)
        all_probs = torch.softmax(all_logits, dim=-1)

        y = token_ids[1:]
        real_topk_probs = all_probs[np.arange(0, y.shape[0], 1), y].cpu().numpy()

        log_seq_prob = sum(math.log(max(p, 1e-10)) for p in real_topk_probs)
        seq_prob = math.exp(log_seq_prob)

        sorted_preds = torch.argsort(all_probs, dim=1, descending=True).cpu()
        
        real_topk_pos = []
        for i in range(y.shape[0]):
            match = np.where(sorted_preds[i] == y[i].item())[0]
            if match.size > 0:
                real_topk_pos.append(int(match[0]))
            else:
                real_topk_pos.append(-1)
        
        real_topk_probs = list(map(lambda x: round(x, 5), real_topk_probs.tolist()))
        real_topk = list(zip(real_topk_pos, real_topk_probs))
        bpe_strings = self.enc.convert_ids_to_tokens(token_ids[:])
        bpe_strings = [self.postprocess(s) for s in bpe_strings]

        topk_prob_values, topk_prob_inds = torch.topk(all_probs, k=topk, dim=1)
        pred_topk = [list(zip(self.enc.convert_ids_to_tokens(topk_prob_inds[i]),
                            topk_prob_values[i].cpu().numpy().tolist())) for i in range(y.shape[0])]
        pred_topk = [[(self.postprocess(t[0]), t[1]) for t in pred] for pred in pred_topk]

        payload = {'bpe_strings': bpe_strings,
                'real_topk': real_topk,
                'pred_topk': pred_topk,
                'sequence_probability': seq_prob}

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return payload

    def postprocess(self, token):
        with_space = False
        with_break = False
        if token.startswith('Ġ'):
            with_space = True
            token = token[1:]
        elif token.startswith('â'):
            token = ' '
        elif token.startswith('Ċ'):
            token = ' '
            with_break = True

        token = '-' if token.startswith('â') else token
        token = '“' if token.startswith('ľ') else token
        token = '”' if token.startswith('Ŀ') else token
        token = "'" if token.startswith('Ļ') else token

        if with_space:
            token = '\u0120' + token
        if with_break:
            token = '\u010A' + token

        return token

class BERTLM(AbstractLanguageChecker):
    def __init__(self, model_name_or_path="./bert-base-chinese"):
        super(BERTLM, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path, do_lower_case=False)
        self.model = BertForMaskedLM.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()
        self.mask_tok = self.tokenizer.convert_tokens_to_ids(["[MASK]"])[0]
        self.pad = self.tokenizer.convert_tokens_to_ids(["[PAD]"])[0]
        print("Loaded BERT model!")

    def check_probabilities(self, in_text, topk=40, max_context=20, batch_size=20):
        if not isinstance(in_text, str):
            print(f"Warning: Input 'in_text' is not a string. Converting to string. Original type: {type(in_text)}")
            in_text = str(in_text)

        in_text = "[CLS] " + in_text + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(in_text)
        y_toks = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0] * len(y_toks)
        y = torch.tensor([y_toks]).to(self.device)
        segments_tensor = torch.tensor([segments_ids]).to(self.device)

        input_batches = []
        target_batches = []
        real_topk_probs_all = []

        for min_ix in range(0, len(y_toks), batch_size):
            max_ix = min(min_ix + batch_size, len(y_toks) - 1)
            cur_input_batch = []
            cur_target_batch = []
            for running_ix in range(max_ix - min_ix):
                tokens_tensor = y.clone()
                mask_index = min_ix + running_ix
                tokens_tensor[0, mask_index + 1] = self.mask_tok

                min_index = max(0, mask_index - max_context)
                max_index = min(tokens_tensor.shape[1] - 1, mask_index + max_context + 1)
                tokens_tensor = tokens_tensor[:, min_index:max_index]
                needed_padding = max_context * 2 + 1 - tokens_tensor.shape[1]

                if min_index == 0 and max_index == y.shape[1] - 1:
                    left_needed = (max_context) - mask_index
                    right_needed = needed_padding - left_needed
                    p = torch.nn.ConstantPad1d((left_needed, right_needed), self.pad)
                    tokens_tensor = p(tokens_tensor)
                elif min_index == 0:
                    p = torch.nn.ConstantPad1d((needed_padding, 0), self.pad)
                    tokens_tensor = p(tokens_tensor)
                elif max_index == y.shape[1] - 1:
                    p = torch.nn.ConstantPad1d((0, needed_padding), self.pad)
                    tokens_tensor = p(tokens_tensor)

                cur_input_batch.append(tokens_tensor)
                cur_target_batch.append(y[:, mask_index + 1])
            
            if len(cur_input_batch) > 0:
                cur_input_batch = torch.cat(cur_input_batch, dim=0)
                cur_target_batch = torch.cat(cur_target_batch, dim=0)
                input_batches.append(cur_input_batch)
                target_batches.append(cur_target_batch)
            else:
                print(f"Warning: Empty batch encountered at index {min_ix}. Skipping this batch.")

        with torch.no_grad():
            for src, tgt in zip(input_batches, target_batches):
                outputs = self.model(src, attention_mask=(src != self.pad).long())
                logits = outputs.logits[:, max_context + 1]
                yhat = torch.softmax(logits, dim=-1)
                real_topk_probs = yhat[np.arange(0, yhat.shape[0], 1), tgt.squeeze(-1)].data.cpu().numpy().tolist()
                real_topk_probs_all.extend([p + 1e-10 for p in real_topk_probs])

        log_seq_prob = np.sum(np.log(np.array(real_topk_probs_all)))
        seq_prob = np.exp(log_seq_prob)

        bpe_strings = [self.postprocess(s) for s in tokenized_text]

        payload = {'bpe_strings': bpe_strings,
                'sequence_probability': seq_prob}
        return payload

    def postprocess(self, token):
        with_space = True
        with_break = token == '[SEP]'
        if token.startswith('##'):
            with_space = False
            token = token[2:]

        if with_space:
            token = '\u0120' + token
        if with_break:
            token = '\u010A' + token
        return token

def process_GLTR(input_file, output_file):
    df = pd.read_csv(input_file, low_memory=False)
    bert_lm = BERTLM()
    gpt2_lm = LM()
    
    results = []
    total_rows = len(df)
    print(f"Total rows to process: {total_rows}")
    
    for idx, row in df.iterrows():
        text = row.get('text', '')
        if isinstance(text, str) and text.strip():
            try:
                bert_result = bert_lm.check_probabilities(text)
                gpt2_result = gpt2_lm.check_probabilities(text)
                
                bert_prob = bert_result.get('sequence_probability', 0)
                gpt2_prob = gpt2_result.get('sequence_probability', 0)
                
                results.append({
                    'user_id': row.get('user_id'),
                    'GLTR_bert_prob': bert_prob,
                    'GLTR_gpt2_prob': gpt2_prob
                })
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                results.append({
                    'user_id': row.get('user_id'),
                    'GLTR_bert_prob': 0,
                    'GLTR_gpt2_prob': 0
                })
        else:
            results.append({
                'user_id': row.get('user_id'),
                'GLTR_bert_prob': 0,
                'GLTR_gpt2_prob': 0
            })
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{total_rows} rows")
    
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"✅ GLTR processing completed. Results saved to {output_file}")

if __name__ == "__main__":
    input_file = "expanded_tweet_data_cleaned.csv"
    output_file = "GLTR_results.csv"
    process_GLTR(input_file, output_file)
