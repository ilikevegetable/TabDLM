import os, random
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
import json
from .word_dict import SPECIALS
from sklearn.preprocessing import QuantileTransformer


def stas_prompt_construct(col_order, stas_info, dataset_name):
    def fmt_num(x, nd=2):
        if x is None: return "nan"
        try:
            return str(int(x)) if float(x).is_integer() else f"{float(x):.{nd}f}"
        except: return str(x)

    stats_items = []
    for c in col_order:
        meta = stas_info[c]
        ty = meta.get("type","").lower()
        if ty in ("float","int","number","numeric"):
            part = (f"{c}:min={fmt_num(meta.get('min'))},max={fmt_num(meta.get('max'))},"
                    f"mean={fmt_num(meta.get('mean'))},median={fmt_num(meta.get('median'))},std={fmt_num(meta.get('std'))}")
        elif ty in ("category","categorical","enum"):
            cands = meta.get("candidates", [])
            part = f"{c}:{{" + ", ".join(map(str, cands)) + "}"
        elif ty in ("text"):
            part = f"{c}:free-form text"
        else:
            part = f"{c}:{{unknown}}"
        stats_items.append(part)

    prompt = f"The following are the statistics information for the columns in the {dataset_name} table: "

    return prompt + SPECIALS['STATS_B'] + "; ".join(stats_items) + SPECIALS['STATS_E']


def schema_prompt_construct(col_order, stas_info, dataset_name):
    schema_items = []
    # TODO: add timestamp features
    for c in col_order:
        meta = stas_info[c]
        ty = meta.get("type", "").lower()
        if ty in ("float","int","number","numeric"):
            schema_items.append(f"{c}: numerical")
        elif ty in ("category","categorical","enum"):
            schema_items.append(f"{c}: categorical")
        else:
            schema_items.append(f"{c}: textual")

    prompt = f"The following is the schema for the {dataset_name} table: "

    return prompt + SPECIALS['SCHEMA_B'] + "; ".join(schema_items) + SPECIALS['SCHEMA_E']


class TabularDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        tokenizer,
        template_probs,
        max_len,
        answer_len,
        normalization='quantile',
        split='train',
        all_numerical=False):
        self.tok = tokenizer
        self.tpl = template_probs
        self.max_len = max_len
        self.answer_len = answer_len
        self.split = split
        self.normalization = normalization
        self.all_numerical = all_numerical
        self.pad_id = tokenizer.pad_token_id # EOS/<|endoftext|> / 126081
        self.num_token_id = 126090
        self.tpl_keys = list(self.tpl.keys())
        self.tpl_weights = [self.tpl[k] for k in self.tpl_keys]

        with open(f"data/tabular/{dataset_name}/info.json", "r") as f:
            self.stas_info = json.load(f)
        # --dataset_name: paths (data/, ckpt/, result/). info.json "name": prompt display only.
        self.dataset_dir = dataset_name
        self.prompt_table_name = "tabular:" + self.stas_info["name"]
        self.numerical_columns = [self.stas_info['column_names'][i] for i in self.stas_info['num_col_idx']]

        if self.all_numerical and dataset_name == 'magic':
            self.stas_info['column_info']['class']['type'] = 'float'

        target_col = self.stas_info['column_names'][self.stas_info['target_col_idx'][0]]
        if self.stas_info['column_info'][target_col]['type'] == 'float':
            self.numerical_columns.append(target_col)

        self.nonNumerical_columns = [col_name for col_name in self.stas_info['column_names'] if col_name not in self.numerical_columns]
        self.col_order = self.numerical_columns + self.nonNumerical_columns

        if self.all_numerical:
            # Promote every remaining categorical column to a numerical channel.
            for col in self.stas_info['column_info'].keys():
                self.stas_info['column_info'][col]['type'] = 'float'
            for col in self.nonNumerical_columns:
                self.numerical_columns.append(col)

        self.num_numerical_columns = len(self.numerical_columns)
        # construct statistic prompt
        self.stats_prompt = stas_prompt_construct(self.col_order, self.stas_info['column_info'], self.prompt_table_name)
        self.schema_prompt = schema_prompt_construct(self.col_order, self.stas_info['column_info'], self.prompt_table_name)
        if split == "train":
            print("SCHEMA PROMPT: " + self.schema_prompt)
            print("STATS PROMPT: " + self.stats_prompt)

        self.df = pd.read_csv(f"data/tabular/{dataset_name}/{split}.csv")
        self.ori_columns = self.df.columns
        self.df_train = pd.read_csv(f"data/tabular/{dataset_name}/train.csv")

        if self.all_numerical:
            # must map categorical strings to integer codes BEFORE `_impute_num_missing_with_train_mean`
            if dataset_name == 'magic':
                self.df['class'] = self.df['class'].map({'g': 1, 'h': 0})
                self.df_train['class'] = self.df_train['class'].map({'g': 1, 'h': 0})

            num2cat_list = []
            bol2cat_list = []
            if dataset_name == 'shoppers':
                num2cat_list = ['OperatingSystems', 'Browser', 'Region', 'TrafficType']
                bol2cat_list = ['Weekend', 'Revenue']
            elif dataset_name == 'beijing':
                num2cat_list = ['year', 'month', 'day', 'hour']
            elif dataset_name == 'default':
                num2cat_list = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2',
                                'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                                'default payment next month']
            for col in num2cat_list + bol2cat_list:
                self.df[col] = self.df[col].astype(str).str.strip()
                self.df_train[col] = self.df_train[col].astype(str).str.strip()

            for col in self.nonNumerical_columns:
                candidates = self.stas_info['column_info'][col]['candidates']
                col_map = {can: idx for idx, can in enumerate(candidates)}
                self.df[col] = self.df[col].map(col_map)
                self.df_train[col] = self.df_train[col].map(col_map)

        self._impute_num_missing_with_train_mean()
        if self.normalization != 'none':
            if self.normalization == 'quantile':
                self._fit_quantile_transformers_on_train()
            self.normalize()
        self.reorder_col_order()

    def _impute_num_missing_with_train_mean(self):
        self.train_num_means = {}
        for col in self.numerical_columns:
            x_train = pd.to_numeric(self.df_train[col], errors="coerce")
            mu = x_train.mean(skipna=True)
            if pd.isna(mu):
                mu = 0.0
            self.train_num_means[col] = float(mu)

        # TODO: Use Mean value to fill null value (fixed)
        for col in self.numerical_columns:
            x = pd.to_numeric(self.df[col], errors="coerce")
            self.df[col] = x.fillna(self.train_num_means[col])

        for col in self.numerical_columns:
            x_train = pd.to_numeric(self.df_train[col], errors="coerce")
            self.df_train[col] = x_train.fillna(self.train_num_means[col])

    def _fit_quantile_transformers_on_train(self, seed=111):
        self.quantile_transformers = {}
        for col in self.numerical_columns:
            x_train = pd.to_numeric(self.df_train[col], errors="coerce")
            transformer = QuantileTransformer(
                output_distribution='normal',
                n_quantiles=max(min(len(x_train) // 30, 1000), 10),
                subsample=int(1e9),
                random_state=seed,
            )
            transformer.fit(x_train.values.reshape(-1, 1))
            self.quantile_transformers[col] = transformer

    def normalize(self, seed=111):
        if self.normalization == "standard":
            for col in self.numerical_columns:
                x = pd.to_numeric(self.df[col], errors="coerce")
                mu = float(self.stas_info['column_info'][col]["mean"])
                sd = float(self.stas_info['column_info'][col]["std"])
                if np.isnan(sd) or sd == 0:
                    sd = 1.0
                self.df[col] = (x - mu) / sd
        elif self.normalization == "quantile":
            for col in self.numerical_columns:
                x = pd.to_numeric(self.df[col], errors="coerce")
                transformer = self.quantile_transformers.get(col)
                if transformer is None:
                    continue
                x_transformed = transformer.transform(x.values.reshape(-1, 1)).flatten()
                self.df[col] = x_transformed

    def denormalize(self, df):
        if self.normalization == "standard":
            for col in self.numerical_columns:
                x = pd.to_numeric(df[col], errors="coerce")
                mu = float(self.stas_info['column_info'][col]["mean"])
                sd = float(self.stas_info['column_info'][col]["std"])
                df[col] = x * sd + mu
        elif self.normalization == "quantile":
            for col in self.numerical_columns:
                if col not in df.columns:
                    continue
                x = pd.to_numeric(df[col], errors="coerce")
                transformer = self.quantile_transformers.get(col)
                if transformer is None:
                    continue
                x_denormalized = transformer.inverse_transform(x.values.reshape(-1, 1)).flatten()
                df[col] = x_denormalized

        return df

    def reorder_col_order(self):
        self.df = self.df[self.col_order]

    def clean_val(self, v):
        return str(v).replace("|", "/").replace("\n", " ").strip()

    def row_text(self, row):
        if self.all_numerical:
            parts = [SPECIALS['NUMBER'] for _ in self.numerical_columns]
            return "".join(parts)
        parts = [f"{self.clean_val(row[c])}" if c not in self.numerical_columns else SPECIALS['NUMBER'] for c in self.col_order]
        # return f"{SPECIALS['ROW_B']}" + f"{SPECIALS['COL_SEP']}".join(parts) + f"{SPECIALS['ROW_E']}"
        return f"{SPECIALS['COL_SEP']}".join(parts)

    def build_prompt_answer(
            self,
            row,
            template):
        if template == "A":
            p = (f"{SPECIALS['BOS']}{SPECIALS['U_ST']}\n\n"
                 f"Generate one {self.prompt_table_name} table row."
                 f"{SPECIALS['EOT']}{SPECIALS['A_ST']}\n\n")

            r = self.row_text(row)
            return p, r

        # TODO: Select several columns as known values; generate multiple rows at once.

        if template == "B":
            p = (f"{SPECIALS['BOS']}{SPECIALS['U_ST']}\n\n"
                 f"{self.schema_prompt}\n"
                 f"Generate one {self.prompt_table_name} table row coherent with the above schema.\n"
                 f"{SPECIALS['EOT']}{SPECIALS['A_ST']}\n\n")
            r = self.row_text(row)
            return p, r

        if template == "C":
            p = (f"{SPECIALS['BOS']}{SPECIALS['U_ST']}\n\n"
                 f"{self.schema_prompt}\n"
                 f"{self.stats_prompt}\n"
                 f"Generate one {self.prompt_table_name} table row coherent with the above schema and stats.\n"
                 f"{SPECIALS['EOT']}{SPECIALS['A_ST']}\n\n")
            r = self.row_text(row)
            return p, r

        raise ValueError("Unknown template")

    def get_num_value_idx(self, prompt):
        r = self.df.iloc[10].to_dict()
        if self.all_numerical:
            text = prompt + self.row_text(r)
        else:
            text = prompt + self.row_text(r) + SPECIALS["EOS"]
        enc_all = self.tok(text, truncation=True, max_length=self.max_len, add_special_tokens=False)
        num_value_index = torch.where(torch.tensor(enc_all['input_ids']) == self.num_token_id)
        return num_value_index[0]

    def compute_answer_len_stats(self, templates=None, verbose=True):
        """Statistics of (len(enc_all['input_ids']) - prompt_len) over self.df,
        i.e. effective DLM content length (answer tokens + EOS) for each template
        with non-zero probability. Useful to sanity-check that `answer_len` is
        large enough to cover all training samples.
        """
        try:
            from tqdm import tqdm
        except Exception:
            tqdm = lambda x, **kw: x

        if templates is None:
            templates = [k for k in self.tpl_keys if self.tpl[k] > 0]

        stats = {}
        for template in templates:
            # prompt p does not depend on the row, so tokenize it once
            sample_row = self.df.iloc[0].to_dict()
            p, _ = self.build_prompt_answer(sample_row, template)
            enc_p = self.tok(p, truncation=True, max_length=self.max_len, add_special_tokens=False)
            prompt_len = len(enc_p["input_ids"])

            lens = []
            iterator = range(len(self.df))
            if verbose:
                iterator = tqdm(iterator, desc=f"[answer_len stats][tpl={template}]")
            for idx in iterator:
                row = self.df.iloc[idx].to_dict()
                _, r = self.build_prompt_answer(row, template)
                text = p + r + SPECIALS["EOS"]
                enc_all = self.tok(text, truncation=True, max_length=self.max_len, add_special_tokens=False)
                lens.append(len(enc_all["input_ids"]) - prompt_len)

            lens_arr = np.array(lens, dtype=np.int64)
            stats[template] = {
                "mean": float(lens_arr.mean()),
                "min": int(lens_arr.min()),
                "max": int(lens_arr.max()),
                "median": float(np.median(lens_arr)),
                "count": int(lens_arr.size),
                "prompt_len": int(prompt_len),
            }
        return stats

    def build_sample_prompt(
            self,
            template):
        if template == "A":
            p = (f"{SPECIALS['BOS']}{SPECIALS['U_ST']}\n\n"
                 f"Generate one {self.prompt_table_name} table row."
                 f"{SPECIALS['EOT']}{SPECIALS['A_ST']}\n\n")
            return p, self.get_num_value_idx(p)

        # TODO: Select several columns as known values; generate multiple rows at once.

        if template == "B":
            p = (f"{SPECIALS['BOS']}{SPECIALS['U_ST']}\n\n"
                 f"{self.schema_prompt}\n"
                 f"Generate one {self.prompt_table_name} table row coherent with the above schema.\n"
                 f"{SPECIALS['EOT']}{SPECIALS['A_ST']}\n\n")
            return p, self.get_num_value_idx(p)

        if template == "C":
            p = (f"{SPECIALS['BOS']}{SPECIALS['U_ST']}\n\n"
                 f"{self.schema_prompt}\n"
                 f"{self.stats_prompt}\n"
                 f"Generate one {self.prompt_table_name} table row coherent with the above schema and stats.\n"
                 f"{SPECIALS['EOT']}{SPECIALS['A_ST']}\n\n")
            return p, self.get_num_value_idx(p)

        raise ValueError("Unknown template")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx].to_dict()
        template = random.choices(self.tpl_keys, weights=self.tpl_weights, k=1)[0]
        p, r = self.build_prompt_answer(row, template)
        if self.all_numerical:
            text = p + r
            enc_all = self.tok(text, truncation=True, max_length=self.max_len, add_special_tokens=False)
            enc_p = self.tok(p, truncation=True, max_length=self.max_len, add_special_tokens=False)
            prompt_len = len(enc_p["input_ids"])
            ans_len = len(enc_all["input_ids"]) - prompt_len
            pad_ids = enc_all["input_ids"]
        else:
            text = p + r + SPECIALS["EOS"]
            enc_all = self.tok(text, truncation=True, max_length=self.max_len, add_special_tokens=False)
            enc_p = self.tok(p, truncation=True, max_length=self.max_len, add_special_tokens=False)
            prompt_len = len(enc_p["input_ids"])
            # pad answers to answer_len
            pad_token_num = max(self.answer_len - (len(enc_all["input_ids"]) - prompt_len), 0)
            ans_len = max(self.answer_len, len(enc_all["input_ids"]) - prompt_len)
            pad_ids = enc_all["input_ids"] + [self.pad_id] * pad_token_num
        num_values = [row[col] for col in self.numerical_columns]

        return {
            "input_ids": torch.tensor(pad_ids, dtype=torch.long),
            "prompt_len": torch.tensor(prompt_len, dtype=torch.long),
            "answer_len": torch.tensor(ans_len, dtype=torch.long),
            "num_value": torch.tensor(num_values, dtype=torch.float),
            "no_pad_answer_len": torch.tensor((len(enc_all["input_ids"]) - prompt_len), dtype=torch.long),
        }


@dataclass
class PadCollator:
    tokenizer: Any
    pad_to_multiple_of: int = 8

    def __call__(self, features: List[Dict[str,torch.Tensor]]) -> Dict[str,torch.Tensor]:
        ids = [f["input_ids"] for f in features]
        lens = [len(x) for x in ids]
        maxl = max(lens)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            maxl = (maxl + m - 1) // m * m
        pad_id = self.tokenizer.pad_token_id or 0

        batch_ids = []
        prompt_lens = []
        answer_lens = []
        num_values = []
        no_pad_answer_lens = []
        for f in features:
            x = f["input_ids"]
            pad = maxl - x.shape[0]
            if pad > 0:
                x = torch.cat([x, torch.full((pad,), pad_id, dtype=torch.long)], dim=0)
            batch_ids.append(x)
            prompt_lens.append(int(f["prompt_len"]))
            answer_lens.append(int(f["answer_len"]))
            num_values.append(f["num_value"])
            no_pad_answer_lens.append(int(f["no_pad_answer_len"]))
        batch = {
            "input_ids": torch.stack(batch_ids, dim=0),
            "prompt_lengths": torch.tensor(prompt_lens, dtype=torch.long),
            "answer_lengths": torch.tensor(answer_lens, dtype=torch.long),
            "num_values": torch.stack(num_values, dim=0),
            "no_pad_answer_len": torch.tensor(no_pad_answer_lens, dtype=torch.long)
        }
        return batch