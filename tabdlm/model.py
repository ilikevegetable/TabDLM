import torch.nn as nn
import os, math, random, json, argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

from .diffusion import diffusion_model
import wandb, time
import numpy as np
import pandas as pd
from tqdm import tqdm

S_churn = 1
S_min = 0
S_max = float('inf')
S_noise = 1


class TabDLM(nn.Module):
    def __init__(self,
                 args,
                 model_name,
                 num_numerical_features,
                 mask_token_id,
                 num_token_id,
                 eps,
                 use_bf16,
                 lora_parameters,
                 lora_r,
                 lora_alpha,
                 lora_dropout,
                 floatenc,
                 floatdec,
                 ae_hidden_dim,
                 dlm_hidden_dim,
                 loss_type='no_divide_pmask',
                 device='cuda',
                 noise_schedule_params={},
                 edm_params={},
                 sampler_params={},
                 num_timesteps=64,
                 scheduler='power_mean',
                 cat_scheduler='log_linear_per_column',
                 noise_dist='uniform_t',
                 all_numerical=False,
                 decoupled_steps=False,
                 **kwargs):
        super(TabDLM, self).__init__()
        self.args = args
        self.model_name = model_name
        self.mask_token_id = mask_token_id
        self.num_token_id = num_token_id
        self.eps = eps
        self.use_bf16 = use_bf16
        self.device = device
        self.loss_type = loss_type
        self.num_timesteps = num_timesteps
        self.scheduler = scheduler
        self.cat_scheduler = cat_scheduler
        self.noise_dist = noise_dist
        self.ae_hidden_dim = ae_hidden_dim
        self.dlm_hidden_dim = dlm_hidden_dim
        self.sampler_params = sampler_params
        self.num_numerical_features = num_numerical_features
        self.all_numerical = all_numerical
        self.decoupled_steps = decoupled_steps
        if self.num_numerical_features == 0:
            self.sampler_params['stochastic_sampler'] = False
            self.sampler_params['second_order_correction'] = False

        ### Diffusion Language Model ###
        self.dlm = AutoModel.from_pretrained(
            model_name, trust_remote_code=True,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float16
        )

        # In Llada: MLP module are: ff_proj / up_proj / ff_out (Notice there is a llm head called transformer.ff_out)
        # Attn Module are: q_proj / k_proj / v_proj / attn_out
        lora_targets = []
        for name, mod in self.dlm.named_modules():
            # only inside transformer.blocks.*，avoid including transformer.ff_out（lm head）
            if name.startswith("model.transformer.blocks."):
                last = name.split(".")[-1]
                if last in lora_parameters:
                    lora_targets.append(name)

        peft_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            bias="none", task_type="CAUSAL_LM",
            target_modules=lora_targets
        )
        self.dlm = prepare_model_for_kbit_training(self.dlm)
        self.dlm = get_peft_model(self.dlm, peft_cfg)
        self.dlm.print_trainable_parameters()
        self.tok_embed = self.dlm.get_input_embeddings()

        # --- RMS Align ---
        with torch.no_grad():
            tgt_rms = self.tok_embed.weight.float().pow(2).mean().sqrt().item()
        # --------------------------------

        ##############################

        ### Diffusion Model ###
        self.diffusion_model = diffusion_model(sigma_data=edm_params['sigma_data'],
                                               net_conditioning=edm_params['net_conditioning'],
                                               input_dim=self.ae_hidden_dim,
                                               output_dim=self.dlm_hidden_dim,
                                               scheduler=self.scheduler,
                                               tgt_rms=tgt_rms,
                                               noise_schedule_params=noise_schedule_params,
                                               num_numerical_features=num_numerical_features,
                                               edm_params=edm_params,
                                               floatenc=floatenc,
                                               floatdec=floatdec)
        ##############################

    def dlm_forward_process(self, input_ids):
        b, l = input_ids.shape
        t = torch.rand(b, device=self.device)
        p_mask = (1 - self.eps) * t + self.eps
        p_mask = p_mask[:, None].repeat(1, l)
        masked_indices = torch.rand((b, l), device=self.device) < p_mask
        return masked_indices, p_mask, t

    def forward(self, x):
        input_ids = x["input_ids"].to(self.device)  # (b, L)
        prompt_lens = x["prompt_lengths"].to(self.device)  # (b,)
        answer_lens = x["answer_lengths"].to(self.device)  # (b,)
        x_num = x["num_values"].to(self.device)
        b, L = input_ids.shape

        masked_indices, p_mask, t = self.dlm_forward_process(input_ids)

        # numerical diffusion should under float32
        with torch.cuda.amp.autocast(enabled=False):
            if self.noise_dist == "uniform_t":
                t = t[:, None]
                sigma_num = self.diffusion_model.num_schedule.total_noise(t)

            c_skip, x_num_t, c_out, num_features = self.diffusion_model(x_num, t, sigma_num)
            num_features = num_features.reshape(-1, num_features.size(-1))

        # for numerical features, don't calculate next-token-prediction loss
        num_index = torch.where(input_ids == self.num_token_id)
        if not self.all_numerical:
            n_num = num_index[0].numel()
            if n_num != num_features.shape[0]:
                raise RuntimeError(
                    f"NUMBER tokens ({n_num}) != flattened num_features rows ({num_features.shape[0]}). "
                    "Usually tokenizer truncation dropped <|reserved_token_6|> tokens while num_values is still full — "
                    "raise max_len, shorten prompts, or truncate num_values to match tokens."
                )
        masked_indices[num_index] = False

        noisy = input_ids.clone()
        noisy[masked_indices] = self.mask_token_id

        pos = torch.arange(L, device=self.device).unsqueeze(0).expand(b, L)
        prompt_mask = (pos < prompt_lens.unsqueeze(1))
        noisy[prompt_mask] = input_ids[prompt_mask]
        masked_indices[prompt_mask] = False
        # p_mask[prompt_mask] = 1.0
        # for the eos tokens that exceed the answer_length, it will still engage into generating
        # but we don't cal loss for them
        batch_eos_mask = (pos >= (prompt_lens + answer_lens).unsqueeze(1))
        masked_indices[batch_eos_mask] = False

        inputs_embeds = self.tok_embed(noisy)
        num_features = num_features.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        inputs_embeds[num_index] = num_features

        out = self.dlm(inputs_embeds=inputs_embeds, return_dict=True, output_hidden_states=True)
        logits = out.logits

        # numerical diffusion should under float32
        with torch.cuda.amp.autocast(enabled=False):
            num_last_states = out.hidden_states[-1][num_index]
            num_prediction = self.diffusion_model.back_projection(c_skip, x_num_t, c_out, num_last_states)
            num_diffusion_loss = self.diffusion_model._edm_loss(num_prediction, x_num, sigma_num).mean()

        if self.all_numerical:
            zero_text_loss = (logits[..., :1] * 0.0).sum()
            return zero_text_loss, num_diffusion_loss, 0

        # logits = self.dlm(input_ids=noisy).logits
        midx = masked_indices
        if midx.sum() == 0:
            loss = (logits[..., :1] * 0.0).sum()
            return loss, num_diffusion_loss, 0.0

        if self.loss_type == 'no_divide_pmask':
            ce_per_tok = F.cross_entropy(logits[midx], input_ids[midx], reduction='none')

            # ans_lens = answer_lens.clamp(min=1).float()
            sample_ids = torch.arange(b, device=self.device).unsqueeze(1).expand(b, L)
            masked_sample_ids = sample_ids[midx]
            # denom = ans_lens[masked_sample_ids]
            counts = torch.bincount(masked_sample_ids, minlength=b).clamp_min(1)
            norm = counts[masked_sample_ids].float()
            ce_scaled = ce_per_tok / norm
            loss = ce_scaled.sum() / b

        if self.loss_type == 'dream_loss':
            # loss like Dream
            ce_per_tok = F.cross_entropy(logits[midx], input_ids[midx], reduction='none')
            # token_reweighting = False
            # if token_reweighting:
            #     ce_per_tok = (
            #             config.diffusion.alpha
            #             * (1 - torch.exp(-ce_per_tok)) ** config.diffusion.gamma
            #             * ce_per_tok
            #     )
            # time_reweighting:
            t_eps = 5e-2
            p_mask_safe = p_mask.clamp_min(t_eps)
            ce_per_tok = ce_per_tok / p_mask_safe[midx]

            loss = ce_per_tok.mean()

        return loss, num_diffusion_loss, midx.sum().item()

    def save_model(self, save_dir, description):
        self.dlm.save_pretrained(os.path.join(save_dir, f"{description}"))
        torch.save(self.diffusion_model.state_dict(), os.path.join(save_dir, f"{description}", "diffusion_model.pt"))

    def load_model(self, save_dir, description):
        ckpt_dir = os.path.join(save_dir, f"{description}")
        diff_path = os.path.join(ckpt_dir, "diffusion_model.pt")

        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"LoRA checkpoint directory not found: {ckpt_dir}")
        if not os.path.exists(diff_path):
            raise FileNotFoundError(f"Diffusion checkpoint not found: {diff_path}")

        dtype = torch.bfloat16 if self.use_bf16 else torch.float16
        base_model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
        )

        base_model = prepare_model_for_kbit_training(base_model)
        self.dlm = PeftModel.from_pretrained(
            base_model,
            ckpt_dir,
        )
        self.tok_embed = self.dlm.get_input_embeddings()

        state_dict = torch.load(diff_path, map_location="cpu")
        self.diffusion_model.load_state_dict(state_dict, strict=True)

        print(f"Loaded LoRA+dllm from {ckpt_dir}")
        print(f"Loaded diffusion_model from {diff_path}")
        return self

    @torch.no_grad()
    def sample_synthetic_all(self, num_samples, batch_size, prompt_ids, num_value_idx, gen_length, block_length, steps,
                             temperature, remasking, tokenizer, train_ds, description, save_description, top_k,
                             keep_nan_samples=False,
                             text_steps=None, num_steps=None):
        if self.all_numerical:
            return self._sample_synthetic_all_onlynum(
                num_samples=num_samples,
                batch_size=batch_size,
                prompt_ids=prompt_ids,
                num_value_idx=num_value_idx,
                gen_length=gen_length,
                block_length=block_length,
                steps=steps,
                temperature=temperature,
                remasking=remasking,
                tokenizer=tokenizer,
                train_ds=train_ds,
                description=description,
                save_description=save_description,
                top_k=top_k,
                keep_nan_samples=keep_nan_samples,
            )
        if self.decoupled_steps:
            if text_steps is None or num_steps is None:
                raise ValueError(
                    "decoupled_steps=True requires both text_steps and num_steps."
                )
            return self._sample_synthetic_all_diffstep(
                num_samples=num_samples,
                batch_size=batch_size,
                prompt_ids=prompt_ids,
                num_value_idx=num_value_idx,
                gen_length=gen_length,
                block_length=block_length,
                steps=steps,
                text_steps=text_steps,
                num_steps=num_steps,
                temperature=temperature,
                remasking=remasking,
                tokenizer=tokenizer,
                train_ds=train_ds,
                description=description,
                save_description=save_description,
                top_k=top_k,
                keep_nan_samples=keep_nan_samples,
            )

        b = batch_size
        rep_token = tokenizer.encode("number", add_special_tokens=False)[0]

        ds_name = getattr(train_ds, "dataset_dir", None) or "unknown"

        all_samples = []
        num_generated = 0
        while num_generated < num_samples:
            print(f"Samples left to generate: {num_samples - num_generated}")
            if top_k == 1:
                out, num = self.sample(b, prompt_ids, num_value_idx, gen_length, block_length, steps, temperature,
                                       tokenizer, remasking)
            else:
                out, num = self.sample_topK(b, prompt_ids, num_value_idx, gen_length, block_length, steps, temperature,
                                       tokenizer, remasking, top_k)
            out[:, num_value_idx] = rep_token
            batch_res = tokenizer.batch_decode(out[:, prompt_ids.size(0):], skip_special_tokens=True)
            df_num = pd.DataFrame(num.cpu(), columns=train_ds.numerical_columns)
            df_num = train_ds.denormalize(df_num)
            # df_num = df_num.round().astype(int)
            df_num = df_num.astype(float)
            new_batch_res = []

            for i in range(len(batch_res)):
                s = batch_res[i]
                vals = df_num.iloc[i].tolist()

                for k in range(num_value_idx.size(0)):
                    s = s.replace("number", str(vals[k]), 1)
                new_batch_res.append(s)

            out_path = os.path.join(
                "result", ds_name or "unknown", "raw_sampling_result",
                f"{description}{save_description}.txt",
            )
            dir_name = os.path.dirname(out_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(out_path, "a", encoding="utf-8") as f:
                for res in new_batch_res:
                    f.write(f"{res}\n")
                    # print(res)

            all_samples.extend(new_batch_res)
            num_generated += len(new_batch_res)

        x_gen = all_samples

        return x_gen

    @torch.no_grad()
    def _sample_synthetic_all_onlynum(self, num_samples, batch_size, prompt_ids, num_value_idx,
                                      gen_length, block_length, steps, temperature, remasking,
                                      tokenizer, train_ds, description, save_description, top_k,
                                      keep_nan_samples=False):
        b = batch_size
        ds_name = getattr(train_ds, "dataset_dir", None) or "unknown"

        num_generated = 0
        while num_generated < num_samples:
            print(f"Samples left to generate: {num_samples - num_generated}")
            if top_k == 1:
                _, num = self._sample_onlynum(b, prompt_ids, num_value_idx, gen_length, block_length,
                                              steps, temperature, tokenizer, remasking)
            else:
                _, num = self.sample_topK(b, prompt_ids, num_value_idx, gen_length, block_length,
                                          steps, temperature, tokenizer, remasking, top_k)

            df_num = pd.DataFrame(num.cpu(), columns=train_ds.col_order)
            df_num = train_ds.denormalize(df_num)
            df_num = df_num.astype(float)
            # with pd.option_context(
            #         'display.max_rows', None,
            #         'display.max_columns', None,
            #         'display.width', None,
            #         'display.max_colwidth', None,
            # ):
            #     print(df_num)

            csv_path = os.path.join(
                "result", ds_name or "unknown", "raw_sampling_result",
                f"{description}{save_description}.csv",
            )
            dir_name = os.path.dirname(csv_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            file_exists = os.path.exists(csv_path)
            df_num.to_csv(
                csv_path,
                mode='a' if file_exists else 'w',
                header=not file_exists,
                index=False,
            )
            num_generated += len(df_num)

    @torch.no_grad()
    def _sample_synthetic_all_diffstep(self, num_samples, batch_size, prompt_ids, num_value_idx,
                                       gen_length, block_length, steps, text_steps, num_steps,
                                       temperature, remasking, tokenizer, train_ds,
                                       description, save_description, top_k,
                                       keep_nan_samples=False):
        b = batch_size
        rep_token = tokenizer.encode("number", add_special_tokens=False)[0]

        ds_name = getattr(train_ds, "dataset_dir", None) or "unknown"

        all_samples = []
        num_generated = 0
        while num_generated < num_samples:
            print(f"Samples left to generate: {num_samples - num_generated}")
            if top_k == 1:
                out, num = self._sample_diffstep(
                    batch_size, prompt_ids, num_value_idx, gen_length, block_length,
                    text_steps, num_steps, temperature, tokenizer, remasking,
                )
            else:
                out, num = self.sample_topK(b, prompt_ids, num_value_idx, gen_length, block_length,
                                            steps, temperature, tokenizer, remasking, top_k)
            out[:, num_value_idx] = rep_token
            batch_res = tokenizer.batch_decode(out[:, prompt_ids.size(0):], skip_special_tokens=True)
            df_num = pd.DataFrame(num.cpu(), columns=train_ds.numerical_columns)
            df_num = train_ds.denormalize(df_num)
            df_num = df_num.astype(float)
            new_batch_res = []
            for i in range(len(batch_res)):
                s = batch_res[i]
                vals = df_num.iloc[i].tolist()
                for k in range(num_value_idx.size(0)):
                    s = s.replace("number", str(vals[k]), 1)
                new_batch_res.append(s)

            out_path = os.path.join(
                "result", ds_name or "unknown", "raw_sampling_result",
                f"{description}{save_description}.txt",
            )
            dir_name = os.path.dirname(out_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(out_path, "a", encoding="utf-8") as f:
                for res in new_batch_res:
                    f.write(f"{res}\n")
                    # print(res)

            all_samples.extend(new_batch_res)
            num_generated += len(new_batch_res)

        return all_samples

    @torch.no_grad()
    def sample(self, batch_size, prompt_ids, num_value_idx, gen_length, block_length, steps, temperature, tokenizer,
               remasking='low_confidence', print_inter_res=False):
        B, L = batch_size, prompt_ids.size(0)
        x = torch.full((B, L + gen_length), self.mask_token_id, dtype=torch.long).to(self.device)
        x[:, :L] = prompt_ids.clone()

        prompt_index = (x != self.mask_token_id)

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks

        t, t_hat_list, z_norm, sigma_num_cur, sigma_num_next, sigma_num_hat = self.num_diffusion_init(batch_size)

        pbar_list = []
        for i in range(num_blocks):
            start_step = self.num_timesteps - steps * (i + 1)
            end_step = self.num_timesteps - steps * i
            pbar = tqdm(reversed(range(start_step, end_step)), total=steps)
            pbar.set_description(f"Sampling Progress Block[{i + 1}/{num_blocks}]")
            pbar_list.append(pbar)

        x_num_cur = z_norm
        for num_block in range(num_blocks):
            block_start = L + num_block * block_length
            block_end = L + (num_block + 1) * block_length
            block_mask_index = (x[:, block_start:block_end] == self.mask_token_id)
            num_transfer_tokens = self.get_num_transfer_tokens(block_mask_index, steps)

            for i in pbar_list[num_block]:
                mask_index = (x == self.mask_token_id)

                # Get x_num_hat by move towards the noise by a small step
                x_num_hat = x_num_cur + (
                        sigma_num_hat[i] ** 2 - sigma_num_cur[i] ** 2).sqrt() * S_noise * torch.randn_like(
                    x_num_cur)

                t_cur = t[i]
                t_next = t[i - 1] if i > 0 else None
                t_hat = t_hat_list[i]

                # numerical diffusion should under float32
                with torch.cuda.amp.autocast(enabled=False):
                    c_skip, x_num_t, c_out, num_features = self.diffusion_model(x_num_hat.float(),
                                                                                t_hat.squeeze().repeat(B),
                                                                                sigma_num_hat[i].unsqueeze(0).repeat(B,
                                                                                                                     1),
                                                                                sampling_stage=True)

                inputs_embeds = self.tok_embed(x)
                inputs_embeds[:, num_value_idx] = num_features

                out = self.dlm(inputs_embeds=inputs_embeds, return_dict=True, output_hidden_states=True)
                logits = out.logits

                # numerical diffusion should under float32
                # with torch.cuda.amp.autocast(enabled=False):
                with torch.amp.autocast("cuda", enabled=False):
                    num_last_states = out.hidden_states[-1][:, num_value_idx]
                    num_prediction = self.diffusion_model.back_projection(c_skip, x_num_t, c_out, num_last_states)

                #### text sampling step ####
                logits_with_noise = self.add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

                need_p_x0 = remasking == 'low_confidence'
                if need_p_x0:
                    p = F.softmax(logits, dim=-1)
                    p_x0 = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
                else:
                    p_x0 = None

                if remasking == 'low_confidence':
                    x0_p = p_x0
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, block_end:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -np.inf))

                table_idx = steps - 1 - (i - steps * (num_blocks - 1 - num_block))
                k_unmask_j = num_transfer_tokens[:, table_idx]

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    k = int(k_unmask_j[j].item())
                    if k <= 0:
                        continue
                    _, select_index = torch.topk(confidence[j], k=k)
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]

                if print_inter_res:
                    step_select_index = []
                    step_select_tokens = []
                    step_select_tokens_prob = []
                    for j in range(confidence.shape[0]):
                        k = int(k_unmask_j[j].item())
                        if k <= 0:
                            step_select_index.append(torch.empty(0, dtype=torch.long))
                            step_select_tokens.append(torch.empty(0, dtype=x0.dtype, device=x0.device))
                            step_select_tokens_prob.append(torch.empty(0, dtype=confidence.dtype, device=confidence.device))
                            continue
                        _, select_index = torch.topk(confidence[j], k=k)
                        step_select_index.append(select_index - prompt_ids.size(0))
                        step_select_tokens.append(x0[j, select_index])
                        step_select_tokens_prob.append(confidence[j, select_index])
                    batch_res = tokenizer.batch_decode(x[:, prompt_ids.size(0):], skip_special_tokens=False)
                    print(f'\nStep{self.num_timesteps - i}: select_index: {step_select_index}')
                    print('Generated tokens:', tokenizer.batch_decode(step_select_tokens, skip_special_tokens=False))
                    print('Generated tokens prob:', step_select_tokens_prob)
                    for res in batch_res:
                        print(res.strip("<|endoftext|>"))
                        print()
                #### text sampling stage ####

                #### num sampling step ####
                # Euler step
                d_cur = (x_num_hat - num_prediction) / sigma_num_hat[i]
                x_num_next = x_num_hat + (sigma_num_next[i] - sigma_num_hat[i]) * d_cur

                if self.sampler_params['second_order_correction']:
                    if i > 0:
                        # numerical diffusion should under float32
                        with torch.cuda.amp.autocast(enabled=False):
                            c_skip, x_num_t, c_out, num_features = self.diffusion_model(x_num_next.float(),
                                                                                        t_next.squeeze().repeat(B),
                                                                                        sigma_num_next[i].unsqueeze(
                                                                                            0).repeat(B, 1),
                                                                                        sampling_stage=True)

                        inputs_embeds[:, num_value_idx] = num_features

                        out = self.dlm(inputs_embeds=inputs_embeds, return_dict=True, output_hidden_states=True)

                        # numerical diffusion should under float32
                        with torch.cuda.amp.autocast(enabled=False):
                            num_last_states = out.hidden_states[-1][:, num_value_idx]
                            num_prediction = self.diffusion_model.back_projection(c_skip, x_num_t, c_out,
                                                                                  num_last_states)

                        d_prime = (x_num_next - num_prediction) / sigma_num_next[i]
                        x_num_next = x_num_hat + (sigma_num_next[i] - sigma_num_hat[i]) * (0.5 * d_cur + 0.5 * d_prime)

                x_num_cur = x_num_next
                #### num sampling step ####

        return x, x_num_cur

    @torch.no_grad()
    def sample_topK(self, batch_size, prompt_ids, num_value_idx, gen_length, block_length, steps, temperature,
                    tokenizer,
                    remasking='low_confidence', top_k=10, print_inter_res=False):
        B, L = batch_size, prompt_ids.size(0)
        x = torch.full((B, L + gen_length), self.mask_token_id, dtype=torch.long).to(self.device)
        x[:, :L] = prompt_ids.clone()

        prompt_index = (x != self.mask_token_id)

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks

        t, t_hat_list, z_norm, sigma_num_cur, sigma_num_next, sigma_num_hat = self.num_diffusion_init(batch_size)

        pbar_list = []
        for i in range(num_blocks):
            start_step = self.num_timesteps - steps * (i + 1)
            end_step = self.num_timesteps - steps * i
            pbar = tqdm(reversed(range(start_step, end_step)), total=steps)
            pbar.set_description(f"Sampling Progress Block[{i + 1}/{num_blocks}]")
            pbar_list.append(pbar)

        K_max = top_k
        mode = 'uniform'
        pos_temp = 1
        print(f"K_max={K_max}; mode={mode}; pos_temp={pos_temp}")

        x_num_cur = z_norm
        for num_block in range(num_blocks):
            block_mask_index = (x[:, L + num_block * block_length: L + (
                    num_block + 1) * block_length:] == self.mask_token_id)
            num_transfer_tokens = self.get_num_transfer_tokens(block_mask_index, steps)
            for local_step, i in enumerate(pbar_list[num_block]):
                mask_index = (x == self.mask_token_id)

                # Get x_num_hat by move towards the noise by a small step
                x_num_hat = x_num_cur + (
                        sigma_num_hat[i] ** 2 - sigma_num_cur[i] ** 2).sqrt() * S_noise * torch.randn_like(
                    x_num_cur)

                t_cur = t[i]
                t_next = t[i - 1] if i > 0 else None
                t_hat = t_hat_list[i]

                # numerical diffusion should under float32
                with torch.cuda.amp.autocast(enabled=False):
                    c_skip, x_num_t, c_out, num_features = self.diffusion_model(x_num_hat.float(),
                                                                                t_hat.squeeze().repeat(B),
                                                                                sigma_num_hat[i].unsqueeze(0).repeat(B,
                                                                                                                     1),
                                                                                sampling_stage=True)

                inputs_embeds = self.tok_embed(x)
                inputs_embeds[:, num_value_idx] = num_features

                out = self.dlm(inputs_embeds=inputs_embeds, return_dict=True, output_hidden_states=True)
                logits = out.logits

                # numerical diffusion should under float32
                with torch.cuda.amp.autocast(enabled=False):
                    num_last_states = out.hidden_states[-1][:, num_value_idx]
                    num_prediction = self.diffusion_model.back_projection(c_skip, x_num_t, c_out, num_last_states)

                #### text sampling step ####
                logits_with_noise = self.add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

                if remasking == 'low_confidence':
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l

                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, L + (num_block + 1) * block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

                block_start = L + num_block * block_length
                block_end = L + (num_block + 1) * block_length

                for j in range(confidence.shape[0]):
                    block_mask_j = mask_index[j, block_start:block_end]
                    valid_local = torch.where(block_mask_j)[0]
                    M = valid_local.numel()
                    if M == 0:
                        continue

                    k_step = int(num_transfer_tokens[
                        j, steps - 1 - (i - steps * (num_blocks - 1 - num_block))].item())
                    k_step = min(k_step, M)
                    if k_step <= 0:
                        continue
                    conf_block = confidence[j, block_start:block_end]
                    conf_j = conf_block[valid_local]
                    K = min(K_max, M)
                    top_conf, top_rel_idx = torch.topk(conf_j, k=K)
                    cand_idx = block_start + valid_local[top_rel_idx]

                    if mode == "uniform":
                        perm = torch.randperm(K, device=x.device)
                        chosen_rel = perm[:k_step]
                    elif mode == "conf_weighted":
                        scaled = top_conf / pos_temp
                        probs = torch.softmax(scaled, dim=-1)
                        chosen_rel = torch.multinomial(probs, num_samples=k_step, replacement=False)
                    else:
                        raise ValueError(f"Unknown pos_sampling mode: {mode}")

                    chosen_idx = cand_idx[chosen_rel]
                    transfer_index[j, chosen_idx] = True

                x[transfer_index] = x0[transfer_index]

                if print_inter_res:
                    step_select_index = []
                    step_select_tokens = []
                    step_select_tokens_prob = []
                    for j in range(confidence.shape[0]):
                        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[
                            j, steps - 1 - (i - steps * (num_blocks - 1 - num_block))])
                        step_select_index.append(select_index - prompt_ids.size(0))
                        step_select_tokens.append(x0[j, select_index])
                        step_select_tokens_prob.append(confidence[j, select_index])
                    batch_res = tokenizer.batch_decode(x[:, prompt_ids.size(0):], skip_special_tokens=False)
                    print(f'\nStep{self.num_timesteps - i}: select_index: {step_select_index}')
                    print('Generated tokens:', tokenizer.batch_decode(step_select_tokens, skip_special_tokens=False))
                    print('Generated tokens prob:', step_select_tokens_prob)
                    for res in batch_res:
                        print(res.strip("<|endoftext|>"))
                        print()
                #### text sampling stage ####

                #### num sampling step ####
                # Euler step
                d_cur = (x_num_hat - num_prediction) / sigma_num_hat[i]
                x_num_next = x_num_hat + (sigma_num_next[i] - sigma_num_hat[i]) * d_cur

                if self.sampler_params['second_order_correction']:
                    if i > 0:
                        # numerical diffusion should under float32
                        with torch.cuda.amp.autocast(enabled=False):
                            c_skip, x_num_t, c_out, num_features = self.diffusion_model(x_num_next.float(),
                                                                                        t_next.squeeze().repeat(B),
                                                                                        sigma_num_next[i].unsqueeze(
                                                                                            0).repeat(B, 1),
                                                                                        sampling_stage=True)

                        inputs_embeds[:, num_value_idx] = num_features

                        out = self.dlm(inputs_embeds=inputs_embeds, return_dict=True, output_hidden_states=True)

                        # numerical diffusion should under float32
                        with torch.cuda.amp.autocast(enabled=False):
                            num_last_states = out.hidden_states[-1][:, num_value_idx]
                            num_prediction = self.diffusion_model.back_projection(c_skip, x_num_t, c_out,
                                                                                  num_last_states)

                        d_prime = (x_num_next - num_prediction) / sigma_num_next[i]
                        x_num_next = x_num_hat + (sigma_num_next[i] - sigma_num_hat[i]) * (0.5 * d_cur + 0.5 * d_prime)

                x_num_cur = x_num_next
                #### num sampling step ####

        return x, x_num_cur

    @torch.no_grad()
    def _sample_onlynum(self, batch_size, prompt_ids, num_value_idx, gen_length, block_length,
                        steps, temperature, tokenizer, remasking='low_confidence',
                        print_inter_res=False):
        B, L = batch_size, prompt_ids.size(0)
        x = torch.full((B, L + gen_length), self.mask_token_id, dtype=torch.long).to(self.device)
        x[:, :L] = prompt_ids.clone()

        num_blocks = 1
        steps = steps // num_blocks

        t, t_hat_list, z_norm, sigma_num_cur, sigma_num_next, sigma_num_hat = self.num_diffusion_init(batch_size)

        pbar_list = []
        for i in range(num_blocks):
            start_step = self.num_timesteps - steps * (i + 1)
            end_step = self.num_timesteps - steps * i
            pbar = tqdm(reversed(range(start_step, end_step)), total=steps)
            pbar.set_description(f"Sampling Progress (onlynum) Block[{i + 1}/{num_blocks}]")
            pbar_list.append(pbar)

        x_num_cur = z_norm
        for num_block in range(num_blocks):
            for i in pbar_list[num_block]:
                x_num_hat = x_num_cur + (
                        sigma_num_hat[i] ** 2 - sigma_num_cur[i] ** 2).sqrt() * S_noise * torch.randn_like(
                    x_num_cur)

                t_hat = t_hat_list[i]
                with torch.cuda.amp.autocast(enabled=False):
                    c_skip, x_num_t, c_out, num_features = self.diffusion_model(
                        x_num_hat.float(),
                        t_hat.squeeze().repeat(B),
                        sigma_num_hat[i].unsqueeze(0).repeat(B, 1),
                        sampling_stage=True,
                    )

                inputs_embeds = self.tok_embed(x)
                inputs_embeds[:, num_value_idx] = num_features
                out = self.dlm(inputs_embeds=inputs_embeds, return_dict=True, output_hidden_states=True)

                with torch.cuda.amp.autocast(enabled=False):
                    num_last_states = out.hidden_states[-1][:, num_value_idx]
                    num_prediction = self.diffusion_model.back_projection(c_skip, x_num_t, c_out, num_last_states)

                d_cur = (x_num_hat - num_prediction) / sigma_num_hat[i]
                x_num_next = x_num_hat + (sigma_num_next[i] - sigma_num_hat[i]) * d_cur
                x_num_cur = x_num_next

        return x, x_num_cur

    @staticmethod
    def _split_steps_across_blocks(total_steps, num_blocks):
        base = total_steps // num_blocks
        rem = total_steps % num_blocks
        return [base + (1 if i < rem else 0) for i in range(num_blocks)]

    @torch.no_grad()
    def _sample_diffstep(self, batch_size, prompt_ids, num_value_idx, gen_length, block_length,
                         text_steps, num_steps, temperature, tokenizer,
                         remasking='low_confidence', print_inter_res=False):
        """Block-aware two-phase decoupled sampling.

        Phase 1: text-driven block sampling, numerical paired 1:1 when budget
                 allows. Once the numerical budget is exhausted the remaining
                 text steps run with the numerical channel frozen.
        Phase 2: if ``num_steps > text_steps``, run the remaining numerical
                 Euler steps with the fully decoded text context.
        """
        B, L = batch_size, prompt_ids.size(0)
        device = self.device

        if gen_length % block_length != 0:
            raise ValueError(
                f"gen_length={gen_length} must be divisible by block_length={block_length}."
            )
        num_blocks = gen_length // block_length

        if text_steps < num_blocks:
            raise ValueError(
                f"text_steps={text_steps} must be >= num_blocks={num_blocks} so that "
                f"every block gets at least one text step."
            )

        text_per_block_list = self._split_steps_across_blocks(text_steps, num_blocks)

        x = torch.full((B, L + gen_length), self.mask_token_id, dtype=torch.long, device=device)
        x[:, :L] = prompt_ids.clone()

        t, t_hat_list, z_norm, sigma_num_cur, sigma_num_next, sigma_num_hat = self.num_diffusion_init(batch_size)
        x_num_cur = z_norm

        if num_steps > len(t):
            raise ValueError(
                f"num_steps={num_steps} exceeds available numerical diffusion schedule length={len(t)}."
            )

        num_start_idx = len(t) - num_steps
        num_step_counter = 0

        num_only_tail = max(num_steps - text_steps, 0)
        text_only_tail_per_block_total = max(text_steps - num_steps, 0)
        print(
            f"[diffstep+block 2-phase] gen_length={gen_length}, block_length={block_length}, "
            f"num_blocks={num_blocks}, text_per_block={text_per_block_list}, "
            f"text_steps={text_steps}, num_steps={num_steps}, "
            f"num_only_tail={num_only_tail}, "
            f"text_with_frozen_num={text_only_tail_per_block_total}, "
            f"num_schedule_len={len(t)}"
        )

        # ---------- Phase 1 ----------
        for num_block in range(num_blocks):
            block_start = L + num_block * block_length
            block_end = L + (num_block + 1) * block_length

            text_per_block = text_per_block_list[num_block]
            if text_per_block == 0:
                continue

            block_mask_index = (x[:, block_start:block_end] == self.mask_token_id)
            num_transfer_tokens = self.get_num_transfer_tokens(block_mask_index, text_per_block)

            pbar = tqdm(range(text_per_block), total=text_per_block)
            pbar.set_description(
                f"Phase1 Block[{num_block + 1}/{num_blocks}] text={text_per_block}"
            )

            for text_step_in_block in pbar:
                num_active = num_step_counter < num_steps
                mask_index = (x == self.mask_token_id)

                if num_active:
                    num_i = len(t) - 1 - num_step_counter
                    x_num_hat = x_num_cur + (
                            sigma_num_hat[num_i] ** 2 - sigma_num_cur[num_i] ** 2
                    ).clamp_min(0).sqrt() * S_noise * torch.randn_like(x_num_cur)
                    t_next = t[num_i - 1] if num_i > 0 else None
                    with torch.cuda.amp.autocast(enabled=False):
                        c_skip, x_num_t, c_out, num_features = self.diffusion_model(
                            x_num_hat.float(),
                            t_hat_list[num_i].squeeze().repeat(B),
                            sigma_num_hat[num_i].unsqueeze(0).repeat(B, 1),
                            sampling_stage=True,
                        )
                else:
                    frozen_i = max(len(t) - 1 - num_step_counter, num_start_idx)
                    with torch.cuda.amp.autocast(enabled=False):
                        c_skip, x_num_t, c_out, num_features = self.diffusion_model(
                            x_num_cur.float(),
                            t_hat_list[frozen_i].squeeze().repeat(B),
                            sigma_num_hat[frozen_i].unsqueeze(0).repeat(B, 1),
                            sampling_stage=True,
                        )
                    num_i = None
                    t_next = None
                    x_num_hat = None

                inputs_embeds = self.tok_embed(x)
                inputs_embeds[:, num_value_idx] = num_features
                out = self.dlm(inputs_embeds=inputs_embeds, return_dict=True, output_hidden_states=True)
                logits = out.logits

                logits_with_noise = self.add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if remasking == 'low_confidence':
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, block_end:] = -np.inf
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -np.inf))

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(B):
                    k = int(num_transfer_tokens[j, text_step_in_block].item())
                    if k <= 0:
                        continue
                    valid_pool = torch.isfinite(confidence[j])
                    k_eff = min(k, int(valid_pool.sum().item()))
                    if k_eff <= 0:
                        continue
                    _, select_index = torch.topk(confidence[j], k=k_eff)
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]

                if print_inter_res:
                    batch_res = tokenizer.batch_decode(x[:, L:], skip_special_tokens=False)
                    print(
                        f"\n[Phase1] Block {num_block + 1}/{num_blocks} "
                        f"text_step {text_step_in_block + 1}/{text_per_block} "
                        f"num_active={num_active}"
                    )
                    for res in batch_res:
                        print(res.strip("<|endoftext|>"))
                        print()

                if num_active:
                    with torch.cuda.amp.autocast(enabled=False):
                        num_last_states = out.hidden_states[-1][:, num_value_idx]
                        num_prediction = self.diffusion_model.back_projection(
                            c_skip, x_num_t, c_out, num_last_states
                        )
                    d_cur = (x_num_hat - num_prediction) / sigma_num_hat[num_i]
                    x_num_next = x_num_hat + (sigma_num_next[num_i] - sigma_num_hat[num_i]) * d_cur

                    if self.sampler_params['second_order_correction'] and num_i > 0:
                        with torch.cuda.amp.autocast(enabled=False):
                            c_skip_2, x_num_t_2, c_out_2, num_features_2 = self.diffusion_model(
                                x_num_next.float(),
                                t_next.squeeze().repeat(B),
                                sigma_num_next[num_i].unsqueeze(0).repeat(B, 1),
                                sampling_stage=True,
                            )
                        inputs_embeds_2 = self.tok_embed(x)
                        inputs_embeds_2[:, num_value_idx] = num_features_2
                        out_2 = self.dlm(inputs_embeds=inputs_embeds_2, return_dict=True, output_hidden_states=True)
                        with torch.cuda.amp.autocast(enabled=False):
                            num_last_states_2 = out_2.hidden_states[-1][:, num_value_idx]
                            num_prediction_2 = self.diffusion_model.back_projection(
                                c_skip_2, x_num_t_2, c_out_2, num_last_states_2
                            )
                        d_prime = (x_num_next - num_prediction_2) / sigma_num_next[num_i]
                        x_num_next = x_num_hat + (sigma_num_next[num_i] - sigma_num_hat[num_i]) * (
                                0.5 * d_cur + 0.5 * d_prime
                        )

                    x_num_cur = x_num_next
                    num_step_counter += 1

        # ---------- Phase 2 ----------
        if num_only_tail > 0:
            pbar2 = tqdm(range(num_only_tail), total=num_only_tail)
            pbar2.set_description(
                f"Phase2 num-only refinement (full text context, {num_only_tail} steps)"
            )
            for tail_step in pbar2:
                num_i = len(t) - 1 - num_step_counter
                x_num_hat = x_num_cur + (
                        sigma_num_hat[num_i] ** 2 - sigma_num_cur[num_i] ** 2
                ).clamp_min(0).sqrt() * S_noise * torch.randn_like(x_num_cur)
                t_next = t[num_i - 1] if num_i > 0 else None
                with torch.cuda.amp.autocast(enabled=False):
                    c_skip, x_num_t, c_out, num_features = self.diffusion_model(
                        x_num_hat.float(),
                        t_hat_list[num_i].squeeze().repeat(B),
                        sigma_num_hat[num_i].unsqueeze(0).repeat(B, 1),
                        sampling_stage=True,
                    )

                inputs_embeds = self.tok_embed(x)
                inputs_embeds[:, num_value_idx] = num_features
                out = self.dlm(inputs_embeds=inputs_embeds, return_dict=True, output_hidden_states=True)

                with torch.cuda.amp.autocast(enabled=False):
                    num_last_states = out.hidden_states[-1][:, num_value_idx]
                    num_prediction = self.diffusion_model.back_projection(
                        c_skip, x_num_t, c_out, num_last_states
                    )
                d_cur = (x_num_hat - num_prediction) / sigma_num_hat[num_i]
                x_num_next = x_num_hat + (sigma_num_next[num_i] - sigma_num_hat[num_i]) * d_cur

                if self.sampler_params['second_order_correction'] and num_i > 0:
                    with torch.cuda.amp.autocast(enabled=False):
                        c_skip_2, x_num_t_2, c_out_2, num_features_2 = self.diffusion_model(
                            x_num_next.float(),
                            t_next.squeeze().repeat(B),
                            sigma_num_next[num_i].unsqueeze(0).repeat(B, 1),
                            sampling_stage=True,
                        )
                    inputs_embeds_2 = self.tok_embed(x)
                    inputs_embeds_2[:, num_value_idx] = num_features_2
                    out_2 = self.dlm(inputs_embeds=inputs_embeds_2, return_dict=True, output_hidden_states=True)
                    with torch.cuda.amp.autocast(enabled=False):
                        num_last_states_2 = out_2.hidden_states[-1][:, num_value_idx]
                        num_prediction_2 = self.diffusion_model.back_projection(
                            c_skip_2, x_num_t_2, c_out_2, num_last_states_2
                        )
                    d_prime = (x_num_next - num_prediction_2) / sigma_num_next[num_i]
                    x_num_next = x_num_hat + (sigma_num_next[num_i] - sigma_num_hat[num_i]) * (
                            0.5 * d_cur + 0.5 * d_prime
                    )

                x_num_cur = x_num_next
                num_step_counter += 1

        return x, x_num_cur

    def add_gumbel_noise(self, logits, temperature):
        '''
        The Gumbel max is a method for sampling categorical distributions.
        According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
        Thus, we use float64.
        '''
        if temperature == 0:
            return logits
        logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=torch.float64)
        gumbel = -torch.log(-torch.log(noise))
        return logits / temperature + gumbel
        # gumbel_noise = (- torch.log(noise)) ** temperature
        # return logits.exp() / gumbel_noise

    def get_num_transfer_tokens(self, mask_index, steps):
        '''
        In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
        This function is designed to precompute the number of tokens that need to be transitioned at each step.
        '''
        mask_num = mask_index.sum(dim=1, keepdim=True)

        base = mask_num // steps
        remainder = mask_num % steps

        num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

        for i in range(mask_num.size(0)):
            num_transfer_tokens[i, :remainder[i]] += 1

        return num_transfer_tokens

    def num_diffusion_init(self, num_samples):
        b = num_samples
        device = self.device
        dtype = torch.float32

        # Create the chain of t
        t = torch.linspace(0, 1, self.num_timesteps, dtype=dtype, device=device)  # times = 0.0,...,1.0
        t = t[:, None]

        # Compute the chains of sigma
        sigma_num_cur = self.diffusion_model.num_schedule.total_noise(t)
        sigma_num_next = torch.zeros_like(sigma_num_cur)
        sigma_num_next[1:] = sigma_num_cur[0:-1]

        # Prepare sigma_hat for stochastic sampling mode
        if self.sampler_params['stochastic_sampler']:
            gamma = min(S_churn / self.num_timesteps, np.sqrt(2) - 1) * (S_min <= sigma_num_cur) * (
                    sigma_num_cur <= S_max)
            sigma_num_hat = sigma_num_cur + gamma * sigma_num_cur
            t_hat = self.diffusion_model.num_schedule.inverse_to_t(sigma_num_hat)
            t_hat = torch.min(t_hat, dim=-1, keepdim=True).values  # take the samllest t_hat induced by sigma_num
            zero_gamma = (gamma == 0).any()
            # zero_gamma = (gamma == 0).squeeze(-1)
            t_hat[zero_gamma] = t[zero_gamma]
            out_of_bound = (t_hat > 1).squeeze()
            sigma_num_hat[out_of_bound] = sigma_num_cur[out_of_bound]
            t_hat[out_of_bound] = t[out_of_bound]
        else:
            t_hat = t
            sigma_num_hat = sigma_num_cur

        # Sample priors for the continuous dimensions
        z_norm = torch.randn((b, self.num_numerical_features), device=device) * sigma_num_cur[-1]

        return t, t_hat, z_norm, sigma_num_cur, sigma_num_next, sigma_num_hat
