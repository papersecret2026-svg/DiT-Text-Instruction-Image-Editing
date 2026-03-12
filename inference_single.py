import os
import json
from argparse import ArgumentParser
import numpy as np
import torch
from PIL import Image
from edit_pipeline import FluxKontextPipeline as EditPipeline
from diffusers.models.embeddings import apply_rotary_emb
from utils import ensure_dir
from utils import regional_mask
from utils.model_control import setup_kontext_pipe
from utils.controller import AttentionController, feature_scaling, ca_scaling, interpolate_attn
from diffusers import FluxKontextPipeline as FluxKontextPipeline_org
import math
import time
import csv

# target_attns = ['ATTN5','ATTN6']
# for target_attn in target_attns:
#    if target_attn == 'ATTN5':
#        # [ATTN5]
#        important_layers_list = [[13,31,29,18,22,30,33,35,34,12], [13,32,31], [13,32,31,29,18], [13,32,31,29,18,36,26,28,22]]
#    else :
#        # [ATTN6]
#        important_layers_list =[[18,13,32], [18,13,32,12,31], [18,13,32,12,31,29,17,22,27]]
#    for important_layers in important_layers_list :
"""
        source_maps, target_maps = [], []
        for s_token_idx in source_token_idx:
            s_map = attn_weight[:, :, 512:512 + 4096, s_token_idx].unsqueeze(-1)
            s_score = s_map.sum()
            source_maps.append(s_map)
            if controller.attn_num > 57 * 26:
                s_heatmap_save_dir = os.path.join(controller.save_root,
                                                  f'{controller.unique_id}_token_heatmap',
                                                  f'{controller.mode}_S_{controller.attn_num}_{s_token_idx}.png')
                _save_token_heatmap(source_maps, s_heatmap_save_dir)
        for t_token_idx in target_token_idx:
            target_map = attn_weight[:, :, 512:512 + 4096, t_token_idx].unsqueeze(-1)
            t_score = target_map.sum()
            target_maps.append(target_map)
            if controller.attn_num > 57 * 26:
                t_heatmap_save_dir = os.path.join(controller.save_root,
                                                  f'{controller.unique_id}_token_heatmap',
                                                  f'{controller.mode}_T_{controller.attn_num}_{t_token_idx}.png')
                _save_token_heatmap(target_maps, t_heatmap_save_dir)
        if controller.attn_num > 57 * 26:
            attn5 = attn_weight[:, :, 512:512 + 4096, 512:512 + 4096]
            _save_heatmap64(attn5, os.path.join(controller.save_root, controller.unique_id,
                                                f'attn5_{controller.attn_num}.png'))
            attn6 = attn_weight[:, :, 512:512 + 4096, 512 + 4096:]
            _save_heatmap64(attn6, os.path.join(controller.save_root, controller.unique_id,
                                                f'attn6_{controller.attn_num}.png'))
            attn8 = attn_weight[:, :, 512 + 4096:, 512:512 + 4096]
            _save_heatmap64(attn8, os.path.join(controller.save_root, controller.unique_id,
                                                f'attn8_{controller.attn_num}.png'))
            attn9 = attn_weight[:, :, 512 + 4096:, 512 + 4096:]
            _save_heatmap64(attn9, os.path.join(controller.save_root, controller.unique_id,
                                                f'attn9_{controller.attn_num}.png'))

            record = [controller.attn_num, s_score, t_score]
            csv_dir = os.path.join(controller.save_root, f'{controller.unique_id}_token_heatmap_score.csv', )
            controller.record_score(record, csv_dir)
        """
txt_len = 512
tgt_len = 4096
src_len = 4096

def wrap_flux_attention_forward(ATTN_module, module_name: str, controller: AttentionController):

    def forward(hidden_states=None,encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None, **kwargs):

        def preparing_attn(hidden_states=None,encoder_hidden_states=None,image_rotary_emb=None,**kwargs):

            source_token_idx = kwargs.get("source_token_idx", None)
            target_token_idx = kwargs.get("target_token_idx", None)
            B = hidden_states.shape[0]
            to_q = ATTN_module.to_q
            to_k = ATTN_module.to_k
            to_v = ATTN_module.to_v
            heads = ATTN_module.heads
            norm_q = ATTN_module.norm_q
            norm_k = ATTN_module.norm_k
            q = to_q(hidden_states)
            k = to_k(hidden_states)
            v = to_v(hidden_states)
            inner_dim = k.shape[-1]
            head_dim = inner_dim // heads
            q = q.view(B, -1, heads, head_dim).transpose(1, 2)
            k = k.view(B, -1, heads, head_dim).transpose(1, 2)
            v = v.view(B, -1, heads, head_dim).transpose(1, 2)
            if norm_q is not None:
                q = norm_q(q)
            if norm_k is not None:
                k = norm_k(k)
            if encoder_hidden_states is not None:
                add_q = ATTN_module.add_q_proj
                add_k = ATTN_module.add_k_proj
                add_v = ATTN_module.add_v_proj
                norm_aq = ATTN_module.norm_added_q
                norm_ak = ATTN_module.norm_added_k

                enc_q = add_q(encoder_hidden_states).view(B, -1, heads, head_dim).transpose(1, 2)
                enc_k = add_k(encoder_hidden_states).view(B, -1, heads, head_dim).transpose(1, 2)
                enc_v = add_v(encoder_hidden_states).view(B, -1, heads, head_dim).transpose(1, 2)

                if norm_aq is not None:
                    enc_q = norm_aq(enc_q)
                if norm_ak is not None:
                    enc_k = norm_ak(enc_k)
                q = torch.cat((enc_q, q), dim=2)
                k = torch.cat((enc_k, k), dim=2)
                v = torch.cat((enc_v, v), dim=2)
            if image_rotary_emb is not None:
                q = apply_rotary_emb(q, image_rotary_emb)
                k = apply_rotary_emb(k, image_rotary_emb)
            return q, k, v, source_token_idx, target_token_idx, head_dim
        def post_attn(attn_prob, v) :
            out = (attn_prob.to(torch.bfloat16) @ v.to(torch.bfloat16))  # (B,H,L,hd)
            controller.attn_num += 1
            out = out.transpose(1, 2).reshape(1, -1, ATTN_module.heads * head_dim).to(q.dtype)
            if encoder_hidden_states is not None:
                enc_len2 = encoder_hidden_states.shape[1]
                encoder_part, hidden_part = out[:, :enc_len2], out[:, enc_len2:]
                hidden_part = ATTN_module.to_out[0](hidden_part)
                hidden_part = ATTN_module.to_out[1](hidden_part)
                encoder_part = ATTN_module.to_add_out(encoder_part)
                return hidden_part, encoder_part
            else :
                return out

        q,k,v,source_token_idx,target_token_idx,head_dim = preparing_attn(hidden_states,encoder_hidden_states, image_rotary_emb,**kwargs)
        # 2.1. Key Scaling
        scale_factor = 1 / math.sqrt(q.size(-1))
        if controller.attn_num < controller.step1_algorithm_num * 57 :
            k,v = feature_scaling(k,v, controller)
        attn_weight = q @ k.transpose(-2, -1) * scale_factor
        # 2.2. Cross Attention Manipulate
        if 'wo' not in controller.search_mode :
            if controller.attn_num < controller.step1_algorithm_num * 57:
                attn_weight = ca_scaling(controller, attn_weight, source_token_idx, target_token_idx)

        # 2.3. Source Preserve
        if controller.attn_num < controller.step2_algorithm_num  * 57 :
            attn_weight = interpolate_attn(attn_weight, controller, txt_len, tgt_len, src_len)
        attn_prob = torch.softmax(attn_weight, dim=-1)
        return post_attn(attn_prob, v)

    return forward

def install_attention(pipe, controller: AttentionController):
    transformer = pipe.transformer
    for module_name, module in transformer.named_modules():
        if module.__class__.__name__ in ["FluxAttention", "Attention"]:
            module.forward = wrap_flux_attention_forward(module, module_name, controller)
    return controller

def main(args):

    if args.origin :
        pipe = FluxKontextPipeline_org.from_pretrained(args.repo_id, torch_dtype=torch.bfloat16).to("cuda")
    else :
        pipe = EditPipeline.from_pretrained(args.repo_id, torch_dtype=torch.bfloat16).to("cuda")
        setup_kontext_pipe(pipe)

    ensure_dir(args.save_folder)
    with open(args.json_file, "r", encoding="utf-8") as f:
        records = json.load(f)
    for rec_idx, rec in enumerate(records):

        img_path  = rec["img_path"]
        mask_path = rec["mask_path"]
        edit_prompt = rec["edit_prompt"]
        task = rec['task'] # structure change or
        assert task in ("structure_change", "structure_preserve"), (
            f"Invalid task: {task}. Must be one of: structure_change, structure_preserve")
        detailed_task = rec.get('detailed_task', None)

        # Step 1 Factor
        ca_scalings = rec['ca_scalings']
        feature_scalings = rec["feature_scalings"]
        source_token_list = rec["source_token"]
        target_token_list = rec["target_token"]
        name_t = ''
        for t in target_token_list:
            name_t += f'{t}_'
        step1_algorithm_nums = rec["step1_algorithm_nums"]
        step2_search_nums = rec["step2_search_nums"]
        step2_algorithm_nums = rec["step2_algorithm_nums"]

        # Preparing
        img_pure_name = os.path.splitext(os.path.split(img_path)[-1])[0]
        img = Image.open(img_path).convert("RGB").resize((args.width, args.height))
        safe_out = edit_prompt.replace(" ", "_").replace(",", "_").replace("/", "_")[:120]
        save_root = args.save_folder
        os.makedirs(save_root, exist_ok=True)
        if args.origin :
            sample_folder = os.path.join(save_root,f'{detailed_task}_{img_pure_name}_{safe_out}')
            os.makedirs(sample_folder, exist_ok=True)

            gen = torch.Generator("cuda").manual_seed(args.seed)
            with torch.inference_mode():

                guidance = 'non'
                unique_id = f"joint_attention_cfg_{guidance}"
                save_path = os.path.join(sample_folder, f"{unique_id}.png")
                out = pipe(prompt=edit_prompt,
                           image=img,
                           height=1024,
                           width=1024,
                           num_inference_steps=28,
                           generator=gen, ).images[0]
                out.save(save_path)
        else :
            red_mask_1d, blue_mask_1d, green_mask_1d, red2d_64_64, blue2d_64_64, green2d_64_64 = regional_mask.get_rgb_color_masks_1d(mask_path,grid_h=64, grid_w=64,device="cuda",tol=20)
            red_mask = red_mask_1d.float()
            blue_mask = blue_mask_1d.float()
            green_mask = green_mask_1d.float()
            sample_folder = os.path.join(save_root, f'{detailed_task}_{img_pure_name}_{safe_out}')
            os.makedirs(sample_folder, exist_ok=True)
            modes = ['wo_ca_scaling','ours', 'wo_interpolation']

            for mode in modes :
                if mode == 'wo_ca_scaling' :
                    ca_scalings = [0]
                    feature_scalings = [0.98]
                    step2_algorithm_nums = [5]
                if mode ==  'wo_feature_scaling' :
                    feature_scalings = [1]
                    ca_scalings = [0]
                    step2_algorithm_nums = [5]
                if mode == 'wo_interpolation' :
                    ca_scalings = [2]
                    feature_scalings = [0.98]
                    step2_algorithm_nums = [0]
                if mode == 'ours' :
                    ca_scalings = [2]
                    feature_scalings = [0.98]
                    step2_algorithm_nums = [5]
                for step1_algorithm_num in rec['step1_algorithm_nums'] :
                    for feature_scaling in feature_scalings:
                        for ca_scaling in ca_scalings:
                            for step2_algorithm_num in step2_algorithm_nums:
                                for step2_search_num in step2_search_nums:
                                        unique_id = (f"{mode}_step1_{step1_algorithm_num}_f_{feature_scaling}_c_{ca_scaling}"
                                                     f"_step2_{step2_search_num}_al_{step2_algorithm_num}")
                                        controller = AttentionController(save_root=sample_folder,
                                                                         unique_id=unique_id,
                                                                         task=task,
                                                                         step2_search_num=step2_search_num,
                                                                         step1_algorithm_num=step1_algorithm_num,
                                                                         step2_algorithm_num=step2_algorithm_num,
                                                                         print_interpolate_algorithm=args.print_interpolate_algorithm,feature_scaling=feature_scaling,ca_scaling=ca_scaling,detailed_task=detailed_task, )
                                        controller.register_mask(red_mask=red_mask,green_mask=green_mask,blue_mask=blue_mask,red2d_64_64=red2d_64_64,blue2d_64_64=blue2d_64_64,green2d_64_64=green2d_64_64, )
                                        install_attention(pipe, controller)
                                        save_path = os.path.join(sample_folder, f"{unique_id}.png")
                                        gen = torch.Generator("cuda").manual_seed(args.seed)
                                        source_save_dir = os.path.join(sample_folder, f'org.png')
                                        if not os.path.exists(source_save_dir):
                                            img.save(source_save_dir)
                                        if not os.path.exists(save_path):
                                            with torch.inference_mode():
                                                start_time = time.time()
                                                out = pipe(prompt=edit_prompt,image=img,height=1024,width=1024,source_token_list=source_token_list,target_token_list=target_token_list,save_intermediate=False,save_folder=sample_folder,num_inference_steps=28,generator=gen, ).images[0]


                                            out.save(save_path)
                                            controller.reset()

    print("Done.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="black-forest-labs/FLUX.1-Kontext-dev")
    parser.add_argument("--save_folder", type=str, default="./result_algorithm")
    parser.add_argument("--json_file", type=str, default="./data/add.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--debug_text_len", type=int, default=512)
    parser.add_argument("--debug_pix_num", type=int, default=4096)
    parser.add_argument("--debug_src_num", type=int, default=4096)
    parser.add_argument("--origin", action="store_true")
    parser.add_argument("--experiment", action="store_true")
    parser.add_argument("--print_interpolate_algorithm", action="store_true")
    args = parser.parse_args()
    main(args)
