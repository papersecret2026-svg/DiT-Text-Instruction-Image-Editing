# teasure 1 : 10steps(local wigh gloal ratio 0.05) + global
# teasure 2 : 10steps(local wigh gloal ratio 0.05) + global
import os
import torch
import math
import json

from diffusers import FluxKontextPipeline
from edit_pipeline import FluxKontextPipeline as RegionalFluxPipeline
from transformer_flux import FluxTransformer2DModel
from PIL import Image
import numpy as np
from diffusers.models.embeddings import apply_rotary_emb

class AttnController:

    def __init__(self,
                 red_1d, blue_1d, green_1d,
                 red_2d, blue_2d, green_2d,
                    source_ratio,
                 feature_scaling_region1, feature_scaling_region2,
                 source_ratio_region1,source_ratio_region2,
                 ca_scaling_region1=0,
                 ca_scaling_region2=0,
                 algorithm_num=15,
                 device=None, dtype=None):
        self.device = device
        self.dtype = dtype
        self.source_ratio = source_ratio
        self.algorithm_num = algorithm_num
        self.red_1d = red_1d
        self.blue_1d = blue_1d
        self.green_1d = green_1d

        self.edit_mask_region1_ = self.blue_1d.view(1, 1, -1)  # .to(device=device, dtype=dtype)
        self.edit_mask_region2_ = self.green_1d.view(1, 1, -1)  # .to(device=device, dtype=dtype)

        self.red_2d = red_2d
        self.blue_2d = blue_2d
        self.green_2d = green_2d

        self.attn_num = 0
        self.feature_scaling_region1 = feature_scaling_region1
        self.feature_scaling_region2 = feature_scaling_region2
        self.ca_scaling_region1 = ca_scaling_region1
        self.ca_scaling_region2 = ca_scaling_region2
        self.source_ratio_region1 = source_ratio_region1
        self.source_ratio_region2 =source_ratio_region2

        self.self_control = False
        self.blue_txt_ratio = 1
        self.green_txt_ratio = 1

    def reset(self):
        self.attn_num = 0
        self.self_control = False

class RegionalFluxAttnProcessor2_0 :

    def __init__(self, controller):
        self.regional_mask = None
        self.controller = controller

    def FluxAttnProcessor2_0_global(self,
                                    attn,
                                    hidden_states,
                                    encoder_hidden_states=None,
                                    image_rotary_emb=None,**kwargs) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
        hidden_states = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (hidden_states[:, : encoder_hidden_states.shape[1]],
                                                    hidden_states[:, encoder_hidden_states.shape[1]:],)
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

    def FluxAttnProcessor2_0_local(self,
                                   attn,
                                   hidden_states,
                                   encoder_hidden_states=None,
                                   image_rotary_emb=None,
                                   **kwargs):
        batch_size, _, _ = hidden_states.shape
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None: query = attn.norm_q(query)
        if attn.norm_k is not None: key = attn.norm_k(key)
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_added_q is not None: encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None: encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        def scaled_dot_product_attention(query, key, value) -> torch.Tensor:
            scale_factor = 1 / math.sqrt(query.size(-1))
            total_length = query.size(-2)
            total_text_length = total_length - 4096 * 2
            num_region = int(total_text_length // 512)
            attn_outs = []
            for region_idx in range(num_region):
                #################################
                # Step 1. Key & Value Scaling ###
                #################################
                k_mask = torch.zeros_like(key)
                k_mask[:,:,region_idx*512:(region_idx+1)*512,:] = 1
                k_mask[:, :, -4096*2:-4096, :] = 1 # target
                if region_idx == 0 : k_mask[:, :, -4096:, :] = self.controller.feature_scaling_region1
                else : k_mask[:, :, -4096:, :] = self.controller.feature_scaling_region2
                k_zero = key * k_mask.to('cuda')
                attn_region = query @ k_zero.transpose(-2,-1) * scale_factor
                #################################
                # Step 2. CA Scaling ###
                #################################
                txt_len1, txt_len2 = 512, 512
                img = slice(512 + txt_len2, txt_len1 + txt_len2 + 4096)
                source_token_idx = kwargs["source_token_idx"]
                target_token_idx = kwargs["target_token_idx"]
                region1_src_token_idx_list, region2_src_token_idx_list = source_token_idx[0], source_token_idx[-1]
                region2_src_token_idx_list = [i + txt_len1 for i in region2_src_token_idx_list]
                region1_trg_token_idx_list, region2_trg_token_idx_list = target_token_idx[0], target_token_idx[-1]
                region2_trg_token_idx_list = [i + txt_len1 for i in region2_trg_token_idx_list]
                ca_scaling_on_region1 = (self.controller.ca_scaling_region1 * self.controller.edit_mask_region1_).to('cuda')
                ca_scaling_on_region2 = (self.controller.ca_scaling_region2 * self.controller.edit_mask_region2_).to('cuda')
                ca_unscaling_on_region1 = (self.controller.ca_scaling_region1 * self.controller.edit_mask_region2_).to('cuda')
                ca_unscaling_on_region2 = (self.controller.ca_scaling_region2 * self.controller.edit_mask_region1_).to('cuda')
                if region_idx == 0:
                    for src_idx in region1_src_token_idx_list:
                        org_map = attn_region[...,img, src_idx]
                        attn_region[...,img, src_idx] = org_map - ca_scaling_on_region1
                    for trg_idx in region1_trg_token_idx_list:
                        org_map = attn_region[..., img, trg_idx]
                        attn_region[..., img, trg_idx] = org_map + ca_scaling_on_region1 - ca_unscaling_on_region2
                    for trg_idx in region2_trg_token_idx_list:
                        org_map = attn_region[..., img, trg_idx]
                        attn_region[..., img, trg_idx] = org_map - ca_scaling_on_region1

                if region_idx == 1:
                    for src_idx in region2_src_token_idx_list:
                        org_map = attn_region[..., img, src_idx]
                        attn_region[..., img, src_idx] = org_map - ca_scaling_on_region2
                    for trg_idx in region2_trg_token_idx_list:
                        org_map = attn_region[..., img, trg_idx]
                        attn_region[..., img, trg_idx] = org_map - ca_unscaling_on_region1 + ca_scaling_on_region2
                    for trg_idx in region1_trg_token_idx_list:
                        org_map = attn_region[..., img, trg_idx]
                        attn_region[..., img, trg_idx] = org_map - ca_scaling_on_region2
                #################################
                # Step 3. Identity Preservation #
                #################################
                attn_region_trg_trg = attn_region[...,total_text_length:total_text_length+4096,total_text_length:total_text_length+4096]
                attn_region_src_trg = attn_region[..., total_text_length + 4096:, total_text_length:total_text_length + 4096]
                self.controller.blue_2d = self.controller.blue_2d.to('cuda') * 1
                self.controller.red_2d = self.controller.red_2d.to('cuda') * 1
                self.controller.green_2d = self.controller.green_2d.to('cuda') * 1
                if region_idx == 0 :
                    attn_txt = attn_region_trg_trg * self.controller.blue_2d
                    attn_img = attn_region_src_trg * self.controller.blue_2d
                    attn_blue_portion = attn_txt * (1-self.controller.source_ratio_region1) + attn_img * (self.controller.source_ratio_region1)
                    attn_non_blue_portion = attn_region_src_trg * (1-self.controller.blue_2d)
                    attn_region[..., total_text_length:total_text_length+4096, total_text_length:total_text_length+4096] = attn_blue_portion + attn_non_blue_portion
                if region_idx == 1 :
                    attn_txt = attn_region_trg_trg * self.controller.green_2d
                    attn_img = attn_region_src_trg * self.controller.green_2d
                    attn_green_portion = attn_txt * (1 - self.controller.source_ratio_region2) + attn_img * (self.controller.source_ratio_region2)
                    attn_non_green_portion = attn_region_src_trg * (1 - self.controller.green_2d)
                    attn_region[..., total_text_length:total_text_length + 4096, total_text_length:total_text_length + 4096] = attn_green_portion + attn_non_green_portion
                v_mask = k_mask
                v_zero = value * v_mask.to('cuda')
                attn_region = attn_region.softmax(dim=-1)
                attn_out = attn_region @ v_zero # b,h,512+512+4096+4096,512+512+4096+4096

                #####################
                # Step 4. ATTN Mask #
                #####################
                if region_idx == 0:
                    blue_1d = self.controller.blue_1d.to("cuda").float()  # (4096,)
                    attn_mask = torch.zeros(total_text_length + 4096 + 4096, device="cuda", dtype=torch.float32)
                    attn_mask[:512] = 1.0
                    s = total_text_length
                    attn_mask[s: s + 4096] = blue_1d  # blue_1d가 0/1 (또는 0~1) 값이면 그대로 반영됨
                else :
                    green_1d = self.controller.green_1d.to("cuda").float()  # (4096,)
                    attn_mask = torch.zeros(total_text_length + 4096 + 4096, device="cuda", dtype=torch.float32)
                    attn_mask[512:1024] = 1.0
                    s = total_text_length
                    attn_mask[s: s + 4096] = green_1d  # blue_1d가 0/1 (또는 0~1) 값이면 그대로 반영됨
                mask = attn_mask.view(1, 1, -1, 1)  # (1,1,L,1)
                attn_out = attn_out * mask  # broadcast -> (B,H,L,D)
                attn_outs.append(attn_out)
            # *** Base *** #
            attn_region = query @ key.transpose(-2, -1) * scale_factor
            attn_region_trg_trg = attn_region[...,1024:1024+4096,1024:1024+4096]
            attn_region_src_trg = attn_region[...,1024+4096:,1024:1024+4096]
            attn_region[..., 1024:1024 + 4096, 1024:1024 + 4096] = attn_region_trg_trg*(1-self.controller.source_ratio) + attn_region_src_trg*(self.controller.source_ratio)
            attn_region = attn_region.softmax(dim=-1)
            attn_out = attn_region @ value
            red_1d = self.controller.red_1d.to("cuda").float()  # (4096,)
            ##############
            attn_mask = torch.zeros(total_text_length + 4096 + 4096, device="cuda", dtype=torch.float32)
            attn_mask[-4096:] = 1.0
            s = total_text_length
            attn_mask[s: s + 4096] = red_1d  # blue_1d가 0/1 (또는 0~1) 값이면 그대로 반영됨
            attn_mask = attn_mask.view(1, 1, -1, 1)  # (1,1,L,1)
            ##############
            attn_out = attn_out * attn_mask.to('cuda')
            attn_outs.append(attn_out)
            final_out = sum(attn_outs)
            return final_out

        hidden_states = scaled_dot_product_attention(query, key, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (hidden_states[:, : encoder_hidden_states.shape[1]], hidden_states[:, encoder_hidden_states.shape[1]:],)
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            image_rotary_emb=None,
            **kwargs,
    ) -> torch.FloatTensor:
        attn_num = self.controller.attn_num
        if attn_num >= self.controller.algorithm_num * 57 :
            attn_out = self.FluxAttnProcessor2_0_global(attn=attn,
                                                        hidden_states=hidden_states,
                                                        encoder_hidden_states=encoder_hidden_states,
                                                        image_rotary_emb=image_rotary_emb,
                                                        **kwargs)
        else :
            attn_out = self.FluxAttnProcessor2_0_local(attn=attn,
                                                       hidden_states=hidden_states,
                                                       encoder_hidden_states=encoder_hidden_states,
                                                       image_rotary_emb=image_rotary_emb,
                                                       **kwargs)
        self.controller.attn_num += 1
        return attn_out

def main(args) :
    print(f' Step 1. Call Model')
    model_path = "black-forest-labs/FLUX.1-Kontext-dev"
    if args.origin :
        pipeline = FluxKontextPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")
        #del pipeline
    else :
        transformer = FluxTransformer2DModel.from_pretrained(model_path, subfolder = 'transformer',torch_dtype=torch.bfloat16)
        pipeline = RegionalFluxPipeline.from_pretrained(model_path, transformer = transformer,
                                                        torch_dtype=torch.bfloat16).to("cuda")

    with open(args.json_file, "r", encoding="utf-8") as f:
        records = json.load(f)
    for rec_idx, rec in enumerate(records):

        image_width = 1024
        image_height = 1024
        seed = 124
        mask_path  = rec['mask_path']
        source_image = Image.open(rec['img_path']).convert("RGB").resize((1024, 1024), Image.NEAREST)
        save_base_folder = args.save_folder  # "./result_multi/bell_pepper"
        os.makedirs(save_base_folder, exist_ok=True)
        mask_img = Image.open(mask_path).convert("RGB")
        mask_img = mask_img.resize((image_width, image_height), Image.NEAREST)
        mask_np = np.array(mask_img)
        def get_bbox_from_color(mask_array, target_color, tolerance=10):
            r, g, b = target_color
            color_mask = ((np.abs(mask_array[:, :, 0] - r) <= tolerance) &
                          (np.abs(mask_array[:, :, 1] - g) <= tolerance) &
                          (np.abs(mask_array[:, :, 2] - b) <= tolerance))
            coords = np.column_stack(np.where(color_mask))
            if coords.size == 0:
                return None
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            return [int(x_min), int(y_min), int(x_max), int(y_max)]

        blue_bbox = get_bbox_from_color(mask_np, target_color=(0, 0, 255))
        green_bbox = get_bbox_from_color(mask_np, target_color=(0, 255, 0))
        regional_prompt_mask_pairs = {}
        if blue_bbox:
            regional_prompt_mask_pairs["0"] = {"description": rec['edit_prompt1'],"mask": blue_bbox}
        if green_bbox:
            regional_prompt_mask_pairs["1"] = {"description": rec['edit_prompt2'], "mask": green_bbox}


        mask_np_64 = np.array(Image.fromarray(mask_np).resize((64, 64), Image.NEAREST))
        tol = 10
        red_position = ((np.abs(mask_np_64[..., 0] - 255) <= tol) & (np.abs(mask_np_64[..., 1] - 0) <= tol) & (np.abs(mask_np_64[..., 2] - 0) <= tol)) # [64,64]
        blue_position = ((np.abs(mask_np_64[..., 0] - 0) <= tol) & (np.abs(mask_np_64[..., 1] - 0) <= tol) & (np.abs(mask_np_64[..., 2] - 255) <= tol)) # [64,64]
        green_position = ((np.abs(mask_np_64[..., 0] - 0) <= tol) & (np.abs(mask_np_64[..., 1] - 255) <= tol) & (np.abs(mask_np_64[..., 2] - 0) <= tol)) # [64,64]
        red_1d = torch.tensor(red_position.flatten())
        red_2d = red_1d.unsqueeze(-1).repeat(1, 4096).unsqueeze(0).unsqueeze(0)
        blue_1d = torch.tensor(blue_position.flatten())
        blue_2d = blue_1d.unsqueeze(-1).repeat(1, 4096).unsqueeze(0).unsqueeze(0)
        green_1d = torch.tensor(green_position.flatten())
        green_2d = green_1d.unsqueeze(-1).repeat(1, 4096).unsqueeze(0).unsqueeze(0)


        print(f' 2.5 base setting')
        image_width = (image_width // pipeline.vae_scale_factor) * pipeline.vae_scale_factor
        image_height = (image_height // pipeline.vae_scale_factor) * pipeline.vae_scale_factor
        regional_prompts = []
        regional_masks = []
        background_mask = torch.ones((image_height, image_width))
        edit_prompt = rec['edit_prompt']
        img_pure_name = os.path.splitext(os.path.split(rec['img_path'])[-1])[0]
        detailed_task = rec.get('detailed_task', None)
        source_ratios = rec['source_ratios']
        safe_out = edit_prompt.replace(" ", "_").replace(",", "_").replace("/", "_")[:120]
        sample_folder = os.path.join(args.save_folder,
                                     f'{detailed_task}_{img_pure_name}_{safe_out}')
        os.makedirs(sample_folder, exist_ok=True)
        source_token_list = rec["source_token"]
        target_token_list = rec["target_token"]
        source_save_dir = os.path.join(sample_folder, 'source.png')
        if not os.path.exists(source_save_dir):
            source_image.save(source_save_dir)

        if args.origin :
            out = pipeline(image = source_image,
                           prompt = rec['edit_prompt']).images[0]
            out.save(os.path.join(sample_folder,'joint_attn.png'))

        else :
            for region_idx, region in regional_prompt_mask_pairs.items():
                description = region['description']
                mask = region['mask']
                x1, y1, x2, y2 = mask
                # [1] mask position (only target position with 1)
                mask = torch.zeros((image_height, image_width))
                mask[y1:y2, x1:x2] = 1.0
                # [2] background is -1 and 0
                background_mask -= mask
                # [3] gathering
                regional_prompts.append(description)
                regional_masks.append(mask)








            for feature_scaling_region1 in rec['feature_scaling_region1s']:
                for feature_scaling_region2 in rec['feature_scaling_region2s'] :
                    for source_ratio_region1 in rec["source_ratio_region1s"] :
                        for source_ratio_region2 in rec["source_ratio_region2s"]:
                            for ca_scaling1 in rec['ca_scaling1s']:
                                for ca_scaling2 in rec['ca_scaling2s']:
                                    for algorithm_num in rec['algorithm_nums'] :
                                        for source_ratio in source_ratios:



                                            controller = AttnController(red_1d=red_1d,
                                                                        blue_1d=blue_1d,
                                                                        green_1d=green_1d,
                                                                        red_2d=red_2d,
                                                                        blue_2d=blue_2d,
                                                                        green_2d=green_2d,
                                                                        algorithm_num=algorithm_num,
                                                                        source_ratio = source_ratio,
                                                                        feature_scaling_region1=feature_scaling_region1,
                                                                        feature_scaling_region2=feature_scaling_region2,
                                                                        ca_scaling_region1=ca_scaling1,
                                                                        ca_scaling_region2=ca_scaling2,

                                                                        source_ratio_region1=source_ratio_region1,
                                                                        source_ratio_region2=source_ratio_region2,
                                                                        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                                                                        dtype=torch.float32)
                                            attn_procs = {}
                                            for name in pipeline.transformer.attn_processors.keys():
                                                if 'transformer_blocks' in name and name.endswith("attn.processor"):
                                                    attn_procs[name] = RegionalFluxAttnProcessor2_0(controller)
                                                else:
                                                    attn_procs[name] = pipeline.transformer.attn_processors[name]
                                            pipeline.transformer.set_attn_processor(attn_procs)
                                            save_path = os.path.join(sample_folder,
                                                                     f's_{source_ratio}_r1_f_{feature_scaling_region1}_c_{ca_scaling1}_r_{source_ratio_region1}_'
                                                                     f'r2_f_{feature_scaling_region2}_c_{ca_scaling2}_r_{source_ratio_region2}_for_{algorithm_num}.png')
                                            if not os.path.exists(save_path):
                                                joint_attention_kwargs = {}
                                                print(f' Making joint attention kwargs')
                                                if blue_bbox:
                                                    edit_prompt1 = rec['edit_prompt1']
                                                    joint_attention_kwargs["region_1"] = {"description": edit_prompt1,
                                                                                          "mask": blue_bbox,
                                                                                          "source_token": rec['source_token1'],
                                                                                          "target_token": rec['target_token1'],
                                                                                          'task1': rec['task1'], }
                                                if green_bbox:
                                                    joint_attention_kwargs["region_2"] = {"description": rec['edit_prompt2'],
                                                                                   "mask": green_bbox,
                                                                                   "source_token": rec['source_token2'],
                                                                                   "target_token": rec['target_token2'],
                                                                                   'task2': rec['task2'], }

                                                images = pipeline(image = source_image,
                                                                  prompt=rec['edit_prompt'],
                                                                  width=image_width,
                                                                  height=image_height,
                                                                  algorithm_num=algorithm_num,
                                                                  num_inference_steps=20,
                                                                  source_token_list=source_token_list,
                                                                  target_token_list=target_token_list,
                                                                  #intermediate_save=True,
                                                                  #save_folder = sample_folder,
                                                                  generator=torch.Generator("cuda").manual_seed(seed),
                                                                  joint_attention_kwargs=joint_attention_kwargs,).images
                                                for idx, image in enumerate(images):
                                                    print(f' save on {save_path}')
                                                    image.save(save_path)
                                                    controller.reset()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_path",type=str,default="./result_multi/bell_pepper/bell_pepper_mask.png",
                        help="Path to the mask image (e.g., ./result_multi/bell_pepper/bell_pepper_mask.png)",)
    parser.add_argument("--origin", action = "store_true",)
    parser.add_argument("--save_folder", type=str, default="./result_algorithm")
    parser.add_argument("--json_file", type=str, default="./data/add.json")
    args = parser.parse_args()
    main(args)
