import torch
from utils.attention_utils import split_blocks
import csv
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class AttentionController:
    def __init__(self,
                 save_root: str,
                 unique_id : str,
                 task = 'structure_change',
                 detailed_task = None,
                 detailed_task1=None,
                 detailed_task2=None,
                 attn_check_time=0,
                 injectiontask = 'Full',
                 injectionratio=1,
                 target_attn = 'ATTN5',
                 # =================================================== #
                 mode='None',
                 algorithm_num = 10,
                 feature_scaling=0.9,
                 feature_scaling_region1=0.9,
                 feature_scaling_region2=0.9,
                 ca_scaling=2,
                 ca_scaling1=2,
                 ca_scaling2=2,
                 step1_num=28,
                 source_ratio=0.4,
                 souce_ratio_lo=0.0,
                 souce_ratio_hi = 0.6,
                 step2_search_num = 3,
                 search_mode = 'ternary',
                 step1_algorithm_num=15,
                 step2_algorithm_num=15,
                 # =================================================== #
                 important_layers = [10, 11, 12, 13, 23, 26, 27, 28, 29],
                 i_token=1,
                 save_ca = False,
                 source_ratio_region1 = 0.2,
                 source_ratio_region2 = 0.2,
                 print_interpolate_algorithm = False,
                 record_ca = False,
                 task1 = None,
                 task2 = None,
                 region1_src_ratio = 0.0,
                 region2_src_ratio=0.0,
                 base_src_ratio=0.0,
                 non_inject = False,
                 non_i_token_preserve = False):



        self.save_root = save_root
        self.unique_id = unique_id
        self.task = task
        self.injectiontask = injectiontask
        self.injectionratio = injectionratio
        self.mode = mode
        self.global_count = 0
        self.attn_num = 0

        self.attn_check_time = attn_check_time
        self.step1_num = step1_num
        self.algorithm_num = algorithm_num
        self.detailed_task = detailed_task
        self.detailed_task1 = detailed_task1
        self.detailed_task2 = detailed_task2
        self.source_ratio_region1 = source_ratio_region1
        self.source_ratio_region2 = source_ratio_region2
        self.step2_search_num = step2_search_num
        self.unique_id = unique_id
        self.print_interpolate_algorithm = print_interpolate_algorithm
        self.search_mode = search_mode
        ################################################################################
        self.mode = mode
        self.source_ratio = source_ratio
        self.source_ratio_lo = souce_ratio_lo
        self.source_ratio_hi = souce_ratio_hi
        self.important_layers = important_layers
        self.score_diff_dict = {}
        ################################################################################
        self.step1_algorithm_num = step1_algorithm_num
        self.step2_algorithm_num = step2_algorithm_num
        # Part 3
        self.save_ca = save_ca
        self.record_ca = record_ca
        # ======================================= #
        self.target_attn = target_attn
        self.feature_scaling = feature_scaling
        self.feature_scaling_region1 = feature_scaling_region1
        self.feature_scaling_region2 = feature_scaling_region2
        self.token_maps = {}
        self.token_scores = {}
        self.ca_bank = []
        self.loss_list = []
        self.e2e_blocks = []
        self.e2i_blocks = []
        self.i2e_blocks = []
        self.i2i_blocks = []
        self.source_token_blocks = []
        self.source_token_blocks = []
        self.loss_dict = {}
        self.ca_scaling = ca_scaling
        self.ca_scaling1 = ca_scaling1
        self.ca_scaling2 = ca_scaling2
        self.i_token = i_token
        ################################################################################
        self.task1 = task1
        self.task2 = task2
        self.region1_src_ratio = region1_src_ratio
        self.region2_src_ratio = region2_src_ratio
        self.base_src_ratio = base_src_ratio
        ################################################################################
        self.non_inject = non_inject
        self.non_i_token_preserve = non_i_token_preserve
        self.final_layer =  max(important_layers)

    def record_score(self, record, csv_dir):
        os.makedirs(os.path.dirname(csv_dir), exist_ok=True)
        file_exists = os.path.exists(csv_dir)
        with open(csv_dir, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                header = ["attn_num", "source_score", "target_score"]
                writer.writerow(header)
            writer.writerow(record)

    def record_final_search(self, search_file, search_rec):
        os.makedirs(os.path.dirname(search_file), exist_ok=True)
        file_exists = os.path.isfile(search_file)
        with open(search_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["step1_al_num","feature_scaling","ca_scaling","step2_search_num","step2_al_num","search_mode","search_result"])
            writer.writerow(search_rec)

    def register_mask(self,
                      red_mask,
                      green_mask,
                      blue_mask,
                      red2d_64_64,
                      green2d_64_64,
                      blue2d_64_64):

        self.red_mask = red_mask
        self.blue_mask = blue_mask
        self.green_mask = green_mask
        self.red_mask2d = red_mask.unsqueeze(-1).repeat(1, 4096).unsqueeze(0).unsqueeze(0)    # [1,1,4096,4096]
        self.blue_mask2d = blue_mask.unsqueeze(-1).repeat(1, 4096).unsqueeze(0).unsqueeze(0)  # [1,1,4096,4096]
        self.green_mask2d = green_mask.unsqueeze(-1).repeat(1, 4096).unsqueeze(0).unsqueeze(0)# [1,1,4096,4096]


        self.red2d_64_64   = torch.tensor(red2d_64_64).to('cuda') # 이게 numpy 라서 cuda 로 안가는데 torch ㄹ
        self.blue2d_64_64  = torch.tensor(blue2d_64_64).to('cuda') # 이게 numpy 라서 cuda 로 안가는데 torch ㄹ
        self.green2d_64_64 = torch.tensor(green2d_64_64).to('cuda') # 이게 numpy 라서 cuda 로 안가는데 torch ㄹ

        self.n_area_64_64 = red2d_64_64.sum()
        self.e_area_64_64 = (64*64 - red2d_64_64.sum())

        # STEP 1.1 Edit Portion
        self.edit_portion = ((1.0 - red_mask).to('cuda').float()).view(1, 1, 4096, 1)
        self.k_mask = 1.0 + (self.feature_scaling - 1.0) * self.edit_portion


        self.region1_portion = blue_mask.to("cuda").float().view(1, 1, 4096, 1)  # region1만 1, 나머지 0
        self.region2_portion = green_mask.to("cuda").float().view(1, 1, 4096, 1)  # region2만 1, 나머지 0
        self.k_mask_multi = (self.region1_portion * self.feature_scaling_region1
                             + self.region2_portion * self.feature_scaling_region2         #
                             + (1.0 - self.region1_portion - self.region2_portion) * 1.0)  # red portion -> 1

        # STEP 1.2
        self.edit_mask_ = self.blue_mask.view(1, 1, -1, 1) # .to(device=device, dtype=dtype)
        if self.ca_scaling != None :
            self.ca_scaling_on_e = self.ca_scaling * self.edit_mask_
            self.ca_scaling_on_n = self.ca_scaling * (1-self.edit_mask_)

        self.edit_mask_region1 = self.blue_mask.view(1, 1, -1, 1)   # .to(device=device, dtype=dtype)
        self.edit_mask_region2 = self.green_mask.view(1, 1, -1, 1)  # .to(device=device, dtype=dtype)

        self.edit_mask_region1_ = self.blue_mask.view(1, 1, -1)  # .to(device=device, dtype=dtype)
        self.edit_mask_region2_ = self.green_mask.view(1, 1,-1)  # .to(device=device, dtype=dtype)

        # STEP 2.1
        self.N = self.red_mask.view(-1)
        self.E = self.blue_mask.view(-1)  # edit query mask
        self.N_area = self.N.sum().clamp_min(1.0)
        self.E_area = self.E.sum().clamp_min(1.0)

    def record_attnweight(self, record):
        csv_file = os.path.join(self.save_root, f"{self.unique_id}_attndiff.csv")
        header = ["attn_num", "attn_id",
                  "N_score5", "E_score5", "ScoreDiff(N-E)5",
                  "N_score6", "E_score6", "ScoreDiff(N-E)6",
                  "N_score8", "E_score8", "ScoreDiff(N-E)8",
                  "N_score9", "E_score9", "ScoreDiff(N-E)9"]
        file_exists = os.path.exists(csv_file)
        with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(record)

    def reset(self, reset_ca=False):

        self.alpha_dict = {}
        self.attn_num = 0
        self.save_root = ""
        self.unique_id = ""

        if reset_ca:
            self.ca_bank = []
        self.dist_logs = []
        self.e2e_blocks = []
        self.e2i_blocks = []
        self.i2e_blocks = []
        self.i2i_blocks = []
        self.source_token_blocks = []
        self.source_token_blocks = []
        self.token_maps = {}
        self.token_scores = {}
        self.loss_dict = {}

def _save_token_heatmap(token_maps, save_path):
    if token_maps is None or len(token_maps) == 0:
        return

    # [(B,H,4096,n1), (B,H,4096,n2), ...] -> (B,H,4096,sum_n)
    x = torch.cat(token_maps, dim=-1)
    x = x.mean(dim=-1)
    x = x.mean(dim=(0, 1))
    # (4096,) -> (1,1,64,64)
    x = x.reshape(1, 1, 64, 64)

    # (64,64) -> (512,512)
    x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)

    x = x[0, 0].detach().float().cpu()

    # normalize
    x_min, x_max = x.min(), x.max()
    if (x_max - x_min) > 1e-8:
        x = (x - x_min) / (x_max - x_min)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # heatmap only
    plt.figure(figsize=(5.12, 5.12), dpi=100)
    plt.imshow(x.numpy(), cmap="viridis")
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()

def _region_extrema(x_bh_hw: torch.Tensor, mask_hw: torch.Tensor):
    """
    x_bh_hw: (B,H,4096,4096)
    mask_hw: (4096,4096) 1이면 해당 영역 포함, 0이면 제외
    return: (max_value, min_value) after mean over (B,H)
    """
    # (4096,4096)
    x_mean = x_bh_hw.mean(dim=(0, 1))

    # bool mask
    m = mask_hw.to(dtype=torch.bool, device=x_mean.device)

    # max over masked region
    x_for_max = x_mean.masked_fill(~m, float("-inf"))
    max_v = x_for_max.max()

    # min over masked region
    x_for_min = x_mean.masked_fill(~m, float("inf"))
    min_v = x_for_min.min()

    return max_v, min_v


    #important_layers = [6, 10, 11, 12, 13, 14, 25, 27, 28, 29, 30, 31, 32, 34, 52, 53, 54, 55]
    # important_layers = [10, 11, 12, 13, 23, 26,27, 28, 29]
    #important_layers = [10, 11, 12, 13, 23, 26]
    #important_layers =   [11, 12, 13]

def make_e2e_new(e2e, i2e, r: float) -> torch.Tensor:
    return e2e * (1.0 - r) + i2e * r

@torch.no_grad()
def feature_scaling(k, v, controller, multi = False):
    if controller.attn_num < controller.step1_num * 57:
        k_src = k[..., -4096:, :]  # (B,H,4096,hd)
        v_src = v[..., -4096:, :]
        k[..., -4096:, :] = k_src * controller.k_mask
        if controller.detailed_task == 'remove' :
            v[..., -4096:, :] = v_src * controller.k_mask
    return k, v


@torch.no_grad()
def ca_scaling(detailed_task, controller, attn_weight_temp,source_token_idx, target_token_idx) :

    device = attn_weight_temp.device
    dtype = attn_weight_temp.dtype
    img = slice(512, 512 + 4096)
    new_source_maps = []
    ca_scaling_on_e = controller.ca_scaling * controller.edit_mask_
    ca_scaling_on_n = controller.ca_scaling * (1-controller.edit_mask_)
    if source_token_idx is not None and len(source_token_idx) > 0 and detailed_task != 'add':
        src_ids = torch.as_tensor(source_token_idx, device=device, dtype=torch.long)
        src_view = attn_weight_temp[..., img, src_ids] # batch, head, 4096,1 -> ORIGINAL SCORE
        #new_source_map = (src_view - ca_scaling_on_e).to(dtype=dtype, device=device)
        new_source_map = (src_view - ca_scaling_on_e - ca_scaling_on_n).to(dtype=dtype, device=device)
        attn_weight_temp[..., img, src_ids] = new_source_map
        new_source_maps.append(new_source_map)
    unique_id = controller.attn_num % 57
    new_target_maps = []
    if detailed_task != "remove" and target_token_idx is not None and len(target_token_idx) > 0:
        trg_ids = torch.as_tensor(target_token_idx, device=device, dtype=torch.long)
        trg_view = attn_weight_temp[..., img, trg_ids] # original weight
        new_target_map = (trg_view + ca_scaling_on_e - ca_scaling_on_n).to(dtype=dtype, device=device)
        new_target_maps.append(new_target_map)
        attn_weight_temp[..., img, trg_ids] = new_target_map.to(dtype=dtype, device=device)
    return attn_weight_temp


#if controller.save_ca :
    #    if len(new_source_maps) > 0:
    #        s_heatmap_save_dir = os.path.join(controller.save_root, controller.unique_id, f'S_heapmap_{controller.attn_num}.png')
    #        _save_token_heatmap(new_source_maps, s_heatmap_save_dir)
    #    if len(new_target_maps) > 0:
    #        t_heatmap_save_dir = os.path.join(controller.save_root, controller.unique_id, f'T_heapmap_{controller.attn_num}.png')
    #        _save_token_heatmap(new_target_maps,t_heatmap_save_dir)
@torch.no_grad()
def interpolate_attn(attn_weight, controller, txt_len, tgt_len, src_len, multi = False):

    blocks = split_blocks(attn_weight, txt_len=txt_len, tgt_len=tgt_len, src_len=src_len)
    e2t = blocks["E2T"]
    e2e = blocks["E2E"]
    e2i = blocks["E2I"]
    i2e = blocks["I2E"]
    unique_layer_id = int(controller.attn_num % 57)
    # -------------------------------------------------
    important_layers = set(controller.important_layers)
    device = attn_weight.device
    dtype = e2t.dtype
    N_area = controller.N_area.to(device=device, dtype=dtype)
    E_area = controller.E_area.to(device=device, dtype=dtype)
    def region_scores_from_logits(e2e_new_logits: torch.Tensor):

        tei_logits = torch.cat([e2t, e2e_new_logits, e2i], dim=-1)   # (B,H,4096, txt+4096+img)
        tei_prob = torch.softmax(tei_logits, dim=-1)
        tei_prob_e = tei_prob[..., 512:512 + 4096] if controller.target_attn == 'ATTN5' else tei_prob[..., 512 + 4096:]
        q_score = tei_prob_e.mean(dim=(0, 1, 3))
        E_score = ((q_score * controller.E).sum() / E_area)
        N_score = (q_score * controller.N).sum() / N_area
        diff = E_score - N_score if controller.target_attn == 'ATTN5' else N_score - E_score
        return N_score, E_score, diff
    base_ratio = float(controller.source_ratio)
    base_e2e_new = make_e2e_new(e2e, i2e, base_ratio)
    do_search = (unique_layer_id in important_layers and controller.attn_num < controller.step2_search_num * 57)
    if do_search:

        if controller.search_mode == 'ternary' :
            lo_mid = 0.5 * (base_ratio + float(controller.source_ratio_lo))
            hi_mid = 0.5 * (base_ratio + float(controller.source_ratio_hi))
            lo_mid_e2e_new = make_e2e_new(e2e, i2e, lo_mid)
            hi_mid_e2e_new = make_e2e_new(e2e, i2e, hi_mid)
            base_N, base_E, base_diff = region_scores_from_logits(base_e2e_new)
            lo_N,   lo_E,   lo_diff   = region_scores_from_logits(lo_mid_e2e_new)
            hi_N,   hi_E,   hi_diff   = region_scores_from_logits(hi_mid_e2e_new)
            # ==================================================================================== #
            # Search
            diffs = torch.stack([base_diff, lo_diff, hi_diff])  # (3,)
            idx = int(torch.argmax(diffs).item())
            ratios = [base_ratio, lo_mid, hi_mid]
            e2e_news = [base_e2e_new, lo_mid_e2e_new, hi_mid_e2e_new]
            best_ratio = float(ratios[idx])
            best_e2e_new = e2e_news[idx]
            controller.source_ratio = best_ratio
            print(f' best ratio = {best_ratio}')
        elif controller.search_mode == 'binary' :
            hi_mid = 0.5 * (base_ratio + float(controller.source_ratio_hi))
            hi_mid_e2e_new = make_e2e_new(e2e, i2e, hi_mid)
            base_N, base_E, base_diff = region_scores_from_logits(base_e2e_new)
            hi_N, hi_E, hi_diff = region_scores_from_logits(hi_mid_e2e_new)
            diffs = torch.stack([base_diff, hi_diff])  # (3,)
            idx = int(torch.argmax(diffs).item())
            ratios = [base_ratio, hi_mid]
            e2e_news = [base_e2e_new, hi_mid_e2e_new]
            best_ratio = float(ratios[idx])
            best_e2e_new = e2e_news[idx]
            controller.source_ratio = best_ratio

        if controller.print_interpolate_algorithm :
            print(f'[{unique_layer_id}] ratio candidates: base {base_ratio:.4f}, lo {lo_mid:.4f}, hi {hi_mid:.4f} | '
                  f'diff: base {base_diff.item():.6f}, lo {lo_diff.item():.6f}, hi {hi_diff.item():.6f} | best {best_ratio:.4f}')
    else :
        best_e2e_new = base_e2e_new
    attn_weight[..., 512:512 + 4096, 512:512 + 4096] = best_e2e_new
    if controller.record_ca :
        _, _, base_diff = region_scores_from_logits(base_e2e_new)
        controller.score_diff_dict[unique_layer_id] = float(base_diff.item())
    return attn_weight

def ca_scaling(controller, attn_weight_temp,source_token_idx, target_token_idx, multi = False) :

    if not multi :
        detailed_task = controller.detailed_task
        device = attn_weight_temp.device
        dtype = attn_weight_temp.dtype
        img = slice(512, 512 + 4096)
        ca_scaling_on_e = controller.ca_scaling * controller.edit_mask_
        ca_scaling_on_n = controller.ca_scaling * (1-controller.edit_mask_)
        if source_token_idx is not None and len(source_token_idx) > 0 and detailed_task != 'add':
            src_ids = torch.as_tensor(source_token_idx, device=device, dtype=torch.long)
            src_view = attn_weight_temp[..., img, src_ids] # batch, head, 4096,1 -> ORIGINAL SCORE
            new_source_map = (src_view - ca_scaling_on_e - ca_scaling_on_n).to(dtype=dtype, device=device)
            attn_weight_temp[..., img, src_ids] = new_source_map
        if target_token_idx is not None and len(target_token_idx) > 0:
            trg_ids = torch.as_tensor(target_token_idx, device=device, dtype=torch.long)
            trg_view = attn_weight_temp[..., img, trg_ids] # original weight
            new_target_map = (trg_view + ca_scaling_on_e - ca_scaling_on_n).to(dtype=dtype, device=device)
            attn_weight_temp[..., img, trg_ids] = new_target_map.to(dtype=dtype, device=device)
    else :
        img = slice(512*2,512*2+4096)
        region1_src_token_idx_list, region2_src_token_idx_list = source_token_idx[0], source_token_idx[-1]
        region1_trg_token_idx_list, region2_trg_token_idx_list = target_token_idx[0], target_token_idx[-1]
        ca_scaling_on_region1 = controller.ca_scaling1 * controller.edit_mask_region1_
        ca_scaling_on_region2 = controller.ca_scaling2 * controller.edit_mask_region2_

        for src_idx in region1_src_token_idx_list :
            org_map = attn_weight_temp[...,img,src_idx]
            attn_weight_temp[...,img,src_idx] = org_map - ca_scaling_on_region1
        for src_idx in region2_src_token_idx_list :
            org_map = attn_weight_temp[...,img,src_idx]
            attn_weight_temp[..., img, src_idx] = org_map - ca_scaling_on_region2
        for trg_idx in region1_trg_token_idx_list :
            org_map = attn_weight_temp[...,img,trg_idx]
            attn_weight_temp[..., img, trg_idx] = org_map + ca_scaling_on_region1 - ca_scaling_on_region2
        for trg_idx in region2_trg_token_idx_list :
            org_map = attn_weight_temp[...,img,trg_idx]
            attn_weight_temp[..., img, trg_idx] = org_map - ca_scaling_on_region1 + ca_scaling_on_region2

    return attn_weight_temp
