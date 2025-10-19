#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_loss.py (customized)
- 固定 data/checkpoint/config/prompt（见下方常量）
- 逐步读取 episode_0.hdf5 的 images + qpos + action
- 调用 policy.infer，稳健抽取单步预测，计算 L1 loss
- 保存 per-step CSV 和 summary JSON
"""

import json
from pathlib import Path
import sys
import time
import csv

import h5py
import numpy as np
import cv2

# openpi imports (确保在激活的环境中可以 import openpi)
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from openpi.shared import normalize as _normalize

# -------------------- 用户固定配置 --------------------
H5_PATH = "/home/elian/Code/openpi/dataset/episode_0.hdf5"
CHECKPOINT_DIR = "/home/elian/Code/openpi/checkpoint"
TRAIN_CONFIG_NAME = "pi0_base_aloha_robotwin_lora"
PROMPT_STR = "Hold the gray kitchenpot with both arms"

# 输出文件
OUT_CSV = "validate_results_episode0.csv"
OUT_SUMMARY = "validate_results_episode0.summary.json"

# 图像预处理参数（如训练时用过 mean/std，请在下面填入）
IMG_SIZE = (224, 224)   # (width, height)
NORMALIZE_IMAGES = False  # 是否把 uint8 -> float32 /255
MEAN = None  # e.g. (0.485,0.456,0.406)
STD = None   # e.g. (0.229,0.224,0.225)
# --------------------------------------------------------------------


def decode_image_from_h5_bytes(img_bytes):
    """
    将 HDF5 中的 bytes/string 解码成 HWC uint8 图像 (RGB)
    """
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError("cv2.imdecode failed - image bytes may be corrupted")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def preprocess_image_for_model(img_rgb, img_size=IMG_SIZE, normalize=NORMALIZE_IMAGES, mean=MEAN, std=STD):
    """
    HWC uint8 RGB -> CHW float32
    - resize
    - /255 if normalize True
    - optional mean/std normalization if mean/std provided (in RGB order)
    """
    img = cv2.resize(img_rgb, img_size)
    # img = img.astype(np.float32) / 255.0 if normalize else img.astype(np.float32)
    # if mean is not None and std is not None:
    #     # expect mean/std length 3 in RGB order
    #     mean_a = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
    #     std_a = np.array(std, dtype=np.float32).reshape(3, 1, 1)
    #     img = (img.transpose(2, 0, 1) - mean_a) / std_a
    # else:
    img = img.transpose(2, 0, 1)
    return img


def extract_pred_vector(pred_action):
    """
    稳健地从 policy.infer 的输出中抽取单步 1D 向量：
    - 支持 pred 为 ndim 1/2/3 或 dict 带 'actions' 等
    - 返回 1D numpy array
    """
    # 如果传入的是 dict（policy 返回 dict），优先取 actions 键
    if isinstance(pred_action, dict):
        if "actions" in pred_action:
            pred = np.array(pred_action["actions"])
        else:
            # 找到第一个 array-like 值
            found = None
            for v in pred_action.values():
                try:
                    tmp = np.array(v)
                    if tmp.size:
                        found = tmp
                        break
                except Exception:
                    continue
            if found is None:
                raise RuntimeError("policy.infer returned dict but no actions-like value found")
            pred = found
    else:
        pred = np.array(pred_action)

    # 根据维度稳健提取单步向量
    if pred.ndim == 0:
        return pred.reshape(-1)
    if pred.ndim == 1:
        return pred.reshape(-1)
    if pred.ndim == 2:
        # ambiguous: treat as (T, D) or (B, D) -> take first row
        return pred[0].reshape(-1)
    if pred.ndim == 3:
        # assume (B, T, D) -> take batch0 timestep0
        return pred[0, 0].reshape(-1)
    # fallback: flatten to (N, D_last) and take first
    last_dim = pred.shape[-1]
    flat = pred.reshape(-1, last_dim)
    return flat[0].reshape(-1)


def safe_get_gt_action(h5f, step_idx):
    """
    从 h5 文件安全获取 GT action（优先级：'action' 顶层 -> joint_action group -> 搜索任何包含 'action' 的 key）
    返回 1D float32 numpy array
    """
    # top-level 'action'
    if "action" in h5f:
        return np.array(h5f["action"][step_idx], dtype=np.float64).reshape(-1)
    # joint_action group
    # if "joint_action" in h5f:
    #     grp = h5f["joint_action"]
    #     parts = []
    #     for k in sorted(grp.keys()):
    #         if isinstance(grp[k], h5py.Dataset):
    #             parts.append(np.asarray(grp[k][step_idx]).reshape(-1))
    #     if parts:
    #         return np.concatenate(parts).astype(np.float32)
    # # fallback: try any top-level key with 'action' in name
    for k in h5f.keys():
        if "action" in k.lower():
            return np.array(h5f[k][step_idx], dtype=np.float64).reshape(-1)
    raise KeyError("No action dataset found in HDF5.")


def main():
    start_time = time.time()
    h5_path = Path(H5_PATH)
    if not h5_path.exists():
        print("HDF5 文件不存在:", h5_path, file=sys.stderr)
        return 2

    # 加载模型 policy，使用固定的 config 与 checkpoint
    print(f"Loading policy: config={TRAIN_CONFIG_NAME}, checkpoint_dir={CHECKPOINT_DIR} ...")
    config = _config.get_config(TRAIN_CONFIG_NAME)
    policy = _policy_config.create_trained_policy(config, checkpoint_dir=CHECKPOINT_DIR)
    print("✅ policy loaded")

    # 打开 HDF5 并遍历每步
    with h5py.File(str(h5_path), "r") as f:
        # 数据布局里 images 在 observations/images，qpos 在 observations/qpos，action 在根部 'action'
        # 支持两种情况：有 observations group 或者直接在根
        if "observations" in f:
            obs_grp = f["observations"]
        else:
            obs_grp = f

        # 校验必须字段
        req_imgs = ["images/cam_high", "images/cam_left_wrist", "images/cam_right_wrist"]
        for k in req_imgs:
            if k not in obs_grp:
                raise KeyError(f"Missing required dataset observations/{k} in {h5_path}")

        if "qpos" not in obs_grp:
            raise KeyError(f"Missing required dataset observations/qpos in {h5_path}")

        # steps
        T = int(np.array(obs_grp["qpos"]).shape[0])
        print(f"Detected T = {T} steps in {h5_path}")

        # 打开 CSV 输出
        csv_path = Path(OUT_CSV)
        with csv_path.open("w", newline="") as csvf:
            # writer = csv.writer(csvf)
            # writer.writerow(["step", "l1_mean", "gt_min", "gt_max", "pred_min", "pred_max"])

            per_step_records = []
            for t in range(T):
                # 读取并解码三路相机
                cam_high_bytes = obs_grp["images/cam_high"][t]
                cam_left_bytes = obs_grp["images/cam_left_wrist"][t]
                cam_right_bytes = obs_grp["images/cam_right_wrist"][t]

                img_high = decode_image_from_h5_bytes(cam_high_bytes)
                img_left = decode_image_from_h5_bytes(cam_left_bytes)
                img_right = decode_image_from_h5_bytes(cam_right_bytes)

                # 预处理成 CHW float32（注意 normalize / mean/std）
                ch = preprocess_image_for_model(img_high, img_size=IMG_SIZE, normalize=NORMALIZE_IMAGES, mean=MEAN, std=STD)
                cl = preprocess_image_for_model(img_left,  img_size=IMG_SIZE, normalize=NORMALIZE_IMAGES, mean=MEAN, std=STD)
                cr = preprocess_image_for_model(img_right, img_size=IMG_SIZE, normalize=NORMALIZE_IMAGES, mean=MEAN, std=STD)

                # 读取 qpos 作为 state（1D）
                qpos = np.array(obs_grp["qpos"][t], dtype=np.float64).reshape(-1)
                if qpos.size != 14:
                    print(f"Warning: qpos size {qpos.size} at step {t} (expect 14)")

                # 构造 observation（包含 prompt）
                observation = {
                    "state": qpos,
                    "images": {
                        "cam_high": ch,
                        "cam_left_wrist": cl,
                        "cam_right_wrist": cr,
                    },
                    "prompt": PROMPT_STR,
                }

                # policy 推理
                out = policy.infer(observation)

                # 抽取预测向量
                pred_vec = extract_pred_vector(out).astype(np.float64)

                # 读取 GT action（优先根目录 action）
                gt_vec = safe_get_gt_action(f, t)

                # 如果长度不一致，取 min dim 进行比较（也可以决定 pad）
                min_d = min(len(pred_vec), len(gt_vec))
                if min_d == 0:
                    print(f"Step {t}: empty pred or gt, skipping")
                    continue

                loss = np.abs(pred_vec[:min_d] - gt_vec[:min_d])
                l1 = float(np.mean(loss))

                # 记录并写 CSV
                gt_min, gt_max = float(np.min(gt_vec)), float(np.max(gt_vec))
                pred_min, pred_max = float(np.min(pred_vec)), float(np.max(pred_vec))
                print(f"step {t:03d}: l1={l1:.6f}, gt_range=({gt_min:.4f},{gt_max:.4f}), pred_range=({pred_min:.4f},{pred_max:.4f})")
                # writer.writerow([t, f"{l1:.6f}", f"{gt_min:.6f}", f"{gt_max:.6f}", f"{pred_min:.6f}", f"{pred_max:.6f}"])
                per_step_records.append({"step": t, "l1": l1, "gt_min": gt_min, "gt_max": gt_max, "pred_min": pred_min, "pred_max": pred_max})

    # summary
    if per_step_records:
        all_l1 = np.array([r["l1"] for r in per_step_records], dtype=np.float32)
        mean_l1 = float(np.mean(all_l1))
    else:
        mean_l1 = float("nan")

    summary = {"steps_evaluated": len(per_step_records), "mean_l1": mean_l1, "h5_path": str(h5_path), "checkpoint": CHECKPOINT_DIR, "config": TRAIN_CONFIG_NAME}
    # with Path(OUT_SUMMARY).open("w") as sf:
    #     json.dump(summary, sf, indent=2)

    print("==== Summary ====")
    print(f"steps evaluated: {summary['steps_evaluated']}")
    print(f"mean L1 loss: {summary['mean_l1']:.6f}")
    # print("Saved per-step CSV to", OUT_CSV)
    # print("Saved summary JSON to", OUT_SUMMARY)
    print("Elapsed: {:.2f}s".format(time.time() - start_time))


if __name__ == "__main__":
    main()
