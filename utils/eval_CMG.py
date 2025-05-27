import os
import logging
import torch
import numpy as np
from sklearn.preprocessing import normalize
from .rerank import re_ranking, pairwise_distance
from torch.nn import functional as F

def get_gallery_names_CMG(perm, cams, ids, trial_id, num_shots=1):
    names = []
    for cam in cams:
        cam_perm = perm[cam - 1].squeeze()  # perm shape: [6, num_ids, num_trials, num_shots]
        #print("cam_perm shape:", cam_perm.shape)
        for i in ids:
            instance_id = cam_perm[i - 1, trial_id][:num_shots]
            names.extend(['camera{}/{:03d}/c{}_{}_{:d}.jpeg'.format(cam, i, cam, i, ins) for ins in instance_id])
        #print(names)
    return names

def get_unique(array):
    _, idx = np.unique(array, return_index=True)
    return array[np.sort(idx)]

def get_cmc(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids):
    gallery_unique_count = get_unique(gallery_ids).shape[0]
    match_counter = np.zeros((gallery_unique_count,))
    result = gallery_ids[sorted_indices]
    cam_locations_result = gallery_cam_ids[sorted_indices]
    valid_probe_sample_count = 0
    for probe_index in range(sorted_indices.shape[0]):
        result_i = result[probe_index, :]
        result_i[np.equal(cam_locations_result[probe_index], query_cam_ids[probe_index])] = -1
        result_i = np.array([i for i in result_i if i != -1])
        result_i_unique = get_unique(result_i)
        match_i = np.equal(result_i_unique, query_ids[probe_index])
        if np.sum(match_i) != 0:
            valid_probe_sample_count += 1
            match_counter += match_i
    rank = match_counter / valid_probe_sample_count
    cmc = np.cumsum(rank)
    return cmc

def get_mAP(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids):
    result = gallery_ids[sorted_indices]
    cam_locations_result = gallery_cam_ids[sorted_indices]
    valid_probe_sample_count = 0
    avg_precision_sum = 0
    for probe_index in range(sorted_indices.shape[0]):
        result_i = result[probe_index, :]
        result_i[cam_locations_result[probe_index, :] == query_cam_ids[probe_index]] = -1
        result_i = np.array([i for i in result_i if i != -1])
        match_i = result_i == query_ids[probe_index]
        true_match_count = np.sum(match_i)
        if true_match_count != 0:
            valid_probe_sample_count += 1
            true_match_rank = np.where(match_i)[0]
            ap = np.mean(np.arange(1, true_match_count + 1) / (true_match_rank + 1))
            avg_precision_sum += ap
    mAP = avg_precision_sum / valid_probe_sample_count
    return mAP

def eval_CMG(query_feats, query_ids, query_cam_ids, gallery_feats, gallery_ids, gallery_cam_ids, gallery_img_paths,
              perm, num_shots=1, num_trials=10, rerank=False):
    # CMG: infrared cams = 4,5,6
    gallery_cams = [1, 2, 3, 4, 5, 6]  # all cams included
    query_feats = F.normalize(query_feats, dim=1)
    gallery_feats = F.normalize(gallery_feats, dim=1)
    #print("query_feats shape:", query_feats.shape)
    # Load permutation file
    #perm = np.load(perm_path)  # shape: [6, num_ids, num_trials, num_shots]
    # Extract gallery info
    gallery_names = np.array(['/'.join(path.replace('\\', '/').split('/')[-3:]) for path in gallery_img_paths])
    #print(gallery_img_paths[0:2])
    #print(gallery_names[0:2])
    gallery_id_set = np.unique(gallery_ids)
    mAP, r1, r5, r10, r20 = 0, 0, 0, 0, 0
    for t in range(num_trials):
        names = get_gallery_names_CMG(perm, gallery_cams, gallery_id_set, t, num_shots)
        flag = np.in1d(gallery_names, names)
        g_feat = gallery_feats[flag]
        print("Flag sum (should > 0):", np.sum(flag))
        g_ids = gallery_ids[flag]
        g_cam_ids = gallery_cam_ids[flag]
        if rerank:
            dist_mat = re_ranking(query_feats, g_feat)
        else:
            dist_mat = pairwise_distance(query_feats, g_feat)
        sorted_indices = np.argsort(dist_mat, axis=1)
        mAP += get_mAP(sorted_indices, query_ids, query_cam_ids, g_ids, g_cam_ids)
        cmc = get_cmc(sorted_indices, query_ids, query_cam_ids, g_ids, g_cam_ids)
        r1 += cmc[0]
        r5 += cmc[4]
        r10 += cmc[9]
        r20 += cmc[19]
    r1 = r1 / num_trials * 100
    r5 = r5 / num_trials * 100
    r10 = r10 / num_trials * 100
    r20 = r20 / num_trials * 100
    mAP = mAP / num_trials * 100
    perf = 'CMG {}-shot: r1 = {:.2f}, r5 = {:.2f}, r10 = {:.2f}, r20 = {:.2f}, mAP = {:.2f}'
    logging.info(perf.format(num_shots, r1, r5, r10, r20, mAP))
    return mAP, r1, r5, r10, r20

