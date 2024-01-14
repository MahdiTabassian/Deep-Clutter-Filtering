"""
A module with functions for computing different MAE and coherence errors.  
"""
import os
import numpy as np
import pandas as pd
import scipy.io as sio

def _compute_sample_temporal_coherency_score(filtered_smp, org_smp):
    abs_diff_flt_org_smp = np.abs(filtered_smp-org_smp)
    frm_pixel_sum = [np.sum(abs_diff_flt_org_smp[:,:,i]) for i in range(abs_diff_flt_org_smp.shape[-1])]
    frm_pixel_sum_shifted = np.roll(frm_pixel_sum, -1)
    frm_diff = np.abs(frm_pixel_sum - frm_pixel_sum_shifted)
    frm_diff = frm_diff[:-1]
    return np.mean(frm_diff)

def _compute_sample_mae(smp, in_ids, filtered_smp, clutter_class):
    clt_smp = sio.loadmat(in_ids[smp])['data_artf']
    org_smp = sio.loadmat(
        in_ids[smp].split(f'data_{clutter_class}')[0] + 'data_org/1.mat')['data_org']
    mae_CltFiltered_CltFree = np.mean(np.abs(255*filtered_smp-org_smp))
    mae_Cltrd_CltFree = np.mean(np.abs(clt_smp-org_smp))
    temporal_coherency_score = _compute_sample_temporal_coherency_score(255*filtered_smp, org_smp)
    return mae_CltFiltered_CltFree, mae_Cltrd_CltFree, temporal_coherency_score

def _make_res_dct():
    res_dct = {'Clutter_class': [],
               'Clutter_spec': [],
               'View': [],
               'Vendor': [],
               'MAE_CltFiltered_CltFree': [],
               'MAE_Cltrd_CltFree': [],
               'temporal_coherency_score': []
               }
    return res_dct

def _id_separation(in_id):
    id_part0 = in_id.split('/A')[0].split('/')
    id_part1 = in_id.split('/data_')[1].split('/')
    v = [v for v in in_id.split('/') if 'A' in v and 'C' in v]
    view = v[0]
    vendor = id_part0[-1]
    clutter_class = id_part1[0]
    clutter_spec = id_part1[1]
    return view, vendor, clutter_class, clutter_spec

def compute_mae(in_ids, filtered_dta, te_subsample=False, te_frames=50):
    res_dct = _make_res_dct()
    for i in range(len(in_ids)):
        view, vendor, clutter_class, clutter_spec = _id_separation(in_ids[i])
        res_dct['Clutter_class'].append(clutter_class)
        res_dct['Clutter_spec'].append(clutter_spec)
        res_dct['Vendor'].append(vendor)
        res_dct['View'].append(view)
        mae_CltFiltered_CltFree, mae_Cltrd_CltFree, temporal_coherency_score = _compute_sample_mae(
            smp=i, in_ids=in_ids, filtered_smp=filtered_dta[i,:,:,:,0],
            clutter_class=clutter_class)
        res_dct['MAE_CltFiltered_CltFree'].append(mae_CltFiltered_CltFree)
        res_dct['MAE_Cltrd_CltFree'].append(mae_Cltrd_CltFree)
        res_dct['temporal_coherency_score'].append(temporal_coherency_score)
    return pd.DataFrame(res_dct)