""" 
Utility functions

"""
import os
import numpy as np

def generate_tr_val_te_subject_ids(subject_list, val_subject_id):
	val_subject = subject_list[val_subject_id]
	te_subject = subject_list[val_subject_id-1]
	subject_list.remove(val_subject)
	subject_list.remove(te_subject)
	tr_subjects = subject_list
	return tr_subjects, val_subject, te_subject

def generate_data_ids(data_dir, subject_list):
    in_ids, out_ids = [], []
    vendor_list = [vendor for vendor in os.listdir(data_dir) if '.' not in vendor]
    for vendor in vendor_list:
        vendor_dir = os.path.join(data_dir, vendor)
        view_list = [view for view in os.listdir(vendor_dir) if '.' not in view]
        for view in view_list:
            view_dir = os.path.join(vendor_dir, view) 
            subject_full_list = [subject for subject in os.listdir(view_dir) if '.' not in subject]
            for subject in subject_full_list:
                if subject in subject_list:
                    subject_dir = os.path.join(view_dir, subject)
                    org_data_dir = os.path.join(subject_dir, 'data_org')
                    org_data_id = os.path.join(org_data_dir, os.listdir(org_data_dir)[0])
                    clutter_list = [clutter for clutter in os.listdir(subject_dir)
                                    if clutter in ['data_NFClt', 'data_NFRvbClt', 'data_RvbClt']
                                    and '.' not in clutter]
                    for clutter in clutter_list:
                        clutter_dir = os.path.join(subject_dir, clutter)
                        clutter_ids = os.listdir(clutter_dir)
                        clutter_ids_dir = [os.path.join(clutter_dir, id_dir) for id_dir in clutter_ids if '.DS' not in id_dir]
                        in_ids += clutter_ids_dir
                        out_ids += [org_data_id]*len(os.listdir(clutter_dir))
    return in_ids, out_ids

def id_preparation(config):
    tr_subjects, val_subject, te_subject = generate_tr_val_te_subject_ids(
        subject_list=config["subject_list"], val_subject_id=config["CV"]["val_subject_id"])
    if config["tr_phase"]:
        in_ids_tr, out_ids_tr = generate_data_ids(config["paths"]["data_path"], tr_subjects)
        in_ids_val, out_ids_val = generate_data_ids(config["paths"]["data_path"], val_subject)
        return in_ids_tr, in_ids_val, out_ids_tr, out_ids_val, val_subject, te_subject
    else:
        in_ids_te, out_ids_te = generate_data_ids(config["paths"]["data_path"], te_subject)
        return in_ids_te, out_ids_te, te_subject, val_subject

def create_weight_dir(val_subject, te_subject, config):
    weight_dir = os.path.join(config["paths"]["save_path"],
                              "Weights", f"ValTeIDs_{val_subject}_{te_subject}")  
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    return weight_dir