# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import os
from keras import backend as K

K.set_image_dim_ordering('th')



def fetch_data_files(data_frame, data_fold):
    '''
    This function assumes the below dataset organization:
        subject_001/
            contrast1/
                contrast1_im_suffixe.nii.gz
                contrast1_target_suffixe.nii.gz
            contrast2/
                contrast2_im_suffixe.nii.gz
                contrast2_target_suffixe.nii.gz
        ...etc.
                
    Input:
        - data_frame: panda dataframe with at least the 2 following columns: subject, contrast_foldname
        - data_fold: absolute path of the data folder
        - im_suffixe: suffixe of the image to process (e.g. _res)
        - target_suffixe: suffixe of the groundtruth mask (e.g. _res_lesion_manual)
        
    Output: a list of list, each sublist containing the absolute path of both the image and its related groundtruth
    '''
    data_files = list()
    for s, t, m, l in zip(data_frame.subject.values, data_frame.Time_point.values, data_frame.Modality.values, data_frame.Labels.values):
        im_fname = data_fold + '\\'+ s +  '\\'+ t + '\\'+ m + '.nii'
        gt_fname = l
        #print(im_fname,gt_fname)
        if os.path.isfile(im_fname):
            subject_files = [im_fname, gt_fname]
            data_files.append(tuple(subject_files))
    return data_files




