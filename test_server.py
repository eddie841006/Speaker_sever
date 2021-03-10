#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 11:03:12 2020

@author: c95csy
"""

import requests
import time
import librosa
import json
import numpy as np



def test_InitSpeakerFeats():
    # InitSpeakerFeats : init speaker server to default speaker feats list
    resp = requests.post(url + "InitSpeakerFeats")
    resp_data = json.loads(resp.text)
    print(resp_data)

def test_speaker(file):
    # predict speaker
    ## load wav
    sr = 16000
    s = time.time()
    wav, sr = librosa.load(file, sr=sr)
    
    if not isinstance(wav, str):
        data = {"speaker":wav.tolist()}
        resp = requests.post(url + "speaker",json=data)
        e = time.time()
        print(e-s)
        resp_data = json.loads(resp.text)
        print(resp_data)
        
    return resp_data

def test_GetSpeakerList():
    # GetSpeakerList : if want to know current speaker mode and label
    # resp_data -> {label", "feats", "feats_mode"}
    resp = requests.post(url + "GetSpeakerList")
    resp_data = json.loads(resp.text)
    print("feats_mode : " + resp_data["feats_mode"])
    print(resp_data["label"])
    
    return resp_data

def test_ModifyFeatsMode(data):
    # ModifyFeatsMode : modify speaker mode
    '''
    post -> data : {"feats_mode": mode}
        mode = "all", "moring", "ai"
    resp_data -> {label", "feats_mode"}
    '''
    resp = requests.post(url + "ModifyFeatsMode", json=data)
    resp_data = json.loads(resp.text)
    print("feats_mode : " + resp_data["feats_mode"])
    print(resp_data["label"])
    
    return resp_data

def test_ModifyFeatsCustom(data):
    # ModifyFeatsCustom : modify custom speaker feas and label
    '''
    post -> data : {"label": custom_label, "feats": custom_feats}
    resp_data -> {label", "feats_mode"}
    '''
    resp = requests.post(url + "ModifyFeatsCustom", json=data)
    resp_data = json.loads(resp.text)
    print("feats_mode : " + resp_data["feats_mode"])
    print(resp_data["label"])
    
    return resp_data


if __name__ == "__main__":
    
    url = "http://127.0.0.1:6000/"
    
    # InitSpeakerFeats : init speaker server to default speaker feats list
    test_InitSpeakerFeats()
    
    # predict speaker
    file = "record_2.wav"
    resp_data = test_speaker(file)
    
    # GetSpeakerList : get current speaker mode and label
    resp_data = test_GetSpeakerList()
    
    # ModifyFeatsMode : modify speaker mode
    data = {"feats_mode":"moring"}
    resp_data = test_ModifyFeatsMode(data)
    
    # ModifyFeatsCustom : modify custom speaker feas and label
    data = np.load("feature_muti_ai_Feb20.npz")
    label = data[data.files[0]][:10]
    feats = data[data.files[1]][:10, ...]
    data = {"label":label.tolist(), "feats":feats.tolist()}
    resp_data = test_ModifyFeatsCustom(data)
