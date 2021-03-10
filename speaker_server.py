#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 09:41:57 2020

@author: c95csy
"""
import os
import numpy as np
import toolkits
import utils as ut
import time
from flask import Flask, request, jsonify

class Speaker_model_inference():
    def __init__(self, moring_feats_path="feature_morning_muti1-12.npz", ai_feats_path="feature_muti_ai_Feb20.npz", feats_mode="all"):
        # init args
        self.moring_feats_path = moring_feats_path
        self.ai_feats_path = ai_feats_path
        self.feats_mode = feats_mode
        
        # init speaker feats
        self.init_default_speaker_feats()
        
        # init speaker model
        self.network_eval = self.init_speaker_model()
        
        
    @staticmethod
    def init_speaker_model():
        # ===========================================
        #        Parse the argument
        # ===========================================
        import argparse
        parser = argparse.ArgumentParser()
        # set up training configuration.
        parser.add_argument('--gpu', default='0', type=str)
        parser.add_argument('--resume', default='weights.h5', type=str)
        parser.add_argument('--batch_size', default=16, type=int)
        parser.add_argument('--data_path', default='/media/weidi/2TB-2/datasets/voxceleb1/wav', type=str)
        # set up network configuration.
        parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
        parser.add_argument('--ghost_cluster', default=2, type=int)
        parser.add_argument('--vlad_cluster', default=8, type=int)
        parser.add_argument('--bottleneck_dim', default=512, type=int)
        parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
        # set up learning rate, training loss and optimizer.
        parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
        parser.add_argument('--test_type', default='normal', choices=['normal', 'hard', 'extend'], type=str)
        
        global args
        args = parser.parse_args()
        
        import model
        toolkits.initialize_GPU(args)
        
        params = {'dim': (257, None, 1),
                  'nfft': 512,
                  'spec_len': 250,
                  'win_length': 400,
                  'hop_length': 160,
                  'n_classes': 5994,
                  'sampling_rate': 16000,
                  'normalize': True,
                  }
        
        network_eval = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                    num_class=params['n_classes'],
                                                    mode='eval', args=args)
        # ==> load pre-trained model ???
        if args.resume:
            # ==> get real_model from arguments input,
            # load the model if the imag_model == real_model.
            if os.path.isfile(args.resume):
                network_eval.load_weights(os.path.join(args.resume), by_name=True)
    
        return network_eval
    
    
    def init_default_speaker_feats(self):
        # load default speaker feats 
        # if default feats have update must modify speaker feats npz
        data = np.load(self.moring_feats_path)
        label_moring = data[data.files[0]]
        feats_moring = data[data.files[1]]
        
        data = np.load(self.ai_feats_path)
        label_ai = data[data.files[0]]
        feats_ai = data[data.files[1]]
        
        if self.feats_mode == "moring":
            self.feats = feats_moring
            self.label = label_moring
        elif self.feats_mode == "ai":
            self.feats = feats_ai
            self.label = label_ai
        else:
            label = np.concatenate((label_ai, label_moring), axis=0)
            feats = np.concatenate((feats_ai, feats_moring), axis=0)
            self.feats = feats
            self.label = label
        
          
    def speaker_predict(self, wav):
        # speaker predict fc
        
        ## wav : np.array , wav to specs
        specs = ut.load_data(wav, win_length=400, sr=16000,
                             hop_length=160, n_fft=512,
                             spec_len=250)
        specs = np.expand_dims(np.expand_dims(specs, 0), -1)
        
        ## model predict
        v = self.network_eval.predict(specs)
        
        return v




app = Flask(__name__)
#app.config['debug'] = True

'''
    POST data format:
    {
        'text': 'Your text'
    }
'''

@app.route('/SpeakerFeats', methods=['GET', 'POST'])
def SpeakerFeats():
    # predict speaker
    post_data = request.json
    
    s = time.time()
    wav = np.array(post_data['speaker'])
    feats = speaker_model.speaker_predict(wav)
    e = time.time()
    print(e-s)
    
    return {"feats":feats.tolist()}

@app.route('/speaker', methods=['GET', 'POST'])
def speaker():
    # predict speaker
    post_data = request.json
    
    s = time.time()
    wav = np.array(post_data['speaker'])
    feats = speaker_model.speaker_predict(wav)
    e = time.time()
    print(e-s)
    score = np.sum(feats*speaker_model.feats,axis=1)
    name = speaker_model.label[np.argmax(score)]
    
    print(name)
    return {"name":name, "score":"%.2f"%np.max(score)}

@app.route('/InitSpeakerFeats', methods=["POST"])
def InitSpeakerFeats():
    # InitSpeakerFeats : init speaker server to default speaker feats list
    speaker_model.feats_mode = init_feats_mode
    speaker_model.init_default_speaker_feats()
    
    return {"label":speaker_model.label.tolist(), "feats_mode":speaker_model.feats_mode}


@app.route('/ModifyFeatsMode', methods=["GET", "POST"])
def modify_feats_mode():
    # ModifyFeatsMode : modify speaker mode
    '''
    post -> data : {"feats_mode": mode}
        mode = "all", "moring", "ai"
    resp_data -> {label", "feats_mode"}
    '''
    
    try:
        post_data = request.json
        
        feats_mode = post_data["feats_mode"]
        speaker_model.feats_mode = feats_mode
        speaker_model.init_default_speaker_feats()
        
        return {"label":speaker_model.label.tolist(), "feats_mode":speaker_model.feats_mode}
    
    except:
        return """must post {"feats_mode":feats_mode} to set speaker feats mode ("all", "ai", "ai")"""    
        
@app.route('/ModifyFeatsCustom', methods=["GET", "POST"])
def modify_feats_custom():
    # ModifyFeatsCustom : modify custom speaker feas and label
    '''
    post -> data : {"label": custom_label, "feats": custom_feats}
    resp_data -> {label", "feats_mode"}
    '''
    try:
        post_data = request.json
    
        label = post_data["label"]
        feats = post_data["feats"]
        feats_mode = "custom"
        speaker_model.feats_mode = feats_mode
        speaker_model.label = np.array(label)
        speaker_model.feats = np.array(feats)
        
        return {"label":speaker_model.label.tolist(), "feats_mode":speaker_model.feats_mode}
    
    except:
        return """must post {"label":label, "feats":feats} to set custom speaker"""


@app.route("/GetSpeakerList", methods=["GET", "POST"])
def get_speaker_list():
    # GetSpeakerList : if want to know current speaker mode and label
    # resp_data -> {label", "feats", "feats_mode"}
    return {"label":speaker_model.label.tolist(), "feats":speaker_model.feats.tolist(), "feats_mode":speaker_model.feats_mode}


if __name__ == "__main__":
    
    # if default feats have update must modify speaker feats npz    
    moring_feats_path="feature_morning_muti1-12.npz"
    ai_feats_path="feature_muti_ai_Feb20.npz"
    
    # init speaker model
    init_feats_mode = "all"
    speaker_model = Speaker_model_inference(moring_feats_path, ai_feats_path, feats_mode=init_feats_mode)
    
    
    #app.run('172.16.121.124', '8002', threaded = False)
    app.run('127.0.0.1', '6000', threaded = False)
    #app.run('192.168.0.110', '8002', threaded = False)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
