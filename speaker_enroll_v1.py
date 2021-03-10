#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 09:34:35 2021

@author: c95csy
"""

import os
import numpy as np
import toolkits
import utils as ut
import model
import argparse

class SpeakerEnroll():
    def __init__(self, args):
        """
        init speaker model params and model
        """
        self.params = {'dim': (257, None, 1),
                       'nfft': 512,
                       'spec_len': 250,
                       'win_length': 400,
                       'hop_length': 160,
                       'n_classes': 5994,
                       'sampling_rate': 16000,
                       'normalize': True,
                       }
        
        # load model
        self.network_eval = model.vggvox_resnet2d_icassp(input_dim=self.params['dim'],
                                                         num_class=self.params['n_classes'],
                                                         mode='eval', args=args)
         
        # ==> load pre-trained model weight
        if args.resume:
            # ==> get real_model from arguments input,
            # load the model if the imag_model == real_model.
            if os.path.isfile(args.resume):
                self.network_eval.load_weights(os.path.join(args.resume), by_name=True)
             
                
    def flow(self, data_path, save_embeds_path=None):
        """
        speaker embeds enroll flow
        1. load wavs path
        2. generate speaker feats
        3. save feats to npz (default not save)
        """
        self.data_path = data_path
        # load wav path
        self.wav_paths = self.load_wav_paths(self.data_path)
        
        # generate speaker feats
        self.speaker_embeds = self.generate_speaker_embeds()
        
        # save feats to npz
        if save_embeds_path:
            np.savez(save_embeds_path, label=np.array(list(self.wav_paths.keys())), train_feats=self.speaker_embeds)
        
        print(f"save speaker embeds : {save_embeds_path}")
        
    def speaker_predict(self, path):
        """
        speaker extract embed fc
        1. load wav
        2. predict
        :params output: v (512,)
        """
        specs = ut.load_data(path, win_length=self.params['win_length'], sr=self.params['sampling_rate'],
                             hop_length=self.params['hop_length'], n_fft=self.params['nfft'],
                             spec_len=self.params['spec_len'], mode='eval')
        specs = np.expand_dims(np.expand_dims(specs, 0), -1)
        
        v = self.network_eval.predict(specs)
        return v
    
    @staticmethod
    def load_wav_paths(data_path):
        """
        load all wav path and speaker to dick
        key : speaker
        values : wavs path
        
        folder tree:
        - data_folder
             - speaker1
                 - wav1
                 - wav2
                 ...
             - speaker2
                 - wav1
                 ...
             ...
             
        """
        speaker = os.listdir(data_path)
        wav_paths = {}
        
        for k in speaker:
            wavs = os.listdir(os.path.join(data_path, k))
            wav_paths[k] = wavs
        
        return wav_paths
    
    def generate_speaker_embeds(self):
        """
        predict all wav to speaker feats
        if speaker has muti wav will extract to one better feats
            extract fc : sum all this speaker feats / len(wav num)
        """
        # init speaker_embeds
        speaker_embeds = np.zeros((0,512))
        
        for k, v in self.wav_paths.items():
            feats = np.zeros((0,512))
            # predict each wav to embed
            for p in v:
                path = os.path.join(self.data_path, k, p)
                embed = self.speaker_predict(path)
                feats = np.concatenate((feats, embed.reshape(1, 512)), axis=0)
            
            # extract fc
            feats = np.sum(feats, axis=0) / len(feats)
            
            # concat all speaker feats
            speaker_embeds = np.concatenate((speaker_embeds, feats.reshape(1, 512)), axis=0)
            print(f"generate {k} embeds")
            
        return speaker_embeds
            
           
    @staticmethod
    def load_speaker_feats(feats_path):
        """
        load speaker feats to nparray
        """
        data = np.load(feats_path)
        label = data[data.files[0]]
        speaker_feats = data[data.files[1]]
        
        return label, speaker_feats
       
if __name__ == "__main__":
    
    # ===========================================
    #        Parse the argument
    # ===========================================
    
    parser = argparse.ArgumentParser()
    # set up training configuration.
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--resume', default='weights.h5', type=str)
    # set up network configuration.
    parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
    parser.add_argument('--ghost_cluster', default=2, type=int)
    parser.add_argument('--vlad_cluster', default=8, type=int)
    parser.add_argument('--bottleneck_dim', default=512, type=int)
    parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
    # set up learning rate, training loss and optimizer.
    parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
    #parser.add_argument('--test_type', default='normal', choices=['normal', 'hard', 'extend'], type=str)
    
    args = parser.parse_args()
    
    toolkits.initialize_GPU(args)
    
    data_path = './speaker_data'
    save_embeds_path = './speaker_feats/ai.npz'
    
    speaker_enroll = SpeakerEnroll(args)
    speaker_enroll.flow(data_path, save_embeds_path)
    
 
        

    




























