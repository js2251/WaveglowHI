# *****************************************************************************
#  Parent class in mel2samp.py at
#  https://github.com/NVIDIA/waveglow
# *****************************************************************************

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 18:15:01 2020

@author: js2251
"""

import random
import torch
import torch.utils.data
import sys

from mel2samp import Mel2Samp, load_wav_to_torch

# We're using the audio processing from TacoTron2 to make sure it matches
sys.path.insert(0, 'tacotron2')

MAX_WAV_VALUE = 32768.0

class Mel2SampTwoFiles( Mel2Samp ):
    def __init__(self, training_files, segment_length, filter_length,
                 hop_length, win_length, sampling_rate, mel_fmin, mel_fmax, dir_normal, dir_hi):
        super().__init__(training_files, segment_length, filter_length, hop_length, win_length, sampling_rate, mel_fmin, mel_fmax)        
        self.dir_normal = dir_normal
        self.dir_hi     = dir_hi
            
    def __getitem__(self, index):
        # Read audio
        filename = self.audio_files[index]
        audio_mel, sampling_rate = load_wav_to_torch(self.dir_normal + '/' + filename)
        audio, sampling_rate = load_wav_to_torch(self.dir_hi + '/' + filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

        # Take segment
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start:audio_start+self.segment_length]
            audio_mel = audio_mel[audio_start:audio_start+self.segment_length]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(0)), 'constant').data
            audio_mel = torch.nn.functional.pad(audio_mel, (0, self.segment_length - audio.size(0)), 'constant').data

        mel = self.get_mel(audio_mel)
        audio = audio / MAX_WAV_VALUE

        return (mel, audio)