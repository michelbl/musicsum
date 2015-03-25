# -*- coding: utf-8 -*-

import numpy
import settings



def timeconv(date, rate, input_unit, output_unit):
    ''' Converts a date from one unit to another unit
    
        second    Time in second
        wav       Number of the frame in the original signal
        mel       Number of the Mel features frame
        feat      Number of the dynamic features frame
    '''
    
    def second_to_wav(date, rate):
        return date*rate
    
    def wav_to_second(date, rate):
        return date/rate
    
    def mel_to_wav(date, rate):
        frame_step = 0.01*rate
        window_size = 0.025*rate
        return date*frame_step + 0.5*window_size
    
    def wav_to_mel(date, rate):
        frame_step = 0.01*rate
        window_size = 0.025*rate
        return (date-0.5*window_size)/frame_step
    
    def mel_to_feature(date):
        frame_step = settings.HOP
        window_size = settings.FFT_SIZE
        return (date-0.5*window_size)/frame_step
        
    def feature_to_mel(date):
        frame_step = settings.HOP
        window_size = settings.FFT_SIZE
        return date*frame_step + 0.5*window_size
    
    switch = {
            ('second', 'second'): lambda date,rate: date,
            ('second', 'wav'): lambda date,rate: second_to_wav(date, rate),
            ('second', 'mel'): lambda date,rate: wav_to_mel(second_to_wav(date, rate), rate),
            ('second', 'feat'): lambda date,rate: mel_to_feature(wav_to_mel(second_to_wav(date, rate), rate)),
            
            ('wav', 'second'): lambda date,rate: wav_to_second(date, rate),
            ('wav', 'wav'): lambda date,rate: date,
            ('wav', 'mel'): lambda date,rate: wav_to_mel(date, rate),
            ('wav', 'feat'): lambda date,rate: mel_to_feature(wav_to_mel(date, rate)),
            
            ('mel', 'second'): lambda date,rate: wav_to_second(mel_to_wav(date, rate), rate),
            ('mel', 'wav'): lambda date,rate: mel_to_wav(date, rate),
            ('mel', 'mel'): lambda date,rate: date,
            ('mel', 'feat'): lambda date,rate: mel_to_feature(date),
            
            ('feat', 'second'): lambda date,rate: wav_to_second(mel_to_wav(feature_to_mel(date), rate), rate),
            ('feat', 'wav'): lambda date,rate: mel_to_wav(feature_to_mel(date), rate),
            ('feat', 'mel'): lambda date,rate: feature_to_mel(date),
            ('feat', 'feat'): lambda date,rate: date,
        }
    
    return switch[(input_unit, output_unit)](date, rate)
    