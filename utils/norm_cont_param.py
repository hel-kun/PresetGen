# 各continuous paramごとの正規化を行う式を定義する
import torch
import torch.nn as nn
import torch.nn.functional as F


# -1〜1に正規化のほうがいいかもしれないが、0~1で学習しても問題ないかも(実際に試してみる必要あり)
# これは-1〜1に正規化する関数群
NORM_CONT_PARAM_FUNCS = {
    "osc2_pitch": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "osc2_fine_tune": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "osc_mix": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "osc_pulse_width": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "osc_key_shift": lambda x:  x / 24.0, # -24~24 -> -1~1
    "osc_mod_env_amount": lambda x: (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "osc_mod_env_attack": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "osc_mod_env_decay": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "filter_attack": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "filter_decay": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "filter_sustain": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "filter_release": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "filter_freq": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "filter_resonance": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~
    "filter_amount": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "filter_kbd_track": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "filter_saturation": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "amp_attack": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "amp_decay": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "amp_sustain": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "amp_release": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "amp_gain": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "amp_velocity_sens": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "arpeggiator_beat": lambda x:  (x / 18.0) * 2.0 - 1.0, # 0~18 -> -1~1
    "arpeggiator_gate": lambda x:  ( (x - 5) / (127.0 - 5) ) * 2.0 - 1.0, # 5~127 -> -1~1
    "delay_time": lambda x:  (x / 19.0) * 2.0 - 1.0, # 0~19 -> -1~1
    "delay_feedback": lambda x:  ( (x - 1) / (120.0 - 1) ) * 2.0 - 1.0, # 1~120 -> -1~1
    "delay_dry_wet": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "portament_time": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "pitch_bend_range": lambda x:  (x / 24.0) * 2.0 - 1.0, # 0~24 -> -1~1
    "lfo1_speed": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "lfo1_depth": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "osc1_FM": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "lfo2_speed": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "lfo2_depth": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "midi_ctrl_sens1": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "midi_ctrl_sens2": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "chorus_delay_time": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "chorus_depth": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "chorus_rate": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "chorus_feedback": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "chorus_level": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "equalizer_tone": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "equalizer_freq": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "equalizer_level": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "equalizer_Q": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "osc12_fine_tune": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "unison_detune": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "osc1_detune": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "effect_control1": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "effect_control2": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "effect_level_mix": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "delay_time_spread": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "unison_pan_spread": lambda x:  (x / 127.0) * 2.0 - 1.0, # 0~127 -> -1~1
    "unison_pitch": lambda x:  (x / 48.0) * 2.0 - 1.0 # 0~48 -> -1~1
}