from dataclasses import dataclass
from typing import List, Tuple, Literal

@dataclass
class CategoricalParam:
  osc1_shape: Literal[0, 1, 2, 3] # 0: Sine, 1: Saw, 2: Square, 3: Triangle
  osc2_shape: Literal[0, 1, 2, 3, 4] # 0: None, 1: Sine, 2: Saw, 3: Square, 4: Triangle
  osc2_kbd_track: Literal[0, 1] # 0: Off, 1: On
  osc2_sync: Literal[0, 1] # 0: Off, 1: On
  osc2_ring_modulation: Literal[0, 1] # 0: Off, 1: On
  osc_mod_env_on_off: Literal[0, 1] # 0: Off, 1: On
  filter_type: Literal[0, 1, 2, 3, 4, 5] # LP12/LP24/HP12/BP12 (の4つなのに何故か5つある)
  filter_velocity_switch: Literal[0, 1] # 0: Off, 1: On
  arpeggiator_type: Literal[0, 1, 2, 3, 4] # 0: None
  arpeggiator_oct_range: Literal[0, 1, 2, 3, 4]
  play_mode_type: Literal[0, 1, 2] # Poly/Mono/Legato
  lfo1_destination: Literal[0, 1, 2, 3, 4, 5, 6, 7] # 0: None, 1~7
  lfo1_type: Literal[0, 1, 2, 3, 4, 5]
  lfo2_destination: Literal[0, 1, 2, 3, 4, 5, 6, 7] # 0: None, 1~7
  lfo2_type: Literal[0, 1, 2, 3, 4, 5]
  lfo1_on_off: Literal[0, 1] # 0: Off, 1: On
  lfo2_on_off: Literal[0, 1] # 0: Off, 1: On
  arpeggiator_on_off: Literal[0, 1] # 0: Off, 1: On
  chorus_type: Literal[0, 1, 2 ,3, 4] # 0: None, 1~4
  delay_on_off: Literal[0, 1] # 0: Off, 1: On
  chorus_on_off: Literal[0, 1] # 0: Off, 1: On
  lfo1_tempo_sync: Literal[0, 1]
  lfo1_key_sync: Literal[0, 1]
  lfo2_tempo_sync: Literal[0, 1]
  lfo2_key_sync: Literal[0, 1]
  osc_mod_dest: Literal[0, 1, 2, 3] # OSC2/FM/PW
  unison_mode: Literal[0, 1]
  portament_auto_mode: Literal[0, 1]
  effect_on_off: Literal[0, 1] # 0: Off, 1: On
  effect_type: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  delay_type: Literal[0, 1, 2]

@dataclass
class ContinuiusParam:
  osc2_pitch: int # 0~127(-60/+60)
  osc2_fine_tune: int # 0~127(-62/+61)
  osc_mix: int # 0~127(100:0/0:100)
  osc_pulse_width: int # 0~127(0.5%~99.5%)
  osc_key_shift: int # -24~24
  osc_mod_env_amount: int # 0~127
  osc_mod_env_attack: int # 0~127
  osc_mod_env_decay: int # 0~127
  filter_attack: int # 0~127
  filter_decay: int # 0~127
  filter_sustain: int # 0~127
  filter_release: int # 0~127
  filter_freq: int # 0~127
  filter_resonance: int # 0~127
  filter_amount: int # 0~127
  filter_kbd_track: int # 0~127
  filter_saturation: int # 0~127
  amp_attack: int # 0~127
  amp_decay: int # 0~127
  amp_sustain: int # 0~127
  amp_release: int # 0~127
  amp_gain: int # 0~127
  amp_velocity_sens: int # 0~127
  arpeggiator_beat: int # 0~18
  arpeggiator_gate: int # 5~127
  delay_time: int # 0~19
  delay_feedback: int # 1~120(0/127)
  delay_dry_wet: int # 0~127
  portament_time: int # 0~127
  pitch_bend_range: int # 0~24
  lfo1_speed: int # 0~127
  lfo1_depth: int # 0~127
  osc1_FM: int # 0~127
  lfo2_speed: int # 0~127
  lfo2_depth: int # 0~127
  midi_ctrl_sens1: int # 0~127
  midi_ctrl_sens2: int # 0~127
  chorus_delay_time: int # 0~127
  chorus_depth: int # 0~127
  chorus_rate: int # 0~127
  chorus_feedback: int # 0~127
  chorus_level: int # 0~127
  equalizer_tone: int # 0~127
  equalizer_freq: int # 0~127
  equalizer_level: int # 0~127
  equalizer_Q: int # 0~127
  osc12_fine_tune: int # 0~127
  unison_detune: int # 0~127
  osc1_detune: int # 0~127
  effect_control1: int # 0~127
  effect_control2: int # 0~127
  effect_level_mix: int # 0~127
  delay_time_spread: int # 0~127
  unison_pan_spread: int # 0~127
  unison_pitch: int # 0~48

# 学習しないパラメータ群
@dataclass
class MiscParam:
  unison_phase_shift: int
  unison_voice_num: int
  osc1_sub_gain: int
  osc1_sub_shape: int
  osc1_sub_octave: int
  delay_tone: int
  midi_ctrl_src1: int = 45057
  midi_ctrl_assign1: int = 44
  midi_ctrl_src2: int = 45057
  midi_ctrl_assign2: int = 43
  # ここ以降は悩みどころ
  pan: int = 64 # 0~127
  osc_phase_shift: int = 0 # 0~127
  unknown_params: int = 16 # 94個目のパラメータ(不明)

@dataclass
class Synth1Preset:
  categorical_param: CategoricalParam
  continuius_param: ContinuiusParam
  misc_param: MiscParam