# パラメータの名前のリスト
CATEGORICAL_PARAM_NAMES = [
    "osc1_shape", "osc2_shape", "osc2_kbd_track", "osc2_sync", "osc2_ring_modulation",
    "osc_mod_env_on_off", "filter_type", "filter_velocity_switch", "arpeggiator_type",
    "arpeggiator_oct_range", "play_mode_type", "lfo1_destination", "lfo1_type",
    "lfo2_destination", "lfo2_type", "lfo1_on_off", "lfo2_on_off", "arpeggiator_on_off",
    "chorus_type", "delay_on_off", "chorus_on_off", "lfo1_tempo_sync", "lfo1_key_sync",
    "lfo2_tempo_sync", "lfo2_key_sync", "osc_mod_dest", "unison_mode", "portament_auto_mode",
    "effect_on_off", "effect_type", "delay_type"
]
CONTINUOUS_PARAM_NAMES = [
    "osc2_pitch", "osc2_fine_tune", "osc_mix", "osc_pulse_width", "osc_key_shift",
    "osc_mod_env_amount", "osc_mod_env_attack", "osc_mod_env_decay", "filter_attack",
    "filter_decay", "filter_sustain", "filter_release", "filter_freq", "filter_resonance",
    "filter_amount", "filter_kbd_track", "filter_saturation", "amp_attack", "amp_decay",
    "amp_sustain", "amp_release", "amp_gain", "amp_velocity_sens", "arpeggiator_beat",
    "arpeggiator_gate", "delay_time", "delay_feedback", "delay_dry_wet", "portament_time",
    "pitch_bend_range", "lfo1_speed", "lfo1_depth", "osc1_FM", "lfo2_speed", "lfo2_depth",
    "midi_ctrl_sens1", "midi_ctrl_sens2", "chorus_delay_time", "chorus_depth", "chorus_rate",
    "chorus_feedback", "chorus_level", "equalizer_tone", "equalizer_freq", "equalizer_level",
    "equalizer_Q", "osc12_fine_tune", "unison_detune", "osc1_detune", "effect_control1",
    "effect_control2", "effect_level_mix", "delay_time_spread", "unison_pan_spread","unison_pitch"
]
MISC_PARAM_NAMES = [
    "midi_ctrl_src1", "midi_ctrl_assign1", "midi_ctrl_src2", "midi_ctrl_assign2",
    "pan", "osc_phase_shift", "unison_phase_shift", "unison_voice_num", "unknown_params",
    "osc1_sub_gain", "osc1_sub_shape", "osc1_sub_octave", "delay_tone"
]

# デフォルト値の定義
CATEGORICAL_DEFAULTS = {
    "osc1_shape": 1,  # Saw
    "osc2_shape": 2,  
    "osc2_kbd_track": 1,  # On
    "osc2_sync": 0,  # Off
    "osc2_ring_modulation": 0,  # Off
    "osc_mod_env_on_off": 0,  # Off
    "filter_type": 0,  # LP12
    "filter_velocity_switch": 0,  # Off
    "arpeggiator_type": 1,
    "arpeggiator_oct_range": 0,
    "play_mode_type": 0,  # Poly
    "lfo1_destination": 1,
    "lfo1_type": 0,
    "lfo2_destination": 1,
    "lfo2_type": 0,
    "lfo1_on_off": 0,  # Off
    "lfo2_on_off": 0,  # Off
    "arpeggiator_on_off": 0,  # Off
    "chorus_type": 1,
    "delay_on_off": 0,  # Off
    "chorus_on_off": 0,  # Off
    "lfo1_tempo_sync": 0,
    "lfo1_key_sync": 0,
    "lfo2_tempo_sync": 0,
    "lfo2_key_sync": 0,
    "osc_mod_dest": 0,  # OSC2
    "unison_mode": 0,
    "portament_auto_mode": 0,
    "effect_on_off": 0,  # Off
    "effect_type": 0,
    "delay_type": 0
}

CONTINUOUS_DEFAULTS = {
    "osc2_pitch": 64,  # 0
    "osc2_fine_tune": 64,  # 0
    "osc_mix": 64,  # 50:50
    "osc_pulse_width": 64,  # 50%
    "osc_key_shift": 0,
    "osc_mod_env_amount": 64,
    "osc_mod_env_attack": 0,
    "osc_mod_env_decay": 64,
    "filter_attack": 0,
    "filter_decay": 64,
    "filter_sustain": 64,
    "filter_release": 64,
    "filter_freq": 127,  # 最大
    "filter_resonance": 0,
    "filter_amount": 64,
    "filter_kbd_track": 64,
    "filter_saturation": 0,
    "amp_attack": 0,
    "amp_decay": 64,
    "amp_sustain": 127,  # 最大
    "amp_release": 64,
    "amp_gain": 100,
    "amp_velocity_sens": 64,
    "arpeggiator_beat": 6,
    "arpeggiator_gate": 64,
    "delay_time": 10,
    "delay_feedback": 64,
    "delay_dry_wet": 64,
    "portament_time": 0,
    "pitch_bend_range": 2,
    "lfo1_speed": 64,
    "lfo1_depth": 64,
    "osc1_FM": 0,
    "lfo2_speed": 64,
    "lfo2_depth": 64,
    "midi_ctrl_sens1": 64,
    "midi_ctrl_sens2": 64,
    "chorus_delay_time": 64,
    "chorus_depth": 64,
    "chorus_rate": 64,
    "chorus_feedback": 64,
    "chorus_level": 64,
    "equalizer_tone": 64,
    "equalizer_freq": 64,
    "equalizer_level": 64,
    "equalizer_Q": 64,
    "osc12_fine_tune": 64,
    "unison_detune": 64,
    "osc1_detune": 0,
    "effect_control1": 64,
    "effect_control2": 64,
    "effect_level_mix": 64,
    "delay_time_spread": 64,
    "unison_pan_spread": 64,
    "unison_pitch": 0
}

MISC_DEFAULTS = {
    "midi_ctrl_src1": 45057,
    "midi_ctrl_assign1": 44,
    "midi_ctrl_src2": 45057,
    "midi_ctrl_assign2": 43,
    "pan": 64,
    "osc_phase_shift": 0,
    "unknown_params": 16,
    "unison_phase_shift": 0,
    "unison_voice_num": 2,
    "osc1_sub_gain": 0,
    "osc1_sub_shape": 0,
    "osc1_sub_octave": 0,
    "delay_tone": 64
}

# パラメータIDから名前へのマッピング
PARAM_ID_TO_NAME = {
  0: "osc1_shape",
  1: "osc2_shape",
  2: "osc2_pitch",
  3: "osc2_fine_tune",
  4: "osc2_kbd_track",
  5: "osc_mix",
  6: "osc2_sync",
  7: "osc2_ring_modulation",
  8: "osc_pulse_width",
  9: "osc_key_shift",
  10: "osc_mod_env_on_off",
  11: "osc_mod_env_amount",
  12: "osc_mod_env_attack",
  13: "osc_mod_env_decay",
  14: "filter_type",
  15: "filter_attack",
  16: "filter_decay",
  17: "filter_sustain",
  18: "filter_release",
  19: "filter_freq",
  20: "filter_resonance",
  21: "filter_amount",
  22: "filter_kbd_track",
  23: "filter_saturation",
  24: "filter_velocity_switch",
  25: "amp_attack",
  26: "amp_decay",
  27: "amp_sustain",
  28: "amp_release",
  29: "amp_gain",
  30: "amp_velocity_sens",
  31: "arpeggiator_type",
  32: "arpeggiator_oct_range",
  33: "arpeggiator_beat",
  34: "arpeggiator_gate",
  35: "delay_time",
  36: "delay_feedback",
  37: "delay_dry_wet",
  38: "play_mode_type",
  39: "portament_time",
  40: "pitch_bend_range",
  41: "lfo1_destination",
  42: "lfo1_type",
  43: "lfo1_speed",
  44: "lfo1_depth",
  45: "osc1_FM",
  46: "lfo2_destination",
  47: "lfo2_type",
  48: "lfo2_speed",
  49: "lfo2_depth",
  50: "midi_ctrl_sens1",
  51: "midi_ctrl_sens2",
  52: "chorus_delay_time",
  53: "chorus_depth",
  54: "chorus_rate",
  55: "chorus_feedback",
  56: "chorus_level",
  57: "lfo1_on_off",
  58: "lfo2_on_off",
  59: "arpeggiator_on_off",
  60: "equalizer_tone",
  61: "equalizer_freq",
  62: "equalizer_level",
  63: "equalizer_Q",
  64: "chorus_type",
  65: "delay_on_off",
  66: "chorus_on_off",
  67: "lfo1_tempo_sync",
  68: "lfo1_key_sync",
  69: "lfo2_tempo_sync",
  70: "lfo2_key_sync",
  71: "osc_mod_dest",
  72: "osc12_fine_tune",
  73: "unison_mode",
  74: "portament_auto_mode",
  75: "unison_detune",
  76: "osc1_detune",
  77: "effect_on_off",
  78: "effect_type",
  79: "effect_control1",
  80: "effect_control2",
  81: "effect_level_mix",
  82: "delay_type",
  83: "delay_time_spread",
  84: "unison_pan_spread",
  85: "unison_pitch",
  86: "midi_ctrl_src1",
  87: "midi_ctrl_assign1",
  88: "midi_ctrl_src2",
  89: "midi_ctrl_assign2",
  90: "pan",
  91: "osc_phase_shift",
  92: "unison_phase_shift",
  93: "unison_voice_num",
  94: "unknown_params", # 94個目のパラメータ(不明)
  95: "osc1_sub_gain",
  96: "osc1_sub_shape",
  97: "osc1_sub_octave",
  98: "delay_tone"
}

CATEG_PARAM_SIZE = {
    "osc1_shape": 4,
    "osc2_shape": 5,
    "osc2_kbd_track": 2,
    "osc2_sync": 2,
    "osc2_ring_modulation": 2,
    "osc_mod_env_on_off": 2,
    "filter_type": 5,
    "filter_velocity_switch": 2,
    "arpeggiator_type": 5,
    "arpeggiator_oct_range": 4,
    "play_mode_type": 3,
    "lfo1_destination": 8,
    "lfo1_type": 6,
    "lfo2_destination": 8,
    "lfo2_type": 6,
    "lfo1_on_off": 2,
    "lfo2_on_off": 2,
    "arpeggiator_on_off": 2,
    "chorus_type": 5,
    "delay_on_off": 2,
    "chorus_on_off": 2,
    "lfo1_tempo_sync": 2,
    "lfo1_key_sync": 2,
    "lfo2_tempo_sync": 2,
    "lfo2_key_sync": 2,
    "osc_mod_dest": 4,
    "unison_mode": 2,
    "portament_auto_mode": 2,
    "effect_on_off": 2,
    "effect_type": 10,
    "delay_type": 3
}