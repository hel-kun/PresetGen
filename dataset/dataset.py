import os, re
import tqdm, logging
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import Dataset
from dotenv import load_dotenv
from utils.types import *
from utils.param import *

class Synth1Dataset(Dataset):
    def __init__(self, logger: logging.Logger = None, embed_dim: int = 512):
        load_dotenv()
        TOKEN = os.getenv("HF_TOKEN")
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("Loading Synth1PresetDataset...")
        self.base_data = load_dataset("hel-kun/Synth1PresetDataset", token=TOKEN, trust_remote_code=True, version="1.0.2")
        self.embed_dim = embed_dim
        self.dataset = self.preprocess()

    def preprocess(self):
        dataset = {
            'train': [],
            'validation': [],
            'test': []
        }
        for split in ['train', 'validation', 'test']:
            bar = tqdm.tqdm(total=len(self.base_data[split]), desc=f"Preprocessing {split} data")
            for item in self.base_data[split]:
                preset = self.create_params_dict(item['preset'])
                dataset[split].append({
                    'preset': preset,
                    'label': item['label']
                })
                bar.update(1)
            bar.close()
        return dataset
    
    def create_params_dict(self, item) -> Synth1Preset:
        categorical_params = {}
        continuius_params = {}
        misc_params = {}

        lines = item.strip().split('\n')
        for line in lines:
            if line.strip().endswith('.'):
                line = line.strip()[:-1]
            if not re.match(r'^-?\d+,-?\d+$', line.strip()): continue
            param_id, value = map(int, line.split(','))
            param_name = PARAM_ID_TO_NAME[param_id] if param_id in PARAM_ID_TO_NAME else None
            if param_name in CATEGORICAL_PARAM_NAMES:
                categorical_params[param_name] = value
            elif param_name in CONTINUOUS_PARAM_NAMES:
                continuius_params[param_name] = value
            elif param_name in MISC_PARAM_NAMES:
                misc_params[param_name] = value
        
        for name, default in CATEGORICAL_DEFAULTS.items():
            if name not in categorical_params:
                self.logger.warning(f"Categorical param {name} missing in preset, setting to default value {default}.")
                categorical_params[name] = default
        for name, default in CONTINUOUS_DEFAULTS.items():
            if name not in continuius_params:
                continuius_params[name] = default
                self.logger.warning(f"Continuous param {name} missing in preset, setting to default value {default}.")
        for name, default in MISC_DEFAULTS.items():
            if name not in misc_params:
                misc_params[name] = default
                self.logger.warning(f"Misc param {name} missing in preset, setting to default value {default}.")
    
        return Synth1Preset(
            categorical_param = CategoricalParam(**categorical_params),
            continuius_param = ContinuiusParam(**continuius_params),
            misc_param = MiscParam(**misc_params)
        )

    def __len__(self) -> int:
        return len(self.base_data)

    def __getitem__(self, idx: int):
        return self.dataset['train'][idx]
    
    def collate_fn(self, batch):
        texts = []
        categ_params = {}
        cont_params = {}
        misc_params = {}

        # param_nameの取得
        for name in CATEGORICAL_PARAM_NAMES:
            categ_params[name] = []
        for name in CONTINUOUS_PARAM_NAMES:
            cont_params[name] = []
        for name in MISC_PARAM_NAMES:
            misc_params[name] = []

        for item in batch:
            label = item["label"]
            text = label["text"]
            texts.append(text)
            preset = item["preset"]
            for name in CATEGORICAL_PARAM_NAMES:
                categ_params[name].append(getattr(preset.categorical_param, name))
            for name in CONTINUOUS_PARAM_NAMES:
                cont_params[name].append(getattr(preset.continuius_param, name))
            for name in MISC_PARAM_NAMES:
                misc_params[name].append(getattr(preset.misc_param, name))

        for name in CATEGORICAL_PARAM_NAMES:
            categ_params[name] = torch.tensor(categ_params[name], dtype=torch.long)
        for name in CONTINUOUS_PARAM_NAMES:
            # 連続値パラメータを0~1の範囲に正規化
            cont_params[name] = torch.tensor(cont_params[name], dtype=torch.float) / 127.0
        for name in MISC_PARAM_NAMES:
            misc_params[name] = torch.tensor(misc_params[name], dtype=torch.float) / 127.0

        batch_size = len(batch)

        categ_embed = torch.zeros(batch_size, 1, self.embed_dim)
        embed_idx = 0
        for name in list(CATEGORICAL_PARAM_NAMES)[:min(5, len(CATEGORICAL_PARAM_NAMES))]:
            if embed_idx + 5 <= self.embed_dim:
                for i in range(batch_size):
                    val = categ_params[name][i].item()
                    if val < 5:  # 5カテゴリまで対応
                        categ_embed[i, 0, embed_idx + val] = 1.0
                embed_idx += 5

        cont_embed = torch.zeros(batch_size, 1, self.embed_dim)
        embed_idx = 0
        for name in list(CONTINUOUS_PARAM_NAMES)[:min(self.embed_dim, len(CONTINUOUS_PARAM_NAMES))]:
            if embed_idx < self.embed_dim:
                cont_embed[:, 0, embed_idx] = cont_params[name].mean()  # バッチの平均値を使用
                embed_idx += 1

        tensor_batch = {
            'categ': categ_embed,
            'cont': cont_embed,
        }
        params_batch = {
            'categ': categ_params,
            'cont': cont_params,
        }
        return texts, tensor_batch, params_batch
                                                     