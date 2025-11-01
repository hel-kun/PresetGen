import os, re
import tqdm, logging
from datasets import load_dataset
from torch.utils.data import Dataset
from dotenv import load_dotenv
from utils.types import *
from utils.param import *

class Synth1Dataset(Dataset):
    def __init__(self, logger: logging.Logger = None):
        load_dotenv()
        TOKEN = os.getenv("HF_TOKEN")
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("Loading Synth1PresetDataset...")
        self.base_data = load_dataset("hel-kun/Synth1PresetDataset", token=TOKEN, trust_remote_code=True, version="1.0.2")
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
            # メタデータ行はスキップ
            if not re.match(r'^\d+,\d+$', line.strip()): continue
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
        params_batch = []
        for item in batch:
            label = item["label"]
            text = label["text"]
            texts.append(text)
            preset = item["params"]
            param_dict = {
                "categorical": {k: v for k, v in preset.categorical_param.__dict__.items()},
                "continuius": {k: v for k, v in preset.continuius_param.__dict__.items()},
                "misc": {k: v for k, v in preset.misc_param.__dict__.items()}
            }
            params_batch.append(param_dict)
  
        return texts, params_batch