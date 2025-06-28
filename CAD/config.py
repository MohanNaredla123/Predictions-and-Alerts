from pathlib import Path
from typing import Tuple
import json
from typing import Dict

from dotenv import load_dotenv

load_dotenv()

class Config:
    def __init__(self) -> None:
        self.PROJ_ROOT = Path(__file__).resolve().parents[1]
        self.DATA_DIR = self.PROJ_ROOT / "data"
        self.RAW_DATA_DIR = self.DATA_DIR / "raw"
        self.INTERIM_DATA_DIR = self.DATA_DIR / "interim"
        self.PROCESSED_DATA_DIR = self.DATA_DIR / "processed"
        self.EXTERNAL_DATA_DIR = self.DATA_DIR / "external"
        self.MODELS_DIR = self.PROJ_ROOT / "models"
        self.REPORTS_DIR = self.PROJ_ROOT / "reports"
        self.FIGURES_DIR = self.REPORTS_DIR / "figures"
        self.CONFIG_FILE = self.PROJ_ROOT / "config.json"

    def getParameterValues(self) -> Dict:
        try:
            with open(self.CONFIG_FILE, 'r') as f:
                config_content = json.load(f)
        except Exception as e:
            print(f'Cannot load the config file: {e}')

        return config_content.get('params', {})
    
    def getMinMaxYear(self) -> Tuple[int, int]:
        files = sorted([file.name for file in self.RAW_DATA_DIR.rglob('*')])

        return ( int(str(files[0])[:4]), int(str(files[-1])[:4]) )

