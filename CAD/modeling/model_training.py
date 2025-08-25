from CAD.config import Config
from CAD.modeling.parameter_tuning import HyperparameterTuner
from CAD.utils import Utils

import joblib
import re
import numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split


class ModelTraining:
    def __init__(self, performParameterTuning: bool = False):
        self.cfg = Config()
        self.utils = Utils()
        self.seed = 999
        self.tune = performParameterTuning
        self.data = pd.read_parquet(self.cfg.PROCESSED_DATA_DIR / "FinalTrainingData.parquet")
        self.min_year, self.max_year = self.cfg.getMinMaxYear()
        
        yr_pat = re.compile(r"_(\d{4})$")
        self.static = [
            c
            for c in self.data.columns
            if not yr_pat.search(c)
            and c
            not in [
                "STUDENT_ID",
                "Attendance_Rate",
                "Total_Days_Present",
                "Total_Days_Enrolled",
                "Total_Days_Unexcused_Absent",
                "Is_Imputed",
            ]
        ]
        obj_cols = self.data.select_dtypes(include="object").columns.tolist()
        for col in obj_cols:
            self.data[col] = self.data[col].astype("category")


    def run(self):
        if self.tune:
            tuner = HyperparameterTuner(self.data, self.static, seed=self.seed, n_trials=80)
            best_params = tuner.tune()
        else:
            best_params = self.cfg.getParameterValues()

        feats = self.utils._features_up_to(year=self.max_year - 1, data=self.data, static=self.static)
        X = self.data[feats]
        y = self.data[f"Attendance_Rate_{self.max_year}"]

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=self.seed)
        mdl = LGBMRegressor(**{**best_params, **{"objective": "regression", "metric": "rmse", "random_state": self.seed}})
        mdl.fit(X_tr, y_tr, categorical_feature=[c for c in self.static if c in feats])
        rmse = np.sqrt(np.mean((y_te - mdl.predict(X_te)) ** 2))
        print(f"Final RMSE (hold-out): {rmse:.5f}")


        Path(self.cfg.MODELS_DIR).mkdir(exist_ok=True, parents=True)
        joblib.dump(mdl, self.cfg.MODELS_DIR / "lgbm_regressor.pkl")
        print(f"Model stored → {self.cfg.MODELS_DIR}/lgbm_regressor.pkl")
