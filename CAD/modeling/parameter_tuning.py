from CAD.config import Config
from CAD.utils import Utils

import numpy as np
import optuna
import re
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split


class HyperparameterTuner:
    def __init__(self, data, static_features, seed=999, n_trials=80):
        self.cfg = Config()
        self.utils = Utils()
        self.data = data
        self.static = static_features
        self.seed = seed
        self.n_trials = n_trials
        self.folds = [(2021, 2022), (2022, 2023), (2023, 2024)]
        self.params = self.cfg.getParameterValues()


    def _objective(self, trial):
        params = {
            **self.params,
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'random_state': self.seed
        }

        rmses = []
        for feat_year, tgt_year in self.folds:
            feats = self.utils._features_up_to(year=feat_year, data=self.data, static=self.static)
            X = self.data[feats]
            y = self.data[f"Attendance_Rate_{tgt_year}"]
            X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=self.seed)

            mdl = LGBMRegressor(**params)
            mdl.fit(
                X_tr,
                y_tr,
                categorical_feature=[c for c in self.static if c in feats],
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
            )
            preds = mdl.predict(X_val, num_iteration=mdl.best_iteration_)
            rmses.append(np.sqrt(np.mean((y_val - preds) ** 2)))

        return float(np.mean(rmses))


    def tune(self):
        import optuna

        study = optuna.create_study(direction="minimize")
        study.optimize(self._objective, n_trials=self.n_trials, show_progress_bar=False)
        return study.best_params
