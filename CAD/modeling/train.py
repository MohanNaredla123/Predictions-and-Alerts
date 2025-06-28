from CAD.modeling.feature_engineering import FeatureEngineering
from CAD.modeling.model_training import ModelTraining

import argparse
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna hyperparameter tuning (80 trials)",
    )
    args = parser.parse_args()

    FeatureEngineering().run()
    ModelTraining(performParameterTuning=args.tune).run()
