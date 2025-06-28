import re
import numpy as np
import pandas as pd
import joblib
from CAD.config import Config


class Predictor:
    def __init__(self) -> None:
        self.cfg = Config()
        self.min_year, self.max_year = self.cfg.getMinMaxYear() 


    def predict(self) -> None:
        model = joblib.load(self.cfg.MODELS_DIR / "lgbm_regressor.pkl")
        df_wide = pd.read_parquet(self.cfg.PROCESSED_DATA_DIR / "FinalTrainingData.parquet")
        rolling_df = pd.read_csv(self.cfg.INTERIM_DATA_DIR / "RollingAvgStudents.csv")
        original_df = pd.read_csv(self.cfg.INTERIM_DATA_DIR / "Merged_Data.csv")

        feature_names = model.feature_name_
        df_pred = df_wide.copy()
        year_pat = re.compile(r"_(\d{4})$")

        for feat in feature_names:
            m = year_pat.search(feat)
            if m:
                yr = int(m.group(1))
                if yr > self.min_year:               
                    src = feat.replace(str(yr), str(yr + 1))
                    if src in df_wide.columns:
                        df_pred[feat] = df_wide[src]
                    else:
                        df_pred[feat] = np.nan

        cat_cols = [
            c for c in feature_names
            if isinstance(df_wide[c], pd.CategoricalDtype)
        ]
        cat_cols += df_pred.select_dtypes(include="object").columns.tolist()
        cat_cols = list(dict.fromkeys(cat_cols))      

        for col in cat_cols:
            df_pred[col] = df_pred[col].astype("category")

        preds = model.predict(df_pred[feature_names])
        model_ids = df_pred["STUDENT_ID"].tolist()
        model_pred = dict(zip(model_ids, preds))
        ra_pred = rolling_df.groupby("STUDENT_ID")["Attendance_Rate"].mean().to_dict()

        all_ids = set(model_ids) | set(ra_pred.keys())
        last_actual = (
            original_df.sort_values("SCHOOL_YEAR")
            .groupby("STUDENT_ID")
            .last()
            .reset_index()
        )
        pred_rows = last_actual[last_actual["STUDENT_ID"].isin(all_ids)].copy()
        pred_rows["SCHOOL_YEAR"] = self.max_year + 1 

        for att_col in [
            "Total_Days_Unexcused_Absent",
            "Total_Days_Enrolled",
            "Total_Days_Present",
        ]:
            if att_col in pred_rows.columns:
                pred_rows[att_col] = np.nan

        pred_rows["Predictions"] = pred_rows["STUDENT_ID"].map({**model_pred, **ra_pred})
        pred_rows["Predictions_District"] = np.nan
        pred_rows["Predictions_School"] = np.nan
        pred_rows["Predictions_Grade"] = np.nan

        for col in ["Predictions", "Predictions_District", "Predictions_School", "Predictions_Grade"]:
            if col not in original_df.columns:
                original_df[col] = np.nan

        df_long = pd.concat([original_df, pred_rows], ignore_index=True, sort=False)

        df_long["Predictions_District"] = df_long.groupby(["DISTRICT_CODE", "SCHOOL_YEAR"])["Predictions"].transform("mean")
        df_long["Predictions_School"] = df_long.groupby(["LOCATION_ID", "SCHOOL_YEAR"])["Predictions"].transform("mean")
        df_long["Predictions_Grade"] = df_long.groupby(["STUDENT_GRADE_LEVEL", "SCHOOL_YEAR"])["Predictions"].transform("mean")
        df_long = df_long.sort_values(by=['STUDENT_ID', 'SCHOOL_YEAR'])
        df_long['SCHOOL_YEAR'] = df_long['SCHOOL_YEAR'].str[:4]
        df_long['SCHOOL_YEAR'] = df_long['SCHOOL_YEAR'].fillna(2025).astype('int')

        out_path = self.cfg.PROCESSED_DATA_DIR / "Predictions.csv"
        df_long.to_csv(out_path, index=False)
        print(f"Predictions written → {out_path}")


pred = Predictor()
pred.predict()