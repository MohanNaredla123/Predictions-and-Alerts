from CAD.config import Config

import re
import numpy as np
import pandas as pd
import joblib
from pandas.api.types import CategoricalDtype


class Predictor:
    def __init__(self) -> None:
        self.cfg = Config()
        self.min_year, self.max_year = self.cfg.getMinMaxYear() 


    def build_student_ra(self, ra: pd.DataFrame) -> dict:                       
        stu_avg = ra.groupby("STUDENT_ID")["Attendance_Rate"].mean()
        dist_avg = ra.groupby("DISTRICT_CODE")["Attendance_Rate"].mean()
        schl_avg = ra.groupby("LOCATION_ID")["Attendance_Rate"].mean()

        latest_grade = (
            ra.sort_values("SCHOOL_YEAR")
            .groupby("STUDENT_ID").tail(1)
            .set_index("STUDENT_ID")["STUDENT_GRADE_LEVEL"]
        )
        grade_avg = ra.groupby("STUDENT_GRADE_LEVEL")["Attendance_Rate"].mean()

        ra_dict = {}
        for sid, s_mean in stu_avg.items():
            pieces = [
                s_mean,
                dist_avg.get(ra.loc[ra["STUDENT_ID"] == sid, "DISTRICT_CODE"].iloc[0], np.nan),
                schl_avg.get(ra.loc[ra["STUDENT_ID"] == sid, "LOCATION_ID"].iloc[0], np.nan),
                grade_avg.get(latest_grade.get(sid, None), np.nan),
            ]
            vals = [v for v in pieces if pd.notna(v)]
            ra_dict[sid] = np.mean(vals) if vals else np.nan
        return ra_dict



    def predict(self) -> None:
        model = joblib.load(self.cfg.MODELS_DIR / "lgbm_regressor.pkl")
        df_wide = pd.read_parquet(self.cfg.PROCESSED_DATA_DIR / "FinalTrainingData.parquet")
        rolling_df = pd.read_csv(self.cfg.INTERIM_DATA_DIR / "RollingAvgStudents.csv")
        original_df = pd.read_csv(self.cfg.INTERIM_DATA_DIR / "TrainLong.csv")

        feature_names = model.feature_name_
        df_pred = df_wide.copy()
        year_pat = re.compile(r"_(\d{4})$")

        for feat in feature_names:
            m = year_pat.search(feat)
            if m:
                yr = int(m.group(1))
                if self.min_year <= yr < self.max_year:               
                    src = feat.replace(str(yr), str(yr + 1))
                    if src in df_wide.columns:
                        df_pred[feat] = df_wide[src]
                    else:
                        df_pred[feat] = np.nan

        cat_cols = [
            c for c in feature_names
            if isinstance(df_wide[c].dtype, CategoricalDtype)
        ]
        cat_cols += df_pred.select_dtypes(include="object").columns.tolist()
        cat_cols = list(dict.fromkeys(cat_cols))      

        for col in cat_cols:
            df_pred[col] = df_pred[col].astype("category")

        model_scores = model.predict(df_pred[feature_names])
        model_pred = dict(zip(df_pred["STUDENT_ID"], model_scores))
        ra_pred = self.build_student_ra(rolling_df) 

        all_pred = model_pred.copy()
        all_pred.update(ra_pred)

        last_actual = (
            original_df.sort_values("SCHOOL_YEAR")
            .groupby("STUDENT_ID")
            .last()
            .reset_index()
        )
        pred_rows = last_actual[last_actual["STUDENT_ID"].isin(all_pred.keys())].copy()
        pred_rows["SCHOOL_YEAR"] = self.max_year + 1 

        covered_ids = set(pred_rows["STUDENT_ID"])
        ra_last = (
            rolling_df.sort_values("SCHOOL_YEAR")       
                    .groupby("STUDENT_ID").last()       
                    .reset_index()
        )

        ra_rows = ra_last[ra_last["STUDENT_ID"].isin(ra_pred.keys() - covered_ids)].copy()
        ra_rows["SCHOOL_YEAR"] = self.max_year + 1

        for col in original_df.columns.difference(ra_rows.columns):
            ra_rows[col] = np.nan

        ra_rows["Predictions"] = ra_rows["STUDENT_ID"].map(ra_pred)
        ra_rows["Predictions_District"] = np.nan
        ra_rows["Predictions_School"] = np.nan
        ra_rows["Predictions_Grade"] = np.nan

        pred_rows = pd.concat([pred_rows, ra_rows], ignore_index=True)


        for att_col in [
            "Total_Days_Unexcused_Absent",
            "Total_Days_Enrolled",
            "Total_Days_Present",
        ]:
            if att_col in pred_rows.columns:
                pred_rows[att_col] = np.nan

        pred_rows["Predictions"] = pred_rows["STUDENT_ID"].map(all_pred)
        pred_rows["Predictions_District"] = np.nan
        pred_rows["Predictions_School"] = np.nan
        pred_rows["Predictions_Grade"] = np.nan

        for col in ["Predictions", "Predictions_District", "Predictions_School", "Predictions_Grade"]:
            if col not in original_df.columns:
                original_df[col] = np.nan

        ra_hist = rolling_df[rolling_df["SCHOOL_YEAR"] <= self.max_year].copy()
        for col in original_df.columns.difference(ra_hist.columns):
            ra_hist[col] = np.nan
        for col in ra_hist.columns.difference(original_df.columns):
            original_df[col] = np.nan       

        for col in ["Predictions", "Predictions_District",
                    "Predictions_School", "Predictions_Grade"]:
            ra_hist[col] = np.nan

        df_long = pd.concat(
            [original_df, ra_hist, pred_rows],
            ignore_index=True,
            sort=False
        )

        df_long["Predictions_District"] = (
            df_long.groupby("DISTRICT_CODE")["Predictions"].transform("mean")
        )
        df_long["Predictions_School"] = (
            df_long.groupby(["DISTRICT_CODE", "LOCATION_ID"])["Predictions"]
                .transform("mean")
        )
        df_long["Predictions_Grade"] = (
            df_long.groupby(["DISTRICT_CODE", "LOCATION_ID", "STUDENT_GRADE_LEVEL"])
                ["Predictions"].transform("mean")
        )

        df_long = df_long.sort_values(["STUDENT_ID", "SCHOOL_YEAR"])


        out_path = self.cfg.PROCESSED_DATA_DIR / "Predictions.parquet"
        df_long.to_parquet(out_path, index=False)
        print(f"Predictions written → {out_path}")


        pred_cols = ['Predictions', 'Predictions_District', 'Predictions_School', 'Predictions_Grade']
        pred_df = df_long[df_long['SCHOOL_YEAR'] == 2025].loc[:, [*pred_cols, 'STUDENT_ID']]

        hist_cols = [col for col in df_long.columns if col not in pred_cols]
        hist_df = df_long[~(df_long['SCHOOL_YEAR'] == 2025)].loc[:, hist_cols]

        alerts = pd.merge(left=hist_df, right=pred_df, on='STUDENT_ID', how='left')
        alerts_out_path = self.cfg.PROCESSED_DATA_DIR / 'Alerts.parquet'
        alerts.to_parquet(alerts_out_path, index=False)


pred = Predictor()
pred.predict()