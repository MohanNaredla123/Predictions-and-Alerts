import numpy as np
import pandas as pd
from CAD.config import Config


class FeatureEngineering:
    def __init__(self) -> None:
        self.cfg = Config()
        self.min_year, self.max_year = self.cfg.getMinMaxYear()


    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        for yr in range(self.min_year, self.max_year + 1):
            prev1_col = f"Att_Rate_Prev1_{yr}"
            prev2_col = f"Att_Rate_Prev2_{yr}"
            delta_col = f"Delta_Attendance_{yr}"
            grade_gender_col = f"Grade_X_Gender_{yr}"
            econ_grade_col = f"Econ_X_Grade_Band_{yr}"

            if f"Attendance_Rate_{yr-1}" in out.columns:
                out[prev1_col] = out[f"Attendance_Rate_{yr-1}"]
            if f"Attendance_Rate_{yr-2}" in out.columns:
                out[prev2_col] = out[f"Attendance_Rate_{yr-2}"]

            if (
                f"Attendance_Rate_{yr}" in out.columns
                and f"Attendance_Rate_{yr-1}" in out.columns
            ):
                out[delta_col] = (
                    out[f"Attendance_Rate_{yr}"] - out[f"Attendance_Rate_{yr-1}"]
                )

            grade_vals = out.get(f"STUDENT_GRADE_LEVEL_{yr}", pd.Series(-2, index=out.index))
            grade_str = grade_vals.astype("Int64").astype(str)
            gender_str = out.get("STUDENT_GENDER", pd.Series("", index=out.index)).astype(str)
            out[grade_gender_col] = grade_str + "_" + gender_str

            econ_code = out.get("ECONOMIC_CODE", pd.Series(np.nan, index=out.index))
            econ_str = econ_code.map({1: "F", 0: "N"}).fillna("N")
            grade_band = pd.Series("Elem", index=out.index)
            grade_band.loc[(grade_vals >= 6) & (grade_vals <= 8)] = "Middle"
            grade_band.loc[grade_vals >= 9] = "High"
            out[econ_grade_col] = econ_str + "_" + grade_band

        return out
    

    def run(self) -> None:
        wide_path = self.cfg.INTERIM_DATA_DIR / "TrainLong.csv"
        wide_df = pd.read_csv(wide_path)

        final_df = self._build_features(wide_df)

        final_df.to_parquet(
            self.cfg.PROCESSED_DATA_DIR / "FinalTrainingData.parquet", index=False
        )
        print("Feature engineering finished → FinalTrainingData.parquet")
