import re
import pandas as pd
from typing import List

class Utils:
    def _features_up_to(self, year: int, data: pd.DataFrame, static: List) -> List:
        pat = re.compile(r"_(\d{4})$")
        cols = []
        for col in data.columns:
            m = pat.search(col)
            if m:
                if int(m.group(1)) <= year:
                    cols.append(col)
            else:
                cols.append(col)
        return cols