import pandas as pd


@pd.api.extensions.register_dataframe_accessor("ta")
class TechnicalAnalysisAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def bbands(self, length=5, std=2):
        close = self._obj["Close"]
        middle = close.rolling(window=length).mean()
        deviation = close.rolling(window=length).std(ddof=0)
        upper = middle + std * deviation
        lower = middle - std * deviation

        suffix = f"{length}_{float(std):.1f}_{float(std):.1f}"
        return pd.DataFrame(
            {
                f"BBL_{suffix}": lower,
                f"BBM_{suffix}": middle,
                f"BBU_{suffix}": upper,
            },
            index=self._obj.index,
        )
