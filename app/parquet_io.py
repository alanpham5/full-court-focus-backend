from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_teams_parquet(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    last_err: BaseException | None = None

    try:
        return pd.read_parquet(path, engine="fastparquet")
    except ImportError:
        pass
    except Exception as e:
        last_err = e

    try:
        return pd.read_parquet(path, engine="pyarrow")
    except ImportError as e:
        last_err = e
    except OSError as e:
        last_err = e

    try:
        import pyarrow.parquet as pq

        return pq.ParquetFile(path).read().to_pandas()
    except ImportError:
        pass
    except OSError as e:
        last_err = e

    msg = (
        "Could not read teams Parquet (fastparquet + PyArrow). "
        "Install dependencies from requirements.txt or re-export the dataset."
    )
    raise OSError(msg) from last_err


def write_teams_parquet(df: pd.DataFrame, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_parquet(path, index=False, engine="fastparquet", compression="snappy")
        return
    except ImportError:
        pass
    except Exception:
        pass

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(df, preserve_index=False)
    kwargs: dict = {"compression": "snappy", "coerce_timestamps": "ms"}
    try:
        kwargs["write_page_index"] = False
        pq.write_table(table, path, **kwargs)
    except TypeError:
        pq.write_table(table, path, compression="snappy", coerce_timestamps="ms")
