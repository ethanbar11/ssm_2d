import meerkat as mk
import numpy as np
import pandas as pd
import terra
from functools import partial
import hashlib
import math


@terra.Task
def balance_dp(
    dp: mk.DataPanel,
    target_col: str = "target",
):

    targets = dp[target_col].data
    print(f"Class balance: {targets.mean():.3f}")
    num_pos = (targets == 1).sum()
    num_neg = (targets == 0).sum()

    new_indices = np.random.choice(
        max(num_neg, num_pos), size=min(num_neg, num_pos), replace=False
    )
    minority_class = num_neg > num_pos
    dp = dp.lz[targets == minority_class].append(
        dp.lz[targets == (not minority_class)].lz[new_indices]
    )

    new_targets = dp[target_col].data
    print(f"New class balance: {new_targets.mean():.3f}")

    return dp


@terra.Task
def split_dp(
    dp: mk.DataPanel,
    split_on: str,
    train_frac: float = 0.7,
    valid_frac: float = 0.1,
    test_frac: float = 0.2,
    other_splits: dict = None,
    salt: str = "",
):
    dp = dp.view()
    other_splits = {} if other_splits is None else other_splits
    splits = {
        "train": train_frac,
        "valid": valid_frac,
        "test": test_frac,
        **other_splits,
    }

    if not math.isclose(sum(splits.values()), 1):
        raise ValueError("Split fractions must sum to 1.")

    dp["split_hash"] = dp[split_on].apply(partial(hash_for_split, salt=salt))
    start = 0
    split_column = pd.Series(["unassigned"] * len(dp))
    for split, frac in splits.items():
        end = start + frac
        split_column[
            ((start < dp["split_hash"]) & (dp["split_hash"] <= end)).data
        ] = split
        start = end

    # need to drop duplicates, since split_on might not be unique
    df = pd.DataFrame({split_on: dp[split_on], "split": split_column}).drop_duplicates()
    return mk.DataPanel.from_pandas(df)


def hash_for_split(example_id: str, salt=""):
    GRANULARITY = 100000
    hashed = hashlib.sha256((str(example_id) + salt).encode())
    hashed = int(hashed.hexdigest().encode(), 16) % GRANULARITY + 1
    return hashed / float(GRANULARITY)


def merge_in_split(dp: mk.DataPanel, split_dp: mk.DataPanel):
    split_dp.columns
    if "split" in dp:
        dp.remove_column("split")
    split_on = [col for col in split_dp.columns if (col != "split") and col != "index"]
    assert len(split_on) == 1
    split_on = split_on[0]

    if split_dp[split_on].duplicated().any():
        # convert the datapanel to one row per split_on id
        df = split_dp[[split_on, "split"]].to_pandas()
        gb = df.groupby(split_on)

        # cannot have multiple splits per `split_on` id
        assert (gb["split"].nunique() == 1).all()
        split_dp = mk.DataPanel.from_pandas(gb["split"].first().reset_index())

    return dp.merge(split_dp, on=split_on)
