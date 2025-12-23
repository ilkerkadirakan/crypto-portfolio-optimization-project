"""Utilities for generating exhaustive portfolio combinations.

This module centralizes logic for loading configured assets and producing full
combination sets that can feed downstream optimization routines.

TODO:
    - Add streaming or chunked exports if future workflows cannot hold all
      combinations in memory at once.
"""

from __future__ import annotations

import pickle
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import yaml

DEFAULT_CONFIG_PATH = Path("configs/assets.yaml")
DEFAULT_OUTPUT_PATH = Path("data/meta/combinations.pkl")
DEFAULT_GROUP_SIZES: Tuple[int, ...] = (2, 3, 5)


def load_assets_from_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> List[str]:
    """Load asset identifiers from the canonical configuration file.

    Parameters
    ----------
    config_path : Path | str, optional
        Location of the YAML file describing the asset universe. Defaults to
        ``configs/assets.yaml``.

    Returns
    -------
    List[str]
        Ordered list of asset symbols suitable for combination generation.

    Raises
    ------
    FileNotFoundError
        If the configuration file cannot be located.
    ValueError
        If the file does not contain a parsable asset list.

    TODO:
        - Extend parsing to support remote configuration sources or alternate
          schema versions once needed by the pipeline.
    """

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Asset configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    raw_assets: Iterable[object]
    if isinstance(data, dict):
        for candidate_key in ("assets", "universe", "symbols"):
            if candidate_key in data:
                raw_assets = data[candidate_key]
                break
        else:
            raw_assets = data.get("assets")
    else:
        raw_assets = data

    if not isinstance(raw_assets, list):
        raise ValueError("Asset configuration must provide a list of assets.")

    assets: List[str] = []
    for item in raw_assets:
        if isinstance(item, str):
            assets.append(item)
        elif isinstance(item, dict):
            symbol = item.get("symbol")
            if not symbol:
                raise ValueError("Asset entry missing 'symbol' field.")
            assets.append(str(symbol))
        else:
            raise ValueError(f"Unsupported asset entry type: {type(item)}")

    if not assets:
        raise ValueError("No assets found in configuration file.")

    # Preserve input order while removing duplicates.
    return list(dict.fromkeys(assets))


def get_all_combinations(
    assets: Sequence[str],
    group_sizes: Sequence[int] = DEFAULT_GROUP_SIZES,
) -> Dict[str, List[Tuple[str, ...]]]:
    """Generate all combinations for the provided assets and group sizes.

    Parameters
    ----------
    assets : Sequence[str]
        Asset identifiers to be grouped.
    group_sizes : Sequence[int], optional
        Collection of target combination sizes. Defaults to ``(2, 3, 5)``.

    Returns
    -------
    Dict[str, List[Tuple[str, ...]]]
        Mapping from group size (string key) to the exhaustive list of ordered
        tuples representing each combination.

    Raises
    ------
    ValueError
        If no group sizes are supplied or requested sizes exceed asset count.

    TODO:
        - Add optional lazy generators or file-backed stores for extremely large
          universes to avoid materializing every combination in memory.
    """

    if not group_sizes:
        raise ValueError("At least one group size must be provided.")

    unique_assets = list(dict.fromkeys(assets))
    if not unique_assets:
        raise ValueError("Asset sequence must contain at least one identifier.")

    max_size = max(group_sizes)
    if max_size > len(unique_assets):
        raise ValueError(
            f"Requested combination size {max_size} exceeds asset count "
            f"{len(unique_assets)}."
        )

    combinations_by_size: Dict[str, List[Tuple[str, ...]]] = {}
    for size in group_sizes:
        if size < 1:
            raise ValueError(f"Combination size must be positive, received {size}.")
        if size > len(unique_assets):
            raise ValueError(
                f"Combination size {size} exceeds asset count {len(unique_assets)}."
            )
        combinations_by_size[str(size)] = list(combinations(unique_assets, size))

    return combinations_by_size


def _ensure_utf8_stdout() -> None:
    """Force UTF-8 capable stdout so Unicode status messages render reliably.

    TODO:
        - Evaluate centralizing encoding guards within a shared IO utility if
          additional modules need similar handling.
    """

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        # On interpreters lacking ``reconfigure`` (or when stdout is detached),
        # we silently rely on the default encoding.
        pass


def cache_combinations(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    group_sizes: Sequence[int] = DEFAULT_GROUP_SIZES,
) -> Dict[str, List[Tuple[str, ...]]]:
    """Generate and persist combinations, caching to disk if missing.

    Parameters
    ----------
    config_path : Path | str, optional
        Path to the asset universe configuration. Defaults to
        ``configs/assets.yaml``.
    output_path : Path | str, optional
        Destination pickle path used as on-disk cache. Defaults to
        ``data/meta/combinations.pkl``.
    group_sizes : Sequence[int], optional
        Combination sizes to materialize. Defaults to ``(2, 3, 5)``.

    Returns
    -------
    Dict[str, List[Tuple[str, ...]]]
        Exhaustive combinations grouped by size.

    TODO:
        - Consider persisting counts alongside combinations to allow lightweight
          reporting without loading large pickle files.
    """

    assets = load_assets_from_config(config_path)
    combinations_by_size = get_all_combinations(assets, group_sizes)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not destination.exists():
        with destination.open("wb") as handle:
            pickle.dump(combinations_by_size, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return combinations_by_size


def main() -> None:
    """Entry point for command-line execution.

    TODO:
        - Expand CLI arguments for custom group sizes or alternate configs if
          future workflows require more flexibility.
    """

    _ensure_utf8_stdout()
    combos = cache_combinations()
    for size_key, combo_list in combos.items():
        print(f"Group size {size_key}: {len(combo_list)} combinations")
    print("✅ Full combination generator ready")


__all__ = ["load_assets_from_config", "get_all_combinations", "cache_combinations"]


if __name__ == "__main__":
    main()
