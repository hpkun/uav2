"""Dedicated MAPPO pursuit entry using the frozen PPO implementation."""
from pathlib import Path
import sys
import argparse
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from algorithm.train_mappo import main

if __name__ == "__main__":
    if "--env-config" not in sys.argv:
        sys.argv.extend(["--env-config", "configs/pursuit_environment.yaml"])
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--env-config")
    args, _ = parser.parse_known_args()
    path = Path(args.env_config)
    if not path.is_absolute():
        path = ROOT / path
    if yaml.safe_load(path.read_text(encoding="utf-8")).get("task_type") != "pursuit":
        raise RuntimeError("train_pursuit requires pursuit task configuration")
    main()
