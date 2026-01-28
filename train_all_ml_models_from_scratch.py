"""
Полное обучение ВСЕХ ML моделей с нуля для всех символов.

Создаёт полный комплект моделей для каждого символа:

- 15m‑only:
    - rf_SYMBOL_15_15m.pkl
    - xgb_SYMBOL_15_15m.pkl
    - ensemble_SYMBOL_15_15m.pkl
    - triple_ensemble_SYMBOL_15_15m.pkl
    - quad_ensemble_SYMBOL_15_15m.pkl

- MTF (15m + 1h + 4h):
    - rf_SYMBOL_15_mtf.pkl
    - xgb_SYMBOL_15_mtf.pkl
    - ensemble_SYMBOL_15_mtf.pkl
    - triple_ensemble_SYMBOL_15_mtf.pkl
    - quad_ensemble_SYMBOL_15_mtf.pkl

(LSTM модели создаются внутри QuadEnsemble обучения автоматически.)

Использование:
    python train_all_ml_models_from_scratch.py

Опции:
    --symbols BTCUSDT,ETHUSDT,SOLUSDT
    --days 180
    --interval 15m
    --modes 15m,mtf        # какие режимы тренировать (по умолчанию оба)
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List
import argparse


PROJECT_ROOT = Path(__file__).parent


def run_cmd(cmd: List[str], ml_mtf_enabled: bool) -> int:
    """Запускает команду с нужным флагом ML_MTF_ENABLED в окружении."""
    env = os.environ.copy()
    env["ML_MTF_ENABLED"] = "1" if ml_mtf_enabled else "0"
    print("\n" + "=" * 80)
    print(f"[train_all] Running: {' '.join(cmd)} (ML_MTF_ENABLED={env['ML_MTF_ENABLED']})")
    print("=" * 80)
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    return proc.returncode


def train_for_symbol(symbol: str, days: int, interval: str, modes: List[str]) -> None:
    """Обучает все типы моделей для одного символа в указанных режимах."""
    base_cmd_python = [sys.executable]

    for mode in modes:
        ml_mtf_enabled = mode.lower() == "mtf"
        mode_label = "MTF" if ml_mtf_enabled else "15m-only"
        print("\n" + "#" * 80)
        print(f"[train_all] SYMBOL: {symbol} | MODE: {mode_label}")
        print("#" * 80)

        # 1) Базовые модели + ensemble/triple_ensemble через retrain_ml_optimized.py
        cmd_retrain_opt = base_cmd_python + [
            "retrain_ml_optimized.py",
            "--symbol",
            symbol,
        ]
        rc = run_cmd(cmd_retrain_opt, ml_mtf_enabled=ml_mtf_enabled)
        if rc != 0:
            print(f"[train_all] WARNING: retrain_ml_optimized failed for {symbol} ({mode_label}), return code {rc}")

        # 2) QuadEnsemble (RF+XGB+LGB+LSTM)
        cmd_quad = base_cmd_python + [
            "train_quad_ensemble.py",
            "--symbol",
            symbol,
            "--days",
            str(days),
            "--interval",
            interval,
        ]
        rc = run_cmd(cmd_quad, ml_mtf_enabled=ml_mtf_enabled)
        if rc != 0:
            print(f"[train_all] WARNING: train_quad_ensemble failed for {symbol} ({mode_label}), return code {rc}")


def main():
    parser = argparse.ArgumentParser(description="Train all ML models from scratch for all symbols")
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTCUSDT,ETHUSDT,SOLUSDT",
        help="Comma-separated list of symbols (default: BTCUSDT,ETHUSDT,SOLUSDT)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="Days of history to use for QuadEnsemble (default: 180)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="15m",
        help="Base timeframe (default: 15m)",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="15m,mtf",
        help="Which modes to train: 15m,mtf or only one of them (default: 15m,mtf)",
    )

    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    modes = [m.strip().lower() for m in args.modes.split(",") if m.strip()] or ["15m", "mtf"]

    print("=" * 80)
    print("🚀 FULL ML MODELS TRAINING FROM SCRATCH")
    print("=" * 80)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Modes: {', '.join(modes)} (15m-only / MTF)")
    print(f"Days for QuadEnsemble: {args.days}")
    print(f"Base interval: {args.interval}")
    print("=" * 80)

    for symbol in symbols:
        train_for_symbol(symbol, days=args.days, interval=args.interval, modes=modes)

    print("\n✅ Training of all ML models completed (check logs above for any warnings).")


if __name__ == "__main__":
    main()

