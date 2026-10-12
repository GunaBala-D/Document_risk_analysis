#!/bin/sh
# One-off job (docker compose --profile bootstrap run --rm bootstrap):
# generate the SYNTHETIC corpus, run it through the extraction pipeline, train + register the
# classifier and anomaly detector in MLflow, write the reports. Takes ~20-30 minutes on a laptop.
set -e

N_CONTRACTS="${N_CONTRACTS:-3000}"
WORKERS="${WORKERS:-4}"

echo "== 1/5 generate synthetic corpus (${N_CONTRACTS} contracts)"
python -m scripts.generate_dataset --n-contracts "$N_CONTRACTS" --n-vendors 200 --seed 42
echo "== 2/5 extraction pipeline + features"
python -m training.build_dataset --workers "$WORKERS"
echo "== 3/5 EDA + label diagnostic"
python -m training.eda
python -m training.diagnose_labels
echo "== 4/5 train classifier (experiments, tuning, quality gate, registry)"
python -m training.train --note "docker bootstrap"
echo "== 5/5 train anomaly detector"
python -m training.train_anomaly
echo "Bootstrap complete. Restart the api service to load the models: docker compose restart api"
