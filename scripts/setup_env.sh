#!/usr/bin/env bash
set -euo pipefail
echo "Setting up bball-shot-analyzer environment..."

python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python: $python_version"

if [[ ! -d "venv" ]]; then
  python3 -m venv venv
  echo "✅ Virtual environment created"
fi
source venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt

mkdir -p data/{raw/{videos,poses},processed/{keypoints,features,sequences},annotations}
mkdir -p models/{bilstm} docs/{eda_report,evaluation} outputs

for d in data/raw data/processed/features data/processed/sequences data/annotations models docs/eda_report docs/evaluation outputs; do
  touch "$d/.gitkeep" 2>/dev/null || true
done

[[ ! -f ".env" ]] && cp .env.example .env && echo "✅ .env created"

echo ""
echo "✅ Setup complete!"
echo "Next: source venv/bin/activate && bash scripts/run_pipeline.sh"
