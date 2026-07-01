#!/bin/bash

# --- CONFIGURATION ---
ENV_DIR="./my_pipeline_env"
REQUIREMENTS="requirements.txt"

# Define your independent commands here in an array
COMMANDS=(
  "python eval.py --begin 0 --end 180 --sim_index gntk --dataset mutag --kaggle 1 --interp_index learned --max_edges 12"
  "python eval.py --begin 0 --end 180 --sim_index vgae --dataset mutag --kaggle 1 --interp_index learned --max_edges 12"
  "python eval.py --begin 0 --end 180 --sim_index gntk --dataset mutag --kaggle 1 --interp_index learned --max_edges 10"
  "python eval.py --begin 0 --end 180 --sim_index vgae --dataset mutag --kaggle 1 --interp_index learned --max_edges 10"
  "python eval.py --begin 0 --end 180 --sim_index gntk --dataset mutag --kaggle 1 --interp_index learned --max_edges 8"
  "python eval.py --begin 0 --end 180 --sim_index vgae --dataset mutag --kaggle 1 --interp_index learned --max_edges 8"
  "python eval.py --begin 0 --end 180 --sim_index gntk --dataset mutag --kaggle 1 --interp_index learned --max_edges 6"
  "python eval.py --begin 0 --end 180 --sim_index vgae --dataset mutag --kaggle 1 --interp_index learned --max_edges 6"
)
# ---------------------

# Exit immediately if a critical setup step fails
set -e

# 1. Validate requirements file
if [ ! -f "$REQUIREMENTS" ]; then
    echo "❌ Error: Required file '$REQUIREMENTS' not found!"
    exit 1
fi

# 2. Setup and activate virtual environment
if [ ! -d "$ENV_DIR" ]; then
    echo "📦 Creating fresh virtual environment at $ENV_DIR..."
    python3 -m venv "$ENV_DIR"
fi

echo "🔄 Activating environment..."
source "$ENV_DIR/bin/activate"

echo "⚙️  Upgrading pip and installing requirements..."
pip install --upgrade pip
pip install -r "$REQUIREMENTS"

# 3. Execute independent commands
# Disable 'set -e' so one failing command does not crash the entire pipeline
set +e 

echo -e "\n🚀 Starting execution of independent commands...\n"

for i in "${!COMMANDS[@]}"; do
    CMD="${COMMANDS[$i]}"
    echo "--------------------------------------------------"
    echo "▶️  Executing Command $((i+1)): $CMD"
    echo "--------------------------------------------------"
    
    # Run the command
    eval "$CMD"
    STATUS=$?
    
    # Check execution status
    if [ $STATUS -eq 0 ]; then
        echo "✅ Command $((i+1)) succeeded."
    else
        echo "❌ Command $((i+1)) failed with exit code $STATUS."
    fi
done

# 4. Cleanup
echo -e "\n🧹 Deactivating environment..."
deactivate
echo "🎉 All tasks finished!"
