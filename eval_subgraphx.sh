#!/bin/bash
# Run eval_subgraphx.py using the dedicated subgraphx_env (Python 3.12 + dgl).
# See requirements_subgraphx.txt for the pinned dependency set.

COMMANDS=(
  "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 180 --sim_index gntk --dataset mutag --kaggle 1 --interp_index learned --max_nodes 12 --eval_stab 0"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 180 --sim_index vgae --dataset mutag --kaggle 1 --interp_index learned --max_nodes 12 --eval_stab 0"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 180 --sim_index gntk --dataset mutag --kaggle 1 --interp_index learned --max_nodes 10 --eval_stab 0"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 180 --sim_index vgae --dataset mutag --kaggle 1 --interp_index learned --max_nodes 10 --eval_stab 0"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 180 --sim_index gntk --dataset mutag --kaggle 1 --interp_index learned --max_nodes 8 --eval_stab 0"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 180 --sim_index vgae --dataset mutag --kaggle 1 --interp_index learned --max_nodes 8 --eval_stab 0"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 180 --sim_index gntk --dataset mutag --kaggle 1 --interp_index learned --max_nodes 6 --eval_stab 0"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 180 --sim_index vgae --dataset mutag --kaggle 1 --interp_index learned --max_nodes 6 --eval_stab 0"
)

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



