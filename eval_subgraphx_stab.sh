#!/bin/bash
# Stability-only SubgraphX evals (--eval_stab 1), kept separate from
# eval_subgraphx.sh (full interp+fidelity sweeps) since stability is ~10x
# more expensive per graph.
#
# mutag (188 graphs) is small enough to run in full. ba2motif, bamultishapes,
# and proteins run on a 200-graph coreset (coreset_<dataset>.npy) selected by
# select_coreset.py: stratified by (predicted class, confidence decile, size
# decile) with a farthest-point-sampling pick within each stratum.
#
# NOTE: this script is NOT executed automatically -- it only prints the
# planned commands. Run it manually (or comment out the dry-run guard below)
# when you have compute available.

COMMANDS=(
  # --- mutag: full dataset, no coreset ---
  "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 188 --sim_index gntk --dataset mutag --kaggle 1 --interp_index learned --max_nodes 12 --eval_stab 1"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 188 --sim_index vgae --dataset mutag --kaggle 1 --interp_index learned --max_nodes 12 --eval_stab 1"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 188 --sim_index gntk --dataset mutag --kaggle 1 --interp_index learned --max_nodes 10 --eval_stab 1"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 188 --sim_index vgae --dataset mutag --kaggle 1 --interp_index learned --max_nodes 10 --eval_stab 1"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 188 --sim_index gntk --dataset mutag --kaggle 1 --interp_index learned --max_nodes 8 --eval_stab 1"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 188 --sim_index vgae --dataset mutag --kaggle 1 --interp_index learned --max_nodes 8 --eval_stab 1"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 188 --sim_index gntk --dataset mutag --kaggle 1 --interp_index learned --max_nodes 6 --eval_stab 1"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --begin 0 --end 188 --sim_index vgae --dataset mutag --kaggle 1 --interp_index learned --max_nodes 6 --eval_stab 1"

  # --- ba2motif: 200-graph coreset ---
  "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_ba2motif.npy --sim_index gntk --dataset ba2motif --kaggle 1 --interp_index learned --max_nodes 12 --eval_stab 1"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_ba2motif.npy --sim_index vgae --dataset ba2motif --kaggle 1 --interp_index learned --max_nodes 12 --eval_stab 1"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_ba2motif.npy --sim_index gntk --dataset ba2motif --kaggle 1 --interp_index learned --max_nodes 10 --eval_stab 1"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_ba2motif.npy --sim_index vgae --dataset ba2motif --kaggle 1 --interp_index learned --max_nodes 10 --eval_stab 1"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_ba2motif.npy --sim_index gntk --dataset ba2motif --kaggle 1 --interp_index learned --max_nodes 8 --eval_stab 1"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_ba2motif.npy --sim_index vgae --dataset ba2motif --kaggle 1 --interp_index learned --max_nodes 8 --eval_stab 1"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_ba2motif.npy --sim_index gntk --dataset ba2motif --kaggle 1 --interp_index learned --max_nodes 6 --eval_stab 1"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_ba2motif.npy --sim_index vgae --dataset ba2motif --kaggle 1 --interp_index learned --max_nodes 6 --eval_stab 1"

  # --- bamultishapes: 200-graph coreset ---
  "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_bamultishapes.npy --sim_index gntk --dataset bamultishapes --kaggle 1 --interp_index learned --max_nodes 12 --eval_stab 1"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_bamultishapes.npy --sim_index vgae --dataset bamultishapes --kaggle 1 --interp_index learned --max_nodes 12 --eval_stab 1"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_bamultishapes.npy --sim_index gntk --dataset bamultishapes --kaggle 1 --interp_index learned --max_nodes 10 --eval_stab 1"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_bamultishapes.npy --sim_index vgae --dataset bamultishapes --kaggle 1 --interp_index learned --max_nodes 10 --eval_stab 1"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_bamultishapes.npy --sim_index gntk --dataset bamultishapes --kaggle 1 --interp_index learned --max_nodes 8 --eval_stab 1"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_bamultishapes.npy --sim_index vgae --dataset bamultishapes --kaggle 1 --interp_index learned --max_nodes 8 --eval_stab 1"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_bamultishapes.npy --sim_index gntk --dataset bamultishapes --kaggle 1 --interp_index learned --max_nodes 6 --eval_stab 1"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_bamultishapes.npy --sim_index vgae --dataset bamultishapes --kaggle 1 --interp_index learned --max_nodes 6 --eval_stab 1"

  # --- proteins: 200-graph coreset ---
  "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_proteins.npy --sim_index gntk --dataset proteins --kaggle 1 --interp_index learned --max_nodes 12 --eval_stab 1"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_proteins.npy --sim_index vgae --dataset proteins --kaggle 1 --interp_index learned --max_nodes 12 --eval_stab 1"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_proteins.npy --sim_index gntk --dataset proteins --kaggle 1 --interp_index learned --max_nodes 10 --eval_stab 1"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_proteins.npy --sim_index vgae --dataset proteins --kaggle 1 --interp_index learned --max_nodes 10 --eval_stab 1"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_proteins.npy --sim_index gntk --dataset proteins --kaggle 1 --interp_index learned --max_nodes 8 --eval_stab 1"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_proteins.npy --sim_index vgae --dataset proteins --kaggle 1 --interp_index learned --max_nodes 8 --eval_stab 1"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_proteins.npy --sim_index gntk --dataset proteins --kaggle 1 --interp_index learned --max_nodes 6 --eval_stab 1"
  "./subgraphx_env/bin/python3 eval_subgraphx.py --indices_path coreset_proteins.npy --sim_index vgae --dataset proteins --kaggle 1 --interp_index learned --max_nodes 6 --eval_stab 1"
)

set +e

echo -e "\n📋 eval_subgraphx_stab.sh: ${#COMMANDS[@]} planned stability commands (dry run, nothing executed)\n"

for i in "${!COMMANDS[@]}"; do
    echo "$((i+1)). ${COMMANDS[$i]}"
done

echo -e "\nThis script only prints the plan. To actually run a command, copy it out"
echo "and execute it directly, or remove the 'exit 0' guard below."

exit 0

# --- execution loop (disabled by the exit 0 above) ---
for i in "${!COMMANDS[@]}"; do
    CMD="${COMMANDS[$i]}"
    echo "--------------------------------------------------"
    echo "▶️  Executing Command $((i+1)): $CMD"
    echo "--------------------------------------------------"

    eval "$CMD"
    STATUS=$?

    if [ $STATUS -eq 0 ]; then
        echo "✅ Command $((i+1)) succeeded."
    else
        echo "❌ Command $((i+1)) failed with exit code $STATUS."
    fi
done
