#!/bin/bash
set -e

# Run the first one sync, this will generate a fresh pickle
python -m backend.scripts.generate_ucache \
    asset-liability \
    --mode 0 \
    --perp-market-index 0

# The next ones will use the --use-snapshot flag, so they will reuse the pickle
# We can run all commands in parallel by adding & at the end

echo "Generating comprehensive price shock cache matrix..."
echo "Asset groups: ignore+stables, jlp+only"
echo "Scenarios: 5 (0.05 distortion), 10 (0.1 distortion)" 
echo "Pool IDs: all, 0, 1, 3"
echo "Total combinations: 16"

# =============================================================================
# ALL POOLS (no pool filter) - 4 combinations
# =============================================================================

# ignore+stables, 5 scenarios, 0.05 distortion, all pools
python -m backend.scripts.generate_ucache \
    --use-snapshot \
    price-shock \
    --asset-group "ignore+stables" \
    --oracle-distortion 0.05 \
    --n-scenarios 5 &

# ignore+stables, 10 scenarios, 0.1 distortion, all pools  
python -m backend.scripts.generate_ucache \
    --use-snapshot \
    price-shock \
    --asset-group "ignore+stables" \
    --oracle-distortion 0.1 \
    --n-scenarios 10 &

# jlp+only, 5 scenarios, 0.05 distortion, all pools
python -m backend.scripts.generate_ucache \
    --use-snapshot \
    price-shock \
    --asset-group "jlp+only" \
    --oracle-distortion 0.05 \
    --n-scenarios 5 &

# jlp+only, 10 scenarios, 0.1 distortion, all pools
python -m backend.scripts.generate_ucache \
    --use-snapshot \
    price-shock \
    --asset-group "jlp+only" \
    --oracle-distortion 0.1 \
    --n-scenarios 10 &

# =============================================================================
# MAIN POOL (pool_id = 0) - 4 combinations  
# =============================================================================

# ignore+stables, 5 scenarios, 0.05 distortion, pool 0
python -m backend.scripts.generate_ucache \
    --use-snapshot \
    price-shock \
    --asset-group "ignore+stables" \
    --oracle-distortion 0.05 \
    --n-scenarios 5 \
    --pool-id 0 &

# ignore+stables, 10 scenarios, 0.1 distortion, pool 0
python -m backend.scripts.generate_ucache \
    --use-snapshot \
    price-shock \
    --asset-group "ignore+stables" \
    --oracle-distortion 0.1 \
    --n-scenarios 10 \
    --pool-id 0 &

# jlp+only, 5 scenarios, 0.05 distortion, pool 0  
python -m backend.scripts.generate_ucache \
    --use-snapshot \
    price-shock \
    --asset-group "jlp+only" \
    --oracle-distortion 0.05 \
    --n-scenarios 5 \
    --pool-id 0 &

# jlp+only, 10 scenarios, 0.1 distortion, pool 0
python -m backend.scripts.generate_ucache \
    --use-snapshot \
    price-shock \
    --asset-group "jlp+only" \
    --oracle-distortion 0.1 \
    --n-scenarios 10 \
    --pool-id 0 &

# =============================================================================
# ISOLATED POOL 1 (pool_id = 1) - 4 combinations
# =============================================================================

# ignore+stables, 5 scenarios, 0.05 distortion, pool 1
python -m backend.scripts.generate_ucache \
    --use-snapshot \
    price-shock \
    --asset-group "ignore+stables" \
    --oracle-distortion 0.05 \
    --n-scenarios 5 \
    --pool-id 1 &

# ignore+stables, 10 scenarios, 0.1 distortion, pool 1
python -m backend.scripts.generate_ucache \
    --use-snapshot \
    price-shock \
    --asset-group "ignore+stables" \
    --oracle-distortion 0.1 \
    --n-scenarios 10 \
    --pool-id 1 &

# jlp+only, 5 scenarios, 0.05 distortion, pool 1
python -m backend.scripts.generate_ucache \
    --use-snapshot \
    price-shock \
    --asset-group "jlp+only" \
    --oracle-distortion 0.05 \
    --n-scenarios 5 \
    --pool-id 1 &

# jlp+only, 10 scenarios, 0.1 distortion, pool 1  
python -m backend.scripts.generate_ucache \
    --use-snapshot \
    price-shock \
    --asset-group "jlp+only" \
    --oracle-distortion 0.1 \
    --n-scenarios 10 \
    --pool-id 1 &

# =============================================================================
# ISOLATED POOL 3 (pool_id = 3) - 4 combinations
# =============================================================================

# ignore+stables, 5 scenarios, 0.05 distortion, pool 3
python -m backend.scripts.generate_ucache \
    --use-snapshot \
    price-shock \
    --asset-group "ignore+stables" \
    --oracle-distortion 0.05 \
    --n-scenarios 5 \
    --pool-id 3 &

# ignore+stables, 10 scenarios, 0.1 distortion, pool 3
python -m backend.scripts.generate_ucache \
    --use-snapshot \
    price-shock \
    --asset-group "ignore+stables" \
    --oracle-distortion 0.1 \
    --n-scenarios 10 \
    --pool-id 3 &

# jlp+only, 5 scenarios, 0.05 distortion, pool 3
python -m backend.scripts.generate_ucache \
    --use-snapshot \
    price-shock \
    --asset-group "jlp+only" \
    --oracle-distortion 0.05 \
    --n-scenarios 5 \
    --pool-id 3 &

# jlp+only, 10 scenarios, 0.1 distortion, pool 3
python -m backend.scripts.generate_ucache \
    --use-snapshot \
    price-shock \
    --asset-group "jlp+only" \
    --oracle-distortion 0.1 \
    --n-scenarios 10 \
    --pool-id 3 &

# =============================================================================
# Wait for all background processes to complete
# =============================================================================
echo "Waiting for all 16 price shock cache generation processes to complete..."
wait

echo "✅ All cache generation completed successfully!"

# Delete old pickles
cd pickles && ls -t | tail -n +4 | xargs rm -rf