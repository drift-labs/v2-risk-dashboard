#!/bin/bash
set -e


# Run the first one sync, this will generate a fresh pickle
python -m backend.scripts.generate_ucache \
    asset-liability \
    --mode 0 \
    --perp-market-index 0


# The next ones will use the --use-snapshot flag, so they will reuse the pickle
# We can run all commands in parallel by adding & at the end
python -m backend.scripts.generate_ucache \
    --use-snapshot \
    price-shock \
    --asset-group "ignore+stables" \
    --oracle-distortion 0.05 \
    --n-scenarios 5 &

python -m backend.scripts.generate_ucache \
    --use-snapshot \
    price-shock \
    --asset-group "jlp+only" \
    --oracle-distortion 0.05 \
    --n-scenarios 5 &

python -m backend.scripts.generate_ucache \
    --use-snapshot \
    price-shock \
    --asset-group "jlp+only" \
    --oracle-distortion 0.1 \
    --n-scenarios 10 &


python -m backend.scripts.generate_ucache \
    --use-snapshot \
    price-shock \
    --asset-group "ignore+stables" \
    --oracle-distortion 0.1 \
    --n-scenarios 10 &

# Wait for all background processes to complete
wait

# Delete old pickles
cd pickles && ls -t | tail -n +4 | xargs rm -rf