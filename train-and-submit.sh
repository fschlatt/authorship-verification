for dir in logistic naive-bayes svc; do
    cd $dir
    echo "Training $dir"
    python3 train.py
    image=$(head -n 1 Dockerfile | cut -d ' ' -f 5)
    echo "Building $image"
    docker build -t $image .
    if [ -d "tira-output" ]; then
        rm -rf tira-output
    fi
    tira-run \
        --input-dataset generative-ai-authorship-verification-panclef-2024/pan24-generative-authorship-tiny-smoke-20240417-training \
        --image $image \
        --output-directory ./tira-output
    if [ -f "tira-output/predictions.jsonl" ]; then
        echo "Uploading $dir"
        rm -rf tira-output
        tira-run \
        --input-dataset generative-ai-authorship-verification-panclef-2024/pan24-generative-authorship-tiny-smoke-20240417-training \
        --image $image \
        --output-directory ./tira-output \
        --push true
    else
        echo "No predictions.jsonl found in $dir"
    fi
    cd ..
done