# Slurm sbatch Examples

## Sol Cluster (A100)
```bash
sbatch \
    --chdir="/path/to/your/Pytorch/directory" \
    --job-name="sol-job" \
    --output="slurm-%j-%x.out" \
    -p public -q public \
    --time="7-00:00:00" \
    --gres="gpu:a100:2" \
    --constraint="a100_80" \
    --mem=128G \
    --cpus-per-task=10 \
    --wrap="bash run-example.sh"
```

## Phoenix Cluster (V100)
```bash
sbatch \
    --chdir="/path/to/your/Pytorch/directory" \
    --job-name="phoenix-job" \
    --output="slurm-%j-%x.out" \
    -p public -q public \
    --time="7-00:00:00" \
    --gres="gpu:v100:4" \
    --constraint="v100_32" \
    --mem=128G \
    --cpus-per-task=12 \
    --wrap="bash run-example.sh"
```
