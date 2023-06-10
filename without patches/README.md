<h1>A Multidimensional State Space Layer for Spatial Transformers - Runs without Patches (isotropic)</h1>
This repository is based on the MEGA repository.

To run the CIFAR100 experiments, run the following command:
bash command here:

```bash
python train.py ${DATA} --seed 0 --ddp-backend c10d --find-unused-parameters -a mega_lra_cifar10 --task lra-image --input-type image --pixel-normalization 0.48 0.24 --encoder-layers 8 --chunk-size ${CHUNK} --activation-fn 'silu' --attention-activation-fn 'laplace' --norm-type 'batchnorm' --sen-rep-type 'mp' --encoder-normalize-before --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric --optimizer adam --lr 0.01 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 --dropout 0 --attention-dropout 0.0 --act-dropout 0.0 --weight-decay 0.02 --batch-size 50 --sentence-avg --update-freq 1 --max-update 180000 --lr-scheduler linear_decay --total-num-update 180000 --end-learning-rate 0 --warmup-updates 9000 --warmup-init-lr '1e-07' --keep-last-epochs 1 --required-batch-size-multiple 1 --save-dir ${SAVE}/2d-seed-0-N1-SSM2-real-2dir --wandb-project grid-search-cifar10 --log-format simple --log-interval 100 --num-workers 8 --ssm_2d True --dataset_percentage 1 --use_positional_encoding True --complex_ssm False --n_ssm 2 --n-dim 1 --directions_amount 2
```

Where ${DATA} is the path to the CIFAR100 dataset and ${SAVE} is the path to the directory where you want to save the
checkpoints.
{CHUNK} is the chunk size, which should be set to -1 for the full image.
