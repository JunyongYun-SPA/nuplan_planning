TRAIN_EPOCHS=100
TRAIN_LR=5e-4
TRAIN_T_MAX=64
TRAIN_ETA_MIN=0.0
TRAIN_LR_MILESTONES=[50,75]
TRAIN_LR_DECAY=0.1
BATCH_SIZE=32
SEED=0

JOB_NAME=training_unimodal_hivt_model_goal_v2
# CACHE_PATH=/home/ssd/dataset/nuplan/cache/training_hivt_model
CACHE_PATH=/home/workspace/dataset/cache/hivt_50m_goal
USE_CACHE_WITHOUT_DATASET=True

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_training.py \
seed=$SEED \
py_func=train \
+training=training_unimodal_hivt_model_goal_v2 \
job_name=$JOB_NAME \
scenario_builder=nuplan \
worker=ray_distributed \
cache.cache_path=$CACHE_PATH \
cache.use_cache_without_dataset=$USE_CACHE_WITHOUT_DATASET \
lightning.trainer.params.max_epochs=$TRAIN_EPOCHS \
data_loader.params.batch_size=$BATCH_SIZE \
data_loader.params.num_workers=0 \
optimizer.lr=$TRAIN_LR \
lr_scheduler=multistep_lr \
lr_scheduler.milestones=$TRAIN_LR_MILESTONES \
lr_scheduler.gamma=$TRAIN_LR_DECAY \
hydra.searchpath="[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.training, pkg://tuplan_garage.planning.script.experiments, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"

