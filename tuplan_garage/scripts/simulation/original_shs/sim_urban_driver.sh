SPLIT=val14_split
CHALLENGE=closed_loop_reactive_agents # open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents
CHECKPOINT_PATH="/home/workspace/nuplan/tuplan_garage/trained_models/urbandriver_checkpoint.ckpt"

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
+simulation=$CHALLENGE \
planner=ml_planner \
scenario_filter=$SPLIT \
scenario_builder=nuplan \
planner.ml_planner.model_config='\${model}' \
planner.ml_planner.checkpoint_path=$CHECKPOINT_PATH \
model=urban_driver_open_loop_model \
hydra.searchpath="[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
