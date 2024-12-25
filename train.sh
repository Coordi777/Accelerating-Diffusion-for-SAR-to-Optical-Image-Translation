NNODES=1
WORLD_SIZE=1
MASTER_ADDR = 
MASTER_PORT =

LOG_PATH=/path/to/log
YAML_PATH=/path/to/yaml

DISTRIBUTED_ARGS="--num_machines $NNODES --num_processes $WORLD_SIZE --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT "

accelerate launch $DISTRIBUTED_ARGS --gpu_ids='all' train.py \
  --config_yaml $YAML_PATH \
  2>&1 | tee $LOG_PATH &


