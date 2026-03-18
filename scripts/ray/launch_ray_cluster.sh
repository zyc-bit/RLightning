# launch the head node
WORK_DIR=$PWD
. $WORK_DIR/.venv/bin/activate

ray start --head --num-gpus 1

# # configure resources and launch the following instruction in the aliyun DLC.
# ray start --address='host_address:port'
# sleep 24h