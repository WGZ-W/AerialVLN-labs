
conda activate AirVLN

cd ./AirVLN
echo $PWD


#nohup python -u ./airsim_plugin/AirVLNSimulatorServerTool.py --gpus 0,1,2,3,4,5,6,7 &
nohup python -u ./airsim_plugin/AirVLNSimulatorServerTool.py --gpus 2,3 &
#CUDA_VISIBLE_DEVICES=2,3 nohup python -u ./airsim_plugin/AirVLNSimulatorServerTool.py --gpus 0,1 &

python -u ./src/vlnce_src/train.py \
--run_type eval \
--policy_type seq2seq \
--collect_type TF \
--name AirVLN-seq2seq \
--batchSize 8 \
--EVAL_CKPT_PATH_DIR ../DATA/models/ddppo-models \
--EVAL_DATASET train \
--EVAL_NUM -1


