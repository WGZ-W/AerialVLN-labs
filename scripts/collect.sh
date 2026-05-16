
#conda activate AirVLN
#
#cd ./AirVLN
#echo $PWD
#
#
#nohup python -u ./airsim_plugin/AirVLNSimulatorServerTool.py --gpus 2,3 &

python -u ./src/vlnce_src/train.py \
--run_type collect \
--policy_type cma \
--collect_type TF \
--name AirVLN-seq2seq \
--batchSize 16


