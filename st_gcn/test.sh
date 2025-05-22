PYTHON=/scratch/ychen855/hand_gesture/st_gcn/test.py
CONFIG=/scratch/ychen855/hand_gesture/st_gcn/config/train_toyTestDataset.yaml
RESUME=/scratch/ychen855/hand_gesture/st_gcn/exps/toyTest-win45/checkpoint_epoch253_val74_train76.pt

python ${PYTHON} --config ${CONFIG} --resume ${RESUME} --test_only