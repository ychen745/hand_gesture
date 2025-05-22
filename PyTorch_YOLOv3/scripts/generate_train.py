import os

script_root = '/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/scripts/sbatch_jobs'
python_path = '/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/train.py'
conf_data = '/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/config/hand_yolo.data'
conf_model = '/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/config/hand_yolo.cfg'
conf_pretrained_weights = '/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/weights/cross-hands.weights'
checkpoints_root = '/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/checkpoints'
log_root = '/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/logs'
epochs = 60
conf_thres = 0.6

continue_train = True
start_epoch = 60

exp_name = 'hand_yolo_lr0.0001_conf0.6'

if exp_name not in os.listdir(checkpoints_root):
    os.mkdir(os.path.join(checkpoints_root, exp_name))
checkpoints_dir = os.path.join(checkpoints_root, exp_name)

if exp_name not in os.listdir(log_root):
    os.mkdir(os.path.join(log_root, exp_name))
logdir = os.path.join(log_root, exp_name)

with open(os.path.join(script_root, exp_name + '.sh'), 'w') as f:
    lines = []
    lines.append('#!/bin/bash\n')
    lines.append('#SBATCH -n 4                        # number of cores')
    lines.append('#SBATCH --mem=64G')
    lines.append('#SBATCH -G a100:1')
    lines.append('#SBATCH -t 0-04:00:00                 # wall time (D-HH:MM:SS)')
    lines.append('#SBATCH -o /scratch/ychen855/hand_gesture/PyTorch_YOLOv3/scripts/job_logs/' + exp_name + '.out')
    lines.append('#SBATCH -e /scratch/ychen855/hand_gesture/PyTorch_YOLOv3/scripts/job_logs/' + exp_name + '.err')
    lines.append('#SBATCH --mail-type=END             # Send a notification when the job starts, stops, or fails')
    lines.append('#SBATCH --mail-user=ychen855@asu.edu # send-to address\n')

    lines.append('source ~/.bashrc')
    lines.append('module load cuda-12.4.1-gcc-12.1.0')
    lines.append('conda activate hand\n')

    lines.append('PYTHON_PATH=' + python_path)
    lines.append('MODEL=' + conf_model)
    lines.append('PRETRAINED_WEIGHTS=' + conf_pretrained_weights)
    lines.append('DATA=' + conf_data)
    lines.append('EPOCHS=' + str(epochs))
    lines.append('CHECKPOINTS_DIR=' + checkpoints_dir)
    lines.append('LOGDIR=' + logdir)
    lines.append('CONF_THRES=' + str(conf_thres))
    if continue_train:
        lines.append('START_EPOCH=' + str(start_epoch))

    command = 'python ${PYTHON_PATH}'
    command += ' --model ${MODEL}'
    command += ' --pretrained_weights ${PRETRAINED_WEIGHTS}'
    command += ' --epochs ${EPOCHS}'
    command += ' --data ${DATA}'
    command += ' --checkpoints_dir ${CHECKPOINTS_DIR}'
    command += ' --logdir ${LOGDIR}'
    command += ' --conf_thres ${CONF_THRES}'
    if continue_train:
        command += ' --start_eopch ${START_EPOCH}'

    lines.append(command + '\n')

    f.write('\n'.join(lines))


