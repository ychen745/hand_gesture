import os

script_root = '/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/scripts/sbatch_jobs'
images = '/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/data/custom/test_images'
python_path = '/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/detect.py'

model = '/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/config/cross-hands.cfg'
weights = '/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/weights/cross-hands.weights'
classes = '/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/data/hand.names'
output = '/scratch/ychen855/hand_gesture/PyTorch_YOLOv3/results/cross-hands'
batch_size = 8
# img_size = (384, 512)
img_size = 416
n_cpu = 4

with open(os.path.join(script_root, 'hand.sh'), 'w') as f:
	lines = []
	lines.append('#!/bin/bash\n')
	lines.append('#SBATCH -n 4                        # number of cores')
	lines.append('#SBATCH --mem=64G')
	lines.append('#SBATCH -G 1')
	lines.append('#SBATCH -t 0-04:00:00                 # wall time (D-HH:MM:SS)')
	lines.append('#SBATCH -o /scratch/ychen855/hand_gesture/PyTorch_YOLOv3/scripts/job_logs/detect.out')
	lines.append('#SBATCH -e /scratch/ychen855/hand_gesture/PyTorch_YOLOv3/scripts/job_logs/detect.err')
	lines.append('#SBATCH --mail-type=NONE             # Send a notification when the job starts, stops, or fails')
	lines.append('#SBATCH --mail-user=ychen855@asu.edu # send-to address\n')

	lines.append('source ~/.bashrc')
	lines.append('module load cuda-12.4.1-gcc-12.1.0')
	lines.append('conda activate hand\n')

	lines.append('PYTHON_PATH=' + python_path)
	lines.append('MODEL=' + model)
	lines.append('WEIGHTS=' + weights)
	lines.append('IMAGES=' + images)
	lines.append('CLASSES=' + classes)
	lines.append('OUTPUT=' + output)
	lines.append('BATCH_SIZE=' + str(batch_size))
	lines.append('IMG_SIZE=' + str(img_size))
	lines.append('N_CPU=' + str(n_cpu))

	command = 'python ${PYTHON_PATH}'
	command += ' --model ${MODEL}'
	command += ' --weights ${WEIGHTS}'
	command += ' --images ${IMAGES}'
	command += ' --classes ${CLASSES}'
	command += ' --output ${OUTPUT}'
	command += ' --batch_size ${BATCH_SIZE}'
	command += ' --img_size ${IMG_SIZE}'
	command += ' --n_cpu ${N_CPU}'

	lines.append(command + '\n')

	f.write('\n'.join(lines))


