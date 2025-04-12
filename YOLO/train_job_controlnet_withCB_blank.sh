#!/bin/bash -l
#SBATCH --job-name=yolo_controlnet_withCB_blank_it3  # ADAPT
#SBATCH --clusters=tinygpu
#SBATCH --ntasks=1

#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100

#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH --mail-type=end,fail
#SBATCH --time=24:00:00  # ADAPT
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
source ~/.bashrc

# Set proxy to access internet from the node
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

module purge
module load python
module load cuda
module load cudnn

conda activate yolov10

echo "Temporary directory:"
echo $TMPDIR
WORKDIR="$TMPDIR/$SLURM_JOBID"
mkdir $WORKDIR
cd $WORKDIR

cp -r ${SLURM_SUBMIT_DIR}/. .
mkdir -p output/

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count())"

# baseline trained model "/home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/train6/weights/best.pt"
# untrained model "yolo11m.pt"
# /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/controlnet_withCB_blank/controlnet_withCB_blank.yaml
# best performing model /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/train71/weights/best.pt

python ./my_yolo_train.py --model "yolo11m.pt" --yaml /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/controlnet_withCB_blank/controlnet_withCB_blank.yaml


if [ $? -ne 0 ]; then
    echo "Python script failed. Exiting."
    exit 1
fi
echo "Python script ran successfully."


export MY_SLURM_OUTPUT="${SLURM_SUBMIT_DIR}/${SLURM_JOB_ID}"
mkdir -p "$MY_SLURM_OUTPUT"


if [ $? -ne 0 ]; then
  echo "Error: Failed to create directory $MY_SLURM_OUTPUT."
  exit 1
fi

echo "Slurm output directory:"
echo "$MY_SLURM_OUTPUT"
cp -r ./output/. "$MY_SLURM_OUTPUT"


if [ $? -ne 0 ]; then
  echo "Error: Failed to copy output files to $MY_SLURM_OUTPUT."
  exit 1
fi

echo "Output Files successfully copied to $MY_SLURM_OUTPUT."
