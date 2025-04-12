#!/bin/bash -l
#SBATCH --job-name=ups_ft_orig_yolo_controlnet_withCB_blank_filled_5  # ADAPT
#SBATCH --clusters=tinygpu
#SBATCH --ntasks=1

#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100

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

# baseline trained model /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/trainOriginalBaseline/train/weights/best.pt
# untrained model "yolo11m.pt"
# /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/controlnet_withCB_blank_filled/controlnet_withCB_blank_filled.yaml
# best performing model /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/train85/weights/best.pt
# merged dataset /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/FINAL_baseline_CN_wCB_filled/FINAL_baseline_CN_wCB_filled.yaml
# best trained augs+orig model /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/train175/weights/best.pt
# only augs dataset /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/FINAL_controlnet_withCB_blank_filled/FINAL_controlnet_withCB_blank_filled.yaml
# trained on augs only model /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/trainCN2203_3/train/weights/best.pt
# trained orig finetuned merged
# /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/baseline_orig/baseline_orig.yaml


# train aug ft original
# train aug
# python ./my_yolo_train.py --output_dir /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/trainCNAug32 --model yolo11m.pt --yaml /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/controlnet_withCB_blank_filled/controlnet_withCB_blank_filled.yaml
# ft orig
# python ./my_yolo_train.py --output_dir /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/trainCNAug32ftOrig --model /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/trainCNAug32/train/weights/best.pt --yaml /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/baseline_orig/baseline_orig.yaml




# finetuning baseline-orig-trained
# python ./my_yolo_train.py --output_dir /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/trainOrigftMergedCN6 --model /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/trainOriginalBaseline/train/weights/best.pt --yaml /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/FINAL_baseline_CN_wCB_filled/FINAL_baseline_CN_wCB_filled.yaml




# corrupt training e1 then ft orig
# corrupt training e1
# python ./my_yolo_train.py --output_dir /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/trainCorCN42 --model yolo11m.pt --yaml /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/TEST8_corrupted_CN/TEST8_corrupted_CN.yaml
# ft afterwards
# python ./my_yolo_train.py --output_dir /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/trainCorCN42ftOrig --model /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/trainCorCN42/train/weights/best.pt --yaml /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/baseline_orig/baseline_orig.yaml


# merged training no CB
python ./my_yolo_train.py --output_dir /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/trainMergedControlNetNoCB --model yolo11m.pt --yaml /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/baseline_CN_noCB/baseline_CN_noCB.yaml


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
