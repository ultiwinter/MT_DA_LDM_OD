#!/bin/bash -l
#SBATCH --job-name=ups_ft_orig_yolo_novel_CB_5 # ADAPT
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
# /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/novel/novel.yaml
# best performing model /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/train77/weights/best.pt
# merged dataset /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/TEST8_MERGED_Novel_CB/TEST8_MERGED_Novel_CB.yaml
# best model trained on novel augs+orig /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/train169/weights/best.pt
# only augs dataset /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/TEST8_FINAL_Novel_CB/TEST8_FINAL_Novel_CB.yaml
# trained on augs only model /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/trainNov2203/train2/weights/best.pt
# trained orig finetuned merged 
# /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/baseline_orig/baseline_orig.yaml
# corrupted e1 aug trained model /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/trainCorNov/train/weights/best.pt
#                               /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/trainCorNov12/train/weights/best.pt





# train aug ft original
# train aug
# python ./my_yolo_train.py --output_dir /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/trainNovAug32 --model yolo11m.pt --yaml /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/TEST8_FINAL_Novel_CB/TEST8_FINAL_Novel_CB.yaml
# ft orig
# python ./my_yolo_train.py --output_dir /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/trainNovAug32ftOrig --model /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/trainNovAug32/train/weights/best.pt --yaml /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/baseline_orig/baseline_orig.yaml



# finetuning baseline-orig-trained
python ./my_yolo_train.py --output_dir /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/trainOrigftMergedNov6 --model /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/trainOriginalBaseline/train/weights/best.pt --yaml /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/TEST8_MERGED_Novel_CB/TEST8_MERGED_Novel_CB.yaml

# corrupt training e1 then ft orig
# corrupt training e1
# python ./my_yolo_train.py --output_dir /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/trainCorNov42 --model yolo11m.pt --yaml /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/FINAL_novel_CB/FINAL_novel_CB.yaml
# ft afterwards
# python ./my_yolo_train.py --output_dir /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/trainCorNov42ftOrig --model /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/trainCorNov42/train/weights/best.pt --yaml /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/baseline_orig/baseline_orig.yaml

# merged training no CB
python ./my_yolo_train.py --output_dir /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/trainMergedNovNoCB --model yolo11m.pt --yaml /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/baseline_novel/baseline_novel.yaml





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
