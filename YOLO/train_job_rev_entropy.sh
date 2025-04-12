#!/bin/bash -l
#SBATCH --job-name=ups_ft_orig_yolo_rev_entropy_CB_5  # ADAPT
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
# /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/reversed_entropy/reversed_entropy.yaml
# best performing model /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/train99/weights/best.pt
# merged dataset /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/TEST8_MERGED_ent_entropy_CB/TEST8_MERGED_ent_entropy_CB.yaml
# best trained on merged (orig+aug) /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/train177/weights/best.onnx
# only augs dataset /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/TEST8_FINAL_ent_entropy_CB/TEST8_FINAL_ent_entropy_CB.yaml
# trained on augs only model /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/trainRevEnt2203/train/weights/best.pt
# trained orig finetuned merged 
# /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/baseline_orig/baseline_orig.yaml
# corrupted aug trained e1 model /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/trainCorRevEntropy/train/weights/best.pt


# train aug ft original
# train aug
# python ./my_yolo_train.py --output_dir /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/trainRevEntropyAug32 --model yolo11m.pt --yaml /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/TEST8_FINAL_ent_entropy_CB/TEST8_FINAL_ent_entropy_CB.yaml
# ft orig
# python ./my_yolo_train.py --output_dir /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/trainRevEntropyAug32ftOrig --model /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/trainRevEntropyAug32/train/weights/best.pt --yaml /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/baseline_orig/baseline_orig.yaml




# finetuning baseline-orig-trained
# python ./my_yolo_train.py --output_dir /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/trainOrigftMergedRevEnt6 --model /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/trainOriginalBaseline/train/weights/best.pt --yaml /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/TEST8_MERGED_ent_entropy_CB/TEST8_MERGED_ent_entropy_CB.yaml

# corrupt training e1 then ft orig
# corrupt training e1
# python ./my_yolo_train.py --output_dir /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/trainCorRevEntropy42 --model yolo11m.pt --yaml /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/FINAL_reversed_entropy_CB/FINAL_reversed_entropy_CB.yaml
# ft afterwards
# python ./my_yolo_train.py --output_dir /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/trainCorRevEntropy42ftOrig --model /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/trainCorRevEntropy42/train/weights/best.pt --yaml /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/baseline_orig/baseline_orig.yaml

# merged training no CB
python ./my_yolo_train.py --output_dir /home/woody/iwi5/iwi5215h/masterarbeit/repos/ultralytics/runs/detect/trainMergedRevEntNoCB --model yolo11m.pt --yaml /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/baseline_revEntropy/baseline_revEntropy.yaml



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
