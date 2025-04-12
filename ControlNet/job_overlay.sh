#!/bin/bash -l
#SBATCH --job-name=overlay_hed  # ADAPT DA_15e_classimbalancetest_281124_b1 finetuning-oilpaint-100e_s128 DA_15e_classimbalancetest_241124_b2 finetuning_after_finetuning20e DA_afterfintuning20e DA_classimbalancetest_231124_b2 finetuning-oilpaint-50e
#SBATCH --clusters=tinygpu
#SBATCH --ntasks=1

#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100

#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH --mail-type=end,fail
#SBATCH --time=24:00:00
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
source ~/.bashrc

# Set proxy to access internet from the node
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80


module purge
# to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
module load python
module load cuda
module load cudnn


conda activate control
pip install xformers

echo "Temporary directory:"
echo $TMPDIR
WORKDIR="$TMPDIR/$SLURM_JOBID"
mkdir $WORKDIR
cd $WORKDIR

cp -r ${SLURM_SUBMIT_DIR}/. .


python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda);"

python overlay_bbox_imgs_plus_random.py 


if [ $? -ne 0 ]; then
    echo "Python script failed. Exiting."
    exit 1
fi
echo "Python script ran successfully."

export MY_SLURM_OUTPUT="${HOME}/${SLURM_JOB_ID}"
mkdir -p "$MY_SLURM_OUTPUT"

if [ $? -ne 0 ]; then
  echo "Error: Failed to create directory $MY_SLURM_OUTPUT."
fi

# echo "Slurm output directory:"
# echo "$MY_SLURM_OUTPUT"
# cp -r ./outputs/. "$MY_SLURM_OUTPUT"

# if [ $? -ne 0 ]; then
#   echo "Error: Failed to copy files to $MY_SLURM_OUTPUT."
#   exit 1
# fi

# echo "Files successfully copied to $MY_SLURM_OUTPUT."


export CKPT_OUTPUT_PATH="${SLURM_SUBMIT_DIR}/checkpoints"
mkdir -p "$CKPT_OUTPUT_PATH"
if [ $? -ne 0 ]; then
  echo "Error: Failed to create directory $CKPT_OUTPUT_PATH."
fi
echo "ckpt output path:"
echo "$CKPT_OUTPUT_PATH"
cp -r ./checkpoints/. "$CKPT_OUTPUT_PATH"
if [ $? -ne 0 ]; then
  echo "Error: Failed to copy files to $CKPT_OUTPUT_PATH."
fi
