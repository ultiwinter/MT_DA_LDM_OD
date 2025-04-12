#!/bin/bash -l
#SBATCH --job-name=testing_baseline  # ADAPT
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

conda activate ldm

echo "Temporary directory:"
echo $TMPDIR
WORKDIR="$TMPDIR/$SLURM_JOBID"
mkdir $WORKDIR
cd $WORKDIR

cp -r ${SLURM_SUBMIT_DIR}/. .
mkdir -p output/


# python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda);"


### stable diffusion from runway repository
# streamlit run scripts/inpaint_st.py -- configs/stable-diffusion/v1-inpainting-inference.yaml models/ldm/inpainting_big/model.ckpt 

# my repo
python ./mysrc/od_training.py -m test --checkptfile 861970/best-checkpoint-epoch=93-val_loss=0.28.ckpt



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

# mkdir -p "$MY_SLURM_OUTPUT/checkpoints/"
# cp -r checkpoints/ "$MY_SLURM_OUTPUT/checkpoints/"

# if [ $? -ne 0 ]; then
#   echo "Error: Failed to copy checkpoints files to $MY_SLURM_OUTPUT."
#   exit 1
# fi

# echo "checkpoints Files successfully copied to $MY_SLURM_OUTPUT."

# cp -r checkpoints/ "$SLURM_SUBMIT_DIR/checkpoints/"
