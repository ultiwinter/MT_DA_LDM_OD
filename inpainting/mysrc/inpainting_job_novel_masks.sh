#!/bin/bash -l
#SBATCH --job-name=novel_inpainting_0901_CB_1  # ADAPT
#SBATCH --clusters=tinygpu
#SBATCH --ntasks=1
#SBATCH --hold
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


# python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda);"

# my repo
python mysrc/automatic_inpaining.py --num_samples 1 --mask_dir /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/novel_masking_masks --output_dir /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/synthesized_novel_RWML_DA_CB


if [ $? -ne 0 ]; then
    echo "Python script failed. Exiting."
    exit 1
fi
echo "Python script ran successfully."


# export MY_SLURM_OUTPUT="${SLURM_SUBMIT_DIR}/${SLURM_JOB_ID}"
# mkdir -p "$MY_SLURM_OUTPUT"

# if [ $? -ne 0 ]; then
#   echo "Error: Failed to create directory $MY_SLURM_OUTPUT."
#   exit 1
# fi

# echo "Slurm output directory:"
# echo "$MY_SLURM_OUTPUT"
# cp -r ./output/. "$MY_SLURM_OUTPUT"

# if [ $? -ne 0 ]; then
#   echo "Error: Failed to copy files to $MY_SLURM_OUTPUT."
#   exit 1
# fi

# echo "Files successfully copied to $MY_SLURM_OUTPUT."