#!/bin/bash -l
#SBATCH --job-name=fill_blankimgs_controlnet_inpainting_0901  # ADAPT
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


# python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda);"

# my repo
# python mysrc/automatic_noncond_inpainting.py --num_samples 1 --mask_dir /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/object_border_nonoverlap_masks --output_dir /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/synthesized_borderobj_DA_images
python mysrc/automatic_noncond_inpainting_blankimgs.py --image_dir /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/FINAL_FINAL_overlayed_imgs_controlnet_hed_f20_smin256max512_withClassBalance_random_overlaying_blank --num_samples 1 --mask_dir /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/FINAL_FINAL_blank_CN_mask_outside_bboxes_masks --output_dir /home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/experimental_blank_filling_1703


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