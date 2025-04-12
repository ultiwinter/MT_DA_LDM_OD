from diffusers import StableDiffusionInpaintPipeline
import torch

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    revision="fp16",
    torch_dtype=torch.float16,
)
prompt = "white dog sitting"
#image and mask_image should be PIL images.
#The mask structure is white for inpainting and black for keeping as is
image = "./demo-images/demo_dog_165388_masked.png"
mask_image= "./demo-images/demo_dog_165388_masked.png"
image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
image.save("demo-images/white-dog-sitting.png")