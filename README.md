# CAP-GEN-EXP
Pipeline combining the usage of BLIP ViT(https://huggingface.co/Salesforce/blip-image-captioning-large) ,fine-tuned version of SmolLM2 360M (https://huggingface.co/HuggingFaceTB/SmolLM2-360M) and Stable-diffusion-v1-4 (https://huggingface.co/CompVis/stable-diffusion-v1-4) for the purpose of image captioning , porviding grad-CAM overlay, self-attention and generating new images based on the extracted caption .
The app focuses on forwarding generated captions into the SmolLM2 in order to explain the word/object in the image in more detail as weel as forwarding into stable diffisuion to generate new images and includeing XAI (grad-cam, self-attention) for the whole process.

App can be tested at the following link : https://huggingface.co/spaces/Fine-Tuning-DLSE-Smol2/dlasw-pipeline-deploy?logs=container. Keep in mind as due to free hosting inference can last upwards to 10min.

# USER INTERFACE
![image](https://github.com/user-attachments/assets/75504a88-f899-42ff-b9ef-a443e6c318ee)
![image](https://github.com/user-attachments/assets/90ee7584-c4c7-4494-9cb8-f871b1da521c)


# WORKFLOW BLOCK DIAGRAM
![image](https://github.com/user-attachments/assets/eebd50ff-0cf3-470c-94cd-c182aac22348)

# ADDITIONAL EXAMPLES
![image](https://github.com/user-attachments/assets/0f9bd3e6-7acf-44b1-a145-0726d397ef30)
![image](https://github.com/user-attachments/assets/90103eac-920f-46b8-92b8-bfb18323f76b)

