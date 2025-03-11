# %%
# %pip install gradio diffusers

# %%
import gradio as gr
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
import io
from diffusers import StableDiffusionPipeline

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load BLIP model and processor
model_name = "Salesforce/blip-image-captioning-large"
blip_processor = BlipProcessor.from_pretrained(model_name)
blip_model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
blip_model.config.vision_config.output_attentions = True

# Load Stable Diffusion model
diffusion_model_name = "CompVis/stable-diffusion-v1-4"
diffusion_pipeline = StableDiffusionPipeline.from_pretrained(diffusion_model_name).to(device)

# Load smol model
smol_model_name = "Michaelj1/INSTRUCT_smolLM2-360M-finetuned-wikitext2-raw-v1"
tokenizer = AutoTokenizer.from_pretrained(smol_model_name)
smol_model = AutoModelForCausalLM.from_pretrained(smol_model_name).to(device)

# %%
def generate_caption(image):
    inputs = blip_processor(images=image, return_tensors="pt").to(device)
    caption_ids = blip_model.generate(**inputs, max_new_tokens=50)
    caption = blip_processor.decode(caption_ids[0], skip_special_tokens=True)
    return caption, inputs

def generate_gradcam(image, inputs):
    with torch.no_grad():
        vision_outputs = blip_model.vision_model(**inputs)
        attentions = vision_outputs.attentions
    last_layer_attentions = attentions[-1]
    avg_attention = last_layer_attentions.mean(dim=1)
    cls_attention = avg_attention[:, 0, 1:]
    num_patches = cls_attention.shape[-1]
    grid_size = int(np.sqrt(num_patches))
    attention_map = cls_attention.cpu().numpy().reshape(grid_size, grid_size)
    attention_map = cv2.resize(attention_map, (image.size[0], image.size[1]))
    attention_map = attention_map - np.min(attention_map)
    attention_map = attention_map / np.max(attention_map)
    img_np = np.array(image)
    heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img_np) / 255
    cam = cam / np.max(cam)
    cam_image = np.uint8(255 * cam)
    return cam_image


def generate_image_from_caption(caption):
    image = diffusion_pipeline(caption).images[0]
    return image


def explain_word(word):
    messages = [{"role": "user", "content": f"Explain the word '{word}' in detail."}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = smol_model.generate(
        inputs,
        max_new_tokens=150,
        temperature=0.9,
        top_p=0.95,
        do_sample=True
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    lines = generated_text.split('\n')
    assistant_response = []
    collect = False
    for line in lines:
        line = line.strip()
        if line.lower() == 'assistant':
            collect = True
            continue
        elif line.lower() in ['system', 'user']:
            collect = False
        if collect and line:
            assistant_response.append(line)
    explanation = '\n'.join(assistant_response).strip()
    return explanation

def get_caption_self_attention(caption):
    text_inputs = blip_processor.tokenizer(
        caption,
        return_tensors="pt",
        add_special_tokens=True
    ).to(device)
    
    with torch.no_grad():
        outputs = blip_model.text_decoder(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
            output_attentions=True,
            return_dict=True,
        )
    decoder_attentions = outputs.attentions
    return decoder_attentions, text_inputs


def generate_self_attention(decoder_attentions, text_inputs):
    last_layer_attentions = decoder_attentions[-1]
    avg_attentions = last_layer_attentions.mean(dim=1)
    attentions = avg_attentions[0].cpu().numpy()
    tokens = blip_processor.tokenizer.convert_ids_to_tokens(text_inputs.input_ids[0])
    cls_token = blip_processor.tokenizer.cls_token or "[CLS]"
    sep_token = blip_processor.tokenizer.sep_token or "[SEP]"
    special_token_indices = [idx for idx, token in enumerate(tokens) if token in [cls_token, sep_token]]
    mask = np.ones(len(tokens), dtype=bool)
    mask[special_token_indices] = False
    filtered_tokens = [token for idx, token in enumerate(tokens) if mask[idx]]
    filtered_attentions = attentions[mask, :][:, mask]
    return filtered_tokens, filtered_attentions

def process_image(image):
    # Ensure input is in the correct format
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    caption, inputs = generate_caption(image)
    cam_image = generate_gradcam(image, inputs)
    diffusion_image = generate_image_from_caption(caption)
    decoder_attentions, text_inputs = get_caption_self_attention(caption)
    filtered_tokens, filtered_attentions = generate_self_attention(decoder_attentions, text_inputs)
    
    # Create visualization grid
    fig, axs = plt.subplots(2, 2, figsize=(18, 18))
    
    axs[0][0].imshow(image)
    axs[0][0].axis('off')
    axs[0][0].set_title('Original Image')
    
    axs[0][1].imshow(cam_image)
    axs[0][1].axis('off')
    axs[0][1].set_title('Grad-CAM Overlay')
    
    axs[1][0].imshow(diffusion_image)
    axs[1][0].axis('off')
    axs[1][0].set_title('Generated Image (Stable Diffusion)')
    
    ax = axs[1][1]
    im = ax.imshow(filtered_attentions, cmap='viridis')
    ax.set_xticks(range(len(filtered_tokens)))
    ax.set_yticks(range(len(filtered_tokens)))
    ax.set_xticklabels(filtered_tokens, rotation=90, fontsize=8)
    ax.set_yticklabels(filtered_tokens, fontsize=8)
    ax.set_title('Caption Self-Attention')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Save visualization to a buffer for display
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close(fig)
    buffer.seek(0)
    visualization_image = Image.open(buffer)
    
    # Generate word options for dropdown
    words = caption.split()
    return caption, visualization_image, gr.Dropdown(label="Select a Word from Caption", choices=words, interactive=True)


def get_word_explanation(word):
    explanation = explain_word(word)
    return f"Explanation for '{word}':\n\n{explanation}"

# %%
# Define Gradio interface
with gr.Blocks() as interface:
    gr.Markdown("# Image Captioning and Visualization with Word Explanation")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload an Image")
            process_button = gr.Button("Process Image")
        with gr.Column():
            caption_output = gr.Textbox(label="Generated Caption")
            visualization_output = gr.Image(type="pil", label="Visualization (Original, Grad-CAM, Stable Diffusion)")
    
    word_dropdown = gr.Dropdown(label="Select a Word from Caption", choices=[], interactive=True)
    word_explanation = gr.Textbox(label="Word Explanation")
    
    # Bind functions to components
    process_button.click(
        process_image,
        inputs=image_input,
        outputs=[caption_output, visualization_output, word_dropdown]
    )
    
    word_dropdown.change(
        get_word_explanation,
        inputs=word_dropdown,
        outputs=word_explanation
    )

# %%
# Launch the Gradio app
interface.launch()

# %%



