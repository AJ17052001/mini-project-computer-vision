pip install streamlit torch torchvision transformers pillow
import streamlit as st
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

st.set_page_config(page_title="Image Caption Generator", page_icon="🖼️")

st.title("🖼️ Image Caption Generator")
st.write("Upload an image to generate a caption.")

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model only once
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    if device == "cuda":
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            torch_dtype=torch.float16
        ).to(device)
    else:
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device)

    model.eval()
    return processor, model

processor, model = load_model()

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Caption"):

        with st.spinner("Generating caption..."):

            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=25,
                    num_beams=3,
                    do_sample=False,
                    early_stopping=True
                )

            caption = processor.decode(
                output[0],
                skip_special_tokens=True
            )

        st.success("Caption Generated")
        st.subheader("Caption")
        st.write(caption.capitalize())
