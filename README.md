# AI Fine-Tuning Project for uP! Life Story Generation

This repository contains the necessary files to fine-tune and evaluate LLMs on the Northflank platform.

## One-Time Setup on Northflank

1.  **Create Project & Service:** Create a project and a "Combined Service" on Northflank, linking it to this GitHub repository and selecting an NVIDIA GPU.
2.  **Add Storage:** Attach a persistent storage volume to the service (e.g., at mount path `/data`). Upload your `training_data.jsonl` and `docxexport.json` files to this `/data` volume.

## One-Time Model Download (Alternative Method)

To avoid depending on a live connection to the Hugging Face Hub during training, we will download the models once to our persistent storage using their official direct download methods.

**For LLaMA 3:**

1.  Go to the [official Meta Llama website](https://llama.meta.com/llama-downloads/) and request access to the Llama 3 8B Instruct model.
2.  Meta will email you a temporary, signed URL to download the model.
3.  Run a **one-off Manual Job** on Northflank with the following command, pasting the URL you received from Meta:

    ```bash
    wget -O - "PASTE_THE_SIGNED_URL_FROM_META_HERE" | tar -x -C /data/Llama-3-8B-Instruct
    ```
    *(Note: You must create the `/data/Llama-3-8B-Instruct` directory first if it doesn't exist)*

**For Mistral-7B:**

The most common distribution is via Hugging Face, but you can also download it via torrent if you wish to completely avoid the Hub. For simplicity in this trial, we recommend using the Hugging Face download method for Mistral as it is open-access and does not require a token.

**Command to run as a Manual Job:**
```bash
huggingface-cli download \
  mistralai/Mistral-7B-Instruct-v0.2 \
  --local-dir /data/Mistral-7B-Instruct-v0.2 \
  --local-dir-use-symlinks False
