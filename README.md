# AI Fine-Tuning Project for uP! Life Story Generation

This repository contains the necessary files to fine-tune and evaluate LLMs on the Northflank platform.

## One-Time Setup on Northflank

1.  **Create Project & Service:** Create a project and a "Combined Service" on Northflank, linking it to this GitHub repository and selecting an NVIDIA GPU.
2.  **Add Storage:** Attach a persistent storage volume to the service (e.g., at mount path `/data`). Upload your `training_data.jsonl` and `docxexport.json` files to this `/data` volume.
3.  **Add Secrets:** Add a project secret named `HF_TOKEN` with your Hugging Face access token.

## One-Time Model Download

Run the following commands as a **one-off Job** on Northflank to download the models to your persistent storage.

**Command to Download LLaMA 3:**
```bash
huggingface-cli download \
  meta-llama/Meta-Llama-3-8B-Instruct \
  --local-dir /data/Llama-3-8B-Instruct \
  --local-dir-use-symlinks False
