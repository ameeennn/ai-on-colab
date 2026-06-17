import os
import json

# List of models to generate
models = [
    # Original Text Generation
    {"owner": "google", "model": "gemma", "parameters": "2b", "id": "google/gemma-2b", "tier": "Free", "task": "text-generation"},
    {"owner": "google", "model": "gemma", "parameters": "7b", "id": "google/gemma-7b", "tier": "Free", "task": "text-generation"},
    {"owner": "google", "model": "gemma-4", "parameters": "sb", "id": "google/gemma-7b", "tier": "Free", "task": "text-generation"},
    {"owner": "meta-llama", "model": "Meta-Llama-3", "parameters": "8B", "id": "meta-llama/Meta-Llama-3-8B", "tier": "Free", "task": "text-generation"},
    {"owner": "mistralai", "model": "Mistral", "parameters": "7B", "id": "mistralai/Mistral-7B-v0.1", "tier": "Free", "task": "text-generation"},

    # Gemma 2
    {"owner": "google", "model": "gemma-2", "parameters": "2b", "id": "google/gemma-2-2b", "tier": "Free", "task": "text-generation"},
    {"owner": "google", "model": "gemma-2", "parameters": "9b", "id": "google/gemma-2-9b", "tier": "Free", "task": "text-generation"},
    {"owner": "google", "model": "gemma-2", "parameters": "27b", "id": "google/gemma-2-27b", "tier": "Paid", "task": "text-generation"},

    # Gemma 3
    {"owner": "google", "model": "gemma-3", "parameters": "1b", "id": "google/gemma-3-1b-it", "tier": "Free", "task": "text-generation"},
    {"owner": "google", "model": "gemma-3", "parameters": "4b", "id": "google/gemma-3-4b-it", "tier": "Free", "task": "text-generation"},
    {"owner": "google", "model": "gemma-3", "parameters": "12b", "id": "google/gemma-3-12b-it", "tier": "Paid", "task": "text-generation"},
    {"owner": "google", "model": "gemma-3", "parameters": "27b", "id": "google/gemma-3-27b-it", "tier": "Paid", "task": "text-generation"},

    # Edge (Gemma 3n / Gemma 4 E)
    {"owner": "google", "model": "gemma-3n", "parameters": "E2B", "id": "google/gemma-3n-E2B-it", "tier": "Free", "task": "text-generation"},
    {"owner": "google", "model": "gemma-4", "parameters": "E2B", "id": "google/gemma-4-E2B", "tier": "Free", "task": "text-generation"},
    {"owner": "google", "model": "gemma-4", "parameters": "E4B", "id": "google/gemma-4-E4B", "tier": "Free", "task": "text-generation"},

    # CodeGemma
    {"owner": "google", "model": "codegemma", "parameters": "2b", "id": "google/codegemma-2b", "tier": "Free", "task": "text-generation"},
    {"owner": "google", "model": "codegemma", "parameters": "7b", "id": "google/codegemma-7b", "tier": "Free", "task": "text-generation"},

    # MedGemma
    {"owner": "google", "model": "medgemma", "parameters": "4b", "id": "google/medgemma-4b-it", "tier": "Free", "task": "text-generation"},
    {"owner": "google", "model": "medgemma", "parameters": "27b", "id": "google/medgemma-27b-it", "tier": "Paid", "task": "text-generation"},

    # PaliGemma
    {"owner": "google", "model": "paligemma", "parameters": "3b", "id": "google/paligemma-3b-pt-224", "tier": "Free", "task": "text-generation"},
    {"owner": "google", "model": "paligemma2", "parameters": "3b", "id": "google/paligemma2-3b-pt-224", "tier": "Free", "task": "text-generation"},
    {"owner": "google", "model": "paligemma2", "parameters": "10b", "id": "google/paligemma2-10b-pt-224", "tier": "Paid", "task": "text-generation"},
    {"owner": "google", "model": "paligemma2", "parameters": "28b", "id": "google/paligemma2-28b-pt-224", "tier": "Paid", "task": "text-generation"},

    # Image Generation
    {"owner": "stable-diffusion-v1-5", "model": "stable-diffusion", "parameters": "v1-5", "id": "stable-diffusion-v1-5/stable-diffusion-v1-5", "tier": "Free", "task": "image-generation"},
    {"owner": "stabilityai", "model": "stable-diffusion-xl", "parameters": "1.0", "id": "stabilityai/stable-diffusion-xl-base-1.0", "tier": "Free", "task": "image-generation"},
    {"owner": "stabilityai", "model": "stable-diffusion-3.5", "parameters": "large", "id": "stabilityai/stable-diffusion-3.5-large", "tier": "Paid", "task": "image-generation"},
    {"owner": "black-forest-labs", "model": "FLUX.1", "parameters": "schnell", "id": "black-forest-labs/FLUX.1-schnell", "tier": "Paid", "task": "image-generation"},
    {"owner": "playgroundai", "model": "playground-v2.5", "parameters": "1024px", "id": "playgroundai/playground-v2.5-1024px-aesthetic", "tier": "Free", "task": "image-generation"},
    {"owner": "kandinsky-community", "model": "kandinsky", "parameters": "2-2", "id": "kandinsky-community/kandinsky-2-2-decoder", "tier": "Free", "task": "image-generation"},
    {"owner": "PixArt-alpha", "model": "PixArt-Sigma", "parameters": "1024", "id": "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", "tier": "Free", "task": "image-generation"},

    # Code Generation
    {"owner": "meta-llama", "model": "CodeLlama", "parameters": "7b", "id": "meta-llama/CodeLlama-7b-hf", "tier": "Free", "task": "text-generation"},
    {"owner": "deepseek-ai", "model": "deepseek-coder", "parameters": "1.3b", "id": "deepseek-ai/deepseek-coder-1.3b-instruct", "tier": "Free", "task": "text-generation"},
    {"owner": "deepseek-ai", "model": "deepseek-coder", "parameters": "33b", "id": "deepseek-ai/deepseek-coder-33b-instruct", "tier": "Paid", "task": "text-generation"},
    {"owner": "Qwen", "model": "Qwen2.5-Coder", "parameters": "7B", "id": "Qwen/Qwen2.5-Coder-7B-Instruct", "tier": "Free", "task": "text-generation"},
    {"owner": "Qwen", "model": "Qwen2.5-Coder", "parameters": "32B", "id": "Qwen/Qwen2.5-Coder-32B-Instruct", "tier": "Paid", "task": "text-generation"},
    {"owner": "bigcode", "model": "starcoder2", "parameters": "3b", "id": "bigcode/starcoder2-3b", "tier": "Free", "task": "text-generation"},
    {"owner": "bigcode", "model": "starcoder2", "parameters": "15b", "id": "bigcode/starcoder2-15b", "tier": "Paid", "task": "text-generation"},

    # Audio Generation
    {"owner": "facebook", "model": "musicgen", "parameters": "medium", "id": "facebook/musicgen-medium", "tier": "Free", "task": "audio-generation"},
    {"owner": "facebook", "model": "musicgen", "parameters": "stereo-large", "id": "facebook/musicgen-stereo-large", "tier": "Paid", "task": "audio-generation"},
    {"owner": "suno", "model": "bark", "parameters": "small", "id": "suno/bark-small", "tier": "Free", "task": "audio-generation"},
    {"owner": "suno", "model": "bark", "parameters": "large", "id": "suno/bark", "tier": "Paid", "task": "audio-generation"},

    # Video Generation
    {"owner": "stabilityai", "model": "stable-video-diffusion", "parameters": "img2vid-xt", "id": "stabilityai/stable-video-diffusion-img2vid-xt", "tier": "Paid", "task": "video-generation"},
    {"owner": "zai-org", "model": "CogVideoX", "parameters": "2b", "id": "zai-org/CogVideoX-2b", "tier": "Free", "task": "video-generation"},
    {"owner": "zai-org", "model": "CogVideoX", "parameters": "5b", "id": "zai-org/CogVideoX-5b", "tier": "Paid", "task": "video-generation"},
    {"owner": "ali-vilab", "model": "text-to-video-ms", "parameters": "1.7b", "id": "ali-vilab/text-to-video-ms-1.7b", "tier": "Free", "task": "video-generation"},

    # More Text Generation
    {"owner": "Qwen", "model": "Qwen2.5-Instruct", "parameters": "7B", "id": "Qwen/Qwen2.5-7B-Instruct", "tier": "Free", "task": "text-generation"},
    {"owner": "meta-llama", "model": "Llama-3.1-Instruct", "parameters": "8B", "id": "meta-llama/Llama-3.1-8B-Instruct", "tier": "Free", "task": "text-generation"},
    {"owner": "meta-llama", "model": "Llama-3.2-Instruct", "parameters": "3B", "id": "meta-llama/Llama-3.2-3B-Instruct", "tier": "Free", "task": "text-generation"},
    {"owner": "meta-llama", "model": "Llama-3.2-Instruct", "parameters": "1B", "id": "meta-llama/Llama-3.2-1B-Instruct", "tier": "Free", "task": "text-generation"},

    # More Image Generation
    {"owner": "Kwai-Kolors", "model": "Kolors", "parameters": "base", "id": "Kwai-Kolors/Kolors", "tier": "Free", "task": "image-generation"},
    {"owner": "fal", "model": "AuraFlow", "parameters": "base", "id": "fal/AuraFlow", "tier": "Free", "task": "image-generation"},

    # More Audio Generation
    {"owner": "parler-tts", "model": "parler_tts", "parameters": "mini_v0.1", "id": "parler-tts/parler_tts_mini_v0.1", "tier": "Free", "task": "audio-generation"},
    {"owner": "facebook", "model": "seamless-m4t", "parameters": "medium", "id": "facebook/seamless-m4t-medium", "tier": "Free", "task": "audio-generation"},
    {"owner": "facebook", "model": "seamless-m4t", "parameters": "v2-large", "id": "facebook/seamless-m4t-v2-large", "tier": "Paid", "task": "audio-generation"},

]

# GitHub repository details
github_user = "ameeennn"
github_repo = "ai-on-colab"
github_branch = "main"

def get_colab_link(filepath):
    # e.g., https://colab.research.google.com/github/ameeennn/ai-on-colab/blob/main/google/gemma/7b.ipynb
    return f"https://colab.research.google.com/github/{github_user}/{github_repo}/blob/{github_branch}/{filepath}"

def generate_notebook(model_id, task):
    if task == "text-generation":
        cells = [
            {
             "cell_type": "markdown",
             "metadata": {},
             "source": [
              f"# Load {model_id} in Google Colab\n",
              f"This notebook helps you easily load and run `{model_id}` in Google Colab using 4-bit quantization."
             ]
            },
            {
             "cell_type": "code",
             "execution_count": None,
             "metadata": {},
             "outputs": [],
             "source": [
              "!pip install -q -U transformers accelerate bitsandbytes"
             ]
            },
            {
             "cell_type": "code",
             "execution_count": None,
             "metadata": {},
             "outputs": [],
             "source": [
              "import torch\n",
              "from transformers import AutoModelForCausalLM, AutoTokenizer\n",
              "\n",
              f"model_id = \"{model_id}\"\n",
              "\n",
              "tokenizer = AutoTokenizer.from_pretrained(model_id)\n",
              "model = AutoModelForCausalLM.from_pretrained(\n",
              "    model_id,\n",
              "    device_map=\"auto\",\n",
              "    torch_dtype=torch.float16,\n",
              "    load_in_4bit=True,\n",
              ")"
             ]
            },
            {
             "cell_type": "code",
             "execution_count": None,
             "metadata": {},
             "outputs": [],
             "source": [
              "text = \"What is the capital of France?\"\n",
              "inputs = tokenizer(text, return_tensors=\"pt\").to(\"cuda\")\n",
              "outputs = model.generate(**inputs, max_new_tokens=50)\n",
              "print(tokenizer.decode(outputs[0], skip_special_tokens=True))"
             ]
            }
        ]
    elif task == "image-generation":
        cells = [
            {
             "cell_type": "markdown",
             "metadata": {},
             "source": [
              f"# Load {model_id} in Google Colab\n",
              f"This notebook helps you easily load and run the `{model_id}` image generation model in Google Colab using diffusers."
             ]
            },
            {
             "cell_type": "code",
             "execution_count": None,
             "metadata": {},
             "outputs": [],
             "source": [
              "!pip install -q -U diffusers transformers accelerate"
             ]
            },
            {
             "cell_type": "code",
             "execution_count": None,
             "metadata": {},
             "outputs": [],
             "source": [
              "import torch\n",
              "from diffusers import DiffusionPipeline\n",
              "\n",
              f"model_id = \"{model_id}\"\n",
              "\n",
              "pipe = DiffusionPipeline.from_pretrained(\n",
              "    model_id,\n",
              "    torch_dtype=torch.float16\n",
              ")\n",
              "pipe = pipe.to(\"cuda\")"
             ]
            },
            {
             "cell_type": "code",
             "execution_count": None,
             "metadata": {},
             "outputs": [],
             "source": [
              "prompt = \"A beautiful sunset over a serene lake, digital art\"\n",
              "image = pipe(prompt).images[0]\n",
              "image"
             ]
            }
        ]
    elif task == "audio-generation":
        cells = [
            {
             "cell_type": "markdown",
             "metadata": {},
             "source": [
              f"# Load {model_id} in Google Colab\n",
              f"This notebook helps you easily load and run the `{model_id}` audio generation model in Google Colab."
             ]
            },
            {
             "cell_type": "code",
             "execution_count": None,
             "metadata": {},
             "outputs": [],
             "source": [
              "!pip install -q -U transformers accelerate scipy"
             ]
            },
            {
             "cell_type": "code",
             "execution_count": None,
             "metadata": {},
             "outputs": [],
             "source": [
              "from transformers import pipeline\n",
              "import scipy\n",
              "\n",
              f"model_id = \"{model_id}\"\n",
              "\n",
              "synthesizer = pipeline(\"text-to-audio\", model=model_id, device=0)\n",
              "\n",
              "prompt = \"A cheerful upbeat pop song\"\n",
              "music = synthesizer(prompt, forward_params={\"max_new_tokens\": 256})\n",
              "\n",
              "scipy.io.wavfile.write(\"output.wav\", rate=music[\"sampling_rate\"], data=music[\"audio\"])\n",
              "print(\"Audio saved to output.wav\")"
             ]
            }
        ]
    elif task == "video-generation":
        cells = [
            {
             "cell_type": "markdown",
             "metadata": {},
             "source": [
              f"# Load {model_id} in Google Colab\n",
              f"This notebook helps you easily load and run the `{model_id}` video generation model in Google Colab using diffusers."
             ]
            },
            {
             "cell_type": "code",
             "execution_count": None,
             "metadata": {},
             "outputs": [],
             "source": [
              "!pip install -q -U diffusers transformers accelerate"
             ]
            },
            {
             "cell_type": "code",
             "execution_count": None,
             "metadata": {},
             "outputs": [],
             "source": [
              "import torch\n",
              "from diffusers import DiffusionPipeline\n",
              "from diffusers.utils import export_to_video\n",
              "\n",
              f"model_id = \"{model_id}\"\n",
              "\n",
              "pipe = DiffusionPipeline.from_pretrained(\n",
              "    model_id,\n",
              "    torch_dtype=torch.float16,\n",
              "    variant=\"fp16\"\n",
              ")\n",
              "pipe = pipe.to(\"cuda\")"
             ]
            },
            {
             "cell_type": "code",
             "execution_count": None,
             "metadata": {},
             "outputs": [],
             "source": [
              "prompt = \"A dog running in the park\"\n",
              "video_frames = pipe(prompt, num_frames=16).frames[0]\n",
              "export_to_video(video_frames, \"output.mp4\", fps=7)\n",
              "print(\"Video saved to output.mp4\")"
             ]
            }
        ]

    notebook = {
     "cells": cells,
     "metadata": {
      "accelerator": "GPU",
      "colab": {
       "gpuType": "T4",
       "provenance": []
      },
      "kernelspec": {
       "display_name": "Python 3",
       "name": "python3"
      },
      "language_info": {
       "name": "python"
      }
     },
     "nbformat": 4,
     "nbformat_minor": 0
    }
    return notebook

def main():
    readme_content = "# ai-on-colab\n\nRun open source AI models easily in Google Colab.\n\n## Models\n\n| Tier | Task | Owner | Model | Parameters | Colab Link |\n|------|------|-------|-------|------------|------------|\n"

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Models on Colab</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        a { color: #0066cc; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .filter-container { margin-top: 20px; margin-bottom: 20px; display: flex; gap: 20px; }
        .filter-group { display: flex; flex-direction: column; }
        .tier-free { color: green; font-weight: bold; }
        .tier-paid { color: orange; font-weight: bold; }
    </style>
    <script>
        function filterTable() {
            var tierFilter = document.getElementById("tierFilter").value.toUpperCase();
            var taskFilter = document.getElementById("taskFilter").value.toUpperCase();

            var table = document.getElementById("modelsTable");
            var tr = table.getElementsByTagName("tr");

            for (var i = 1; i < tr.length; i++) {
                var tdTier = tr[i].getElementsByTagName("td")[0];
                var tdTask = tr[i].getElementsByTagName("td")[1];

                if (tdTier && tdTask) {
                    var tierValue = tdTier.textContent || tdTier.innerText;
                    var taskValue = tdTask.textContent || tdTask.innerText;

                    var tierMatch = (tierFilter === "ALL" || tierValue.toUpperCase() === tierFilter);
                    var taskMatch = (taskFilter === "ALL" || taskValue.toUpperCase() === taskFilter);

                    if (tierMatch && taskMatch) {
                        tr[i].style.display = "";
                    } else {
                        tr[i].style.display = "none";
                    }
                }
            }
        }
    </script>
</head>
<body>
    <h1>AI Models on Colab</h1>
    <p>Click on the links below to easily load and run open source AI models in Google Colab.</p>

    <div class="filter-container">
        <div class="filter-group">
            <label for="tierFilter"><strong>Filter by Tier:</strong></label>
            <select id="tierFilter" onchange="filterTable()">
                <option value="All">All Tiers</option>
                <option value="Free">Free Tier (<= 10B / SDXL)</option>
                <option value="Paid">Paid Tier (Pro/Pro+)</option>
            </select>
        </div>
        <div class="filter-group">
            <label for="taskFilter"><strong>Filter by Task:</strong></label>
            <select id="taskFilter" onchange="filterTable()">
                <option value="All">All Tasks</option>
                <option value="text-generation">Text Generation</option>
                <option value="image-generation">Image Generation</option>
                <option value="audio-generation">Audio Generation</option>
                <option value="video-generation">Video Generation</option>
            </select>
        </div>
    </div>

    <table id="modelsTable">
        <tr>
            <th>Tier</th>
            <th>Task</th>
            <th>Owner</th>
            <th>Model</th>
            <th>Parameters</th>
            <th>Colab Link</th>
        </tr>
"""

    for m in models:
        owner = m["owner"]
        model = m["model"]
        params = m["parameters"]
        model_id = m["id"]
        tier = m["tier"]
        task = m["task"]

        dir_path = os.path.join(owner, model)
        os.makedirs(dir_path, exist_ok=True)

        filepath = os.path.join(dir_path, f"{params}.ipynb")

        notebook = generate_notebook(model_id, task)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=1)

        # Convert filepath to posix for URLs
        posix_filepath = filepath.replace('\\\\', '/')
        colab_link = get_colab_link(posix_filepath)

        tier_class = "tier-free" if tier == "Free" else "tier-paid"

        readme_content += f"| {tier} | {task} | {owner} | {model} | {params} | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_link}) |\n"

        html_content += f"""
        <tr>
            <td class="{tier_class}">{tier}</td>
            <td>{task}</td>
            <td>{owner}</td>
            <td>{model}</td>
            <td>{params}</td>
            <td><a href="{colab_link}" target="_blank">Open in Colab</a></td>
        </tr>
"""

    html_content += """
    </table>
</body>
</html>
"""

    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_content)

if __name__ == "__main__":
    main()
