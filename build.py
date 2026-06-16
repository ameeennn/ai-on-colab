import os
import json

# List of models to generate
models = [
    {"owner": "google", "model": "gemma", "parameters": "2b", "id": "google/gemma-2b"},
    {"owner": "google", "model": "gemma", "parameters": "7b", "id": "google/gemma-7b"},
    {"owner": "google", "model": "gemma-4", "parameters": "sb", "id": "google/gemma-7b"},
    {"owner": "meta-llama", "model": "Meta-Llama-3", "parameters": "8B", "id": "meta-llama/Meta-Llama-3-8B"},
    {"owner": "mistralai", "model": "Mistral", "parameters": "7B", "id": "mistralai/Mistral-7B-v0.1"},
]

# GitHub repository details
github_user = "ameeennn"
github_repo = "ai-on-colab"
github_branch = "main"

def get_colab_link(filepath):
    # e.g., https://colab.research.google.com/github/ameeennn/ai-on-colab/blob/main/google/gemma/7b.ipynb
    return f"https://colab.research.google.com/github/{github_user}/{github_repo}/blob/{github_branch}/{filepath}"

def generate_notebook(model_id):
    notebook = {
     "cells": [
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
     ],
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
    readme_content = "# ai-on-colab\n\nRun open source AI models easily in Google Colab.\n\n## Models\n\n| Owner | Model | Parameters | Colab Link |\n|-------|-------|------------|------------|\n"

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Models on Colab</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        a { color: #0066cc; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>AI Models on Colab</h1>
    <p>Click on the links below to easily load and run open source AI models in Google Colab.</p>
    <table>
        <tr>
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

        dir_path = os.path.join(owner, model)
        os.makedirs(dir_path, exist_ok=True)

        filepath = os.path.join(dir_path, f"{params}.ipynb")

        notebook = generate_notebook(model_id)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=1)

        # Convert filepath to posix for URLs
        posix_filepath = filepath.replace('\\\\', '/')
        colab_link = get_colab_link(posix_filepath)

        readme_content += f"| {owner} | {model} | {params} | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_link}) |\n"

        html_content += f"""
        <tr>
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
