import torch
import os
from msclap import CLAP

def normalize(x):
    return x / x.norm(dim=-1, keepdim=True)

def main():
    audio_dir = "audios"
    file_template = "benchmark-styletransfer-{}.mp3"
    prompts = [
        "a classical music song with a full orchestra",
        "a classical music song with a full orchestra",
        "a classical music song with a full orchestra"
    ]

    clap = CLAP(version="2023", use_cuda=False)

    text_embeddings = clap.get_text_embeddings(prompts)
    text_embeddings = normalize(text_embeddings)  # normalize text embeddings

    for idx, prompt in enumerate(prompts, start=1):
        audio_path = os.path.join(audio_dir, file_template.format(idx))
        if not os.path.isfile(audio_path):
            print(f"[!] File not found: {audio_path}")
            continue

        audio_emb = clap.get_audio_embeddings([audio_path])
        audio_emb = normalize(audio_emb)  # normalize audio embedding

        # now compute dot product (which is cosine similarity after normalization)
        sim = torch.matmul(audio_emb, text_embeddings[idx-1 : idx].T)
        score = sim[0, 0].item()

        print(f"Audio #{idx} — “{prompt}”")
        print(f"  Path : {audio_path}")
        print(f"  CLAP cosine similarity score: {score:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    main()
