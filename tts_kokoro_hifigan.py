import torch
import json
import soundfile as sf

from model import KokoroModel
from checkpoint_manager import load_phoneme_processor

from hifi_gan.models import Generator
from hifi_gan.env import AttrDict

# =========================
# DEVICE
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# PATHS
# =========================
KOKORO_DIR = "./models/kokoro_russian_a100"
KOKORO_CKPT = f"{KOKORO_DIR}/kokoro_russian_final.pth"

HIFIGAN_DIR = "./hifi_gan"
HIFIGAN_CONFIG = f"{HIFIGAN_DIR}/config.json"
HIFIGAN_GEN = f"{HIFIGAN_DIR}/g_02500000"

OUT_WAV = "output.wav"

# =========================
# LOAD KOKORO
# =========================
ckpt = torch.load(KOKORO_CKPT, map_location="cpu")
config = ckpt["config"]

phoneme_processor = load_phoneme_processor(KOKORO_DIR)
vocab_size = phoneme_processor.get_vocab_size()

model = KokoroModel(
    vocab_size=vocab_size,
    mel_dim=config.n_mels,
    hidden_dim=config.hidden_dim,
    n_encoder_layers=6,
    n_decoder_layers=6,
    max_decoder_seq_len=4000,
)

model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.to(DEVICE).eval()

print("✅ Kokoro loaded")

# =========================
# LOAD HiFi-GAN
# =========================
with open(HIFIGAN_CONFIG) as f:
    h = AttrDict(json.load(f))

hifigan = Generator(h).to(DEVICE)
state = torch.load(HIFIGAN_GEN, map_location=DEVICE)
hifigan.load_state_dict(state)
hifigan.eval()
hifigan.remove_weight_norm()

print("✅ HiFi-GAN loaded")

# =========================
# TEXT → MEL
# =========================
text = "Привет, это тест русской модели."

phoneme_ids = phoneme_processor.text_to_indices(text)
phoneme_tensor = torch.tensor(
    phoneme_ids,
    dtype=torch.long
).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    mel = model.forward_inference(
        phoneme_tensor,
        max_len=400
    )

# HiFi-GAN expects [B, 80, T]
mel = mel.transpose(1, 2)

print("Mel:", mel.shape)

# =========================
# MEL → WAV (HiFi-GAN)
# =========================
with torch.no_grad():
    audio = hifigan(mel)
    audio = audio.squeeze().cpu().numpy()

sf.write(
    OUT_WAV,
    audio,
    samplerate=h.sampling_rate,
    subtype="PCM_16"
)

print(f"🔊 WAV saved to {OUT_WAV}")
