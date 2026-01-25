"""Joint multimodal diffusion on MNIST: images (continuous) + labels (discrete).

Demonstrates:
  1. MultimodalCorruptionProcess with FlowMatching (image) + MaskCosine (label).
  2. MultimodalLoss with per-modality DenoisingLoss.
  3. MultimodalSampler with EulerSampler (image) + DiscreteAncestralSampler (label).
  4. A single DiT backbone with a CLS token for joint image+label prediction.

Both image and label are generated simultaneously from pure noise — the model
learns to jointly denoise continuous pixel patches and a discrete class token.

Usage:
    uv run python examples/mnist_multimodal.py
"""

import struct
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from diffusionlab.models.components import SinusoidalEmbedding
from diffusionlab.models.dit import DiTBlock
from diffusionlab.processes.base import MultimodalCorruptionProcess
from diffusionlab.processes.discrete import MaskCosine
from diffusionlab.processes.gaussian import FlowMatching
from diffusionlab.sampling.base import MultimodalSampler, sample
from diffusionlab.sampling.continuous import EulerSampler
from diffusionlab.sampling.discrete import DiscreteAncestralSampler
from diffusionlab.sampling.schedules import uniform_schedule
from diffusionlab.training.base import MultimodalLoss
from diffusionlab.training.denoising import DenoisingLoss
from diffusionlab.training.schedules import logit_normal_time, uniform_time
from diffusionlab.typing import Prediction

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEED = 0
SAMPLES_PER_CLASS = 100
BATCH_SIZE = 64
NUM_STEPS = 50000
LR = 3e-4

PATCH_SIZE = 4  # 28/4 = 7 -> 49 patches of dim 16
SEQ_LEN = 49
PATCH_DIM = PATCH_SIZE * PATCH_SIZE  # 16

VOCAB_SIZE = 10  # digits 0-9; MaskCosine adds MASK at index 10
NUM_LABEL_TOKENS = VOCAB_SIZE + 1  # 11 (includes MASK)

# DiT image backbone
HIDDEN_DIM = 128
NUM_LAYERS = 4
NUM_HEADS = 4
HEAD_DIM = 32
MLP_DIM = 256
TIME_EMB_DIM = 64
COND_DIM = 128
LABEL_EMB_DIM = 32

SAMPLE_STEPS = 100
NUM_SAMPLES = 50


# ---------------------------------------------------------------------------
# Data loading (same as mnist_dit.py)
# ---------------------------------------------------------------------------
def load_mnist_images(path: Path) -> jax.Array:
    with open(path, "rb") as f:
        _magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        raw = jnp.frombuffer(f.read(), dtype=jnp.uint8).reshape(num, rows, cols)
    return raw.astype(jnp.float32) / 127.5 - 1.0


def load_mnist_labels(path: Path) -> jax.Array:
    with open(path, "rb") as f:
        _magic, _num = struct.unpack(">II", f.read(8))
        raw = jnp.frombuffer(f.read(), dtype=jnp.uint8)
    return raw.astype(jnp.int32)


def download_mnist(data_dir: Path) -> tuple[jax.Array, jax.Array]:
    import gzip
    import urllib.request

    data_dir.mkdir(parents=True, exist_ok=True)
    base = "https://storage.googleapis.com/cvdf-datasets/mnist"
    files = {
        "train-images-idx3-ubyte.gz": "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte.gz": "train-labels-idx1-ubyte",
    }
    for gz_name, raw_name in files.items():
        raw_path = data_dir / raw_name
        if not raw_path.exists():
            gz_path = data_dir / gz_name
            if not gz_path.exists():
                print(f"  Downloading {gz_name}...")
                urllib.request.urlretrieve(f"{base}/{gz_name}", gz_path)
            with gzip.open(gz_path, "rb") as gz:
                raw_path.write_bytes(gz.read())

    images = load_mnist_images(data_dir / "train-images-idx3-ubyte")
    labels = load_mnist_labels(data_dir / "train-labels-idx1-ubyte")
    return images, labels


# ---------------------------------------------------------------------------
# Patchify / unpatchify
# ---------------------------------------------------------------------------
def patchify(images: jax.Array) -> jax.Array:
    """(N, 28, 28) -> (N, 49, 16)."""
    n = images.shape[0]
    x = images.reshape(n, 7, PATCH_SIZE, 7, PATCH_SIZE)
    x = x.transpose(0, 1, 3, 2, 4)
    return x.reshape(n, SEQ_LEN, PATCH_DIM)


def unpatchify(patches: jax.Array) -> jax.Array:
    """(N, 49, 16) -> (N, 28, 28)."""
    n = patches.shape[0]
    x = patches.reshape(n, 7, 7, PATCH_SIZE, PATCH_SIZE)
    x = x.transpose(0, 1, 3, 2, 4)
    return x.reshape(n, 28, 28)


# ---------------------------------------------------------------------------
# Model: MultimodalPredictor
# ---------------------------------------------------------------------------
class MultimodalPredictor(eqx.Module):
    """Joint predictor for image (continuous) + label (discrete) via CLS token.

    A single DiT backbone processes image patches with a prepended CLS token.
    The CLS token is initialized from the noisy label embedding concatenated
    with the label diffusion time, providing the transformer with both the
    label content and its noise level. After self-attention across all tokens,
    the CLS token is extracted and projected to class logits, while the image
    patch tokens are projected to velocity predictions.

    Bidirectional coupling between modalities happens naturally through
    self-attention — no separate label head is needed.

    Attributes:
        label_emb: Embeds noisy label tokens (11 vocab) to ``LABEL_EMB_DIM``.
        cls_proj: Projects label embedding + label time to ``HIDDEN_DIM``.
        time_embed: Sinusoidal time embedding (shared, stateless).
        time_proj: Projects image time embedding to ``COND_DIM`` for AdaLN.
        cond_proj: Projects label embedding to ``COND_DIM``; added to time cond
            so image tokens get explicit label conditioning at every block.
        input_proj: Projects image patches from ``PATCH_DIM`` to ``HIDDEN_DIM``.
        pos_embed: Learnable positional embeddings, shape ``(50, HIDDEN_DIM)``.
        blocks: Stack of DiT transformer blocks with AdaLN-Zero.
        img_output_norm: LayerNorm before image output projection.
        img_output_proj: Projects image tokens to ``PATCH_DIM``.
        cls_output_norm: LayerNorm before CLS output projection.
        cls_output_proj: Projects CLS token to ``VOCAB_SIZE`` logits.
    """

    # Label embedding (noisy label token -> vector)
    label_emb: eqx.nn.Embedding

    # CLS token: label_emb || label_time -> hidden_dim
    cls_proj: eqx.nn.Linear

    # Time embedding & projection (image time -> AdaLN conditioning)
    time_embed: SinusoidalEmbedding
    time_proj: eqx.nn.MLP
    cond_proj: eqx.nn.MLP  # label_emb -> COND_DIM (additive with time for AdaLN)

    # Image patch input projection
    input_proj: eqx.nn.Linear

    # Positional embeddings: pos 0 = CLS, pos 1..49 = image patches
    pos_embed: jax.Array  # (SEQ_LEN + 1, HIDDEN_DIM)

    # Transformer blocks
    blocks: list[DiTBlock]

    # Image output head
    img_output_norm: eqx.nn.LayerNorm
    img_output_proj: eqx.nn.Linear

    # CLS (label) output head
    cls_output_norm: eqx.nn.LayerNorm
    cls_output_proj: eqx.nn.Linear

    def __init__(self, *, key):
        (
            key_emb,
            key_cls,
            key_tp,
            key_cp,
            key_ip,
            key_pe,
            key_io,
            key_co,
            *block_keys,
        ) = jax.random.split(key, 8 + NUM_LAYERS)

        # Label embedding: 11 tokens (10 classes + MASK) -> LABEL_EMB_DIM
        self.label_emb = eqx.nn.Embedding(NUM_LABEL_TOKENS, LABEL_EMB_DIM, key=key_emb)

        # CLS projection: concat(label_emb, label_time_emb) -> hidden_dim
        self.cls_proj = eqx.nn.Linear(
            LABEL_EMB_DIM + TIME_EMB_DIM, HIDDEN_DIM, key=key_cls
        )

        # Time embedding (stateless sinusoidal) and projection for AdaLN
        self.time_embed = SinusoidalEmbedding(emb_dim=TIME_EMB_DIM, max_period=10_000.0)
        self.time_proj = eqx.nn.MLP(
            in_size=TIME_EMB_DIM,
            out_size=COND_DIM,
            width_size=HIDDEN_DIM,
            depth=2,
            key=key_tp,
            activation=jax.nn.silu,
        )
        self.cond_proj = eqx.nn.MLP(
            in_size=LABEL_EMB_DIM,
            out_size=COND_DIM,
            width_size=HIDDEN_DIM,
            depth=2,
            key=key_cp,
            activation=jax.nn.silu,
        )

        # Image patch input projection: PATCH_DIM -> HIDDEN_DIM
        self.input_proj = eqx.nn.Linear(PATCH_DIM, HIDDEN_DIM, key=key_ip)

        # Positional embeddings: position 0 = CLS, positions 1..49 = patches
        self.pos_embed = jax.random.normal(key_pe, (SEQ_LEN + 1, HIDDEN_DIM)) * 0.02

        # Transformer blocks with AdaLN-Zero conditioning
        self.blocks = [
            DiTBlock(
                hidden_dim=HIDDEN_DIM,
                num_heads=NUM_HEADS,
                head_dim=HEAD_DIM,
                mlp_dim=MLP_DIM,
                cond_dim=COND_DIM,
                activation=jax.nn.gelu,
                key=k,
            )
            for k in block_keys
        ]

        # Image output head
        self.img_output_norm = eqx.nn.LayerNorm(HIDDEN_DIM)
        self.img_output_proj = eqx.nn.Linear(HIDDEN_DIM, PATCH_DIM, key=key_io)

        # CLS (label) output head
        self.cls_output_norm = eqx.nn.LayerNorm(HIDDEN_DIM)
        self.cls_output_proj = eqx.nn.Linear(HIDDEN_DIM, VOCAB_SIZE, key=key_co)

    def _forward_single(self, x_img, x_lbl, t_img, t_lbl):
        """Forward pass for a single sample.

        Args:
            x_img: Noisy image patches, shape ``(49, 16)``.
            x_lbl: Noisy label token, scalar int.
            t_img: Image diffusion time, scalar.
            t_lbl: Label diffusion time, scalar.

        Returns:
            Tuple of (image velocity prediction ``(49, 16)``,
            label logits ``(10,)``).
        """
        # Noisy label embedding (used for both AdaLN cond and CLS token)
        lbl_vec = self.label_emb(x_lbl)  # (LABEL_EMB_DIM,)

        # AdaLN conditioning: image time + label embedding (so image tokens
        # get explicit "which digit" signal at every block, like original DiT)
        img_t_emb = self.time_embed(t_img[None])[0]  # (TIME_EMB_DIM,)
        cond = self.time_proj(img_t_emb) + self.cond_proj(lbl_vec)  # (COND_DIM,)

        # Build CLS token from noisy label embedding + label time
        lbl_t_emb = self.time_embed(t_lbl[None])[0]  # (TIME_EMB_DIM,)
        cls_token = self.cls_proj(
            jnp.concatenate([lbl_vec, lbl_t_emb])
        )  # (HIDDEN_DIM,)

        # Project image patches to hidden dim
        img_tokens = jax.vmap(self.input_proj)(x_img)  # (49, HIDDEN_DIM)

        # Prepend CLS token -> full sequence (50, HIDDEN_DIM)
        h = jnp.concatenate([cls_token[None], img_tokens], axis=0)
        h = h + self.pos_embed

        # Transformer blocks (self-attention couples CLS <-> image patches)
        for block in self.blocks:
            h = block(h, cond)

        # Split CLS and image tokens
        cls_h = h[0]  # (HIDDEN_DIM,)
        img_h = h[1:]  # (49, HIDDEN_DIM)

        # Image output: norm + project -> velocity prediction
        img_h = jax.vmap(self.img_output_norm)(img_h)
        img_out = jax.vmap(self.img_output_proj)(img_h)  # (49, PATCH_DIM)

        # Label output: extract CLS -> norm + project -> class logits
        cls_h = self.cls_output_norm(cls_h)
        lbl_logits = self.cls_output_proj(cls_h)  # (VOCAB_SIZE,)

        return img_out, lbl_logits

    def __call__(self, x, s, t, cond, aux):
        """Predict both modalities jointly.

        Args:
            x: dict with "image" (batch, 49, 16) and "label" (batch,) int.
            s: ignored.
            t: dict with "image" (batch,) and "label" (batch,) times.
            cond: ignored (unconditional generation).
            aux: ignored.

        Returns:
            dict with "image" Prediction(kind="v_t") and
            "label" Prediction(kind="logits_x0").
        """
        img_preds, lbl_logits = jax.vmap(self._forward_single)(
            x["image"], x["label"], t["image"], t["label"]
        )

        return {
            "image": Prediction(value=img_preds, kind="v_t"),
            "label": Prediction(value=lbl_logits, kind="logits_x0"),
        }


# ===========================================================================
# 1. Load data
# ===========================================================================
print("Loading MNIST...")
data_dir = Path("examples/.data/mnist")
all_images, all_labels = download_mnist(data_dir)

# Take first SAMPLES_PER_CLASS per digit
indices = []
for digit in range(10):
    mask = jnp.where(all_labels == digit)[0]
    indices.append(mask[:SAMPLES_PER_CLASS])
indices = jnp.concatenate(indices)

images = all_images[indices]  # (1000, 28, 28)
labels = all_labels[indices]  # (1000,) int32
patches = patchify(images)  # (1000, 49, 16)
num_data = patches.shape[0]
print(f"  Using {num_data} samples ({SAMPLES_PER_CLASS} per class)")

# ===========================================================================
# 2. Create processes and model
# ===========================================================================
process_img = FlowMatching(data_shape=(SEQ_LEN, PATCH_DIM))
process_lbl = MaskCosine(vocab_size=VOCAB_SIZE, seq_shape=())
process = MultimodalCorruptionProcess({"image": process_img, "label": process_lbl})

key = jax.random.key(SEED)
key, key_model = jax.random.split(key)
model = MultimodalPredictor(key=key_model)

num_params = sum(p.size for p in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
print(f"  Model parameters: {num_params:,}")

# ===========================================================================
# 3. Training setup
# ===========================================================================
loss_img = DenoisingLoss(
    process=process_img, weight_fn=None, loss_fn=None, target_kind=None
)
loss_lbl = DenoisingLoss(
    process=process_lbl, weight_fn=None, loss_fn=None, target_kind=None
)
loss_obj = MultimodalLoss({"image": loss_img, "label": loss_lbl})

optimizer = optax.adamw(learning_rate=LR, weight_decay=1e-4)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))


@eqx.filter_jit
def train_step(model, opt_state, x_0_img, x_0_lbl, key):
    key_t_img, key_t_lbl, key_fwd = jax.random.split(key, 3)
    batch = x_0_img.shape[0]

    # Independent time sampling per modality
    t_img = logit_normal_time(key_t_img, batch, mean=0.0, std=1.0)
    t_lbl = uniform_time(key_t_lbl, batch)
    t = {"image": t_img, "label": t_lbl}

    x_0 = {"image": x_0_img, "label": x_0_lbl}
    x_t, aux = process.forward(key_fwd, x_0, t)

    def compute_loss(m):
        out = loss_obj(predictor=m, x_0=x_0, x_t=x_t, t=t, aux=aux, cond=None)
        return jnp.mean(out.loss), out

    (loss, loss_out), grads = eqx.filter_value_and_grad(compute_loss, has_aux=True)(
        model
    )
    updates, new_opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    new_model = eqx.apply_updates(model, updates)

    # Per-modality losses for logging
    img_loss = jnp.mean(loss_out.per_leaf["image"])
    lbl_loss = jnp.mean(loss_out.per_leaf["label"])
    return new_model, new_opt_state, loss, img_loss, lbl_loss


# ===========================================================================
# 4. Training loop
# ===========================================================================
losses = []
img_losses = []
lbl_losses = []
print("Training...")
for step in range(NUM_STEPS):
    key, key_batch, key_step = jax.random.split(key, 3)
    idx = jax.random.randint(key_batch, (BATCH_SIZE,), 0, num_data)
    x_0_img = patches[idx]
    x_0_lbl = labels[idx]
    model, opt_state, loss, img_loss, lbl_loss = train_step(
        model, opt_state, x_0_img, x_0_lbl, key_step
    )
    losses.append(float(loss))
    img_losses.append(float(img_loss))
    lbl_losses.append(float(lbl_loss))
    if (step + 1) % 500 == 0:
        print(
            f"  step {step + 1:5d}  loss = {loss:.4f}"
            f"  (img: {img_loss:.4f}, lbl: {lbl_loss:.4f})"
        )

# ===========================================================================
# 5. Joint unconditional sampling
# ===========================================================================
print("Sampling jointly...")
sampler = MultimodalSampler(
    {
        "image": EulerSampler(process_img),
        "label": DiscreteAncestralSampler(process_lbl),
    }
)

# Build multimodal time schedule
img_sched = uniform_schedule(SAMPLE_STEPS, NUM_SAMPLES)
lbl_sched = uniform_schedule(SAMPLE_STEPS, NUM_SAMPLES)
time_schedule = [
    {"image": img_sched[i], "label": lbl_sched[i]} for i in range(len(img_sched))
]

key, key_sample = jax.random.split(key)
generated = sample(
    key=key_sample,
    sampler=sampler,
    predictor=model,
    batch_size=NUM_SAMPLES,
    time_schedule=time_schedule,
)

gen_images = generated["image"]  # (NUM_SAMPLES, 49, 16)
gen_labels = generated["label"]  # (NUM_SAMPLES,) int

print(f"  Generated {NUM_SAMPLES} (image, label) pairs")
print(f"  Label distribution: {[int((gen_labels == d).sum()) for d in range(10)]}")

# ===========================================================================
# 6. Plotting
# ===========================================================================

# --- Grid of generated samples ---
ncols = 10
nrows = (NUM_SAMPLES + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.2, nrows * 1.4))
axes = axes.flatten()
for i in range(NUM_SAMPLES):
    ax = axes[i]
    img = unpatchify(gen_images[i : i + 1])[0]
    img = jnp.clip(img, -1, 1)
    ax.imshow(img, cmap="gray", vmin=-1, vmax=1)
    ax.set_title(f"{int(gen_labels[i])}", fontsize=10, color="C0")
    ax.set_xticks([])
    ax.set_yticks([])
# Hide unused axes
for i in range(NUM_SAMPLES, len(axes)):
    axes[i].set_visible(False)
plt.suptitle("Joint Multimodal Generation: Image + Label", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("examples/mnist_multimodal_samples.png", dpi=150, bbox_inches="tight")
print("Saved examples/mnist_multimodal_samples.png")

# --- Loss curves ---
fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 3.5))
window = 100


def plot_loss(ax, data, title, color):
    ax.plot(data, linewidth=0.5, alpha=0.3, color=color)
    if len(data) >= window:
        smoothed = jnp.convolve(
            jnp.array(data), jnp.ones(window) / window, mode="valid"
        )
        ax.plot(range(window - 1, len(data)), smoothed, linewidth=1.5, color=color)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.set_yscale("log")


plot_loss(ax1, losses, "Total Loss", "C0")
plot_loss(ax2, img_losses, "Image Loss (MSE)", "C1")
plot_loss(ax3, lbl_losses, "Label Loss (CE)", "C2")
plt.tight_layout()
plt.savefig("examples/mnist_multimodal_loss.png", dpi=150, bbox_inches="tight")
print("Saved examples/mnist_multimodal_loss.png")
plt.show()
