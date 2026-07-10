"""
Linear Probe on GPT-2 Small — finding a "sentiment direction" in the residual stream.
Phase 1 exercise.

THE PIPELINE (what you are building, top to bottom):
  1. Environment setup (device = "mps")
  2. Load gpt2-small
  3. Build dataset: 20 positive + 20 negative prompts (+ integer labels)
  4. Forward pass with run_with_cache -> grab the layer-6 residual stream
     at the LAST token position of each prompt
  5. Tensors -> numpy: build X (features) and y (labels)
  6. Train a sklearn LogisticRegression (the "probe")
  7. Test on a brand-new prompt

You fill in every TODO. The comments tell you the concept + expected shapes,
not the code.
"""

# ---------------------------------------------------------------------------
# STEP 1: Imports & environment
# ---------------------------------------------------------------------------
import functools
import torch
import numpy as np
from transformer_lens import HookedTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

DEVICE = 'mps'


# ---------------------------------------------------------------------------
# STEP 2: Load the model
# ---------------------------------------------------------------------------
model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
#
# Answer in a comment: how many blocks does gpt2-small have, what is d_model,
# and why is layer 6 a sensible "middle"?


# ---------------------------------------------------------------------------
# STEP 3: Dataset
# ---------------------------------------------------------------------------
# Every topic (film, food, travel, work, weather, tech, sports...) appears in
# BOTH lists on purpose, so the only signal separating them is sentiment.
positive_prompts = [
    "I absolutely loved this movie from start to finish.",
    "The dinner last night was delicious and satisfying.",
    "Our vacation to the coast was pure bliss.",
    "She got promoted and could not stop smiling.",
    "What a beautiful, sunny morning it is today.",
    "This phone works flawlessly and feels amazing.",
    "The concert was thrilling and the crowd was ecstatic.",
    "He is such a kind and generous friend.",
    "That novel was gripping and beautifully written.",
    "Our team won the championship and we celebrated all night.",
    "The new cafe downtown is wonderful and cozy.",
    "I aced the exam and felt incredibly proud.",
    "The puppy is adorable and full of joy.",
    "This software update made everything faster and smoother.",
    "The garden is blooming and it looks gorgeous.",
    "Customer service was helpful and resolved my issue instantly.",
    "The hotel room was spotless and comfortable.",
    "Grandma's recipe turned out perfect and everyone loved it.",
    "The presentation went great and my boss was impressed.",
    "I feel refreshed and happy after a good night's sleep.",
]

negative_prompts = [
    "I completely hated this movie and turned it off halfway.",
    "The dinner last night was bland and disgusting.",
    "Our vacation to the coast was a miserable disaster.",
    "He got fired and felt utterly devastated.",
    "What a gloomy, freezing morning it is today.",
    "This phone keeps crashing and feels cheap.",
    "The concert was boring and the sound was awful.",
    "He is such a rude and selfish person.",
    "That novel was dull and painfully written.",
    "Our team lost badly and everyone was heartbroken.",
    "The new cafe downtown is dirty and unwelcoming.",
    "I failed the exam and felt deeply ashamed.",
    "The stray dog looked sick and terrified.",
    "This software update broke everything and slowed it down.",
    "The garden is dying and it looks dreadful.",
    "Customer service was useless and ignored my complaint.",
    "The hotel room was filthy and uncomfortable.",
    "The recipe turned out terrible and nobody ate it.",
    "The presentation flopped and my boss was furious.",
    "I feel exhausted and depressed after a sleepless night.",
]

# Convention: positive = 1, negative = 0.
prompts = positive_prompts + negative_prompts
labels = [1] * 20 + [0] * 20


# ---------------------------------------------------------------------------
# STEP 4: Forward pass + cache the layer-6 residual stream at the LAST token
# ---------------------------------------------------------------------------
# The residual stream at layer 6 has THREE possible hook points. Pick one and
# be able to justify it:
#     cache["resid_pre",  6]   # stream BEFORE block 6 does anything
#     cache["resid_mid",  6]   # after attention, before the MLP
#     cache["resid_post", 6]   # after the full block 6
# Each has shape [batch, position, d_model] = [1, seq_len, 768] for one prompt.
#
# DESIGN QUESTION (answer BEFORE you code): if you run all 40 prompts in one
# batch, they have different token lengths, so the tokenizer PADS them. Then
# position -1 for a short prompt is a PAD token, not its real last word.
# How do you avoid grabbing a pad? Two options:
#     A) loop one prompt at a time (no padding to worry about)
#     B) batch, but track each prompt's true length and index it directly
#
features = []                      # collect one 768-vector per prompt
for prompt in prompts:
  logits, cache = model.run_with_cache(prompt)
  resid = cache["resid_post", 6]     # what shape?
  last  = resid[0, -1]     # index the POSITION dim -> shape [768]
  features.append(last)


# ---------------------------------------------------------------------------
# STEP 5: Tensors -> numpy
# ---------------------------------------------------------------------------
# sklearn wants X of shape [n_samples, n_features] = [40, 768], y of shape [40].
# TODO: stack your list of 768-vectors into one tensor of shape [40, 768]
tensor_features = torch.stack(features)  # shape [40, 768]
X=tensor_features.detach().cpu().numpy()
y = np.array(labels)

# Hold out 25% (10 prompts) to measure GENERALIZATION, not memorization.
# stratify=y keeps the pos/neg ratio balanced in both halves (5 pos / 5 neg in test).
# random_state fixes the shuffle so the split is reproducible run-to-run.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)


# ---------------------------------------------------------------------------
# STEP 6: Train the probe
# ---------------------------------------------------------------------------
probe = LogisticRegression(max_iter=1000)
probe.fit(X_train, y_train)                       # learn ONLY from the training slice
print("train acc:", probe.score(X_train, y_train))  # will still be ~1.0 (memorization)
print("test  acc:", probe.score(X_test, y_test))    # the HONEST generalization number

# k-fold CV: rotate through 5 folds over ALL 40 examples and average.
# cross_val_score CLONES a fresh probe for each fold, so `probe` above is untouched.
cv_scores = cross_val_score(probe, X, y, cv=5)
print("cv folds:", cv_scores)
print("cv mean +/- std:", cv_scores.mean(), cv_scores.std())


# ---------------------------------------------------------------------------
# STEP 7: Predict on a brand-new prompt
# ---------------------------------------------------------------------------
test_prompt = "The book I just read was quite disturbing and badly written."         # something NOT in your dataset
logits, cache = model.run_with_cache(test_prompt)
test_resid = cache["resid_post", 6]
test_last = test_resid[0, -1]
test_feature = test_last.detach().cpu().numpy().reshape(1, -1)
print(probe.predict(test_feature), probe.predict_proba(test_feature))


# ---------------------------------------------------------------------------
# TEST B: STEERING VECTOR
# ---------------------------------------------------------------------------
# Instead of READING the sentiment direction (the probe), we now WRITE it back
# into the model and watch its behaviour change.
#
# STEP 8: Build the steering vector = mean(positive activations) - mean(negative)
# tensor_features is [40, 768]: rows 0:20 are positive, rows 20:40 are negative.
# Result shape: [768]. It stays on mps (it came from tensor_features).
steering_vector = tensor_features[0:20].mean(dim=0) - tensor_features[20:40].mean(dim=0)
print(steering_vector.norm(), cache["resid_post", 6][0, -1].norm())

# ---------------------------------------------------------------------------
# STEP 9: The hook function
# ---------------------------------------------------------------------------
# TransformerLens calls this during the forward pass. `activation` is the live
# residual stream at our hook point, shape [batch, pos, 768]. Whatever you
# RETURN replaces it. functools.partial (below) binds the two extra args so the
# signature TransformerLens actually calls is just (activation, hook).
def steering_hook(activation, hook, steering_vector, coefficient):
    activation = activation + coefficient * steering_vector
    return activation


coefficient = 8.0
hook_name = "blocks.6.hook_resid_post"   # same point as cache["resid_post", 6]
hook_fn = functools.partial(
    steering_hook, steering_vector=steering_vector, coefficient=coefficient
)


# ---------------------------------------------------------------------------
# STEP 10: Run WITH vs WITHOUT the hook and compare the next-token prediction
# ---------------------------------------------------------------------------
def top_next_tokens(logits, k=5):
    top = logits[0, -1].topk(k).indices.tolist()
    return [model.tokenizer.decode(idx) for idx in top]


steer_prompt = "The movie I watched last night was"
baseline_logits = model(steer_prompt)
steered_logits = model.run_with_hooks(
    steer_prompt,
    fwd_hooks=[(hook_name, hook_fn)],
)
print("baseline:", top_next_tokens(baseline_logits))
print("steered :", top_next_tokens(steered_logits))


# ---------------------------------------------------------------------------
# STEP 11: Coefficient sweep — how hard must we push, and when does it break?
# ---------------------------------------------------------------------------
# For each strength we rebuild the partial (a new coefficient baked in) and re-run.
# c=0 is the CONTROL: 0 * steering_vector adds nothing, so it must match baseline.
for c in [0, 4, 8, 16, 32, 64]:
    swept_fn = functools.partial(
        steering_hook, steering_vector=steering_vector, coefficient=c
    )
    swept_logits = model.run_with_hooks(steer_prompt, fwd_hooks=[(hook_name, swept_fn)])
    print(f"c={c:>3}:", top_next_tokens(swept_logits))


# ---------------------------------------------------------------------------
# STEP 12: A better lens — generate a full continuation with steering ON
# ---------------------------------------------------------------------------
# Top-5 single next-token is a keyhole. A whole sentence makes sentiment obvious.
# model.hooks(...) is a CONTEXT MANAGER: the hook stays registered for EVERY token
# generated inside the `with` block, then auto-removes when the block exits.
# do_sample=False = greedy (deterministic), so baseline vs steered is a fair compare.
gen_fn = functools.partial(
    steering_hook, steering_vector=steering_vector, coefficient=8.0
)
baseline_text = model.generate(steer_prompt, max_new_tokens=30, do_sample=False, verbose=False)
with model.hooks(fwd_hooks=[(hook_name, gen_fn)]):
    steered_text = model.generate(steer_prompt, max_new_tokens=30, do_sample=False, verbose=False)
print("baseline text:", baseline_text)
print("steered  text:", steered_text)
