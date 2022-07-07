# Note: this code is being actively modified by justheuristic. If you want to change anything about it, please warn me.
import os

import torch
import transformers
from hivemind import get_logger, use_hivemind_log_handler

from src.client.remote_model import DistributedBloomForCausalLM

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__file__)


INITIAL_PEERS = os.environ.get("INITIAL_PEERS")
if not INITIAL_PEERS:
    raise RuntimeError("Must specify INITIAL_PEERS environment variable with one or more peer ids")
INITIAL_PEERS = INITIAL_PEERS.split()


MODEL_NAME = os.environ.get("MODEL_NAME")
if not MODEL_NAME:
    raise RuntimeError("Must specify MODEL_NAME as an index of a transformer block to be tested")

REF_NAME = os.environ.get("REF_NAME")


def test_full_model_exact_match(atol_forward=1e-5, atol_inference=1e-3, prefix="bloom6b3"):
    tokenizer = transformers.BloomTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistributedBloomForCausalLM.from_pretrained(MODEL_NAME, initial_peers=INITIAL_PEERS, prefix=prefix)
    assert len(model.transformer.h) == model.config.n_layer

    test_inputs = tokenizer("A cat sat on a mat", return_tensors="pt")["input_ids"]
    parallel_outputs = model.forward(test_inputs).logits
    assert torch.all(torch.isfinite(parallel_outputs))
    logger.info("Forward outputs are finite")

    if REF_NAME:
        ref_model = transformers.AutoModelForCausalLM.from_pretrained(REF_NAME)
        dummy_mask = torch.ones_like(test_inputs, dtype=torch.bool)
        # note: this creates a dummy mask to make the test compatible with older transformer versions
        # prior to https://github.com/huggingface/transformers/pull/17837
        ref_outputs = ref_model.forward(test_inputs, attention_mask=dummy_mask).logits
        assert torch.allclose(ref_outputs, parallel_outputs, rtol=0, atol=atol_forward)
    else:
        logger.warning("Did not test exact match with local model: REF_NAME environment variable is not set")

    embs = model.transformer.word_embeddings(test_inputs)
    embs = model.transformer.word_embeddings_layernorm(embs)
    recurrent_outputs = []
    with model.transformer.h.inference_session() as sess:
        for t in range(embs.shape[1]):
            recurrent_outputs.append(sess.step(embs[:, t : t + 1, :]))
    recurrent_outputs = torch.cat(recurrent_outputs, dim=1)
    recurrent_outputs = model.transformer.ln_f(recurrent_outputs)

    dictionary = model.transformer.word_embeddings.weight.t()
    recurrent_outputs = recurrent_outputs.to(dictionary.dtype)
    recurrent_outputs = (recurrent_outputs @ dictionary).float()
    assert torch.allclose(recurrent_outputs, parallel_outputs, rtol=0, atol=atol_inference)
    logger.info("Inference is consistent with forward")