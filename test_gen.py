#!/usr/bin/env python3

import time
import sys

import torch

from byprot.models.dplm import DiffusionProteinLanguageModel as DPLM
from generate_dplm import initialize_generation


MODEL = "airkingbd/dplm_150m"
SEQUENCE_LENGTH = 50
NUM_SEQUENCES = 5


def main():
    print("=" * 70)
    print("DPLM CPU smoke test")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    print(f"Python : {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA   : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print("WARNING: CUDA is available, but this test is intended for CPU.")

    device = torch.device("cpu")
    print(f"Device : {device}")
    print()

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"Loading model: {MODEL}")
    start = time.time()

    model = DPLM.from_pretrained(MODEL)
    model = model.to(device)
    model.eval()

    print(f"Model loaded in {time.time() - start:.1f} seconds")
    print()

    # ------------------------------------------------------------------
    # Prepare input
    # ------------------------------------------------------------------
    print(f"Preparing {NUM_SEQUENCES} sequence(s), length={SEQUENCE_LENGTH}")

    input_tokens = initialize_generation(
        length=SEQUENCE_LENGTH,
        num_seqs=NUM_SEQUENCES,
        tokenizer=model.tokenizer,
        device=device,
    )

    print("Input prepared.")
    print()

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------
    print("Generating...")
    print("CPU inference can take a while.")
    print()

    start = time.time()

    with torch.no_grad():
        samples = model.generate(
            input_tokens=input_tokens,
            max_iter=50,
        )

    elapsed = time.time() - start

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------
    sequences = model.tokenizer.batch_decode(
        samples,
        skip_special_tokens=True,
    )

    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    for i, sequence in enumerate(sequences, start=1):
        sequence = "".join(sequence.split())

        print(f">dplm_test_{i}")
        print(sequence)
        print(f"Length: {len(sequence)}")
        print()

    print(f"Generation time: {elapsed:.1f} seconds")
    print("=" * 70)
    print("DPLM test completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()