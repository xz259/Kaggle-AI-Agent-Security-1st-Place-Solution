# Hybrid GCG

Hybrid GCG is a compact reference implementation of the optimization method
used in a first-place Kaggle agent-security solution. It uses gradients from an
upstream Transformers checkpoint to propose token substitutions, but lets the
quantized GGUF model running in llama.cpp decide which substitutions are real
improvements.

The included demonstration has one deliberately narrow objective:

> Given an existing two-hop prompt, make the first token of hop 2 an EOG token
> while preserving the observed hop-1 token sequence exactly.

No optimized prompts, model weights, competition SDK, or experiment archive are
included. Bring your own prompt and matching upstream/GGUF model pair.

## How it works

```
llama.cpp bootstrap
        |
        v
Freeze observed hop 1
        |
        v
BF16 HotFlip rankings
        |
        v
Sample or ridge-rerank radius-1 edits <---+
        |                                  |
        v                                  |
Cached-prefix GGUF shortlist               |
        |                                  |
        v                                  |
Full-context GGUF recheck                  |
        |                                  |
        v                                  |
Exact hop 1 preserved? -- no --------------+
        |
       yes
        |
        v
Accept and checkpoint
```

The cached-prefix path is only a shortlist mechanism. A mutation can be
accepted only after full-context GGUF rescoring, teacher-forced verification of
every observed hop-1 token, and a free greedy hop-1 replay.

## Repository scope

This release intentionally contains only:

- automatic llama.cpp trajectory bootstrapping;
- BF16 HotFlip proposal rankings;
- the Faster-GCG power-law rank sampler;
- optional GGUF-calibrated ridge reranking;
- radius-1 coordinate search;
- cached-prefix GGUF shortlisting;
- full-context GGUF acceptance and exact hop-1 hard rejection;
- persistent checkpoints and JSONL events.

The larger research codebase also explored semantic evolution, alternative
losses, and multi-token moves. Those branches are not required to understand or
run this demonstration.

## Installation

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[hf,test]"
```

Install `llama-cpp-python==0.3.34` separately using the wheel or build flags
appropriate for your CUDA, Metal, or CPU environment. For a normal CPU build:

```bash
pip install "llama-cpp-python==0.3.34"
```

The full search generally requires enough accelerator memory to keep the BF16
proposal model and GGUF validation model resident together.

Run the model-free tests first:

```bash
python -m unittest discover -s tests -v
```

## Prepare your trajectory

Copy the example directory and replace every placeholder:

```bash
cp -R examples/two_hop my_experiment
mv my_experiment/config.toml.example my_experiment/config.toml
mv my_experiment/prompt.txt.example my_experiment/prompt.txt
mv my_experiment/templates/hop1.txt.example my_experiment/templates/hop1.txt
mv my_experiment/templates/hop2.txt.example my_experiment/templates/hop2.txt
```

The templates operate on the exact raw text consumed by llama.cpp:

- `{{PROMPT}}` is the user-controlled token span and must occur exactly once.
- `{{HOP1}}`, when used, is the complete sampled hop-1 text, including its EOG
  token.
- `{{HOP1_VISIBLE}}`, when used, excludes the terminal EOG token.

For an agent or tool-use application, the hop-2 template must reproduce the
real post-tool context exactly. This small release uses a text-template adapter
instead of bundling a competition-specific SDK. Some frameworks parse the
sampled tool call and then render a canonical structured call into the next
context. In that case, put that canonical post-tool history directly in the
hop-2 template and omit `{{HOP1}}`; splicing in the sampled raw text would
optimize a different trajectory.

The prompt must tokenize as an independent span inside both templates. The
bootstrap command fails rather than silently optimizing across a changed token
boundary.

## Bootstrap

```bash
hybrid-gcg bootstrap --config my_experiment/config.toml
```

This command:

1. greedily decodes hop 1 with the GGUF;
2. records all sampled token IDs, including the EOG token;
3. constructs the configured hop-2 context;
4. confirms that the requested hop-2 target is one GGUF token and is classified
   as EOG by llama.cpp;
5. records the llama.cpp version plus the GGUF filename, size, and vocabulary
   size;
6. writes `runs/demo/baseline.json`.

Inspect the artifact without loading a model:

```bash
hybrid-gcg inspect --baseline my_experiment/runs/demo/baseline.json
```

## Search

Begin with a bounded smoke panel:

```toml
[search]
steps = 1
candidate_budget = 32
top_k = 64
full_recheck_count = 4
```

Then run:

```bash
hybrid-gcg search --config my_experiment/config.toml
```

For a larger run, increase `steps`, `candidate_budget`, and `top_k` only after
the smoke test reproduces the bootstrapped hop 1. Every panel is checkpointed;
rerunning the same command resumes from `checkpoint.json`.

To enable GGUF-calibrated reranking, add:

```toml
[search.ridge]
enabled = true
minimum_observations = 256
regularization = 0.01
exploration_fraction = 0.25
```

The first part of a panel collects GGUF labels from ordinary HotFlip proposals.
Once enough labels are available, two local ridge models rerank unseen
proposals by predicted target-NLL change and hop-2 margin gain. A reserved
fraction is still sampled from the gradient-ranked lists. These observations
are checkpointed while the incumbent is unchanged and cleared after an
accepted edit; the regression never bypasses full-context GGUF rescoring or the
exact hop-1 gate.

The default objective is the first-hop-2-token margin

```math
z_{\mathrm{EOG}}-\max_{v\ne\mathrm{EOG}}z_v.
```

Search stops once the EOG token is the greedy winner. `result.json` contains the
final prompt, margin, token IDs, and validation status.

## Important constraints

- The HF and GGUF tokenizers must assign identical IDs to the captured context.
- The requested EOG must also have the same single token ID in both tokenizers.
- Candidate prompts retain the same token length as the bootstrap prompt.
- Hop 1 is preserved token-for-token, not merely by decoded text.
- The hop-2 template is part of the experimental specification; an inaccurate
  template optimizes the wrong trajectory.
- Quantization, llama.cpp versions, hardware, cache boundaries, and kernels can
  change close logits. Revalidate final prompts in the deployment environment.
- Use this software only on models and systems you are authorized to evaluate.

For the competition narrative and motivation, see
[docs/solution.md](docs/solution.md). The implementation details and acceptance
rules are described in [docs/method.md](docs/method.md).

## Contributors

- [xz259](https://github.com/xz259) — competition solution, research, and project direction
- Codex (OpenAI) — implementation, testing, and documentation support

## License

Released under the [MIT License](LICENSE).
