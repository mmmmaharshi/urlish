# Baseline reimplementation notes

Issue #20 asks which three published detectors to reimplement as
baselines, and the cheapest faithful reimplementation of each.
The README already cites URLNet, URLTran, and Phishpedia.

This note records, for each: input features, model architecture, training
protocol, whether code or a checkpoint exists, and the effort to reproduce
on the PhreshPhish corpus. PhreshPhish facts are in
[phreshphish.md](phreshphish.md).

Every claim cites its source. Items that could not be verified are marked
"unverified".

## 1. URLNet (Le et al., 2018)

Source: arXiv 1802.03162 v2 (HTML, accessed 2026-10-26);
official code at `github.com/Antimalweb/URLNet`.

- Input features: the raw URL string only. No host, DNS, or page content.
  - Character branch: the URL as a sequence of characters, padded or
    truncated to 200 characters. 96 unique characters after rare-character
    folding (arXiv 1802.03162 v2, Section 3.2).
  - Word branch: the URL split into words by special characters, padded or
    truncated to 200 words; special characters also treated as words
    (Sections 3.3.1 and 3.3.2).
- Model architecture: two parallel CNN branches.
  - Character-level CNN: embedding size 32, four filter widths (3, 4, 5, 6)
    with 256 filters each, max-pooling, dropout, fully connected layer.
  - Word-level CNN: same CNN shape; word embedding size 32; advanced
    embedding adds a character-level word embedding (each word padded or
    truncated to 20 characters) summed with the word embedding.
  - Branch outputs concatenated, then 4 fully connected layers to a
    binary classifier (Sections 3.2, 3.3.3, 3.4).
- Training protocol: Adam optimizer, dropout 0.5, 5 epochs. Two corpus
  sizes reported: 1 million and 5 million URLs sampled from 5 million
  training URLs; test on 10 million URLs. Labels: negative if on no VirusTotal
  blacklist, malicious if on 5 or more of 64 blacklists; URLs on 1 to 4
  blacklists discarded. Dataset sorted by query timestamp; first 60% of
  URLs used for training (5 million sampled), last 40% for testing
  (10 million sampled). Duplicate URLs removed; per-domain frequency capped
  below 5% (Section 4.1.1, Table 1).
- Pretrained checkpoint or code: official TensorFlow 1.8 / tflearn 0.3
  code exists at `Antimalweb/URLNet`. No pretrained checkpoint is published
  in the repo as of the search date (unverified: the repo's `runs/`
  checkpoints are training outputs, not a release).
- Effort to reproduce on PhreshPhish: medium. PhreshPhish has URL and label
  columns, so the input is available. The protocol needs a timestamp sort
  and a 60/40 temporal cut; PhreshPhish's `date` column is a collection
  date (phreshphish.md, Section 2), so a temporal split is possible but is
  a URL-level approximation of PhreshPhish's own 70/30 cut plus an
  HTML-based leakage filter. URLNet's per-domain frequency cap and
  blacklist-threshold labelling do not apply to PhreshPhish, because
  PhreshPhish labels are `phish`/negative from its own collection. Training
  is cheap (no GPU needed for the CNN at this scale, though a GPU helps).
  Estimated effort: 1 to 2 weeks, mostly data adaptation and tuning.

## 2. URLTran (Maneriker et al., 2021)

Source: arXiv 2106.05256 v3 (HTML, accessed 2026-10-26);
code at `github.com/bfilar/URLTran`.

- Input features: the raw URL string only, tokenised into subword tokens.
  - URLTran_BERT: BERT wordpiece tokenizer.
  - URLTran_RoBERTa: BPE tokenizer.
  - URLTran_CustVoc: character-level or byte-level BPE with vocabulary
    sizes 1,000 or 10,000, trained on the URL corpus.
  Token sequence length varied between 128 and 256 tokens
  (Section 3.1.1, Appendix A).
- Model architecture: a three-stage transformer pipeline.
  - Subword tokenizer produces the token sequence.
  - Transformer encoder (BERT or RoBERTa architecture, or a custom
    transformer with 3, 6, or 12 layers) produces an embedding.
  - A single dense two-class classification layer on the pooled `[CLS]`
    token. A dense layer of vocabulary size is used for masked token
    prediction during pre-training (Sections 3.1.2, 3.2).
- Training protocol:
  - Data: six weeks of Microsoft Edge/IE browsing telemetry, summer 2019.
    Four weeks train, one week validate, one week test. Negative set
    down-sampled to a 1:20 phishing-to-negative ratio. Train size 1,039,413;
    validation 259,854; test 1,784,155 (8,742 phishing). Labels are
    analyst-verified (Section 2).
  - Pre-training: masked language modelling, 15% of tokens masked
    (80% replaced, 10% unchanged, 10% random token), dynamic masking.
    URLTran_CustVoc pre-trained on the training URLs only.
  - Fine-tuning: Adam optimizer with cross-entropy loss; triangular
    learning-rate scheduler. Training time for URLTran_BERT: 4 hours 57
    minutes on an NVIDIA V100; inference 0.36 ms per sample
    (Section 3.2, Section 4).
  - Adversarial augmentation (optional): homoglyph substitution, compound
    word splits with hyphens, and query-parameter reordering
    (Section 3.3).
- Pretrained checkpoint or code: PyTorch 1.2 code exists at
  `bfilar/URLTran`, extending Hugging Face and Fairseq. No pretrained
  checkpoint is published (unverified: the repo contains training scripts
  only).
- Effort to reproduce on PhreshPhish: high. The model needs a transformer
  stack (PyTorch + Hugging Face), GPU for pre-training and fine-tuning, and
  either a BERT/RoBERTa backbone download or a custom pre-training run on
  the URL corpus. URL-only input is available in PhreshPhish, but the
  class ratio (1:20 in URLTran) differs from PhreshPhish's prevalence; the
  temporal split maps onto PhreshPhish's collection dates only as an
  approximation. Estimated effort: 3 to 6 weeks, dominated by pre-training
  and hyperparameter search.

## 3. Phishpedia (Lin et al., 2021)

Source: USENIX Security 2021 (paper page and presentation video);
code at `github.com/lindsey98/Phishpedia`.

- Input features: a URL and a screenshot of the rendered page. Phishpedia
  is not a URL-only detector. It detects the brand logo in the screenshot
  with an object detector and compares it to a reference list of brand logos
  (USENIX page; repo README).
- Model architecture: two-stage pipeline.
  - Stage 1: Faster R-CNN object detector (ResNet backbone) that predicts
    logo and input boxes in the screenshot. Weights file
    `rcnn_bet365.pth`, config `faster_rcnn.yaml`.
  - Stage 2: Siamese network based on BiT-M-R50x1 (ResNet v2) that
    matches the cropped logo against the reference brand-logo list. Weights
    file `resnetv2_rgb_new.pth.tar`. A domain map (`domain_map.pkl`) maps
    brands to their legitimate domains. Decision rule: if the detected
    brand's domain does not match the page's domain, the page is phishing
    (repo `logo_recog.py`, `logo_matching.py`, `setup.sh`).
- Training protocol:
  - Object detector trained on 30,000 labelled legitimate websites with
    annotated logo and input boxes. Siamese trained in two stages: first
    pre-trained on a large logo dataset (Logo-2K+), then fine-tuned on
    Phishpedia's own logo set covering the top 181 brands attacked in
    OpenPhish. No positives are used for training
    (presentation video; repo).
  - Reference list covers the top 181 brands most attacked in OpenPhish
    (presentation video).
- Pretrained checkpoint or code: official code exists and `setup.sh`
  downloads pretrained weights (`rcnn_bet365.pth`,
  `faster_rcnn.yaml`, `resnetv2_rgb_new.pth.tar`, `expand_targetlist.zip`,
  `domain_map.pkl`) via Google Drive. This is the only baseline of the
  three with published weights (verified from `setup.sh`).
- Effort to reproduce on PhreshPhish: very high, and only partially
  comparable. PhreshPhish is a URL corpus; it has no screenshots or HTML
  in the columns we fetch (phreshphish.md, Section 1). Running Phishpedia
  requires rendering each URL to a screenshot, which needs a browser and
  network access, plus the reference brand list. It also cannot be
  evaluated as a URL-only detector, so it does not fit the benchmark's
  client-side URL-only constraint. Estimated effort: 4 to 8 weeks,
  dominated by screenshot collection and rendering infrastructure. A
  faithful comparison would need a separate screenshot-based evaluation
  track, not the URL benchmark.

## 4. PhishAgent (Cao et al., 2024)

Source: arXiv 2408.10738 v3 (HTML, accessed 2026-10-26). Note: the
README cites Phishpedia, not PhishAgent. PhishAgent is included here
because the task named it.

- Input features: URL, HTML, and a screenshot. PhishAgent is a multimodal
  agent, not a URL-only detector. It uses GPT-3.5-turbo-instruct on HTML
  and URL text, ChatGPT-4o on screenshots, Google search for brand
  information, and the Phishpedia logo extractor for logo detection
  (arXiv 2408.10738 v3, Section 3 and appendix).
- Model architecture: a ReAct-style agent that extracts a target brand
  from HTML and URL, detects logos in the screenshot with the Phishpedia
  Faster R-CNN extractor, retrieves candidate brands from an offline
  knowledge base, and checks domain-brand consistency. No single trained
  classifier; behaviour depends on LLM calls and retrieval
  (arXiv 2408.10738 v3).
- Training protocol: none in the traditional sense. Components are
  off-the-shelf (Phishpedia logo extractor, GPT-3.5, GPT-4o). Evaluated
  on three datasets; the code and knowledge base are withheld for
  commercial reasons (KnowPhish repo README states the detector is
  deployed commercially and code is not released; PhishAgent is built on
  it).
- Pretrained checkpoint or code: no public code or checkpoint found.
  PhishAgent builds on KnowPhish, whose authors state they cannot release
  code or the knowledge base (KnowPhish repo README). Unverified: whether
  any third-party reimplementation exists.
- Effort to reproduce on PhreshPhish: not applicable as a URL-only
  baseline. It requires screenshots, HTML, LLM API access, and Google
  search, and it is not deterministic or cheap. It cannot be reproduced
  faithfully on PhreshPhish without the withheld knowledge base.
  Estimated effort: not estimable; likely blocked on unavailable data.

## Recommendation

For a URL-only benchmark on PhreshPhish, the realistic baselines are
URLNet and URLTran. URLNet is the cheaper of the two and has a public
codebase. URLTran needs a GPU and pre-training but is the stronger
transformer baseline. Phishpedia is not a URL-only detector and needs
screenshots; it belongs in a separate visual evaluation track, not in this
benchmark. PhishAgent depends on withheld code and data and is not
reproducible.
