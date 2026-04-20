# Reference 논문 정리

## Category

* AE (Acoustic Encoder)
* SE (Semantic Encoder)
* ALM (Audio Language Model)
* Benchmark
* Representation Learning
* Alignment / Disentanglement
* Other

---

## Paper Summary Table


| BibTeX Alias | Title                                                                                                 | Venue (Year)           | Category                    | Key Idea / Contribution                                                                                                                                            | #  |
| ------------ | ----------------------------------------------------------------------------------------------------- | ---------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -- |
| `LSTN26`     | Do Audio LLMs Really LISTEN, or Just Transcribe? Measuring Lexical vs. Acoustic Emotion Cues Reliance | EACL (2026)            | Benchmark                   | Introduces LISTEN, a diagnostic benchmark that measures whether audio LLMs rely on lexical or acoustic emotion cues. SE+LM이 acoustic task에서 못한다는 걸 보여줌. | 1  |
| `Q2A24`      | Qwen2-Audio Technical Report                                                                          | arXiv (2024)           | ALM                         | Presents Qwen2-Audio, a large audio-language model built around strong speech-semantic understanding and instruction following.                                    | 2  |
| `QA23`       | Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models     | arXiv (2023)           | ALM                         | Proposes a unified audio-language model for broad audio understanding, representative of semantic-first ALM design.                                                | 3  |
| `GPT4o24`    | GPT-4o                                                                                                | OpenAI Index (2024)    | ALM                         | Introduces a natively multimodal frontier model that processes text, audio, and vision in an integrated framework.                                                 | 4  |
| `Wspr23`     | Whisper                                                                                               | GitHub / OpenAI (2023) | SE                          | Provides a large-scale ASR model whose representations are optimized for transcription-oriented semantic extraction.                                               | 5  |
| `Q3ASR26`    | Qwen3-ASR Technical Report                                                                            | arXiv (2026)           | SE                          | Presents an ASR-focused model emphasizing strong speech recognition performance and semantic recovery from audio.                                                  | 6  |
| `RPA26`      | Resurfacing Paralinguistic Awareness in Large Audio Language Models                                   | arXiv (2026)           | Alignment / Disentanglement | Analyzes how semantic bottlenecks weaken paralinguistic awareness in large audio language models and how such awareness can be recovered.                          | 7  |
| `FLM25`      | Frozen Large Language Models Can Perceive Paralinguistic Aspects of Speech                            | Interspeech (2025)     | Motivation                  | Shows that frozen LLMs can recover semantic and paralinguistic information when paired with an acoustic front-end. (저희의 선행 연구격)                            | 8  |
| `DAM26`      | Scaling Open Discrete Audio Foundation Models with Interleaved Semantic, Acoustic, and Text Tokens    | arXiv (2026)           | Alignment / Disentanglement | Explores joint modeling of semantic, acoustic, and text tokens to build scalable open audio foundation models.                                                     | 9  |
| `LMA26`      | When Audio-LLMs Don't Listen: A Cross-Linguistic Study of Modality Arbitration                        | arXiv (2026)           | Alignment / Disentanglement | Studies how audio LLMs arbitrate between modalities across languages and when they fail to truly attend to audio.                                                  | 10 |
| `XCodec25` | Codec Does Matter: Exploring the Semantic Shortcoming of Codec for Audio Language Model | AAAI (2025) | Representation Learning | Shows acoustic codecs lack semantics; proposes semantic-guided codec (X-Codec) | 11 |
| `SpTok24` | SpeechTokenizer: Unified Speech Tokenizer for Speech Language Models | ICLR (2024) | Representation Learning | Disentangles semantic vs acoustic info across RVQ layers | 12 |
| `Moshi24` | Moshi: a Speech-Text Foundation Model for Real-Time Dialogue | arXiv (2024) | ALM | Real-time full-duplex speech LM with low-latency codec | 13 |
| `WavTok25` | WavTokenizer: an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling | ICLR (2025) | AE | Acoustic tokenizer optimized for ALM | 14 |
| `ALMTok25` | ALMTokenizer: A Low-bitrate and Semantic-rich Audio Codec Tokenizer for Audio Language Modeling | arXiv (2025) | Representation Learning | Low-bitrate semantic-rich codec tokenizer | 15 |
| `EAA25` | Emotion-Aware Audio Large Language Models with Dual Encers | Interspeech (2025) | Alignment / Disentanglement | Dual encoder (semantic + acoustic) fusion | 16 |
| `PaM25` | Enhancing Speech LLMs with Prompt-Aware Mixture of Audio Encoders | EMNLP (2025) | Alignment / Disentanglement | Mixture-of-encoders with task-dependent routing | 17 |
| `Entropy26` | Rethinking Entropy Allocation in LLM-based ASR | arXiv (2026) | Representation Learning | Analyzes encoder vs LLM information allocation via CKA | 18 |
| `WavLM22` | WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing | IEEE JSTSP (2022) | SE | SSL encoder capturing both acoustic and semantic info | 19 |
| `HuBERT21` | HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction | IEEE TASLP (2021) | SE | Masked prediction-based semantic SSL encoder | 20 |
| `SurveySLM24` | A Survey on Speech Large Language Models | arXiv (2024) | Other | Overview of encoder-projector-LLM architecture | 21 |
| `SurveyALM24` | Towards Audio Language Modeling: An Overview | arXiv (2024) | Other | Overview of ALM and tokenization strategies | 22 |
| `DATSurvey24` | Discrete Audio Tokens: More Than a Survey | arXiv (2024) | Other | Taxonomy of semantic vs acoustic tokens | 23 |
| `DAC23` | Descript Audio Codec (DAC) | NeurIPS (2023) | AE | High-fidelity neural audio codec | 24 |
| `DACVAE24` | DAC-VAE | GitHub (2024) | AE | Continuous latent acoustic encoder (reconstruction-based) | 25 |
| `PAL25` | Probing Audio Encoders via LLMs | arXiv (2025) | Representation Learning | Studies encoder→LLM transfer; attention injection improves performance | 26 |
| `IS26Chal` | Audio Encoder Capability Challenge for LALMs | Interspeech (2026) | Benchmark | Benchmarks different encoders under unified LLM setting | 27 |
---

## Notes

* **Category**: 새로운 카테고리 추가 시 위에 명시
* **Key Idea / Contribution**: 한 줄 요약 (문제 정의 + 핵심 접근)
* **BibTeX Alias**: `ABBRyear[2:]` 형태 권장 (e.g., `AAYN17` for Attention is All You Need, 2017)
