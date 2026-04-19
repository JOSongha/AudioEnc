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

---

## Notes

* **Category**: 새로운 카테고리 추가 시 위에 명시
* **Key Idea / Contribution**: 한 줄 요약 (문제 정의 + 핵심 접근)
* **BibTeX Alias**: `ABBRyear[2:]` 형태 권장 (e.g., `AAYN17` for Attention is All You Need, 2017)
