# Awesome-ML-Security

A curated list of awesome machine learning security references, guidance, tools, and more.

**Table of Contents**

- [Awesome-ML-Security](#awesome-ml-security)
  - [Relevant work, standards, literature](#relevant-work-standards-literature)
    - [CIA of the model](#cia-of-the-model)
      - [Confidentiality](#confidentiality)
      - [Integrity](#integrity)
      - [Availability](#availability)
    - [Degraded model performance](#degraded-model-performance)
    - [ML-Ops](#ml-ops)
    - [AI’s effect on attacks/security elsewhere](#ais-effect-on-attackssecurity-elsewhere)
      - [Self-driving cars](#self-driving-cars)
  - [Regulatory actions](#regulatory-actions)
    - [US](#us)
    - [EU](#eu)
  - [Safety standards](#safety-standards)
  - [Taxonomies and frameworks](#taxonomies-and-frameworks)
  - [Security tools and techniques](#security-tools-and-techniques)
    - [API probing](#api-probing)
    - [Model backdoors](#model-backdoors)
  - [DeepFakes, disinformation, and abuse](#deepfakes-disinformation-and-abuse)
    - [AI's impact on security tools and techniques](#ais-impact-on-security-tools-and-techniques)
  - [Background information](#background-information)
  - [Notable incidents](#notable-incidents)
  - [Notable harms](#notable-harms)

## Relevant work, standards, literature

### CIA of the model
Membership attacks, model inversion attacks, model extraction, adversarial perturbation, prompt injections, etc.
* [Towards the Science of Security and Privacy in Machine Learning](https://arxiv.org/abs/1611.03814)
* [SoK: Machine Learning Governance](https://arxiv.org/abs/2109.10870)
* [Not with a Bug, But with a Sticker: Attacks on Machine Learning Systems and What To Do About Them](https://www.goodreads.com/book/show/125075266-not-with-a-bug-but-with-a-sticker)
* [On the Impossible Safety of Large AI Models](https://arxiv.org/abs/2209.15259)

#### Confidentiality
Reconstruction (model inversion; attribute inference; gradient and information leakage), theft of data, Membership inference and reidentification of data, Model extraction (model theft), property inference (leakage of dataset properties), etc.
* [awesome-ml-privacy-attacks](https://github.com/stratosphereips/awesome-ml-privacy-attacks)
* [Privacy Side Channels in Machine Learning Systems](https://arxiv.org/abs/2309.05610#:~:text=Most%20current%20approaches%20for%20protecting,%2C%20output%20monitoring%2C%20and%20more)

#### Integrity 
Backdoors/neural trojans (same as for non-ML systems), adversarial evasion (perturbation of an input to evade a certain classification or output), data poisoning and ordering (providing malicious data or changing the order of the data flow into an ML model). 
* [Adversarial Machine Learning: A Systematic Survey of Backdoor Attack, Weight Attack and Adversarial Examples](https://arxiv.org/abs/2302.09457)
* [Poisoning Web-Scale Training Datasets is Practical](https://arxiv.org/abs/2302.10149)
* [Planting Undetectable Backdoors in Machine Learning Models](https://arxiv.org/abs/2204.06974)
* [Motivating the Rules of the Game for Adversarial Example Research](https://arxiv.org/abs/1807.06732)
* [Manipulating SGD with Data Ordering Attacks](https://arxiv.org/abs/2104.09667)
* [Adversarial reprogramming](https://arxiv.org/abs/1806.11146) - repurposing a model for a different task than its original intended purpose 
* [Model spinning attacks](https://arxiv.org/abs/2107.10443) (meta backdoors) - forcing a model to produce output that adheres to a meta task (for ex. making a general LLM produce propaganda)
* [Sleeper agent](https://huggingface.co/yifever/sleeper-agent)
* Prompt injection (not really an attack) - craft an input that makes a model bypass its current “safety” measures
  
##### Prompt Injection 
* [Securing LLM Systems Against Prompt Injection](https://developer.nvidia.com/blog/securing-llm-systems-against-prompt-injection/) (and [NeMo-Guardrails Security Guidelines](https://github.com/NVIDIA/NeMo-Guardrails/blob/main/docs/security/guidelines.md))
* [LLM Censorship: A Machine Learning Challenge or a Computer Security Problem?](https://arxiv.org/abs/2307.10719)
* CTFs: [GPT Prompt Attack](https://gpa.43z.one/); [Merlin’s Defense. GPT-4 Prompt Attack CTF](http://mcaledonensis.blog/merlins-defense/); [Gandalf](https://gandalf.lakera.ai/)
* [Garak: LLM Scanner](https://github.com/leondz/garak/)
* [Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173)

#### Availability
* [Energy-latency attacks](https://arxiv.org/abs/2006.03463) - denial of service for neural networks 

### Degraded model performance
* [Robustness Testing of Autonomy Software](https://users.ece.cmu.edu/~koopman/pubs/hutchison18_icse_robustness_testing_autonomy_software.pdf)
* [Can robot navigation bugs be found in simulation? An exploratory study](https://hal.science/hal-01534235/file/PID4832685.pdf)
* [Bugs can optimize for bad behavior (OpenAI GPT-2)](https://openai.com/research/fine-tuning-gpt-2)
* [You Only Look Once Run time errors](https://www.york.ac.uk/assuring-autonomy/guidance/body-of-knowledge/implementation/2-3/2-3-3/cross-domain-automotive/)

### ML-Ops
* [Facebook’s LLAMA being openly distributed via torrents](https://news.ycombinator.com/item?id=35007978)
* [Summoning Demons: The Pursuit of Exploitable Bugs in Machine Learning](https://arxiv.org/abs/1701.04739) 
* [DeepPayload: Black-box Backdoor Attack on Deep Learning Models through Neural Payload Injection](https://arxiv.org/abs/2101.06896) 
* [Weaponizing Machine Learning Models with Ransomware](https://hiddenlayer.com/research/weaponizing-machine-learning-models-with-ransomware/) (and [Machine Learning Threat Roundup](https://hiddenlayer.com/research/machine-learning-threat-roundup/)) 
* [Hacking AI: System and Cloud Takeover via MLflow Exploit](https://protectai.com/blog/hacking-ai-system-takeover-exploit-in-mlflow) 
* [Trail of Bits's Audit of the Safetensors Library](https://github.com/trailofbits/publications/blob/master/reviews/2023-03-eleutherai-huggingface-safetensors-securityreview.pdf)

### AI’s effect on attacks/security elsewhere
* [How AI will affect cybersecurity: What we told the CFTC](https://blog.trailofbits.com/2023/07/31/how-ai-will-affect-cybersecurity-what-we-told-the-cftc/)
* [Lost at C: A User Study on the Security Implications of Large Language Model Code Assistants](https://arxiv.org/abs/2208.09727)
* [Examining Zero-Shot Vulnerability Repair with Large Language Models](https://arxiv.org/pdf/2112.02125.pdf) 
* [Do Users Write More Insecure Code with AI Assistants?](https://arxiv.org/pdf/2211.03622.pdf) 
* [Learned Systems Security](https://arxiv.org/abs/2212.10318) 
* [Beyond the Hype: A Real-World Evaluation of the Impact and Cost of Machine Learning-Based Malware Detection](https://arxiv.org/abs/2012.09214)
* [Data-Driven Offense](https://player.vimeo.com/video/133292422) from Infiltrate 2015
* [Codex (and GPT-4) can’t beat humans on smart contract audits](https://blog.trailofbits.com/2023/03/22/codex-and-gpt4-cant-beat-humans-on-smart-contract-audits/)
  
#### Self-driving cars 
* [Driving to Safety: How Many Miles of Driving Would It Take to Demonstrate Autonomous Vehicle Reliability?](https://www.rand.org/pubs/research_reports/RR1478.html)

## Regulatory actions

### US
* [FTC: Keep your AI claims in check](https://www.ftc.gov/business-guidance/blog/2023/02/keep-your-ai-claims-check)
* [FAA - Unmanned Aircraft Vehicles](https://www.faa.gov/regulations_policies/rulemaking/committees/documents/index.cfm/committee/browse/committeeID/837)
* [NHTSA - Automated Vehicle safety](https://www.nhtsa.gov/technology-innovation/automated-vehicles-safety)
* [AI Bill of Rights](https://www.whitehouse.gov/ostp/ai-bill-of-rights/)

### EU
- [The Artificial Intelligence Act](https://artificialintelligenceact.eu/) (proposed)

## Safety standards
* [Toward Comprehensive Risk Assessments and Assurance of AI-Based Systems](https://blog.trailofbits.com/2023/03/14/ai-security-safety-audit-assurance-heidy-khlaaf-odd/)
* ISO/IEC 42001 — Artificial intelligence — Management system
* ISO/IEC 22989 — Artificial intelligence — Concepts and terminology
* ISO/IEC 38507 — Governance of IT — Governance implications of the use of artificial intelligence by organizations
* ISO/IEC 23894 — Artificial Intelligence — Guidance on Risk Management
* ANSI/UL 4600 Standard for Safety for the Evaluation of Autonomous Products — addresses fully autonomous systems that move such as self-driving cars, and other vehicles including lightweight unmanned aerial vehicles (UAVs). Includes safety case construction, risk analysis, design process, verification and validation, tool qualification, data integrity, human-machine interaction, metrics and conformance assessment.
* High-Level Expert Group on AI in European Commission — Ethics Guidelines for Trustworthy Artificial Intelligence

## Taxonomies and frameworks
* [NIST AI 100-2e2023](https://csrc.nist.gov/publications/detail/white-paper/2023/03/08/adversarial-machine-learning-taxonomy-and-terminology/draft)
* [MITRE ATLAS](https://atlas.mitre.org/) 
* [AI Incident Database](https://incidentdatabase.ai/) 

## Security tools and techniques
### API probing
* [PrivacyRaven](https://github.com/trailofbits/PrivacyRaven): runs different privacy attacks against ML models; the tool only runs black-box label-only attacks
* [Counterfit](https://github.com/Azure/counterfit): runs different adversarial ML attacks against ML models 

### Model backdoors
* [Fickling](https://github.com/trailofbits/fickling): a decompiler, static analyzer, and bytecode rewriter for Python pickle files; injects backdoors into ML model files
* [Semgrep rules for ML](https://blog.trailofbits.com/2022/10/03/semgrep-maching-learning-static-analysis/) 
* [API Rate Limiting](https://platform.openai.com/docs/guides/rate-limits/overview)

### AI's impact on security tools and techniques 
* [Awesome Large Language Model Tools for Cybersecurity Research](https://github.com/tenable/awesome-llm-cybersecurity-tools)
* [EscalateGPT](https://github.com/tenable/EscalateGPT)
* [BurpGPT](https://github.com/tenable/Burp-extension-for-GPT#burpgpt)

## Background information
* [Machine Learning Glossary | Google Developers](https://developers.google.com/machine-learning/glossary)
* [Hugging Face NLP course](https://huggingface.co/learn/nlp-course/chapter1/1)
* [Making Large Language Models work for you](https://simonwillison.net/2023/Aug/27/wordcamp-llms/)
* [Neural Networks: Zero to Hero](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
* [Normcore LLM Reading List](https://gist.github.com/veekaybee/be375ab33085102f9027853128dc5f0e)
* [3blue1brown's Guide to Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
* Licensing:
  * [From RAIL to Open RAIL: Topologies of RAIL Licenses](https://www.licenses.ai/blog/2022/8/18/naming-convention-of-responsible-ai-licenses)
  * [Hugging Face - OpenRAIL ](https://huggingface.co/blog/open_rail)
  * [Hugging Face - AI Release Models](https://arxiv.org/abs/2302.04844)
  * [Open LLMs](https://github.com/eugeneyan/open-llms)
  * [Prompt Engineering Guide](https://github.com/trailofbits/awesome-ml-security/blob/main/prompt-engineering.md)

## DeepFakes, disinformation, and abuse
* [How to Prepare for the Deluge of Generative AI on Social Media](https://knightcolumbia.org/content/how-to-prepare-for-the-deluge-of-generative-ai-on-social-media)
* [Generative ML and CSAM: Implications and Mitigations](https://purl.stanford.edu/jv206yg3793)

## Notable incidents
| **Incident** | **Type** | **Loss** |
| ----- | ----- | ----- |
| Tay | Poor training set selection | Reputational |
| [Apple NeuralHash](https://www.theverge.com/2021/8/18/22630439/apple-csam-neuralhash-collision-vulnerability-flaw-cryptography) | Adversarial evasion (led to hash collisions) | Reputational |
| [PyTorch Compromise](https://pytorch.org/blog/compromised-nightly-dependency/) | Dependency confusion |
| [Proofpoint - CVE-2019-20634](https://github.com/moohax/Proof-Pudding) | Model extraction |
| [ClearviewAI Leak](https://techcrunch.com/2020/04/16/clearview-source-code-lapse/) | Source Code misconfiguration |
| [Kubeflow Crypto-mining attack ](https://sysdig.com/blog/crypto-mining-kubeflow-tensorflow-falco/) | System misconfiguration |
| [OpenAI - takeover someone's account, view their chat history, and access their billing information ](https://twitter.com/naglinagli/status/1639343866313601024) | Web Cache Deception | Reputational |
| [OpenAI- first message of a newly-created conversation was visible in someone else’s chat history](https://openai.com/blog/march-20-chatgpt-outage) | [Cache - Redis Async I/O](https://github.com/redis/redis-py/issues/2624) | Reputational |
| [OpenAI- ChatGPT's new Browser SDK was using some relatively recently known-vulnerable code (specifically MinIO CVE-2023-28432)](https://twitter.com/Andrew___Morris/status/1639325397241278464) | [Security vulnerability resulting in information disclosure of all environment variables, including MINIO_SECRET_KEY and MINIO_ROOT_PASSWORD.](https://www.greynoise.io/blog/openai-minio-and-why-you-should-always-use-docker-cli-scan-to-keep-your-supply-chain-clean) | Reputational              |
| ML Flow | [Protect AI tested the security of MLflow and found a combined Local File Inclusion/Remote File Inclusion vulnerability which can lead to a complete system or cloud provider takeover.](https://protectai.com/blog/hacking-ai-system-takeover-exploit-in-mlflow) | Monetary and Reputational |
| [Microsoft AI Data Leak](https://www.wiz.io/blog/38-terabytes-of-private-data-accidentally-exposed-by-microsoft-ai-researchers) | SAS token misconfiguration |

## Notable harms
| **Incident** | **Type** | **Loss** |
| ----- | ----- | ----- |
| Google Photos Gorillas | Algorithmic bias | Reputational |
| [Uber hits a pedestrian](https://incidentdatabase.ai/cite/4/) | Model failure |
| [Facebook mistranslation leads to arrest](https://incidentdatabase.ai/cite/72/) | Algorithmic bias |
