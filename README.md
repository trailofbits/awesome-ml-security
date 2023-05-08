# Awesome-ML-Security
# Relevant Work, Standards, Literature

## Security issues

- CIA and privacy of the model: *membership attacks, model inversion attacks, model extraction, adversarial perturbation, prompt injections*

  - [Towards the Science of Security and Privacy in Machine Learning](https://arxiv.org/abs/1611.03814)

  - [SoK: Machine Learning Governance](https://arxiv.org/abs/2109.10870)

  - Confidentiality/privacy ([awesome-ml-privacy-attacks](https://github.com/stratosphereips/awesome-ml-privacy-attacks))

    - Reconstruction (model inversion; attribute inference; gradient and information leakage) - theft of data
    - Membership inference - reidentification of data
    - Model extraction - theft of model
    - Property inference - leakage of dataset properties

  - Availability

    - [Energy-latency attacks](https://arxiv.org/abs/2006.03463) - denial of service for neural networks 

  - Integrity 

    - Backdoors/neural trojans - same as for non-ML systems
    - Adversarial evasion - perturbation of an input to evade a certain classification or output 
    - Data poisoning and [ordering](https://arxiv.org/abs/2104.09667) - providing malicious data or changing the order of the data flow into an ML model 
    - [Adversarial reprogramming](https://arxiv.org/abs/1806.11146) - repurposing a model for a different task than its original intended purpose 
    - [Model spinning attacks](https://arxiv.org/abs/2107.10443) (meta backdoors) - forcing a model to produce output that adheres to a meta task (for ex. making a general LLM produce propaganda)
    - Prompt injection (not really an attack) - craft an input that makes a model bypass its current “safety” measures

- Degraded Model performance (due to bugs)

  - [Robustness Testing of Autonomy Software](https://users.ece.cmu.edu/~koopman/pubs/hutchison18_icse_robustness_testing_autonomy_software.pdf)
  - [Can robot navigation bugs be found in simulation? An exploratory study](https://hal.science/hal-01534235/file/PID4832685.pdf)
  - [Bugs can optimize for bad behavior (OpenAI GPT-2)](https://openai.com/research/fine-tuning-gpt-2)
  - [You Only Look Once Run time errors](https://www.york.ac.uk/assuring-autonomy/guidance/body-of-knowledge/implementation/2-3/2-3-3/cross-domain-automotive/)

- AI’s effect on attacks/security elsewhere

  - [Lost at C: A User Study on the Security Implications of Large Language Model Code Assistants](https://arxiv.org/abs/2208.09727)
  - [Examining Zero-Shot Vulnerability Repair with Large Language Models](https://arxiv.org/pdf/2112.02125.pdf) 
  - [Do Users Write More Insecure Code with AI Assistants?](https://arxiv.org/pdf/2211.03622.pdf) 
  - [Learned Systems Security](https://arxiv.org/abs/2212.10318) 
  - [Beyond the Hype: A Real-World Evaluation of the Impact and Cost of Machine Learning-Based Malware Detection](https://arxiv.org/abs/2012.09214) 

- ML-Ops and pipeline vulnerabilities and exploits: model assets (data, weights, model)

  - [Facebook’s LLAMA being openly distributed via torrents](https://news.ycombinator.com/item?id=35007978)
  - [Summoning Demons: The Pursuit of Exploitable Bugs in Machine Learning](https://arxiv.org/abs/1701.04739) 
  - [DeepPayload: Black-box Backdoor Attack on Deep Learning Models through Neural Payload Injection](https://arxiv.org/abs/2101.06896) 
  - [Weaponizing Machine Learning Models with Ransomware](https://hiddenlayer.com/research/weaponizing-machine-learning-models-with-ransomware/) (and [Machine Learning Threat Roundup](https://hiddenlayer.com/research/machine-learning-threat-roundup/)) 
  - [Hacking AI: System and Cloud Takeover via MLflow Exploit](https://protectai.com/blog/hacking-ai-system-takeover-exploit-in-mlflow) 

## Regulatory Actions

### US

- [FTC: Keep your AI claims in check](https://www.ftc.gov/business-guidance/blog/2023/02/keep-your-ai-claims-check)
- [FAA - Unmanned Aircraft Vehicles](https://www.faa.gov/regulations_policies/rulemaking/committees/documents/index.cfm/committee/browse/committeeID/837)
- [NHTSA - Automated Vehicle safety](https://www.nhtsa.gov/technology-innovation/automated-vehicles-safety)
- [AI Bill of Rights](https://www.whitehouse.gov/ostp/ai-bill-of-rights/)

### EU

- [The Artificial Intelligence Act](https://artificialintelligenceact.eu/) (proposed)

## Safety Standards

- ISO/IEC 42001 — Artificial intelligence — Management system

- ISO/IEC 22989 — Artificial intelligence — Concepts and terminology

- ISO/IEC 38507 — Governance of IT — Governance implications of the use of artificial intelligence by organizations

- ISO/IEC 23894 — Artificial Intelligence — Guidance on Risk Management

- ANSI/UL 4600 Standard for Safety for the Evaluation of Autonomous Products — addresses fully autonomous systems that move such as self-driving cars, and other vehicles including lightweight unmanned aerial vehicles (UAVs).

  - safety case construction
  - risk analysis
  - design process
  - verification and validation
  - tool qualification
  - data integrity
  - human-machine interaction
  - metrics and conformance assessment

- High-Level Expert Group on AI in European Commission — Ethics Guidelines for Trustworthy Artificial Intelligence

- Toward Comprehensive Risk Assessments and Assurance of AI-Based Systems (Use of MIL-STD-882E)

## Relevant taxonomies and frameworks

- [NIST AI 100-2e2023](https://csrc.nist.gov/publications/detail/white-paper/2023/03/08/adversarial-machine-learning-taxonomy-and-terminology/draft)
- [MITRE ATLAS](https://atlas.mitre.org/) 
- [AI Incident Database](https://incidentdatabase.ai/) 

## Security tools and techniques

- API probing

  - PrivacyRaven: runs different privacy attacks against ML models; the tool only runs black-box label-only attacks
  - Counterfeit: runs different adversarial ML attacks against ML models 

- Model backdoors

  - Fickling: a decompiler, static analyzer, and bytecode rewriter for Python pickle files; injects backdoors into ML model files 

- [API Rate Limiting](https://platform.openai.com/docs/guides/rate-limits/overview)

- [Semgrep rules for ML](https://blog.trailofbits.com/2022/10/03/semgrep-maching-learning-static-analysis/) 

## Essential background information

- [Machine Learning Glossary | Google Developers](https://developers.google.com/machine-learning/glossary) 

- Licensing:

  - [From RAIL to Open RAIL: Topologies of RAIL Licenses](https://www.licenses.ai/blog/2022/8/18/naming-convention-of-responsible-ai-licenses)
  - [Hugging Face - OpenRAIL ](https://huggingface.co/blog/open_rail)
  - [Hugging Face - AI Release Models](https://arxiv.org/abs/2302.04844)

# Notable Incidents

| **Incident**                                                 | **Type**                                                     | **Loss**                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------- |
| Tay                                                          | Poor training set selection                                  | Reputational              |
| [Apple NeuralHash](https://www.theverge.com/2021/8/18/22630439/apple-csam-neuralhash-collision-vulnerability-flaw-cryptography) | Adversarial evasion (led to hash collisions)                 | Reputational              |
| [PyTorch Compromise](https://pytorch.org/blog/compromised-nightly-dependency/) | Dependency confusion                                         |                           |
| [Proofpoint - CVE-2019-20634](https://github.com/moohax/Proof-Pudding) | Model extraction                                             |                           |
| [ClearviewAI Leak](https://techcrunch.com/2020/04/16/clearview-source-code-lapse/) | Source Code misconfiguration                                 |                           |
| [Kubeflow Crypto-mining attack ](https://sysdig.com/blog/crypto-mining-kubeflow-tensorflow-falco/) | System misconfiguration                                      |                           |
| [OpenAI - takeover someone's account, view their chat history, and access their billing information ](https://twitter.com/naglinagli/status/1639343866313601024) | Web Cache Deception                                          | Reputational              |
| [OpenAI- first message of a newly-created conversation was visible in someone else’s chat history](https://openai.com/blog/march-20-chatgpt-outage) | [Cache - Redis Async I/O](https://github.com/redis/redis-py/issues/2624) | Reputational              |
| [OpenAI- ChatGPT's new Browser SDK was using some relatively recently known-vulnerable code (specifically MinIO CVE-2023-28432)](https://twitter.com/Andrew___Morris/status/1639325397241278464) | [Security vulnerability resulting in information disclosure of all environment variables, including MINIO_SECRET_KEY and MINIO_ROOT_PASSWORD.](https://www.greynoise.io/blog/openai-minio-and-why-you-should-always-use-docker-cli-scan-to-keep-your-supply-chain-clean) | Reputational              |
| ML Flow                                                      | [Protect AI tested the security of MLflow and found a combined Local File Inclusion/Remote File Inclusion vulnerability which can lead to a complete system or cloud provider takeover.](https://protectai.com/blog/hacking-ai-system-takeover-exploit-in-mlflow) | Monetary and Reputational |

# Notable Harms

| **Incident**                                                 | **Type**         | **Loss**     |
| ------------------------------------------------------------ | ---------------- | ------------ |
| Google Photos Gorillas                                       | Algorithmic bias | Reputational |
| [Uber hits a pedestrian](https://incidentdatabase.ai/cite/4/) | Model failure    |              |
| [Facebook mistranslation leads to arrest](https://incidentdatabase.ai/cite/72/) | Algorithmic bias |              |
