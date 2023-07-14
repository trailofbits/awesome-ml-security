This is an overview of recent prompt engineering research. Tips and tricks have been extracted from relevant works. Each of these techniques should be taken with a grain of salt as it may not generalize to the chosen task, model, or settings.

### General Prompt Engineering Advice

-   [Prompt Programming for Large Language Models: Beyond the Few-Shot Paradigm](https://arxiv.org/abs/2102.07350) 

    - "Use declarative and direct signifiers for tasks such as translate or rephrase this paragraph so that a 2nd grader can understand it.

    -   Use few-shot demonstrations when the task requires a bespoke format, recognizing that few-shot examples may be interpreted holistically by the model rather than as independent samples.

    -   Specify tasks using characters or characteristic situations as a proxy for an intention such as asking Gandhi or Nietzsche to solve a task. Here you are tapping into LLMs' sophisticated understanding of analogies.

    -   Constrain the possible completion output using careful syntactic and lexical prompt formulations such as saying "Translate this French sentence to English" or by adding quotes around the French sentence.

-   Encourage the model to break down problems into subproblems via step-by-step reasoning." (Summary from [Mihail Eric's Prompt Engineering Guide](https://www.mihaileric.com/posts/a-complete-introduction-to-prompt-engineering/) )

-   [Prompt Engineering Tips and Tricks with GPT-3](https://blog.andrewcantino.com/blog/2021/04/21/prompt-engineering-tips-and-tricks/) 

    -   "Make sure your inputs are grammatically correct and have good writing quality as LLMs tend to preserve stylistic consistency in their completions.

    -   Rather than generating a list of N items, generate a single item N times. This avoids the language model getting stuck in a repetitive loop.

    -   In order to improve output quality, generate many completions and then rank them heuristically." (Summary from [Mihail Eric's Prompt Engineering Guide](https://www.mihaileric.com/posts/a-complete-introduction-to-prompt-engineering/) )

-   [Reframing Instructional Prompts to GPTk's Language](https://arxiv.org/abs/2109.07830) 

    -   "Use low-level patterns from other examples to make a given prompt easier to understand for an LLM.

    -   Explicitly itemize instructions into bulleted lists. 

    -   Turn negative statements such as don't create questions which are not to create questions which are.

    -   When possible, break down a top-level task into different sub-tasks that can be executed in parallel or sequentially.

    -   Avoid repeated and generic statements when trying to solve a very specific task. " (Summary from [Mihail Eric's Prompt Engineering Guide](https://www.mihaileric.com/posts/a-complete-introduction-to-prompt-engineering/) )

-   [How to get Codex to produce the code you want! | Prompt Engineering](https://microsoft.github.io/prompt-engineering/)

    -   Provide the model with high level task descriptions, high level context, examples, and previous user input.  

    -   Set the temperature to 0 if you want the same output each time. 

    -   Use the stop sequence to stop Codex from generating variations of similar code. 

    -   "Imagine that you already live in a timeline where the output you want exists. If you were using/quoting it in a blog post, what caption or context might you write for it?" ([Twitter](https://twitter.com/davidad/status/1551143240065228800?lang=en))

-   [Ask Me Anything: A simple strategy for prompting language models](https://arxiv.org/abs/2210.02441)

    -   Prioritize open-ended questions over restricted ones. 

    -   Consider using AMA prompting, which combines collections of open-ended prompts with weak supervision. 

-   [Legal Prompting: Teaching a Language Model to Think Like a Lawyer](https://arxiv.org/abs/2212.01326)

### Few-Shot and Least-to-Most Prompting

-   [Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity](https://arxiv.org/abs/2104.08786)

    -   The order of the examples matter; Consider using a probing technique to identify the optimal order. 

-   [Calibrate Before Use: Improving Few-Shot Performance of Language Models](https://arxiv.org/abs/2102.09690) 

    -   Few-shot examples can have majority label bias, recency bias, or common token bias. You can use a calibration technique to overcome this. 

-   [Zero-Label Prompt Selection](https://arxiv.org/abs/2211.04668) 

    -   Consider using ZPS, a statistical prompt selection technique. 

-   [Least-to-Most Prompting Enables Complex Reasoning in Large Language Models](https://arxiv.org/abs/2205.10625) 

    -   This is how you use least-to-most prompting: 

    -   "1. The first stage is for problem reduction. The prompt in this stage contains constant examples that demonstrate the reduction followed by the specific question to be reduced. 

    -   2\. The second stage is for problem solving---sequentially solve the generated subproblems from the first stage. The prompt in this stage consists of three parts: (1) constant examples demonstrating how subproblems are solved; (2) a potentially empty list of previously"

-   [Compositional Semantic Parsing with Large Language Models](https://arxiv.org/abs/2209.15003)

    -   "We address these challenges with dynamic least to-most prompting, a generic refinement of least-to-most prompting that involves the following steps: (1) tree-structured decomposition of natural language inputs through LM-predicted syntactic parsing, (2) use the decomposition to dynamically select exemplars, and (3) linearize the decomposition tree and prompt the model to sequentially generate answers to subproblems."

-   [Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing](https://arxiv.org/abs/2107.13586)   

    -   This is an older survey of prompt engineering techniques.

### Chain-of-Thought Prompting

-   [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916) 

    -   You can add the exact phrase "Let's think step by step" for chain-of-thought (CoT) prompting.  

-   [Chain of Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)  

    -   You can write a sequence of questions to force the model to think through the problem step-by-step in what is known as handcrafted CoT. 

-   [Complexity-Based Prompting for Multi-Step Reasoning](https://arxiv.org/abs/2210.00720)

    -   You can use a complexity-based scheme to choose answers from prompts with higher reasoning complexity. More broadly, when using CoT, chains with more reasoning steps can perform better. 

-   [Measuring and Narrowing the Compositionality Gap in Language Models](https://arxiv.org/abs/2210.03350) ([Twitter thread](https://twitter.com/ofirpress/status/1577302733383925762))

    -   For complex tasks, force the model to ask follow-up questions to arrive at the answer (add "Are follow-up questions needed here? Yes" to the prompt"). In addition, consider integrating a search engine.

### Soft Prompting 

-   [Injecting World Knowledge into Language Models through Soft Prompts](https://arxiv.org/abs/2210.04726) 

    -   Using soft prompting, provide the LLM with an external memory with domain-specific knowledge by adding continuous vectors to the input sequence. 

-   [XPrompt: Exploring the Extreme of Prompt Tuning](https://arxiv.org/abs/2210.04457) 

    -   Consider learning and tuning soft prompts. 

-   [Learning to Compose Soft Prompts for Compositional Zero-Shot Learning](https://arxiv.org/abs/2204.03574)

    -   Try using compositional soft prompting for compositional problems.

### Automated Prompt Generation 

-   [Large Language Models Are Human-Level Prompt Engineers](https://arxiv.org/abs/2211.01910) & [AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts](https://arxiv.org/abs/2010.15980) 

    -   You can use an automated prompt generation technique that relies on gradient-based optimization. 

-   [How Can We Know What Language Models Know?](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00324/96460/How-Can-We-Know-What-Language-Models-Know) 

    -   You can use an automated prompt generation technique that relies on mining and paraphrasing.  

-   [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190) 

    -   You can optimize continuous prefixes for the input instead of discrete trigger words in the prompt. 

-   [Automatic Chain of Thought Prompting in Large Language Models](https://arxiv.org/abs/2210.03493)

    -   You can sample a diverse set of questions to automatically generate reasoning chains.

### Evaluation Research & Applications

-   [A Hazard Analysis Framework for Code Synthesis Large Language Models](https://arxiv.org/abs/2207.14157) 

-   [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) 

-   [Examining Zero-Shot Vulnerability Repair with Large Language Models](https://arxiv.org/abs/2112.02125) 

-   [Security Implications of Large Language Model Code Assistants: A User Study](https://arxiv.org/abs/2208.09727)

-   [Asleep at the Keyboard? Assessing the Security of GitHub Copilot's Code Contributions](https://arxiv.org/abs/2108.09293)

-   [Pop Quiz! Can a Large Language Model Help With Reverse Engineering?](https://arxiv.org/abs/2202.01142)   

-   [Conversing with Copilot: Exploring Prompt Engineering for Solving CS1 Problems Using Natural Language](https://arxiv.org/abs/2210.15157) 

-   [Red Teaming Language Models with Language Models](https://arxiv.org/abs/2202.03286) & [Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned](https://www.anthropic.com/red_teaming.pdf)

-   [Evaluate & Evaluation on the Hub: Better Best Practices for Data and Model Measurements](https://arxiv.org/abs/2210.01970) 

-   [A Systematic Evaluation of Large Language Models of Code](https://arxiv.org/abs/2202.13169) 

-   [Large Language Models Struggle to Learn Long-Tail Knowledge](https://arxiv.org/abs/2211.08411) 

-   [Do Users Write More Insecure Code with AI Assistants?](https://arxiv.org/pdf/2211.03622.pdf)

### Prompt Engineering Infrastructure 

-   [LangChain: A library that helps build applications from LLMs, includes a wide variety of  prompt engineering techniques](https://github.com/hwchase17/langchain/)

-   [PromptSource: An Integrated Development Environment and Repository for Natural Language Prompts](https://arxiv.org/abs/2202.01279#:~:text=PromptSource%3A%20An%20Integrated%20Development%20Environment%20and%20Repository%20for%20Natural%20Language%20Prompts,-Stephen%20H.&text=PromptSource%20is%20a%20system%20for,language%20input%20and%20target%20output)

-   [Interactive and Visual Prompt Engineering for Ad-hoc Task Adaptation with Large Language Models](https://arxiv.org/abs/2208.07852)

-   [PromptChainer: Chaining Large Language Model Prompts through Visual Programming](https://arxiv.org/abs/2203.06566)  

-   [microsoft/prompt-engine: A library for helping developers craft prompts for LLMs](https://github.com/microsoft/prompt-engine)

-   [GPT with Python interpreter](https://twitter.com/sergeykarayev/status/1569377881440276481?s=46&t=voSylmKII0grJoj8juIOYQ)

-   [GPT with a browser](https://twitter.com/sharifshameem/status/1405462642936799247) ([GitHub](https://github.com/nat/natbot))

### Fine-Tuning; Similar Techniques; and Other Experiments 

-   [Aligning Language Models to Follow Instructions](https://openai.com/blog/instruction-following/)

-   [Large Language Models Can Self-Improve](https://arxiv.org/abs/2210.11610)

-   [Learning by Distilling Context](https://arxiv.org/abs/2209.15189)

-   [Large Language Models with Controllable Working Memory](https://arxiv.org/abs/2211.05110) 

-   [Prompt Injection: Parameterization of Fixed Inputs](https://arxiv.org/abs/2206.11349) 

-   [Do Prompt-Based Models Really Understand the Meaning of their Prompts?](https://arxiv.org/abs/2109.01247)

-   [Self-Programming Artificial Intelligence Using Code-Generating Language Models](https://openreview.net/forum?id=SKat5ZX5RET)

-   [GitHub - semiosis/prompts: A free and open-source curation of prompts](https://github.com/semiosis/prompts)

-   [Machine Learning for Big Code and Naturalness](https://ml4code.github.io/)
