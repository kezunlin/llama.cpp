```mermaid
graph TD
    A[Start] --> B{Input Processing};
    B --> B1["Parse CLI Arguments & User Input <br/>(prompt, file, chat, sys_prompt, n_gpu_layers, etc.)"];

    B1 --> C{Data Preparation};
    C --> C1["Tokenize Text <br/>(common_tokenize)"];
    C1 --> C2["Apply Chat Templates <br/>(if applicable)"];

    C2 --> D{Backend & Model Initialization};
    D --> D1["Initialize Backends <br/>(llama_backend_init(), specific GPU inits)"];
    D1 --> D2["Load GGUF Model Metadata <br/>(llama_model_loader)<br/><i>Parses GGUF, reads tensor structures, handles splits</i>"];
    D2 --> D3["Determine Tensor Placement Strategy <br/>(CPU/GPU)<br/><i>Based on n_gpu_layers, main_gpu, tensor_split</i>"];
    D3 --> D4["Allocate Tensors & Populate Data <br/><i>Mmap to CPU / Direct Read to CPU / <br/>Direct Read to Host & <b>Async Copy to GPU</b></i>"];

    D4 --> E{Inference Loop};
    E --> E1["Loop while <br/>(tokens to generate AND <br/>NOT (antiprompt OR EOG))"];
    
    E1 -- Generation continues --> E2["`llama_decode` (Process Input Tokens)<br/><i>Builds GGML graph. <br/>GGML executes ops on CPU/<b>GPU</b> where tensors reside</i>"];
    E2 --> E3["Update KV Cache"];
    E3 --> E4["Sample Next Token <br/>(common_sampler_sample)<br/><i>Uses logits from llama_decode</i>"];
    E4 --> E5["Update Sampler State <br/>(common_sampler_accept)"];
    E5 --> E1;

    E1 -- Generation stops --> F{Post-processing};
    F --> F1["Detokenize Output Tokens <br/>(common_token_to_piece)"];
    F1 --> F2["Check for Antiprompts / EOG"];
    F2 --> F3["Display Output to User / <br/>Handle Interactive Input"];
    
    F3 --> G[End];

    %% Styling for GPU interaction emphasis
    style D4 fill:#D6EAF8,stroke:#2E86C1,stroke-width:2px;
    style E2 fill:#D6EAF8,stroke:#2E86C1,stroke-width:2px;
```
