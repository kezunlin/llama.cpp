# Data Preparation Deep Dive

This report provides an in-depth look at the data preparation stage within the llama.cpp inference pipeline. It includes a detailed flowchart and an enhanced textual description with C++ code snippets to illustrate the process from input acquisition to final tokenization.

## Flowchart: Data Preparation

```mermaid
graph TD
    A["Start Data Preparation"] --> B{"Parse CLI Arguments & Files"};
    B -- common_params_parse --> C["Initial 'common_params' set 
(prompt, system_prompt, chat_template, antiprompt, etc.)"];
    
    C --> D{"Mode Selection"};
    D -- "Interactive Mode 
('params.interactive_first' or no initial prompt)" --> E["Interactive Input Loop"];
    D -- "Pre-set Prompt Mode" --> F["Process Pre-set Prompt"];

    F --> F1["Load 'params.prompt' (from CLI -p or file -f)"];
    F1 --> F2{"Conversation & Chat Template? 
('params.conversation_mode && params.enable_chat_template')"};
    F2 -- Yes --> G["Chat Templating"];
    G --> G1["'common_chat_templates_init(model, params.chat_template)'"];
    G1 --> G2["Format System Prompt (if 'params.system_prompt') 
using 'chat_add_and_format' (calls 'common_chat_format_single')"];
    G2 --> G3["Format Initial User Prompt (if 'params.prompt') 
using 'chat_add_and_format'"];
    G3 --> G4["'common_chat_templates_apply' -> 'final_prompt_string'"];
    F2 -- No --> H["'final_prompt_string = params.prompt'"];
    
    G4 --> I["Tokenize Initial Prompt"];
    H --> I;

    E --> E1["'console::readline()' -> 'buffer'"];
    E1 --> E2{"Conversation & Chat Template?"};
    E2 -- Yes --> E3["'chat_add_and_format(\\'user\\', buffer)' -> 'user_input_string'"];
    E2 -- No --> E4["'user_input_string = buffer'"];
    E3 --> E5["Handle 'params.input_prefix', 'params.input_suffix' 
(if chat template disabled for interactive)"];
    E4 --> E5;
    E5 --> J["Tokenize Interactive Input"];

    I --> K["Set 'embd_inp' from Initial Prompt Tokens"];
    J --> K; 
    K --> L["'common_tokenize(ctx, text_to_tokenize, add_special, parse_special)'"];
    L --> M{"Special Token Handling"};
    M --> M1["Determine 'add_bos' based on vocab & 'params.use_jinja'"];
    M1 --> M2["If 'embd_inp' is empty & 'add_bos', add BOS token"];
    M2 --> N["Output: 'embd_inp' (tokenized input)"];
    N --> Z["End Data Preparation"];

    subgraph legend [Flowchart Legend]
        direction LR
        legend_input["Input/Output"]
        legend_process["Process Step"]
        legend_decision{"Decision"}
        legend_subroutine["/Sub-routine Call/"]
    end
    classDef input fill:#f9f,stroke:#333,stroke-width:2px;
    classDef process fill:#9cf,stroke:#333,stroke-width:2px;
    classDef decision fill:#f96,stroke:#333,stroke-width:2px;
    classDef subroutine fill:#9f9,stroke:#333,stroke-width:2px;
    
    class A,C,E,E1,F,F1,G,G1,G2,G3,G4,H,I,E3,E4,E5,J,K,L,M1,M2,Z process;
    class N input; %% N is an output, so class input
    class B,D,F2,E2,M decision;
    
    class legend_input input;
    class legend_process process;
    class legend_decision decision;
    class legend_subroutine subroutine;

```

## Detailed Explanation with Code Snippets

This section expands upon the Data Preparation phase, integrating C++ code snippets from `tools/main/main.cpp` and `common/common.cpp` for clarity.

### 1. Input Acquisition

Input text for the language model can be provided through several mechanisms, primarily configured via command-line arguments parsed by `common_params_parse` (defined in `common/arg.cpp`).

**Command-Line Arguments & `common_params_parse`:**

The `common_params_parse` function processes `argv` and populates the `common_params` struct (defined in `common/common.h`). This struct holds various settings, including input text sources.

Key fields in `common_params` related to input text:
```cpp
// From common/common.h (simplified)
struct common_params {
    // ... other params ...
    std::string prompt               = ""; // From -p or -f
    std::string system_prompt        = ""; // From -sys or -sysf
    std::string chat_template      = ""; // From --chat-template or --chat-template-file
    std::vector<std::string> antiprompt;   // From -r
    bool interactive           = false;
    bool interactive_first     = false;
    common_conversation_mode conversation_mode = COMMON_CONVERSATION_MODE_AUTO;
    bool enable_chat_template  = true; // Controls whether chat template is used if available
    // ... other params ...
};
```

The parsing itself in `common/arg.cpp` involves defining `common_arg` structures for each command-line option, which then call handler functions to set these `common_params` fields. For example (conceptual):
```cpp
// Conceptual structure from common/arg.cpp for parsing "-p"
// add_opt(common_arg(
//     {"-p", "--prompt"}, "PROMPT",
//     "prompt to start generation with...",
//     [](common_params & params, const std::string & value) {
//         params.prompt = value; // Sets the prompt
//     }
// ));
// add_opt(common_arg(
//     {"-f", "--file"}, "FNAME",
//     "a file containing the prompt...",
//     [](common_params & params, const std::string & value) {
//         params.prompt = read_file(value); // Reads prompt from file
//         params.prompt_file = value;
//     }
// ));
// add_opt(common_arg(
//     {"-sys", "--system-prompt"}, "PROMPT",
//     "system prompt to use...",
//     [](common_params & params, const std::string & value) {
//         params.system_prompt = value;
//     }
// ));
// add_opt(common_arg(
//     {"--chat-template"}, "JINJA_TEMPLATE",
//     "set custom jinja chat template...",
//     [](common_params & params, const std::string & value) {
//         params.chat_template = value;
//     }
// ));
```

**Initial Prompt Construction in `tools/main/main.cpp`:**

Based on the parsed `params`, `tools/main/main.cpp` constructs the initial `prompt` string that will be tokenized.
```cpp
// In tools/main/main.cpp, within main()
    std::string prompt; // This will hold the final string to be tokenized
    { // Block for initial prompt setup
        if (params.conversation_mode && params.enable_chat_template) {
            // Initialize chat templates
            auto chat_templates = common_chat_templates_init(model, params.chat_template);
            LOG_INF("%s: chat template example:\n%s\n", __func__, common_chat_format_example(chat_templates.get(), params.use_jinja).c_str());

            // Lambda for formatting and adding messages to chat_msgs
            auto chat_add_and_format = 
                [&chat_msgs, &chat_templates](const std::string & role, const std::string & content) {
                common_chat_msg new_msg;
                new_msg.role = role;
                new_msg.content = content;
                // common_chat_format_single formats one message according to the template
                auto formatted = common_chat_format_single(
                    chat_templates.get(), chat_msgs, new_msg, role == "user", g_params->use_jinja);
                chat_msgs.push_back(new_msg);
                LOG_DBG("formatted: '%s'\n", formatted.c_str());
                return formatted;
            };

            if (!params.system_prompt.empty()) {
                chat_add_and_format("system", params.system_prompt);
            }

            if (!params.prompt.empty()) { // User-provided initial prompt
                chat_add_and_format("user", params.prompt);
            } else {
                waiting_for_first_input = true; // No initial user prompt, wait for interactive
            }

            if (!params.system_prompt.empty() || !params.prompt.empty()) {
                common_chat_templates_inputs inputs;
                inputs.messages = chat_msgs;
                // add_generation_prompt controls if the template adds an assistant generation prefix
                inputs.add_generation_prompt = !params.prompt.empty(); 

                // common_chat_templates_apply assembles the full prompt from all messages
                prompt = common_chat_templates_apply(chat_templates.get(), inputs).prompt;
            }
        } else {
            // Not using chat templates, use the prompt as is (from -p or -f)
            prompt = params.prompt;
            if (params.conversation_mode && !params.enable_chat_template) {
                 LOG_INF("%s: in-suffix/prefix is specified, chat template will be disabled\n", __func__);
            }
        }
    // ... tokenization follows ...
    }
```

**Interactive Input:**

If no initial prompt is provided in conversation mode, or if `params.interactive_first` is set, the application enters an interactive loop.
```cpp
// In tools/main/main.cpp, inside the main while loop
if ((n_past > 0 || waiting_for_first_input) && is_interacting) {
    // ...
    std::string buffer; // Raw text from user
    std::string line;
    bool another_line = true;
    do {
        another_line = console::readline(line, params.multiline_input);
        buffer += line;
    } while (another_line);

    // ... (buffer processing, e.g., pop_back for newline) ...

    if (!buffer.empty()) {
        // ...
        bool format_chat = params.conversation_mode && params.enable_chat_template;
        std::string user_inp = format_chat
            ? chat_add_and_format("user", std::move(buffer)) // Apply chat template
            : std::move(buffer);                             // Use raw buffer

        // Tokenize user_inp (shown in tokenization section)
        const auto line_pfx = common_tokenize(ctx, params.input_prefix, false, true);
        const auto line_inp = common_tokenize(ctx, user_inp, false, format_chat); // format_chat influences parse_special
        const auto line_sfx = common_tokenize(ctx, params.input_suffix, false, true);

        embd_inp.insert(embd_inp.end(), line_pfx.begin(), line_pfx.end());
        embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());
        embd_inp.insert(embd_inp.end(), line_sfx.begin(), line_sfx.end());
        // ...
    }
}
```

### 2. Tokenization

The processed prompt string (or interactive input) is then tokenized.

**Tokenization Call in `tools/main/main.cpp`:**
For the initial prompt:
```cpp
// In tools/main/main.cpp, after initial prompt construction
if (params.interactive_first || !prompt.empty() || session_tokens.empty()) {
    LOG_DBG("tokenize the prompt\n");
    // add_special = true to potentially add BOS, parse_special = true for chat templates
    embd_inp = common_tokenize(ctx, prompt, true, true); 
} else {
    LOG_DBG("use session tokens\n");
    embd_inp = session_tokens;
}
```
For interactive input (as seen above, within the interactive loop):
```cpp
// const auto line_inp = common_tokenize(ctx, user_inp, false, format_chat);
// `add_special` is false as BOS/EOS are typically handled by the surrounding logic or template.
// `parse_special` is true if chat templates were used (`format_chat`).
```

**`common_tokenize()` Wrapper in `common/common.cpp`:**
This function is a convenience wrapper around the core `llama_tokenize` function.
```cpp
// From common/common.cpp
std::vector<llama_token> common_tokenize(
  const struct llama_context * ctx,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    // Delegates to another overload
    return common_tokenize(vocab, text, add_special, parse_special);
}

std::vector<llama_token> common_tokenize(
    const struct llama_vocab * vocab,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special) {
    // Pre-allocate a generous buffer
    int n_tokens = text.length() + 2 * add_special; // Max possible if all chars are tokens + BOS/EOS
    std::vector<llama_token> result(n_tokens);
    
    // llama_tokenize is the core library function
    n_tokens = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
    
    if (n_tokens < 0) { // If buffer was too small
        result.resize(-n_tokens);
        int check = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
        GGML_ASSERT(check == -n_tokens); // Ensure second attempt with correct size works
    } else {
        result.resize(n_tokens); // Resize to actual number of tokens
    }
    return result;
}
```
*   **`add_special` parameter:** If true, instructs `llama_tokenize` to potentially add BOS (Beginning Of Sequence) or EOS (End Of Sequence) tokens if the vocabulary is configured for them.
*   **`parse_special` parameter:** If true, allows `llama_tokenize` to interpret special tokens (e.g., `<|user|>`, `<|im_start|>`) within the text and convert them to their corresponding token IDs. This is crucial when using chat templates that embed such tokens.

### 3. Prompt Formatting and Chat Templating

As shown in the "Initial Prompt Construction" section, `tools/main/main.cpp` uses several helper functions from `common/chat.cpp` (though their implementation details are not shown here, their roles are important):

*   **`common_chat_templates_init(model, params.chat_template)`:** Loads the chat template definition, either from the model file itself or from a user-provided template string (`params.chat_template`).
*   **`common_chat_format_single(chat_templates.get(), chat_msgs, new_msg, ...)`:** Formats a single message (`new_msg`) according to the loaded template, considering the existing conversation history (`chat_msgs`). This is used by the `chat_add_and_format` lambda.
*   **`common_chat_templates_apply(chat_templates.get(), inputs)`:** Takes the complete list of formatted messages and applies the overall template structure (e.g., ensuring correct turn prefixes/suffixes, adding a generation prompt for the assistant if needed) to produce the final string to be tokenized.

### 4. Special Token Handling (BOS)

The Beginning-Of-Sequence (BOS) token is handled explicitly in a few places:

**Determining `add_bos`:**
```cpp
// In tools/main/main.cpp
// llama_vocab_get_add_bos(vocab) checks if the model's vocab expects a BOS token.
// !params.use_jinja ensures BOS is not added if Jinja templating is active (Jinja handles its own BOS/EOS).
const bool add_bos = llama_vocab_get_add_bos(vocab) && !params.use_jinja; 
```

**Adding BOS if `embd_inp` is empty:**
If, after all initial prompt processing, the input token list `embd_inp` is empty (and it's not waiting for the first interactive input), a BOS token might be added.
```cpp
// In tools/main/main.cpp
if (!waiting_for_first_input && embd_inp.empty()) {
    if (add_bos) {
        embd_inp.push_back(llama_vocab_bos(vocab)); // llama_vocab_bos(vocab) gets the actual BOS token ID
        LOG_WRN("embd_inp was considered empty and bos was added: %s\n", string_from(ctx, embd_inp).c_str());
    } else {
        LOG_ERR("input is empty\n");
        // return -1; // This would typically exit if input is truly empty and no BOS can be added.
    }
}
```

**Adding BOS before interactive input (`params.input_prefix_bos`):**
Users can also force a BOS token before each interactive input.
```cpp
// In tools/main/main.cpp, within the interactive input section
if ((n_past > 0 || waiting_for_first_input) && is_interacting) {
    // ...
    if (params.input_prefix_bos) {
        LOG_DBG("adding input prefix BOS token\n");
        embd_inp.push_back(llama_vocab_bos(vocab));
    }
    // ...
}
```
This detailed preparation ensures that the model receives a correctly formatted and tokenized sequence, ready for the inference process.
