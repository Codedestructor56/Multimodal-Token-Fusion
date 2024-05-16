from llama_cpp import Llama

llm = Llama(
      model_path="./models/ggml-model-f16.gguf",
      n_gpu_layers=5,
      # seed=1337, # Uncomment to set a specific seed
      # n_ctx=2048, # Uncomment to increase the context window
)
output = llm(
    "Q: summarize the following text: I love playing in the outdoor heat,where the birds chirp. A: ", # Prompt
      max_tokens=None, # Generate up to 32 tokens, set to None to generate up to the end of the context window
      stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
      echo=True # Echo the prompt back in the output
) # Generate a completion, can also call create_completion
print(output)
