import tiktoken
import torch
from GPTModel import GPTModel

GPT_CONFIG_124M={
    "vocab_size":50217,
    "context_length":1024,
    "emb_dim":768,
    "n_heads":12,
    "drop_rate":0.2,
    "qkv_bias":False,
    "n_layers":12,
}

tokenizer=tiktoken.get_encoding("gpt2")
batch=[]
txt1="Every effort moves you"
txt2="Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))

batch=torch.stack(batch,dim=0)

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)
# print("Input batch:\n", batch)
# print("\nOutput shape:", out.shape)
# print(out)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

def generate_text_simple(model,idx,max_new_tokens,context_size):

    for _ in range(max_new_tokens):
        idx_cond=idx[:,-context_size:]
        with torch.no_grad():
            logits=model(idx_cond)
        
        logits=logits[:,-1,:]
        probas=torch.softmax(logits,dim=-1)
        idx_next=torch.argmax(probas,dim=-1,keepdim=True)
        idx=torch.cat((idx,idx_next),dim=1)

    return idx

start_context = "Hello"
encoded = tokenizer.encode(start_context)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)   

model.eval()    

out = generate_text_simple(
    model=model,
    idx=encoded_tensor, 
    max_new_tokens=6, 
    context_size=GPT_CONFIG_124M["context_length"]
 )

print("Output:", out)

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)