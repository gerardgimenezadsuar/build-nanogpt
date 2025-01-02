import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def demonstrate_batch_structure():
    """Demonstrates the structure of batches and sequences"""
    print("\n1. BATCH STRUCTURE DEMONSTRATION")
    print("-" * 50)
    
    # Create example batches
    B = 4  # batch size
    T = 8  # sequence length
    
    # Create input and target tensors
    x = torch.tensor([
        [1, 2, 3, 4, 5, 6, 7, 8],  # sequence 1
        [2, 3, 4, 5, 6, 7, 8, 9],  # sequence 2
        [3, 4, 5, 6, 7, 8, 9, 1],  # sequence 3
        [4, 5, 6, 7, 8, 9, 1, 2]   # sequence 4
    ])
    
    y = torch.tensor([
        [2, 3, 4, 5, 6, 7, 8, 9],  # shifted sequence 1
        [3, 4, 5, 6, 7, 8, 9, 1],  # shifted sequence 2
        [4, 5, 6, 7, 8, 9, 1, 2],  # shifted sequence 3
        [5, 6, 7, 8, 9, 1, 2, 3]   # shifted sequence 4
    ])
    
    print(f"Input shape (B, T): {x.shape}")
    print("\nInput sequences:")
    print(x)
    print("\nTarget sequences (shifted by 1):")
    print(y)

def demonstrate_embeddings():
    """Demonstrates token and position embeddings"""
    print("\n2. EMBEDDINGS DEMONSTRATION")
    print("-" * 50)
    
    B = 4  # batch size
    T = 8  # sequence length
    n_embd = 6  # embedding dimension (smaller for demonstration)
    
    # Create dummy embedding layers
    token_embedding = torch.nn.Embedding(100, n_embd)
    position_embedding = torch.nn.Embedding(T, n_embd)
    
    # Create input tensor
    x = torch.tensor([
        [1, 2, 3, 4, 5, 6, 7, 8],
        [2, 3, 4, 5, 6, 7, 8, 9],
        [3, 4, 5, 6, 7, 8, 9, 1],
        [4, 5, 6, 7, 8, 9, 1, 2]
    ])
    
    # Get token embeddings
    tok_emb = token_embedding(x)  # shape: (B=4, T=8, n_embd=6)
    
    # Get position embeddings
    pos = torch.arange(0, T)  # [0,1,2,3,4,5,6,7]
    pos_emb = position_embedding(pos)  # shape: (T=8, n_embd=6)
    
    # Combine embeddings
    combined = tok_emb + pos_emb
    
    print(f"Token embeddings shape: {tok_emb.shape}")
    print(f"Position embeddings shape: {pos_emb.shape}")
    print(f"Combined embeddings shape: {combined.shape}")
    
    print("\nExample of first token embedding:")
    print(tok_emb[0, 0])  # First sequence, first token

def demonstrate_loss_calculation():
    """Demonstrates how the loss is calculated"""
    print("\n3. LOSS CALCULATION DEMONSTRATION")
    print("-" * 50)
    
    B, T, V = 4, 8, 5  # V is vocabulary size (small for demonstration)
    
    # Create random logits and targets
    logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))
    
    # Calculate loss
    loss = F.cross_entropy(
        logits.view(-1, V),
        targets.view(-1)
    )
    
    print(f"Logits shape: {logits.shape}")
    print(f"Logits sample: {logits.view(-1, V)}")
    print(f"Targets shape: {targets.shape}")
    print(f"Targets sample: {targets.view(-1)}")
    print(f"Calculated loss: {loss.item():.4f}")
    print(f"Loss {loss}")
    
    # Show example for one position
    print("\nProbabilities for first sequence, first position:")
    probs = F.softmax(logits[0, 0], dim=-1)
    print(probs)
    print(f"Target token: {targets[0, 0]}")

def demonstrate_gradient_accumulation():
    """Demonstrates gradient accumulation"""
    print("\n4. GRADIENT ACCUMULATION DEMONSTRATION")
    print("-" * 50)
    
    # Simple model for demonstration
    model = torch.nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters())
    
    total_batch_size = 32
    micro_batch_size = 4
    grad_accum_steps = total_batch_size // micro_batch_size
    
    print(f"Total batch size: {total_batch_size}")
    print(f"Micro batch size: {micro_batch_size}")
    print(f"Gradient accumulation steps: {grad_accum_steps}")
    
    # Training loop example
    optimizer.zero_grad()
    loss_accum = 0.0
    
    for micro_step in range(grad_accum_steps):
        # Fake batch
        x = torch.randn(micro_batch_size, 10)
        y = torch.randint(0, 5, (micro_batch_size,))
        
        # Forward pass
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        print(f"Y: {y}")
        print(f"Logits: {logits}")
        #print(f"Loss: {loss}")
        
        # Scale loss
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        
        # Backward pass
        loss.backward()
        
        print(f"Micro-step {micro_step + 1}, Loss: {loss.item():.4f}")
    
    print(f"\nAccumulated loss: {loss_accum.item():.4f}")
    optimizer.step()

def demonstrate_lr_schedule():
    """Demonstrates learning rate schedule"""
    print("\n5. LEARNING RATE SCHEDULE DEMONSTRATION")
    print("-" * 50)
    
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 100
    max_steps = 1000
    
    def get_lr(it):
        # Linear warmup
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        # Cosine decay
        if it > max_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)
    
    # Plot learning rate schedule
    steps = torch.linspace(0, max_steps, 100)
    steps_np = steps.numpy()  # Convert to numpy
    lrs = [get_lr(it.item()) for it in steps]
    
    plt.figure(figsize=(10, 5))
    plt.plot(steps, lrs)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.savefig('lr_schedule.png')
    plt.close()
    
    print(f"Learning rate at step 0: {get_lr(0):.2e}")
    print(f"Learning rate at warmup end: {get_lr(warmup_steps):.2e}")
    print(f"Learning rate at final step: {get_lr(max_steps):.2e}")
    print("\nLearning rate schedule plot saved as 'lr_schedule.png'")

if __name__ == "__main__":
    import math
    
    print("GPT-2 Training Components Demonstration")
    print("=" * 50)
    
    demonstrate_batch_structure()
    demonstrate_embeddings()
    demonstrate_loss_calculation()
    demonstrate_gradient_accumulation()
    demonstrate_lr_schedule() 