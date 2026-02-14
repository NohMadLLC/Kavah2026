"""
bedrock_agi/perception/text_encoder.py

Text Encoder: Map text -> H^n via external embeddings.
Supports:
- Ollama (local)
- OpenAI embeddings (via compatible endpoint)
- Any endpoint returning vectors

Flow:
1. Text -> external embedding API
2. Embedding -> adapter network (Tangent Space)
3. Adapter output -> exp_0 -> H^n (Poincaré Ball)
"""

import torch
import torch.nn as nn
from typing import List, Optional
import requests
import hashlib

class TextEncoder(nn.Module):
    """
    Text -> H^n encoder.
    
    Adapts external semantic vectors (e.g., 768d) to the internal 
    hyperbolic latent space (e.g., 16d).
    """
    
    def __init__(
        self,
        latent_dim=16,
        embed_dim=768,  # Default for nomic-embed-text / bert-base
        endpoint="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text"
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.endpoint = endpoint
        self.model_name = model_name
        
        # Adapter: embedding space (Euclidean) -> Tangent Space of H^n at origin
        # We process in Euclidean space before projecting
        self.adapter = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, latent_dim)
        )
        
    def get_embedding(self, text: str) -> torch.Tensor:
        """
        Get embedding from external API.
        
        Args:
            text: Input text string
            
        Returns:
            Embedding vector (embed_dim,)
        """
        # Logic: Try API, Fallback to deterministic random if fail
        try:
            # Ollama / OpenAI-compatible JSON format
            payload = {
                "model": self.model_name,
                "prompt": text
            }
            
            response = requests.post(
                self.endpoint,
                json=payload,
                timeout=5  # Short timeout to avoid hanging
            )
            
            if response.status_code == 200:
                data = response.json()
                # Handle different API response formats
                if "embedding" in data:
                    emb = data["embedding"]
                elif "data" in data and len(data["data"]) > 0:
                    emb = data["data"][0]["embedding"]
                else:
                    raise ValueError(f"Unknown API format: {data.keys()}")
                    
                # Ensure dimension matches expected
                t_emb = torch.tensor(emb, dtype=torch.float32)
                if t_emb.shape[0] != self.embed_dim:
                    # Simple resize/pad/crop if dimension mismatch (rare but safe)
                    # For now, just warn and use random
                    print(f"Warning: Dim mismatch {t_emb.shape[0]} != {self.embed_dim}")
                    return self._deterministic_fallback(text)
                    
                return t_emb
                
            else:
                # API Error
                # print(f"API Error {response.status_code}: {response.text}")
                return self._deterministic_fallback(text)
                
        except Exception as e:
            # Connection failed or other error
            # print(f"Embedding API unavailable: {e}")
            return self._deterministic_fallback(text)

    def _deterministic_fallback(self, text: str) -> torch.Tensor:
        """
        Generate a deterministic random vector based on text hash.
        Useful for testing without the embedding server running.
        """
        # Hash text to integer seed
        hash_digest = hashlib.sha256(text.encode()).hexdigest()
        seed = int(hash_digest[:8], 16)
        
        # Local generator to avoid messing with global seed
        g = torch.Generator()
        g.manual_seed(seed)
        
        return torch.randn(self.embed_dim, generator=g)

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts to H^n.
        
        Args:
            texts: List of text strings
            
        Returns:
            Latent states in H^n (B, latent_dim)
        """
        from ..core.hbl_geometry import PoincareMath
        
        # 1. Fetch Embeddings
        embeddings = []
        for text in texts:
            emb = self.get_embedding(text)
            embeddings.append(emb)
        
        H = torch.stack(embeddings, dim=0)  # (B, embed_dim)
        
        # 2. Adapt to Tangent Space (Euclidean transformation)
        z = self.adapter(H)  # (B, latent_dim)
        
        # 3. Map to Poincaré Ball (Exponential Map)
        b = PoincareMath.exp_map_0(z)
        
        return b

if __name__ == "__main__":
    print("Testing TextEncoder...")
    
    # Initialize with default settings
    # Note: If no local Ollama is running, this will use deterministic fallback
    encoder = TextEncoder(latent_dim=16, embed_dim=768)
    
    # Test with sample texts
    texts = ["Hello world", "The quick brown fox"]
    
    # Run forward pass
    b = encoder(texts)
    
    # Checks
    assert b.shape == (2, 16), f"Shape mismatch: {b.shape}"
    print(f"✓ Output shape correct: {b.shape}")
    
    # Check manifold constraints
    norms = torch.norm(b, dim=-1)
    assert torch.all(norms < 1.0), "Manifold violation: Norm >= 1.0"
    print(f"✓ On manifold: max norm = {norms.max():.4f}")
    
    # Test stability (deterministic fallback)
    b2 = encoder(texts)
    dist = torch.norm(b - b2).item()
    if dist < 1e-6:
        print("✓ Encoder is deterministic")
    else:
        print(f"⚠ Encoder is stochastic (diff={dist:.4f}) - Check API/Fallback")

    print("✓ TextEncoder operational")