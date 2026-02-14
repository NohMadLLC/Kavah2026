"""
tests/test_perception.py

Perception Layer Integration Test.
Verifies that all encoders map correctly to the Poincaré Ball H^n.
"""

import torch
import sys
import os

# Ensure we can import from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bedrock_agi.perception.text_encoder import TextEncoder
from bedrock_agi.perception.vision_encoder import VisionEncoder
from bedrock_agi.perception.audio_encoder import AudioEncoder
from bedrock_agi.perception.state_encoder import StateEncoder
from bedrock_agi.core.hbl_geometry import HyperbolicDistance

def test_text_encoder():
    """Test text encoding."""
    print("\nTesting TextEncoder...")
    
    # Initialize
    encoder = TextEncoder(latent_dim=16, embed_dim=768)
    
    # Test inputs
    texts = ["Hello world", "Bedrock AGI"]
    
    # Forward pass
    b = encoder(texts)
    
    # Assertions
    assert b.shape == (2, 16), f"Shape mismatch: {b.shape}"
    norms = torch.norm(b, dim=-1)
    assert torch.all(norms < 1.0), "Manifold violation"
    
    print("✓ TextEncoder works")

def test_vision_encoder():
    """Test vision encoding."""
    print("\nTesting VisionEncoder...")
    
    encoder = VisionEncoder(latent_dim=16)
    
    # Random image batch
    imgs = torch.randn(2, 3, 224, 224)
    
    b = encoder(imgs)
    
    assert b.shape == (2, 16)
    assert torch.all(torch.norm(b, dim=-1) < 1.0)
    
    print("✓ VisionEncoder works")

def test_audio_encoder():
    """Test audio encoding."""
    print("\nTesting AudioEncoder...")
    
    encoder = AudioEncoder(latent_dim=16)
    
    # Random spectrogram batch
    specs = torch.randn(2, 1, 80, 200)
    
    b = encoder(specs)
    
    assert b.shape == (2, 16)
    assert torch.all(torch.norm(b, dim=-1) < 1.0)
    
    print("✓ AudioEncoder works")

def test_state_encoder():
    """Test state encoding."""
    print("\nTesting StateEncoder...")
    
    encoder = StateEncoder(state_dim=32, latent_dim=16)
    
    # Random state vectors
    states = torch.randn(2, 32)
    
    b = encoder(states)
    
    assert b.shape == (2, 16)
    assert torch.all(torch.norm(b, dim=-1) < 1.0)
    
    print("✓ StateEncoder works")

def test_multimodal_integration():
    """Test that all encoders produce compatible latents in H^n."""
    print("\nTesting multimodal integration...")
    
    # Initialize all
    text_enc = TextEncoder(latent_dim=16)
    vision_enc = VisionEncoder(latent_dim=16)
    state_enc = StateEncoder(state_dim=32, latent_dim=16)
    
    # Get latents from different modalities
    b_text = text_enc(["The concept of roundness"])
    b_vision = vision_enc(torch.randn(1, 3, 224, 224)) # Image of a circle
    b_state = state_enc(torch.randn(1, 32)) # Robot holding a ball
    
    # All should be in same space H^16
    assert b_text.shape == b_vision.shape == b_state.shape == (1, 16)
    
    # Can compute Hyperbolic Distances between modalities
    dist_fn = HyperbolicDistance()
    
    d_text_vision = dist_fn(b_text, b_vision)
    d_text_state = dist_fn(b_text, b_state)
    
    assert d_text_vision >= 0
    assert d_text_state >= 0
    
    print(f"  Distance(text, vision): {d_text_vision.item():.4f}")
    print(f"  Distance(text, state):  {d_text_state.item():.4f}")
    
    print("✓ Multimodal geometry verified")

if __name__ == "__main__":
    try:
        test_text_encoder()
        test_vision_encoder()
        test_audio_encoder()
        test_state_encoder()
        test_multimodal_integration()
        print("\nPERCEPTION LAYER: SUCCESS.")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)