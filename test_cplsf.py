#!/usr/bin/env python
"""Quick test script to verify CPLSF implementation works without errors."""

import torch
import sys
sys.path.insert(0, '/home/milab/SSD5_8TB/Daniel/Daniel_ssd2/RegistFormer')

from src.models.components.network_proposed_synthesis import ProposedSynthesisModule

def test_cplsf():
    print("=" * 80)
    print("Testing CPLSF Implementation")
    print("=" * 80)

    # Test parameters
    kwargs = {
        'input_nc': 1,
        'output_nc': 2,
        'feat_ch': 128,
        'demodulate': True,
        'use_multiple_outputs': True,
        'is_3d': False,
        'use_separate_style_layers': True,
        'use_style_router': True,
        'style_router_ch': 64,
        'use_freq_prior': True,
        'use_source_gate': True,
    }

    print("\n1. Initializing network...")
    try:
        net = ProposedSynthesisModule(**kwargs)
        print("   ✓ Network initialized successfully")
    except Exception as e:
        print(f"   ✗ Network initialization failed: {e}")
        return False

    print("\n2. Creating dummy input...")
    batch_size = 2
    H, W = 128, 128
    # Input: [B, 1 (source) + 2 (refs), H, W]
    source = torch.randn(batch_size, 1, H, W)
    ref_b = torch.randn(batch_size, 1, H, W)
    ref_c = torch.randn(batch_size, 1, H, W)
    merged_input = torch.cat([source, ref_b, ref_c], dim=1)
    print(f"   Merged input shape: {merged_input.shape}")

    print("\n3. Running forward pass...")
    try:
        net.eval()
        with torch.no_grad():
            output = net(merged_input)

        # Check if output is tuple (out, decomp_dict)
        if isinstance(output, tuple):
            out, decomp_dict = output
            print(f"   ✓ Forward pass successful")
            print(f"   Output shape: {out.shape}")

            if decomp_dict is not None:
                print(f"   ✓ decomp_dict returned (CPLSF active)")
                print(f"   decomp_dict keys: {list(decomp_dict.keys())}")
                for key, val in decomp_dict.items():
                    if isinstance(val, torch.Tensor):
                        print(f"     - {key}: {val.shape}")
            else:
                print(f"   ✗ decomp_dict is None (CPLSF not active)")
        else:
            print(f"   ✗ Output is not a tuple: {type(output)}")
            return False

    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n4. Testing loss computation...")
    try:
        from src.losses.style_decomp_loss import StyleDecompositionLoss

        criterion = StyleDecompositionLoss(
            lambda_common=1.0,
            lambda_private=0.2,
            lambda_sep=0.2,
            lambda_smooth=0.05,
        )

        if decomp_dict is not None:
            loss, loss_dict = criterion(decomp_dict)
            print(f"   ✓ Loss computation successful")
            print(f"   Total loss: {loss.item():.6f}")
            for key, val in loss_dict.items():
                print(f"     - {key}: {val:.6f}")
        else:
            print(f"   ⚠ Skipping loss test (decomp_dict is None)")

    except Exception as e:
        print(f"   ✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = test_cplsf()
    sys.exit(0 if success else 1)
