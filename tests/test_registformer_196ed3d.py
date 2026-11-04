import unittest
from types import SimpleNamespace

import torch

from src.models.components.network_registformer_196ed3d import (
    RegistFormer, flow_to_grid, flow_guide_sampler,
)
from src.models.components.network_voxelmorph_original import SpatialTransformer
from src.models.registformer_196ed3d_module import IXIRegistFormer196ed3dModule


class RegistFormerRoutingTests(unittest.TestCase):
    def test_ground_truth_is_reserved_for_evaluation(self):
        module = IXIRegistFormer196ed3dModule.__new__(IXIRegistFormer196ed3dModule)
        torch.nn.Module.__init__(module)
        module.params = SimpleNamespace(is_3d=False)
        module.has_multiple_outputs = False
        seen = []
        module.forward = lambda a, ref: seen.append((a, ref)) or a + ref
        source, target, ref = [torch.full((1, 1, 8, 12), x) for x in (0., 1., 2.)]
        module.training = True
        result = module.model_step((source, target, ref))
        self.assertIs(result[1], ref)
        self.assertIs(seen[-1][1], ref)
        module.training = False
        result = module.model_step((source, target, ref))
        self.assertIs(result[1], target)
        self.assertIs(seen[-1][1], ref)

    def test_flow_uses_real_reference_moving_and_converts_voxelmorph_axes(self):
        class DAM(torch.nn.Module):
            def forward(self, source, ref):
                return source * 0.5

        class Flow(torch.nn.Module):
            def forward(self, moving, fixed, registration=False):
                self.inputs = moving, fixed
                flow = torch.empty(1, 2, 8, 12)
                flow[:, 0] = 1  # dy
                flow[:, 1] = 2  # dx
                return moving, flow

        class Features(torch.nn.Module):
            def forward(self, value):
                return [value[:, :1]] * 6

        class Attention(torch.nn.Module):
            def forward(self, q, k, v, flow):
                torch.testing.assert_close(flow[:, 0], torch.full_like(flow[:, 0], 2))
                torch.testing.assert_close(flow[:, 1], torch.full_like(flow[:, 1], 1))
                return q

        net = RegistFormer.__new__(RegistFormer)
        torch.nn.Module.__init__(net)
        net.DAM, net.flow_estimator = DAM(), Flow()
        net.dam_type, net.flow_type = 'proposed_synthesis', 'voxelmorph_lightning'
        net.flow_size, net.is_moved_feat, net.daca_loc = [8, 12], False, None
        net.unet_q, net.unet_k = Features(), Features()
        net.trans_unit = torch.nn.ModuleList([Attention() for _ in range(3)])
        for index in range(7):
            setattr(net, 'conv' + str(index), torch.nn.Identity())
        source, ref = torch.rand(1, 1, 8, 12), torch.rand(1, 1, 8, 12)
        net(torch.cat((source, ref), 1))
        torch.testing.assert_close(net.flow_estimator.inputs[0], ref)
        torch.testing.assert_close(net.flow_estimator.inputs[1], source * 0.5)

    def test_daca_sampling_matches_voxelmorph_spatial_transform(self):
        ref = torch.arange(8 * 12, dtype=torch.float32).reshape(1, 1, 8, 12)
        flow_yx = torch.empty(1, 2, 8, 12)
        flow_yx[:, 0], flow_yx[:, 1] = 1.25, -0.5
        expected = SpatialTransformer((8, 12))(ref, flow_yx)
        actual = flow_guide_sampler(ref, flow_to_grid(flow_yx[:, [1, 0]], k_size=1), k_size=1)
        torch.testing.assert_close(actual[:, 0], expected)


if __name__ == '__main__':
    unittest.main()
