from unittest import TestCase
import torch
from libs.models import GenericMultiHeadLinearModel


class TestModels(TestCase):
    def test_generic_multi_head_model(self):
        model = GenericMultiHeadLinearModel(
            in_size=3,
            units=[2, 2],
            outs=[3, 4, 5],
            flatten=False,
        )
        inputs = torch.rand((2, 3))
        output = model(inputs)

        expected_shapes = [[2, 3], [2, 4], [2, 5]]

        for i in range(len(output)):
            self.assertEqual(expected_shapes[i], list(output[i].shape))
