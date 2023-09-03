import unittest
import torch
from sloppygrad.tensor import Tensor

class TestTensors(unittest.TestCase):

    def test_math_tensors(self):
        # we want to test addition against pytorch
        #define two tensors from pytorch
        ta,tb = torch.tensor([2.0]).double(),torch.tensor([3.0]).double()
        # define two tensors from sloppygrad 
        sa,sb = Tensor(2.0),Tensor(3.0)
        assert (ta + tb).item() ==  (sa + sb).data
        tc = torch.tensor([4.0]).double()
        sc = Tensor(4.0)
        assert ((ta + tb ) * tc).item() == ((sa + sb) * sc).data
        assert (ta**2).item() == (sa**2).data        
        # test out tanh function
        ttanh = torch.nn.Tanh()
        assert ttanh(ta).item() == sa.tanh().data
    
