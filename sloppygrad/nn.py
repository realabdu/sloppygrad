from sloppygrad.tensor import Tensor
import random 
class Neuron:
    def __init__(self,num_ws):
        # dimensoin of the nueron
        self.ws = [ Tensor(random.uniform(-1,1)) for _ in range(num_ws) ]
        self.b  =  Tensor(random.uniform(-1,1))
    def __call__(self,xs):
        assert len(xs) == len(self.ws),"number of wights does not match number of data"
        act = sum(([x*w for x,w in zip(xs,self.ws)]),self.b) 
        o = act.tanh()
        return o
    def parameters(self):
        return self.ws + [self.b] 
        
        

class Layer:
    def __init__(self,n_dim,n_out):
        self.neurons = [Neuron(n_dim) for _ in range(n_out)]
    def __call__(self,x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()] 
class MLP:
    def __init__(self,num_xs,num_neu):
        # we want to take two consctuve pair in this list     
        specs = [num_xs] + num_neu 
        self.layers = [Layer(specs[i],specs[i+1]) for i in range(len(num_neu))]

    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [l for layer in self.layers for l in layer.parameters()]
