import math
class Tensor:
    """Tensor is a mathmatical object/data structure in a Neural Network
    it's the base building block, and the smallest form of data
    """
    def __init__(self, data,_children=(),_op='',label=''):
        # we put underscore, to mark variable/function for interanl class use.
        self.data = data
        self._prev = set(_children)
        self._op = _op
        # we set_backward as None, the default for a leaf node. it does not have a gradiant
        self._backward = lambda : None
        self.label = label
        self.grad = 0.0 # the derivate of the node with resepct to the output value( loss ) 
    def __repr__(self):
        return f"Tensor(data={self.data})"
    def __add__(self,obj):
        # to accpet, Tensor object and othey datatypes.
        obj = obj if isinstance(obj,Tensor) else Tensor(obj)
        out = Tensor(data = self.data + obj.data,_children=(self,obj),_op="+" )
        
        #TODO: is there a bette way to implement backward for each mlop ?
        #like tinygrad, we can have a mlop class, that all ops derivce from and implement 
        #thier own backward functionalities
        def _backward():
            """
                propagate through add operation
            """
            self.grad += 1.0 * out.grad
            obj.grad += 1.0 * out.grad
        #out being the output node.
        out._backward = _backward
        return out
              
    def __mul__(self,obj):
        obj = obj if isinstance(obj,Tensor) else Tensor(obj)
        out = Tensor(data=self.data * obj.data,_children=(self,obj),_op="*" )

        def _backward():
            #propagate through multiplication operation
            self.grad += obj.data * out.grad
            obj.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self,other):
        # make sure other, is a float or int 
        assert isinstance(other,(float,int)) , "invalid type supporting only float,int"
        out = Tensor(self.data**other, (self,), f'**{other}')

        def _backward():
            #TODO : look more into calcuating grad for power operation,
            #this is essential for more complex operation like ReLU, Tanh
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    #exmaple (  1 * Tensor ) 
    def __rmul__(self,other):
        # here we will swap the order of operands
        return self * other
    def __radd__(self,other):
        return self + other
    def __truediv__(self,other):
        # look more into the relation between division and multiplications
        return self * other**-1
    def __rtruediv__(self,other):
        return other * self**1
    def __sub__(self,other):
        return self + (-other)
    def __rsub__(self,other):
        return other + (-self)
        
    def tanh(self):
        # TODO: look into https://en.wikipedia.org/wiki/Hyperbolic_functions
        x = self.data
        n = (math.exp(2*x) - 1) / (math.exp(2*x) + 1 )
        out = Tensor(data = n,_children=(self,),_op='tanh')

        def _backward():
            self.grad += (1 - n**2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Tensor(math.exp(x),(self,), 'exp') 

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    def backward(self):
        # backward trough nodes topoligically
        topo =[]
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        self.grad=1.0
        build_topo(self)
        
        for node in reversed(topo):
            node._backward()

