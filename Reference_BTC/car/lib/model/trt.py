import tensorrt as trt
import torch
import numpy as np
import torch.nn as nn

from collections import OrderedDict, namedtuple

class trt_model(nn.Module):
    def __init__(self, weights, imgsize=(640,640)):
        super().__init__()

        # self.weights = weights
        w = weights
        print('Loading %s for TensorRT inference...'%w)

        self.device = torch.device('cuda:0')
        p = torch.cuda.get_device_properties(0)
        print('Using CUDA:0 (%s, %.0fMiB)'%(p.name, p.total_memory / (1 << 20)))
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.INFO)
        with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        context = model.create_execution_context()
        bindings = OrderedDict()
        output_names = []
        fp16 = False  # default updated below
        dynamic = False
        for i in range(model.num_bindings):
            name = model.get_binding_name(i)
            # print(name)
            dtype = trt.nptype(model.get_binding_dtype(i))
            # print(dtype)
            if model.binding_is_input(i):
                if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                    dynamic = True
                    context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                if dtype == np.float16:
                    fp16 = True
            else:  # output
                output_names.append(name)
            shape = tuple(context.get_binding_shape(i))
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
            bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        batch_size = bindings['images'].shape[0]  # if dynamic, this is instead max batch size

        self.__dict__.update(locals())


    def forward(self, im, augment=False, visualize=False):
 
        if self.dynamic and im.shape != self.bindings['images'].shape:
            # print('hello')
            i = self.model.get_binding_index('images')
            self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
            self.bindings['images'] = self.bindings['images']._replace(shape=im.shape)
            for name in self.output_names:
                # print(name)
                i = self.model.get_binding_index(name)
                self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
        
        s = self.bindings['images'].shape
        assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
        self.binding_addrs['images'] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        y = [self.bindings[x].data for x in sorted(self.output_names)]

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)


    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warmup model by running inference once
        im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
        for _ in range(3):  #
            self.forward(im)  # warmup

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x
    


