require 'nn'
json=require 'cjson'

ip = torch.random(1,20)
op = torch.random(1,20)
kH = torch.random(1,10)
kW = torch.random(1,10)
iH = torch.random(kH, 64)
iW = torch.random(kW, 64)

-- test W: 9x6x10x1 I: 6x45x21
-- test non-equal kernels
mod = nn.SpatialConvolution(ip,op,kH,kW)
inp = torch.randn(ip, iH, iW)
out = mod:forward(inp)

enc = {}
enc.weight = mod.weight:storage():totable()
enc.bias =  mod.bias:storage():totable()
enc.inp = inp:storage():totable()
enc.out = out:storage():totable()
enc.ip = ip
enc.op = op
enc.kH = kH
enc.kW = kW
enc.iH = iH
enc.iW = iW


jenc = json.encode(enc)

f = io.open('conv.json', 'w')
f:write(jenc)
f:close()
