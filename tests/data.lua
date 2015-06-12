require 'nn'
json=require 'cjson'

ip = 1
op = 5
kH = 3
kW = 3
iH = 7
iW = 7

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
