require 'nn'
json=require 'cjson'

ip = torch.random(1,20)
op = torch.random(1,20)
kH = torch.random(1,10)
kW = torch.random(1,10)
iH = torch.random(kH, 64)
iW = torch.random(kW, 64)

--[[
ip = 1
op = 1
kH = 1
kW = 2
iH = 2
iW = 2
]]--

print('conv', ip, op, kH, kW, iH, iW)

mod = nn.SpatialConvolution(ip,op,kW,kH)
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
f = io.open('data/conv.json', 'w')
f:write(jenc)
f:close()

np = torch.random(1,20)
kH = torch.random(2,4)
kW = torch.random(2,4)
dH = torch.random(2,kH)
dW = torch.random(2,kW)
iH = torch.random(kH, 64)
iW = torch.random(kW, 64)

--[[
np = 8
kH = 2
kW = 2
dH = 2
dW = 2
iH = 2
iW = 2
]]--

mod = nn.SpatialMaxPooling(kW, kH, dW, dH)
inp = torch.randn(np, iH, iW)
out = mod:forward(inp)

enc = {}
enc.inp = inp:storage():totable()
enc.out = out:storage():totable()
enc.np = np
enc.kH = kH
enc.kW = kW
enc.iH = iH
enc.iW = iW
enc.dH = dH
enc.dW = dW

local oH = math.floor((iH - kH) / dH + 1)
local oW = math.floor((iW - kW) / dW + 1)
inpHWD = torch.randn(iH, iW, np)
outHWD = torch.randn(oH, oW, np)
for i = 0, np do
   for j = 0, iH do
      for k = 0, iW do
         inpHWD[{j, k, i}] = inp[{i, j, k}]
      end
   end
   for j = 0, oH do
      for k = 0, oW do
         outHWD[{j, k, i}] = out[{i, j, k}]
      end
   end
end
enc.inpHWD = inpHWD:storage():totable()
enc.outHWD = outHWD:storage():totable()

oH = out:size(2)
oW = out:size(3)

print('pool', np, kH, kW, dH, dW, iH, iW, oH, oW)

jenc = json.encode(enc)
f = io.open('data/pool.json', 'w')
f:write(jenc)
f:close()
--------------------------------------------------------
inSize = torch.random(1,100)
outSize = torch.random(1,100)
mod = nn.Linear(inSize, outSize)
inp = torch.randn(inSize)
out = mod:forward(inp)

enc = {}
enc.inp = inp:storage():totable()
enc.out = out:storage():totable()
enc.weight = mod.weight:storage():totable()
enc.bias = mod.bias:storage():totable()
enc.inSize = inSize
enc.outSize = outSize
print('linear', inSize, outSize)

jenc = json.encode(enc)
f = io.open('data/full.json', 'w')
f:write(jenc)
f:close()
