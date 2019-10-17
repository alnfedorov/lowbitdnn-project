# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#pylint: disable-msg=too-many-arguments, too-many-locals, assignment-from-no-return
""" Conv Int8 functional and performance testing"""
import sys
import logging
import numpy as np
import tvm
import topi
from tvm import autotvm
from topi.cuda.conv2d import conv2d_NCHWc_int8, schedule_conv2d_NCHWc_int8
# import os
# os.environ["PATH"] += "/usr/local/cuda-10.1/bin/"
@autotvm.template
def conv2d(data, kernel, *args):
    cfg = autotvm.get_config()
    out = conv2d_NCHWc_int8(cfg, data, kernel, *args)
    s = tvm.create_schedule(out.op)
    s = schedule_conv2d_NCHWc_int8(cfg, s, out)
    fadd = tvm.build(s, [data, kernel, out], 'cuda', name="conv")
    dev_module = fadd.imported_modules[0]
    print("-----GPU code-----")
    print(dev_module.get_source())
    print(s)
    return s, [data, kernel, out]

data_shape = (32, 32, 128, 128, 4)
kernel_shape = (32, 4, 40, 40, 4, 4)

# Create TVM placeholders
data = tvm.placeholder(data_shape, name='data', dtype='int8')
kernel = tvm.placeholder(kernel_shape, name='kernel', dtype='int8')

stride = 1
padding = 1
dilation = 1
layout = 'NCHW4c'
out_dtype = 'int32'

task = autotvm.task.create(conv2d, args=(data, kernel, stride, padding, dilation, layout, out_dtype), target='cuda')

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
LOGGER = logging.getLogger('test_conv_int8_intel')
LOGGER.disabled = False

TARGET_NAME = 'cuda'
CTX = tvm.context(TARGET_NAME)


data_array = tvm.nd.array(np.random.randint(100, size=data_shape).astype('int8'))
kernel_array = tvm.nd.array(np.random.randint(100, size=kernel_shape).astype('int8'))

# c_orig will be used for declaration ouptut
# c_sch will be used for scheduled computation output
c_orig = tvm.nd.array(np.zeros(data_shape, dtype='int8'), CTX)
c_sch = tvm.nd.array(np.zeros(data_shape, dtype='int8'), CTX)

cfg = ConfigEntity()
with tvm.target.create(TARGET_NAME):
    conv = topi.cuda.conv2d.conv2d_NCHWc_int8(data, kernel, stride=1, dilation=1,
                                padding=1, layout='NCHWc',
                                out_layout='NCHWc', out_dtype='int8')
    out = conv
    sch = tvm.create_schedule(out.op)
    func = tvm.build(sch, [data, kernel, out], target=TARGET_NAME, name='out')
    func(data_array, kernel_array, c_orig)
    LOGGER.debug(tvm.lower(sch, [data, kernel], simple_mode=True))

    # Generate and run the optimized schedule
    sconv = topi.generic.nn.schedule_conv2d_NCHWc(outs=[out])
    func = tvm.build(sconv, [data, kernel, out], target=TARGET_NAME, name='conv')
    func(data_array, kernel_array, c_sch)

    # Functional check
    assert np.allclose(c_orig.asnumpy(), c_sch.asnumpy())

    evaluator = func.time_evaluator(func.entry_name, CTX, number=1000)
    LOGGER.debug(tvm.lower(sconv, [data, kernel], simple_mode=True))
    print(evaluator(data_array, kernel_array, c_sch).mean)




















import numpy as np
import tvm

# The sizes of inputs and filters
batch = 256
in_channel = 256
out_channel = 512
in_size = 14
kernel = 3
pad = 1
stride = 1

# Algorithm
A = tvm.placeholder((in_size, in_size, in_channel, batch), name='A')
W = tvm.placeholder((kernel, kernel, in_channel, out_channel), name='W')
out_size = (in_size - kernel + 2*pad) // stride + 1
# Pad input
Apad = tvm.compute(
    (in_size + 2*pad, in_size + 2*pad, in_channel, batch),
    lambda yy, xx, cc, nn: tvm.if_then_else(
        tvm.all(yy >= pad, yy - pad < in_size,
                xx >= pad, xx - pad < in_size),
        A[yy - pad, xx - pad, cc, nn], tvm.const(0., "float32")),
    name='Apad')
# Create reduction variables
rc = tvm.reduce_axis((0, in_channel), name='rc')
ry = tvm.reduce_axis((0, kernel), name='ry')
rx = tvm.reduce_axis((0, kernel), name='rx')
# Compute the convolution
B = tvm.compute(
    (out_size, out_size, out_channel, batch),
    lambda yy, xx, ff, nn: tvm.sum(
        Apad[yy * stride + ry, xx * stride + rx, rc, nn] * W[ry, rx, rc, ff],
        axis=[ry, rx, rc]),
    name='B')

# Designate the memory hierarchy
s = tvm.create_schedule(B.op)
s[Apad].compute_inline() # compute Apad inline
AA = s.cache_read(Apad, 'shared', [B])
WW = s.cache_read(W, "shared", [B])
AL = s.cache_read(AA, "local", [B])
WL = s.cache_read(WW, "local", [B])
BL = s.cache_write(B, "local")

# tile consts
tile = 8
num_thread = 8
block_factor = tile * num_thread
step = 8
vthread = 2

# Get the GPU thread indices
block_x = tvm.thread_axis("blockIdx.x")
block_y = tvm.thread_axis("blockIdx.y")
block_z = tvm.thread_axis("blockIdx.z")
thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
thread_y = tvm.thread_axis((0, num_thread), "threadIdx.y")
thread_xz = tvm.thread_axis((0, vthread), "vthread", name="vx")
thread_yz = tvm.thread_axis((0, vthread), "vthread", name="vy")

# Split the workloads
hi, wi, fi, ni = s[B].op.axis
bz = s[B].fuse(hi, wi)
by, fi = s[B].split(fi, factor=block_factor)
bx, ni = s[B].split(ni, factor=block_factor)

# Bind the iteration variables to GPU thread indices
s[B].bind(bz, block_z)
s[B].bind(by, block_y)
s[B].bind(bx, block_x)

tyz, fi = s[B].split(fi, nparts=vthread)  # virtual thread split
txz, ni = s[B].split(ni, nparts=vthread)  # virtual thread split
ty, fi = s[B].split(fi, nparts=num_thread)
tx, ni = s[B].split(ni, nparts=num_thread)
s[B].reorder(bz, by, bx, tyz, txz, ty, tx, fi, ni)

s[B].bind(tyz, thread_yz)
s[B].bind(txz, thread_xz)
s[B].bind(ty, thread_y)
s[B].bind(tx, thread_x)


# Schedule BL local write
s[BL].compute_at(s[B], tx)
yi, xi, fi, ni = s[BL].op.axis
ry, rx, rc = s[BL].op.reduce_axis
rco, rci = s[BL].split(rc, factor=step)
s[BL].reorder(rco, ry, rx, rci, fi, ni)

# Attach computation to iteration variables
s[AA].compute_at(s[BL], rx)
s[WW].compute_at(s[BL], rx)
s[AL].compute_at(s[BL], rci)
s[WL].compute_at(s[BL], rci)

# Schedule for A's shared memory load
yi, xi, ci, ni = s[AA].op.axis
ty, ci = s[AA].split(ci, nparts=num_thread)
tx, ni = s[AA].split(ni, nparts=num_thread)
_, ni = s[AA].split(ni, factor=4)
s[AA].reorder(ty, tx, yi, xi, ci, ni)
s[AA].bind(ty, thread_y)
s[AA].bind(tx, thread_x)
s[AA].vectorize(ni)  # vectorize memory load

# Schedule for W's shared memory load
yi, xi, ci, fi = s[WW].op.axis
ty, ci = s[WW].split(ci, nparts=num_thread)
tx, fi = s[WW].split(fi, nparts=num_thread)
_, fi = s[WW].split(fi, factor=4)
s[WW].reorder(ty, tx, yi, xi, ci, fi)
s[WW].bind(ty, thread_y)
s[WW].bind(tx, thread_x)
s[WW].vectorize(fi)  # vectorize memory load

func = tvm.build(s, [A, W, B], 'cuda')
ctx = tvm.gpu(0)
a_np = np.random.uniform(size=(in_size, in_size, in_channel, batch)).astype(A.dtype)
w_np = np.random.uniform(size=(kernel, kernel, in_channel, out_channel)).astype(W.dtype)
a = tvm.nd.array(a_np, ctx)
w = tvm.nd.array(w_np, ctx)
b = tvm.nd.array(np.zeros((out_size, out_size, out_channel, batch), dtype=B.dtype), ctx)
func(a, w, b)
evaluator = func.time_evaluator(func.entry_name, ctx, number=1)
print('Convolution: %f ms' % (evaluator(a, w, b).mean * 1e3))
