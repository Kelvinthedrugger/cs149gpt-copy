import argparse
import time
import math
import random
import inspect
from dataclasses import dataclass
import sys, getopt
from os import getcwd, path
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.cpp_extension import load
from torch.profiler import profile, record_function, ProfilerActivity
import module_ref as ms

NUM_THREADS=8
torch.set_num_threads(NUM_THREADS)

ispc_path = getcwd() + "/module_ispc.o"
if not path.exists(ispc_path): ispc_path = ""

print("\nCompiling code into a PyTorch module...\n\n")
mr = load(name="custom_module", sources=["module.cpp"],  extra_cflags=["-mavx", "-O3", "-fopenmp"], extra_ldflags=[ispc_path])
correctness_error_message = "\n-------------------------------------------\n YOUR ATTENTION PRODUCED INCORRECT RESULTS"

# generates dummy matrices for use in part0
def createQKVSimple(N,d,B,H):
    Q = torch.empty(B,H,N,d)
    K = torch.empty(B,H,d,N)
    V = torch.empty(B,H,N,d)
    for b in range(B):
        for h in range(H):
            for i in range(N):
                for j in range(d):
                    Q[b][h][i][j] = 0.0002 * i + 0.0001 * j
                    K[b][h][j][i] = 0.0006 * i + 0.0003 * j
                    V[b][h][i][j] = 0.00015 * i + 0.0008 * j
    K=K.transpose(-2,-1)
    return Q,K,V


def test(Q,K,V):
    with profile(activities=[ProfilerActivity.CPU],
            profile_memory=True, record_shapes=True) as prof:

        start = time.time()
        #compute QK^T
        QK = Q @ K.transpose(-2,-1)
        #compute softmax of QK^T
        QKSoftmax = F.softmax(QK, dim=3)
        QKV = QKSoftmax @ V
        end = time.time()
        pytorch_time = end - start

        #compute QK^TV

        attentionModule = CustomAttention(Q,K,V)
        start = time.time()
        QKS1 = attentionModule.myFusedAttention()
        QKS2 = attentionModule.myFlashAttention()
        QKS2 = attentionModule.myUnfusedAttention()
        #QKS1 = ms.my_attention(Q, K, V)
        end = time.time()
        manual_time = end - start
        print("Pytorch Execution Time:", pytorch_time, "\n")
        print("Manual Execution Time: ", manual_time, "\n")
        print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
        print(Q.shape)
        print(Q.stride())

def badSoftmax(Q, K, V):
    QK = Q @ K.transpose(-2,-1)
    #compute softmax of QK^T
    QKSoftmax = F.softmax(QK, dim=3)
    QKV = QKSoftmax @ V
    return QKV

def accessTest(B, H, N, d):
    Q,_ ,_ = createQKVSimple(N,d,B,H)
    print("\nTensor Shape:", Q.size())
    print("\n4D Tensor Contents:\n", Q)
    b = random.randrange(B)
    h = random.randrange(H)
    i = random.randrange(N)
    j = random.randrange(d)
    print("\nIndexing Value When: x = " + str(b) + ", y = " + str(h) + ", z = " + str(i) + ", b = " + str(j))
    expected = round(Q[b][h][i][j].item(), 6)
    result = round(mr.fourDimRead(Q.flatten().tolist(), b, h, i, j, H, N, d), 6)
    print("Expected:", expected)
    print("Result:", result)
    assert abs(expected - result) < 1e-5


# rewrite the whole thing so that it's concise
# design: use 'testcase' (str) to refer:
#  which module to test
#  whether there are additional intermediate tensors to be allocated upfront
# 1. testtemplate: don't have to create QKV over & over again
# @loops: (not designed yet) run the module 'loops' times to mitigate overhead of kickstarting the whole thing
# 2. get rid of the class of different attn modules: just use functions
# @func_name: use this to refer which function to get,
#   (use similar approach on str passed into 'record_function')
def get_intermediates(N, d, bc, br):
    # return list/tuple of tensors
    Qi = torch.zeros((br, d))
    Kj = torch.zeros((bc, d))
    Vj = torch.zeros((bc, d))
    Sij = torch.zeros((br, bc))
    Pij = torch.zeros((br, bc))
    PV = torch.zeros((br, d))
    Oi = torch.zeros((br, d))
    L = torch.zeros((N))
    Lnew = torch.zeros((br))
    Lij = torch.zeros((br))
    Li = torch.zeros((br))
    temp = [Qi, Kj, Vj, Sij, Pij, PV, Oi, L, Lnew, Lij, Li, bc, br]
    return temp

func_name =  {"part1": ["myNaiveAttention", "Naive Attention"],
  "part2": ["myUnfusedAttentionBlocked", "Blocked Unfused Attention"],
  "part3": ["myFusedAttention", "Fused Attention"],
  "part4": ["myFlashAttention", "Flash Attention"],
  }
# @dims: (N,d,B,H)
# testit() handles part 1 to 4
def testit(testcase, dims, is_ref=False, bc=0, br=0):
    N = dims[0]; d = dims[1]
    # qkv: tuple of (Q,K,V)
    qkv = createQKVSimple(*dims)
    start = time.time()
    QKV = badSoftmax(*qkv)
    end = time.time()
    pytorch_time = end - start
    # @temp: list of tensors to be passed into the attention module
    temp = None
    # @tsidx: int, extracted test number
    tsidx = int(testcase[4:])
    if tsidx == 0:
        pass
    elif tsidx < 3:
        temp = [torch.zeros((N, N))]
    elif tsidx == 3:
        temp = [torch.zeros((NUM_THREADS, N))]
    else: #part 4
        # list of intermediates
        temp = get_intermediates(N,d,bc,br)
    # if is_ref: is reference solution (.so file)
    #  -> test it as well
    func = None
    test_key = ""
    if is_ref:
        func = getattr(ms,func_name[testcase][0])
        test_key = "REFERENCE - "
    else:
        func = getattr(mr,func_name[testcase][0])
        test_key = "STUDENT - "
    test_key += func_name[testcase][1]
    with profile(activities=[ProfilerActivity.CPU],
            profile_memory=True, record_shapes=True) as prof:
        with record_function("model_inference"):
            start = time.time()
            with record_function(test_key):
                # @dims: (N,d,B,H); but we need B,H,N,d here
                QKS1 = func(*qkv, *temp, *(dims[2:]+dims[:2]) )
            end = time.time()
            manual_time = end - start
    assert torch.allclose(QKV,QKS1, atol=1e-4), correctness_error_message
    print("manual attention == pytorch attention",torch.allclose(QKV,QKS1, atol=1e-4))
    print("Pytorch Execution Time:", pytorch_time, "\n")
    print("Manual Execution Time: ", manual_time, "\n")
    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
    r = prof.key_averages()
    for rr in r:
        if rr.key == test_key:
            key, cpu_time, mem_usage = rr.key, rr.cpu_time, rr.cpu_memory_usage
            print (test_key+ " statistics")
            print("cpu time: ", str(cpu_time / 1000.0) + "ms")
            print("mem usage: ", mem_usage, "bytes")
    return

def part0Test(N, d, B, H):
    print("Running part 0 test: Pytorch Matmul + Softmax")
    Q,K,V = createQKVSimple(N,d,B,H)
    with profile(activities=[ProfilerActivity.CPU],
            profile_memory=True, record_shapes=True) as prof:
        start = time.time()
        #compute pytorch unfused softmax
        QKV = badSoftmax(Q,K,V)
        end = time.time()
        pytorch_time = end - start
    print("Pytorch Execution Time:", pytorch_time, "\n")
    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

def partNTest(testname, dims, bc, br):
    # REFERENCE solution
    print("-----RUNNING REFERENCE IMPLEMENTATION-----\n")
    testit(testname, dims, True, bc, br)
    time.sleep(3)
    # STUDENT's implementation
    print("-----RUNNING STUDENT IMPLEMENTATION-----\n")
    testit(testname, dims, False, bc, br)
    return

def main():

    d=32
    B=1
    H=4

    parser = argparse.ArgumentParser()
    parser.add_argument("testname", default="part0", help="name of test to run: part0, part1, part2, part3, part4, 4Daccess")
    parser.add_argument("-m", "--model", default="shakes128", help="name of model to use: shakes128, shakes1024, shakes2048, kayvon")
    parser.add_argument("--inference", action="store_true", default=False, help="run gpt inference")
    parser.add_argument("-bc",  default="256", help="Flash Attention Bc Size")
    parser.add_argument("-br", default="256", help="Flash Attention Br Size")
    parser.add_argument("-N", default="1024", help="Flash Attention Br Size")

    args = parser.parse_args()

    if args.model == "shakes128":
        N = 128
        model_filename = "out-shakespeare-char2048Good"
    elif args.model == "shakes256":
        N = 256
        model_filename = "out-shakespeare-char2048Good"
    elif args.model == "shakes1024":
        N = 1024
        model_filename = "out-shakespeare-char2048Good"
    elif args.model == "shakes2048":
        N = 2048
        model_filename = "out-shakespeare-char2048Good"
    else:
        print("Unknown model name: %s" % args.model)
        return

    if args.inference == False:
        N = int(args.N)
        if args.testname == "part0":
            part0Test(N, d, B, H)
        elif args.testname in {"part1", "part2", "part3", "part4",}:
            partNTest(args.testname, (N, d, B, H), int(args.bc), int(args.br))
        elif args.testname == "4Daccess":
            accessTest(1, 2, 4, 4)
        else:
            print("Unknown test name: %s" % args.testname)
    else:
        print("Running inference using dnn model %s" % (args.model))
        from sample import run_sample
        run_sample(N, model_filename, args.testname)


if __name__ == "__main__":
    main()
