""""
Benchmark for Neural Edge

Measures inference latency (p50/p95) and FPS 
so everey variant (FP32 / pruned / INT8) is compared with the same rules and tools.

"""
import numpy as np
import cv2
import time
import torch
from ultralytics import YOLO

#TODO : Load Model
#TODO : read N frames from a video by default, into a list once (Important to keep I/O out of the timing loop)
#TODO : warm up loop, need to run inference X times, discard timings of loading and setting up (that can wrongly alterate the results)
#TODO : torch.cuda.synchronize() once after warm up (to keep the process synchrone and not async, so to get accurate results in the benchmark)
#TODO : measure loop per frame. t0 --> inference --> synchronize --> t1 --> append ms to list of times
#TODO : print the results

