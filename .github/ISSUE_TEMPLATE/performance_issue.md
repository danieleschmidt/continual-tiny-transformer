---
name: Performance Issue
about: Report a performance problem or bottleneck
title: '[PERFORMANCE] '
labels: performance
assignees: ''

---

## Performance Issue Description
**Describe the performance problem**
A clear description of what is running slower than expected.

## Current Performance
**Measurements and observations**
- Current execution time: ___
- Memory usage: ___
- CPU utilization: ___
- GPU utilization (if applicable): ___

## Expected Performance  
**What performance did you expect?**
- Expected execution time: ___
- Expected memory usage: ___
- Baseline or comparison: ___

## Reproduction Steps
**Steps to reproduce the performance issue:**
1. Dataset/model size: ___
2. Hardware configuration: ___
3. Run command: ___
4. Measurement method: ___

## Environment Details
- OS: [e.g. Ubuntu 20.04]
- Python version: [e.g. 3.9.0]
- Package version: [e.g. 1.0.0]
- Hardware: [CPU, GPU, Memory]
- Dependencies: [PyTorch, CUDA versions]

## Profiling Data
**Attach profiling results if available**
- Memory profiler output
- CPU profiler results  
- GPU profiler data (if applicable)
- Benchmarking results

## Suggested Optimizations
**Ideas for performance improvements**
- Algorithm optimizations
- Memory usage reductions
- Caching strategies
- Parallelization opportunities

## Impact Assessment
**How critical is this performance issue?**
- [ ] Blocking - unusable performance
- [ ] High - significantly impacts usability  
- [ ] Medium - noticeable but manageable
- [ ] Low - minor performance concern