# CUDA checkers

Extremely fast checkers solver using [Monte Carlo Tree Search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)

## Usage

Requirements:
- CUDA-capable GPU
- [nvcc](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

Linux:
```bash
cmake -S . -B build
cmake --build build --config Release
```

Windows:
```bash
cmake -S . -B build -G "Visual Studio 17 2022"
cmake --build build --config Release
```


See [logs](https://github.com/FreePlacki/cuda-checkers/edit/main/logs) directory for file syntax.

## Performance

Currently achieving about 30 milion playouts (games simulated till the end) per second on a 3060 Ti.
