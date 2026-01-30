![](https://github.com/FreePlacki/cuda-checkers/blob/main/checkers.gif)

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

Remove `-DINTRIN` to compile without intrinsics (if it doesn't compile on older cpus)
Remove `-DFORMATTING` to compile without custom console formatting.

Run:
```bash
./build/cuda-checkers logs/log.txt [init.txt]
```

See [logs](https://github.com/FreePlacki/cuda-checkers/edit/main/logs) directory for file syntax.

Tip: pressing Ctrl+D (Ctrl+Z on Windows, but probably won't work) when running AI vs AI will make the
game play out automatically ;)

## Detailed rules

- draw after 80 non-capture moves
- pieces move only forward
- kings move forward and backward one square
- captures are forced (can choose any)

## Performance

Currently achieving about 30 milion playouts (games simulated till the end) per second on a 3060 Ti.
