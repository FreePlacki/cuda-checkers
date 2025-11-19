# CUDA checkers

Extremely fast checkers solver using [Monte Carlo Tree Search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)

## Usage

Requirements:
- CUDA-capable GPU
- [nvcc](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

```bash
# build
make
# run
./cuda-checkers log_file.txt [game_init_file.txt]
```

See [logs](https://github.com/FreePlacki/cuda-checkers/edit/main/logs) directory for file syntax.

NOTE: on Windows add to nvcc flags: `-ccbin "path\to\cl.exe directory"`,
ex. "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"

## Performance

Currently achieving about 10 milion playouts (games simulated till the end) per second on a 3060 Ti.
