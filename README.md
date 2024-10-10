
# Linear Solver Experiment

This project demonstrates the use of the HYPRE library in solving linear systems through simple examples. The project includes three executables:
- `hello_hypre`: Basic introduction to HYPRE.
- `poisson`: Solves the Poisson equation using HYPRE.
- `diffusion`: Solves a diffusion equation using HYPRE.

## Prerequisites

Before compiling, ensure that you have the following dependencies installed on your system:

- **CMake** (minimum version 3.10)
- **HYPRE** (properly built and accessible in your environment)
- **MPI** (for parallel execution)

## Project Setup

Make sure that you have HYPRE installed and built. The `HYPRE_DIR` variable in the `CMakeLists.txt` file should point to the directory where HYPRE is installed.

```bash
# Example HYPRE directory structure (adjust as necessary)
HYPRE_DIR="../hypre/src"
```

If you have a different path, you can modify the `HYPRE_DIR` in the `CMakeLists.txt` to point to the correct location.

## Compilation Instructions

Follow these steps to compile the project:

1. **Clone the repository** (or move to the directory where the source code is):
   
   ```bash
   cd linear_solver_experiment
   ```

2. **Create a build directory**:
   
   It’s recommended to build the project in a separate directory to keep things clean.

   ```bash
   mkdir build
   cd build
   ```

3. **Run CMake**:

   Use CMake to generate the build files, pointing it to the directory containing the `CMakeLists.txt`.

   ```bash
   cmake ..
   ```

   This will configure the project and set up the necessary build files.

4. **Compile the project**:

   After running CMake, you can compile the executables using `make`.

   ```bash
   make
   ```

   If successful, this will generate the following executables:
   - `hello_hypre`
   - `poisson`
   - `diffusion`
