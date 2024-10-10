#include <stdio.h>
#include "HYPRE.h"
#include "HYPRE_krylov.h"
#include "HYPRE_parcsr_ls.h"
#include "mpi.h"

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);                 // Initialize MPI
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    HYPRE_Init();  // Initialize HYPRE

    if (myid == 0) {
        printf("Hello, HYPRE World!\n");
    }

    // Finalize HYPRE and MPI
    HYPRE_Finalize();
    MPI_Finalize();

    return 0;
}

