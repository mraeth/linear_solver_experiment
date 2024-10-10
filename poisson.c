#include <stdio.h>
#include <string>
#include <stdlib.h>
#include "_hypre_utilities.h"
#include "HYPRE_struct_ls.h"
#include "HYPRE.h"

void write_array(HYPRE_StructVector x, int ilower, int iupper, const char * file_name){
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    FILE *file;
    if (myid == 0) {
        file = fopen(file_name, "w");
        if (file == NULL) {
            printf("Error opening file for writing.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    /* Write solution in matrix form */
    for (int i = ilower; i <= iupper; i++) {
        if (myid == 0) {
            // Start a new row
            for (int j = ilower; j <= iupper; j++) {
                int index[2] = {i, j};
                double u_value;
                HYPRE_StructVectorGetValues(x, index, &u_value);

                // Write the value followed by a space
                fprintf(file, "%f ", u_value);
            }
            // End the row with a newline character
            fprintf(file, "\n");
        }
    }

    if (myid == 0) {
        fclose(file);
    }
}

/*
 * This example solves the 2D variable-coefficient Poisson equation with Dirichlet boundary conditions
 * on a regular grid using HYPRE's structured grid interface and the SMG (Semi-Coarsening Multigrid) solver.
 *
 * The equation being solved is of the form:
 *
 *     div(grad(u)) = f
 *
 * where:
 *     - u is the unknown solution we seek (e.g., temperature, pressure, etc.)
 *     - a(x, y) is a spatially varying diffusion coefficient (defined on the grid)
 *     - f(x, y) is the source term (right-hand side), which in this case is set to 1 for simplicity
 *
 * The equation is solved on a 2D grid with dimensions nx x ny.
 * The grid is discretized using a 5-point stencil that approximates the divergence and gradient operators
 * (center, left, right, up, down), and the matrix system corresponding to the discretized equation is constructed.
 * 
 * Dirichlet boundary conditions are imposed (u = 0) on the boundary of the grid, meaning the solution is set to zero
 * at the edges of the domain.
 *
 * The matrix equation is then solved using the SMG solver, and the solution u is written to an output file 'result.dat'.
 */

int main (int argc, char *argv[])
{
   int i, j;

   int myid, num_procs;

   int n, N, pi, pj;
   double h, h2;
   int ilower[2], iupper[2];

   int solver_id;
   int n_pre, n_post;

   HYPRE_StructGrid     grid;
   HYPRE_StructStencil  stencil;
   HYPRE_StructMatrix   A;
   HYPRE_StructVector   b;
   HYPRE_StructVector   x;
   HYPRE_StructSolver   solver;
   HYPRE_StructSolver   precond;

   int num_iterations;
   double final_res_norm;

   int vis;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   /* Set defaults */
   n = 65;
   solver_id = 0;
   n_pre  = 1;
   n_post = 1;
   vis = 0;


   /* Initialize HYPRE */
   HYPRE_Initialize();

   /* Print GPU info */
   /* HYPRE_PrintDeviceInfo(); */

   /* Figure out the processor grid (N x N).  The local problem
      size for the interior nodes is indicated by n (n x n).
      pi and pj indicate position in the processor grid. */
   N  = sqrt(num_procs);
   h  = 1.0 / (N * n + 1); /* note that when calculating h we must
                          remember to count the boundary nodes */
   h2 = h * h;
   pj = myid / N;
   pi = myid - pj * N;

   /* Figure out the extents of each processor's piece of the grid. */
   ilower[0] = pi * n;
   ilower[1] = pj * n;

   iupper[0] = ilower[0] + n - 1;
   iupper[1] = ilower[1] + n - 1;

   /* 1. Set up a grid */
   {
      /* Create an empty 2D grid object */
      HYPRE_StructGridCreate(MPI_COMM_WORLD, 2, &grid);

      /* Add a new box to the grid */
      HYPRE_StructGridSetExtents(grid, ilower, iupper);
      int periodic[2] = {n+1, n+1}; // Periodicity in x and y directions
      HYPRE_StructGridSetPeriodic(grid,periodic);

      /* This is a collective call finalizing the grid assembly.
         The grid is now ``ready to be used'' */
      HYPRE_StructGridAssemble(grid);
   }

   /* 2. Define the discretization stencil */
   {
      /* Create an empty 2D, 5-pt stencil object */
      HYPRE_StructStencilCreate(2, 5, &stencil);

      /* Define the geometry of the stencil */
      {
         int entry;
         int offsets[5][2] = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};

         for (entry = 0; entry < 5; entry++)
         {
            HYPRE_StructStencilSetElement(stencil, entry, offsets[entry]);
         }
      }
   }

   /* 3. Set up a Struct Matrix */
   {
      int nentries = 5;
      int nvalues = nentries * n * n;
      double *values;
      int stencil_indices[5];

      /* Create an empty matrix object */
      HYPRE_StructMatrixCreate(MPI_COMM_WORLD, grid, stencil, &A);

      /* Indicate that the matrix coefficients are ready to be set */
      HYPRE_StructMatrixInitialize(A);

      values = (double*) calloc(nvalues, sizeof(double));

      for (j = 0; j < nentries; j++)
      {
         stencil_indices[j] = j;
      }

      /* Set the standard stencil at each grid point,
         we will fix the boundaries later */
      for (i = 0; i < nvalues; i += nentries)
      {
         values[i] = 4.0;
         for (j = 1; j < nentries; j++)
         {
            values[i + j] = -1.0;
         }
      }

      HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, nentries,
                                     stencil_indices, values);

      free(values);
   }

   /* 4. Incorporate the zero boundary conditions: go along each edge of
         the domain and set the stencil entry that reaches to the boundary to
         zero.*/
   {
      int bc_ilower[2];
      int bc_iupper[2];
      int nentries = 1;
      int nvalues  = nentries * n; /*  number of stencil entries times the length
                                     of one side of my grid box */
      double *values;
      int stencil_indices[1];

      values = (double*) calloc(nvalues, sizeof(double));
      for (j = 0; j < nvalues; j++)
      {
         values[j] = 0.0;
      }

      /* Recall: pi and pj describe position in the processor grid */
      if (pj == 0)
      {
         /* Bottom row of grid points */
         bc_ilower[0] = pi * n;
         bc_ilower[1] = pj * n;

         bc_iupper[0] = bc_ilower[0] + n - 1;
         bc_iupper[1] = bc_ilower[1];

         stencil_indices[0] = 3;

         HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, nentries,
                                        stencil_indices, values);
      }

      if (pj == N - 1)
      {
         /* upper row of grid points */
         bc_ilower[0] = pi * n;
         bc_ilower[1] = pj * n + n - 1;

         bc_iupper[0] = bc_ilower[0] + n - 1;
         bc_iupper[1] = bc_ilower[1];

         stencil_indices[0] = 4;

         HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, nentries,
                                        stencil_indices, values);
      }

      if (pi == 0)
      {
         /* Left row of grid points */
         bc_ilower[0] = pi * n;
         bc_ilower[1] = pj * n;

         bc_iupper[0] = bc_ilower[0];
         bc_iupper[1] = bc_ilower[1] + n - 1;

         stencil_indices[0] = 1;

         HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, nentries,
                                        stencil_indices, values);
      }

      if (pi == N - 1)
      {
         /* Right row of grid points */
         bc_ilower[0] = pi * n + n - 1;
         bc_ilower[1] = pj * n;

         bc_iupper[0] = bc_ilower[0];
         bc_iupper[1] = bc_ilower[1] + n - 1;

         stencil_indices[0] = 2;

         HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, nentries,
                                        stencil_indices, values);
      }

      free(values);
   }

   /* This is a collective call finalizing the matrix assembly.
      The matrix is now ``ready to be used'' */
   HYPRE_StructMatrixAssemble(A);

   /* 5. Set up Struct Vectors for b and x */
   {
      int    nvalues = n * n;
      double *values;

      values = (double*) calloc(nvalues, sizeof(double));

      /* Create an empty vector object */
      HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &b);
      HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &x);

      /* Indicate that the vector coefficients are ready to be set */
      HYPRE_StructVectorInitialize(b);
      HYPRE_StructVectorInitialize(x);

      /* Set the values */
      for (i = 0; i < nvalues; i ++)
      {
         values[i] = 0.3*h2*i ;
      }



      values[2000] = 10.0;
      values[1000] = +2.0;
      values[3000] = 3.0;
      values[500] = -5.0;

      values[2500] = -5.0;
      values[1500] = -10.0;
      values[n+1] = -20.0;


      HYPRE_StructVectorSetBoxValues(b, ilower, iupper, values);

      for (i = 0; i < nvalues; i ++)
      {
         values[i] = 1.0;
      }
      HYPRE_StructVectorSetBoxValues(x, ilower, iupper, values);

      free(values);

      /* This is a collective call finalizing the vector assembly.
         The vector is now ``ready to be used'' */
      HYPRE_StructVectorAssemble(b);
      HYPRE_StructVectorAssemble(x);
   }

    write_array(x, ilower[0], iupper[0], "x.dat");
    write_array(b, ilower[0], iupper[0], "b.dat");


   /* 6. Set up and use a struct solver
      (Solver options can be found in the Reference Manual.) */
   {
      HYPRE_StructSMGCreate(MPI_COMM_WORLD, &solver);
      HYPRE_StructSMGSetMemoryUse(solver, 0);
      HYPRE_StructSMGSetMaxIter(solver, 50);
      HYPRE_StructSMGSetTol(solver, 1.0e-06);
      HYPRE_StructSMGSetRelChange(solver, 0);
      HYPRE_StructSMGSetNumPreRelax(solver, n_pre);
      HYPRE_StructSMGSetNumPostRelax(solver, n_post);
      /* Logging must be on to get iterations and residual norm info below */
      HYPRE_StructSMGSetLogging(solver, 1);

      /* Setup and solve */
      HYPRE_StructSMGSetup(solver, A, b, x);
      HYPRE_StructSMGSolve(solver, A, b, x);

      /* Get some info on the run */
      HYPRE_StructSMGGetNumIterations(solver, &num_iterations);
      HYPRE_StructSMGGetFinalRelativeResidualNorm(solver, &final_res_norm);

      /* Clean up */
      HYPRE_StructSMGDestroy(solver);
   }

   write_array(x, ilower[0], iupper[0], "result.dat");


   if (myid == 0)
   {
      printf("\n");
      printf("Iterations = %d\n", num_iterations);
      printf("Final Relative Residual Norm = %g\n", final_res_norm);
      printf("\n");
   }

   /* Free memory */
   HYPRE_StructGridDestroy(grid);
   HYPRE_StructStencilDestroy(stencil);
   HYPRE_StructMatrixDestroy(A);
   HYPRE_StructVectorDestroy(b);
   HYPRE_StructVectorDestroy(x);

   /* Finalize HYPRE */
   HYPRE_Finalize();

   /* Finalize MPI */
   MPI_Finalize();

   return (0);
}