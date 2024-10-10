/*
   Simplified program based on example 4

   Interface:      Structured interface (Struct)


   We recommend viewing examples 1, 2, and 3 before viewing this
                   example.

   Description:    This example differs from the previous structured example
                   (Example 3) in that a more sophisticated stencil and
                   boundary conditions are implemented. The method illustrated
                   here to implement the boundary conditions is much more general
                   than that in the previous example.  Also symmetric storage is
                   utilized when applicable.

                   This code solves the convection-reaction-diffusion problem
                   div (-K grad u ) = F in the unit square with
                   boundary condition u = U0.  The domain is split into N x N
                   processor grid.  Thus, the given number of processors should
                   be a perfect square. Each processor has a n x n grid, with
                   nodes connected by a 5-point stencil. Note that the struct
                   interface assumes a cell-centered grid, and, therefore, the
                   nodes are not shared.

                   To incorporate the boundary conditions, we do the following:
                   Let x_i and x_b be the interior and boundary parts of the
                   solution vector x. If we split the matrix A as
                             A = [A_ii A_ib; A_bi A_bb],
                   then we solve
                             [A_ii 0; 0 I] [x_i ; x_b] = [b_i - A_ib u_0; u_0].
                   Note that this differs from the previous example in that we
                   are actually solving for the boundary conditions (so they
                   may not be exact as in ex3, where we only solved for the
                   interior).  This approach is useful for more general types
                   of b.c.


*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <cstdlib> // for std::atoi
// #include "HYPRE_krylov.h"
#include "HYPRE_struct_ls.h"

#define PI 3.14159265358979

/* Macro to evaluate a function F in the grid point (i,j) */
#define Eval(F,i,j) (F( (ilower[0]+(i))*h, (ilower[1]+(j))*h )*n*n)
#define bcEval(F,i,j) (F( (bc_ilower[0]+(i))*h, (bc_ilower[1]+(j))*h )*n*n)

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

/* Diffusion coefficient */
double K(double x, double y)
{
    return  0.1+exp(3*x*x)*abs(sin(2*PI*y)*sin(2*PI*x)); //x * x + exp(y);
}

/* Boundary condition */
double U0(double x, double y)
{
    return 0.;
}

/* Right-hand side */
double F(double x, double y)
{
   return exp(-((x-0.5)*(x-0.5)+(y-0.5)*(y-0.5))*10.);
}

int main (int argc, char *argv[])
{
   int i, j, k;

   int myid, num_procs;

   int n, N, pi, pj;
   double h, h2;
   int ilower[2], iupper[2];

   int n_pre, n_post;
   double mytime = 0.0;
   double walltime = 0.0;

   int num_iterations;
   double final_res_norm;


    int solver_id = 0;

    // Check if the solver_id is provided as a command-line argument
    if (argc > 1) {
        solver_id = std::atoi(argv[1]); // Convert argument to an integer

        // Validate the solver_id
        if (solver_id != 0 && solver_id != 1) {
            std::cerr << "Invalid solver_id: " << solver_id 
                      << ". Allowed values are 0 (SMG) or 1 (GMRES)." << std::endl;
            return 1;
        }
    }

    // Choose solver based on solver_id
    if (solver_id == 0) {
        std::cout << "Selected solver: SMG (ID: 0)" << std::endl;
    } else if (solver_id == 1) {
        std::cout << "Selected solver: GMRES (/w SMG preconditioner)(ID: 1)" << std::endl;
    }



   HYPRE_StructGrid     grid;
   HYPRE_StructStencil  stencil;
   HYPRE_StructMatrix   A;
   HYPRE_StructVector   b;
   HYPRE_StructVector   x;
   HYPRE_StructVector   alpha;
   HYPRE_StructSolver   solver;
   HYPRE_StructSolver   precond;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   /* Initialize HYPRE */
   HYPRE_Initialize();

   /* Print GPU info */

   /* Set default parameters */
   n         = 1000;
   n_pre     = 1;
   n_post    = 1;

   /* Figure out the processor grid (N x N).  The local
      problem size is indicated by n (n x n). pi and pj
      indicate position in the processor grid. */
   N  = sqrt(num_procs);
   h  = 1.0 / (N * n - 1);
   h2 = h * h;
   pj = myid / N;
   pi = myid - pj * N;

   /* Define the nodes owned by the current processor (each processor's
      piece of the global grid) */
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

      /* This is a collective call finalizing the grid assembly.
         The grid is now ``ready to be used'' */
      HYPRE_StructGridAssemble(grid);
   }

   /* 2. Define the discretization stencil */
   {
      /* Define the geometry of the stencil */
      int offsets[5][2] = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};

      /* Create an empty 2D, 5-pt stencil object */
      HYPRE_StructStencilCreate(2, 5, &stencil);

      /* Assign stencil entries */
      for (i = 0; i < 5; i++)
      {
         HYPRE_StructStencilSetElement(stencil, i, offsets[i]);
      }
   }


   /* 3. Set up Struct Vectors for b and x */
   {
      double *values;

      /* Create an empty vector object */
      HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &b);
      HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &x);
      HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &alpha);

      /* Indicate that the vector coefficients are ready to be set */
      HYPRE_StructVectorInitialize(b);
      HYPRE_StructVectorInitialize(x);
      HYPRE_StructVectorInitialize(alpha);

      values = (double*) calloc((n * n), sizeof(double));

      /* Set the values of b in left-to-right, bottom-to-top order */
      for (k = 0, j = 0; j < n; j++)
         for (i = 0; i < n; i++, k++)
         {
            values[k] = h2 * Eval(F, i, j);
         }
      HYPRE_StructVectorSetBoxValues(b, ilower, iupper, values);

      /* Set x = 0 */
      for (i = 0; i < (n * n); i ++)
      {
         values[i] = 0.0;
      }
      HYPRE_StructVectorSetBoxValues(x, ilower, iupper, values);

      /* Set x = 0 */
      for (k = 0, j = 0; j < n; j++)
         for (i = 0; i < n; i++, k++)
         {
            values[k] = h2 * Eval(K, i, j);
         }
      HYPRE_StructVectorSetBoxValues(alpha, ilower, iupper, values);

      free(values);

      /* Assembling is postponed since the vectors will be further modified */
   }

   /* 4. Set up a Struct Matrix */
   {
      /* Create an empty matrix object */
      HYPRE_StructMatrixCreate(MPI_COMM_WORLD, grid, stencil, &A);

      /* Indicate that the matrix coefficients are ready to be set */
      HYPRE_StructMatrixInitialize(A);

      /* Set the stencil values in the interior. Here we set the values
         at every node. We will modify the boundary nodes later. */
      {
         int stencil_indices[5] = {0, 1, 2, 3, 4}; /* labels correspond
                                                      to the offsets */
         double *values;

         values = (double*) calloc(5 * (n * n), sizeof(double));

         /* The order is left-to-right, bottom-to-top */
         for (k = 0, j = 0; j < n; j++)
            for (i = 0; i < n; i++, k += 5)
            {
               values[k + 1] = - Eval(K, i - 0.5, j) ;

               values[k + 2] = - Eval(K, i + 0.5, j);

               values[k + 3] = - Eval(K, i, j - 0.5);

               values[k + 4] = - Eval(K, i, j + 0.5);

               values[k] =
                           + Eval(K, i - 0.5, j) + Eval(K, i + 0.5, j)
                           + Eval(K, i, j - 0.5) + Eval(K, i, j + 0.5);
            }

         HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 5,
                                        stencil_indices, values);

         free(values);
      }

   /* 5. Set the boundary conditions, while eliminating the coefficients
         reaching ouside of the domain boundary. We must modify the matrix
         stencil and the corresponding rhs entries. */
   {
      int bc_ilower[2];
      int bc_iupper[2];

      int stencil_indices[5] = {0, 1, 2, 3, 4};
      double *values, *bvalues;

      int nentries;
      nentries = 5;


      values  = (double*) calloc(nentries * n, sizeof(double));
      bvalues = (double*) calloc(n, sizeof(double));

      /* The stencil at the boundary nodes is 1-0-0-0-0. Because
         we have I x_b = u_0; */
      for (i = 0; i < nentries * n; i += nentries)
      {
         values[i] = 1.0;
         for (j = 1; j < nentries; j++)
         {
            values[i + j] = 0.0;
         }
      }

      /* Processors at y = 0 */
      if (pj == 0)
      {
         bc_ilower[0] = pi * n;
         bc_ilower[1] = pj * n;

         bc_iupper[0] = bc_ilower[0] + n - 1;
         bc_iupper[1] = bc_ilower[1];

         /* Modify the matrix */
         HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, nentries,
                                        stencil_indices, values);

         /* Put the boundary conditions in b */
         for (i = 0; i < n; i++)
         {
            bvalues[i] = bcEval(U0, i, 0);
         }

         HYPRE_StructVectorSetBoxValues(b, bc_ilower, bc_iupper, bvalues);
      }

      /* Processors at y = 1 */
      if (pj == N - 1)
      {
         bc_ilower[0] = pi * n;
         bc_ilower[1] = pj * n + n - 1;

         bc_iupper[0] = bc_ilower[0] + n - 1;
         bc_iupper[1] = bc_ilower[1];

         /* Modify the matrix */
         HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, nentries,
                                        stencil_indices, values);

         /* Put the boundary conditions in b */
         for (i = 0; i < n; i++)
         {
            bvalues[i] = bcEval(U0, i, 0);
         }


         HYPRE_StructVectorSetBoxValues(b, bc_ilower, bc_iupper, bvalues);
      }

      /* Processors at x = 0 */
      if (pi == 0)
      {
         bc_ilower[0] = pi * n;
         bc_ilower[1] = pj * n;

         bc_iupper[0] = bc_ilower[0];
         bc_iupper[1] = bc_ilower[1] + n - 1;

         /* Modify the matrix */
         HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, nentries,
                                        stencil_indices, values);

         /* Put the boundary conditions in b */
         for (j = 0; j < n; j++)
         {
            bvalues[j] = bcEval(U0, 0, j);
         }

         HYPRE_StructVectorSetBoxValues(b, bc_ilower, bc_iupper, bvalues);
      }

      /* Processors at x = 1 */
      if (pi == N - 1)
      {
         bc_ilower[0] = pi * n + n - 1;
         bc_ilower[1] = pj * n;

         bc_iupper[0] = bc_ilower[0];
         bc_iupper[1] = bc_ilower[1] + n - 1;

         /* Modify the matrix */
         HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, nentries,
                                        stencil_indices, values);

         /* Put the boundary conditions in b */
         for (j = 0; j < n; j++)
         {
            bvalues[j] = bcEval(U0, 0, j);
         }

         HYPRE_StructVectorSetBoxValues(b, bc_ilower, bc_iupper, bvalues);
      }

      /* Recall that the system we are solving is:
         [A_ii 0; 0 I] [x_i ; x_b] = [b_i - A_ib u_0; u_0].
         This requires removing the connections between the interior
         and boundary nodes that we have set up when we set the
         5pt stencil at each node. We adjust for removing
         these connections by appropriately modifying the rhs.
         For the symm ordering scheme, just do the top and right
         boundary */

      /* Processors at y = 0, neighbors of boundary nodes */
      if (pj == 0)
      {
         bc_ilower[0] = pi * n;
         bc_ilower[1] = pj * n + 1;

         bc_iupper[0] = bc_ilower[0] + n - 1;
         bc_iupper[1] = bc_ilower[1];

         stencil_indices[0] = 3;

         /* Modify the matrix */
         for (i = 0; i < n; i++)
         {
            bvalues[i] = 0.0;
         }

         HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, 1,
                                           stencil_indices, bvalues);

         /* Eliminate the boundary conditions in b */
         for (i = 0; i < n; i++)
         {
            bvalues[i] = bcEval(U0, i, -1) * bcEval(K, i, -0.5) ;
         }

         if (pi == 0)
         {
            bvalues[0] = 0.0;
         }

         if (pi == N - 1)
         {
            bvalues[n - 1] = 0.0;
         }

         /* Note the use of AddToBoxValues (because we have already set values
            at these nodes) */
         HYPRE_StructVectorAddToBoxValues(b, bc_ilower, bc_iupper, bvalues);
      }

      /* Processors at x = 0, neighbors of boundary nodes */
      if (pi == 0)
      {
         bc_ilower[0] = pi * n + 1;
         bc_ilower[1] = pj * n;

         bc_iupper[0] = bc_ilower[0];
         bc_iupper[1] = bc_ilower[1] + n - 1;

         stencil_indices[0] = 1;

         /* Modify the matrix */
         for (j = 0; j < n; j++)
         {
            bvalues[j] = 0.0;
         }

            HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, 1,
                                           stencil_indices, bvalues);

         /* Eliminate the boundary conditions in b */
         for (j = 0; j < n; j++)
         {
            bvalues[j] = bcEval(U0, -1, j) * (bcEval(K, -0.5, j) );
         }

         if (pj == 0)
         {
            bvalues[0] = 0.0;
         }

         if (pj == N - 1)
         {
            bvalues[n - 1] = 0.0;
         }

         HYPRE_StructVectorAddToBoxValues(b, bc_ilower, bc_iupper, bvalues);
      }

      /* Processors at y = 1, neighbors of boundary nodes */
      if (pj == N - 1)
      {
         bc_ilower[0] = pi * n;
         bc_ilower[1] = pj * n + (n - 1) - 1;

         bc_iupper[0] = bc_ilower[0] + n - 1;
         bc_iupper[1] = bc_ilower[1];

         stencil_indices[0] = 4;

         /* Modify the matrix */
         for (i = 0; i < n; i++)
         {
            bvalues[i] = 0.0;
         }

         HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, 1,
                                        stencil_indices, bvalues);

         /* Eliminate the boundary conditions in b */
         for (i = 0; i < n; i++)
         {
            bvalues[i] = bcEval(U0, i, 1) * (bcEval(K, i, 0.5) );
         }

         if (pi == 0)
         {
            bvalues[0] = 0.0;
         }

         if (pi == N - 1)
         {
            bvalues[n - 1] = 0.0;
         }

         HYPRE_StructVectorAddToBoxValues(b, bc_ilower, bc_iupper, bvalues);
      }

      /* Processors at x = 1, neighbors of boundary nodes */
      if (pi == N - 1)
      {
         bc_ilower[0] = pi * n + (n - 1) - 1;
         bc_ilower[1] = pj * n;

         bc_iupper[0] = bc_ilower[0];
         bc_iupper[1] = bc_ilower[1] + n - 1;

         stencil_indices[0] = 2;

         /* Modify the matrix */
         for (j = 0; j < n; j++)
         {
            bvalues[j] = 0.0;
         }

         HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, 1,
                                        stencil_indices, bvalues);

         /* Eliminate the boundary conditions in b */
         for (j = 0; j < n; j++)
         {
            bvalues[j] = bcEval(U0, 1, j) * (bcEval(K, 0.5, j));
         }

         if (pj == 0)
         {
            bvalues[0] = 0.0;
         }

         if (pj == N - 1)
         {
            bvalues[n - 1] = 0.0;
         }

         HYPRE_StructVectorAddToBoxValues(b, bc_ilower, bc_iupper, bvalues);
      }

      free(values);
      free(bvalues);
   }
   }

   /* Finalize the vector and matrix assembly */
   HYPRE_StructMatrixAssemble(A);
   HYPRE_StructVectorAssemble(b);
   HYPRE_StructVectorAssemble(x);

   /* 6. Set up and use a solver */


   if (solver_id)
   {
      /* Start timing */
      mytime -= MPI_Wtime();

      /* Options and setup */
      HYPRE_StructSMGCreate(MPI_COMM_WORLD, &solver);
      HYPRE_StructSMGSetMemoryUse(solver, 0);
      HYPRE_StructSMGSetMaxIter(solver, 50);
      HYPRE_StructSMGSetTol(solver, 1.0e-12);
      HYPRE_StructSMGSetRelChange(solver, 0);
      HYPRE_StructSMGSetNumPreRelax(solver, n_pre);
      HYPRE_StructSMGSetNumPostRelax(solver, n_post);
      HYPRE_StructSMGSetPrintLevel(solver, 1);
      HYPRE_StructSMGSetLogging(solver, 1);
      HYPRE_StructSMGSetup(solver, A, b, x);

      /* Finalize current timing */
      mytime += MPI_Wtime();
      MPI_Allreduce(&mytime, &walltime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      if (myid == 0)
      {
         printf("\nSMG Setup time = %f seconds\n\n", walltime);
      }

      /* Start timing again */
      mytime -= MPI_Wtime();

      /* Solve */
      HYPRE_StructSMGSolve(solver, A, b, x);


      /* Finalize current timing */
      mytime += MPI_Wtime();
      MPI_Allreduce(&mytime, &walltime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      if (myid == 0)
      {
         printf("\nSMG Solve time = %f seconds\n\n", walltime);
      }

      /* Get info and release memory */
      HYPRE_StructSMGGetNumIterations(solver, &num_iterations);
      HYPRE_StructSMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
      HYPRE_StructSMGDestroy(solver);
   }
   else{

      HYPRE_StructGMRESCreate(MPI_COMM_WORLD, &solver);

      /* Note that GMRES can be used with all the interfaces - not
         just the struct.  So here we demonstrate the
         more generic GMRES interface functions. Since we have chosen
         a struct solver then we must type cast to the more generic
         HYPRE_Solver when setting options with these generic functions.
         Note that one could declare the solver to be
         type HYPRE_Solver, and then the casting would not be necessary.*/

      HYPRE_GMRESSetMaxIter((HYPRE_Solver) solver, 500 );
      HYPRE_GMRESSetKDim((HYPRE_Solver) solver, 30);
      HYPRE_GMRESSetTol((HYPRE_Solver) solver, 1.0e-012 );
      HYPRE_GMRESSetPrintLevel((HYPRE_Solver) solver, 0 );
      HYPRE_GMRESSetLogging((HYPRE_Solver) solver, 0 );


         /* use symmetric SMG as preconditioner */
         HYPRE_StructSMGCreate(MPI_COMM_WORLD, &precond);
         HYPRE_StructSMGSetMemoryUse(precond, 0);
         HYPRE_StructSMGSetMaxIter(precond, 1);
         HYPRE_StructSMGSetTol(precond, 0.0);
         HYPRE_StructSMGSetZeroGuess(precond);
         HYPRE_StructSMGSetNumPreRelax(precond, n_pre);
         HYPRE_StructSMGSetNumPostRelax(precond, n_post);
         HYPRE_StructSMGSetPrintLevel(precond, 0);
         HYPRE_StructSMGSetLogging(precond, 0);
         HYPRE_StructGMRESSetPrecond(solver,
                                     HYPRE_StructSMGSolve,
                                     HYPRE_StructSMGSetup,
                                     precond);




      HYPRE_StructGMRESSetup(solver, A, b, x );

      mytime += MPI_Wtime();
      MPI_Allreduce(&mytime, &walltime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      if (myid == 0)
      {
         printf("\nGMRES Setup time = %f seconds\n\n", walltime);
      }

      mytime -= MPI_Wtime();

      /* GMRES Solve */
      HYPRE_StructGMRESSolve(solver, A, b, x);

      mytime += MPI_Wtime();
      MPI_Allreduce(&mytime, &walltime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      if (myid == 0)
      {
         printf("\nGMRES Solve time = %f seconds\n\n", walltime);
      }

      /* Get info and release memory */
      HYPRE_StructGMRESGetNumIterations(solver, &num_iterations);
      HYPRE_StructGMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
      HYPRE_StructGMRESDestroy(solver);
      HYPRE_StructSMGDestroy(precond);
   }

    write_array(x, ilower[0], iupper[0], "result.dat");
    write_array(b, ilower[0], iupper[0], "b.dat");
    write_array(alpha, ilower[0], iupper[0], "alpha.dat");

   if (myid == 0)
   {
      printf("\n");
      printf("Iterations = %d\n", num_iterations);
      printf("Final Relative Residual Norm = %e\n", final_res_norm);
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
