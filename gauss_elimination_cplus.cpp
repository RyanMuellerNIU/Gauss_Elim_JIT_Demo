/*
 *  Gaussian Elimination Solver, based on a version distributed by Sandhya
 *  Dwarkadas, University of Rochester
 */

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <assert.h>
#include <getopt.h>
/*
 *  Helper macro: swap two doubles
 */
#define DSWAP(a, b) { double tmp; tmp = a; a = b; b = tmp; }

/*
 *  Helper macro: swap two pointers
 */
#define PSWAP(a, b) { double *tmp; tmp = a; a = b; b = tmp; }

/*
 *  Helper macro: absolute value
 */
#define ABS(a)      (((a) > 0) ? (a) : -(a))

/*
 *  The 2-d matrix that holds the coefficients for gaussian elimination.  Since
 *  the size is an input parameter, we implement the matrix as an array of
 *  arrays.  This also makes swapping rows easy... we just swap pointers.
 */
double **matrix;

/*
 *  The "left hand side" vector of values
 */
double *B;

/*
 *  To verify our work, we'll back-substitute into M after it is in triangular
 *  form.  This vector gives us the solution.  As with 'V', we declare it early
 *  and allocate it early to avoid out-of-memory errors later.
 */
double *C;

/*
 * Allocate the arrays
 */
void allocate_memory(int size)
{
    /* hold [size] pointers*/
    matrix = (double**)malloc(size * sizeof(double*));
    assert(matrix != NULL);
    /* get a [size x size] array of doubles */
    double *tmp = (double*)malloc(size*size*sizeof(double));
    assert(tmp != NULL);
    /* allocate parts of the array to the rows of the matrix */
    for (int i = 0; i < size; i++) {
        matrix[i] = tmp;
        tmp = tmp + size;
    }
    /* allocate the LHS vector */
    B = (double*)malloc(size * sizeof(double));
    assert(B != NULL);
    /* allocate the solution vector */
    C = (double*)malloc(size * sizeof(double));
    assert(C != NULL);
}

/*
 * Initialize the matrix with some values that we know yield a solution that is
 * easy to verify. A correct solution should yield -0.5 and 0.5 for the first
 * and last C values, and 0 for the rest.
 */
void initMatrix(int nsize)
{
    for (int i = 0; i < nsize; i++) {
        for (int j = 0; j < nsize; j++) {
            matrix[i][j] = ((j < i )? 2*(j+1) : 2*(i+1));
        }
        B[i] = (double)i;
    }
}

/*
 * Get the pivot row.  For best numerical stability, always choose the row with
 * the largest column value.  If the column only has zeros, then we have a
 * singluar array and must fail.
 */
void getPivot(int nsize, int currow)
{
    /* irow is the row we're going to use */
    int irow = currow;
    /* find the biggest value in this column */
    double big = ABS(matrix[currow][currow]);
    for (int i = currow + 1; i < nsize; i++) {
        double tmp = ABS(matrix[i][currow]);
        if (tmp > big) {
            big = tmp;
            irow = i;
        }
    }
    /* make sure we can progress */
    if (big == 0.0) {
        printf("The matrix is singular\n");
        exit(-1);
    }
    /* Do we have to swap? */
    if (irow != currow) {
        PSWAP(matrix[irow], matrix[currow]);
        DSWAP(B[irow], B[currow]);
    }
    /* Now normalize the current row so pivot point is 1.0 */
    double pivotVal = matrix[currow][currow];
    if (pivotVal != 1.0) {
        matrix[currow][currow] = 1.0;
        for (int i = currow + 1; i < nsize; i++) {
            matrix[currow][i] /= pivotVal;
        }
        B[currow] /= pivotVal;
    }
}

/*
 * For all the rows, get the pivot and eliminate all rows and columns for that
 * particular pivot row.
 */
void computeGauss(int nsize)
{
    for (int i = 0; i < nsize; i++) {
        getPivot(nsize, i);
        for (int j = i + 1; j < nsize; j++) {
            double pivotVal = matrix[j][i];
            matrix[j][i] = 0.0;
            for (int k = i + 1; k < nsize; k++) {
                matrix[j][k] -= pivotVal * matrix[i][k];
            }
            B[j] -= pivotVal * B[i];
        }
    }
}

/*
 * Do back-substitution to get a solution vector
 */
void solveGauss(int nsize)
{
    C[nsize-1] = B[nsize-1];
    for (int row = nsize - 2; row >= 0; row--) {
        C[row] = B[row];
        for (int col = nsize - 1; col > row; col--) {
            C[row] -= matrix[row][col] * C[col];
        }
    }
}

/*
 *  Main routine: parse command args, create array, compute solution
 */
int main(int argc, char *argv[])
{
    /* start and end time */
    struct timeval t0;
    struct timeval t1;
    /* two temps and the size of the array */
    int i, s, nsize = 1024;
    /* get the size from the command-line */
    while ((i = getopt(argc,argv,"s:")) != -1) {
        switch(i) {
          case 's':
            s = atoi(optarg);
            if (s > 0)
                nsize = s;
            else
                printf("  -s is negative... using %d\n", nsize);
            break;
          default:
            assert(0);
            break;
        }
    }
    /* allocate memory */
    allocate_memory(nsize);
    /* get start time, initialize, compute, solve, get end time */
    gettimeofday(&t0, 0);
    initMatrix(nsize);
    computeGauss(nsize);
    solveGauss(nsize);
    gettimeofday(&t1, 0);
    /* print compute time */
    unsigned long usecs = t1.tv_usec - t0.tv_usec;
    usecs  += (1000000 * (t1.tv_sec - t0.tv_sec));
    printf("Size: %d rows\n", nsize);
    printf("Time: %f seconds\n", ((double)usecs)/1000000.0);
    /* verify that the code was correct */
    for (int n = 0; n < nsize; n++) {
        if (n == 0)
            assert (C[n] == -0.5);
        else if (n == nsize-1)
            assert (C[n] == 0.5);
        else
            assert (C[n] == 0);
    }
    printf("Correct solution found.\n");
}
