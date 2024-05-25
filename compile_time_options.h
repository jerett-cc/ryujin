//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#pragma once

/* Compile-time options: */

#define NUMBER 

/* #undef CHECK_BOUNDS */
#if defined(DEBUG) && !defined(CHECK_BOUNDS)
#define CHECK_BOUNDS
#endif

/* #undef ASYNC_MPI_EXCHANGE */
#define DEBUG_OUTPUT
/* #undef DENORMALS_ARE_ZERO */
/* #undef FORCE_DEAL_II_SPARSE_MATRIX */

/* External packages: */

/* #undef WITH_EOSPAC */
/* #undef WITH_LIKWID */
/* #undef WITH_OPENMP */
/* #undef WITH_VALGRIND */

/* Discretization: */

#define ORDER_FINITE_ELEMENT 
#define ORDER_MAPPING 
#define ORDER_QUADRATURE 
