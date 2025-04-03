#!/bin/bash

# the .prm is $1
# the nWorld is $2
# the nX is $3
# the number of threads is $4
# the rest are the refinement levels.
## warning, these need to be sorted into ascending order eg. 1 3 6 levels of refinement
## since the logic of this file assumes that $5 is the coarsest refinement.

#TODO: make some kind of assertion here?

prm="$1"
nworld="$2"
nx="$3"
nthreads="$4"
s_refinements="${@:5:($#)}"
refinements=(${s_refinements})

echo "${prm}"
echo "${nworld}"
echo "${nx}"
echo "${nthreads}"
echo ${s_refinements}
echo ${refinements[0]}

DEAL_II_NUM_THREADS="${nthreads}" mpirun -n "${nworld}" startup "${prm}" "${refinements[0]}"

# once the setupfiles are written, we start the regular program.

time DEAL_II_NUM_THREADS="${nthreads}" mpirun -n "${nworld}" cpp_myapp_testing "${prm}" "${nx}" ${s_refinements} | tee CURRENT.log
