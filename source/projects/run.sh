#!/bin/bash

# the .prm is $1
# the nWorld is $2
# the nX is $3
# the rest are the refinement levels.
## warning, these need to be sorted into ascending order eg. 1 3 6 levels of refinement
## since the logic of this file assumes that $4 is the coarsest refinement.

#TODO: make some kind of assertion here?

prm="$1"
nworld="$2"
nx="$3"
s_refinements="${@:4:($#)}"
refinements=(${s_refinements})

echo "${prm}"
echo "${nworld}"
echo "${nx}"
echo ${s_refinements}
echo ${refinements[0]}

mpirun -n "${nworld}" startup "${prm}" "${refinements[0]}"

# once the setupfiles are written, we start the regular program.

mpirun -n "${nworld}" cpp_myapp_testing "${prm}" "${nx}" ${s_refinements}
