/*
 * level_structures.cc
 *
 *  Created on: Oct 5, 2023
 *      Author: jerett
 */

#include "level_structures.template.h"
#include <instantiate.h>

namespace ryujin{
  namespace mgrit{
    /* instantiations */
    template class LevelStructures<Description, 1, NUMBER>;
    template class LevelStructures<Description, 2, NUMBER>;
    template class LevelStructures<Description, 3, NUMBER>;

  } // namespace mgrit
}//namespace ryujin
