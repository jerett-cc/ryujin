//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#include "time_loop_mgrit.h"
#include "time_loop_mgrit.template.h"
#include <instantiate.h>

namespace ryujin
{
  namespace mgrit{
    /* instantiations */
    template class TimeLoopMgrit<Description, 1, NUMBER>;
    template class TimeLoopMgrit<Description, 2, NUMBER>;
    template class TimeLoopMgrit<Description, 3, NUMBER>;
  }//namespace mgrit
} // namespace ryujin
