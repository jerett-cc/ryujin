##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2022 - 2023 by the ryujin authors
##

include_directories(
  ${CMAKE_BINARY_DIR}/source/
  )
# MGRIT is only for euler equations now.
set(TEST_LIBRARIES obj_common obj_euler obj_euler_dependent obj_mgrit obj_mgrit_dependent)

  deal_ii_pickup_tests()

