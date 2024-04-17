#pragma once

#include <deal.II/base/tensor.h>
#include "my_app.h"

namespace mgrit_functions{

/// @brief This function calculates the drag around a ryujin boundary marked
/// with BoundaryID::object.
/// @param app The MyApp object which knows about all the data structures at the
/// level of MyVector.
/// @param u The vector to calculate on.
/// @param t The time we are calculating at.
/// @param dim The spatial dimension of the problem.
/// @return A dealii::tensor of the components of the net force acting on the
/// part of the boundary.
template <int dim>
dealii::Tensor<1, dim> calculate_drag_and_lift(mgrit::MyApp *app,
                                               const mgrit::MyVector &u,
                                               const braid_Real t);

}// Namespace mgrit_functions