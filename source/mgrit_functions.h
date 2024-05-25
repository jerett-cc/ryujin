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
template <typename Number, typename Description, int dim>
dealii::Tensor<1, dim> calculate_drag_and_lift(mgrit::MyApp<Number, Description, dim> *app,
                                               const mgrit::MyVector<Number, Description, dim> &u,
                                               const braid_Real t);

/// @brief This function ensures that the solution state @u produces physical
/// quantities. For example, if u represents the conservative states of a
/// system, we ensure that the pressure, density, and entropy remain
/// non-negative.
/// @tparam Description The description of the system, providing a pressure
/// function and an entropy function.
/// @tparam dim The spatial dimension.
/// @tparam Number Either a double or float.
/// @param u The solution state we want to modify to remain conservative.
/// @param level The level that describes the vector we want to enforce physicality.
/// @param app The app containing the level structures we need to do work on u.
//FIXME: reorder the template parameters to match the library
template <typename Description, int dim, typename Number>
void enforce_physicality_bounds(mgrit::MyVector<Number, Description, dim> &u,
                                const unsigned int level,
                                const mgrit::MyApp<Number, Description, dim> &app,
                                const Number t);

}// Namespace mgrit_functions