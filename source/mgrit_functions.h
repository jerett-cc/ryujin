#pragma once

#include <deal.II/base/tensor.h>
#include "my_app.h"

namespace mgrit_functions{

  /// @brief This function calculates the drag around a ryujin boundary marked
  /// with BoundaryID::object.
  /// @tparam Number The numeric storage, e.g. double or float.
  /// @tparam Description The desctiption of the hyperbolic system, see ryujin.
  /// @param app The MyApp object which knows about all the data structures at the
  /// level of MyVector.
  /// @param u The vector to calculate on.
  /// @param t The time we are calculating at.
  /// @param dim The spatial dimension of the problem.
  /// @return A dealii::tensor of the components of the net force acting on the
  /// part of the boundary.
  template <typename Number, typename Description>
  dealii::Tensor<1, 2>
  calculate_drag_and_lift(mgrit::MyApp<Number, Description, 2> *app,
			  const mgrit::MyVector<Number, Description, 2> &u,
			  const braid_Real t);

  /// @brief This function is the dim=1 version of the above. It is not implemented. FIXME: change this brief if ever implemented.
  /// @tparam Number The numeric storage, e.g. double or float.
  /// @tparam Description The desctiption of the hyperbolic system, see ryujin.
  /// @param app The calling MyApp function which knows about problem dimension and data handlers.
  /// @param u The solution state to perform drag calculations on.
  /// @param t The simulation time at which we are performing the calculation.
  /// @return A tensor of the net force. Here, is will just return the 0 tensor.
  template <typename Number, typename Description>
  dealii::Tensor<1, 1>
  calculate_drag_and_lift(mgrit::MyApp<Number, Description, 1> *app,
			  const mgrit::MyVector<Number, Description, 1> &u,
			  const braid_Real t);

  /// @brief This function is the dim=3 version of the above. It is not implemented. FIXME: change this brief if ever implemented.
  /// @tparam Number The numeric storage, e.g. double or float.
  /// @tparam Description The desctiption of the hyperbolic system, see ryujin.
  /// @param app The calling MyApp function which knows about problem dimension and data handlers.
  /// @param u The solution state to perform drag calculations on.
  /// @param t The simulation time at which we are performing the calculation.
  /// @return A tensor of the net force. Here, is will just return the 0 tensor.
  template <typename Number, typename Description>
  dealii::Tensor<1, 3>
  calculate_drag_and_lift(mgrit::MyApp<Number, Description, 3> *app,
			  const mgrit::MyVector<Number, Description, 3> &u,
			  const braid_Real t);

  /// @brief This function ensures that the solution state @u produces physical
  /// quantities. For example, if u represents the conservative states of a
  /// system, we ensure that the pressure, density, and entropy remain
  /// non-negative. This function makes sure that the pressure is above a min-
  /// imum and that the density is above a mimimum.
  /// MODIFICATION: we also make sure that the pressure quantity is not larger
  ///  than the average in the region.
  /// @tparam Description The description of the system, providing a pressure
  /// function and an entropy function.
  /// @tparam dim The spatial dimension.
  /// @tparam Number Either a double or float.
  /// @param u The solution state we want to modify to remain conservative.
  /// @param level The level that describes the vector we want to enforce physicality.
  /// @param app The app containing the level structures we need to do work on u.
  //FIXME/TODO: reorder the template parameters to match the library
  template <typename Description, int dim, typename Number>
  void enforce_physicality_bounds(mgrit::MyVector<Number, Description, dim> &u,
				  const unsigned int level,
				  const mgrit::MyApp<Number, Description, dim> &app,
				  const Number t);
  
  /// @brief This function measures the conserved quantities and entropy in the system,
  ///        only works for the most accurate MGRIT level.
  /// @tparam Description A description of the equation of state.
  /// @tparam dim         The spatial dimension.
  /// @tparam Number      The numerical expression of a real number.
  /// @param u            A reference to a solution state on our mesh.
  /// @param app          A reference to the my_app structure which contains
  ///                     mesh specific data.
  /// @param level        Required to specify the level in this function.
  /// @param time         The time at which we calculate the entropy.
  template <typename Description, int dim, typename Number>
  void conserved_and_entropy_in_system(const mgrit::MyVector<Number, Description, dim> &u,
				       const mgrit::MyApp<Number, Description, dim> &app,
				       const unsigned int level,
				       const Number time);
  
}// Namespace mgrit_functions
