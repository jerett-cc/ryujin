#include "mgrit_functions.h"
#include "level_structures.h"

namespace mgrit_functions{

  template <int dim>
  dealii::Tensor<1, dim>
  calculate_drag_and_lift(mgrit::MyApp *app, const mgrit::MyVector &u, const braid_Real t)
  {
    if (dim == 2) {
      using scalar_type = dealii::LinearAlgebra::distributed::Vector<NUMBER>;

      UNUSED(t);

      const auto offline_data = app->levels[app->finest_level]->offline_data;
      const auto mpi_communicator = app->comm_x;
      const auto hyperbolic_system_view =
          app->levels[app->finest_level]
              ->hyperbolic_system->template view<2 /*dim*/, NUMBER>();
      // first, set up the finite element, the data, and the facevalues
      // const dealii::FiniteElement<2,2> fe =
      // app->levels[app->finest_level]->offline_data->discretization().finite_element();
      // // the finite element const int degree =
      // app->levels[app->finest_level]->offline_data->discretization().finite_element().degree;
      // dealii::QGauss<1> face_quadrature_formula =
      // app->levels[app->finest_level]->offline_data->discretization().quadrature_1d();
      const int n_q_points = app->levels[app->finest_level]
                                 ->offline_data->discretization()
                                 .quadrature_1d()
                                 .size();

      std::vector<double> pressure_values(n_q_points);

      scalar_type density, pressure;
      std::vector<scalar_type> momentum(dim);

      // initialize partitions
      density.reinit(offline_data->scalar_partitioner(), mpi_communicator);
      pressure.reinit(offline_data->scalar_partitioner(), mpi_communicator);
      for (unsigned int c = 0; c < dim; c++)
        momentum.at(c).reinit(offline_data->scalar_partitioner(),
                              mpi_communicator);

      dealii::Tensor<1, 2 /*dim*/> normal_vector;
      dealii::SymmetricTensor<2, 2 /*dim*/> fluid_stress;
      dealii::SymmetricTensor<2, 2 /*dim*/> fluid_pressure;
      dealii::Tensor<1, 2 /*dim*/> forces;

      dealii::FEFaceValues<2 /*dim*/> fe_face_values(
          app->levels[app->finest_level]
              ->offline_data->discretization()
              .finite_element() /*FE_Q<dim>*/,
          app->levels[app->finest_level]
              ->offline_data->discretization()
              .quadrature_1d() /*QGauss<dim-1*/,
          dealii::update_values | dealii::update_quadrature_points |
              dealii::update_gradients | dealii::update_JxW_values |
              dealii::update_normal_vectors); // the face values

      // Create vectors that store the locally owned parts on every process
      u.U.extract_component(density, 0);        // extract density
      u.U.extract_component(pressure, dim + 1); // extract density

      // extract momentum, and convert to velocity
      for (unsigned int c = 0; c < dim; c++) {
        int comp =
            c +
            1; // momentum is stored in positions [1,...,dim], so add one to c
        u.U.extract_component(momentum.at(c), comp);
      }

      // extract energy
      u.U.extract_component(pressure, dim + 1);

      // convert E to pressure
      for (unsigned int k = 0; k < offline_data->n_locally_owned(); k++) {

        // calculate momentum norm squared
        const double &E = pressure.local_element(k);
        const double &rho = density.local_element(k);
        double m_square = 0;
        for (unsigned int d = 0; d < dim; d++)
          m_square += std::pow(momentum.at(d).local_element(k), 2);

        // pressure = (gamma-1)*internal_energy
        pressure.local_element(k) =
            (hyperbolic_system_view.gamma() - 1.0) * (E - 0.5 * m_square / rho);
      }

      density.update_ghost_values();
      pressure.update_ghost_values();
      for (auto mom : momentum)
        mom.update_ghost_values();

      double drag = 0.;
      double lift = 0.;

      for (const auto &cell :
           offline_data->dof_handler().active_cell_iterators()) {
        if (cell->is_locally_owned()) {
          for (unsigned int face = 0; face < cell->n_faces(); ++face) {
            if (cell->face(face)->at_boundary() &&
                cell->face(face)->boundary_id() == ryujin::Boundary::object) {
              // if on circle, we do the calculation
              // first, find if the face center is on the circle

              fe_face_values.reinit(cell, face);

              // pressure values
              fe_face_values.get_function_values(pressure, pressure_values);

              // now, loop over quadrature points calculating their contribution
              // to the forces acting on the face
              for (int q = 0; q < n_q_points; ++q) {
                normal_vector = -fe_face_values.normal_vector(q);

                // form the contributions from pressure
                for (unsigned int d = 0; d < dim; ++d)
                  fluid_pressure[d][d] = pressure_values[q];

                fluid_stress = -fluid_pressure; // for the euler equations, the
                                                // only contribution to stresses
                                                // comes from pressure
                forces = fluid_stress * normal_vector * fe_face_values.JxW(q);
                // the drag is in the x direction, the lift is in the y
                // direction but FIXME: does this hold true in higher dimension?
                // look below for this
                drag += forces[0];
                lift += forces[1];
              } // loop over q points
            }   // if cell face is at boundary && on the object
          }     // face loop
        }       // locally_owned cells
      }         // cell loop

      // now, sum the values across all processes.
      lift = dealii::Utilities::MPI::sum(lift, mpi_communicator);
      drag = dealii::Utilities::MPI::sum(drag, mpi_communicator);

      forces[0] = drag;
      forces[1] = lift;
      return forces;

    } /* dim is 2, this code only works for dim=2, unfortunately.*/ else {
      Assert(false,
             dealii::ExcMessage(
                 "You are trying to call calculate_drag_and_lift,"
                 " which only works for dim=2. you called with dim = " +
                 std::to_string(dim)));
      dealii::Tensor<1, dim> forces;
      return forces;
    }
  }

  template <typename Description, int dim, typename Number>
  void enforce_physicality_bounds(mgrit::MyVector &u,
                                  const unsigned int level,
                                  const mgrit::MyApp &app,
                                  const Number t)
  {
    // Create Hyperbolic System View, where we can compute functions like pressure.
    const auto view = app.levels[level]->hyperbolic_system->template view<dim,Number>();

    // For each node, translate the conserved quantities into the primitive quantities.
    // If we dip below the minimums, set the primitive to their minimum.
    for(unsigned int node=0; node < app.n_locally_owned_at_level(level); node++)
    {
      const auto state = u.U.get_tensor(node); // The current conserved state at this node.
      auto primitive_state = view.to_primitive_state(state); // The primitive state at this node.

      // Modify the primitive state to be physical.
      // We only need to ensure that the density and pressure are positive. Velocities can be negative.
      // todo: is this true? do we need to make sure that the velocities are not too large if we decrease pressure? is the number here good enough (1e-8)?
      primitive_state[0] = std::max(primitive_state[0], Number(1e-8));
      primitive_state[dim + 1] = std::max(primitive_state[dim + 1], Number(1e-8));

      // Translate new state to conserved, then place in to spot.
      u.U.write_tensor(view.from_primitive_state(primitive_state), node);
    }

    u.U.update_ghost_values();

    // Make sure boundary conditions are satisfied on these states.
    //FIXME: this function also calls update_ghost_values(), do I need the one above?
    app.levels[level]->hyperbolic_module->apply_boundary_conditions(u.U, t);
  }


} // Namespace mgrit_functions