#include "mgrit_functions.h"
#include "level_structures.h"

// This preprocessor macro is used on function arguments
// that are not used in the function. It is used to
// suppress compiler warnings.
#define UNUSED(x) (void)(x)

namespace mgrit_functions{

  // Specialize dim=2
  template <typename Number, typename Description>
  dealii::Tensor<1, 2>
  calculate_drag_and_lift(mgrit::MyApp<Number, Description, 2> *app,
                          const mgrit::MyVector<Number, Description, 2> &u,
                          const braid_Real t)
  {
    using scalar_type = dealii::LinearAlgebra::distributed::Vector<Number>;
    // In this function, dim = 2 is a valid assumption
    constexpr int dim = 2;
    // We do not care what t is in this case.
    UNUSED(t);

      const auto offline_data = app->levels[app->finest_level]->offline_data;
      const auto mpi_communicator = app->comm_x;
      const auto hyperbolic_system_view =
          app->levels[app->finest_level]
	  ->hyperbolic_system->get().template view<2, Number>();
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
    for (int c = 0; c < dim; c++)
      momentum.at(c).reinit(offline_data->scalar_partitioner(),
                            mpi_communicator);

    dealii::Tensor<1, dim> normal_vector;
    dealii::SymmetricTensor<2, dim> fluid_stress;
    dealii::SymmetricTensor<2, dim> fluid_pressure;
    dealii::Tensor<1, dim> forces;

    dealii::FEFaceValues<dim> fe_face_values(
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
    std::get<0>(u.U).extract_component(density, 0);        // extract density
    std::get<0>(u.U).extract_component(pressure, dim + 1); // extract density

    // extract momentum, and convert to velocity
    for (int c = 0; c < dim; c++) {
      int comp =
          c + 1; // momentum is stored in positions [1,...,dim], so add one to c
      std::get<0>(u.U).extract_component(momentum.at(c), comp);
    }

    // extract energy
    std::get<0>(u.U).extract_component(pressure, dim + 1);

    // convert E to pressure
    for (unsigned int k = 0; k < offline_data->n_locally_owned(); k++) {

      // calculate momentum norm squared
      const double &E = pressure.local_element(k);
      const double &rho = density.local_element(k);
      double m_square = 0;
      for (int d = 0; d < dim; d++)
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
              for (int d = 0; d < dim; ++d)
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
          } // if cell face is at boundary && on the object
        } // face loop
      } // locally_owned cells
    } // cell loop

    // now, sum the values across all processes.
    lift = dealii::Utilities::MPI::sum(lift, mpi_communicator);
    drag = dealii::Utilities::MPI::sum(drag, mpi_communicator);

    forces[0] = drag;
    forces[1] = lift;
    return forces;
  }

   // Specialize dim=1
  template <typename Number, typename Description>
  dealii::Tensor<1, 1>
  calculate_drag_and_lift(mgrit::MyApp<Number, Description, 1> *app,
                          const mgrit::MyVector<Number, Description, 1> &u,
                          const braid_Real t)
  {
    UNUSED(app);
    UNUSED(u);
    UNUSED(t);
    dealii::ExcNotImplemented();
    return dealii::Tensor<1, 1>();
  }

   // Specialize dim=3
  template <typename Number, typename Description>
  dealii::Tensor<1, 3>
  calculate_drag_and_lift(mgrit::MyApp<Number, Description, 3> *app,
                          const mgrit::MyVector<Number, Description, 3> &u,
                          const braid_Real t)
  {
    UNUSED(app);
    UNUSED(u);
    UNUSED(t);
    dealii::ExcNotImplemented();
    return dealii::Tensor<1, 3>();
  }

  template <typename Description, int dim, typename Number>
  void enforce_physicality_bounds(mgrit::MyVector<Number, Description, dim> &u,
                                  const unsigned int level,
                                  const mgrit::MyApp<Number, Description, dim> &app,
                                  const Number t)
  {
    // Create Hyperbolic System View, where we can compute functions like pressure.
    const auto view = app.levels[level]->hyperbolic_system->get().template view<dim,Number>();

    // Compute the local average in pressure and use this as a limit on pressure.
    Number average_pressure = 0.0;
    for(unsigned int node=0; node < app.n_locally_owned_at_level(level); node++)
    {
      const auto state = std::get<0>(u.U).get_tensor(node);
      auto primitive_state_pressure = view.to_primitive_state(state)[dim+1];

      average_pressure += primitive_state_pressure;
    }

    Assert((app.n_locally_owned_at_level(level) > 0), dealii::ExcInternalError());
    average_pressure /= app.n_locally_owned_at_level(level); 

    //TODO: do we need to take the global average?
    
    // For each node, translate the conserved quantities into the primitive quantities.
    // If we dip below the minimums, set the primitive to their minimum, and for
    // pressure we disallow very large pressures.
    for(unsigned int node=0; node < app.n_locally_owned_at_level(level); node++)
    {
      const auto state = std::get<0>(u.U).get_tensor(node); // The current conserved state at this node.
      auto primitive_state = view.to_primitive_state(state); // The primitive state at this node.

      // Modify the primitive state to be physical.
      // We only need to ensure that the density and pressure are positive. Velocities can be negative.
      // todo: is this true? do we need to make sure that the velocities are not too large if we decrease pressure? is the number here good enough (1e-8)?
      // todo: we need to make this actually a template, and use the physicality from the description...
      primitive_state[0] = std::max(primitive_state[0], Number(1e-8));
      primitive_state[dim + 1] = std::max(primitive_state[dim + 1], Number(1e-8));
      // disallow very large pressures.
      if(std::abs(primitive_state[dim+1]/average_pressure) > 1e1)
	primitive_state[dim+1] = average_pressure;

      // Translate new state to conserved, then place in to spot.
      std::get<0>(u.U).write_tensor(view.from_primitive_state(primitive_state), node);
    }

    std::get<0>(u.U).update_ghost_values();

    // Make sure boundary conditions are satisfied on these states.
    //FIXME: this function also calls update_ghost_values(), do I need the one above?
    app.levels[level]->hyperbolic_module->prepare_state_vector(u.U, t);
  }

  template <typename Description, int dim, typename Number>
  void conserved_and_entropy_in_system(const mgrit::MyVector<Number, Description, dim> &u,
				       const mgrit::MyApp<Number, Description, dim> *app,
				       const braid_Int level,
				       const Number time)
  {
    [[maybe_unused]] int n_dofs       = app->n_locally_owned_at_level(level);
    [[maybe_unused]] int n_components = app->problem_dimension;
    [[maybe_unused]] int n_vector_dof = std::get<0>(u.U).locally_owned_size()/n_components;

    Assert((level == app->finest_level),
	   dealii::ExcMessage("Can only calculate entropy on finest level."));
    Assert((n_dofs == n_vector_dof),
	   dealii::ExcMessage("Total entropy can only be calculated when dofs match on mesh and u."
			      "Here, level="+ std::to_string(level)+ " which has "+
		      std::to_string(n_vector_dof)+ " when the expected "+
		      "number of dofs on the finest level is "+
		      std::to_string(n_dofs)));
    // Calculate the entropy in the system
    const auto hyperbolic_system_view =
      app->levels[level]->hyperbolic_system->get().template view<dim,Number>();

    Number total_entropy = 0;
    Number mass          = 0;
    Number momentum_sqr  = 0;
    Number E             = 0;

    const auto comm_x = app->comm_x;
    const auto od     = app->levels[app->finest_level]->offline_data;
    const auto &fe    = od->discretization().finite_element();
    const auto &quad  = od->discretization().quadrature();

    dealii::FEValues<dim> fe_vals(fe,
				  quad,
				  dealii::update_values | dealii::update_JxW_values);
    
    for (const auto &cell: od->dof_handler().active_cell_iterators())
      {
	fe_vals.reinit(cell);
	if(cell->is_locally_owned())
	  {
	    for (const unsigned int q_idx: fe_vals.quadrature_point_indices())
	      {
		const auto state = std::get<0>(u.U).get_tensor(q_idx);
		const Number point_entropy = hyperbolic_system_view.specific_entropy(state);
		const Number point_rho = state[0];
		Number point_mom_sqr = 0;
		for (int d=0; d<dim; d++)
		  point_mom_sqr += state[1+d]*state[1+d];
		const Number point_E = state[dim+1];
		
		for (const unsigned int i: fe_vals.dof_indices())
		  {
		    total_entropy += (fe_vals.shape_value(i,q_idx) *
				      point_entropy *
				      fe_vals.JxW(q_idx));
		    mass          += (fe_vals.shape_value(i,q_idx) *
				      point_rho *
				      fe_vals.JxW(q_idx));
		    momentum_sqr  += (fe_vals.shape_value(i,q_idx) *
				      point_mom_sqr *
				      fe_vals.JxW(q_idx));
		    E             += (fe_vals.shape_value(i,q_idx) *
				      point_E *
				      fe_vals.JxW(q_idx));
		  }// dof contributions for each cell
	      } // quadrature points in cell
	  } // locally owned
      } //cells
   
    // communicate across space
    dealii::Utilities::MPI::sum(total_entropy, comm_x);
    dealii::Utilities::MPI::sum(mass, comm_x);
    dealii::Utilities::MPI::sum(momentum_sqr, comm_x);
    dealii::Utilities::MPI::sum(E, comm_x);
    if(dealii::Utilities::MPI::this_mpi_process(app->comm_x)==0){
      std::cout << "Total entropy at time t= " << time << " is " << total_entropy << std::endl;
      std::cout << "Total Mass at time t= " << time << " is " << mass << std::endl;
      std::cout << "Total Momentum Squared at time t= " << time << " is " << momentum_sqr << std::endl;
      std::cout << "Total E at time t= " << time << " is " << E << std::endl;
    }
  }

  template <typename Description, int dim, typename Number>
  bool does_E_exceed_threshold(mgrit::MyVector<Number, Description, dim> &u,
			       mgrit::MyApp<Number, Description, dim> &app,
			       [[maybe_unused]]const braid_Int level,
			       const Number time,
			       const braid_Int t_idx,
			       const braid_Int calling,
			       const braid_Real E_threshold,
			       const bool do_print,
			       std::string fname)
  {
    // Assert(level == app.finest_level,
    // 	   dealii::ExcMessage("Can only verify E is not too large on the finest level."));
    
    [[maybe_unused]] const auto od = app.levels[level]->offline_data;
    // Check that the level claimed matches the level of the vector by comparing sizes.
    //std::cout << "n_locally_owned_on_level(" << level << ") is " << app.n_locally_owned_at_level(level) << std::endl;
    Assert(od->n_locally_owned() == (int)std::get<0>(u.U).locally_owned_size()/app.problem_dimension,
	   dealii::ExcMessage("The number of DOF's from the vector is "
			      + std::to_string(std::get<0>(u.U).locally_owned_size()/app.problem_dimension)
			      + " which does not match the number that the offline data requires "
			      + std::to_string(od->n_locally_owned())
			      + "for checking if E is too large."));
    
    bool violates = false;
    const unsigned int local_size = od->n_locally_owned();
    for (unsigned int i = 0; i < local_size; i++)
    {
      const auto state = std::get<0>(u.U).get_tensor(i);
      const Number point_E = state[dim+1];
      if ( point_E > E_threshold )
      {
	violates = true;
	std::string ostring = "E is too large (E=" + std::to_string(point_E) 
	  + ") compared to threshold "
	  + std::to_string(E_threshold)
	  + " at t " + std::to_string(time) + " at local_dof_index "
	  + std::to_string(i) + " with xbraid calling function "
	  + std::to_string(calling);
	//FIXME: make this a conditional ostream?
	std::cout << ostring << std::endl;
	break;
      }
    } // locally dofs

    if (violates && do_print)
    {
      fname+=std::to_string(calling);
      app.print_solution(u.U, time, level, fname, t_idx);
    }

    return violates;
  }
  
} // Namespace mgrit_functions
