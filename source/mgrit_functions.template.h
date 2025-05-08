#include "mgrit_functions.h"
#include "level_structures.h"

#include <type_traits>

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
                                  [[maybe_unused]]const Number t,
				  const braid_Int calling)
  {
    // The incoming u.U should already respect the following equalities:
    // (1) u.U[0]     = rho
    // (2) u.U[i]     = rho*vi, for each spatial dimension
    // (3) u.U[dim+1] = E

    // Here, we assume an ideal gas (with a gamma law).
    // Further, these variable are related to each other via the following equality:
    // where e is the specific energy density
    // (4)               E = rho*e + 1/2*(|m|^2)/rho
    
    // And we have the following equations of interest, which define the invariant domain of
    // ryujin's authors. When we are done with this function, we hope that the following
    // inequalities hold: 
    // (Density)          rho                                   > 0
    // (Specific Entropy) psi   = internal_energy*(1/rho)^gamma > 0
    // (Internal Energy)  rho*e = (E - 0.5/rho*|m|^2)           > 0

    // For this projection, we also know that the pressure is given by the following EOS:
    // (Pressure) P = (gamma-1)*internal_energy

    // This function works in the following steps
    //   - The tau-like corrections from MGRIT cause negative densities, so we first
    //     set a small, but nonzero density.
    //   - We observed spikes in E, presumably caused by the tau corrections, so we
    //     next limit E at each node to the maximum of its neighbors if the node's
    //     E contributes more than 90% of the total E in the area. (A spike).
    //   - Next, we have to ensure that all of our changes are in the invariant domain
    //     described by (Density), (Specific Entropy), (Internal Energy). Density ought
    //     to be OK, but we need to check the other conditions. If these fail, we then
    //     compute incremental delta E energy increases, until both (Specific Entropy)
    //     and (Internal Energy) are satisfied. In principle, thes could be calculated
    //     independently for different PDE systems or EOS. For the assumptions outlined above,
    //     we note that (Internal Energy) implies (Specific Entropy). So we just add to E until
    //     E > 0.5/rho*|m|^2 + epsilon (the parameter epsilon is to ensure we are small,
    //     but not negative).
    //   - Finally, we update the vector reference u with all the states that needed modifying .
    
    // Since this function modifies rho and E, we need to make sure this is done in a compatible way.
    // Meaning that if we limit rho >= 0, then we need to update all the states, since (1), (2), and
    // (4) all involve rho.

    // Similarly, modifying E should entail modifying rho*e, and hence needs to modify
    // somehow rho, e, and m?
    
    // Create Hyperbolic System View, where we can compute functions like pressure.
    const auto view = app.levels[level]->hyperbolic_system->get().template view<dim,Number>();
    auto &od_level = app.levels[level]->offline_data;// TODO: const?
    const auto &sparsity_level = od_level->sparsity_pattern();

    Assert(app.vector_size_match_level(u.U, level),
	   dealii::ExcMessage("enforce_physicality only works if the vector's size and the size "
			      "from the level match. This is because a copy is made from the size "
			      "from the offline_data."));
    Assert((std::is_same<Description,typename ryujin::Euler::Description>::value),
	   dealii::ExcMessage("enforce_physicality only designed for the Euler case"
			      " with a gamma law EOS.")); 
    

    // Calculate global averages in density and total energy, for use in the eps
    // terms which limit the sensity and internal energy, rather than a non-physical
    // term like 1e-8.

    // state  avgs    = global_average_rho_E();
    // Number eps_rho = avgs[0]     * 1e-2;
    // Number eps_E   = avgs[dim+1] * 1e-2;
    
    // First, we limit the density to be non-negative, and update all the relations
    // with this new density.
    for(unsigned int node=0; node < app.n_locally_owned_at_level(level); node++)
    {
      auto state = std::get<0>(u.U).get_tensor(node);
      Number old_rho = state[0];
      
      state[0] = std::max(old_rho, 1e-8);// set this to eps_rho
      std::get<0>(u.U).write_tensor(state, node);
    }
    // Communicate the changes.
    std::get<0>(u.U).update_ghost_values();
    
    // Now that the densities are fixed, let's ensure that E is not too large
    // by checking if it contributes greater than 90% of the total enegry of a local
    // stencil. Then, if it does, we replace it with the largest (hopefully reasonable
    // FIXME) of the surrounding connected nodes, in all solution components. This way,
    // we are guranteed to satisfy (1), (2), (3), (4) since we assume that incoming data
    // already satisfies this.
    
    // First, set up some temporary data which we will store our modifications, if needed.
    mgrit::MyVector<Number, Description,dim> copy;
    app.reinit_to_level(&copy,level);
    std::get<0>(copy.U) = std::get<0>(u.U);
    
    // Compute the local average in E and use this as a limit on E in the copy.
    for(unsigned int node=0; node < app.n_locally_owned_at_level(level); node++)
    {
      // For this node, we loop over the local stencil and calculate an average
      // of the other nodes.
      Number surrounding_E_sum = 0.0;
      Number maximum_surrounding_E = 0.0;// FIXME: does this 
      int    max_surrounding_idx   = 0; // this should not be node after we do the next step.
      auto stencil_size = sparsity_level.row_length(node);
      Assert(stencil_size > 1,
	     dealii::ExcMessage("Enforce physicality only works for now on "
				"triangulations without constraints."));
      for(auto jt = sparsity_level.begin(node); jt != sparsity_level.end(node); ++jt)
      {
	const auto stencil_node_j = jt->column();
	// we only calculate on the other connected nodes.
	if(stencil_node_j == node)// the diagonal is stored at 0 since our sparsity is square.
	  continue;
	const auto state_j = std::get<0>(u.U).get_tensor(stencil_node_j);
	auto E = state_j[dim+1];
	
	if(E >= maximum_surrounding_E)
	{
	  // We've found the new max, note the column and update the max.
	  max_surrounding_idx = stencil_node_j;
	  maximum_surrounding_E = E;  
	}
	surrounding_E_sum += E;
      }

      Assert(max_surrounding_idx != node,
	     dealii::ExcMessage("The maximum surrounding index cannot be the current node "
				"what we really want is the max of the "
				"connected SURROUNDING nodes, not the current one."));
      auto state_node = std::get<0>(u.U).get_tensor(node);
      auto E_node = state_node[dim+1];
      
      const Number total_stencil_E = surrounding_E_sum + E_node;
      
      // Calculate the average of the other nodes.
      // guranteed to not divide by zero since we assert above
      // that this mesh has no constraints.
      Number surrounding_avg = surrounding_E_sum/(stencil_size-1);

#ifdef DEBUG_OUTPUT
      std::cout << "Node: " << node << " row length is " << sparsity_level.row_length(node)
		<< " and the node_avg E is " << total_stencil_E/stencil_size
		<< " and the average from the surrouning nodes is "
		<< surrounding_avg <<  std::endl;
      std::cout << "While the E_node is " <<  E_node
		<< " and the surrounding maximum E " << maximum_surrounding_E
		<< std::endl;
#endif
      // Prevent E from being small.
      state_node[dim+1] = std::max(E_node, Number(1e-8));//TODO: change this to eps_E

      // Finally, disallow very large E.
      if(std::abs(state_node[dim+1]) > 3*surrounding_avg)
      {
	// If E is too large compared to the surrounding nodes, we replace all the vector data from
	// the data of the largest surrounding node.
	state_node = std::get<0>(u.U).get_tensor(max_surrounding_idx);
	std::cout << "Replacing E in projection operation." << std::endl;
      }

      // Next, we need to verify that the each node's state is admissible, if not, then
      // it is likely that the internal energy is negative, so we calculate a delta E based
      // on (Internal Energy).
      const Number eps = 1e-8;//TODO: change to eps_E
      if(!view.is_admissible(state_node))
      {
#ifdef DEBUG
	std::cout << "calling=" << calling
		  << " enforce_physicality() state after limiting density and E/Pressure "
		  << "on level=" << level
		  << " is not admissible node="
		  << node << " and state=" << state_node << std::endl;
#endif
	Number deltaE = -view.internal_energy(state_node)+eps;//TODO: change to eps_E
	state_node[dim+1] += deltaE;
      }

#ifdef DEBUG
      // Double ckeck that now the state is admissible.
      if(view.is_admissible(state_node))
      {
	std::cout << "calling=" << calling
		  << " enforce_physicality() state after limiting density and E/Pressure "
		  << "on level=" << level
		  << " has been made admissible node="
		  << node << " and state=" << state_node << std::endl;
      }
#endif
      
      // Write new state in the copied vector. TODO: does this need to happen every time
      // or only in the case that the above if(...) triggers?
      std::get<0>(copy.U).write_tensor(state_node, node);
    }

    //TODO: the equations (1), (2), (3),(4) maybe not satisfied with these
    //      arbitrary additions and limitations?

    // Exchange projection changes in copy.
    std::get<0>(copy.U).update_ghost_values();

    // now that the copy is fixed up, we move the copied data into the one we wish to change,
    // and update ghost to finish change.
    std::get<0>(u.U) = std::get<0>(copy.U);
    //std::cout << "---------------------------------------------------" << std::endl;
    // Make sure boundary conditions are satisfied on these states.
    //FIXME: this function also calls update_ghost_values(), do I need the one above?
    //app.levels[level]->hyperbolic_module->prepare_state_vector(u.U, t);
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

  template <typename Description, int dim, typename Number>
  bool state_admissible_everywhere(mgrit::MyVector<Number, Description, dim> &u,
				   const unsigned int level,
				   const mgrit::MyApp<Number, Description, dim> &app,
				   const Number t,
				   const braid_Int calling)
  {
    // Create Hyperbolic System View, where we can query admissibility.
    const auto view = app.levels[level]->hyperbolic_system->get().template view<dim,Number>();
    Assert(app.vector_size_match_level(u.U, level),
	   dealii::ExcMessage("admissible_everywhere() only works if the vector's size and the size "
			      "from the level match. This is because the function loops over "
			      "all the locally owned dofs at the supposed level."));

    for(unsigned int node=0; node < app.n_locally_owned_at_level(level); node++)
    {
      auto state = std::get<0>(u.U).get_tensor(node);
      if(!view.is_admissible(state))
      {
	std::cout << "calling=" << calling
		  << " a state at time t=" << t
		  << " is not admissible node="
		  << node << " and state=" << state << std::endl;
	return false;
      }
    }

    return true;
  }
  
} // Namespace mgrit_functions
