#include "level_structures.h"
#include "mgrit_functions.h"

#include <type_traits>

namespace mgrit_functions
{

  template <typename Number, typename Description, int dim>
  dealii::Tensor<1, dim>
  calculate_forces_on_object(mgrit::MyApp<Number, Description, dim> *app,
                             const mgrit::MyVector<Number, Description, dim> &u,
                             const braid_Real t,
                             const braid_Int cycle,
                             const braid_Int t_idx)
  {
    using scalar_type = dealii::LinearAlgebra::distributed::Vector<Number>;

    const auto offline_data = app->levels[app->finest_level]->offline_data;
    const auto mpi_communicator = app->comm_x;
    const auto hyperbolic_system_view = app->levels[app->finest_level]
                                            ->hyperbolic_system->get()
                                            .template view<dim, Number>();
    const int n_q_points = app->levels[app->finest_level]
                               ->offline_data->discretization()
                               .face_quadrature()
                               .size();

    std::vector<Number> pressure_values(n_q_points);

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
            .face_quadrature() /*QGauss<dim-1>*/,
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

    dealii::Tensor<1, dim> output_forces;
    std::ostringstream ostring;

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
              output_forces += forces;

              ostring << std::setprecision(16) << "Q point " << q
                      << " is located at " << fe_face_values.quadrature_point(q)
                      << " has a pressure P of " << pressure_values[q]
                      << " and has forces equal to ";
              for (int i = 0; i < dim; i++)
                ostring << forces[i] << " ";
              ostring << "at time " << t << std::endl;

            } // loop over q points
          }   // if cell face is at boundary && on the object
        }     // face loop
      }       // locally_owned cells
    }         // cell loop

    // now, sum the values across all processes.
    dealii::Utilities::MPI::sum(output_forces, mpi_communicator);
    // collect the output strings so we know forces at quadrature points on
    // object.
    std::vector<std::string> all_output =
        dealii::Utilities::MPI::gather(mpi_communicator, ostring.str());
    if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
      std::ostringstream cycle_stream;
      cycle_stream << std::setw(app->cycle_io_width) << std::setfill('0')
                   << std::to_string(cycle);
      std::ofstream o;
      o.open(app->base_name + "_brick" + std::to_string(t_idx) + "_cycle" +
             cycle_stream.str() + "_forces_quadrature_points.csv");
      for (auto s : all_output)
        o << s;
    }

    return output_forces;
  }


  template <typename Description, int dim, typename Number>
  dealii::Tensor<1, dim + 2, Number>
  global_average_state(mgrit::MyVector<Number, Description, dim> &u,
                       const unsigned int level,
                       const mgrit::MyApp<Number, Description, dim> &app)
  {
    using Tensor = dealii::Tensor<1, dim + 2, Number>;
    Tensor avg_state;

    // Loop over all locally owned nodes, adding to avg_state.
    auto &od_level = app.levels[level]->offline_data; // TODO: const?

    Assert(
        (std::is_same<Description, typename ryujin::Euler::Description>::value),
        dealii::ExcMessage(
            "global_average_state only designed for the Euler case"
            " so that problem dimension = space_dim+2, since this function"
            " returns a (1,dim+2) tensor, which only makes sense if"
            " the state vector is dim+2 long."));

    unsigned int n_local_dof = app.n_locally_owned_at_level(level);
    for (unsigned int node = 0; node < n_local_dof; node++) {
      auto state = std::get<0>(u.U).get_tensor(node);
      avg_state += state;
    }

    // Now all local contributions have been calculated, we do a
    // communication/allreduce to sum all the states, and then divide by the
    // global n_dofs.
    const unsigned int n_global_dof = od_level->dof_handler().n_dofs();
    const unsigned int tensor_size = Tensor::dimension;
    Assert(
        tensor_size == dim + 2,
        dealii::ExcMessage("global_average_state fails vector dimension sanity "
                           "check for the Euler equations."));
    const std::function<Number(const Number &, const Number &)> sum(
        [](const Number &u, const Number &v) { return u + v; });

    // FIXME: is there a more efficient way to do this without the loop?
    for (unsigned int i = 0; i < tensor_size; i++)
      avg_state[i] =
          dealii::Utilities::MPI::all_reduce(avg_state[i], app.comm_x, sum);

    avg_state /= n_global_dof;

    return avg_state;
  }

  namespace internal
  {
    /// This function takes a node and sets the component to the global average
    /// by stealing some of the mass from the local stencil in a way that after
    /// we set the node to the global average the total global average did not
    /// change.
    template <typename Description, int dim, typename Number>
    void redistribute_to_maintain_global_average(
        mgrit::MyVector<Number, Description, dim> &U,
        const unsigned int node,
        const unsigned int level,
        const unsigned int component,
        const Number global_avg,
        const mgrit::MyApp<Number, Description, dim> &app)
    {
      const auto &sparsity_level =
          app.levels[level]->offline_data->sparsity_pattern();
#ifdef DEBUG
      const auto &view = app.levels[level]
                             ->hyperbolic_system->get()
                             .template view<dim, Number>();
      // Pre-compute global averages.
      const auto avgs =
          global_average_state<Description, dim, Number>(U, level, app);
#endif
      // Loop over the sparsity on this node, and set the value to the global
      // average, subtracting off equal parts from the surrounding nodes to
      // ensure that the global average stays the same.
      auto stencil_size = sparsity_level.row_length(node);
      Assert(stencil_size > 1,
             dealii::ExcMessage(
                 "Average invariant mass re-sistribution only works for now on "
                 "triangulations without constraints."));
      // TAG: #2.1 loop over local nodes, subtracting off part of the node's
      // value in this
      //           component to redistribute the mass in this component.
      auto state_node = std::get<0>(U.U).get_tensor(node);
      const Number mass_to_subtract =
          (global_avg + state_node[component]) / Number(stencil_size);

      for (auto jt = sparsity_level.begin(node); jt != sparsity_level.end(node);
           ++jt) {
        const auto j = jt->column();
        // skip the current node.
        if (j == node)
          continue;

        auto state_j = std::get<0>(U.U).get_tensor(j);
        state_j[component] -= mass_to_subtract;

        Assert(view.is_admissible(state_j),
               dealii::ExcMessage(
                   "In trying to redistribute mass from other nodes to "
                   "this node, you've created an inadmissible state. Likely, "
                   "the node's value is too small, and the surronding nodes "
                   "are not large enough to take any mass from."));

        std::get<0>(U.U).write_tensor(state_j, j);
      }

      // now, reset the state node to global average
      state_node[component] = global_avg;
      Assert(
          view.is_admissible(state_node),
          dealii::ExcMessage(
              "In trying to redistribute mass from other nodes to "
              "this node, you've created an inadmissible state at this node."));

      std::get<0>(U.U).update_ghost_values();

#ifdef DEBUG
      // Compute new global averages and compare to old. Nothing should have
      // changed.
      auto avgs_new =
          global_average_state<Description, dim, Number>(U, level, app);
      avgs_new -= avgs;
      Assert(std::abs(avgs_new.norm()) < 1e-16,
             dealii::ExcMessage(
                 "In trying to redistribute mass from other nodes to "
                 "this node, you've changed the global averages. The intent of"
                 "the function is to MAINTAIN the global averages."));
#endif
    }
  } // namespace internal

  template <typename Description, int dim, typename Number>
  void
  enforce_physicality_bounds(mgrit::MyVector<Number, Description, dim> &u,
                             const unsigned int level,
                             const mgrit::MyApp<Number, Description, dim> &app,
                             [[maybe_unused]] const Number t)
  {
    // The incoming u.U should already respect the following equalities:
    // (1) u.U[0]     = rho
    // (2) u.U[i]     = rho*vi, for each spatial dimension
    // (3) u.U[dim+1] = E

    // Here, we assume an ideal gas (with a gamma law).
    // Further, these variable are related to each other via the following
    // equality: where e is the specific energy density (4)               E =
    // rho*e + 1/2*(|m|^2)/rho

    // And we have the following equations of interest, which define the
    // invariant domain of ryujin's authors. When we are done with this
    // function, we hope that the following inequalities hold: (Density) rho > 0
    // (Specific Entropy) psi   = internal_energy*(1/rho)^gamma > 0
    // (Internal Energy)  rho*e = (E - 0.5/rho*|m|^2)           > 0

    // For this projection, we also know that the pressure is given by the
    // following EOS: (Pressure) P = (gamma-1)*internal_energy

    // This function works in the following steps
    //   - The tau-like corrections from MGRIT cause negative densities, so we
    //   first
    //     set a small, but nonzero density.
    //   - We observed spikes in E, presumably caused by the tau corrections, so
    //   we
    //     next limit E at each node to the maximum of its neighbors if the
    //     node's E contributes more than 90% of the total E in the area. (A
    //     spike).
    //   - Next, we have to ensure that all of our changes are in the invariant
    //   domain
    //     described by (Density), (Specific Entropy), (Internal Energy).
    //     Density ought to be OK, but we need to check the other conditions. If
    //     these fail, we then compute incremental delta E energy increases,
    //     until both (Specific Entropy) and (Internal Energy) are satisfied. In
    //     principle, thes could be calculated independently for different PDE
    //     systems or EOS. For the assumptions outlined above, we note that
    //     (Internal Energy) implies (Specific Entropy). So we just add to E
    //     until E > 0.5/rho*|m|^2 + epsilon (the parameter epsilon is to ensure
    //     we are small, but not negative).
    //   - Finally, we update the vector reference u with all the states that
    //   needed modifying .

    // Since this function modifies rho and E, we need to make sure this is done
    // in a compatible way. Meaning that if we limit rho >= 0, then we need to
    // update all the states, since (1), (2), and (4) all involve rho.

    // Similarly, modifying E should entail modifying rho*e, and hence needs to
    // modify somehow rho, e, and m?

    Assert(
        app.vector_size_match_level(u.U, level),
        dealii::ExcMessage(
            "enforce_physicality only works if the vector's size and the size "
            "from the level match. This is because a copy is made from the "
            "size "
            "from the offline_data."));
    Assert(
        (std::is_same<Description, typename ryujin::Euler::Description>::value),
        dealii::ExcMessage(
            "enforce_physicality only designed for the Euler case"
            " with a gamma law EOS."));


    // Calculate minimum allowable values in density and internal energy.

    // Here, we could calculate an averaged density and
    // set a minimum density to be one hundredth of the average density.
    // Likewise for the total energy.
    // const auto avgs =
    //     global_average_state<Description, dim, Number>(u, level, app);
    // Number eps_rho = avgs[0] * 1e-2;
    // Number eps_E = avgs[dim + 1] * 1e-2;

    // Here, we could have the user define their own reference density and
    // internal energy, then scale by user defined scale factor.
    const Number eps_rho = app.reference_rho * app.reference_scale;
    const Number eps_e = app.reference_e * app.reference_scale;
    const auto view = app.levels[level]
                          ->hyperbolic_system->get()
                          .template view<dim, Number>();

    // First, we loop over all the local nodes and
    // TAG: #1 projection: Loop over all nodes.
    for (unsigned int node = 0; node < app.n_locally_owned_at_level(level);
         node++) {
      auto state = std::get<0>(u.U).get_tensor(node);
      // if the current state is OK, move on.
      if (view.is_admissible(state))
        continue;

      // here, we are not admissible, so we first check the internal energy
      // and then set the
      const Number old_e = view.internal_energy(state);
      const Number old_rho = view.density(state);

      // density needs to be set, we also need to update the total energy.
      if (old_rho < eps_rho) {
        state[0] = eps_rho;
        state[dim + 1] =
            old_e + 0.5 / eps_rho * (view.momentum(state).norm_square());
      }

      // If density was the only problem, then we skip checking the internal
      // energy.
      if (view.is_admissible(state)) {
        std::get<0>(u.U).write_tensor(state, node);
        continue;
      }

      // If we are still not on stable manifold, then internal energy needs to
      // be set, wherein we modify the total energy. We do this via the
      // relationship that TE = IE_new + KE.
      if (old_e < eps_e) {
        // No need to set the density, that's already done.
        state[dim + 1] =
            eps_e + 0.5 / state[0] * (view.momentum(state).norm_square());
        // we could mimic how we write the state, but instead we assert we are
        // good and
      }

      // If at this point the state is still not admissible, then we have a
      // major issue and should crash after printing the state.
      const auto rho_new = view.density(state);
      const auto e_new = view.internal_energy(state);
      const auto s_new = view.specific_entropy(state);

      constexpr auto gt = dealii::SIMDComparison::greater_than;
      using T = Number;
      const auto test =
          dealii::compare_and_apply_mask<gt>(rho_new, T(0.), T(0.), T(-1.)) + //
          dealii::compare_and_apply_mask<gt>(e_new, T(0.), T(0.), T(-1.)) +   //
          dealii::compare_and_apply_mask<gt>(s_new, T(0.), T(0.), T(-1.));

      if (!(test == Number(0.))) {
        std::cout << std::fixed << std::setprecision(16);
        std::cout << "Bounds violation: Negative state [rho, e, s] detected!\n";
        std::cout << "\t\trho: " << rho_new << "\n";
        std::cout << "\t\tint: " << e_new << "\n";
        std::cout << "\t\tent: " << s_new << "\n" << std::endl;
        std::cout << "[WARNING]: Projection failed." << std::endl;
        exit(EXIT_FAILURE);
      }
      // TODO: do we need to modify the velocities to conserve the total energy?

      // in debug mode, we check that the state is all good now.
      // if its not ok at this point, we have an issue.
      Assert(view.is_admissible(state),
             dealii::ExcMessage("Projection did not work. The state "
                                "after projection limiting \rho e and \rho "
                                "is still not admissible."));

      std::get<0>(u.U).write_tensor(state, node);
    }
    // Communicate the changes.
    std::get<0>(u.U).update_ghost_values();
  }

  template <typename Description, int dim, typename Number>
  void conserved_and_entropy_in_system(
      const mgrit::MyVector<Number, Description, dim> &u,
      const mgrit::MyApp<Number, Description, dim> *app,
      const braid_Int level,
      const Number time)
  {
    [[maybe_unused]] int n_dofs = app->n_locally_owned_at_level(level);
    [[maybe_unused]] int n_components = app->problem_dimension;
    [[maybe_unused]] int n_vector_dof =
        std::get<0>(u.U).locally_owned_size() / n_components;

    Assert((level == app->finest_level),
           dealii::ExcMessage("Can only calculate entropy on finest level."));
    Assert((n_dofs == n_vector_dof),
           dealii::ExcMessage("Total entropy can only be calculated when dofs "
                              "match on mesh and u."
                              "Here, level=" +
                              std::to_string(level) + " which has " +
                              std::to_string(n_vector_dof) +
                              " when the expected " +
                              "number of dofs on the finest level is " +
                              std::to_string(n_dofs)));
    // Calculate the entropy in the system
    const auto hyperbolic_system_view = app->levels[level]
                                            ->hyperbolic_system->get()
                                            .template view<dim, Number>();

    Number total_entropy = 0;
    Number mass = 0;
    Number momentum_sqr = 0;
    Number E = 0;

    const auto comm_x = app->comm_x;
    const auto od = app->levels[app->finest_level]->offline_data;
    const auto &fe = od->discretization().finite_element();
    const auto &quad = od->discretization().quadrature();

    dealii::FEValues<dim> fe_vals(
        fe, quad, dealii::update_values | dealii::update_JxW_values);

    for (const auto &cell : od->dof_handler().active_cell_iterators()) {
      fe_vals.reinit(cell);
      if (cell->is_locally_owned()) {
        for (const unsigned int q_idx : fe_vals.quadrature_point_indices()) {
          const auto state = std::get<0>(u.U).get_tensor(q_idx);
          const Number point_entropy =
              hyperbolic_system_view.specific_entropy(state);
          const Number point_rho = state[0];
          Number point_mom_sqr = 0;
          for (int d = 0; d < dim; d++)
            point_mom_sqr += state[1 + d] * state[1 + d];
          const Number point_E = state[dim + 1];

          for (const unsigned int i : fe_vals.dof_indices()) {
            total_entropy += (fe_vals.shape_value(i, q_idx) * point_entropy *
                              fe_vals.JxW(q_idx));
            mass += (fe_vals.shape_value(i, q_idx) * point_rho *
                     fe_vals.JxW(q_idx));
            momentum_sqr += (fe_vals.shape_value(i, q_idx) * point_mom_sqr *
                             fe_vals.JxW(q_idx));
            E += (fe_vals.shape_value(i, q_idx) * point_E * fe_vals.JxW(q_idx));
          } // dof contributions for each cell
        }   // quadrature points in cell
      }     // locally owned
    }       // cells

    // communicate across space
    dealii::Utilities::MPI::sum(total_entropy, comm_x);
    dealii::Utilities::MPI::sum(mass, comm_x);
    dealii::Utilities::MPI::sum(momentum_sqr, comm_x);
    dealii::Utilities::MPI::sum(E, comm_x);
    if (dealii::Utilities::MPI::this_mpi_process(app->comm_x) == 0) {
      std::cout << "Total entropy at time t= " << time << " is "
                << total_entropy << std::endl;
      std::cout << "Total Mass at time t= " << time << " is " << mass
                << std::endl;
      std::cout << "Total Momentum Squared at time t= " << time << " is "
                << momentum_sqr << std::endl;
      std::cout << "Total E at time t= " << time << " is " << E << std::endl;
    }
  }

  template <typename Description, int dim, typename Number>
  bool does_E_exceed_threshold(mgrit::MyVector<Number, Description, dim> &u,
                               mgrit::MyApp<Number, Description, dim> &app,
                               [[maybe_unused]] const braid_Int level,
                               const Number time,
                               const braid_Int t_idx,
                               const braid_Int calling,
                               const braid_Real E_threshold,
                               const bool do_print,
                               std::string fname)
  {
    // Assert(level == app.finest_level,
    // 	   dealii::ExcMessage("Can only verify E is not too large on the finest
    // level."));

    [[maybe_unused]] const auto od = app.levels[level]->offline_data;
    // Check that the level claimed matches the level of the vector by comparing
    // sizes.
    // std::cout << "n_locally_owned_on_level(" << level << ") is " <<
    // app.n_locally_owned_at_level(level) << std::endl;
    Assert(
        od->n_locally_owned() ==
            (int)std::get<0>(u.U).locally_owned_size() / app.problem_dimension,
        dealii::ExcMessage(
            "The number of DOF's from the vector is " +
            std::to_string(std::get<0>(u.U).locally_owned_size() /
                           app.problem_dimension) +
            " which does not match the number that the offline data requires " +
            std::to_string(od->n_locally_owned()) +
            "for checking if E is too large."));

    bool violates = false;
    const unsigned int local_size = od->n_locally_owned();
    for (unsigned int i = 0; i < local_size; i++) {
      const auto state = std::get<0>(u.U).get_tensor(i);
      const Number point_E = state[dim + 1];
      if (point_E > E_threshold) {
        violates = true;
        std::string ostring =
            "E is too large (E=" + std::to_string(point_E) +
            ") compared to threshold " + std::to_string(E_threshold) +
            " at t " + std::to_string(time) + " at local_dof_index " +
            std::to_string(i) + " with xbraid calling function " +
            std::to_string(calling);
        // FIXME: make this a conditional ostream?
        std::cout << ostring << std::endl;
        break;
      }
    } // locally dofs

    if (violates && do_print) {
      fname += std::to_string(calling);
      app.print_solution(u.U, time, level, fname, t_idx);
    }

    return violates;
  }

  template <typename Description, int dim, typename Number>
  bool
  state_admissible_everywhere(mgrit::MyVector<Number, Description, dim> &u,
                              const unsigned int level,
                              const mgrit::MyApp<Number, Description, dim> &app,
                              const Number t,
                              const braid_Int calling)
  {
    // Create Hyperbolic System View, where we can query admissibility.
    const auto view = app.levels[level]
                          ->hyperbolic_system->get()
                          .template view<dim, Number>();
    Assert(app.vector_size_match_level(u.U, level),
           dealii::ExcMessage(
               "admissible_everywhere() only works if the vector's size and "
               "the size "
               "from the level match. This is because the function loops over "
               "all the locally owned dofs at the supposed level."));
    ryujin::Scope scope(app.computing_timer, "state_admissible_everywhere");
    for (unsigned int node = 0; node < app.n_locally_owned_at_level(level);
         node++) {
      auto state = std::get<0>(u.U).get_tensor(node);
      if (!view.is_admissible(state)) {
#ifdef DEBUG_MGRIT
        std::cout << "calling=" << calling << " a state at time t=" << t
                  << " is not admissible node=" << node
                  << " and state=" << state << std::endl;
#endif
        return false;
      }
    }

    return true;
  }

} // Namespace mgrit_functions
