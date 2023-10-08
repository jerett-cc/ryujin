/*
 * level_structures.h
 *
 *  Created on: Oct 5, 2023
 *      Author: jerett
 */

#pragma once

//deal.II includes
#include <deal.II/base/parameter_acceptor.h>

//ryujin includes
#include "hyperbolic_module.h"
#include "offline_data.h"
#include "geometry_cylinder.h"
#include "discretization.h"
#include "hyperbolic_system.h"
#include "euler/parabolic_system.h"
#include "time_loop.h"
#include "euler/description.h"
#include "initial_values.h"
#include "offline_data.h"
#include "parabolic_module.h"
#include "postprocessor.h"
#include "quantities.h"
#include "time_integrator.h"
#include "vtu_output.h"

namespace ryujin{
  namespace mgrit{
    ///A struct containing all relevant objects to a given MGRIT level
    /**
     * This struct stores all the relevant information for a given level
     * in the mgrit algorithm.
     *
     * Intended usage is as follows:
     *
     *    Vector<LevelStructures> levels;
     *    for(const auto l : mgrit_levels)
     *      levels[l].offline_data = std::make_shared<ryujin::OfflineData<dim, Number>(...);
     *
     *    //select the relevant level data
     *    int level = 0;
     *
     *    f(...,levels[level].offline_data,...);
     */
    template<typename Description, int dim, typename Number = double>
    class LevelStructures : public dealii::ParameterAcceptor
    {

      public:

        LevelStructures(const MPI_Comm &comm_x,
            const int refinement);

        void prepare();

        using HyperbolicSystem = typename Description::HyperbolicSystem;
        using ParabolicSystem = typename Description::ParabolicSystem;

        MPI_Comm level_comm_x;
        const int level_refinement;

        std::shared_ptr<ryujin::OfflineData<dim,Number>> offline_data;
        std::shared_ptr<ParabolicSystem> parabolic_system;
        std::shared_ptr<HyperbolicSystem> hyperbolic_system;
        std::shared_ptr<ryujin::HyperbolicModule<Description,
        dim,
        Number>> hyperbolic_module;
        std::shared_ptr<ryujin::ParabolicModule<Description,
        dim,
        Number>> parabolic_module;
        std::shared_ptr<ryujin::Discretization<dim>> discretization;
        std::shared_ptr<ryujin::TimeIntegrator<Description,
        dim,
        Number>> time_integrator;
        std::shared_ptr<ryujin::InitialValues<Description,
        dim,
        Number>> initial_values;
        std::shared_ptr<ryujin::Postprocessor<Description,
        dim,
        Number>> postprocessor;
        std::shared_ptr<ryujin::VTUOutput<Description,dim,Number>> vtu_output;
        std::shared_ptr<ryujin::Quantities<Description,dim,Number>> quantities;

    };
  }//namespace mgrit
}//namespace ryujin







