/*
 * level_structures.h
 *
 *  Created on: Oct 5, 2023
 *      Author: jerett
 */

#pragma once

#include <compile_time_options.h>

//deal.II includes
#include <deal.II/base/parameter_acceptor.h>

//ryujin includes
#include "discretization.h"
#include "hyperbolic_module.h"
#include "initial_values.h"
#include "offline_data.h"
#include "parabolic_module.h"
#include "postprocessor.h"
#include "quantities.h"
#include "time_integrator.h"
#include "vtu_output.h"

#include <deal.II/base/timer.h>
#include <deal.II/base/tensor.h>

#include <fstream>
#include <future>
#include <sstream>

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
     *    for(unsigned int i=0; i<levels.size(); i++)
     *      levels.at(i) = LevelStructures(...);
     *
     *    //select the relevant level data, eg. offline data
     *    int level = 0;
     *    const auto current_offline_data = levels[level]->offline_data;
     *
     */
    template<typename Description, int dim, typename Number = double>
    class LevelStructures : public dealii::ParameterAcceptor
    {

      public:

        LevelStructures(const MPI_Comm &comm_x,
            const int refinement = 0);

        void prepare();

        /*Describe relevant typenames*/
        using HyperbolicSystem = typename Description::HyperbolicSystem;
        using HyperbolicSystemView
            = typename Description::template HyperbolicSystemView<dim, Number>;
        using ParabolicSystem
            = typename Description::ParabolicSystem;
        using OfflineData
            = typename ryujin::OfflineData<dim, Number>;
        using HyperbolicModule
            = typename ryujin::HyperbolicModule<Description,dim, Number>;
        using Discretization = ryujin::Discretization<dim>;
        using ParabolicModule = ryujin::ParabolicModule<Description,dim, Number>;
        using TimeIntegrator = ryujin::TimeIntegrator<Description, dim, Number>;
        using InitialValues = ryujin::InitialValues<Description, dim, Number>;
        using VTUOutput = ryujin::VTUOutput<Description, dim, Number>;
        using Postprocessor = ryujin::Postprocessor<Description, dim, Number>;
        using Quantities = ryujin::Quantities<Description,dim,Number>;

        //The communicator used for computations spatially on this level
        MPI_Comm level_comm_x;
        //The amount of refinement for the mesh describing this level.
        const int level_refinement;
        //The dimension for this problem, i.e. the number of unknowns for each
        //node in the mesh.
        static constexpr unsigned int problem_dimension
          = HyperbolicSystemView::problem_dimension;

        //TODO: do I need this timer?
        //A map that is used to time different sections of code
        //The string is typically the section, and the timer corresponds to this.
        std::map<std::string, dealii::Timer> computing_timer;

        //The necessary structures to run a simulation on this level.
        std::shared_ptr<HyperbolicSystem> hyperbolic_system;
        std::shared_ptr<ParabolicSystem> parabolic_system;
        std::shared_ptr<Discretization> discretization;
        std::shared_ptr<OfflineData> offline_data;
        std::shared_ptr<InitialValues> initial_values;
        std::shared_ptr<HyperbolicModule> hyperbolic_module;
        std::shared_ptr<ParabolicModule> parabolic_module;
        std::shared_ptr<TimeIntegrator> time_integrator;
        std::shared_ptr<Postprocessor> postprocessor;
        std::shared_ptr<VTUOutput> vtu_output;
        std::shared_ptr<Quantities> quantities;

    };

    //Constructor that takes a @p MPI_Comm to be used by all objects, and
    //a global @p refinement for the underlying triangulation.
    template<typename Description, int dim, typename Number>
    LevelStructures<Description,dim, Number>::LevelStructures(const MPI_Comm& comm_x,
                                                              const int refinement)
    : ParameterAcceptor("/LevelStructures")
    , level_comm_x(comm_x)//what constructor is used here? copy?
    , level_refinement(refinement)
    , hyperbolic_system(std::make_shared<HyperbolicSystem>("/Equation"))
    , parabolic_system(std::make_shared<ParabolicSystem>("/Equation"))
    , discretization(std::make_shared<Discretization>(comm_x,
                                                      level_refinement,
                                                      "/Discretization"))
    , offline_data(std::make_shared<OfflineData>(comm_x,
                                                 *discretization,
                                                 "/OfflineData"))
    , initial_values(std::make_shared<InitialValues>(*hyperbolic_system,
                                                     *offline_data,
                                                     "/InitialValues"))
    , hyperbolic_module(std::make_shared<HyperbolicModule>(comm_x,
                                                           computing_timer,
                                                           *offline_data,
                                                           *hyperbolic_system,
                                                           *initial_values,
                                                           "/HyperbolicModule"))
    , parabolic_module(std::make_shared<ParabolicModule>(comm_x,
                                                         computing_timer,
                                                         *offline_data,
                                                         *hyperbolic_system,
                                                         *parabolic_system,
                                                         *initial_values,
                                                         "/ParabolicModule"))
    , time_integrator(std::make_shared<TimeIntegrator>(comm_x,
                                                       computing_timer,
                                                       *offline_data,
                                                       *hyperbolic_module,
                                                       *parabolic_module,
                                                       "/TimeIntegrator"))
    , postprocessor(std::make_shared<Postprocessor>(comm_x,
                                                    *hyperbolic_system,
                                                    *offline_data,
                                                    "/VTUOutput"))
    , vtu_output(std::make_shared<VTUOutput>(comm_x,
                                             *offline_data,
                                             *hyperbolic_module,
                                             *postprocessor,
                                             "/VTUOutput"))
    , quantities(std::make_shared<Quantities>(comm_x,
                                             *hyperbolic_system,
                                             *offline_data,
                                             "/Quantities"))
    {}

    /**
     * This function prepares all the data structure pointers.
     *
     * The implementation essentially calls .prepare() for all
     * underlying structures.
     */
    template<typename Description, int dim, typename Number>
    void LevelStructures<Description, dim, Number>::prepare()
    {
      discretization->prepare();
      offline_data->prepare(problem_dimension);
      initial_values->parse_parameters_callback();
      hyperbolic_module->prepare();
      parabolic_module->prepare();
      time_integrator->prepare();
      postprocessor->prepare();
      vtu_output->prepare();
      quantities->prepare("MGRIT_"/*base name*/, 1 /*output cycle*/);
    }

  }//namespace mgrit
}//namespace ryujin







