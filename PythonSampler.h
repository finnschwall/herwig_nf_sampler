#ifndef HERWIG_PYTHONSAMPLER_H
#define HERWIG_PYTHONSAMPLER_H

#include "ThePEG/Interface/ClassDocumentation.h"
#include "ThePEG/EventRecord/Particle.h"
#include "ThePEG/Repository/UseRandom.h"
#include "ThePEG/Repository/EventGenerator.h"
#include "ThePEG/Repository/Repository.h"
#include "ThePEG/Utilities/DescribeClass.h"

#include "ThePEG/Interface/Parameter.h"
#include "ThePEG/Interface/Switch.h"

#include "ThePEG/Persistency/PersistentOStream.h"
#include "ThePEG/Persistency/PersistentIStream.h"

#include "ThePEG/Handlers/StandardEventHandler.h"
#include "ThePEG/Handlers/StandardXComb.h"

#include "Herwig/Sampling/GeneralSampler.h"
#include "Herwig/Sampling/BinSampler.h"


#include <pybind11/embed.h>
namespace py = pybind11;

namespace Herwig {
    using namespace ThePEG;

    class PythonSampler : public BinSampler {
       
    public:
        PythonSampler();
        virtual ~PythonSampler();

    public:
        /**
         * Clone this object.
         */
        Ptr<PythonSampler>::ptr cloneMe() const {
            return dynamic_ptr_cast<Ptr<PythonSampler>::ptr>(clone());
        }
    private:
        py::module_ sampling;
        bool interpreterInitialized = false;
        size_t nSamplesCache = 100000;
        size_t cachedIndex = -1;

        std::vector<std::vector<double>> cachedPsPoints;
        std::vector<double> cachedProbabilities;
        std::vector<double> cachedFuncValues;
    public:
        // std::unique_ptr<py::scoped_interpreter> guard;
        
        double dSigDR(std::vector<double> p);
        double dSigDRRun(std::vector<double> p, double probability, double channelWeight);
        std::vector<double> dSigDRMatrix(const std::vector<std::vector<double>>& matrix);

        /**
         * Generate the next point; store the point in lastPoint() and its
         * weight using select(); if noMaxInfo is true, do not throw
         * NewMaximum or UpdateCrossSections exceptions.
         */
        virtual double generate();

        /**
         * Initialize this bin sampler.
         */
        virtual void initialize(bool progress);

        /**
         * Finalize this sampler.
         */
        virtual void finalize(bool);

        /**
         * Adapt/retrain the normalizing flow model
         */
        virtual void adapt();



    public:
        /**
         * The standard Init function used to initialize the interfaces.
         * Called exactly once for each class by the class description system
         * before the main function starts or
         * when this class is dynamically loaded.
         */
        static void Init();

    protected:
        /** @name Clone Methods. */
        //@{
        /**
         * Make a simple clone of this object.
         * @return a pointer to the new object.
         */
        virtual IBPtr clone() const;

        /** Make a clone of this object, possibly modifying the cloned object
         * to make it sane.
         * @return a pointer to the new object.
         */
        virtual IBPtr fullclone() const;
        //@}

    };
}

#endif // HERWIG_PYTHONSAMPLER_H