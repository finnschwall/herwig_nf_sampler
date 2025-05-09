#ifndef HERWIG_DIAGRAMSAMPLER_H
#define HERWIG_DIAGRAMSAMPLER_H

#include "ThePEG/Interface/ClassDocumentation.h"
#include "ThePEG/EventRecord/Particle.h"
#include "ThePEG/Repository/UseRandom.h"
#include "ThePEG/Repository/EventGenerator.h"
#include "ThePEG/Repository/Repository.h"
#include "ThePEG/Utilities/DescribeClass.h"

#include "ThePEG/Interface/Parameter.h"
#include "ThePEG/Interface/ParVector.h"
#include "ThePEG/Interface/Switch.h"

#include "ThePEG/Persistency/PersistentOStream.h"
#include "ThePEG/Persistency/PersistentIStream.h"

#include "ThePEG/Handlers/StandardEventHandler.h"
#include "ThePEG/Handlers/StandardXComb.h"

#include "Herwig/Utilities/Progress.h"

#include "Herwig/Sampling/BinSampler.h"
#include "Herwig/Sampling/GeneralSampler.h"
#include "SingleSampler.h"

namespace Herwig {
    using namespace ThePEG;

    class SingleSampler;

    class DiagramSampler : public BinSampler {
    public:
        DiagramSampler();

        virtual ~DiagramSampler();

    public:

        /**
         * Clone this object.
         */
        Ptr<DiagramSampler>::ptr cloneMe() const {
            return dynamic_ptr_cast<Ptr<DiagramSampler>::ptr>(clone());
        }

    public:

        /**
         * Generate the next point; store the point in lastPoint() and its
         * weight using select(); if noMaxInfo is true, do not throw
         * NewMaximum or UpdateCrossSections exceptions.
         */
        virtual double generate();

        /**
         * Initialize this bin sampler. This default version calls runIteration.
         */
        virtual void initialize(bool progress);

        /**
         * Finalize this sampler.
         */
        virtual void finalize(bool);

        /**
         * Adapt
         */
        virtual void adapt();

        /**
         * Return true, if grid data exists for this sampler.
         */
        virtual bool existsGrid() const;

        /**
         * Save grid data
         */
        virtual void saveGrid() const;

    private:
        /**
         * Adapts this sampler. To remove redundant code this can be used with parameter false for the explore step.
         * @param doAdapt true iff the SingleSampler instances should also get adapted
         */
        void adaptImpl(bool doAdapt = true);

    public:
        /** @name Getters for parameters of a CellGridSampler */
        //@{
        bool unweightCells() const {
            return theUnweightCells;
        }

        const vector<int>& pre_adaption_splits() const {
            return the_pre_adaption_splits;
        }

        size_t explorationPoints() const {
            return theExplorationPoints;
        }

        size_t explorationSteps() const {
            return theExplorationSteps;
        }

        double gain() const {
            return theGain;
        }

        double epsilon() const {
            return theEpsilon;
        }

        void epsilon(double theEpsilon) {
            DiagramSampler::theEpsilon = theEpsilon;
        }

        double minimumSelection() const {
            return theMinimumSelection;
        }

        double minDiagSelection() const {
            return theMinDiagramSelection;
        }

        size_t diagramDimension() const {
            return theDiagramDimension;
        }

        double epsilonInterpolLimit() const {
            return theEpsilonInterpolLimit;
        }

        size_t minAdaptPoints() const {
            return theMinAdaptPoints;
        }

        size_t diagramCount() const {
            return theSamplers.size();
        }
        //@}

    public:

        /** @name Functions used by the persistent I/O system. */
        //@{
        /**
         * Function used to write out object persistently.
         * @param os the persistent output stream written to.
         */
        void persistentOutput(PersistentOStream& os) const;

        /**
         * Function used to read in object persistently.
         * @param is the persistent input stream read from.
         * @param version the version number of the object when written.
         */
        void persistentInput(PersistentIStream& is, int version);
        //@}

        /**
         * The standard Init function used to initialize the interfaces.
         * Called exactly once for each class by the class description system
         * before the main function starts or
         * when this class is dynamically loaded.
         */
        static void Init();

        void fromXML(const XML::Element& elem);

        XML::Element toXML() const;

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

    private:

        /**
         * The assignment operator is private and must never be called.
         * In fact, it should not even be implemented.
         */
        DiagramSampler& operator=(const DiagramSampler&) = delete;

    private:

        /**
         * Selector to randomly pick a diagram
         */
        Selector<SingleSampler*> theSelector;

        /**
         * List of all SingleSamplers
         * This object owns them
         */
        vector <SingleSampler> theSamplers;

        /**
         * Index of the dimension for diagram selection
         */
        size_t theDiagramDimension{};

        /*
         * Minimum selection of a diagram
         */
        double theMinDiagramSelection;

        /**
         * Minimum contribution to the integral for which the SingleSampler will still get adapted
         */
        double theMinAdaptPart;

        /**
         * Limit up to which point epsilon changes
         */
        double theEpsilonInterpolLimit;

        /**
         * Reduction factor for the selection probability of diagrams with integral 0
         */
        double theProbDampening;

        /**
         * Minimum points used for adaption/exploration of a cell
         */
        size_t theMinAdaptPoints;

        //See CellGridSampler:

        /**
         * The number of points used to explore a cell
         */
        size_t theExplorationPoints;

        /**
         * The number of exploration steps
         */
        size_t theExplorationSteps;

        /**
         * The adaption threshold.
         */
        double theGain;

        /**
         * The adaption threshold.
         */
        double theEpsilon;

        /**
         * The minimum probability for cell selection.
         */
        double theMinimumSelection;

        /**
         * The splittings for each dimension befor adaption.
         */
        vector<int> the_pre_adaption_splits;

        /**
         * The number of splits to put into parton luminiosity degrees of
         * freedom.
         */
        int theLuminositySplits;

        /**
         * Perform unweighting in cells
         */
        bool theUnweightCells;
    };

}


#endif //HERWIG_DIAGRAMSAMPLER_H


