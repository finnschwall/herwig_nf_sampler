#include "SingleSampler.h"

using namespace Herwig;
using namespace ThePEG;
using namespace ExSample;

SingleSampler::SingleSampler() = default;

SingleSampler::SingleSampler(const RCPtr<DiagramBase>& diag, DiagramSampler* parent) : SingleSampler() {
    theParent = parent;
    theDiagram = diag;
    rootNode.boundaries(vector<double>(theParent->dimension() - 1, 0.0),
                        vector<double>(theParent->dimension() - 1, 1.0));
    rootNode.weightInformation().resize(theParent->dimension() - 1);
}

SingleSampler::~SingleSampler() {
}

void SingleSampler::updateIntegral() {
    rootNode.setWeights();
    rootNode.updateIntegral();
    theIntegral = rootNode.integral();
    if (!std::isfinite(theIntegral))
        theIntegral = 0;
    rootNode.minimumSelection(parent()->minimumSelection());
}

double SingleSampler::generate() {
    UseRandom rnd;
    //see CellGridSampler (here all flags and data is stored in theParent)
    vector<double> point(parent()->dimension() - 1, 0);
    double w = rootNode.sample(rnd, *this, point,
                               !parent()->weighted() && parent()->initialized() && parent()->unweightCells(),
                               !parent()->initialized());

    //convert the sampled point to a point as used by the BinSampler
    for (size_t i = 0; i < parent()->diagramDimension(); i++) {
        parent()->lastPoint()[i] = point[i];
    }
    point[parent()->diagramDimension()] = diagramPosition();
    for (size_t i = parent()->diagramDimension() + 1; i < size_t(parent()->dimension()); i++) {
        parent()->lastPoint()[i] = point[i - 1];
    }

    ++thePointCount;
    return w;
}

double SingleSampler::evaluate(std::vector<double> point) {
    std::vector<double> pointInt(parent()->dimension(), 0);

    // convert the incoming n-1 dimensional point to the form used for the BinSampler
    for (size_t i = 0; i < parent()->diagramDimension(); i++) {
        pointInt[i] = point[i];
    }
    pointInt[parent()->diagramDimension()] = diagramPosition();
    for (size_t i = parent()->diagramDimension() + 1; i < size_t(parent()->dimension()); i++) {
        pointInt[i] = point[i - 1];
    }

    double r = parent()->evaluate(pointInt);
    return r;
}

void SingleSampler::adapt(double factor) {
    UseRandom rnd;
    set < SimpleCellGrid * > newCells;
    //lerp epsilon
    double lerpLimit = parent()->epsilonInterpolLimit();
    double eps = factor < lerpLimit ? parent()->epsilon() * factor / lerpLimit : parent()->epsilon();

    //lerp  adaptPoints
    size_t minPoints = parent()->minAdaptPoints();
    size_t adaptPoints = minPoints + (parent()->explorationPoints() - minPoints) * factor;

    rootNode.adapt(parent()->gain(), eps, newCells);
    rootNode.explore(adaptPoints, rnd, *this, newCells,
                     Repository::clog());

    rootNode.setWeights();
    rootNode.minimumSelection(parent()->minimumSelection());
}

void SingleSampler::split() {
    std::set < SimpleCellGrid * > newCells;
    UseRandom rnd;
    //Based on the CellGridSampler with correction for the different dimensions
    for (int splitdim = 0;
         splitdim < min(parent()->dimension(), (int) parent()->pre_adaption_splits().size()); splitdim++) {
        int realDim = splitdim > parent()->diagramDimension() ? splitdim - 1 : splitdim;
        rootNode.splitter(realDim, parent()->pre_adaption_splits()[splitdim]);
    }

    rootNode.explore(parent()->explorationPoints(), rnd, *this, newCells, Repository::clog());
}

void SingleSampler::finishExplore() {
    //see CellGridSampler
    rootNode.setWeights();
    rootNode.minimumSelection(parent()->minimumSelection());
}

bool SingleSampler::explore(progress_display* progressBar, double frac) {
    //Single Iteration of the explore Loop in CellGridSampler::initialise
    if (exploreDone) return true;
    std::set < SimpleCellGrid * > newCells;
    UseRandom rnd;
    rootNode.adapt(parent()->gain(), parent()->epsilon(), newCells);
    if (progressBar)
        ++(*progressBar);

    if (newCells.empty())
        exploreDone = true;
    else {
        size_t minPoints = parent()->minAdaptPoints();
        //lerp with a limit
        size_t points =
                min(frac * parent()->diagramCount(), 1.0) * (parent()->explorationPoints() - minPoints) + minPoints;
        rootNode.explore(points, rnd, *this, newCells, Repository::clog());
    }
    return exploreDone;
}

void SingleSampler::fromXML(const XML::Element& elem) {
    int id;
    elem.getFromAttribute("diagram", id);
    assert(id == diagram()->id()); //sanity check
    rootNode.fromXML(elem);
}

XML::Element SingleSampler::toXML() const {
    XML::Element grid = rootNode.toXML();
    grid.appendAttribute("diagram", diagram()->id());
    return grid;
}