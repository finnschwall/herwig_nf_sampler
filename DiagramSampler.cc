#include "DiagramSampler.h"

#include "Herwig/API/RunDirectories.h"
#include "Herwig/MatrixElement/Matchbox/Utility/DiagramDrawer.h"
#include <fstream>
#include <chrono>

using namespace Herwig;
using namespace ThePEG;

DiagramSampler::DiagramSampler()
        : BinSampler(), theSelector(), theSamplers(), theMinDiagramSelection(0.0001), theMinAdaptPart(0.01),
          theEpsilonInterpolLimit(0.1),
          theProbDampening(0.65), theMinAdaptPoints(50), theExplorationPoints(1000), theExplorationSteps(8),
          theGain(0.3),
          theEpsilon(0.01), theMinimumSelection(0.0001), theLuminositySplits(0),
          theUnweightCells(true) {}

DiagramSampler::~DiagramSampler() = default;

double DiagramSampler::generate() {
    auto s = theSelector.select(UseRandom::current());
    double w = s->generate() / (theSamplers.size() * (s->probability() / theSelector.sum()));

    if (!weighted() && initialized()) {
        double p = min(abs(w), kappa() * referenceWeight()) / (kappa() * referenceWeight());
        double sign = w >= 0. ? 1. : -1.;
        if (p < 1 && UseRandom::rnd() > p)
            w = 0.;
        else
            w = sign * max(abs(w), referenceWeight() * kappa());
    }
    select(w);

    if (w != 0.0)
        accept();
    assert(kappa() == 1. || sampler()->almostUnweighted());
    return w;
}

void DiagramSampler::initialize(bool progress) {
    //check for grid
    bool haveGrid = false;
    auto git = sampler()->grids().children().begin();
    for (; git != sampler()->grids().children().end(); ++git) {
        if (git->type() != XML::ElementTypes::Element)
            continue;
        if (git->name() != "DiagramGrid")
            continue;
        string proc;
        git->getFromAttribute("process", proc);
        if (proc == id()) {
            haveGrid = true;
            break;
        }
    }

    if (haveGrid) {
        fromXML(*git);
        sampler()->grids().erase(git);
        didReadGrids();
    }

    lastPoint().resize(dimension());
    if (!randomNumberString().empty())
        for (size_t i = 0; i < lastPoint().size(); i++) {
            RandomNumberHistograms[RandomNumberIndex(id(), i)] = make_pair(RandomNumberHistogram(), 0.);
        }

    //calculate diagram dimension
    const StandardEventHandler* handler = eventHandler(); // needs to be done to call the public const variant of XCombs()
    theDiagramDimension = eventHandler()->lumiDim() +
                          handler->xCombs()[bin()]->partonDimensions().first;

    if (initialized()) {
        if (!hasGrids())
            throw Exception() << "DiagramSampler: Require existing grid when starting to run.\n"
                              << "Did you miss setting --setupfile?"
                              << Exception::abortnow;
        return;
    }

    if (haveGrid) {
        if (!integrated())
            runIteration(initialPoints(), progress);
        isInitialized();
        return;
    }

    //Load diagrams and setup SingleSamplers
    {
        auto& diagrams = handler->xCombs()[bin()]->diagrams();
        std::vector <Ptr<DiagramBase>::ptr> sortSet(diagrams);
        std::sort(sortSet.begin(), sortSet.end()); //TreePhasespace uses this ordering so we use it too
        double prob = 1.0 / (double) diagrams.size();

        if (diagrams.size() <= 1) {
            throw Exception() << "DiagramSampler: The selected process only has one diagram.\n"
                              << "Please use the CellGridSampler!" << Exception::abortnow;
            return;
        }
        if (prob < theMinDiagramSelection) {
            throw Exception() << "DiagramSampler: There are more diagrams than allowed by MinDiagramSelection.\n"
                              << "Consider lowering this parameter to at least " << prob << "."
                              << Exception::abortnow;
            return;
        }

        theSamplers.reserve(diagrams.size());

        for (size_t i = 0; i < sortSet.size(); i++) {
            theSamplers.emplace_back(sortSet[i], this);
            theSamplers.back().probability(prob);
            theSamplers.back().diagramPosition((i + 0.5) / diagrams.size());
            //the position won't change, because enough memory is already reserved
            theSelector.insert(prob, &theSamplers.back());
        }
    }

    progress_display* progressBar = nullptr;
    if (progress) {
        Repository::clog() << "exploring " << process();
        progressBar = new progress_display{theExplorationSteps * theSamplers.size(), Repository::clog()};
    }

    //generate Splits
    if (pre_adaption_splits().empty() && theLuminositySplits) {
        const StandardEventHandler& eh = *eventHandler();
        const StandardXComb& xc = *eh.xCombs()[bin()];
        the_pre_adaption_splits.resize(dimension(), 0);
        const pair<int, int>& pdims = xc.partonDimensions();
        if (theLuminositySplits && dimension() >= pdims.first + pdims.second) {
            for (int n = 0; n < pdims.first; ++n)
                the_pre_adaption_splits[n] = theLuminositySplits;
            for (int n = dimension() - pdims.second; n < dimension(); ++n)
                the_pre_adaption_splits[n] = theLuminositySplits;
        }
    }

    for (auto& sampler: theSamplers) {
        sampler.split();
    }
    adaptImpl(false);

    //explore all samplers
    bool notAll = false;
    for (size_t i = 1; i < explorationSteps(); ++i) {
        for (auto& sampler: theSamplers) {
            notAll |= sampler.explore(progressBar, sampler.probability() / theSelector.sum());
        }
        adaptImpl(false);
    }

    for (auto& sampler: theSamplers) {
        sampler.finishExplore();
        if (progressBar)
            ++(*progressBar);
    }

    if (progressBar) {
        if (notAll)
            cout << "\n" << flush;
        delete progressBar;
    }


    unsigned long points = initialPoints();
    for (unsigned long k = 0; k < nIterations(); ++k) {
        runIteration(points, progress);
        if (k < nIterations() - 1) {
            points = (unsigned long) (points * enhancementFactor());
            adapt();
            nextIteration();
        }
    }

    didReadGrids();
    isInitialized();
}

void DiagramSampler::finalize(bool) {
    XML::Element grid = toXML();
    grid.appendAttribute("process", id());
    sampler()->grids().append(grid);
    if (!randomNumberString().empty())
        for (auto& histogram: RandomNumberHistograms)
            histogram.second.first.dump(randomNumberString(), histogram.first.first, shortprocess(),
                                        histogram.first.second);
}

void DiagramSampler::adapt() {
    adaptImpl(true);
}

void DiagramSampler::adaptImpl(bool doAdapt) {
    double totalIntegral = 0;
    double oldIntegral = 0;
    //calculate total integrals
    for (auto& sampler: theSamplers) {
        oldIntegral += sampler.integral();
        sampler.updateIntegral();
        totalIntegral += sampler.integral();
    }

    //adapt samplers if needed
    if (doAdapt) {
        double threshold = totalIntegral * theMinAdaptPart;
        double oldTotal = totalIntegral;
        totalIntegral = 0;
        for (auto& sampler: theSamplers) {
            if (sampler.integral() > threshold)
                sampler.adapt(sampler.integral() / oldTotal);

            totalIntegral += sampler.integral();
        }
    }

    double minIntegral = minDiagSelection() * totalIntegral;

    //during init generate is never called thus pointCount=0 for all samplers
    unsigned pointLimit = doAdapt ? minIterationPoints() : 0;
    if (totalIntegral == 0)
        totalIntegral = 1;
    if (oldIntegral == 0)
        oldIntegral = 1;


    //update the selector
    theSelector.clear();
    for (auto& sampler: theSamplers) {
        double newProb = sampler.integral();

        if (sampler.pointCount() < pointLimit || sampler.integral() == 0)
            newProb = theProbDampening * sampler.probability() * totalIntegral / oldIntegral;
        if (newProb < minIntegral)
            newProb = minIntegral;

        sampler.probability(newProb);
        theSelector.insert(newProb, &sampler);
        sampler.resetStats();
    }

}

bool DiagramSampler::existsGrid() const {
    for (auto& git: sampler()->grids().children()) {
        if (git.type() != XML::ElementTypes::Element)
            continue;
        if (git.name() != "DiagramGrid")
            continue;
        string proc;
        git.getFromAttribute("process", proc);
        if (proc == id())
            return true;
    }
    return false;
}

void DiagramSampler::saveGrid() const {
    XML::Element grid = toXML();
    grid.appendAttribute("process", id());
    sampler()->grids().append(grid);
}

void DiagramSampler::fromXML(const XML::Element& elem) {
    const StandardEventHandler* handler = eventHandler(); // needs to be done to call the public const variant of XCombs()
    auto& diagrams = handler->xCombs()[bin()]->diagrams();
    theSamplers.reserve(diagrams.size());
    std::vector<RCPtr<DiagramBase>> sortSet(diagrams);
    std::sort(sortSet.begin(), sortSet.end()); //TreePhasespace uses this ordering so we use it too

    for (auto& git: elem.children()) {
        if (git.type() != XML::ElementTypes::Element)
            continue;
        if (git.name() != "CellGrid")
            continue;
        int id;
        git.getFromAttribute("diagram", id);
        double prob;
        git.getFromAttribute("probability", prob);

        auto diagr = std::find_if(sortSet.begin(), sortSet.end(), [id](auto item) { return item->id() == id; });
        int idx = diagr - sortSet.begin();
        assert(diagr != sortSet.end());
        theSamplers.emplace_back(*diagr, this);
        theSamplers.back().probability(prob);
        theSamplers.back().fromXML(git);
        theSamplers.back().diagramPosition((idx + 0.5) / diagrams.size());
        theSelector.insert(prob, &theSamplers.back());
    }
}

XML::Element DiagramSampler::toXML() const {
    XML::Element root(XML::ElementTypes::Element, "DiagramGrid");

    for (auto& samp: theSamplers) {
        auto sampNode = samp.toXML();
        sampNode.appendAttribute("probability", samp.probability());
        root.append(sampNode);
    }
    return root;
}

void DiagramSampler::persistentOutput(PersistentOStream& os) const {
    BinSampler::put(os);
    os << theExplorationPoints << theExplorationSteps
       << theGain << theEpsilon << theMinimumSelection
       << the_pre_adaption_splits
       << theLuminositySplits << theMinDiagramSelection
       << theUnweightCells
       << theMinAdaptPart << theEpsilonInterpolLimit
       << theMinAdaptPoints;
}

void DiagramSampler::persistentInput(PersistentIStream& is, int) {
    BinSampler::get(is);
    is >> theExplorationPoints >> theExplorationSteps
       >> theGain >> theEpsilon >> theMinimumSelection
       >> the_pre_adaption_splits
       >> theLuminositySplits >> theMinDiagramSelection
       >> theUnweightCells
       >> theMinAdaptPart >> theEpsilonInterpolLimit
       >> theMinAdaptPoints;
}


IBPtr DiagramSampler::clone() const {
    return new_ptr(*this);
}

IBPtr DiagramSampler::fullclone() const {
    return new_ptr(*this);
}

DescribeClass <DiagramSampler, BinSampler>
        describeHerwigDiagramSampler("Herwig::DiagramSampler", "HwSampling.so");

void DiagramSampler::Init() {

    static ClassDocumentation<DiagramSampler> documentation
            ("DiagramSampler samples XCombs bins using information provided by the respective Feynman-Diagrams.");

    static Parameter<DiagramSampler, size_t> interfaceExplorationPoints
            ("ExplorationPoints",
             "The number of points to use for cell exploration.",
             &DiagramSampler::theExplorationPoints, 1000, 1, 0,
             false, false, Interface::lowerlim);

    static Parameter<DiagramSampler, size_t> interfaceExplorationSteps
            ("ExplorationSteps",
             "The number of exploration steps to perform.",
             &DiagramSampler::theExplorationSteps, 8, 1, 0,
             false, false, Interface::lowerlim);

    static Parameter<DiagramSampler, double> interfaceGain
            ("Gain",
             "The gain factor used for adaption.",
             &DiagramSampler::theGain, 0.3, 0.0, 1.0,
             false, false, Interface::limited);

    static Parameter<DiagramSampler, double> interfaceEpsilon
            ("Epsilon",
             "The efficieny threshold used for adaption.",
             &DiagramSampler::theEpsilon, 0.01, 0.0, 1.0,
             false, false, Interface::limited);

    static Parameter<DiagramSampler, double> interfaceEpsilonInterpolLimit
            ("EpsilonInterpolLimit",
             "Epsilon is adapted linearly based on the selection probability of a diagram.\n"
             "If a diagram has a probability larger than this value, epsilon has the value set as the parameter epsilon.\n"
             "Note: this value must be nonzero",
             &DiagramSampler::theEpsilonInterpolLimit, 0.1, 0, 1.0,
             false, false, Interface::limited);

    static Parameter<DiagramSampler, double> interfaceAdaptLimit
            ("AdaptLimit",
             "The probability that a diagram must have such that the underlying CellGrid will be divided further.",
             &DiagramSampler::theMinAdaptPart, 0.01, 0.0, 1.0,
             false, false, Interface::limited);

    static Parameter<DiagramSampler, double> interfaceProbDampening
            ("ProbDampening",
             "Factor between 0 and 1 that reduces the probability of diagrams with integral 0 in each adaption.",
             &DiagramSampler::theProbDampening, 0.65, 0.0, 1.0,
             false, false, Interface::limited);

    static Parameter<DiagramSampler, size_t> interfaceMinAdaptPoints
            ("MinAdaptPoints",
             "The number of points at least used to explore new cells.\n"
             "(The exact value depends on the diagram probability)",
             &DiagramSampler::theMinAdaptPoints, 50, 0, 0,
             false, false, Interface::lowerlim);

    static Parameter<DiagramSampler, double> interfaceMinimumSelection
            ("MinimumSelection",
             "The minimum cell selection probability.",
             &DiagramSampler::theMinimumSelection, 0.0001, 0.0, 1.0,
             false, false, Interface::limited);

    static ParVector<DiagramSampler, int> interfacethe_pre_adaption_splits
            ("preadaptionsplit",
             "The splittings for each dimension before adaption.",
             &DiagramSampler::the_pre_adaption_splits, 1., -1, 0.0, 0.0, 0,
             false, false, Interface::lowerlim);

    static Parameter<DiagramSampler, int> interfaceLuminositySplits
            ("LuminositySplits",
             "",
             &DiagramSampler::theLuminositySplits, 0, 0, 0,
             false, false, Interface::lowerlim);

    static Parameter<DiagramSampler, double> interfaceMinDiagProb(
            "MinDiagramSelection", "The minimum selection probability for a diagram.", &DiagramSampler::theMinDiagramSelection, 0, 0, 0, false, false,
            Interface::lowerlim);

    static Switch<DiagramSampler, bool> interfaceUnweightCells
            ("UnweightCells",
             "",
             &DiagramSampler::theUnweightCells, true, false, false);
    static SwitchOption interfaceUnweightCellsYes
            (interfaceUnweightCells,
             "Yes",
             "",
             true);
    static SwitchOption interfaceUnweightCellsNo
            (interfaceUnweightCells,
             "No",
             "",
             false);


}
