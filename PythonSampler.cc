#include "PythonSampler.h"

#include <fstream>
#include <chrono>

using namespace Herwig;
using namespace ThePEG;
using namespace std;

#include <pybind11/embed.h>
#include <pybind11/stl.h> 
#include <typeinfo>
//#include "ThePEG/MatrixElement/MEBase.h"
//#include "Herwig/MatrixElement/Matchbox/Base/MatchboxMEBase.h"
//#include "ThePEG/Handlers/LuminosityFunction.h"
//#include "ThePEG/Utilities/SimplePhaseSpace.h"
namespace py = pybind11;

PythonSampler::PythonSampler()
    : BinSampler() {
}

// PythonSampler::~PythonSampler() = default;

PythonSampler::~PythonSampler(){
    if(initialized()){
        sampling.attr("finalize")();
    }
    // try{
    //     sampling.attr("finalize")();
    // }
    // catch(const std::exception& e)
    // {
    //     std::cerr << e.what() << '\n';
    // }
}


double PythonSampler::generate() {
    if (cachedIndex >= maxCachedIndex || cachedIndex == -1) {
        try{
            pybind11::tuple result = sampling.attr("generate")(nSamplesCache);
            auto psPointsMatrix = result[0].cast<std::vector<std::vector<double>>>();
            auto probabilities = result[1].cast<std::vector<double>>();
            auto funcValues = result[2].cast<std::vector<double>>();
            auto nSamples = result[3].cast<int>();

            maxCachedIndex = nSamples;
    
            for (size_t i = 0; i < nSamples; ++i) {
                if (cachedPsPoints[i].capacity() >= psPointsMatrix[i].size()) {
                    cachedPsPoints[i].swap(psPointsMatrix[i]);
                } else {
                    cachedPsPoints[i] = std::move(psPointsMatrix[i]);
                }
                cachedProbabilities[i] = probabilities[i];
                cachedFuncValues[i] = funcValues[i];
            }            
            cachedIndex = 0;
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            exit(0);
        }
        
    
    }
    
    // Get the current sample
    double probability = cachedProbabilities[cachedIndex];
    double funcValue = cachedFuncValues[cachedIndex];
    const auto& psPoint = cachedPsPoints[cachedIndex];
    
    double w = probability > 0 ? funcValue / probability : 0.0;
    
    if (!weighted() && initialized()) {
        double p = min(abs(w), kappa() * referenceWeight()) / (kappa() * referenceWeight());
        double sign = w >= 0. ? 1. : -1.;
        if (p < 1 && UseRandom::rnd() > p)
            w = 0.;
        else
            w = sign * max(abs(w), referenceWeight() * kappa());
    }
    
    lastPoint() = psPoint;
    select(w);
    if (w != 0.0)
        accept();
    cachedIndex++;
    
    assert(kappa() == 1. || sampler()->almostUnweighted());
    return w;
}



double PythonSampler::dSigDR(std::vector<double> p) {
    double w = eventHandler()->dSigDR(p) / nanobarn;
    return w;
}

double PythonSampler::dSigDRRun(std::vector<double> p, double probability, double w) {
    // double w = eventHandler()->dSigDR(p) / nanobarn;
    lastPoint() = p;
    select(w);
    if(probability == 0.0 || w == 0.0) {
        return 0;
    }
    w  = w/probability;
    if ( w != 0.0 )
        accept();
    return w;
}



std::vector<double> PythonSampler::dSigDRMatrix(const std::vector<std::vector<double>>& matrix) {
    std::vector<double> results;
    results.reserve(matrix.size());

    for (const auto& row : matrix) {
        results.push_back(dSigDR(row));
    }

    return results;
}


  

void PythonSampler::initialize(bool progress) {
    cachedPsPoints.resize(nSamplesCache);
    cachedProbabilities.resize(nSamplesCache);
    cachedFuncValues.resize(nSamplesCache);


    const StandardEventHandler* handler = eventHandler();
    auto theDiagramDimension = eventHandler()->lumiDim() +
                          handler->xCombs()[bin()]->partonDimensions().first;

    // std::cout << "p1 phasespace dim: " << handler->xCombs()[bin()]->partonDimensions().first << std::endl;
    // std::cout << "p2 phasespace dim: " << handler->xCombs()[bin()]->partonDimensions().second << std::endl;
    // cout << "lumiDIm:" << eventHandler()->lumiDim() << endl;
    // cout << "theBin dim:" << eventHandler()->nDim(0) << endl;
    // cout << "theBin:" << bin() << endl;
    // cout << "process: " << process() << endl;
    int binCount = handler->xCombs().size();


    auto xComb = *(handler->xCombs()[bin()]);

    auto lastDiagram = xComb.lastDiagram();
    // cout << lastDiagram->getTag() << endl;

    auto matrixElem = xComb.matrixElement();
    auto matrixElemName = matrixElem->name();
    std::cout << "matrix element:" << matrixElem->name() << std::endl;
    int nDims2 = dimension();
    // std::cout << "nDims matrix element" << nDims2 << std::endl;
    int diagDim = xComb.diagrams().size();
    auto xc = handler->xCombs()[0];
    auto channelSelectionDim = xc->partonDimensions().first;

    try{
        py::initialize_interpreter();
    }
    catch(const std::exception& e)
    {
        // std::cerr << e.what() << '\n';
    }

    int nDims = dimension();
    try{
       sampling = py::module_::import("sampling");
       if(initialized()){
        py::object result = sampling.attr("load")(py::cast(this), nDims, diagDim, matrixElemName, bin(), binCount, channelSelectionDim, referenceWeight());
       }
       else{
        py::object result = sampling.attr("train")(py::cast(this), nDims, diagDim, matrixElemName, bin(), binCount, channelSelectionDim);
       }
      
    //   std::cout << "Returned float from sampling.py: " << result.cast<double>() << std::endl;
    }
    catch (py::error_already_set &e) {
      std::cout << "Failed to import sampling" << std::endl;
      std::cout << e.what() << std::endl;
    }
    // runIteration(initialPoints(),progress);
    // runIteration(10000,false);
    isInitialized();
}

PYBIND11_EMBEDDED_MODULE(herwig_python, m) {
    py::class_<PythonSampler>(m, "PythonSampler")
        .def(py::init<>())
        .def("dimension", &PythonSampler::dimension)
        .def("dSigDR", &PythonSampler::dSigDR)
        .def("dSigDRRun", &PythonSampler::dSigDRRun)
        .def("dSigDRMatrix", &PythonSampler::dSigDRMatrix);
  }


void PythonSampler::finalize(bool) {
    try
    {
        cout << "final integrated cross section is ( "
        << integratedXSec()/nanobarn << " +/- "
        << integratedXSecErr()/nanobarn << " ) nb\n" << endl;
        sampling.attr("finalize")();
        // std::cout << "Python Sampler finalize" << std::endl;
        // py::finalize_interpreter();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

void PythonSampler::adapt() {
    std::cout << "PythonSampler::adapt()" << std::endl;
}



IBPtr PythonSampler::clone() const {
    return new_ptr(*this);
}

IBPtr PythonSampler::fullclone() const {
    return new_ptr(*this);
}

DescribeClass<PythonSampler, BinSampler>
    describeHerwigPythonSampler("Herwig::PythonSampler", "HwSampling.so");

void PythonSampler::Init() {
    static ClassDocumentation<PythonSampler> documentation
        ("PythonSampler samples XCombs bins using a normalizing flow implemented in PyTorch.");

}