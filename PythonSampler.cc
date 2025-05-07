#include "PythonSampler.h"

#include <fstream>
#include <chrono>

using namespace Herwig;
using namespace ThePEG;
using namespace std;

#include <pybind11/embed.h>
#include <pybind11/stl.h> 
#include <typeinfo>
#include "ThePEG/MatrixElement/MEBase.h"
#include "Herwig/MatrixElement/Matchbox/Base/MatchboxMEBase.h"
#include "ThePEG/Handlers/LuminosityFunction.h"
#include "ThePEG/Utilities/SimplePhaseSpace.h"
namespace py = pybind11;

PythonSampler::PythonSampler()
    : BinSampler() {
}

PythonSampler::~PythonSampler() = default;


double PythonSampler::generate() {
    // Get point, probability, and function value from your NF
    pybind11::tuple result = sampling.attr("generate")();
    std::vector<double> psPoint = result[0].cast<std::vector<double>>();  // The sampled point
    double probability = result[1].cast<double>();                       // PDF at the point
    double funcValue = result[2].cast<double>();                         // Function value at the point
    
    // Calculate the weight (func_val / prob)
    double w = probability > 0 ? funcValue / probability : 0.0;
    
    // Apply importance sampling similar to the original code
    if (!weighted() && initialized()) {
        double p = min(abs(w), kappa() * 10.4) / (kappa() * 10.4);
        double sign = w >= 0. ? 1. : -1.;
        if (p < 1 && UseRandom::rnd() > p)
            w = 0.;
        else
            w = sign * max(abs(w), 10.4 * kappa());
    }
    
    // Update the last point (you'll need to implement this)
    lastPoint() = psPoint;
    
    // Select this weight
    select(w);
    
    // Accept the point if weight is non-zero
    if (w != 0.0)
        accept();
        
    assert(kappa() == 1. || sampler()->almostUnweighted());
    // std::cout << "w:" << w << " kappa:" << kappa() << " ref:" << referenceWeight() << std::endl;
    return w;
}

// double PythonSampler::generate(){
//     pybind11::tuple result = sampling.attr("generate")();
//     std::vector<double> p = result[0].cast<std::vector<double>>();
//     double probability = result[1].cast<double>();
//     double w = result[2].cast<double>()/probability;
//     // double maxWeight = result[3].cast<double>();

//     double rand = UseRandom::rnd();
//     if (rand < w / referenceWeight()) {
//         // Accept the event
//         lastPoint() = p;
//         w=referenceWeight();
//     } else {
//         select(0.0);
//         return 0.0;
//     }
   
//     select(w);
//     if ( w != 0.0 )
//       accept();
//     assert(kappa()==1.||sampler()->almostUnweighted());
//     return w;
// }

// double PythonSampler::generate() {
//     pybind11::tuple result = sampling.attr("generate")();
//     std::vector<double> p = result[0].cast<std::vector<double>>();
//     double probability = result[1].cast<double>();
//     double w = result[2].cast<double>();
//     // double w = eventHandler()->dSigDR(p) / nanobarn;
//     lastPoint() = p;
//     select(w);
//     if(probability == 0.0 || w == 0.0) {
//         return 0;
//     }
//     w  = w/probability;
//     if ( w != 0.0 )
//         accept();

//     return w;
//     // py::tuple result = sampling.attr("generate")(1);
//     // double w = result[1].cast<double>();
//     // lastPoint() = result[0].cast<std::vector<double>>();
    
//     // select(w);
//     // if ( w != 0.0 )
//     //   accept();
//     // return w;
// }

double PythonSampler::dSigDR(std::vector<double> p) {
    double w = eventHandler()->dSigDR(p) / nanobarn;
    // lastPoint() = p;
    // select(w);
    // if ( w != 0.0 )
    //   accept();
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
    // std::cout << "matrix element:" << matrixElem->name() << std::endl;
    int nDims2 = dimension();
    // std::cout << "nDims matrix element" << nDims2 << std::endl;
    int diagDim = xComb.diagrams().size();

    cout << "WEIGHTED:" << weighted() << endl;
    cout << "REFWEIGHT:" << referenceWeight() << endl;
    cout << "KAPPA:" << kappa() << endl;
    try{

        // auto xc = handler->xCombs()[0];
        // cout << "dia dim:"<< xc->diagrams().size() << endl;

        // vector<double> vec = {0.001, 10.1212, 0.1515};//, 0.0344, 0.6610, 0.8498};//, 0.3029};
        // double w =0;
        // w = eventHandler()->dSigDR(vec) / nanobarn;
        // MatchboxMEBase* matchboxME = dynamic_cast<MatchboxMEBase*>(matrixElem.operator->());
        // cout << "havex1x2:" << matchboxME->haveX1X2() << endl; 


        // auto phasespace = matchboxME->phasespace();
        // phasespace->generateKinematics(&vec[0], xc->meMomenta());

        // cout << "PS gen dim for 2 outgoing: " << phasespace->nDimPhasespace(2) << endl;
        // cout << "wantCMS: " << phasespace->wantCMS() << endl;
        

        // auto meMomenta = xc->meMomenta();

        // cout << "meMomenta:" << meMomenta.size() << endl;


        // pair<double, double> ll(0.0, 0.0);
        // const LuminosityFunction &lumi = handler->lumiFn();
        // Energy2 maxS = sqr(lumi.maximumCMEnergy())/exp(ll.first + ll.second);
        // PPair inc = make_pair(handler->incoming().first->produceParticle(),
		// 	handler->incoming().second->produceParticle());
        // SimplePhaseSpace::CMS(inc, maxS);
        // xc->prepare(inc);

        // w = xc->dSigDR(ll, 7,&vec[0])/ nanobarn;;
        // cout << "wXcomb" << w << endl;

   
        // auto temp = xc->meMomenta();
        // cout << "temp" << temp.size() << endl;


        // const vector<double> r = {};
        // double jac = 1.0;
        // const LuminosityFunction &lumi = handler->lumiFn();
        // auto res = lumi.generateLL(&r[0], jac);

        // cout << res.first <<"," << res.second << endl;

        // auto maxS = sqr(lumi.maximumCMEnergy());

        // cout << lumi.maximumCMEnergy()/GeV << endl;

        // cout << "binDim" << handler->nDim(0) << endl;

        // pair<double, double> ll(0.0, 0.0);
        // double *vec = new double[7];
        // auto cr = eventHandler()->dSigDR(ll, maxS, 0, 7, vec);

    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
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
        py::object result = sampling.attr("load")(py::cast(this), nDims, diagDim, matrixElemName, bin(), binCount);
       }
       else{
        py::object result = sampling.attr("train")(py::cast(this), nDims, diagDim, matrixElemName, bin(), binCount);
       }
      
    //   std::cout << "Returned float from sampling.py: " << result.cast<double>() << std::endl;
    }
    catch (py::error_already_set &e) {
      std::cout << "Failed to import sampling" << std::endl;
      std::cout << e.what() << std::endl;
    }
    cout << "INITIALPOINTS:" << 1000 << endl;
    runIteration(10000,progress);
    cout << "REFWEIGHT:" << referenceWeight() << endl;
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